import pyterrier as pt
import pandas as pd
import os
from pyterrier.measures import *

# Stocker les top 3 résultats globaux
top_results_summary = []

def clean_qrels(qrels):
    """Clean and validate the qrels dataframe"""
    qrels['label'] = pd.to_numeric(qrels['label'], errors='coerce')
    qrels = qrels.dropna(subset=['label'])
    qrels['label'] = qrels['label'].astype(int)
    qrels = qrels[qrels['label'].isin([0, 1])]
    return qrels

def load_and_verify_data():
    """Load and verify qrels and queries with proper validation"""
    try:
        qrels = pd.read_csv("qrels.tsv", sep='\s+', header=None, 
                            names=["qid", "iter", "docno", "label"])
        qrels = clean_qrels(qrels)
        qrels['docno'] = qrels['docno'].astype(str)
        qrels['qid'] = qrels['qid'].astype(str)

        print("\n🔍 Qrels Verification:")
        print(f"Total valid judgments: {len(qrels)}")
        print(f"Unique query IDs: {qrels['qid'].unique()}")
        print(f"Relevance distribution:\n{qrels['label'].value_counts()}")

        queries = pd.DataFrame([
            {"qid": "MB39", "query": "Gaza under attack"},
            {"qid": "MB40", "query": "Military occupation Gaza"},
            {"qid": "MB41", "query": "Israel Genocide Gaza"},
            {"qid": "MB42", "query": "Ceasefire in Gaza"},
            {"qid": "MB43", "query": "Massacres in Gaza"},
            {"qid": "MB44", "query": "Palestinian rights"},
            {"qid": "MB45", "query": "Refugees from Gaza"},
        ])

        missing_qids = set(queries['qid']) - set(qrels['qid'])
        if missing_qids:
            print(f"⚠ Warning: Missing judgments for QIDs: {missing_qids}")
            print("Evaluation will only work for queries with judgments")

        return queries, qrels
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def evaluate_model(model, model_name, index_name, topics_df, qrels):
    """Evaluate a single model and record top 3 results per query"""
    print(f"\n● Evaluating {model_name} (Index: {index_name})")

    try:
        valid_qids = qrels['qid'].unique()
        eval_topics = topics_df[topics_df['qid'].isin(valid_qids)]

        if len(eval_topics) == 0:
            print("⚠ No queries with judgments available for evaluation")
            return None

        results = model.transform(eval_topics)

        # Top 3 results per query
        top3_per_query = results.groupby('qid').apply(
            lambda df: df.nlargest(3, 'score')).reset_index(drop=True)

        print("Top 3 results:")
        print(top3_per_query[['qid', 'docno', 'score']].head(3))

        # Enregistrer les résultats pour tableau final
        for _, row in top3_per_query.iterrows():
            top_results_summary.append({
                "model": model_name,
                "index": index_name,
                "qid": row["qid"],
                "docno": row["docno"],
                "score": row["score"]
            })

        return None
    except Exception as e:
        print(f"❌ Evaluation failed for {model_name}: {e}")
        return None

def main():
    pt.java.init()
    topics_df, qrels = load_and_verify_data()
    if topics_df is None or qrels is None:
        return

    index_configs = [
        ('Original', "./tweet_index"),
        ('Stemmed', "./index_stem"),
        ('Lemmatized', "./index_lemma")
    ]

    for index_name, index_path in index_configs:
        print(f"\n{'='*50}\n📈 Evaluating {index_name} Index\n{'='*50}")

        try:
            index_ref = pt.IndexFactory.of(index_path)
            stats = index_ref.getCollectionStatistics()
            print(f"\n📊 {index_name} Index Statistics:")
            print(f"- Documents: {stats.getNumberOfDocuments()}")
            print(f"- Unique terms: {stats.getNumberOfUniqueTerms()}")
            print(f"- Avg length: {stats.getAverageDocumentLength():.1f} terms/doc")

            models = {
                "BM25": pt.terrier.Retriever(index_ref, wmodel="BM25"),
                "TF-IDF": pt.terrier.Retriever(index_ref, wmodel="TF_IDF"),
                "PL2": pt.terrier.Retriever(index_ref, wmodel="PL2"),
                "DLH": pt.terrier.Retriever(index_ref, wmodel="DLH"),
                "DFRee": pt.terrier.Retriever(index_ref, wmodel="DFRee"),
                "DirichletLM": pt.terrier.Retriever(index_ref, wmodel="DirichletLM"),
                "DFIZ": pt.terrier.Retriever(index_ref, wmodel="DFIZ"),
                "LGD": pt.terrier.Retriever(index_ref, wmodel="LGD"),
                "BM25 (k1=0.9,b=0.3)": pt.terrier.Retriever(index_ref, controls={"wmodel": "BM25", "bm25.k1": 0.9, "bm25.b": 0.3})
            }

            for model_name, model in models.items():
                evaluate_model(model, model_name, index_name, topics_df, qrels)

        except Exception as e:
            print(f"❌ Error processing index {index_name}: {e}")
            continue

    # Afficher le tableau final
    if top_results_summary:
        df_top3 = pd.DataFrame(top_results_summary)
        df_top3 = df_top3.sort_values(by=["index", "model", "qid", "score"], ascending=[True, True, True, False])

        print("\n📋 Final Top 3 Results per Model and Index:")
        print(df_top3.to_string(index=False, float_format="%.4f"))

        os.makedirs("results", exist_ok=True)
        df_top3.to_csv("results/top3_results.csv", index=False)
        print("\n✅ Résultats Top 3 sauvegardés dans: results/top3_results.csv")
    else:
        print("\n❌ Aucun résultat Top 3 généré")

if __name__ == "__main__":
    main()
