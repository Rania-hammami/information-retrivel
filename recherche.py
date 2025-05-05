import pyterrier as pt
import pandas as pd
from pyterrier.measures import *

# Initialiser PyTerrier si ce n’est pas déjà fait
if not pt.started():
    pt.init()

# Charger les requêtes
def charger_requetes():
    return pd.DataFrame([
        {"qid": "MB39", "query": "Gaza under attack"},
        {"qid": "MB40", "query": "Military occupation Gaza"},
        {"qid": "MB41", "query": "Israel Genocide Gaza"},
        {"qid": "MB42", "query": "Ceasefire in Gaza"},
        {"qid": "MB43", "query": "Massacres in Gaza"},
        {"qid": "MB44", "query": "Palestinian rights"},
        {"qid": "MB45", "query": "Refugees from Gaza"},
    ])

# Charger les qrels proprement (supporte espaces multiples comme séparateurs)
def charger_qrels():
    qrels = pd.read_csv("qrels.tsv", delim_whitespace=True, header=None, names=["qid", "iter", "docno", "label"])
    return qrels

# Fonction d’évaluation
def evaluer_modeles_etendus():
    topics_df = charger_requetes()
    qrels_df = charger_qrels()

    print("\n✔ QIDs dans topics :", topics_df['qid'].unique())
    print("✔ QIDs dans qrels :", qrels_df['qid'].unique())

    for index_name, index_path in [
        ('Original', "./tweet_index"),
        ('Stemmisé', "./index_stem"),
        ('Lemmatisé', "./index_lemma")
    ]:
        print(f"\n====================\n🔎 Évaluation de l’index : {index_name}")

        # Charger l'index
        index_ref = pt.IndexFactory.of(index_path)

        # Définir les modèles
        models = {
            "BM25": pt.BatchRetrieve(index_ref, wmodel="BM25"),
            "TF-IDF": pt.BatchRetrieve(index_ref, wmodel="TF_IDF"),
            "PL2": pt.BatchRetrieve(index_ref, wmodel="PL2"),
            "DLH": pt.BatchRetrieve(index_ref, wmodel="DLH"),
            "DFRee": pt.BatchRetrieve(index_ref, wmodel="DFRee"),
            "DirichletLM": pt.BatchRetrieve(index_ref, wmodel="DirichletLM"),
            "DFIZ": pt.BatchRetrieve(index_ref, wmodel="DFIZ"),
            "LGD": pt.BatchRetrieve(index_ref, wmodel="LGD"),
            "BM25 (k1=0.9,b=0.3)": pt.BatchRetrieve(index_ref, controls={"wmodel": "BM25", "bm25.k1": 0.9, "bm25.b": 0.3}),
        }

        for model_name, model in models.items():
            print(f"\n📌 Modèle : {model_name}")

            # Obtenir les résultats
            results = model.transform(topics_df)

            # Afficher les 3 premiers résultats
            print("Top 3 résultats :")
            print(results[['qid', 'docno', 'score']].head(3))

            # Évaluer le modèle avec qrels
            try:
                eval = pt.Experiment(
                    [model],
                    topics_df,
                    qrels_df,
                    eval_metrics=["AP", "P@5", "P@10", "RR", "nDCG@10"]
                )
                print("\n📊 Résultats d’évaluation :")
                print(eval)
            except ValueError as e:
                print(f"⚠ Erreur d’évaluation : {e}")

        # Afficher les statistiques de l'index
        stats = index_ref.getCollectionStatistics()
        print(f"\n📚 Statistiques de l’index {index_name} :")
        print(f"- Documents : {stats.getNumberOfDocuments()}")
        print(f"- Termes uniques : {stats.getNumberOfUniqueTerms()}")
        print(f"- Longueur moyenne : {stats.getAverageDocumentLength():.2f}")

# Lancer l’évaluation
evaluer_modeles_etendus()
print("\n✅ Évaluation complète terminée.")
