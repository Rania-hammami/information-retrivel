import nltk
import os
import pyterrier as pt
import pandas as pd
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize



def stem_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def lemmatize_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


nltk.download('punkt')
nltk.download('wordnet')

# Initialiser le stemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()



# Initialiser PyTerrier
if not pt.started():
    pt.init()

# Charger les tweets depuis le fichier TSV
try:
    df = pd.read_csv("tweets.tsv", sep="\t")
except FileNotFoundError:
    raise FileNotFoundError("Le fichier 'tweets.tsv' est introuvable dans le répertoire courant.")

# Vérifier les colonnes nécessaires
required_cols = {'docno', 'text'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Le fichier doit contenir les colonnes suivantes : {required_cols}")

# Convertir docno en string
df['docno'] = df['docno'].astype(str)

# Créer des versions stemmisées et lemmatisées
df_stemmed = df.copy()
df_stemmed['text'] = df_stemmed['text'].apply(stem_text)

df_lemmatized = df.copy()
df_lemmatized['text'] = df_lemmatized['text'].apply(lemmatize_text)

# Préparer les données pour PyTerrier
tweet_dicts = df[['docno', 'text']].to_dict(orient='records')


# Créer les trois index différents

print("Création de l'index original...")
index_ref = pt.IterDictIndexer("./tweet_index", overwrite=True).index(df[['docno', 'text']].to_dict('records'))

print("Création de l'index stemmisé...")
index_stem = pt.IterDictIndexer("./index_stem", overwrite=True).index(df_stemmed.to_dict("records"))

print("Création de l'index lemmatisé...")
index_lemma = pt.IterDictIndexer("./index_lemma", overwrite=True).index(df_lemmatized.to_dict("records"))

# Afficher les statistiques pour chaque index
print("\nStatistiques de l'index original:")
print(pt.IndexFactory.of(index_ref).getCollectionStatistics())

print("\nStatistiques de l'index stemmisé:")
print(pt.IndexFactory.of(index_stem).getCollectionStatistics())

print("\nStatistiques de l'index lemmatisé:")
print(pt.IndexFactory.of(index_lemma).getCollectionStatistics())

print("\n✅ Tous les index ont été créés avec succès.")