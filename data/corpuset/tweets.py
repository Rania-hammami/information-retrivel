import os
import json
import pandas as pd

# Étape 1 : Lister tous les fichiers JSON dans le répertoire courant
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

# Étape 2 : Lire et agréger les tweets
tweets = []
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            tweets.extend(data)
        else:
            tweets.append(data)

# Étape 3 : Extraire le texte des tweets
# On suppose que chaque tweet est un dict avec une clé 'text'
texts = [tweet.get('text', '').strip() for tweet in tweets if tweet.get('text')]

# Étape 4 : Construire le DataFrame avec docno et text
df = pd.DataFrame({
    'docno': range(1, len(texts) + 1),
    'text': texts
})

# Étape 5 : Sauvegarder au format TSV (tabulé)
df.to_csv('tweets.tsv', sep='\t', index=False, encoding='utf-8')

print(f"{len(df)} tweets ont été enregistrés dans 'tweets.tsv'")
