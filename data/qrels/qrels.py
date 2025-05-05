import os
import pandas as pd

# Étape 1 : Lister tous les fichiers texte dans le répertoire courant
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]

# Étape 2 : Lire et agréger le contenu des fichiers texte
texts = []
for file in txt_files:
    with open(file, 'r', encoding='utf-8') as f:
        # Lire tout le contenu du fichier
        content = f.read().strip()
        if content:  # seulement si le fichier n'est pas vide
            texts.append(content)

# Étape 3 : Construire le DataFrame avec docno et text
df = pd.DataFrame({
    'docno': range(1, len(texts) + 1),
    'text': texts
})

# Étape 4 : Sauvegarder au format TSV (tabulé)
df.to_csv('qrels.tsv', sep='\t', index=False, encoding='utf-8')

print(f"{len(df)} textes ont été enregistrés dans 'qrels.tsv'")