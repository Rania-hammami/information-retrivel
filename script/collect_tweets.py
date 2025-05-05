import tweepy
import json
import os
import time 

# 🔹 Configuration
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAE%2BU0QEAAAAADRmE%2FVuUwzjH7XN%2BF6Isrr%2FYxoA%3DutwtPahqi5ozMCBuw8zfU72g1Pe4ZZLwYn1Xl2NJE2V4QqHamY"
QUERIES = ["chacun execute son propre requet .....",]
DATA_DIR = "../data"


# 🔹 Connexion à l'API Tweepy
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# 🔹 Initialisation des listes de données
corpus = []
topics = []
qrels = []

print("🔎 Début de la collecte des tweets...")

for idx, query in enumerate(QUERIES, start=1):
    query_id = f"MB{str(idx + 38).zfill(2)}"
    topics.append({"num": query_id, "title": query})

    try:
        # 🔹 Recherche de tweets
        response = client.search_recent_tweets(
            query=query + " lang:en -is:retweet",
            tweet_fields=["created_at", "public_metrics", "lang"],
            max_results=100
        )

        tweets = response.data if response.data else []
        if not tweets:
            print(f"[!] Aucun tweet trouvé pour '{query}'")
            continue

        for i, tweet in enumerate(tweets):
            tweet_data = {
                "id": str(tweet.id),
                "timestamp": tweet.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "user": "anonymized_user",
                "text": tweet.text,
                "lang": tweet.lang,
                "retweets": tweet.public_metrics["retweet_count"],
                "likes": tweet.public_metrics["like_count"]
            }
            corpus.append(tweet_data)

            relevance = 1 if i < 30 else 0
            qrels.append(f"{query_id} 0 {tweet.id} {relevance}")

        print(f"[✓] {len(tweets)} tweets collectés pour '{query}'")


    except Exception as e:
        print(f"[!] Erreur lors de la collecte des tweets pour '{query}': {e}")

# 🔹 Sauvegarde des fichiers JSON même en cas d'erreur
with open(os.path.join(DATA_DIR, "corpus.json"), "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)

with open(os.path.join(DATA_DIR, "topics.json"), "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=4, ensure_ascii=False)

with open(os.path.join(DATA_DIR, "qrels.txt"), "w", encoding="utf-8") as f:
    for line in qrels:
        f.write(line + "\n")

print("✅ Tous les fichiers ont été enregistrés avec succès dans 'DATA_DIR'.")