import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('evaluation_results.csv')

# Préparation des données
metrics = ["MAP", "P@1", "P@5", "P@10", "R-P"]
models = results['name'].unique()

# Création des graphiques
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 3, i+1)
    for model in models:
        values = results[results['name'] == model][metric]
        plt.plot(values, label=model)
    plt.title(metric)
    plt.xlabel('Requêtes')
    plt.ylabel('Score')
    plt.legend()

plt.tight_layout()
plt.savefig('evaluation_metrics.png')
plt.show()