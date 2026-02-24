import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
sbert = pd.read_csv("results/results_sbert.csv")
labse = pd.read_csv("results/results_labse.csv")

# Compute accuracy per language
sbert_acc = sbert.groupby("language")["is_correct_top1"].mean()
labse_acc = labse.groupby("language")["is_correct_top1"].mean()

languages = sorted(sbert_acc.index)

x = np.arange(len(languages))
width = 0.35

plt.figure()

plt.bar(x - width/2, [sbert_acc[lang] for lang in languages], width, label="SBERT")
plt.bar(x + width/2, [labse_acc[lang] for lang in languages], width, label="LaBSE")

plt.xticks(x, languages)
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Accuracy per Language per Model")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_by_language.png", dpi=300)
plt.show()