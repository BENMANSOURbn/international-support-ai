import pandas as pd
import matplotlib.pyplot as plt

m = pd.read_csv("results/metrics_summary.csv")

plt.figure()
plt.bar(m["model"], m["MRR"])
plt.ylabel("MRR")
plt.ylim(0, 1.05)
plt.title("Mean Reciprocal Rank (MRR) per Model")

plt.tight_layout()
plt.savefig("mrr_per_model.png", dpi=300)
plt.tight_layout()
plt.savefig("overall_accuracy.png", dpi=300)
plt.show()