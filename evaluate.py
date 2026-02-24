import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Config
# -------------------------
print("evaluate.py started running")
FAQ_FILE = "faq.csv"
QUERIES_FILE = "test_queries.csv"
RESULTS_DIR = "results"

MODELS = {
    "sbert": "paraphrase-multilingual-MiniLM-L12-v2",
    "labse": "sentence-transformers/LaBSE",
}

# -------------------------
# Helpers
# -------------------------
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    faq_df = pd.read_csv(FAQ_FILE)
    queries_df = pd.read_csv(QUERIES_FILE)

    # Remove completely empty rows (Excel sometimes adds them)
    queries_df = queries_df.dropna(how="all")

    # Remove rows missing required fields (prevents NaN -> int crash)
    required_cols = ["query_id", "query_text", "language", "correct_faq_id"]
    queries_df = queries_df.dropna(subset=required_cols)

    # Convert IDs safely
    faq_df["faq_id"] = faq_df["faq_id"].astype(int)
    queries_df["query_id"] = queries_df["query_id"].astype(int)
    queries_df["correct_faq_id"] = queries_df["correct_faq_id"].astype(int)

    # Clean language column
    queries_df["language"] = queries_df["language"].astype(str).str.strip().str.lower()

    return faq_df, queries_df


def evaluate_model(model_name: str, model: SentenceTransformer, faq_df: pd.DataFrame, queries_df: pd.DataFrame):
    """
    Runs retrieval for all queries and returns:
    - results_df: per-query predictions + ranks
    - metrics: summary metrics dict
    """

    # 1) Embed FAQ questions once
    faq_questions = faq_df["question_en"].tolist()
    faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

    # Map index -> faq_id (because embeddings are in the same order as faq_df rows)
    faq_ids = faq_df["faq_id"].tolist()

    rows = []

    # 2) Loop through every query
    for _, q in queries_df.iterrows():
        query_id = int(q["query_id"])
        query_text = str(q["query_text"])
        lang = str(q["language"])
        correct_id = int(q["correct_faq_id"])

        # Embed query
        q_emb = model.encode(query_text, convert_to_tensor=True)

        # Similarity scores (1 x num_faqs)
        scores = util.cos_sim(q_emb, faq_embeddings)[0]

        # Sort indices by descending score
        ranked_idx = scores.argsort(descending=True)

        # Top-1 prediction
        top1_idx = int(ranked_idx[0])
        pred_faq_id = int(faq_ids[top1_idx])
        top1_score = float(scores[top1_idx])

        # Find rank of correct FAQ (1-based)
        correct_rank = None
        for r, idx in enumerate(ranked_idx, start=1):
            if int(faq_ids[int(idx)]) == correct_id:
                correct_rank = r
                break

        rr = 1.0 / correct_rank if correct_rank is not None else 0.0

        rows.append({
            "model": model_name,
            "query_id": query_id,
            "language": lang,
            "query_text": query_text,
            "correct_faq_id": correct_id,
            "predicted_faq_id": pred_faq_id,
            "correct_rank": correct_rank,
            "reciprocal_rank": rr,
            "top1_similarity": top1_score,
            "is_correct_top1": int(pred_faq_id == correct_id),
        })

    results_df = pd.DataFrame(rows)

    # 3) Compute metrics
    mrr = results_df["reciprocal_rank"].mean()
    overall_acc = results_df["is_correct_top1"].mean()
    acc_by_lang = results_df.groupby("language")["is_correct_top1"].mean().to_dict()

    correct_only = results_df[results_df["is_correct_top1"] == 1]
    avg_sim_correct = correct_only["top1_similarity"].mean() if len(correct_only) > 0 else 0.0

    metrics = {
        "model": model_name,
        "MRR": float(mrr),
        "overall_accuracy": float(overall_acc),
        "accuracy_en": float(acc_by_lang.get("en", 0.0)),
        "accuracy_ar": float(acc_by_lang.get("ar", 0.0)),
        "accuracy_fr": float(acc_by_lang.get("fr", 0.0)),
        "avg_similarity_correct": float(avg_sim_correct),
        "num_queries": int(len(results_df)),
    }

    return results_df, metrics


# -------------------------
# Main
# -------------------------
def main():
    ensure_results_dir()
    faq_df, queries_df = load_data()

    all_metrics = []

    for short_name, hf_name in MODELS.items():
        print(f"\nLoading model: {short_name} -> {hf_name}")
        model = SentenceTransformer(hf_name)

        print(f"Evaluating {short_name} on {len(queries_df)} queries...")
        results_df, metrics = evaluate_model(short_name, model, faq_df, queries_df)

        out_path = os.path.join(RESULTS_DIR, f"results_{short_name}.csv")
        results_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved: {out_path}")

        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"\nSaved metrics summary: {metrics_path}")

    print("\n=== METRICS SUMMARY ===")
    print(metrics_df)

if __name__ == "__main__":
    main()