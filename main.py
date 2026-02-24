import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# Load datasets
faq_df = pd.read_csv("faq.csv")
queries_df = pd.read_csv("test_queries.csv")

print("FAQ Loaded:", faq_df.shape)
print("Queries Loaded:", queries_df.shape)

# Load SBERT model
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("SBERT model loaded successfully.")

# Encode FAQ questions
faq_questions = faq_df["question_en"].tolist()
faq_embeddings_sbert = sbert_model.encode(faq_questions, convert_to_tensor=True)
print("FAQ embeddings generated (SBERT).")

# Test with first query
test_query = queries_df.iloc[0]["query_text"]
query_embedding = sbert_model.encode(test_query, convert_to_tensor=True)

# Similarity + best match
scores = util.cos_sim(query_embedding, faq_embeddings_sbert)
best_match_idx = int(scores.argmax())
best_score = float(scores[0][best_match_idx])

print("\nQuery:", test_query)
print("Predicted FAQ:", faq_df.iloc[best_match_idx]["question_en"])
print("Similarity Score:", best_score)
print("RUNNING FILE:", __file__)
print("CURRENT FOLDER:", os.getcwd())
labse_model = SentenceTransformer("sentence-transformers/LaBSE")
print("LaBSE model loaded successfully.")
faq_embeddings_labse = labse_model.encode(faq_questions, convert_to_tensor=True)
print("FAQ embeddings generated (LaBSE).")
query_embedding_labse = labse_model.encode(test_query, convert_to_tensor=True)

scores_labse = util.cos_sim(query_embedding_labse, faq_embeddings_labse)
best_idx_labse = int(scores_labse.argmax())
best_score_labse = float(scores_labse[0][best_idx_labse])

print("\n[LaBSE]")
print("Query:", test_query)
print("Predicted FAQ:", faq_df.iloc[best_idx_labse]["question_en"])
print("Similarity Score:", best_score_labse)
