# Multilingual FAQ Chatbot using SBERT and LaBSE

## 📌 Project Overview
This project presents the design and implementation of a multilingual FAQ chatbot that retrieves the most relevant answer from an English knowledge base using semantic similarity. The system supports queries in English, Arabic, and French by leveraging transformer-based sentence embedding models.

Two models are evaluated and compared:
- SBERT (paraphrase-multilingual-MiniLM-L12-v2)
- LaBSE (Language-agnostic BERT Sentence Embedding)

The chatbot is implemented as a retrieval-based system and includes a full evaluation pipeline using metrics such as Mean Reciprocal Rank (MRR) and accuracy.

---

## 🎯 Objectives
- Build a multilingual FAQ retrieval system
- Compare SBERT and LaBSE for cross-lingual retrieval
- Evaluate performance using standard IR metrics
- Develop an interactive chatbot interface using Streamlit

---

## 🧠 System Architecture
The system follows a retrieval-based pipeline:

1. Load FAQ dataset (English)
2. Load multilingual query dataset
3. Convert text into embeddings using SBERT / LaBSE
4. Compute cosine similarity
5. Rank FAQ results
6. Return the best matching answer

---

## 📁 Project Structure
