# Hackathon2026
End-to-end project combining Data Engineering and Generative AI, using a biomedical article corpus inspired by PubMed.

# Mission
Build a measured data-and-AI system for biomedical long documents:
- Ingest PubMed-style articles,
- Store them cleanly,
- Add new articles,
- Summarize them with AI,
- Search the corpus with RAG,
- Benchmark environmental, cost, latency, and quality trade-offs.

# 1. Domain and Dataset Context
## 1.1 What is PubMed?
PubMed is a free resource for searching biomedical literature, mainly providing citations and abstracts rather than full-text articles. It is used for discovery, while PubMed Central offers free access to some full texts, and the hackathon dataset is a teaching version inspired by this system.
## 1.2 Dataset used in the hackathon
Dataset: ccdv/pubmed-summarization from Hugging Face. 
Parquet Format.
