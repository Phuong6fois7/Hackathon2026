# Hackathon2026
** End-to-end project combining Data Engineering and Generative AI, using a biomedical article corpus inspired by PubMed.**

# Mission
Build a measured data-and-AI system for biomedical long documents:
- Ingest PubMed-style articles and structuring,
- Store them in Parquet format,
- Add new articles to the database,
- Summarize them with AI-generated,
- Search the corpus with RAG,
- Evaluate performance in terms of cost, latency, quality, and environmental impact.

# 1. Domain and Dataset Context
## 1.1. What is PubMed?
PubMed is a free resource for searching biomedical literature, mainly providing citations and abstracts rather than full-text articles. It is used for discovery, while PubMed Central offers free access to some full texts, and the hackathon dataset is a teaching version inspired by this system.
## 1.2. Dataset used in the Hackathon
We use the **ccdv/pubmed-summarization** dataset from **Hugging Face**:  
[PubMed Summarization Dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization)

This dataset is inspired by biomedical articles similar to those indexed in PubMed and is used for educational and experimentation purposes. It is provided in **Parquet format**, enabling efficient storage and processing within our data and AI pipeline.

# 3. Setup 
## 3.1. Recommended local setup

```bash
python -m venv .venv

source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install datasets pandas pyarrow polars codecarbon tiktoken

pip install sentence-transformers faiss-cpu scikit-learn streamlit matplotlib

pip freeze > requirements.txt