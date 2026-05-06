# Hackathon2026
**End-to-end project combining Data Engineering and Generative AI, using a biomedical article corpus inspired by PubMed.**

<p align="center">
  <img src="assets/logo_pub_med.png" alt="logo_pub_med" />
</p>

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

# 2. Core Concepts

## 2.1. Data Engineering Concepts

| Concept | Description | Usage in Project |
|---|---|---|
| **Bronze / Silver / Gold** | Layered data architecture for increasing data quality and usability. | Bronze = raw data, Silver = cleaned data, Gold = embeddings, summaries, metadata. |
| **Schema Management** | Defines columns, data types, and constraints. | Prevents pipeline failures when new biomedical articles are added. |
| **Parquet Format** | Compressed columnar storage optimized for analytics. | Main storage format for efficient processing and reduced storage footprint. |
| **Append Strategy** | Adds new records without recomputing the entire dataset. | Simulates continuous ingestion of newly published articles. |
| **Idempotent Pipelines** | Running the pipeline multiple times should not duplicate or corrupt data. | Uses deterministic IDs and duplicate checks. |
| **Batch Processing** | Processes data in groups instead of one item at a time. | Optimizes embeddings generation, summarization, and API costs. |

## 2.2. AI Concepts

| Concept | Description | Project Implementation |
|---|---|---|
| **Summarization** | Generates concise versions of long biomedical articles. | Produces abstracts and compares them with reference summaries. |
| **Embeddings** | Numerical vector representations of semantic meaning. | Encodes article chunks for semantic retrieval. |
| **Vector Search** | Finds semantically similar text using embedding similarity. | Retrieves relevant biomedical article chunks. |
| **RAG** | Retrieval-Augmented Generation: retrieves context before generating an answer. | Answers biomedical questions using retrieved scientific content. |
| **Prompt Optimization** | Reduces prompt size to lower token usage, latency, cost, and energy consumption. | Compares long-context prompting with compact RAG-based prompting. |

## 2.3. Green AI & Sustainability Metrics

| Tool / Concept | Purpose | Expected Output |
|---|---|---|
| **CodeCarbon** | Measures energy consumption and carbon emissions of local computations. | Generates `emissions.csv` with duration, energy usage, and CO₂ estimates. |
| **EcoLogits** | Estimates environmental impact of LLM API calls. | Provides per-call ecological impact analysis. |
| **compar:IA** | Supports manual model comparison and environmental awareness. | Produces an evaluation table comparing quality and ecological indicators. |
| **Token Counting** | Measures prompt and completion size. | Tracks input tokens, output tokens, cost estimation, and prompt strategy. |

# 3. Setup 
## 3.1. Recommended local setup

```bash
python -m venv .venv

source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install datasets pandas pyarrow polars codecarbon tiktoken

pip install sentence-transformers faiss-cpu scikit-learn streamlit matplotlib

pip freeze > requirements.txt
