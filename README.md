# Recruitment Assessment: RAG-based Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot that answers questions from the provided FAQ knowledge base.

The current pipeline uses `data/FAQ.docx` as the main retrieval source.

Local embedding model files under `models/` are optional runtime artifacts and should not be committed to GitHub.

## Features

- Document ingestion pipeline for `.docx` (FAQ), with `.csv`/`.xlsx` fallback support
- Text chunking with overlap
- Embedding generation using `sentence-transformers` (with offline local hashing fallback)
- Local vector store using NumPy (cosine similarity retrieval)
- Gemini v2.5 Flash integration via **Python `requests` POST**
- Prompt-engineered answer generation grounded on retrieved context
- Basic guardrails for malicious/prompt-injection-style queries
- Minimal web interface using Streamlit

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   ├── FAQ.docx
│   └── create_sample_data.py      # optional utility, not used by default pipeline
├── scripts/
│   └── build_index.py
├── storage/                  # generated index files
└── src/
    ├── config.py
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── llm_client.py
    ├── prompting.py
    ├── guardrails.py
    ├── rag_pipeline.py
    └── types.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
copy .env.example .env
```

Then set values in `.env`:

- `GEMINI_API_KEY=<your_key>`
- `GEMINI_MODEL=gemini-2.5-flash`
- Optional: `TOP_K`, `EMBEDDING_MODEL`

## Build Vector Index

```bash
python scripts/build_index.py
```

The pipeline will load `data/FAQ.docx` by default.

This creates:

- `storage/vectors.npy`
- `storage/records.json`

If your environment cannot download models from Hugging Face (for example SSL/corporate certificate issues), indexing will automatically fall back to a local hashing embedder.

## Run Chatbot UI

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in terminal.

## Notes for Submission

- Keep source code in this repository.
- Include this `README.md` and `requirements.txt`.
- Deploy on Streamlit Cloud or Hugging Face Spaces and share the URL.
- Do not commit the confidential lesson-learned spreadsheet. It is excluded in [.gitignore](.gitignore).
- Do not commit local model files in the `models/` folder. They are excluded in [.gitignore](.gitignore).

## Data Source Notes

- Default source: `data/FAQ.docx`
- Optional utility: [data/create_sample_data.py](data/create_sample_data.py) (not used in the default FAQ pipeline)

## Upload to GitHub

1. Initialize Git in the project root:

```bash
git init
```

2. Review ignored files and current status:

```bash
git status
```

3. Add the public project files:

```bash
git add .
```

4. Create the first commit:

```bash
git commit -m "Initial RAG chatbot assessment submission"
```

5. Create an empty GitHub repository, then connect it:

```bash
git remote add origin <your-github-repo-url>
```

6. Push the project:

```bash
git branch -M main
git push -u origin main
```

Before pushing, confirm that the confidential spreadsheet does not appear in `git status`.

## Example Gemini API Request (via requests)

Implemented in `src/llm_client.py` using:

- `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}`
- JSON payload with `contents` and `generationConfig`
