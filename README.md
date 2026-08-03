# Zepto Data Engineering & Analytics Capstone

This repository contains three complete modules, forming an end-to-end pipeline from raw data collection through predictive modeling to a deployable GenAI service.

## Modules

| Module | Path | Description |
|---|---|---|
| 1. Data Pipeline | [`/data_pipeline`](./data_pipeline) | Scrapes books.toscrape.com, cleans and converts currency, loads into a normalized SQLite schema, and runs SQL + pandas queries |
| 2. Analytics Pipeline | [`/analytics`](./analytics) | EDA, missing-value handling, and a full classification + regression modeling pipeline on the Titanic dataset |
| 3. Support Assistant | [`/support_assistant`](./support_assistant) | A RAG-based customer support assistant using ChromaDB, LangGraph, and FastAPI, with a fully offline deterministic mock mode |

Each module has its own README with install/run steps, design decisions, and (where applicable) example outputs — see the links above.

## Requirements

- Python **3.12+** (Module 3's dependencies — `fastapi`, `langgraph`, `sentence-transformers`, `torch` — require Python ≥3.10; 3.12 is what this project was built and tested against)
- ~4GB free disk space (Module 3 pulls in `torch` and downloads the `all-MiniLM-L6-v2` embedding model on first run)
- Docker (optional, only needed to build/run Module 3's container)

## Setup

All three modules share a single virtual environment and a single `requirements.txt` at the project root.

```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install all dependencies for all three modules
pip install -r requirements.txt
```

## Running each module

With the venv above activated:

```bash
# Module 1 — Data Pipeline
cd data_pipeline
python3 scrape_books.py

# Module 2 — Analytics Pipeline
cd ../analytics
jupyter notebook
# then run 01_eda.ipynb followed by 02_modeling.ipynb, in that order

# Module 3 — Support Assistant
cd ../support_assistant
python3 main.py
# API docs available at http://localhost:8000/docs once running
```

### Module 3 via Docker (optional, locally-runnable baseline)

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 8000:8000 zepto-support-assistant
```

## Repository structure

```
capstone_project/
├── requirements.txt          # shared across all three modules
├── .gitignore
├── data_pipeline/
│   ├── scrape_books.py
│   ├── books.db
│   └── README.md
├── analytics/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv            # raw offline fallback
│   ├── titanic_cleaned.csv
│   ├── best_pipeline.pkl
│   └── README.md
└── support_assistant/
    ├── main.py
    ├── docs/                  # 8 policy documents (corpus)
    ├── Dockerfile
    ├── requirements.txt        # copy of root file, needed for Docker build context
    ├── .env.example
    └── README.md
```

## Git workflow note

Per the assignment requirements, a feature branch was created, committed to at least twice, and merged back into `main` — visible in this repository's commit history (`git log --graph --all`).

## AI tool use disclosure

AI tools were used during development of this project. All code was reviewed, tested, and is understood by the author, who can explain any part of it on request, per the assignment's permitted-use terms.
