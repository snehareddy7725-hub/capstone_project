# Zepto Data Engineering & Analytics Project

This repository contains three complete modules for a data engineering and analytics pipeline.

## Modules

1. **Data Pipeline** (`/data_pipeline`): Web scraping, data cleaning, ETL, and SQL database
2. **Analytics Pipeline** (`/analytics`): EDA, data profiling, modeling, and evaluation
3. **Support Assistant** (`/support_assistant`): RAG-based customer support system

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd <repository-name>

# Module 1 - Data Pipeline
cd data_pipeline
pip install -r requirements.txt
python scrape_books.py

# Module 2 - Analytics Pipeline
cd ../analytics
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb

# Module 3 - Support Assistant
cd ../support_assistant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
