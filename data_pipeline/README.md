# Data Pipeline Module

## Setup
1. Install dependencies: `pip install requests beautifulsoup4 pandas sqlite3`
2. Run scraper: `python scrape_books.py`

## Design Decisions
- **Scraping Scope**: Scraped first 5 paginated pages (60 books) from books.toscrape.com
- **Missing Data Handling**: Dropped rows with missing data (< 5% of dataset)
- **Currency Conversion**: Fixed rate: 1 GBP = 105.50 INR
- **Database Schema**: Normalized with categories and books tables

## Queries Implemented
1. SELECT/WHERE: Top 10 books by price
2. ORDER BY: Books sorted by rating
3. LIMIT: First 5 books
4. DISTINCT: Unique categories
5. BETWEEN: Books in price range
6. JOIN: Books with category names
