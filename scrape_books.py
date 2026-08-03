import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
from typing import Dict, List, Optional
import time
import os

class BookScraper:
    def __init__(self):
        self.base_url = "http://books.toscrape.com"
        self.books_data = []
        self.GBP_TO_INR = 105.50  # Fixed conversion rate
        
    def parse_price(self, price_text: str):
        """Extract and convert price to float"""
        match = re.search(r'\d+\.?\d*', price_text)
        if match:
            return float(match.group())
        return None
    
    def parse_rating(self, rating_text: str):
        """Convert text rating to integer (1-5)"""
        rating_map = {
            'One': 1, 'Two': 2, 'Three': 3, 
            'Four': 4, 'Five': 5
        }
        return rating_map.get(rating_text.strip())
    
    def parse_availability(self, avail_text: str):
        """Convert availability text to boolean"""
        return 'In stock' in avail_text
    
    def scrape_page(self, page_url: str):
        """Scrape a single page of books"""
        print(f"Scraping: {page_url}")
        response = requests.get(page_url)
        response.encoding = 'utf-8'  
        soup = BeautifulSoup(response.text, 'html.parser')
        
        books = []
        book_elements = soup.select('article.product_pod')
        
        for book in book_elements:
            try:
                # Extract data
                title = book.select('h3 a')[0].get('title')
                price_elem = book.select('p.price_color')[0]
                price_text = price_elem.text
                
                # Rating
                rating_class = book.select('p.star-rating')[0].get('class')[1]
                rating = self.parse_rating(rating_class)
                
                # Availability
                avail_elem = book.select('p.instock.availability')
                if avail_elem:
                    avail_text = avail_elem[0].text.strip()
                    in_stock = self.parse_availability(avail_text)
                else:
                    in_stock = False
                
                # Category (get from detail page)
                detail_url = book.select('h3 a')[0].get('href')
                detail_response = requests.get(f"{self.base_url}/catalogue/{detail_url}")
                detail_response.encoding = 'utf-8'
                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                
                category_elem = detail_soup.select('ul.breadcrumb li a')
                category = category_elem[-1].text if len(category_elem) > 2 else 'Unknown'
                
                # Store data
                book_data = {
                    'title': title,
                    'price_gbp': price_text,
                    'star_rating': rating_class,
                    'availability': avail_text if avail_elem else 'Out of stock',
                    'category': category,
                    'rating_int': rating,
                    'in_stock': in_stock
                }
                books.append(book_data)
                
                time.sleep(0.5)  # Be respectful to the server
                
            except Exception as e:
                print(f"Error scraping book: {e}")
                continue
        
        return books
    
    def scrape_all(self, num_pages: int = 5):
        """Scrape multiple pages from the catalog"""
        for page_num in range(1, num_pages + 1):
            if page_num == 1:
                url = f"{self.base_url}/catalogue/page-1.html"
            else:
                url = f"{self.base_url}/catalogue/page-{page_num}.html"
            
            page_books = self.scrape_page(url)
            self.books_data.extend(page_books)
    
    def clean_data(self):
        """Clean and transform scraped data"""
        df = pd.DataFrame(self.books_data)
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna(subset=['price_gbp', 'star_rating', 'rating_int'])
        
        # Convert price to float
        df['price_gbp'] = df['price_gbp'].apply(self.parse_price)
        df = df.dropna(subset=['price_gbp'])
        
        # Convert to INR
        df['price_inr'] = df['price_gbp'] * self.GBP_TO_INR
        
        # Ensure proper types
        df['rating'] = df['rating_int'].astype(int)
        df['in_stock'] = df['in_stock'].astype(bool)
        df['price_gbp'] = df['price_gbp'].astype(float)
        df['price_inr'] = df['price_inr'].astype(float)
        
        print(f"Cleaned data: {len(df)} rows (removed {initial_rows - len(df)} rows)")
        print(f"Distinct categories: {df['category'].nunique()} -> {sorted(df['category'].unique())}")
        assert df['category'].nunique() >= 3, "Need at least 3 distinct categories"
        return df
    
    def create_database(self, df: pd.DataFrame, db_path: str = 'books.db'):
        """Create normalized SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE
            )
        ''')
        
        # Create books table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                price_gbp REAL,
                price_inr REAL,
                rating INTEGER,
                in_stock INTEGER,
                category_id INTEGER,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            )
        ''')
        
        # Insert categories
        categories = df['category'].unique()
        for category in categories:
            cursor.execute(
                "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
                (category,)
            )
        
        # Get category IDs
        cursor.execute("SELECT category_id, category_name FROM categories")
        category_map = {name: id for id, name in cursor.fetchall()}
        
        # Insert books
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO books 
                (title, price_gbp, price_inr, rating, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                row['title'],
                row['price_gbp'],
                row['price_inr'],
                row['rating'],
                int(row['in_stock']),
                category_map[row['category']]
            ))
        
        conn.commit()
        conn.close()
        print(f"Database created at {db_path} with {len(df)} books")
    
    def run_queries(self, db_path: str = 'books.db'):
        """Execute and display SQL queries"""
        conn = sqlite3.connect(db_path)
        
        queries = [
            # SELECT/WHERE
            ("Top 10 most expensive books",
             "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10"),
            
            # ORDER BY
            ("Books sorted by rating (highest first)",
             "SELECT title, rating FROM books ORDER BY rating DESC LIMIT 5"),
            
            # LIMIT
            ("First 5 books in database",
             "SELECT title FROM books LIMIT 5"),
            
            # DISTINCT
            ("Unique categories",
             "SELECT DISTINCT c.category_name FROM categories c"),
            
            # BETWEEN
            ("Books with price between 10 and 20 GBP",
             "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20"),
            
            # JOIN
            ("Books with category names",
             "SELECT b.title, c.category_name, b.price_gbp FROM books b "
             "JOIN categories c ON b.category_id = c.category_id LIMIT 10")
        ]
        
        print("\n" + "="*60)
        print("EXECUTING SQL QUERIES")
        print("="*60)
        
        for description, query in queries:
            print(f"\n{description}:")
            print("-" * 40)
            result = pd.read_sql_query(query, conn)
            print(result)
            print()
        
        # Verify pd.read_sql vs pd.merge produce equivalent results for the JOIN query
        print("\n" + "="*60)
        print("VERIFYING pd.read_sql vs pd.merge EQUIVALENCE")
        print("="*60)

        books_df = pd.read_sql_query("SELECT * FROM books", conn)
        categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

        merged_df = pd.merge(
            books_df, categories_df, on='category_id'
        )[['title', 'category_name', 'price_gbp']].head(10)

        sql_join_result = pd.read_sql_query(
            "SELECT b.title, c.category_name, b.price_gbp FROM books b "
            "JOIN categories c ON b.category_id = c.category_id LIMIT 10", conn
        )

        print("\nSQL JOIN result (pd.read_sql):")
        print(sql_join_result)
        print("\npd.merge result (in-memory, no SQL):")
        print(merged_df)

        are_equal = sql_join_result.reset_index(drop=True).equals(merged_df.reset_index(drop=True))
        print(f"\nEquivalent: {are_equal}")

        conn.close()

def main():
    print("Starting book scraper...")
    scraper = BookScraper()
    
    print("Scraping books...")
    scraper.scrape_all(num_pages=5)
    
    print(f"Scraped {len(scraper.books_data)} books")
    assert len(scraper.books_data) >= 60, (
        f"Only scraped {len(scraper.books_data)} books, need >= 60. "
        "Increase num_pages or check scrape_page() for silent errors above."
    )
    
    print("Cleaning data...")
    df = scraper.clean_data()
    
    print("Creating database...")
    scraper.create_database(df)
    print(f"books.db file size: {os.path.getsize('books.db')} bytes")
    
    print("Running queries...")
    scraper.run_queries()
    
    print("\nDone!")

if __name__ == "__main__":
    main()