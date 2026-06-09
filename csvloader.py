import pandas as pd
import os
from typing import List, Tuple, Dict

class CSVLoader:
    """Load and manage CNN-DailyMail dataset from CSV"""
    
    def __init__(self, csv_path: str):
        """
        Initialize the CSV loader
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load CSV into DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"[OK] Loaded {len(self.df)} articles from {os.path.basename(self.csv_path)}")
            # Menampilkan kolom yang terdeteksi untuk memudahkan debugging
            print(f"[OK] Columns detected: {list(self.df.columns)}") 
        except Exception as e:
            print(f"[ERROR] Error loading CSV: {e}")
            raise
            
    def _extract_text(self, row, possible_cols: List[str]) -> str:
        """Helper internal untuk mencari kolom dengan aman (Anti KeyError)"""
        for col in possible_cols:
            if col in self.df.columns:
                val = row[col]
                # Pastikan nilainya bukan NaN (Not a Number) atau kosong
                if pd.notna(val):
                    return str(val).strip()
        return ""

    def get_article_summary_pair(self, index: int) -> Tuple[str, str]:
        """
        Get article-summary pair by index
        """
        if index >= len(self.df):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.df)} rows")
        
        row = self.df.iloc[index]
        
        # Cari data artikel dengan berbagai variasi nama kolom
        article_cols = ['article', 'Article', 'text', 'Text', 'content', 'Content']
        article = self._extract_text(row, article_cols)
        
        # Cari data ringkasan dengan berbagai variasi nama kolom
        summary_cols = ['highlights', 'Highlights', 'summary', 'Summary', 'abstract', 'Abstract']
        summary = self._extract_text(row, summary_cols)
        
        return article, summary
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all article-summary pairs"""
        pairs = []
        for i in range(len(self.df)):
            try:
                article, summary = self.get_article_summary_pair(i)
                if article and summary:
                    pairs.append((article, summary))
            except Exception as e:
                continue
        return pairs
    
    def get_sample(self, n: int = 10) -> List[Dict]:
        """Get random sample of articles for testing"""
        sample_df = self.df.sample(min(n, len(self.df)), random_state=42)
        samples = []
        
        article_cols = ['article', 'Article', 'text', 'Text', 'content']
        summary_cols = ['highlights', 'Highlights', 'summary', 'Summary']
        
        for _, row in sample_df.iterrows():
            article = self._extract_text(row, article_cols)
            summary = self._extract_text(row, summary_cols)
            
            if article and summary:
                samples.append({
                    'article': article,
                    'summary': summary
                })
        return samples