"""Table extraction from PDFs using pdfplumber."""

import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


class TableExtractor:
    """Extract tables from PDF documents."""
    
    def __init__(self):
        """Initialize table extractor."""
        self.logger = logger
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        Extract all tables from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tables with metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Extracting tables from: {pdf_path.name}")
        
        all_tables = []
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            table_data = self._process_table(
                                table,
                                page_num,
                                table_idx,
                                pdf_path.stem
                            )
                            if table_data:
                                all_tables.append(table_data)
            
            self.logger.info(f"Extracted {len(all_tables)} tables from {pdf_path.name}")
            return all_tables
            
        except Exception as e:
            self.logger.error(f"Error extracting tables from {pdf_path}: {str(e)}")
            return []
    
    def _process_table(
        self,
        table: List[List[str]],
        page_num: int,
        table_idx: int,
        doc_id: str
    ) -> Optional[Dict]:
        """
        Process raw table data into structured format.
        
        Args:
            table: Raw table data from pdfplumber
            page_num: Page number
            table_idx: Table index on page
            doc_id: Document ID
            
        Returns:
            Processed table data or None if invalid
        """
        try:
            # Clean None values
            cleaned_table = []
            for row in table:
                cleaned_row = [cell if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)
            
            if len(cleaned_table) < 2:  # Need at least header + 1 row
                return None
            
            # Create DataFrame
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            
            # Remove empty rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                return None
            
            # Filter out garbage tables: minimum 3 rows, 2 columns
            if len(df) < 3 or len(df.columns) < 2:
                return None
            
            # Skip if all cells are empty/whitespace
            if df.astype(str).apply(lambda x: x.str.strip()).eq('').all().all():
                return None
            
            # Filter out chart/graph artifacts:
            # 1. Tables with mostly numeric-only rows (likely axis labels)
            # 2. Tables with very short cells (< 3 chars on average)
            # 3. Tables where most cells are just numbers without context
            non_empty_cells = df.astype(str).apply(lambda x: x.str.strip()).replace('', None).stack()
            if len(non_empty_cells) > 0:
                avg_cell_length = non_empty_cells.str.len().mean()
                # If average cell length is very short, likely a chart artifact
                if avg_cell_length < 3:
                    return None
                
                # Check if first column has meaningful labels (not just numbers/years)
                first_col = df.iloc[:, 0].astype(str).str.strip()
                first_col_words = first_col.str.split().str.len().mean()
                # If first column is mostly single words/numbers, likely not a real table
                if first_col_words < 1.5:
                    return None
            
            # Create a clean plain text representation optimized for LLM parsing
            plain_text = ""
            
            # Try to extract key-value pairs for better LLM comprehension
            for idx, row in df.iterrows():
                row_values = [str(val).strip() for val in row if str(val).strip() and str(val) != 'nan']
                
                if len(row_values) >= 2:
                    # First value is typically the label/metric
                    metric = row_values[0]
                    values = row_values[1:]
                    
                    # Format: "Metric: value1, value2, value3"
                    plain_text += f"{metric}: {', '.join(values)}\n"
                elif len(row_values) == 1:
                    # Single value row (might be a header or category)
                    plain_text += f"\n{row_values[0]}:\n"
            
            # Generate markdown representation (for display only)
            try:
                markdown = "TABLE:\n" + df.to_markdown(index=False, tablefmt='grid')
            except:
                # Fallback if markdown fails
                markdown = plain_text
            
            # Generate CSV representation
            csv = df.to_csv(index=False)
            
            # Generate CSV representation
            csv = df.to_csv(index=False)
            
            # Count tokens (rough estimate: 1 token ~= 4 characters)
            token_count = len(plain_text) // 4
            
            table_data = {
                "doc_id": doc_id,
                "page": page_num,
                "table_idx": table_idx,
                "type": "table",
                "markdown": markdown,
                "plain_text": plain_text,
                "csv": csv,
                "dataframe": df.to_dict('records'),
                "shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                },
                "token_count": token_count,
                "content_text": plain_text  # Use plain_text for embedding
            }
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error processing table on page {page_num}: {str(e)}")
            return None
    
    def extract_table_from_page(
        self,
        pdf_path: str,
        page_num: int
    ) -> List[Dict]:
        """
        Extract tables from specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            List of tables from the page
        """
        tables = []
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if 0 < page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    raw_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(raw_tables):
                        if table and len(table) > 0:
                            table_data = self._process_table(
                                table,
                                page_num,
                                table_idx,
                                Path(pdf_path).stem
                            )
                            if table_data:
                                tables.append(table_data)
        
        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_num}: {str(e)}")
        
        return tables
