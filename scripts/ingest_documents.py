"""CLI script to ingest PDF documents."""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_config, setup_logging, save_json, ensure_dir
from src.ingestion import PDFParser, TableExtractor, OCRProcessor, SmartChunker

logger = setup_logging(level="INFO")


def ingest_documents(input_dir: str, output_dir: str = None):
    """
    Ingest PDF documents from directory.
    
    Args:
        input_dir: Input directory with PDF files
        output_dir: Output directory for processed chunks
    """
    config = get_config()
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if output_dir is None:
        output_dir = config.processed_data_path
    else:
        output_dir = Path(output_dir)
    
    ensure_dir(output_dir)
    
    # Initialize components
    logger.info("Initializing ingestion pipeline...")
    pdf_parser = PDFParser()
    table_extractor = TableExtractor()
    ocr_processor = OCRProcessor(languages=['en'], gpu=False)
    chunker = SmartChunker(
        chunk_size=config.text_chunk_size,
        chunk_overlap=config.text_overlap
    )
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"\nProcessing: {pdf_file.name}")
            
            # Parse PDF
            doc_data = pdf_parser.parse(str(pdf_file))
            
            # Extract tables
            logger.info("Extracting tables...")
            tables = table_extractor.extract_tables(str(pdf_file))
            
            # Chunk document
            logger.info("Chunking document...")
            text_chunks = chunker.chunk_document(doc_data)
            
            # Chunk tables
            if tables:
                table_chunks = chunker.chunk_tables(tables)
                text_chunks.extend(table_chunks)
            
            # Process OCR for image chunks
            logger.info("Processing OCR for images...")
            ocr_results = {}
            for chunk in text_chunks:
                if chunk.get("chunk_type") == "image":
                    try:
                        image_data = chunk.get("image_data", {})
                        if image_data.get("image_bytes"):
                            ocr_result = ocr_processor.process_image_bytes(
                                image_data["image_bytes"]
                            )
                            ocr_results[chunk["chunk_id"]] = ocr_result
                    except Exception as e:
                        logger.warning(f"OCR failed for chunk {chunk['chunk_id']}: {str(e)}")
            
            # Add OCR text to image chunks
            if ocr_results:
                text_chunks = chunker.add_ocr_to_image_chunks(text_chunks, ocr_results)
            
            all_chunks.extend(text_chunks)
            logger.info(f"Created {len(text_chunks)} chunks for {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    
    # Save all chunks
    output_file = output_dir / "chunks.json"
    logger.info(f"\nSaving {len(all_chunks)} chunks to {output_file}")
    
    # Remove image bytes before saving (too large for JSON)
    for chunk in all_chunks:
        if "image_data" in chunk and "image_bytes" in chunk["image_data"]:
            del chunk["image_data"]["image_bytes"]
    
    save_json(all_chunks, output_file)
    logger.info("Ingestion complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest PDF documents")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/",
        help="Input directory with PDF files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for processed chunks"
    )
    
    args = parser.parse_args()
    
    ingest_documents(args.input, args.output)


if __name__ == "__main__":
    main()
