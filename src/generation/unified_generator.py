"""Unified generator with Mistral API and local fallback."""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from .mistral_generator import MistralGenerator
from .generator import Generator
from ..utils import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class UnifiedGenerator:
    """Generator that tries Mistral API first, falls back to local FLAN-T5."""
    
    def __init__(
        self,
        local_model_name: str = "google/flan-t5-large",
        use_local_fallback: bool = True,
        max_new_tokens: int = 512
    ):
        """
        Initialize unified generator.
        
        Args:
            local_model_name: Local fallback model name
            use_local_fallback: Whether to fallback to local model
            max_new_tokens: Maximum tokens to generate
        """
        self.logger = logger
        self.use_local_fallback = use_local_fallback
        self.max_new_tokens = max_new_tokens
        
        # Try to initialize Mistral
        self.mistral_generator = None
        self.local_generator = None
        
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        if mistral_api_key and mistral_api_key.strip():
            try:
                self.logger.info("Initializing Mistral API generator...")
                self.mistral_generator = MistralGenerator(
                    api_key=mistral_api_key,
                    model=os.getenv("MISTRAL_MODEL", "mistral-medium-latest"),
                    max_tokens=max_new_tokens
                )
                self.logger.info("Mistral API generator initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Mistral API: {e}")
                self.mistral_generator = None
        else:
            self.logger.info("Mistral API key not found in .env file")
        
        # Initialize local fallback if enabled
        if use_local_fallback or not self.mistral_generator:
            self.logger.info(f"Initializing local fallback generator: {local_model_name}")
            self.local_generator = Generator(
                model_name=local_model_name,
                max_new_tokens=max_new_tokens
            )
            self.logger.info("Local generator initialized successfully")
        
        # Determine which generator to use
        if self.mistral_generator:
            self.logger.info("PRIMARY: Mistral API | FALLBACK: Local FLAN-T5")
        else:
            self.logger.info("Using local FLAN-T5 only (Mistral not available)")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using Mistral API with local fallback.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        
        # Try Mistral API first
        if self.mistral_generator:
            try:
                self.logger.info("Generating with Mistral API...")
                result = self.mistral_generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                self.logger.info("Mistral API generation successful")
                return result
                
            except Exception as e:
                self.logger.warning(f"Mistral API failed: {e}")
                
                if not self.use_local_fallback or not self.local_generator:
                    self.logger.error("No fallback available")
                    raise Exception(f"Mistral API failed and fallback disabled: {e}")
                
                self.logger.info("Falling back to local FLAN-T5 model...")
        
        # Use local generator (either as fallback or primary)
        if self.local_generator:
            try:
                self.logger.info("Generating with local FLAN-T5 model...")
                result = self.local_generator.generate(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature is not None else 0.0
                )
                self.logger.info("Local generation successful")
                return result
                
            except Exception as e:
                self.logger.error(f"Local generation failed: {e}")
                raise Exception(f"Both Mistral and local generation failed: {e}")
        
        raise Exception("No generator available")
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_new_tokens: Optional[int] = None
    ) -> Dict:
        """
        Generate answer with source tracking and post-processing.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            max_new_tokens: Override max tokens
            
        Returns:
            Dictionary with answer and sources
        """
        from .prompts import create_grounded_prompt
        import re
        
        # Create prompt
        prompt = create_grounded_prompt(query, retrieved_chunks)
        
        # Generate answer using unified generator (tries Mistral first)
        self.logger.info(f"Generating answer for query: {query[:50]}...")
        raw_answer = self.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Post-process answer (same logic as local generator)
        # 1. Remove inline citations like [qatar_test_doc, page 14]
        answer = re.sub(r'\[\w+_[\w_]+,\s*page\s*\d+\]', '', raw_answer)
        
        # 2. Remove standalone page references
        answer = re.sub(r'\[page\s*\d+\]', '', answer)
        answer = re.sub(r'\s*\[\d+\]', '', answer)
        
        # 3. Remove incomplete fragments (like "2024H1." or short codes at end)
        answer = re.sub(r'\b\d{4}H\d\.?\s*$', '', answer)
        answer = re.sub(r'\b[A-Z]\d+\.?\s*$', '', answer)
        answer = re.sub(r'\b\w{1,3}\.?\s*$', '', answer)  # Remove 1-3 char fragments at end
        
        # 4. Clean up extra whitespace and newlines
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'\s*\.\s*\.', '.', answer)  # Remove double periods
        
        # 5. Ensure proper sentence ending
        if answer and answer[-1] not in '.!?':
            if not answer[-1].isdigit():
                answer += '.'
        
        # 6. Validate answer quality - must be substantial
        words = answer.split()
        if len(words) < 5 or not any(c.isalpha() for c in answer):
            answer = "I was unable to generate a complete answer from the available information."
        
        # Extract sources from chunks
        sources = []
        seen_pages = set()
        for chunk in retrieved_chunks[:10]:
            page = chunk.get("page_num") or chunk.get("page")  # Try both field names
            doc_id = chunk.get("doc_id", "unknown")
            chunk_type = chunk.get("chunk_type", "text")
            
            if page and page not in seen_pages:
                sources.append({
                    "doc_id": doc_id,
                    "page_num": page,
                    "chunk_type": chunk_type,
                    "score": chunk.get("rrf_score", 0)
                })
                seen_pages.add(page)
        
        # Add clean source reference to answer
        if sources and not any(word in answer.lower() for word in ["source:", "page", "pages"]):
            pages = sorted([s["page_num"] for s in sources])
            if len(pages) == 1:
                page_str = str(pages[0])
            elif len(pages) <= 3:
                page_str = ", ".join(map(str, pages))
            else:
                page_str = ", ".join(map(str, pages[:3])) + f", +{len(pages)-3} more"
            
            answer += f" (Source: pages {page_str})"
        
        return {
            "answer": answer,
            "sources": sources,
            "raw_answer": raw_answer
        }
