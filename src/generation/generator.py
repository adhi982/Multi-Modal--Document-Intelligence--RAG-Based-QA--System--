"""FLAN-T5 based generation."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Optional

from ..utils import get_logger
from .prompts import create_grounded_prompt

logger = get_logger(__name__)


class Generator:
    """FLAN-T5 based answer generator."""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ):
        """
        Initialize generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (0 = deterministic)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.logger = logger
        
        # Auto-detect device with GPU priority
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.logger.warning("No GPU detected, using CPU")
        else:
            self.device = device
        
        self.logger.info(f"Loading generation model: {model_name} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use 8-bit quantization if on GPU to save memory
        if self.device == "cuda":
            try:
                self.logger.info("Loading model with 8-bit quantization for GPU efficiency")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
            except Exception as e:
                self.logger.warning(f"8-bit loading failed: {e}. Using standard loading.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Set to eval mode
        self.model.eval()
        
        self.logger.info("Generation model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        
        # Tokenize input with more space for context
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Generate with optimized parameters
        with torch.no_grad():
            if temp > 0:
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    do_sample=True,
                    num_beams=4,
                    repetition_penalty=1.2
                )
            else:
                # Beam search for better quality
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=0.6
                )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Generate grounded answer from query and retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        self.logger.info(f"Generating answer for query: {query[:50]}...")
        
        # Create grounded prompt
        prompt = create_grounded_prompt(query, retrieved_chunks)
        
        # Generate answer
        raw_answer = self.generate(prompt)
        
        # Post-process answer:
        # 1. Remove inline citations like [qatar_test_doc, page 14]
        import re
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
            # Don't add period if ends with a number (might be intentional)
            if not answer[-1].isdigit():
                answer += '.'
        
        # 6. Validate answer quality - must be substantial
        words = answer.split()
        if len(words) < 5 or not any(c.isalpha() for c in answer):
            answer = "I was unable to generate a complete answer from the available information."
        
        # 7. Add source citations at the end in a clean format
        if answer and "unable to" not in answer.lower() and retrieved_chunks:
            pages = sorted(set(chunk.get('page', '?') for chunk in retrieved_chunks[:3]))
            page_str = ', '.join(str(p) for p in pages)
            answer += f" (Source: pages {page_str})"
        
        # Extract sources
        sources = [
            {
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "chunk_type": chunk.get("chunk_type")
            }
            for chunk in retrieved_chunks
        ]
        
        result = {
            "answer": answer,
            "query": query,
            "sources": sources,
            "num_sources": len(sources),
            "context_length": len(prompt)
        }
        
        self.logger.info(f"Generated answer: {answer[:100]}...")
        return result
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            max_new_tokens: Override max tokens
            
        Returns:
            List of generated texts
        """
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                num_beams=1,
                early_stopping=True
            )
        
        # Decode all outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
