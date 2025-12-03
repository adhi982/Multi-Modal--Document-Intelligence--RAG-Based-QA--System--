"""Prompt templates for generation."""


GROUNDED_QA_PROMPT = """You are an expert financial analyst writing a clear, professional summary.

Context:
{context_with_citations}

Question: {user_query}

Critical Instructions:
1. Write ONLY complete sentences in natural language - never output codes, abbreviations, or fragments
2. Your answer must be 2-5 complete sentences that directly address the question
3. Include specific numbers and percentages naturally: "GDP growth was 2.2% in 2024"
4. When showing data: "The nominal GDP increased from 813.6 billion Qatari Riyals in 2024 to 828.9 billion in 2025"
5. Connect ideas smoothly using transitions like "Additionally," "Furthermore," "This represents"
6. Calculate changes when relevant: "an increase of 1.9%" or "grew by 15 billion"
7. Write as if explaining to a colleague - professional but conversational
8. Do NOT include any citations, page numbers, or document references in your answer
9. Do NOT end with incomplete phrases, codes like "2024H1", or abbreviations
10. End with a complete thought and proper punctuation

Write a complete, natural language answer:"""


SUMMARIZATION_PROMPT = """Summarize the following passages concisely, preserving key information:

{passages}

Summary:"""


MULTI_DOC_BRIEFING_PROMPT = """Create a brief executive summary of the following information:

{clustered_content}

Provide a concise briefing that covers the main points. Include source citations.

Briefing:"""


QUESTION_DECOMPOSITION_PROMPT = """Break down the following complex question into simpler sub-questions:

Question: {question}

Sub-questions:"""


def format_context_with_citations(chunks: list) -> str:
    """
    Format retrieved chunks with citation markers.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Formatted context string
    """
    formatted = []
    
    for i, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("doc_id", "unknown")
        page = chunk.get("page", "?")
        chunk_type = chunk.get("chunk_type", "text")
        content = chunk.get("content_text", "")
        
        # Special handling for tables - extract key information
        if chunk_type == "table":
            table_data = chunk.get("table_data", {})
            plain_text = table_data.get("plain_text", "")
            
            # Use plain_text directly with better formatting
            if plain_text:
                lines = [line.strip() for line in plain_text.split('\n') if line.strip()]
                
                # Add table header and structure
                formatted_lines = ["TABLE DATA:"]
                
                # Process each line to make it clearer
                for line in lines:
                    if ':' in line and any(char.isdigit() for char in line):
                        # This is a data row: "Metric: val1, val2, val3"
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            metric = parts[0].strip()
                            values = parts[1].strip()
                            formatted_lines.append(f"  • {metric}: {values}")
                        else:
                            formatted_lines.append(f"  {line}")
                    elif line.endswith(':'):
                        # Category header
                        formatted_lines.append(f"\n{line}")
                    elif any(char.isdigit() for char in line):
                        # Years or time periods
                        formatted_lines.append(f"  Years: {line}")
                
                content = '\n'.join(formatted_lines) if len(formatted_lines) > 1 else plain_text
            else:
                content = chunk.get("content_text", "")
        
        citation = f"[{doc_id}, page {page}]"
        
        # Add type indicator for tables
        if chunk_type == "table":
            formatted.append(f"[{i}] TABLE FROM PAGE {page}:\n{content}\n{citation}")
        else:
            formatted.append(f"[{i}] {content} {citation}")
    
    return "\n\n".join(formatted)


def create_grounded_prompt(query: str, chunks: list) -> str:
    """
    Create grounded QA prompt.
    
    Args:
        query: User query
        chunks: Retrieved chunks
        
    Returns:
        Formatted prompt
    """
    context = format_context_with_citations(chunks)
    return GROUNDED_QA_PROMPT.format(
        context_with_citations=context,
        user_query=query
    )


def create_summarization_prompt(passages: list) -> str:
    """
    Create summarization prompt.
    
    Args:
        passages: List of text passages
        
    Returns:
        Formatted prompt
    """
    formatted_passages = "\n\n".join([f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)])
    return SUMMARIZATION_PROMPT.format(passages=formatted_passages)


def create_briefing_prompt(clusters: list) -> str:
    """
    Create multi-document briefing prompt.
    
    Args:
        clusters: List of cluster dictionaries with content and sources
        
    Returns:
        Formatted prompt
    """
    formatted_clusters = []
    
    for i, cluster in enumerate(clusters, start=1):
        content = cluster.get("content", "")
        sources = cluster.get("sources", [])
        
        formatted_clusters.append(f"Topic {i}:\n{content}\nSources: {', '.join(sources)}")
    
    clustered_content = "\n\n".join(formatted_clusters)
    return MULTI_DOC_BRIEFING_PROMPT.format(clustered_content=clustered_content)
