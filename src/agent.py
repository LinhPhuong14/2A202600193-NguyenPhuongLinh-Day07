from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Execute the RAG loop: Retrieve, Augment, Generate.
        """
        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)
        
        if not results:
            return "I'm sorry, I couldn't find any relevant information in my knowledge base to answer your question."

        # 2. Augment (Build Context with Metadata for Citation)
        context_parts = []
        for i, res in enumerate(results):
            # Senior AI Engineer: Extract source from metadata for proper citation
            source = res['metadata'].get('source', 'unknown')
            doc_id = res['metadata'].get('doc_id', 'unknown')
            score = res.get('score', 0.0)
            
            # Logging retrieval for quality monitoring (Senior practice)
            print(f"[Retrieval] Hit {i+1}: {doc_id} (score: {score})")
            
            context_parts.append(f"--- SOURCE: {source} (ID: {doc_id}) ---\n{res['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Professional RAG Prompt with Grounding and Citation Rules
        prompt = f"""You are a senior AI research assistant. Use the following pieces of context to answer the question at the end. 

STRICT RULES:
1. If the answer is not contained within the CONTEXT provided below, explicitly state "I don't know" or that the information is missing. Do NOT make up answers.
2. When you provide information from a specific context block, cite it using the format [Source: source_name].
3. Maintain a professional, technical tone.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # 3. Generate
        return self.llm_fn(prompt)
