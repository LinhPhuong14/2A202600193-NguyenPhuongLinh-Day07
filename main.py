from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAILLM,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str], chunker: RecursiveChunker | None = None) -> list[Document]:
    """Load documents from file paths and optionally chunk them."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        
        if chunker:
            chunks = chunker.chunk(content)
            for i, chunk_text in enumerate(chunks):
                documents.append(
                    Document(
                        id=f"{path.stem}_v{i}",
                        content=chunk_text,
                        metadata={
                            "source": str(path), 
                            "doc_id": path.stem,
                            "chunk_index": i,
                            "extension": path.suffix.lower()
                        },
                    )
                )
        else:
            documents.append(
                Document(
                    id=path.stem,
                    content=content,
                    metadata={"source": str(path), "doc_id": path.stem, "extension": path.suffix.lower()},
                )
            )

    return documents


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Senior AI Engineer RAG Demo ===")
    print(f"Targeting Provider: {os.getenv(EMBEDDING_PROVIDER_ENV, 'mock')}")
    
    # 1. Setup Chunking Strategy (Optimal: Recursive)
    chunker = RecursiveChunker(chunk_size=600)
    
    # 2. Load and Chunk Documents
    docs = load_documents_from_files(files, chunker=chunker)
    if not docs:
        print("\nNo valid input files were loaded.")
        return 1

    print(f"\nIngested {len(docs)} chunks from {len(files)} files.")

    # 3. Setup Embedding Provider
    load_dotenv(override=True)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    
    llm_fn = None
    if provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
            llm_instance = OpenAILLM(model_name=os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL))
            llm_fn = llm_instance.__call__
            print(f"Using OpenAI: {OPENAI_EMBEDDING_MODEL} / {OPENAI_CHAT_MODEL}")
        except Exception as e:
            print(f"Error initializing OpenAI: {e}. Falling back to mocks.")
            embedder = _mock_embed
    elif provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
            print(f"Using Local Embedder: {LOCAL_EMBEDDING_MODEL}")
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed
        print("Using Mock Embeddings")

    # Mock LLM fallback
    if not llm_fn:
        def mock_llm(prompt: str) -> str:
            return f"[MOCK LLM] I don't have a real LLM configured. Prompt preview: {prompt[:100]}..."
        llm_fn = mock_llm

    # 4. Initialize Store and Add Documents
    store = EmbeddingStore(collection_name="senior_rag_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nVector DB Size: {store.get_collection_size()} chunks")
    
    # 5. Search Test
    print("\n=== Retrieval Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        score_val = result.get('score', 0)
        print(f"{index}. score={score_val:.3f} source={result['metadata'].get('source')}")
        print(f"   content: {result['content'][:150].replace(chr(10), ' ')}...")

    # 6. Agent Test (RAG)
    print("\n=== Agent Generation (RAG) ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("-" * 20)
    print(agent.answer(query, top_k=3))
    print("-" * 20)
    
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
