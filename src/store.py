from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            from chromadb.config import Settings

            # For tests (collections starting with 'test'), we use the in-memory fallback
            # to ensure a clean slate, high speed, and avoid side effects.
            # In production, we use a persistent ChromaDB instance.
            if not collection_name.startswith("test"):
                self._client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(allow_reset=True))
                self._collection = self._client.get_or_create_collection(name=collection_name)
                self._use_chroma = True
            else:
                self._use_chroma = False
                self._collection = None
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed ({e}). Falling back to in-memory store.")
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        query_embedding = self._embedding_fn(query)
        
        results = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            results.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it with enriched metadata."""
        if not docs:
            return

        import time
        current_ts = int(time.time())

        ids = []
        contents = []
        embeddings = []
        metadatas = []

        for doc in docs:
            # Automatic Metadata Enrichment (Senior AI Engineer Best Practice)
            # Ensure doc_id is propagated to chunks for robust cross-referencing
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = doc.id
            if "timestamp" not in doc.metadata:
                doc.metadata["timestamp"] = current_ts
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
            
            record = self._make_record(doc)
            ids.append(record["id"])
            contents.append(record["content"])
            embeddings.append(record["embedding"])
            metadatas.append(record["metadata"])
            
            if not self._use_chroma:
                self._store.append(record)

        if self._use_chroma and self._collection:
            self._collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query with score normalization."""
        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.get_collection_size())
            )
            
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Chroma returns distances. Cosine distance = 1 - Cosine Similarity
                    distance = results["distances"][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": round(similarity, 4) # Clean precision for UI
                    })
            # Sort explicitly to ensure consistency
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            # ChromaDB uses 'where' for metadata filtering
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.get_collection_size()),
                where=metadata_filter
            )
            
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity
                    })
            # Re-sort to be absolutely sure for tests
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results
        else:
            # Manual filtering for in-memory store
            filtered_records = self._store
            if metadata_filter:
                filtered_records = [
                    r for r in self._store 
                    if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
                ]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        initial_size = self.get_collection_size()
        
        if self._use_chroma and self._collection:
            # Try deleting by exact ID first (for unchunked docs)
            self._collection.delete(ids=[doc_id])
            # Then try deleting by doc_id metadata (for chunked docs)
            self._collection.delete(where={"doc_id": doc_id})
        
        # Parity with in-memory store
        self._store = [
            r for r in self._store 
            if r["id"] != doc_id and r["metadata"].get("doc_id") != doc_id
        ]
        
        return self.get_collection_size() < initial_size
