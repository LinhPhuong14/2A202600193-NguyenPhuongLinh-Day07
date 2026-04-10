from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Fixed-width lookbehind chains (Python 're' limitation)
        # Handle 2-char, 3-char and 4-char abbreviations separately
        # Group 1: 2-char (Mr, St, Dr, Ms, Jr, vs)
        # Group 2: 3-char (Mrs, etc)
        # Group 3: 4-char (Prof)
        lookbehind = (
            r'(?<!\bMr)(?<!\bSt)(?<!\bDr)(?<!\bMs)(?<!\bJr)(?<!\bvs)'
            r'(?<!\bMrs)(?<!\betc)'
            r'(?<!\bProf)'
        )
        sentence_pattern = f'{lookbehind}\\.(?=\\s+|$)|(?<=[!?])\\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = ". ".join(sentences[i : i + self.max_sentences_per_chunk])
            # Ensure the chunk ends with a period if it was split on one
            if not chunk.endswith((".", "!", "?")):
                chunk += "."
            chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500, overlap: int = 50) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2) # Safety cap: overlap shouldn't exceed half the size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """
        Thuật toán chia nhỏ đệ quy tối ưu:
        1. Nếu đoạn văn bản đủ nhỏ -> Trả về luôn.
        2. Nếu hết dấu phân tách -> Chia cứng theo độ dài.
        3. Chia theo dấu phân tách ưu tiên nhất, sau đó gộp lại thành các chunk kèm overlap.
        """
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        sep_len = len(separator)

        # Chia nhỏ bản văn theo separator hiện tại
        parts = current_text.split(separator)
        
        final_chunks: list[str] = []
        buffer: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part)
            
            # Nếu một phần đơn lẻ quá lớn -> Đệ quy sâu hơn vào phần đó
            if part_len > self.chunk_size:
                if buffer:
                    final_chunks.extend(self._process_buffer(buffer, separator, next_separators))
                    buffer = []
                    current_len = 0
                final_chunks.extend(self._split(part, next_separators))
                continue

            # Tính toán độ dài mới nếu thêm phần này vào buffer
            # (current_len + part_len + separator_len)
            new_len = current_len + part_len + (sep_len if buffer else 0)

            if new_len > self.chunk_size:
                # Flush buffer thành một chunk
                if buffer:
                    final_chunks.extend(self._process_buffer(buffer, separator, next_separators))
                
                # Tính toán Overlap: Lấy các phần cuối của buffer cũ để đưa vào buffer mới
                # Đảm bảo tổng độ dài overlap không vượt quá self.overlap
                overlap_buffer: list[str] = []
                overlap_len = 0
                for p in reversed(buffer):
                    p_len = len(p)
                    potential_len = overlap_len + p_len + (sep_len if overlap_buffer else 0)
                    if potential_len <= self.overlap:
                        overlap_buffer.append(p)
                        overlap_len = potential_len
                    else:
                        break
                
                buffer = list(reversed(overlap_buffer))
                buffer.append(part)
                current_len = overlap_len + sep_len + part_len if len(buffer) > 1 else part_len
            else:
                buffer.append(part)
                current_len = new_len

        if buffer:
            final_chunks.extend(self._process_buffer(buffer, separator, next_separators))

        return final_chunks

    def _process_buffer(self, buffer: list[str], separator: str, next_separators: list[str]) -> list[str]:
        """Tạo chunk từ buffer hoặc đệ quy tiếp nếu cần (Senior Optimization)."""
        if not buffer:
            return []
        
        chunk_text = separator.join(buffer)
        if len(chunk_text) <= self.chunk_size:
            return [chunk_text]
        # Trường hợp hy hữu: một dấu phân tách cấp cao lại chứa các đoạn văn quá dài
        return self._split(chunk_text, next_separators)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b|| + epsilon)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    
    epsilon = 1e-9
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    
    if norm_a < epsilon or norm_b < epsilon:
        return 0.0
        
    return dot_product / (norm_a * norm_b + epsilon)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            results[name] = {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
        return results
