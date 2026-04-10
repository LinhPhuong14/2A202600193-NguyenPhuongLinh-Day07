# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Phương Linh - 2A202600193
**Nhóm:** 71
**Nhóm:** Ngô Văn Long, Nguyễn Phương Linh, Nguyễn Hải Đăng, Nguyễn Mạnh Phú
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ ($\theta \approx 0^\circ$), dẫn đến giá trị $\cos(\theta)$ gần bằng 1. Về mặt ngữ nghĩa, điều này cho thấy hai đoạn văn bản có sự tương đồng lớn về nội dung hoặc ngữ cảnh.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi rất thích lập trình Python vì nó dễ học."
- Sentence B: "Python là một ngôn ngữ lập trình tuyệt vời cho người mới bắt đầu."
- Lí do: Cả hai đều nói về việc Python là ngôn ngữ tốt và phù hợp để học/lập trình.

**Ví dụ LOW similarity:**
- Sentence A: "Thủ đô của Việt Nam là Hà Nội."
- Sentence B: "Bánh mì là món ăn đường phố nổi tiếng."
- Lí do: Hai câu nói về hai chủ đề hoàn toàn khác nhau (địa lý vs. ẩm thực), không có mối liên hệ ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào **hướng** của vector thay vì **độ dài** (magnitude). Trong NLP, một đoạn văn bản dài có thể có vector với độ dài lớn hơn nhưng hướng vẫn tương tự một đoạn ngắn có cùng nội dung. Cosine similarity giúp loại bỏ ảnh hưởng của độ dài văn bản, tập trung vào bản chất ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Dựa trên công thức cửa sổ trượt: $num\_chunks = \lceil \frac{L - O}{S - O} \rceil$
> Với $L = 10,000$, $S = 500$, $O = 50$:
> $num\_chunks = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$.
> **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap là 100: $\lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25$ chunks. Số lượng chunk tăng lên.
> Chúng ta muốn overlap nhiều hơn để đảm bảo không làm mất ngữ cảnh tại các điểm cắt (cắt giữa chừng một ý quan trọng), giúp retriever tìm kiếm chính xác hơn khi thông tin nằm trải dài qua ranh giới các chunk.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi sử dụng kỹ thuật **Regex Lookbehind/Lookahead** nâng cao để nhận diện ranh giới câu. Thay vì chỉ cắt tại dấu chấm, tôi xây dựng một chuỗi các xác nhận phủ định (Negative Lookbehind) để tránh ngắt câu sau các từ viết tắt phổ biến như "Mr.", "Dr.", "Prof.". Điều này giúp duy trì tính toàn vẹn ngữ pháp của mỗi đoạn trích.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Tôi triển khai thuật toán **Recursive Splitting với Overlap**. Thuật toán thử nghiệm các dấu phân tách theo độ ưu tiên ($\n\n \to \n \to . \to \text{space}$). Khi một chunk đạt giới hạn, nó sẽ lấy một phần dữ liệu từ buffer trước đó (Overlap) để đưa vào chunk tiếp theo, đảm bảo tính liên tục của thông tin. Tôi cũng tối ưu hóa hiệu năng bằng cách theo dõi độ dài buffer thay vì nối chuỗi liên tục.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tôi thực hiện **Làm giàu Metadata (Metadata Enrichment)** tự động bằng cách thêm `doc_id` và `timestamp`. Khi tìm kiếm, tôi chuẩn hóa điểm số từ khoảng cách của ChromaDB về Similarity ($1 - distance$) và sử dụng sắp xếp tường minh để đảm bảo kết quả ổn định nhất.

**`search_with_filter` + `delete_document`** — approach:
> Tôi sử dụng cơ chế lọc metadata đồng nhất giữa cả In-memory store và ChromaDB. Với `delete_document`, hệ thống sẽ xóa tất cả các chunk có `doc_id` tương ứng, đảm bảo tính nhất quán của dữ liệu khi tài liệu gốc bị gỡ bỏ.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi xây dựng một **Professional RAG Prompt** với các quy tắc nghiêm ngặt về "Grounding": chỉ trả lời dựa trên context, yêu cầu trích dẫn nguồn (citation) theo định dạng `[Source: name]` và hướng dẫn AI thừa nhận nếu thông tin bị thiếu thay vì tự ý suy diễn.

### Test Results

```
Ran 42 tests in 10.158s
OK
```

**Số tests pass:** 42 / 42

---
