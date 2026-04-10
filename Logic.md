# Logic & Cơ sở Toán học: Embedding, Vector Store & RAG

Chào mừng bạn đến với tài liệu giải thích kiến thức cốt lõi của Lab 07. Tài liệu này được thiết kế để giúp bạn hiểu sâu về bản chất toán học và logic lập trình phía sau hệ thống Tìm kiếm ngữ nghĩa (Semantic Search).

---

## 1. Không gian Vector & Embedding (Embedding Space)

### Khái niệm
Trong AI, **Embedding** là quá trình chuyển đổi một dữ liệu phi cấu trúc (như văn bản) thành một danh sách các số thực (Vector) trong không gian $n$-chiều.

$$f(\text{text}) \to \mathbf{V} = [x_1, x_2, \dots, x_n]$$

### Tại sao cần Embedding?
Máy tính không hiểu chữ cái, nó chỉ hiểu con số. Embedding giúp biểu diễn **ngữ nghĩa** sao cho các từ có ý nghĩa giống nhau sẽ nằm **gần nhau** trong không gian toán học.

---

## 2. Độ tương đồng Cosine (Cosine Similarity)

Đây là "thước đo" quan trọng nhất trong Lab này để so sánh hai đoạn văn bản.

### Công thức toán học
Độ tương đồng Cosine giữa hai vector $\mathbf{A}$ và $\mathbf{B}$ được tính bằng:

$$\text{Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\| + \epsilon}$$

Trong đó:
- $\mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i$: Tích vô hướng (Dot Product).
- $\|\mathbf{A}\| = \sqrt{\sum_{i=1}^{n} A_i^2}$: Độ dài (Norm) của vector $\mathbf{A}$.
- $\epsilon = 10^{-9}$: Một hằng số cực nhỏ (Epsilon) để đảm bảo **tính ổn định số học** (tránh lỗi chia cho 0).

### Tại sao dùng Cosine thay vì khoảng cách Euclid?
Khoảng cách Euclid bị ảnh hưởng bởi độ dài văn bản (Magnitude). Cosine chỉ quan tâm đến **hướng** (ngữ nghĩa), giúp hệ thống vẫn nhận diện tốt sự tương đồng kể cả khi một đoạn văn dài và một đoạn văn ngắn.

---

## 3. Thuật toán Recursive Chunking với Overlap

Để AI có thể đọc các tài liệu dài hàng nghìn trang, chúng ta phải chia nhỏ chúng (Chunking).

### Cơ chế chia nhỏ đệ quy (Recursive)
Thay vì cắt bừa bãi, thuật toán thử nghiệm các dấu phân tách theo thứ tự ưu tiên:
1.  `\n\n` (Đoạn văn)
2.  `\n` (Dòng)
3.  `. ` (Câu)
4.  ` ` (Từ)

### Toán học của Overlap (Cửa sổ trượt)
Khi chia nhỏ, chúng ta giữ lại một phần của chunk trước đưa vào chunk sau. 
Nếu $L$ là độ dài tài liệu, $S$ là kích thước chunk, $O$ là độ chồng lấp (Overlap):

$$num\_chunks = \lceil \frac{L - O}{S - O} \rceil$$

**Lý do:** Overlap giúp bảo toàn ngữ cảnh. Nếu một câu quan trọng bị cắt đôi, phần Overlap sẽ giúp cả hai chunk đều giữ được một phần thông tin đó, giúp Retrieval không bị "đứt gãy" ý nghĩa.

---

## 4. RAG - Retrieval Augmented Generation

Mục tiêu cuối cùng của Lab là xây dựng Agent theo mô hình RAG:

1.  **Retrieve (Truy xuất):** Dùng Query Embedding để tìm $k$ chunk có Cosine Similarity cao nhất từ Vector Store.
2.  **Augment (Làm giàu):** Đưa các chunk này vào Prompt làm ngữ cảnh (Context).
3.  **Generate (Tạo phản hồi):** Yêu cầu LLM trả lời dựa trên Context đó.

### Metadata Enrichment
Chúng ta không chỉ lưu văn bản mà còn lưu **Metadata** (nguồn, ngày tháng, ID). Điều này cho phép:
- **Filtering:** "Tìm cho tôi các luật về thuế *trong năm 2023*".
- **Citation:** "Dựa trên [Nguồn: tailieu.pdf], câu trả lời là...".

---

## 5. Tối ưu hóa hiệu năng (Senior Optimization)

Trong mã nguồn, các điểm tối ưu quan trọng bao gồm:
- **Regex Lookbehind:** Tránh ngắt câu sai ở các từ viết tắt ("Dr.", "Mr.").
- **Buffer Management:** Theo dõi độ dài buffer bằng biến thay vì nối chuỗi liên tục (giảm độ phức tạp từ $O(N^2)$ xuống $O(N)$).
- **Exact Deletion:** Xóa tài liệu theo `doc_id` trong metadata để đảm bảo sạch sẽ dữ liệu.

---
*Tài liệu soạn thảo bởi Senior AI Engineer - Lab 07.*
