- Tập cơ sở dữ liệu NTCIR12: http://www.cs.rit.edu/~rlaz/NTCIR12_MathIR_WikiCorpus_v2.1.0.tar.bz2

- Tạo folder chứa cơ sở dữ liệu như sau: NTCIR-12_MathIR_Wikipedia_Corpus\MathTagArticles\wpmath...

- Tập 20 công thức truy vấn được cung cấp bởi cuộc thi NTCIR-12 nằm trong folder TestQueries, tập 8 công thức truy vấn tuỳ chỉnh nằm trong folder CustomQueries

- Mô hình chạy trên python 3.10, các thư viện được sử dụng trong file requirements.txt

- Chạy mô hình SLT: slt_run.py

- Chạy mô hình OPT: opt_run.py

- Chạy mô hình SLT-Type: slt_type_run.py

- Chạy mô hình kết hợp: combined_run.py

- Kết quả được lưu trong folder Retrieval_Results:
combined_res: kết quả tìm kiếm công thức của mô hình nhúng fastText (SLT + OPT + SLT-Type)

combined_res_2: kết quả tìm kiếm công thức của mô hình nhúng fastText (SLT + OPT)

custom_res: kết quả tìm kiếm công thức của mô hình nhúng fastText (SLT + OPT + SLT-Type) đối với tập 8 công thức truy vấn tự tạo tuỳ chỉnh (khác 20 công thức truy vấn có sẵn)

judge.dat: tập đánh giá (đã chuẩn hoá tên các tài liệu bị lỗi mã hoá tên)

opt_res.tsv: kết quả tìm kiếm công thức của mô hình nhúng fastText (OPT)

slt_res.tsv: kết quả tìm kiếm công thức của mô hình nhúng fastText (SLT)

slt_res_type.tsv: kết quả tìm kiếm công thức của mô hình nhúng fastText (SLT-Type)

approach0_res.dat: kết quả tìm kiếm công thức của mô hình approach0

tangents_res.dat: kết quả tìm kiếm công thức của mô hình tangent-S

cbow_res: folder chứa kết quả tìm kiếm công thức khi sử dụng mô hình CBOW (SLT + OPT)

euclidean_res: folder chứa kết quả tìm kiếm công thức khi sử dụng khoảng cách euclidean (SLT + OPT)

manhattan_res: folder chứa kết quả tìm kiếm công thức khi sử dụng khoảng cách manhattan (SLT + OPT)

- Công cụ hỗ trợ đánh giá kết quả tìm kiếm: Trec_eval tool (https://trec.nist.gov/trec_eval/)
