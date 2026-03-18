# MÔ TẢ ASSIGNMENT

## Yêu cầu

- Tìm hiểu topic mà nhóm đã chọn, lựa chọn algorithm cụ thể trong đề cương môn học, có thể sử dụng các công nghệ mới, tuy nhiên, cần so sánh đánh giá với các kỹ thuật có trong môn học.
- Thực hiện các kỹ thuật của Khai phá dữ liệu (theo đề cương môn học).
  - Logistic Regression
  - Decision Tree (nâng cao với random forest)
  - Neutral Network
- Visualize kết quả và đưa ra kết luận (Decision making/Conclusion) từ vấn đề được đặt ra và giải quyết trong topic mà nhóm đã chọn.
  - Trực quan hóa kết quả của các thuật toán: đã giải quyết những vấn đề gì mà nhóm đặt ra.

## Outline bài tập lớn

1. TỔNG QUAN ĐỀ TÀI
   1.1 Vấn đề đặt ra
       - Thực trạng trì hoãn chuyến bay tại Mỹ năm 2015
       - Hạn chế của các phân tích thủ công truyền thống
       - Nhu cầu có hệ thống phân tích tự động và dự báo
   1.2 Mục tiêu
       - Xây dựng Data Warehouse chuẩn hóa dữ liệu hàng không
       - Cung cấp dashboard phân tích OLAP đa chiều
       - Dự báo khả năng delay chuyến bay mới
   1.3 Phạm vi
       - Dữ liệu: Kaggle US Flight Delays 2015
         (5.8M chuyến bay, 14 hãng, 322 sân bay)
       - Thời gian: năm 2015, nội địa Hoa Kỳ
   1.4 Đóng góp
       - ETL pipeline streaming xử lý 5.8M dòng, RAM ~200MB
       - So sánh 5 thuật toán Machine Learning
       - Dashboard tương tác 2 cấp phân tích

2. CƠ SỞ LÝ THUYẾT
   2.1 Data Warehouse và OLAP
       - Khái niệm Data Warehouse, Star Schema
       - OLAP và các phép phân tích đa chiều
   2.2 Tiền xử lý dữ liệu (theo slide thầy)
       - Data Cleaning: missing values, outliers, inconsistency
       - Data Integration: tích hợp 3 nguồn CSV
       - Data Transformation: normalization, feature construction
       - Data Reduction: attribute selection, sampling
   2.3 Các thuật toán khai phá dữ liệu
       - Logistic Regression
       - Random Forest
       - Neural Network
       - XGBoost / LightGBM
       - Các chỉ số đánh giá: AUC-ROC, F1, Precision, Recall

3. THIẾT KẾ HỆ THỐNG
   3.1 Kiến trúc tổng thể (Layer Architecture)
       - Sơ đồ 5 tầng: Sources → ETL → DW → Analytics → Mining
   3.2 Pipeline ETL
       - Thiết kế streaming chunk 100K dòng
       - Vectorized transform với numpy.select()
       - Chiến lược xử lý missing values, outliers
   3.3 Star Schema
       - Thiết kế fact_flights + 4 dimension tables
       - Lý do chọn Star Schema thay vì Snowflake
   3.4 OLAP Queries
       - 9 câu truy vấn phân tích đa chiều
       - Minh họa Data Cube theo tháng/quý/hãng/sân bay
   3.5 Giao diện Dashboard
       - Kiến trúc 2 cấp: Tổng quan và Nguyên nhân
       - Thiết kế bộ lọc động
   3.6 Mô hình khai phá dữ liệu
       - Feature engineering và lý do chọn 10 features
       - Kiến trúc từng thuật toán
       - Chiến lược xử lý mất cân bằng lớp (21% delay)

4. HIỆN THỰC VÀ THỰC NGHIỆM
   4.1 Môi trường thực nghiệm
       - Cấu hình máy, thư viện, phiên bản
   4.2 Kết quả ETL
       - Hình ảnh CSV thô → Database
       - Thống kê dữ liệu sau làm sạch
       - So sánh RAM trước/sau tối ưu
   4.3 Kết quả OLAP và Dashboard
       - Hình ảnh dashboard
       - Một số insight rút ra từ OLAP queries
   4.4 Kết quả Data Mining
       - Bảng so sánh 5 thuật toán
       - Biểu đồ ROC, Confusion Matrix, Feature Importance
       - Phân tích tại sao Gradient Boosting thắng
   4.5 Thảo luận
       - Trả lời câu hỏi ban đầu: giải quyết được vấn đề gì?
       - Những trường hợp model còn sai, tại sao

5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
   5.1 Kết luận
       - Tóm tắt những gì đạt được so với mục tiêu
       - Insight thực tế rút ra từ dữ liệu 2015
   5.2 Hạn chế
       - Dữ liệu chỉ có 2015
       - SQLite không phù hợp production
       - AUC 0.73 còn hạn chế
   5.3 Hướng phát triển
       - Mở rộng dữ liệu nhiều năm
       - Deploy lên cloud
       - Cải thiện model với SHAP/XAI
