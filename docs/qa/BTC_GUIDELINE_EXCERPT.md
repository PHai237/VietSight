# Quy ước BTC — nội dung người dùng cung cấp

Nhận trong chat ngày **27/11/2026, GMT+7**, sau snapshot review ban đầu. Người dùng xác nhận đây là phần “Tóm tắt quy ước gán nhãn” trên website cuộc thi. Chưa có URL hoặc revision của trang để đối chiếu online. Nội dung dưới đây được dùng làm căn cứ ngữ nghĩa cho QA; không sửa `docs/LABEL_GUIDELINE.md` vì đó là tài liệu chung Claude đang phụ trách.

1. **Motorcyclist/Motorcycle**: Có người ngồi trên xe, gồm dừng đèn đỏ, người ngồi sau, trẻ em → một instance xe + mọi người ngồi trên xe, class Motorcyclist. Xe không người/đỗ/đang được dắt → Motorcycle; người dắt → Pedestrian, hai instance riêng.
2. **Cyclist/Bicycle**: Cùng quy ước người + xe. Xe đạp điện dạng xe máy điện không bàn đạp → Motorcycle/Motorcyclist.
3. **Đầu kéo/rơ-moóc**: Đầu kéo → Truck; rơ-moóc hoặc sơ-mi rơ-moóc, kể cả container → Other vehicle; **hai instance riêng**. Ba gác, xích lô, xe công trình → Other vehicle.
4. **Visible mask**: Chỉ phần nhìn thấy, các instance không chồng nhau. Vật phía trước giữ pixel; không gồm bóng đổ, phản chiếu hoặc ánh đèn pha. Vật bị cột/dây che vẫn một instance.
5. **Sign/Traffic light**: Mỗi tấm biển một instance, chỉ mặt biển, không cột; không gán biển quảng cáo/cửa hàng. Mỗi hộp đèn giao thông một instance, phủ cả hộp; không gán đèn xe/đèn trang trí.
6. **Không gán**: vật sau kính cửa hàng, người trong ô tô thấy qua kính, hình trên áp phích, động vật, xe đẩy tay; đối tượng quá nhỏ với cạnh ngắn dưới khoảng **1,25% cạnh dài ảnh**. Ngưỡng hình học này không đồng nhất với bộ lọc diện tích 64 pixel ở 640 của notebook; không thay validator để hợp nhất hai quy tắc.
7. **Cây quyết định**: Đoạn trích ban đầu nói xe hai bánh theo có người/không người; xe từ bốn bánh nhỏ dưới khoảng 9 chỗ → Car, kiểu xe buýt → Bus, xe tải/đầu kéo → Truck, còn lại → Other vehicle. **Cập nhật theo chỉ dẫn trực tiếp tiếp theo của người dùng: xe 9–15 chỗ thuộc Car.** Dùng quy tắc mới này khi đối chiếu; không xem “dưới 9 chỗ” là giới hạn trên hiện hành của Car. Ghi quyết định khó trong quy ước đội để nhất quán. Đoạn được cung cấp chưa nêu trực tiếp chữ “pickup”; phải xác định đúng loại xe theo guideline đã duyệt, không đổi mọi source `truck` theo số ghế.

Các điểm chưa có trong đoạn trích: mặt sau biển/đèn, hàng hóa trên xe máy, quyền sử dụng dataset. Giữ các điểm này pending thay vì suy diễn.
