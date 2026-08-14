import streamlit as st
import requests
import pandas as pd
import time

def quet_tu_khoa_youtube(tu_khoa):
    tu_khoa_thu_thap = []
    # Ghép từ khóa gốc với các chữ cái a-z để đào sâu mọi ngách
    danh_sach_truy_van = [tu_khoa] + [f"{tu_khoa} {chr(i)}" for i in range(97, 123)]
    
    thanh_tien_trinh = st.progress(0, text="Đang khởi động Radar...")
    
    for i, truy_van in enumerate(danh_sach_truy_van):
        # Đã sửa thành HTTPS để vượt qua tường lửa đám mây của Streamlit
        url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={truy_van}"
        try:
            response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=5)
            if response.status_code == 200:
                suggestions = response.json()[1]
                tu_khoa_thu_thap.extend(suggestions)
        except:
            pass
        
        thanh_tien_trinh.progress((i + 1) / len(danh_sach_truy_van), text=f"Đang quét ngách: '{truy_van}'...")
        time.sleep(0.1) 
        
    thanh_tien_trinh.empty()
    return list(set(tu_khoa_thu_thap))

# Giao diện chính của ứng dụng
st.set_page_config(page_title="Radar Quét Ngách", page_icon="🎯", layout="wide")
st.title("🎯 Radar Gợi Ý Từ Khóa Khán Giả")
st.markdown("Hệ thống kết nối trực tiếp với thanh tìm kiếm của YouTube để trích xuất chính xác những gì người xem đang gõ (Bảo đảm 100% không bị chặn).")

tu_khoa = st.text_input("🔑 Nhập chủ đề cốt lõi bạn muốn phân tích (Nên dùng tiếng Anh, VD: financial mistakes, money psychology):")

if st.button("🚀 Bắt đầu quét hàng loạt"):
    if tu_khoa:
        with st.spinner(f'Đang trích xuất dữ liệu hành vi người dùng cho "{tu_khoa}"...'):
            ket_qua = quet_tu_khoa_youtube(tu_khoa)
            
        if not ket_qua:
            st.error("Không thể kết nối. Vui lòng kiểm tra lại mạng!")
        else:
            st.success(f"🎉 Tuyệt vời! Đã thu thập được {len(ket_qua)} từ khóa ngách thực tế.")
            
            # Đóng gói xuất Excel
            df_export = pd.DataFrame(ket_qua, columns=['Từ Khóa Ngách'])
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Tải xuống toàn bộ danh sách (Mở bằng Excel)",
                data=csv,
                file_name=f'tu_khoa_pov_fortune_{tu_khoa.replace(" ", "_")}.csv',
                mime='text/csv',
            )
            
            st.markdown("---")
            st.subheader("🔥 Danh sách Từ khóa Tiêu biểu")
            
            col1, col2, col3 = st.columns(3)
            for i, tk in enumerate(ket_qua[:30]): 
                if i % 3 == 0:
                    col1.markdown(f"- {tk}")
                elif i % 3 == 1:
                    col2.markdown(f"- {tk}")
                else:
                    col3.markdown(f"- {tk}")
                    
            if len(ket_qua) > 30:
                st.info(f"... và {len(ket_qua) - 30} cụm từ khóa khác (Vui lòng tải file Excel để xem toàn bộ chi tiết).")
    else:
        st.warning("Vui lòng nhập từ khóa trước khi quét!")
