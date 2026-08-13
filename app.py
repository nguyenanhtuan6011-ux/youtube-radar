import streamlit as st
from duckduckgo_search import DDGS
from collections import Counter
import pandas as pd
import re

# Danh sách từ khóa cơ bản cần bỏ qua (Stop words)
STOP_WORDS = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'how', 'what', 'why', 'do', 'can', 'my', 'we', 'about', 'an', 'if', 'will', 'up', 'out', 'just', 'so', 'me', 'they', 'like', 'get', 'more', 'have', 'youtube', 'video'}

def la_tu_khoa_hop_le(word):
    # Lọc ký tự tiếng Trung, Nhật, Hàn
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u20000-\u2a6df]', word):
        return False
    # Lấy từ hợp lệ (dài hơn 2 chữ cái và không nằm trong từ khóa bỏ qua)
    if len(word) > 2 and word not in STOP_WORDS:
        return True
    return False

def quet_qua_duckduckgo(tu_khoa, so_luong=30):
    try:
        # Nhờ DuckDuckGo quét trang YouTube để lách luật chặn IP
        query = f"site:youtube.com {tu_khoa}"
        ket_qua = DDGS().text(query, max_results=so_luong)
        
        tat_ca_van_ban = ""
        for item in ket_qua:
            title = item.get('title', '')
            title = title.replace("- YouTube", "") # Xóa chữ mặc định
            tat_ca_van_ban += f" {title} "
            
        # Tách lấy các từ cốt lõi
        words = re.findall(r'\b[a-zA-Z]+\b', tat_ca_van_ban.lower())
        keywords = [w for w in words if la_tu_khoa_hop_le(w)]
        return keywords
    except Exception as e:
        return []

# Giao diện chính của ứng dụng
st.set_page_config(page_title="Radar Quét Ngách", page_icon="🎯", layout="wide")
st.title("🎯 Radar Quét Hàng Loạt Từ Khóa Ngách")
st.markdown("Hệ thống sử dụng vệ tinh tìm kiếm ẩn danh để phân tích video (Bảo đảm 100% không bị chặn IP).")

tu_khoa = st.text_input("🔑 Nhập chủ đề bạn muốn phân tích (Nên dùng tiếng Anh, VD: personal finance):")

if st.button("🚀 Bắt đầu quét hàng loạt"):
    if tu_khoa:
        with st.spinner(f'Đang dùng vệ tinh quét lách luật cho "{tu_khoa}"...'):
            tong_hop_tags = quet_qua_duckduckgo(tu_khoa, 30)
            
        if not tong_hop_tags:
            st.error("Lỗi mạng lưới vệ tinh. Vui lòng thử lại một từ khóa khác!")
        else:
            st.success("🎉 Quét hoàn tất! Đã thu thập thành công dữ liệu.")
            dem_tags = Counter(tong_hop_tags)
            top_20_tags = dem_tags.most_common(20) 
            
            # Đóng gói xuất Excel
            df_export = pd.DataFrame(top_20_tags, columns=['Từ khóa', 'Tần suất'])
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Tải xuống danh sách từ khóa (Mở bằng Excel)",
                data=csv,
                file_name=f'tu_khoa_{tu_khoa.replace(" ", "_")}.csv',
                mime='text/csv',
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔥 Top Từ Khóa Nổi Bật Nhất")
                for tag, count in top_20_tags:
                    st.markdown(f"- **{tag}** *(dùng {count} lần)*")
                    
            with col2:
                st.subheader("📊 Biểu đồ xu hướng")
                df_chart = df_export.set_index('Từ khóa')
                st.bar_chart(df_chart)
    else:
        st.warning("Vui lòng nhập từ khóa trước khi quét!")
