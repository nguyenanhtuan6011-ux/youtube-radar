import streamlit as st
from youtubesearchpython import VideosSearch
from collections import Counter
import pandas as pd
import re

# Danh sách từ khóa cơ bản cần bỏ qua (Stop words tiếng Anh)
STOP_WORDS = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'how', 'what', 'why', 'do', 'can', 'my', 'we', 'about', 'an', 'if', 'will', 'up', 'out', 'just', 'so', 'me', 'they', 'like', 'get', 'more', 'have'}

def la_tu_khoa_hop_le(word):
    # Lọc ký tự tiếng Trung, Nhật, Hàn
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u20000-\u2a6df]', word):
        return False
    # Chỉ lấy chữ cái, độ dài > 2, không nằm trong danh sách từ bỏ qua
    if len(word) > 2 and word not in STOP_WORDS:
        return True
    return False

def quet_tu_khoa_hien_dai(tu_khoa, so_luong=15, ma_quoc_gia="US"):
    try:
        videosSearch = VideosSearch(tu_khoa, limit=so_luong, region=ma_quoc_gia)
        ket_qua = videosSearch.result()['result']
        
        tat_ca_van_ban = ""
        for video in ket_qua:
            # Thu thập tiêu đề của các video top đầu
            title = video.get('title', '')
            tat_ca_van_ban += f" {title}"
            
        # Tách từ và làm sạch dữ liệu
        words = re.findall(r'\b[a-zA-Z]+\b', tat_ca_van_ban.lower())
        
        # Lọc ra các từ khóa cốt lõi
        keywords = [w for w in words if la_tu_khoa_hop_le(w)]
        return keywords
    except Exception as e:
        return []

# Giao diện chính của ứng dụng
st.set_page_config(page_title="Radar Quét Ngách", page_icon="🎯", layout="wide")
st.title("🎯 Radar Quét Hàng Loạt Từ Khóa Ngách")
st.markdown("Hệ thống phân tích Tiêu đề của Top 15 video dẫn đầu để tìm ra Keyword cốt lõi (Đã kích hoạt chế độ chống chặn).")

# Bố cục giao diện
col_input, col_market = st.columns([2, 1])

with col_input:
    tu_khoa = st.text_input("🔑 Nhập chủ đề bạn muốn phân tích:")

with col_market:
    lua_chon_thi_truong = st.selectbox(
        "🌍 Chọn thị trường mục tiêu:",
        ("Hoa Kỳ (US)", "Châu Âu (Anh Quốc - UK)", "Toàn cầu")
    )

ma_quoc_gia = "US"
if lua_chon_thi_truong == "Châu Âu (Anh Quốc - UK)":
    ma_quoc_gia = "GB"
elif lua_chon_thi_truong == "Toàn cầu":
    ma_quoc_gia = ""

if st.button("🚀 Bắt đầu quét hàng loạt"):
    if tu_khoa:
        with st.spinner(f'Đang quét Top 15 video thị trường {lua_chon_thi_truong} (Thuật toán mới cực nhẹ)...'):
            tong_hop_tags = quet_tu_khoa_hien_dai(tu_khoa, 15, ma_quoc_gia)
            
        if not tong_hop_tags:
            st.error("Không tìm thấy dữ liệu. Hãy kiểm tra lại kết nối hoặc thử một từ khóa khác!")
        else:
            st.success(f"🎉 Quét hoàn tất! Đã lấy dữ liệu chuẩn từ thị trường {lua_chon_thi_truong}.")
            dem_tags = Counter(tong_hop_tags)
            top_20_tags = dem_tags.most_common(20) 
            
            # Xuất file
            df_export = pd.DataFrame(top_20_tags, columns=['Từ khóa', 'Tần suất'])
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Tải xuống danh sách từ khóa (Mở bằng Excel)",
                data=csv,
                file_name=f'tu_khoa_ngach_{tu_khoa.replace(" ", "_")}.csv',
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
