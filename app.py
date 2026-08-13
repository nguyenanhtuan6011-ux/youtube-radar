import streamlit as st
import yt_dlp
from collections import Counter
import pandas as pd
import re

# Bộ lọc: Trả về False nếu phát hiện có ký tự tiếng Trung, Nhật, Hàn
def la_tu_khoa_hop_le(tag):
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\u20000-\u2a6df]', tag):
        return False
    return True

# Hàm quét và bóc tách dữ liệu ép theo quốc gia
def quet_youtube(tu_khoa, so_luong=10, ma_quoc_gia="US"):
    danh_sach_tags = []
    query = f"ytsearch{so_luong}:{tu_khoa}"
    
    # Cấu hình giả lập vị trí và trình duyệt
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'extract_flat': False,
        'http_headers': {
            'Accept-Language': 'en-US,en;q=0.9' # Ép YouTube ưu tiên tiếng Anh
        }
    }
    
    # Kích hoạt chuyển vùng nếu có chọn quốc gia
    if ma_quoc_gia:
        ydl_opts['geo_bypass_country'] = ma_quoc_gia
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info:
                for video in info['entries']:
                    tags = video.get('tags', [])
                    if tags:
                        # Lọc tiếng Trung
                        tags_hop_le = [t for t in tags if la_tu_khoa_hop_le(t)]
                        danh_sach_tags.extend(tags_hop_le)
            return danh_sach_tags
    except Exception as e:
        return []

# Giao diện chính của ứng dụng
st.set_page_config(page_title="Radar Quét Ngách", page_icon="🎯", layout="wide")
st.title("🎯 Radar Quét Hàng Loạt Từ Khóa Ngách")
st.markdown("Nhập một chủ đề, hệ thống sẽ tìm 10 video top đầu và tổng hợp từ khóa đối thủ đang dùng.")

# Bố cục giao diện: 2 cột (Nhập từ khóa và Chọn thị trường)
col_input, col_market = st.columns([2, 1])

with col_input:
    tu_khoa = st.text_input("🔑 Nhập chủ đề bạn muốn phân tích:")

with col_market:
    lua_chon_thi_truong = st.selectbox(
        "🌍 Chọn thị trường mục tiêu:",
        ("Hoa Kỳ (US)", "Châu Âu (Anh Quốc - UK)", "Toàn cầu")
    )

# Dịch lựa chọn thành mã quốc gia cho hệ thống
ma_quoc_gia = "US"
if lua_chon_thi_truong == "Châu Âu (Anh Quốc - UK)":
    ma_quoc_gia = "GB"
elif lua_chon_thi_truong == "Toàn cầu":
    ma_quoc_gia = ""

if st.button("🚀 Bắt đầu quét hàng loạt"):
    if tu_khoa:
        with st.spinner(f'Đang kết nối tới máy chủ {lua_chon_thi_truong} để quét từ khóa cho "{tu_khoa}"...'):
            tong_hop_tags = quet_youtube(tu_khoa, 10, ma_quoc_gia)
            
        if not tong_hop_tags:
            st.error("Không tìm thấy từ khóa nào phù hợp hoặc hệ thống tạm thời bị chặn. Hãy thử lại!")
        else:
            st.success(f"🎉 Quét hoàn tất! Đã lấy dữ liệu chuẩn từ thị trường {lua_chon_thi_truong}.")
            dem_tags = Counter(tong_hop_tags)
            top_20_tags = dem_tags.most_common(20) 
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔥 Top Từ Khóa Nổi Bật Nhất")
                for tag, count in top_20_tags:
                    st.markdown(f"- **{tag}** *(dùng {count} lần)*")
                    
            with col2:
                st.subheader("📊 Biểu đồ xu hướng")
                df = pd.DataFrame(top_20_tags, columns=['Từ khóa', 'Tần suất'])
                df = df.set_index('Từ khóa')
                st.bar_chart(df)
    else:
        st.warning("Vui lòng nhập từ khóa trước khi quét!")
