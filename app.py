import streamlit as st
import yt_dlp
from collections import Counter
import pandas as pd

# Hàm quét và bóc tách dữ liệu trực tiếp bằng yt-dlp để chống chặn
def quet_youtube(tu_khoa, so_luong=10):
    danh_sach_tags = []
    query = f"ytsearch{so_luong}:{tu_khoa}"
    
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'extract_flat': False, 
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if 'entries' in info:
                for video in info['entries']:
                    tags = video.get('tags', [])
                    if tags:
                        danh_sach_tags.extend(tags)
            return danh_sach_tags
    except Exception as e:
        return []

# Giao diện chính của ứng dụng
st.set_page_config(page_title="Radar Quét Ngách", page_icon="🎯", layout="wide")
st.title("🎯 Radar Quét Hàng Loạt Từ Khóa Ngách")
st.markdown("Nhập một chủ đề, hệ thống sẽ tìm 10 video top đầu và tổng hợp từ khóa đối thủ đang dùng.")

# Ô nhập liệu
tu_khoa = st.text_input("🔑 Nhập chủ đề bạn muốn phân tích:")

if st.button("🚀 Bắt đầu quét hàng loạt"):
    if tu_khoa:
        with st.spinner(f'Đang lách qua tường lửa và quét ngầm YouTube cho "{tu_khoa}". Quá trình này mất khoảng 30s - 1 phút...'):
            tong_hop_tags = quet_youtube(tu_khoa, 10)
            
        if not tong_hop_tags:
            st.error("Không tìm thấy từ khóa nào hoặc hệ thống tạm thời bị YouTube chặn. Hãy thử lại từ khóa khác!")
        else:
            st.success("🎉 Quét hoàn tất! Dưới đây là kết quả phân tích thị trường ngách của bạn.")
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
