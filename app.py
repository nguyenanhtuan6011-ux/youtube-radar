import streamlit as st
import yt_dlp
from youtubesearchpython import VideosSearch
from collections import Counter
import pandas as pd

# Hàm 1: Tìm kiếm 10 link video từ 1 chủ đề lớn
def tim_kiem_video(tu_khoa, so_luong=10):
    try:
        videosSearch = VideosSearch(tu_khoa, limit=so_luong)
        ket_qua = videosSearch.result()
        danh_sach_link = []
        for video in ket_qua['result']:
            danh_sach_link.append(video['link'])
        return danh_sach_link
    except Exception as e:
        return []

# Hàm 2: Bóc tách thẻ tag của video
def lay_tags_video(url):
    ydl_opts = {'skip_download': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get('tags', [])
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
        with st.spinner(f'Đang tìm 10 video hàng đầu cho "{tu_khoa}"...'):
            links = tim_kiem_video(tu_khoa, 10)
            
        if not links:
            st.error("Không tìm thấy video nào. Hãy thử từ khóa khác!")
        else:
            tong_hop_tags = []
            thanh_tien_trinh = st.progress(0, text="Đang bóc tách dữ liệu hàng loạt. Vui lòng đợi...")
            
            for i, link in enumerate(links):
                tags = lay_tags_video(link)
                if tags:
                    tong_hop_tags.extend(tags)
                thanh_tien_trinh.progress((i + 1) / len(links), text=f"Đang bóc tách video {i+1}/10...")
            
            thanh_tien_trinh.empty() 
            
            if tong_hop_tags:
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
                st.warning("Các video top đầu không sử dụng thẻ tag nào.")
    else:
        st.warning("Vui lòng nhập từ khóa trước khi quét!")
