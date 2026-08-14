import streamlit as st
import requests
import pandas as pd
from collections import Counter
import re
import time
import io
from bs4 import BeautifulSoup
import docx

# Set up giao diện Streamlit
st.set_page_config(page_title="Radar Phân Tích Từ Khóa Đa Nguồn", page_icon="🎯", layout="wide")
st.title("🎯 Radar Phân Tích & Bóc Tách Từ Khóa Đa Nguồn")
st.markdown("Hỗ trợ trích xuất từ khóa từ **Thanh gợi ý YouTube**, **Kênh/Video YouTube**, **Đường link Website**, và **Tệp Office (Excel/Word/TXT)**.")

# Danh sách từ bỏ qua (Stopwords Anh & Việt)
STOP_WORDS = {
    # Tiếng Anh
    'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 
    'be', 'are', 'from', 'at', 'as', 'your', 'how', 'what', 'why', 'do', 'can', 'my', 'we', 'about', 'an', 'if', 'will', 
    'up', 'out', 'just', 'so', 'me', 'they', 'like', 'get', 'more', 'have', 'has', 'was', 'were', 'been', 'which', 'when',
    'https', 'http', 'com', 'www', 'youtube', 'watch', 'channel', 'video', 'v', 'org', 'net',
    # Tiếng Việt
    'và', 'là', 'của', 'cho', 'trong', 'với', 'các', 'những', 'đó', 'này', 'được', 'khi', 'về', 'có', 'không', 'như',
    'đã', 'đang', 'sẽ', 'để', 'một', 'mọi', 'ra', 'vào', 'hoặc', 'vì', 'theo', 'tại', 'từ', 'nên', 'cần', 'nhưng', 'bị'
}

# --- CÁC HÀM XỬ LÝ DỮ LIỆU CỐT LÕI ---

def lam_sach_text(text):
    """Làm sạch văn bản, loại bỏ URL và ký tự đặc biệt"""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    words = re.findall(r'\b[a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+\b', text.lower())
    return words

def trich_xuat_tu_khoa(danh_sach_tu, loai_ngram=1, top_n=50):
    """Trích xuất từ đơn (1-gram), cụm 2 từ (Bigram) hoặc cụm 3 từ (Trigram)"""
    tu_hop_le = [w for w in danh_sach_tu if len(w) > 1 and w not in STOP_WORDS and not w.isdigit()]
    
    if loai_ngram == 1:
        tokens = tu_hop_le
    elif loai_ngram == 2:
        tokens = [' '.join(tu_hop_le[i:i+2]) for i in range(len(tu_hop_le)-1)]
    else:
        tokens = [' '.join(tu_hop_le[i:i+3]) for i in range(len(tu_hop_le)-2)]
        
    counts = Counter(tokens)
    return counts.most_common(top_n)

def tao_file_excel(df):
    """Xuất DataFrame ra file Excel (.xlsx) dưới dạng Binary Buffer"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TuKhoa')
    buffer.seek(0)
    return buffer

def hien_thi_ket_qua(ket_qua_tu_khoa, ten_file_xuat="tu_khoa"):
    """Hiển thị giao diện kết quả và nút tải về tệp Office Excel/CSV"""
    if not ket_qua_tu_khoa:
        st.warning("⚠️ Không tìm thấy từ khóa phù hợp. Vui lòng thử dữ liệu đầu vào khác!")
        return
        
    st.success(f"🎉 Phân tích hoàn tất! Đã trích xuất được {len(ket_qua_tu_khoa)} nhóm từ khóa nổi bật.")
    
    df = pd.DataFrame(ket_qua_tu_khoa, columns=['Từ khóa / Cụm từ', 'Tần suất xuất hiện'])
    
    # Khu vực tải về File Office (CSV & XLSX)
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải file CSV (Chuẩn Excel tiếng Việt)",
            data=csv,
            file_name=f'{ten_file_xuat}.csv',
            mime='text/csv'
        )
        
    with col_dl2:
        excel_data = tao_file_excel(df)
        st.download_button(
            label="📊 Tải file Office Excel (.xlsx)",
            data=excel_data,
            file_name=f'{ten_file_xuat}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    st.markdown("---")
    
    # Hiển thị biểu đồ & bảng dữ liệu
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🔥 Bảng Tần Suất Từ Khóa")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.subheader("📊 Biểu Đồ Top Từ Khóa")
        df_chart = df.set_index('Từ khóa / Cụm từ')
        st.bar_chart(df_chart.head(15))

# --- GIAO DIỆN CHÍNH (TABS) ---

tab1, tab2, tab3 = st.tabs([
    "📺 YouTube & Kênh YouTube", 
    "🌐 Đường Link Web (URL)", 
    "📁 Tệp Office (.xlsx, .docx, .txt)"
])

# BỘ LỌC CHUNG CHO N-GRAM
st.sidebar.header("⚙️ Cấu Hình Phân Tích")
ngram_choice = st.sidebar.radio(
    "Độ dài từ khóa (N-gram):",
    ("Từ đơn (1 từ)", "Cụm 2 từ (Bigram)", "Cụm 3 từ (Trigram)")
)
ngram_map = {"Từ đơn (1 từ)": 1, "Cụm 2 từ (Bigram)": 2, "Cụm 3 từ (Trigram)": 3}
selected_ngram = ngram_map[ngram_choice]

# ================= TAB 1: YOUTUBE =================
with tab1:
    st.subheader("📺 Phân Tích Từ Khóa YouTube & Kênh YouTube")
    che_do_yt = st.radio("Chọn chế độ:", ("Gợi ý thanh tìm kiếm YouTube", "Trích xuất từ Link Kênh / Video Web"))
    
    if che_do_yt == "Gợi ý thanh tìm kiếm YouTube":
        tu_khoa_yt = st.text_input("🔑 Nhập từ khóa chủ đề (VD: financial management, money, credit card):", key="yt_kw")
        if st.button("🚀 Quét Gợi Ý YouTube"):
            if tu_khoa_yt:
                danh_sach_truy_van = [tu_khoa_yt] + [f"{tu_khoa_yt} {chr(i)}" for i in range(97, 123)]
                tat_ca_suggests = []
                bar = st.progress(0)
                
                for i, q in enumerate(danh_sach_truy_van):
                    url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={q}"
                    try:
                        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if res.status_code == 200:
                            tat_ca_suggests.extend(res.json()[1])
                    except:
                        pass
                    bar.progress((i + 1) / len(danh_sach_truy_van))
                    time.sleep(0.05)
                bar.empty()
                
                words = lam_sach_text(' '.join(tat_ca_suggests))
                res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                hien_thi_ket_qua(res_kw, f"yt_keywords_{tu_khoa_yt.replace(' ', '_')}")
            else:
                st.warning("Vui lòng nhập từ khóa!")
                
    else:
        link_yt = st.text_input("🔗 Dán đường link Kênh hoặc Video YouTube (VD: https://www.youtube.com/@channel):", key="yt_link")
        if st.button("🚀 Phân Tích Link YouTube"):
            if link_yt:
                with st.spinner("Đang kết nối và thu thập dữ liệu tiêu đề/mô tả..."):
                    try:
                        res = requests.get(link_yt, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                        soup = BeautifulSoup(res.text, 'html.parser')
                        title = soup.title.string if soup.title else ""
                        meta_desc = ""
                        for m in soup.find_all('meta'):
                            if m.get('name') in ['description', 'keywords']:
                                meta_desc += " " + str(m.get('content', ''))
                        
                        full_content = f"{title} {meta_desc} " + ' '.join([t.text for t in soup.find_all(['h1', 'h2', 'h3', 'span'])])
                        words = lam_sach_text(full_content)
                        res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                        hien_thi_ket_qua(res_kw, "yt_channel_keywords")
                    except Exception as e:
                        st.error(f"Không thể truy cập link: {e}")
            else:
                st.warning("Vui lòng nhập đường link!")

# ================= TAB 2: WEBPAGE URL =================
with tab2:
    st.subheader("🌐 Phân Tích Đường Link Website (URL)")
    web_url = st.text_input("🔗 Nhập địa chỉ đường link Web (VD: https://vnexpress.net/... hoặc trang tin tức/blog):")
    
    if st.button("🚀 Bóc Tách Từ Khóa Website"):
        if web_url:
            with st.spinner("Đang tải trang web và bóc tách nội dung bài viết..."):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    res = requests.get(web_url, headers=headers, timeout=10)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, 'html.parser')
                        # Xóa các thẻ script, style không cần thiết
                        for script in soup(["script", "style", "nav", "footer"]):
                            script.decompose()
                            
                        text_blocks = [tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'article'])]
                        full_text = ' '.join(text_blocks)
                        
                        words = lam_sach_text(full_text)
                        res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                        hien_thi_ket_qua(res_kw, "web_url_keywords")
                    else:
                        st.error(f"Không thể truy cập trang web (Mã lỗi: {res.status_code})")
                except Exception as e:
                    st.error(f"Lỗi truy cập URL: {e}")
        else:
            st.warning("Vui lòng nhập đường link URL!")

# ================= TAB 3: OFFICE FILES =================
with tab3:
    st.subheader("📁 Tải Lên Tệp Office Để Phân Tích (.xlsx, .csv, .docx, .txt)")
    uploaded_file = st.file_uploader("Chọn file từ máy tính của bạn:", type=['xlsx', 'csv', 'docx', 'txt'])
    
    if st.button("🚀 Phân Tích Tệp Office"):
        if uploaded_file is not None:
            with st.spinner("Đang đọc nội dung file Office..."):
                file_text = ""
                file_name = uploaded_file.name
                
                try:
                    if file_name.endswith('.csv'):
                        df_in = pd.read_csv(uploaded_file)
                        file_text = ' '.join(df_in.astype(str).values.flatten())
                    elif file_name.endswith('.xlsx'):
                        df_in = pd.read_excel(uploaded_file)
                        file_text = ' '.join(df_in.astype(str).values.flatten())
                    elif file_name.endswith('.docx'):
                        doc = docx.Document(uploaded_file)
                        file_text = ' '.join([p.text for p in doc.paragraphs])
                    elif file_name.endswith('.txt'):
                        file_text = uploaded_file.read().decode('utf-8', errors='ignore')
                        
                    words = lam_sach_text(file_text)
                    res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                    hien_thi_ket_qua(res_kw, f"office_keywords_{file_name.split('.')[0]}")
                except Exception as e:
                    st.error(f"Lỗi đọc file Office: {e}")
        else:
            st.warning("Vui lòng chọn tệp trước khi bấm phân tích!")
