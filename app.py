import streamlit as st
import requests
import pandas as pd
from collections import Counter
import re
import time
import io
from bs4 import BeautifulSoup
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# Set up giao diện Streamlit
st.set_page_config(page_title="Radar Phân Tích Từ Khóa Pro", page_icon="🎯", layout="wide")
st.title("🎯 Radar Phân Tích Từ Khóa Đa Nguồn (Bản PRO)")
st.markdown("Hỗ trợ trích xuất từ khóa, tạo Word Cloud, phân tích cảm xúc, so sánh đối thủ và quét thị trường quốc tế.")

# Khởi tạo Lịch sử tìm kiếm trong Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 1. XỬ LÝ STOPWORDS TÙY CHỈNH (Nâng cấp UX/UI) ---
STOP_WORDS = {
    'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 
    'be', 'are', 'from', 'at', 'as', 'your', 'how', 'what', 'why', 'do', 'can', 'my', 'we', 'about', 'an', 'if', 'will', 
    'up', 'out', 'just', 'so', 'me', 'they', 'like', 'get', 'more', 'have', 'has', 'was', 'were', 'been', 'which', 'when',
    'https', 'http', 'com', 'www', 'youtube', 'watch', 'channel', 'video', 'v', 'org', 'net',
    'và', 'là', 'của', 'cho', 'trong', 'với', 'các', 'những', 'đó', 'này', 'được', 'khi', 'về', 'có', 'không', 'như',
    'đã', 'đang', 'sẽ', 'để', 'một', 'mọi', 'ra', 'vào', 'hoặc', 'vì', 'theo', 'tại', 'từ', 'nên', 'cần', 'nhưng', 'bị'
}

st.sidebar.header("⚙️ Cấu Hình Phân Tích")
uploaded_stopwords = st.sidebar.file_uploader("Tải lên file Stopwords riêng (.txt)", type=['txt'])
if uploaded_stopwords:
    custom_words = uploaded_stopwords.read().decode('utf-8').splitlines()
    STOP_WORDS.update([w.strip().lower() for w in custom_words if w.strip()])
    st.sidebar.success(f"Đã thêm {len(custom_words)} từ khóa bỏ qua!")

# Cấu hình N-gram
ngram_choice = st.sidebar.radio("Độ dài từ khóa:", ("Từ đơn", "Cụm 2 từ", "Cụm 3 từ"))
ngram_map = {"Từ đơn": 1, "Cụm 2 từ": 2, "Cụm 3 từ": 3}
selected_ngram = ngram_map[ngram_choice]

# Hiển thị Lịch sử (Nâng cấp Năng suất)
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Lịch Sử Phân Tích")
for item in reversed(st.session_state.history[-5:]): # Hiển thị 5 cái gần nhất
    st.sidebar.caption(f"✓ {item}")

# --- CÁC HÀM XỬ LÝ CỐT LÕI ---
@st.cache_data(show_spinner=False)
def lam_sach_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    words = re.findall(r'\b[a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+\b', text.lower())
    return words

def trich_xuat_tu_khoa(danh_sach_tu, loai_ngram=1, top_n=50):
    tu_hop_le = [w for w in danh_sach_tu if len(w) > 1 and w not in STOP_WORDS and not w.isdigit()]
    if loai_ngram == 1:
        tokens = tu_hop_le
    elif loai_ngram == 2:
        tokens = [' '.join(tu_hop_le[i:i+2]) for i in range(len(tu_hop_le)-1)]
    else:
        tokens = [' '.join(tu_hop_le[i:i+3]) for i in range(len(tu_hop_le)-2)]
    return Counter(tokens).most_common(top_n)

def tao_file_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TuKhoa')
    buffer.seek(0)
    return buffer

def luu_lich_su(hanh_dong):
    if hanh_dong not in st.session_state.history:
        st.session_state.history.append(hanh_dong)

def phan_tich_cam_xuc(text):
    """Sử dụng TextBlob để đo lường cảm xúc (chủ yếu tốt cho tiếng Anh)"""
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.1: return "Tích cực 😊"
    elif score < -0.1: return "Tiêu cực 😠"
    return "Trung lập 😐"

def hien_thi_ket_qua(ket_qua_tu_khoa, nguyen_ban_text, ten_file_xuat="tu_khoa"):
    if not ket_qua_tu_khoa:
        st.warning("⚠️ Không tìm thấy từ khóa phù hợp.")
        return
        
    st.success(f"🎉 Phân tích hoàn tất! Trích xuất được {len(ket_qua_tu_khoa)} nhóm từ khóa.")
    
    # Nâng cấp Cảm xúc & Gợi ý (Năng suất & Intelligence)
    cam_xuc = phan_tich_cam_xuc(nguyen_ban_text)
    top_1 = ket_qua_tu_khoa[0][0]
    
    col_info1, col_info2 = st.columns(2)
    col_info1.info(f"**Cảm xúc chung văn bản:** {cam_xuc}")
    col_info2.info(f"**Gợi ý tiêu đề bài viết:** 'Bí quyết hiểu rõ {top_1.capitalize()} năm 2024'")

    df = pd.DataFrame(ket_qua_tu_khoa, columns=['Từ khóa / Cụm từ', 'Tần suất'])
    
    # Khu vực tải về
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Tải file CSV", data=df.to_csv(index=False).encode('utf-8-sig'), file_name=f'{ten_file_xuat}.csv', mime='text/csv')
    with col_dl2:
        st.download_button("📊 Tải file Excel", data=tao_file_excel(df), file_name=f'{ten_file_xuat}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    st.markdown("---")
    
    # Hiển thị WordCloud & Bảng
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("🔥 Bảng Tần Suất")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.subheader("☁️ Đám Mây Từ Khóa (Word Cloud)")
        words_dict = dict(ket_qua_tu_khoa)
        wc = WordCloud(width=600, height=400, background_color='white', colormap='viridis').generate_from_frequencies(words_dict)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# --- GIAO DIỆN CHÍNH (TABS) ---
tab1, tab2, tab3 = st.tabs(["🔍 Google & YouTube Suggest", "🌐 Phân Tích & So Sánh URL", "📁 Tệp Office"])

# ================= TAB 1: GOOGLE & YOUTUBE QUỐC TẾ =================
with tab1:
    st.subheader("🔍 Lấy gợi ý tìm kiếm đa quốc gia")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        nguon_gợi_ý = st.selectbox("Chọn nền tảng:", ("YouTube", "Google Search"))
    with col_opt2:
        quoc_gia = st.selectbox(
            "🌍 Chọn thị trường (Quốc gia):",
            ("Việt Nam (VN)", "Mỹ (US)", "Anh - UK (Châu Âu)", "Đức - DE (Châu Âu)", "Pháp - FR (Châu Âu)", "Toàn cầu")
        )
    
    tu_khoa_nhap = st.text_input("🔑 Nhập từ khóa gốc (VD: make money online, credit card):", key="kw_search")
    
    market_map = {
        "Việt Nam (VN)": {"gl": "vn", "hl": "vi"},
        "Mỹ (US)": {"gl": "us", "hl": "en"},
        "Anh - UK (Châu Âu)": {"gl": "gb", "hl": "en"},
        "Đức - DE (Châu Âu)": {"gl": "de", "hl": "de"},
        "Pháp - FR (Châu Âu)": {"gl": "fr", "hl": "fr"},
        "Toàn cầu": {"gl": "", "hl": "en"}
    }
    
    if st.button("🚀 Quét Từ Khóa Theo Thị Trường"):
        if tu_khoa_nhap:
            gl = market_map[quoc_gia]["gl"]
            hl = market_map[quoc_gia]["hl"]
            
            danh_sach_truy_van = [tu_khoa_nhap] + [f"{tu_khoa_nhap} {chr(i)}" for i in range(97, 123)]
            tat_ca_suggests = []
            bar = st.progress(0)
            
            client_type = "yt" if nguon_gợi_ý == "YouTube" else "chrome"
            
            for i, q in enumerate(danh_sach_truy_van):
                if nguon_gợi_ý == "YouTube":
                    url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds={client_type}&q={q}&hl={hl}&gl={gl}"
                else:
                    url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={q}&hl={hl}&gl={gl}"
                
                try:
                    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                    if res.status_code == 200:
                        tat_ca_suggests.extend(res.json()[1])
                except: 
                    pass
                
                bar.progress((i + 1) / len(danh_sach_truy_van))
                time.sleep(0.05)
            
            bar.empty()
            full_text = ' '.join(tat_ca_suggests)
            res_kw = trich_xuat_tu_khoa(lam_sach_text(full_text), selected_ngram)
            
            luu_lich_su(f"Quét {nguon_gợi_ý} ({quoc_gia}): {tu_khoa_nhap}")
            ten_file = f"{nguon_gợi_ý}_{market_map[quoc_gia]['gl']}_{tu_khoa_nhap.replace(' ', '_')}"
            hien_thi_ket_qua(res_kw, full_text, ten_file)
        else:
            st.warning("⚠️ Vui lòng nhập từ khóa gốc trước khi quét!")

# ================= TAB 2: SO SÁNH WEBSITE (Intelligence) =================
with tab2:
    st.subheader("🌐 Phân Tích & So Sánh Website")
    che_do_web = st.radio("Chế độ:", ("Phân tích 1 Website", "So sánh 2 Đối thủ (A vs B)"))
    
    url1 = st.text_input("🔗 Nhập URL trang 1:")
    url2 = st.text_input("🔗 Nhập URL trang 2 (đối thủ):") if che_do_web == "So sánh 2 Đối thủ (A vs B)" else None
    
    def lay_chu_web(url):
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for script in soup(["script", "style"]): script.decompose()
        return ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'article'])])

    if st.button("🚀 Tiến Hành Phân Tích URL"):
        if che_do_web == "Phân tích 1 Website" and url1:
            text1 = lay_chu_web(url1)
            res_kw = trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram)
            luu_lich_su(f"Phân tích web: {url1}")
            hien_thi_ket_qua(res_kw, text1, "Web_Keywords")
            
        elif che_do_web == "So sánh 2 Đối thủ (A vs B)" and url1 and url2:
            text1, text2 = lay_chu_web(url1), lay_chu_web(url2)
            kw1 = dict(trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram))
            kw2 = dict(trich_xuat_tu_khoa(lam_sach_text(text2), selected_ngram))
            
            # Kết hợp dữ liệu 2 trang web
            all_keys = set(kw1.keys()).union(set(kw2.keys()))
            compare_data = [{"Từ khóa": k, "Tần suất Web 1": kw1.get(k, 0), "Tần suất Web 2": kw2.get(k, 0)} for k in all_keys]
            df_compare = pd.DataFrame(compare_data).sort_values(by="Tần suất Web 1", ascending=False).head(30)
            
            st.success("✅ Đã so sánh xong 2 trang web!")
            st.dataframe(df_compare, use_container_width=True)
            luu_lich_su(f"So sánh: {url1} vs {url2}")

# ================= TAB 3: OFFICE FILES =================
with tab3:
    st.subheader("📁 Tải Lên Tệp Office")
    uploaded_file = st.file_uploader("Chọn file (xlsx, csv, docx, txt):", type=['xlsx', 'csv', 'docx', 'txt'])
    
    if st.button("🚀 Phân Tích Tệp Office") and uploaded_file:
        file_text, file_name = "", uploaded_file.name
        if file_name.endswith('.csv'): file_text = ' '.join(pd.read_csv(uploaded_file).astype(str).values.flatten())
        elif file_name.endswith('.xlsx'): file_text = ' '.join(pd.read_excel(uploaded_file).astype(str).values.flatten())
        elif file_name.endswith('.docx'): file_text = ' '.join([p.text for p in docx.Document(uploaded_file).paragraphs])
        elif file_name.endswith('.txt'): file_text = uploaded_file.read().decode('utf-8', errors='ignore')
            
        res_kw = trich_xuat_tu_khoa(lam_sach_text(file_text), selected_ngram)
        luu_lich_su(f"Phân tích file: {file_name}")
        hien_thi_ket_qua(res_kw, file_text, f"office_{file_name}")
