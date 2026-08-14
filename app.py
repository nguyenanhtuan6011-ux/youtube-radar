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
import google.generativeai as genai

# Set up giao diện Streamlit
st.set_page_config(page_title="Radar SEO & Content Pro", page_icon="🚀", layout="wide")
st.title("🚀 Radar SEO & Trợ Lý Viết Bài AI (Bản Ultimate)")
st.markdown("Hỗ trợ trích xuất từ khóa đa nguồn, phân loại ý định tìm kiếm, phân tích bình luận và tự động viết dàn ý bài chuẩn SEO bằng AI.")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- CẤU HÌNH API KEY (SIDEBAR) ---
st.sidebar.header("🔑 Cấu Hình API (Tùy chọn)")
gemini_api_key = st.sidebar.text_input("Gemini API Key (Dùng để AI viết bài):", type="password")
yt_api_key = st.sidebar.text_input("YouTube Data API v3 Key (Để quét Comment):", type="password")
st.sidebar.caption("Lấy Gemini API miễn phí tại: aistudio.google.com")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Cấu Hình Phân Tích")

# --- XỬ LÝ STOPWORDS ---
STOP_WORDS = {
    'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 
    'be', 'are', 'from', 'at', 'as', 'your', 'how', 'what', 'why', 'do', 'can', 'my', 'we', 'about', 'an', 'if', 'will', 
    'https', 'http', 'com', 'www', 'youtube', 'watch', 'channel', 'video', 'v', 'org', 'net',
    'và', 'là', 'của', 'cho', 'trong', 'với', 'các', 'những', 'đó', 'này', 'được', 'khi', 'về', 'có', 'không', 'như',
    'đã', 'đang', 'sẽ', 'để', 'một', 'mọi', 'ra', 'vào', 'hoặc', 'vì', 'theo', 'tại', 'từ', 'nên', 'cần', 'nhưng', 'bị'
}

uploaded_stopwords = st.sidebar.file_uploader("Tải lên file Stopwords (.txt)", type=['txt'])
if uploaded_stopwords:
    custom_words = uploaded_stopwords.read().decode('utf-8').splitlines()
    STOP_WORDS.update([w.strip().lower() for w in custom_words if w.strip()])

ngram_choice = st.sidebar.radio("Độ dài từ khóa:", ("Từ đơn", "Cụm 2 từ", "Cụm 3 từ"))
ngram_map = {"Từ đơn": 1, "Cụm 2 từ": 2, "Cụm 3 từ": 3}
selected_ngram = ngram_map[ngram_choice]

# --- CÁC HÀM XỬ LÝ CỐT LÕI (CÓ NÂNG CẤP) ---

@st.cache_data(show_spinner=False)
def lam_sach_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    words = re.findall(r'\b[a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+\b', text.lower())
    return words

def trich_xuat_tu_khoa(danh_sach_tu, loai_ngram=1, top_n=50):
    tu_hop_le = [w for w in danh_sach_tu if len(w) > 1 and w not in STOP_WORDS and not w.isdigit()]
    if loai_ngram == 1: tokens = tu_hop_le
    elif loai_ngram == 2: tokens = [' '.join(tu_hop_le[i:i+2]) for i in range(len(tu_hop_le)-1)]
    else: tokens = [' '.join(tu_hop_le[i:i+3]) for i in range(len(tu_hop_le)-2)]
    return Counter(tokens).most_common(top_n)

# TÍNH NĂNG 3: PHÂN LOẠI Ý ĐỊNH TÌM KIẾM (Search Intent)
def phan_loai_y_dinh(kw):
    kw_lower = kw.lower()
    if any(w in kw_lower for w in ['mua', 'giá', 'bán', 'rẻ', 'bao nhiêu', 'khuyến mãi', 'shop']): return "🛒 Giao dịch"
    if any(w in kw_lower for w in ['cách', 'hướng dẫn', 'là gì', 'tại sao', 'khi nào', 'review', 'đánh giá', 'không']): return "📖 Thông tin"
    if any(w in kw_lower for w in ['facebook', 'youtube', 'shopee', 'tiki', 'lazada', 'login']): return "🧭 Điều hướng"
    return "💡 Khám phá"

# TÍNH NĂNG 2: GOM NHÓM TỪ KHÓA (Clustering)
def gom_nhom(kw):
    words = kw.split()
    if len(words) > 1: return words[0].capitalize()
    return "Từ Đơn"

# TÍNH NĂNG 4.1: CẢI THIỆN CẢM XÚC TIẾNG VIỆT
def phan_tich_cam_xuc_vn(text):
    tich_cuc = ['tuyệt', 'hay', 'tốt', 'giỏi', 'đẹp', 'xuất sắc', 'thích', 'ok', 'ngon', 'đỉnh', 'cảm ơn']
    tieu_cuc = ['tệ', 'chán', 'dở', 'xấu', 'lỗi', 'kém', 'lừa đảo', 'scam', 'thất vọng']
    text_lower = text.lower()
    score = sum(text_lower.count(w) for w in tich_cuc) - sum(text_lower.count(w) for w in tieu_cuc)
    
    if score > 0: return "Tích cực 😊"
    elif score < 0: return "Tiêu cực 😠"
    
    # Fallback cho Tiếng Anh
    blob_score = TextBlob(text).sentiment.polarity
    if blob_score > 0.1: return "Tích cực 😊"
    elif blob_score < -0.1: return "Tiêu cực 😠"
    return "Trung lập 😐"

def tao_file_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='TuKhoa')
    buffer.seek(0)
    return buffer

def luu_lich_su(hanh_dong):
    if hanh_dong not in st.session_state.history:
        st.session_state.history.append(hanh_dong)

# TÍNH NĂNG 1: TÍCH HỢP AI TẠO NỘI DUNG (Generative AI)
def tao_dan_y_ai(tu_khoa_list):
    if not gemini_api_key:
        st.error("⚠️ Bạn cần nhập Gemini API Key ở thanh bên (Sidebar) để dùng tính năng này!")
        return
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Bạn là một chuyên gia SEO hàng đầu. Dựa vào Top các từ khóa sau đây mà tôi thu thập được: {tu_khoa_list}. 
        Hãy giúp tôi: 
        1. Đề xuất 3 Tiêu đề bài viết siêu hấp dẫn (Giật tít, thu hút click).
        2. Lập một Dàn ý (Outline) chi tiết chuẩn SEO (H2, H3) để viết bài bao trùm các từ khóa trên."""
        
        with st.spinner("🤖 Trợ lý AI đang vắt óc suy nghĩ để viết dàn ý..."):
            response = model.generate_content(prompt)
            st.success("✅ AI đã hoàn thành!")
            st.markdown(response.text)
    except Exception as e:
        st.error(f"Lỗi khi gọi AI: {e}")

def hien_thi_ket_qua(ket_qua_tu_khoa, nguyen_ban_text, ten_file_xuat="tu_khoa"):
    if not ket_qua_tu_khoa:
        st.warning("⚠️ Không tìm thấy từ khóa phù hợp.")
        return
        
    st.success(f"🎉 Trích xuất thành công {len(ket_qua_tu_khoa)} nhóm từ khóa.")
    
    col_info1, col_info2 = st.columns(2)
    col_info1.info(f"**Cảm xúc văn bản:** {phan_tich_cam_xuc_vn(nguyen_ban_text)}")
    col_info2.info(f"**Top 1 Keyword:** {ket_qua_tu_khoa[0][0].capitalize()}")

    # Nâng cấp DataFrame với Ý định và Gom nhóm
    df = pd.DataFrame(ket_qua_tu_khoa, columns=['Từ khóa / Cụm từ', 'Tần suất'])
    df['Ý định tìm kiếm (Intent)'] = df['Từ khóa / Cụm từ'].apply(phan_loai_y_dinh)
    df['Nhóm (Cluster)'] = df['Từ khóa / Cụm từ'].apply(gom_nhom)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Tải file CSV", data=df.to_csv(index=False).encode('utf-8-sig'), file_name=f'{ten_file_xuat}.csv', mime='text/csv')
    with col_dl2:
        st.download_button("📊 Tải file Excel", data=tao_file_excel(df), file_name=f'{ten_file_xuat}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    st.markdown("---")
    
    # Hiển thị Nút gọi AI
    top_10_words = [item[0] for item in ket_qua_tu_khoa[:10]]
    if st.button("✨ Dùng AI viết Dàn Ý Bài Viết dựa trên Top Từ Khóa"):
        tao_dan_y_ai(top_10_words)
        
    st.markdown("---")
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("🔥 Bảng Tần Suất & Intent")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.subheader("☁️ Word Cloud")
        words_dict = dict(ket_qua_tu_khoa)
        wc = WordCloud(width=600, height=400, background_color='white', colormap='viridis').generate_from_frequencies(words_dict)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# --- GIAO DIỆN CHÍNH (TABS) ---
tab1, tab2, tab3 = st.tabs(["📺 YouTube & Gợi Ý", "🌐 Phân Tích & So Sánh URL", "📁 Tệp Office"])

# ================= TAB 1: YOUTUBE & GỢI Ý =================
with tab1:
    st.subheader("🔍 Phân Tích YouTube & Gợi Ý Tìm Kiếm")
    
    che_do_yt = st.radio(
        "Chọn chế độ phân tích:", 
        ("🌍 Gợi ý tìm kiếm Đa quốc gia", "🔗 Bóc tách từ Link Kênh/Video (Hỗ trợ cào Comment)")
    )
    st.markdown("---")
    
    if che_do_yt == "🌍 Gợi ý tìm kiếm Đa quốc gia":
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            nguon_gợi_ý = st.selectbox("Chọn nền tảng:", ("YouTube", "Google Search"))
        with col_opt2:
            quoc_gia = st.selectbox("🌍 Chọn thị trường:", ("Việt Nam (VN)", "Mỹ (US)", "Anh (UK)", "Đức (DE)", "Pháp (FR)", "Toàn cầu"))
        
        tu_khoa_nhap = st.text_input("🔑 Nhập từ khóa gốc:", key="kw_search")
        market_map = {"Việt Nam (VN)": {"gl": "vn", "hl": "vi"}, "Mỹ (US)": {"gl": "us", "hl": "en"}, "Anh (UK)": {"gl": "gb", "hl": "en"}, "Đức (DE)": {"gl": "de", "hl": "de"}, "Pháp (FR)": {"gl": "fr", "hl": "fr"}, "Toàn cầu": {"gl": "", "hl": "en"}}
        
        if st.button("🚀 Quét Từ Khóa"):
            if tu_khoa_nhap:
                gl, hl = market_map[quoc_gia]["gl"], market_map[quoc_gia]["hl"]
                danh_sach_truy_van = [tu_khoa_nhap] + [f"{tu_khoa_nhap} {chr(i)}" for i in range(97, 123)]
                tat_ca_suggests = []
                bar = st.progress(0)
                client_type = "yt" if nguon_gợi_ý == "YouTube" else "chrome"
                
                for i, q in enumerate(danh_sach_truy_van):
                    url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds={client_type}&q={q}&hl={hl}&gl={gl}" if nguon_gợi_ý == "YouTube" else f"https://suggestqueries.google.com/complete/search?client=firefox&q={q}&hl={hl}&gl={gl}"
                    try:
                        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if res.status_code == 200: tat_ca_suggests.extend(res.json()[1])
                    except: pass
                    bar.progress((i + 1) / len(danh_sach_truy_van))
                    time.sleep(0.05)
                
                bar.empty()
                full_text = ' '.join(tat_ca_suggests)
                res_kw = trich_xuat_tu_khoa(lam_sach_text(full_text), selected_ngram)
                luu_lich_su(f"Quét {nguon_gợi_ý} ({quoc_gia})")
                hien_thi_ket_qua(res_kw, full_text, f"{nguon_gợi_ý}_{market_map[quoc_gia]['gl']}")
            else:
                st.warning("Vui lòng nhập từ khóa!")
                
    else:
        link_yt = st.text_input("🔗 Nhập link Video/Kênh YouTube:")
        
        # TÍNH NĂNG 4.2: CÀO COMMENT YOUTUBE
        quet_comment = st.checkbox("💬 Quét luôn cả bình luận của người xem (Chỉ dùng cho Link Video)")
        
        if st.button("🚀 Phân Tích YouTube"):
            if link_yt:
                with st.spinner("Đang thu thập dữ liệu..."):
                    try:
                        res = requests.get(link_yt, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                        soup = BeautifulSoup(res.text, 'html.parser')
                        title = soup.title.string if soup.title else ""
                        meta_desc = " ".join([str(m.get('content', '')) for m in soup.find_all('meta') if m.get('name') in ['description', 'keywords']])
                        
                        full_content = f"{title} {meta_desc} " + ' '.join([t.text for t in soup.find_all(['h1', 'h2', 'h3', 'span'])])
                        
                        # Xử lý Cào Comment qua API
                        if quet_comment:
                            if not yt_api_key:
                                st.warning("Bạn chưa nhập YouTube API Key ở thanh bên. Hệ thống chỉ lấy tiêu đề & mô tả.")
                            else:
                                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', link_yt)
                                if video_id_match:
                                    vid_id = video_id_match.group(1)
                                    api_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={vid_id}&maxResults=100&key={yt_api_key}"
                                    cmt_res = requests.get(api_url).json()
                                    if 'items' in cmt_res:
                                        for item in cmt_res['items']:
                                            full_content += " " + item['snippet']['topLevelComment']['snippet']['textOriginal']
                                        st.success("✅ Đã lấy thành công Insight từ bình luận!")
                                    else:
                                        st.error("API Key sai hoặc Video tắt bình luận.")
                                else:
                                    st.error("Không trích xuất được Video ID. Bạn chắc chắn đây là link Video chứ?")

                        words = lam_sach_text(full_content)
                        res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                        luu_lich_su(f"Phân tích Link YT")
                        hien_thi_ket_qua(res_kw, full_content, "yt_link_keywords")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
            else:
                st.warning("Vui lòng nhập đường link!")

# ================= TAB 2: SO SÁNH WEBSITE =================
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

    if st.button("🚀 Phân Tích URL"):
        if che_do_web == "Phân tích 1 Website" and url1:
            text1 = lay_chu_web(url1)
            hien_thi_ket_qua(trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram), text1, "Web_Keywords")
            
        elif che_do_web == "So sánh 2 Đối thủ (A vs B)" and url1 and url2:
            text1, text2 = lay_chu_web(url1), lay_chu_web(url2)
            kw1 = dict(trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram))
            kw2 = dict(trich_xuat_tu_khoa(lam_sach_text(text2), selected_ngram))
            
            all_keys = set(kw1.keys()).union(set(kw2.keys()))
            compare_data = [{"Từ khóa": k, "Tần suất Web 1": kw1.get(k, 0), "Tần suất Web 2": kw2.get(k, 0)} for k in all_keys]
            df_compare = pd.DataFrame(compare_data).sort_values(by="Tần suất Web 1", ascending=False).head(30)
            
            st.success("✅ Đã so sánh xong!")
            st.dataframe(df_compare, use_container_width=True)

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
            
        hien_thi_ket_qua(trich_xuat_tu_khoa(lam_sach_text(file_text), selected_ngram), file_text, f"office_{file_name}")
