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
st.set_page_config(page_title="Radar SEO & Content Pro", page_icon="🚀", layout="wide")
st.title("🚀 Radar SEO & Content Pro (Bản All-in-One)")
st.markdown("Hệ sinh thái phân tích từ khóa, đối thủ và tự động hóa sản xuất nội dung bằng Trí Tuệ Nhân Tạo.")

# ================= KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE) =================
if 'history' not in st.session_state: st.session_state.history = []
if 'current_kw' not in st.session_state: st.session_state.current_kw = None
if 'current_text' not in st.session_state: st.session_state.current_text = ""
if 'current_file' not in st.session_state: st.session_state.current_file = ""
if 'ai_result' not in st.session_state: st.session_state.ai_result = ""

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

# --- CÁC HÀM XỬ LÝ CỐT LÕI ---
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

def phan_loai_y_dinh(kw):
    kw_lower = kw.lower()
    if any(w in kw_lower for w in ['mua', 'giá', 'bán', 'rẻ', 'bao nhiêu', 'khuyến mãi', 'shop', 'lazada', 'shopee']): return "🛒 Giao dịch"
    if any(w in kw_lower for w in ['cách', 'hướng dẫn', 'là gì', 'tại sao', 'khi nào', 'review', 'đánh giá', 'không']): return "📖 Thông tin"
    if any(w in kw_lower for w in ['facebook', 'youtube', 'tiki', 'login', 'web']): return "🧭 Điều hướng"
    return "💡 Khám phá"

def gom_nhom(kw):
    words = kw.split()
    if len(words) > 1: return words[0].capitalize()
    return "Từ Đơn"

def phan_tich_cam_xuc_vn(text):
    tich_cuc = ['tuyệt', 'hay', 'tốt', 'giỏi', 'đẹp', 'xuất sắc', 'thích', 'ok', 'ngon', 'đỉnh', 'cảm ơn']
    tieu_cuc = ['tệ', 'chán', 'dở', 'xấu', 'lỗi', 'kém', 'lừa đảo', 'scam', 'thất vọng']
    text_lower = text.lower()
    score = sum(text_lower.count(w) for w in tich_cuc) - sum(text_lower.count(w) for w in tieu_cuc)
    if score > 0: return "Tích cực 😊"
    elif score < 0: return "Tiêu cực 😠"
    blob_score = TextBlob(text).sentiment.polarity
    if blob_score > 0.1: return "Tích cực 😊"
    elif blob_score < -0.1: return "Tiêu cực 😠"
    return "Trung lập 😐"

def tao_file_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Tat_Ca_Tu_Khoa')
        df.head(10).to_excel(writer, index=False, sheet_name='Top_10_Tu_Khoa')
    buffer.seek(0)
    return buffer

def luu_lich_su(hanh_dong):
    if hanh_dong not in st.session_state.history:
        st.session_state.history.append(hanh_dong)

def luu_ket_qua_vao_bo_nho(res_kw, full_text, ten_file):
    st.session_state.current_kw = res_kw
    st.session_state.current_text = full_text
    st.session_state.current_file = ten_file
    st.session_state.ai_result = ""

# --- HÀM TẠO NỘI DUNG AI ĐA NĂNG (ĐÃ TĂNG THỜI GIAN CHỜ LÊN 120 GIÂY) ---
def xu_ly_ai_da_nang(che_do, tu_khoa_list, text_goc):
    if not gemini_api_key:
        st.error("⚠️ Bạn cần dán Gemini API Key ở thanh bên (Sidebar) để dùng tính năng này!")
        return
        
    url_list = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {'x-goog-api-key': gemini_api_key, 'Content-Type': 'application/json'}
    
    try:
        res_list = requests.get(url_list, headers=headers, timeout=10)
        if res_list.status_code != 200:
            st.error(f"❌ Khóa API không hợp lệ: {res_list.text}")
            return
            
        models_data = res_list.json().get('models', [])
        valid_models = [m['name'] for m in models_data if 'generateContent' in m.get('supportedGenerationMethods', []) and 'gemini' in m['name']]
        
        if not valid_models:
            st.error("❌ Không tìm thấy mô hình khả dụng.")
            return

        # Soạn Prompt dựa theo lựa chọn của người dùng
        if che_do == "📝 Lập Dàn Ý SEO (Outline)":
            prompt = f"Bạn là chuyên gia SEO. Dựa vào Top từ khóa: {tu_khoa_list}. Hãy đề xuất 3 Tiêu đề hấp dẫn và lập Dàn ý (H2, H3) chi tiết chuẩn SEO."
        elif che_do == "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)":
            prompt = f"Bạn là một Copywriter chuyên nghiệp. Dựa vào bộ từ khóa: {tu_khoa_list}. Hãy viết một bài Blog chuẩn SEO hoàn chỉnh, dài khoảng 1000 từ. Bố cục rõ ràng, có mở bài, thân bài (chia các Heading H2, H3) và kết bài. Lồng ghép từ khóa một cách tự nhiên nhất."
        elif che_do == "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)":
            prompt = f"Bạn là một chuyên gia sáng tạo nội dung Video ngắn. Từ các từ khóa xu hướng này: {tu_khoa_list}. Hãy viết một kịch bản Video dọc (dưới 60 giây). Bao gồm: 1. Hook (Câu móc nối 3 giây đầu gây sốc/tò mò), 2. Body (Nội dung chính diễn giải chi tiết), 3. CTA (Kêu gọi hành động). Cung cấp cả gợi ý hình ảnh/chữ chạy trên màn hình (Text on screen)."
        elif che_do == "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)":
            prompt = f"Bạn là chuyên gia chốt sale trên các sàn thương mại điện tử. Dựa vào danh sách từ khóa nhu cầu này: {tu_khoa_list}. Hãy viết: 1. 3 phương án Tiêu đề Sản phẩm giật tít, chuẩn SEO. 2. Một đoạn Mô tả sản phẩm (Product Description) đánh mạnh vào nỗi đau khách hàng, nêu bật lợi ích và lồng ghép từ khóa để tối ưu thứ hạng tìm kiếm."
        elif che_do == "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)":
            if "youtube" not in st.session_state.current_file.lower() and "yt" not in st.session_state.current_file.lower():
                st.warning("Tính năng này hoạt động tốt nhất khi bạn nhập Link YouTube của đối thủ ở Tab 1. AI sẽ phân tích text hiện tại, nhưng có thể không chính xác nếu đây không phải dữ liệu từ YouTube.")
            prompt = f"Dưới đây là dữ liệu (Tiêu đề, mô tả, và possibly comments) trích xuất từ Video YouTube của đối thủ: \n\n{text_goc[:5000]}\n\nBạn là một YouTuber chuyên nghiệp. Hãy phân tích nội dung trên và viết ra một Kịch Bản Video YouTube hoàn chỉnh, MỚI MẺ và HẤP DẪN HƠN cho kênh của tôi. Cấu trúc kịch bản bao gồm: Tiêu đề thu hút, Hook, Intro, Body (Các luận điểm chính được diễn giải chi tiết hơn đối thủ), và Outro/CTA."

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        success = False
        with st.spinner(f"🤖 Trợ lý AI đang xử lý yêu cầu: {che_do} (Quá trình này có thể mất 1-2 phút, vui lòng chờ)..."):
            for model_name in valid_models:
                url_gen = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
                # ĐÃ TĂNG TIMEOUT LÊN 120 GIÂY Ở ĐÂY
                resp_gen = requests.post(url_gen, headers=headers, json=payload, timeout=120) 
                
                if resp_gen.status_code == 200:
                    data = resp_gen.json()
                    st.session_state.ai_result = data['candidates'][0]['content']['parts'][0]['text']
                    success = True
                    break
                    
        if success:
            st.rerun()
        else:
            st.error("❌ Máy chủ Google đang quá tải hoặc từ chối xử lý, vui lòng thử lại.")
            
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")

# ================= GIAO DIỆN NHẬP LIỆU (TABS) =================
tab1, tab2, tab3 = st.tabs(["📺 YouTube & Gợi Ý", "🌐 Phân Tích & So Sánh URL", "📁 Tệp Office"])

with tab1:
    che_do_yt = st.radio("Chọn chế độ phân tích:", ("🌍 Gợi ý tìm kiếm Đa quốc gia", "🔗 Bóc tách từ Link Kênh/Video (Hỗ trợ cào Comment)"))
    st.markdown("---")
    
    if che_do_yt == "🌍 Gợi ý tìm kiếm Đa quốc gia":
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1: nguon_gợi_ý = st.selectbox("Chọn nền tảng:", ("YouTube", "Google Search"))
        with col_opt2: quoc_gia = st.selectbox("🌍 Chọn thị trường:", ("Việt Nam (VN)", "Mỹ (US)", "Anh (UK)", "Đức (DE)", "Pháp (FR)", "Toàn cầu"))
        
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
                luu_ket_qua_vao_bo_nho(res_kw, full_text, f"{nguon_gợi_ý}_{market_map[quoc_gia]['gl']}")
            else:
                st.warning("Vui lòng nhập từ khóa!")
                
    else:
        link_yt = st.text_input("🔗 Nhập link Video/Kênh YouTube:")
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
                        
                        if quet_comment and yt_api_key:
                            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', link_yt)
                            if video_id_match:
                                vid_id = video_id_match.group(1)
                                api_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={vid_id}&maxResults=100&key={yt_api_key}"
                                cmt_res = requests.get(api_url).json()
                                if 'items' in cmt_res:
                                    for item in cmt_res['items']: full_content += " " + item['snippet']['topLevelComment']['snippet']['textOriginal']
                        
                        words = lam_sach_text(full_content)
                        res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                        luu_lich_su(f"Phân tích Link YT")
                        luu_ket_qua_vao_bo_nho(res_kw, full_content, "yt_link_keywords")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
            else:
                st.warning("Vui lòng nhập đường link!")

with tab2:
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
            luu_ket_qua_vao_bo_nho(trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram), text1, "Web_Keywords")

with tab3:
    uploaded_file = st.file_uploader("Chọn file (xlsx, csv, docx, txt):", type=['xlsx', 'csv', 'docx', 'txt'])
    if st.button("🚀 Phân Tích Tệp Office") and uploaded_file:
        file_text, file_name = "", uploaded_file.name
        if file_name.endswith('.csv'): file_text = ' '.join(pd.read_csv(uploaded_file).astype(str).values.flatten())
        elif file_name.endswith('.xlsx'): file_text = ' '.join(pd.read_excel(uploaded_file).astype(str).values.flatten())
        elif file_name.endswith('.docx'): file_text = ' '.join([p.text for p in docx.Document(uploaded_file).paragraphs])
        elif file_name.endswith('.txt'): file_text = uploaded_file.read().decode('utf-8', errors='ignore')
        luu_ket_qua_vao_bo_nho(trich_xuat_tu_khoa(lam_sach_text(file_text), selected_ngram), file_text, f"office_{file_name}")

# ================= GIAO DIỆN HIỂN THỊ KẾT QUẢ =================
if st.session_state.current_kw:
    st.markdown("---")
    st.success(f"🎉 Trích xuất thành công {len(st.session_state.current_kw)} nhóm từ khóa.")
    
    col_info1, col_info2 = st.columns(2)
    col_info1.info(f"**Cảm xúc văn bản:** {phan_tich_cam_xuc_vn(st.session_state.current_text)}")
    col_info2.info(f"**Top 1 Keyword:** {st.session_state.current_kw[0][0].capitalize()}")

    df = pd.DataFrame(st.session_state.current_kw, columns=['Từ khóa / Cụm từ', 'Tần suất'])
    df['Ý định tìm kiếm (Intent)'] = df['Từ khóa / Cụm từ'].apply(phan_loai_y_dinh)
    df['Nhóm (Cluster)'] = df['Từ khóa / Cụm từ'].apply(gom_nhom)
    df['Phân loại Top'] = ['🏆 Top 10 Thịnh hành' if i < 10 else 'Thông thường' for i in range(len(df))]
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1: st.download_button("📥 Tải file CSV", data=df.to_csv(index=False).encode('utf-8-sig'), file_name=f'{st.session_state.current_file}.csv', mime='text/csv')
    with col_dl2: st.download_button("📊 Tải file Excel", data=tao_file_excel(df), file_name=f'{st.session_state.current_file}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    st.markdown("---")
    
    # 🤖 KHU VỰC TRỢ LÝ AI MỚI (MENU ĐA NĂNG)
    st.subheader("🤖 Trợ Lý Trí Tuệ Nhân Tạo (Đa Năng)")
    che_do_ai = st.selectbox("Chọn hành động bạn muốn AI thực hiện:", [
        "📝 Lập Dàn Ý SEO (Outline)",
        "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)",
        "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)",
        "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)",
        "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)"
    ])
    
    top_10_words = [item[0] for item in st.session_state.current_kw[:10]]
    
    if st.button(f"✨ Kích Hoạt Trợ Lý AI: {che_do_ai.split(' ')[1]}"):
        xu_ly_ai_da_nang(che_do_ai, top_10_words, st.session_state.current_text)
        
    if st.session_state.ai_result:
        st.success("✅ Trợ lý AI đã hoàn thành xuất sắc nhiệm vụ!")
        st.markdown(st.session_state.ai_result)
        
    st.markdown("---")
    
    # 📊 KHU VỰC BÁO CÁO PHÂN TÍCH
    st.subheader("📊 Báo Cáo Phân Tích Dữ Liệu")
    
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        st.markdown("**Bảng Dữ Liệu Từ Khóa**")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.markdown("**Biểu Đồ Ý Định Tìm Kiếm (Intent)**")
        intent_counts = df['Ý định tìm kiếm (Intent)'].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(intent_counts, labels=intent_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        ax_pie.axis('equal') 
        st.pyplot(fig_pie)
    with c3:
        st.markdown("**Đám Mây Từ Khóa (Word Cloud)**")
        words_dict = dict(st.session_state.current_kw)
        wc = WordCloud(width=400, height=400, background_color='white', colormap='viridis').generate_from_frequencies(words_dict)
        fig_wc, ax_wc = plt.subplots(figsize=(4, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
