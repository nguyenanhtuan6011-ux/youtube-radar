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
import base64

# Set up giao diện Streamlit
st.set_page_config(page_title="Radar SEO & Content Pro", page_icon="🚀", layout="wide")
st.title("🚀 Radar SEO & Content Pro (Bản All-in-One + X-Ray)")
st.markdown("Hệ sinh thái phân tích từ khóa, tình báo đối thủ và tự động hóa sản xuất nội dung bằng Trí Tuệ Nhân Tạo.")

# ================= KHỞI TẠO BỘ NHỚ TẠM =================
if 'history' not in st.session_state: st.session_state.history = []
if 'current_kw' not in st.session_state: st.session_state.current_kw = None
if 'current_text' not in st.session_state: st.session_state.current_text = ""
if 'current_file' not in st.session_state: st.session_state.current_file = ""
if 'ai_result' not in st.session_state: st.session_state.ai_result = ""
if 'sc_memory' not in st.session_state: st.session_state.sc_memory = ""
if 'sc_part' not in st.session_state: st.session_state.sc_part = 1
if 'yt_xray_data' not in st.session_state: st.session_state.yt_xray_data = None
# Bộ nhớ cho tính năng Chat
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- CẤU HÌNH API KEY TỰ ĐỘNG ---
default_gemini = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
default_yt = st.secrets.get("YOUTUBE_API_KEY", "") if "YOUTUBE_API_KEY" in st.secrets else ""

st.sidebar.header("🔑 Cấu Hình API")
gemini_api_key = st.sidebar.text_input("Gemini API Key:", value=default_gemini, type="password")
yt_api_key = st.sidebar.text_input("YouTube API v3 Key:", value=default_yt, type="password")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Tùy chỉnh Model AI")

if st.sidebar.button("🔍 Quét Model Khả Dụng"):
    if not gemini_api_key:
        st.sidebar.error("Vui lòng nhập API Key trước!")
    else:
        with st.sidebar.status("Đang quét máy chủ Google..."):
            try:
                url_check = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}"
                res_check = requests.get(url_check, timeout=10)
                if res_check.status_code == 200:
                    models = [m['name'].replace('models/', '') for m in res_check.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
                    st.sidebar.success("✅ Danh sách Model của bạn:")
                    for m in models:
                        st.sidebar.code(m)
                else:
                    st.sidebar.error(f"Lỗi: {res_check.status_code} - {res_check.text}")
            except Exception as e:
                st.sidebar.error(f"Lỗi kết nối: {e}")

ai_model_choice = st.sidebar.text_input("Nhập chính xác tên Model:", value="gemini-1.5-flash")

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
    st.session_state.sc_memory = ""
    st.session_state.sc_part = 1
    st.session_state.yt_xray_data = None
    st.session_state.chat_history = []

# --- HÀM AI CHẤM ĐIỂM THUMBNAIL TỪ ẢNH ---
def ai_cham_diem_thumbnail(img_url, title):
    if not gemini_api_key: return "⚠️ Thiếu Gemini API Key để chấm điểm ảnh."
    try:
        img_resp = requests.get(img_url, timeout=10)
        if img_resp.status_code != 200: return "⚠️ Không thể tải ảnh Thumbnail để phân tích."
        img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
        model_path = ai_model_choice.strip().replace("models/", "")
        url_gen = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
        headers = {'x-goog-api-key': gemini_api_key, 'Content-Type': 'application/json'}
        prompt = f"""Bạn là chuyên gia Marketing và Thiết kế YouTube. Nhìn bức ảnh Thumbnail này và Tiêu đề: "{title}". Hãy chấm điểm (/10) và đánh giá:
        1. Độ nổi bật (Màu sắc & Bố cục).
        2. Chữ có dễ đọc không.
        3. Sự liên kết kích thích CTR."""
        payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]}]}
        
        # Nâng cấp Timeout lên 5 phút (300 giây)
        res = requests.post(url_gen, headers=headers, json=payload, timeout=300)
        if res.status_code == 200: return res.json()['candidates'][0]['content']['parts'][0]['text']
        else: return f"⚠️ Lỗi từ Google ({res.status_code})."
    except Exception as e:
        return f"⚠️ Lỗi mạng: {e}"

# --- HÀM TẠO NỘI DUNG AI ĐA NĂNG ---
def xu_ly_ai_da_nang(che_do, tu_khoa_list, text_goc, is_continue=False):
    if not gemini_api_key:
        st.error("❌ Vui lòng dán Gemini API Key ở thanh bên trái!")
        return

    model_path = ai_model_choice.strip().replace("models/", "")
    url_gen = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
    headers = {'x-goog-api-key': gemini_api_key, 'Content-Type': 'application/json'}

    if che_do == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)" and not is_continue:
        st.session_state.sc_memory = ""
        st.session_state.sc_part = 1

    if che_do == "📝 Lập Dàn Ý SEO (Outline)": prompt = f"Chuyên gia SEO. Dựa vào Top từ khóa: {tu_khoa_list}. Hãy lập Dàn ý chi tiết chuẩn SEO."
    elif che_do == "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)": prompt = f"Viết bài Blog chuẩn SEO dài khoảng 1000 từ dựa vào bộ từ khóa: {tu_khoa_list}."
    elif che_do == "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)": prompt = f"Viết kịch bản Video dọc (dưới 60s) từ từ khóa: {tu_khoa_list}. Cấu trúc: Hook, Body, CTA."
    elif che_do == "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)": prompt = f"Viết 3 Tiêu đề giật tít và Mô tả sản phẩm đánh vào nỗi đau khách hàng từ từ khóa: {tu_khoa_list}."
    elif che_do == "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)": prompt = f"Dữ liệu từ YouTube đối thủ: {text_goc[:5000]}. Viết Kịch Bản Video YouTube hoàn chỉnh, MỚI MẺ HƠN."
    elif che_do == "🧠 Phản Biện Kịch Bản (Script Doctor)": prompt = f"Đóng vai người phản biện khắt khe nội dung sau: {st.session_state.sc_memory if st.session_state.sc_memory else text_goc[:3000]}. Chỉ ra điểm mù logic, nhịp điệu và 3 đề xuất."
    elif che_do == "📈 Dự Báo Thuật Toán YouTube (Pre-Publish)": prompt = f"Chuyên gia Thuật toán. Chấm điểm kịch bản dự kiến: {st.session_state.sc_memory if st.session_state.sc_memory else text_goc[:4000]}. Dự báo CTR, Retention và 3 bước tối ưu khẩn."
    elif che_do == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)":
        prompt = f"""Biên kịch Phim tài liệu. Viết DUY NHẤT PART {st.session_state.sc_part} nối tiếp Kịch bản: {st.session_state.sc_memory}.
        Cấu trúc BẮT BUỘC: [PART {st.session_state.sc_part}], VOICE SCRIPT, VIETNAMESE TRANSLATION, KEY MESSAGE, PRODUCTION NOTE."""

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    with st.spinner(f"🤖 Đang gọi Model '{model_path}' để xử lý..."):
        try:
            # Nâng cấp Timeout lên 5 phút (300 giây)
            resp_gen = requests.post(url_gen, headers=headers, json=payload, timeout=300) 
            if resp_gen.status_code == 200:
                try:
                    new_content = resp_gen.json()['candidates'][0]['content']['parts'][0]['text']
                    
                    # Lưu vào hệ thống Chat để tương tác tiếp
                    st.session_state.chat_history = [
                        {"role": "user", "parts": [{"text": prompt}]},
                        {"role": "model", "parts": [{"text": new_content}]}
                    ]

                    if che_do == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)":
                        st.session_state.sc_memory += f"\n\n{new_content}"
                        st.session_state.ai_result = st.session_state.sc_memory
                        st.session_state.sc_part += 1
                    else:
                        st.session_state.ai_result = new_content
                    st.rerun()
                except KeyError: st.error(f"❌ Nội dung nhạy cảm bị chặn. Chi tiết: {resp_gen.text}")
            else: st.error(f"❌ Lỗi từ Google API (Mã {resp_gen.status_code}): {resp_gen.text}")
        except Exception as e: st.error(f"Lỗi kết nối mạng: {e}")

# ================= GIAO DIỆN NHẬP LIỆU =================
tab1, tab2, tab3 = st.tabs(["📺 YouTube & Gợi Ý", "🌐 Phân Tích & So Sánh URL", "📁 Tệp Office"])

with tab1:
    che_do_yt = st.radio("Chọn chế độ phân tích:", ("🌍 Gợi ý tìm kiếm Đa quốc gia", "🔗 Bóc tách từ Link Kênh/Video (X-Ray Đối thủ)"))
    st.markdown("---")
    if che_do_yt == "🌍 Gợi ý tìm kiếm Đa quốc gia":
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1: nguon_gợi_ý = st.selectbox("Chọn nền tảng:", ("YouTube", "Google Search"))
        with col_opt2: quoc_gia = st.selectbox("🌍 Chọn thị trường:", ("Việt Nam (VN)", "Mỹ (US)", "Anh (UK)", "Toàn cầu"))
        tu_khoa_nhap = st.text_input("🔑 Nhập từ khóa gốc:")
        market_map = {"Việt Nam (VN)": {"gl": "vn", "hl": "vi"}, "Mỹ (US)": {"gl": "us", "hl": "en"}, "Anh (UK)": {"gl": "gb", "hl": "en"}, "Toàn cầu": {"gl": "", "hl": "en"}}
        
        if st.button("🚀 Quét Từ Khóa"):
            if tu_khoa_nhap:
                gl, hl = market_map[quoc_gia]["gl"], market_map[quoc_gia]["hl"]
                danh_sach_truy_van = [tu_khoa_nhap] + [f"{tu_khoa_nhap} {chr(i)}" for i in range(97, 123)]
                tat_ca_suggests = []
                bar = st.progress(0)
                client_type = "yt" if nguon_gợi_ý == "YouTube" else "chrome"
                for i, q in enumerate(danh_sach_truy_van):
                    try:
                        url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds={client_type}&q={q}&hl={hl}&gl={gl}" if nguon_gợi_ý == "YouTube" else f"https://suggestqueries.google.com/complete/search?client=firefox&q={q}&hl={hl}&gl={gl}"
                        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if res.status_code == 200: tat_ca_suggests.extend(res.json()[1])
                    except: pass
                    bar.progress((i + 1) / len(danh_sach_truy_van))
                    time.sleep(0.05)
                bar.empty()
                full_text = ' '.join(tat_ca_suggests)
                res_kw = trich_xuat_tu_khoa(lam_sach_text(full_text), selected_ngram)
                luu_ket_qua_vao_bo_nho(res_kw, full_text, f"{nguon_gợi_ý}_{market_map[quoc_gia]['gl']}")
            else: st.warning("Vui lòng nhập từ khóa!")
                
    else:
        link_yt = st.text_input("🔗 Nhập link Video YouTube:")
        quet_comment = st.checkbox("💬 Quét luôn cả bình luận", value=True)
        if st.button("🚀 Phân Tích & X-Ray Đối Thủ"):
            if not link_yt: st.warning("Vui lòng nhập đường link Video!")
            elif not yt_api_key: st.error("⚠️ Cần nhập YouTube API v3 Key!")
            else:
                with st.spinner("🕵️ Đang bóc tách dữ liệu..."):
                    try:
                        vid_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', link_yt).group(1)
                        full_content, xray = "", {}
                        vid_res = requests.get(f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={vid_id}&key={yt_api_key}").json()
                        if 'items' in vid_res and len(vid_res['items']) > 0:
                            v = vid_res['items'][0]
                            xray.update({'v_title': v['snippet']['title'], 'v_desc': v['snippet']['description'], 'v_tags': v['snippet'].get('tags', []), 'v_views': v['statistics'].get('viewCount', '0'), 'v_likes': v['statistics'].get('likeCount', '0'), 'v_comments': v['statistics'].get('commentCount', '0'), 'channel_id': v['snippet']['channelId'], 'channel_title': v['snippet']['channelTitle']})
                            thumbs = v['snippet']['thumbnails']
                            xray['v_thumb'] = thumbs.get('maxres', thumbs.get('high', thumbs.get('default')))['url']
                            full_content += f"{xray['v_title']} {xray['v_desc']} " + " ".join(xray['v_tags'])
                            ch_res = requests.get(f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={xray['channel_id']}&key={yt_api_key}").json()
                            if 'items' in ch_res:
                                xray.update({'c_subs': ch_res['items'][0]['statistics'].get('subscriberCount', 'Ẩn'), 'c_views': ch_res['items'][0]['statistics'].get('viewCount', '0'), 'c_videos': ch_res['items'][0]['statistics'].get('videoCount', '0')})
                        if quet_comment:
                            cmt_res = requests.get(f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={vid_id}&maxResults=100&key={yt_api_key}").json()
                            if 'items' in cmt_res: full_content += " " + " ".join([i['snippet']['topLevelComment']['snippet']['textOriginal'] for i in cmt_res['items']])
                        xray['thumb_review'] = ai_cham_diem_thumbnail(xray['v_thumb'], xray['v_title'])
                        st.session_state.yt_xray_data = xray
                        luu_ket_qua_vao_bo_nho(trich_xuat_tu_khoa(lam_sach_text(full_content), selected_ngram), full_content, "youtube_xray_data")
                    except Exception as e: st.error(f"Lỗi: {e}")

with tab2:
    url1 = st.text_input("🔗 Nhập URL trang web:")
    if st.button("🚀 Phân Tích URL") and url1:
        res = requests.get(url1, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for script in soup(["script", "style"]): script.decompose()
        text1 = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'article'])])
        luu_ket_qua_vao_bo_nho(trich_xuat_tu_khoa(lam_sach_text(text1), selected_ngram), text1, "Web_Keywords")

with tab3:
    uploaded_file = st.file_uploader("Chọn file (xlsx, csv, docx, txt):", type=['xlsx', 'csv', 'docx', 'txt'])
    if st.button("🚀 Phân Tích Tệp Office") and uploaded_file:
        file_text = ""
        if uploaded_file.name.endswith('.csv'): file_text = ' '.join(pd.read_csv(uploaded_file).astype(str).values.flatten())
        elif uploaded_file.name.endswith('.txt'): file_text = uploaded_file.read().decode('utf-8', errors='ignore')
        luu_ket_qua_vao_bo_nho(trich_xuat_tu_khoa(lam_sach_text(file_text), selected_ngram), file_text, "office_file")

# ================= BÁO CÁO X-RAY =================
if st.session_state.yt_xray_data:
    x = st.session_state.yt_xray_data
    st.markdown("---")
    st.header("🕵️ BẢNG ĐIỀU KHIỂN X-RAY ĐỐI THỦ")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(x['v_thumb'], use_column_width=True)
        st.markdown(f"**{x['channel_title']}** | Subs: {x.get('c_subs', 'N/A')}")
    with col2:
        st.subheader(x['v_title'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Views", f"{int(x['v_views']):,}")
        c2.metric("Likes", f"{int(x['v_likes']):,}")
        c3.metric("Comments", f"{int(x['v_comments']):,}")
        st.info("Tags Ẩn: " + (", ".join(x['v_tags']) if x['v_tags'] else "Không có"))
        st.success(f"**🤖 AI Đánh Giá Thumbnail:** {x['thumb_review']}")

# ================= TỪ KHÓA & AI =================
if st.session_state.current_kw:
    st.markdown("---")
    df = pd.DataFrame(st.session_state.current_kw, columns=['Từ khóa', 'Tần suất'])
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("🤖 Xưởng Sản Xuất Nội Dung AI")
    che_do_ai = st.selectbox("Chọn hành động:", ["📝 Lập Dàn Ý SEO (Outline)", "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)", "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)", "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)", "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)", "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)", "🧠 Phản Biện Kịch Bản (Script Doctor)", "📈 Dự Báo Thuật Toán YouTube (Pre-Publish)"])
    
    if st.button(f"✨ Kích Hoạt Trợ Lý AI"):
        xu_ly_ai_da_nang(che_do_ai, [i[0] for i in st.session_state.current_kw[:10]], st.session_state.current_text, False)
        
    if st.session_state.ai_result:
        st.success("✅ Trợ lý AI đã hoàn thành nhiệm vụ!")
        st.markdown(st.session_state.ai_result)
        
        # --- KHU VỰC CHAT ĐÍNH KÈM TỆP ĐA NĂNG ---
        st.markdown("---")
        st.subheader("💬 Trợ Lý Tinh Chỉnh & Phân Tích Kèm Tệp")
        st.caption("Chat với AI để gọt giũa kịch bản. Bạn có thể đính kèm Ảnh, file TXT, CSV, DOCX để AI đọc và phân tích theo yêu cầu.")
        
        # In lại lịch sử chat (Bỏ qua prompt gốc hệ thống tạo ra ban đầu)
        for msg in st.session_state.chat_history[2:]:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                for part in msg["parts"]:
                    if "text" in part:
                        st.markdown(part["text"])
                    if "inline_data" in part:
                        img_data = base64.b64decode(part["inline_data"]["data"])
                        st.image(img_data, width=300)

        # Trạm tải tệp đính kèm ngay trên khung chat
        chat_file = st.file_uploader("📎 Đính kèm tệp vào tin nhắn (Ảnh, TXT, DOCX, CSV)", type=['png', 'jpg', 'jpeg', 'txt', 'docx', 'csv'], key="chat_upload")

        # Khung chat chính
        if prompt_chat := st.chat_input("VD: Đánh giá bức ảnh đính kèm và viết lại đoạn mở đầu kịch bản..."):
            
            user_parts = [{"text": prompt_chat}]
            
            # Xử lý nếu người dùng có up tệp
            if chat_file:
                fname = chat_file.name.lower()
                # Nếu là ảnh -> Cho AI nhìn ảnh
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    img_b64 = base64.b64encode(chat_file.getvalue()).decode('utf-8')
                    mime = "image/png" if fname.endswith('.png') else "image/jpeg"
                    user_parts.append({"inline_data": {"mime_type": mime, "data": img_b64}})
                # Nếu là file Text/Word/CSV -> Ép AI đọc toàn bộ chữ
                elif fname.endswith('.txt'):
                    text_content = chat_file.getvalue().decode('utf-8', errors='ignore')
                    user_parts[0]["text"] += f"\n\n[DỮ LIỆU ĐÍNH KÈM TỪ TỆP {fname}]:\n{text_content}"
                elif fname.endswith('.csv'):
                    df_chat = pd.read_csv(chat_file)
                    text_content = ' '.join(df_chat.astype(str).values.flatten())
                    user_parts[0]["text"] += f"\n\n[DỮ LIỆU ĐÍNH KÈM TỪ TỆP {fname}]:\n{text_content}"
                elif fname.endswith('.docx'):
                    doc_chat = docx.Document(chat_file)
                    text_content = ' '.join([p.text for p in doc_chat.paragraphs])
                    user_parts[0]["text"] += f"\n\n[DỮ LIỆU ĐÍNH KÈM TỪ TỆP {fname}]:\n{text_content}"
            
            # In ra màn hình tin nhắn của người dùng
            with st.chat_message("user"):
                st.markdown(prompt_chat)
                if chat_file:
                    if chat_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        st.image(chat_file.getvalue(), width=300)
                    else:
                        st.info(f"📄 Đã đính kèm tệp: {chat_file.name}")
            
            # Lưu lại vào lịch sử trò chuyện
            st.session_state.chat_history.append({"role": "user", "parts": user_parts})
            
            # Đóng gói và Gửi yêu cầu sang Google
            model_path = ai_model_choice.strip().replace("models/", "")
            url_chat = f"https://generativelanguage.googleapis.com/v1beta/models/{model_path}:generateContent"
            headers_chat = {'x-goog-api-key': gemini_api_key, 'Content-Type': 'application/json'}
            payload_chat = {"contents": st.session_state.chat_history}
            
            # Chờ AI trả lời
            with st.chat_message("assistant"):
                with st.spinner("🤖 Đang phân tích dữ liệu đính kèm và xử lý yêu cầu..."):
                    try:
                        # Nâng cấp Timeout lên 5 phút (300 giây)
                        res_chat = requests.post(url_chat, headers=headers_chat, json=payload_chat, timeout=300)
                        if res_chat.status_code == 200:
                            reply = res_chat.json()['candidates'][0]['content']['parts'][0]['text']
                            st.markdown(reply)
                            st.session_state.chat_history.append({"role": "model", "parts": [{"text": reply}]})
                        else:
                            st.error(f"Lỗi API: {res_chat.text}")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
