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

# --- CẤU HÌNH API KEY TỰ ĐỘNG ---
default_gemini = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
default_yt = st.secrets.get("YOUTUBE_API_KEY", "") if "YOUTUBE_API_KEY" in st.secrets else ""

st.sidebar.header("🔑 Cấu Hình API")
gemini_api_key = st.sidebar.text_input("Gemini API Key:", value=default_gemini, type="password")
yt_api_key = st.sidebar.text_input("YouTube API v3 Key:", value=default_yt, type="password")

st.sidebar.markdown("---")
st.sidebar.header("🤖 Tùy chỉnh Model AI")

# TÍNH NĂNG MỚI: QUÉT DANH SÁCH MODEL TỪ GOOGLE
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

# Mặc định đổi sang bản có đánh số -001 để đảm bảo tỷ lệ sống 100%
ai_model_choice = st.sidebar.text_input("Nhập chính xác tên Model:", value="gemini-1.5-flash-001")
st.sidebar.caption("Gợi ý an toàn: gemini-1.5-flash-001, gemini-1.5-flash-002, gemini-1.5-pro-001")

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
        prompt = f"""Bạn là một chuyên gia Marketing và Thiết kế YouTube. 
        Hãy nhìn vào bức ảnh Thumbnail này và Tiêu đề video: "{title}". 
        Hãy chấm điểm (trên thang 10) và đánh giá ngắn gọn theo 3 tiêu chí:
        1. 🎨 **Độ nổi bật (Màu sắc & Bố cục):** Ảnh có nổi bật trên nền giao diện tối (Dark mode) không?
        2. ✍️ **Chữ (Text on Image):** Chữ có dễ đọc trên màn hình điện thoại nhỏ không? Thông điệp có khơi gợi sự tò mò không?
        3. 💡 **Sự liên kết (Tiêu đề & Ảnh):** Ảnh và tiêu đề có bổ trợ cho nhau để tạo ra tỷ lệ click (CTR) cao không? Đề xuất 1 cách cải thiện."""
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }]
        }
        
        res = requests.post(url_gen, headers=headers, json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"⚠️ Lỗi từ Google ({res.status_code}): Không thể xử lý ảnh bằng model {model_path}."
    except Exception as e:
        return f"⚠️ Lỗi mạng khi xử lý ảnh: {e}"

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

    if che_do == "📝 Lập Dàn Ý SEO (Outline)":
        prompt = f"Bạn là chuyên gia SEO. Dựa vào Top từ khóa: {tu_khoa_list}. Hãy đề xuất 3 Tiêu đề hấp dẫn và lập Dàn ý (H2, H3) chi tiết chuẩn SEO."
    elif che_do == "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)":
        prompt = f"Bạn là một Copywriter chuyên nghiệp. Dựa vào bộ từ khóa: {tu_khoa_list}. Hãy viết một bài Blog chuẩn SEO hoàn chỉnh, dài khoảng 1000 từ. Bố cục rõ ràng, lồng ghép từ khóa một cách tự nhiên nhất."
    elif che_do == "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)":
        prompt = f"Bạn là chuyên gia sáng tạo nội dung Video ngắn. Từ các từ khóa xu hướng này: {tu_khoa_list}. Hãy viết kịch bản Video dọc (dưới 60 giây). Bao gồm: 1. Hook, 2. Body, 3. CTA. Có text chạy trên màn hình."
    elif che_do == "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)":
        prompt = f"Bạn là chuyên gia chốt sale TMĐT. Dựa vào danh sách từ khóa: {tu_khoa_list}. Hãy viết: 1. 3 Tiêu đề Sản phẩm giật tít. 2. Đoạn Mô tả sản phẩm đánh mạnh vào nỗi đau khách hàng, lồng ghép từ khóa."
    elif che_do == "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)":
        prompt = f"Dữ liệu trích xuất từ Video YouTube của đối thủ: \n\n{text_goc[:5000]}\n\nBạn là YouTuber chuyên nghiệp. Hãy viết ra một Kịch Bản Video YouTube hoàn chỉnh, MỚI MẺ và HẤP DẪN HƠN cho kênh của tôi."
    elif che_do == "🧠 Phản Biện Kịch Bản (Script Doctor)":
        script_to_critique = st.session_state.sc_memory if st.session_state.sc_memory else text_goc[:3000]
        prompt = f"""Bạn là một Đạo diễn và Cố vấn Kịch bản (Script Doctor) cực kỳ khắt khe. 
        Hãy đọc kỹ nội dung/kịch bản sau đây:
        {script_to_critique}
        Hãy đóng vai trò người phản biện để giúp tôi nâng cấp kịch bản này lên mức xuất sắc nhất. Phân tích: Điểm mù Logic, Điểm nghẽn Cảm xúc, Nhịp điệu (Pacing) và đưa ra 3 Đề xuất sửa đổi cụ thể."""
    elif che_do == "📈 Dự Báo Thuật Toán YouTube (Pre-Publish)":
        content_to_audit = st.session_state.sc_memory if st.session_state.sc_memory else text_goc[:4000]
        prompt = f"""Bạn là một Chuyên gia Thuật toán YouTube (YouTube Growth Hacker) nội bộ.
        Dựa trên bộ từ khóa đang nhắm mục tiêu: {tu_khoa_list}
        Và dữ liệu nội dung/kịch bản (hoặc mô tả) dự kiến xuất bản: 
        {content_to_audit}
        
        Hãy đóng vai hệ thống AI của YouTube để "chấm điểm" và dự báo mức độ thành công của video này TRƯỚC KHI TẢI LÊN. Trình bày báo cáo rõ ràng theo 4 phần:
        
        1. 🎯 **Chỉ Số Đề Xuất (Algorithm Score):** Chấm điểm /100 dựa trên khả năng Viral và chuẩn SEO.
        2. 🧲 **Dự Báo Tỷ Lệ Nhấp (CTR):** Nội dung và chủ đề này có đủ sức nặng kích thích sự tò mò không?
        3. ⏳ **Tỷ Lệ Giữ Chân (Retention Rate):** Cấu trúc nội dung có hứa hẹn giữ chân người xem qua 30 giây đầu (Hook) tốt không?
        4. 🚀 **3 Bước Tối Ưu Khẩn Cấp:** Đưa ra 3 hành động cụ thể BẮT BUỘC phải sửa để video dễ dàng cắn luồng Traffic đề xuất tốt nhất."""
    elif che_do == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)":
        prompt = f"""Bạn là một Biên kịch Phim tài liệu chuyên nghiệp. Đang viết kịch bản chuẩn "SILENT CAPITAL".
        Từ khóa: {tu_khoa_list}
        Kịch bản cũ đã viết (KHÔNG VIẾT LẠI CHÚNG): {st.session_state.sc_memory}
        
        HÃY VIẾT DUY NHẤT PART {st.session_state.sc_part}. (Khoảng 80-95 từ tiếng Anh, thiết kế đọc trong 30-35 giây).
        Cấu trúc BẮT BUỘC:
        [PART {st.session_state.sc_part} - TÊN PART IN HOA]
        English word count: ...
        VOICE SCRIPT
        [Tiếng Anh - Ngắt dòng từng câu ngắn]
        VIETNAMESE TRANSLATION
        [Tiếng Việt - Ngắt dòng tương ứng]
        KEY MESSAGE
        [1 câu tóm tắt Anh-Việt]
        PRODUCTION NOTE
        [Hướng dẫn góc máy, ánh sáng, Master Style: Flat 2D stickman, cinematic lighting]"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    with st.spinner(f"🤖 Đang gọi Model '{model_path}' để xử lý {'tiếp Part ' + str(st.session_state.sc_part) if is_continue else 'yêu cầu: ' + che_do} ..."):
        try:
            resp_gen = requests.post(url_gen, headers=headers, json=payload, timeout=90) 
            if resp_gen.status_code == 200:
                try:
                    new_content = resp_gen.json()['candidates'][0]['content']['parts'][0]['text']
                    if che_do == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)":
                        st.session_state.sc_memory += f"\n\n{new_content}"
                        st.session_state.ai_result = st.session_state.sc_memory
                        st.session_state.sc_part += 1
                    else:
                        st.session_state.ai_result = new_content
                    st.rerun()
                except KeyError:
                    st.error(f"❌ Nội dung nhạy cảm bị Bộ lọc an toàn (Safety Filter) của Google chặn lại. Chi tiết lỗi: {resp_gen.text}")
            else:
                st.error(f"❌ Lỗi từ Google API (Mã {resp_gen.status_code}). Nguyên nhân: {resp_gen.text}")
        except Exception as e:
            st.error(f"Lỗi kết nối mạng hoặc quá hạn chờ: {e}")


# ================= GIAO DIỆN NHẬP LIỆU (TABS) =================
tab1, tab2, tab3 = st.tabs(["📺 YouTube & Gợi Ý", "🌐 Phân Tích & So Sánh URL", "📁 Tệp Office"])

with tab1:
    che_do_yt = st.radio("Chọn chế độ phân tích:", ("🌍 Gợi ý tìm kiếm Đa quốc gia", "🔗 Bóc tách từ Link Kênh/Video (X-Ray Đối thủ)"))
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
        link_yt = st.text_input("🔗 Nhập link Video YouTube (Để kích hoạt X-Ray, cần có YouTube API Key):")
        quet_comment = st.checkbox("💬 Quét luôn cả bình luận (Gom nhặt nỗi đau khách hàng)", value=True)
        
        if st.button("🚀 Phân Tích & X-Ray Đối Thủ"):
            if not link_yt:
                st.warning("Vui lòng nhập đường link Video!")
            elif not yt_api_key:
                st.error("⚠️ Bạn cần nhập YouTube Data API v3 Key ở cột bên trái để sử dụng tính năng X-Ray nâng cao!")
            else:
                with st.spinner("🕵️ Đang bóc tách toàn bộ dữ liệu tình báo của đối thủ..."):
                    try:
                        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', link_yt)
                        if not video_id_match:
                            st.error("Link Video không hợp lệ.")
                        else:
                            vid_id = video_id_match.group(1)
                            full_content = ""
                            xray = {}
                            
                            # 1. LẤY DATA VIDEO (X-Ray Video)
                            vid_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={vid_id}&key={yt_api_key}"
                            vid_res = requests.get(vid_url).json()
                            if 'items' in vid_res and len(vid_res['items']) > 0:
                                v_data = vid_res['items'][0]
                                xray['v_title'] = v_data['snippet']['title']
                                xray['v_desc'] = v_data['snippet']['description']
                                xray['v_tags'] = v_data['snippet'].get('tags', [])
                                xray['v_views'] = v_data['statistics'].get('viewCount', '0')
                                xray['v_likes'] = v_data['statistics'].get('likeCount', '0')
                                xray['v_comments'] = v_data['statistics'].get('commentCount', '0')
                                xray['channel_id'] = v_data['snippet']['channelId']
                                xray['channel_title'] = v_data['snippet']['channelTitle']
                                
                                # Tìm Thumbnail nét nhất
                                thumbs = v_data['snippet']['thumbnails']
                                if 'maxres' in thumbs: xray['v_thumb'] = thumbs['maxres']['url']
                                elif 'high' in thumbs: xray['v_thumb'] = thumbs['high']['url']
                                else: xray['v_thumb'] = thumbs['default']['url']
                                
                                full_content += f"{xray['v_title']} {xray['v_desc']} " + " ".join(xray['v_tags'])
                                
                                # 2. LẤY DATA KÊNH (Trinh sát kênh)
                                ch_url = f"https://www.googleapis.com/youtube/v3/channels?part=statistics&id={xray['channel_id']}&key={yt_api_key}"
                                ch_res = requests.get(ch_url).json()
                                if 'items' in ch_res:
                                    xray['c_subs'] = ch_res['items'][0]['statistics'].get('subscriberCount', 'Ẩn')
                                    xray['c_views'] = ch_res['items'][0]['statistics'].get('viewCount', '0')
                                    xray['c_videos'] = ch_res['items'][0]['statistics'].get('videoCount', '0')
                                
                                # 3. LẤY TOP 5 VIDEO CỦA KÊNH
                                search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={xray['channel_id']}&maxResults=5&order=viewCount&type=video&key={yt_api_key}"
                                search_res = requests.get(search_url).json()
                                xray['top_videos'] = []
                                if 'items' in search_res:
                                    for item in search_res['items']:
                                        xray['top_videos'].append({
                                            'title': item['snippet']['title'],
                                            'vid_id': item['id']['videoId']
                                        })

                            # 4. QUÉT COMMENT (Nếu chọn)
                            if quet_comment:
                                cmt_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={vid_id}&maxResults=100&key={yt_api_key}"
                                cmt_res = requests.get(cmt_url).json()
                                if 'items' in cmt_res:
                                    for item in cmt_res['items']: 
                                        full_content += " " + item['snippet']['topLevelComment']['snippet']['textOriginal']
                            
                            # 5. CHẤM ĐIỂM THUMBNAIL (AI)
                            st.session_state.yt_xray_data = xray
                            st.session_state.yt_xray_data['thumb_review'] = ai_cham_diem_thumbnail(xray['v_thumb'], xray['v_title'])

                            # Trích xuất từ khóa
                            words = lam_sach_text(full_content)
                            res_kw = trich_xuat_tu_khoa(words, selected_ngram)
                            luu_lich_su(f"X-Ray YouTube")
                            luu_ket_qua_vao_bo_nho(res_kw, full_content, "youtube_xray_data")

                    except Exception as e:
                        st.error(f"Lỗi: {e}")

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

# ================= GIAO DIỆN BÁO CÁO X-RAY =================
if st.session_state.yt_xray_data:
    xdata = st.session_state.yt_xray_data
    st.markdown("---")
    st.header("🕵️ BẢNG ĐIỀU KHIỂN X-RAY ĐỐI THỦ")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.image(xdata['v_thumb'], use_column_width=True, caption="Thumbnail Thực Tế")
        st.markdown(f"**Tên Kênh:** {xdata['channel_title']}")
        st.markdown(f"👥 **Subscribers:** {xdata.get('c_subs', 'N/A')}")
        st.markdown(f"🎬 **Tổng Video Kênh:** {xdata.get('c_videos', 'N/A')}")
        st.markdown(f"👁️ **Tổng View Kênh:** {xdata.get('c_views', 'N/A')}")
    
    with col_v2:
        st.subheader(xdata['v_title'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Lượt Xem (Views)", f"{int(xdata['v_views']):,}")
        c2.metric("Lượt Thích (Likes)", f"{int(xdata['v_likes']):,}")
        c3.metric("Bình luận (Comments)", f"{int(xdata['v_comments']):,}")
        
        st.markdown("**🏷️ Thẻ Tags Bị Ẩn (Bí mật kéo view):**")
        st.info(", ".join(xdata['v_tags']) if xdata['v_tags'] else "Video này không dùng thẻ tag ẩn.")
        
        st.markdown("**🤖 AI Đánh Giá Ảnh Bìa (Thumbnail Audit):**")
        st.success(xdata['thumb_review'])

    with st.expander("🔥 Tình Báo Kênh: Top 5 Video Nhiều View Nhất Kênh Này", expanded=False):
        for i, vid in enumerate(xdata['top_videos']):
            st.markdown(f"{i+1}. [{vid['title']}](https://www.youtube.com/watch?v={vid['vid_id']})")


# ================= GIAO DIỆN HIỂN THỊ TỪ KHÓA & AI =================
if st.session_state.current_kw:
    st.markdown("---")
    st.subheader(f"🔑 Khai Thác Từ Khóa & Nỗi Đau Khách Hàng")
    
    df = pd.DataFrame(st.session_state.current_kw, columns=['Từ khóa / Cụm từ', 'Tần suất'])
    df['Ý định tìm kiếm (Intent)'] = df['Từ khóa / Cụm từ'].apply(phan_loai_y_dinh)
    df['Nhóm (Cluster)'] = df['Từ khóa / Cụm từ'].apply(gom_nhom)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1: st.download_button("📥 Tải file CSV", data=df.to_csv(index=False).encode('utf-8-sig'), file_name=f'{st.session_state.current_file}.csv', mime='text/csv')
    with col_dl2: st.download_button("📊 Tải file Excel", data=tao_file_excel(df), file_name=f'{st.session_state.current_file}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        st.dataframe(df.head(20), use_container_width=True)
    with c2:
        intent_counts = df['Ý định tìm kiếm (Intent)'].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(intent_counts, labels=intent_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        ax_pie.axis('equal') 
        st.pyplot(fig_pie)
    with c3:
        words_dict = dict(st.session_state.current_kw)
        wc = WordCloud(width=400, height=400, background_color='white', colormap='viridis').generate_from_frequencies(words_dict)
        fig_wc, ax_wc = plt.subplots(figsize=(4, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    st.markdown("---")
    
    # 🤖 KHU VỰC TRỢ LÝ AI (MENU ĐA NĂNG)
    st.subheader("🤖 Xưởng Sản Xuất Nội Dung Bằng Trí Tuệ Nhân Tạo")
    che_do_ai = st.selectbox("Chọn hành động bạn muốn AI thực hiện:", [
        "📝 Lập Dàn Ý SEO (Outline)",
        "✍️ Viết Bài Full Chuẩn SEO (1000+ từ)",
        "🎬 Kịch Bản Video Ngắn (TikTok/Reels/Shorts)",
        "🕵️ Tái tạo Kịch Bản YouTube Đối Thủ (Phân tích Link)",
        "🛒 Tối Ưu Gian Hàng TMĐT (Tiêu đề & Mô tả)",
        "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)",
        "🧠 Phản Biện Kịch Bản (Script Doctor)",
        "📈 Dự Báo Thuật Toán YouTube (Pre-Publish)"
    ])
    
    top_10_words = [item[0] for item in st.session_state.current_kw[:10]]
    
    if st.button(f"✨ Kích Hoạt Trợ Lý AI"):
        xu_ly_ai_da_nang(che_do_ai, top_10_words, st.session_state.current_text, is_continue=False)
        
    if st.session_state.ai_result:
        st.success("✅ Trợ lý AI đã hoàn thành nhiệm vụ!")
        st.markdown(st.session_state.ai_result)
        
        if che_do_ai == "🎬 Kịch Bản Phim Tài Liệu (Chuẩn Silent Capital)":
            st.info(f"💡 AI vừa hoàn thành **Part {st.session_state.sc_part - 1}**. Bạn có thể yêu cầu AI viết tiếp hoặc tải file về máy.")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(f"✍️ Viết tiếp Part {st.session_state.sc_part}"):
                    xu_ly_ai_da_nang(che_do_ai, top_10_words, st.session_state.current_text, is_continue=True)
            with col_btn2:
                if st.session_state.sc_memory:
                    st.download_button("📥 Tải File Kịch Bản (.TXT)", data=st.session_state.sc_memory, file_name="Kich_Ban_Silent_Capital.txt", mime="text/plain")
