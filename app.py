from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.saving import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from glob import glob
from skimage import io
import re
from login_ui import login_ui
from firebase_auth import auth

def main():
    st.title("MedPredict - Dự đoán khối u não")
    with st.sidebar:
        selected = option_menu("Menu", ["🏠 Home", "💬 Chatbot Gemini", "🧪 Dự đoán ảnh MRI","💉 Dự đoán phương pháp điều trị","🔐 Sign In"],
        icons=["house", "chat", "image", "syringe", "person"],
        menu_icon="cast", default_index=0)

    

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_key")  
genai.configure(api_key=GEMINI_API_KEY)
MODEL_PATH = "model_epoch_03.keras"
INPUT_SIZE = (128, 128)
CLASS_LABELS =  ['Normal','glioma_tumor','meningioma_tumor','pituitary_tumor']

st.set_page_config(page_title="MedPred App", layout="wide", page_icon="🧠")
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)



st.markdown("""
    <style>
    /* Toàn bộ nền app */
    .stApp {
        background-color: #1e2a38;
        color: #f0f0f0;
    }

    /* Tất cả chữ */
    * {
        color: #f0f0f0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2b3e50;
    }

    section[data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }

    /* Nút bấm */
    .stButton>button {
        background-color: #4db8ff;
        color: #001f33;
        border-radius: 8px;
        font-weight: bold;
    }

    .stNumberInput label, .stSelectbox label {
        color: #cccccc;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
        <style>
        .treat-card {
            background: #222c3c;
            border-radius: 14px;
            padding: 20px 20px 10px 20px;
            box-shadow: 0 2px 8px #0002;
            margin-bottom: 18px;
        }
        .treat-label {
            color: #4db8ff;
            font-size: 17px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .treat-value {
            font-size: 20px;
            font-weight: bold;
            color: #fff;
            margin-bottom: 8px;
        }
        .treat-icon {
            font-size: 22px;
            margin-right: 8px;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

Gemini_model = genai.GenerativeModel("gemini-1.5-flash")
Gemini_model = genai.GenerativeModel("gemini-1.5-flash",
                                     system_instruction=f"""
Bạn là một trợ lý sức khỏe AI thân thiện,có khả năng ghi nhớ và liên kết hội thoại, được tích hợp vào một ứng dụng phân tích ảnh MRI để hỗ trợ người dùng phát hiện và tìm hiểu về các loại khối u não. Bạn luôn nói chuyện gần gũi, dễ hiểu, không dùng từ ngữ máy móc hoặc đe dọa. App này tên là MedPredict, người phát triển là Nguyễn Bá Khang, một học sinh lớp 10 có niềm đam mê với lập trình và AI. Với thông tin về người phát triển, chỉ trả lời khi người dùng hỏi về nguồn gốc ứng dụng hoặc muốn biết thêm về tác giả. Với các câu hỏi về hướng dẫn, không đề cập đến người phát triển.
Bạn có thể trả lời các câu hỏi về kết quả phân tích ảnh MRI, hướng dẫn sử dụng ứng dụng, thông tin về các loại khối u não và phương pháp điều trị. Bạn cũng có thể cung cấp kiến thức y khoa cơ bản nhưng không thay thế ý kiến bác sĩ.

🎯 Cách ứng xử:
Chỉ trả lời một lần duy nhất cho mỗi câu hỏi. Không lặp lại hướng dẫn hay câu văn đã nói trừ khi người dùng hỏi lại rõ ràng.

Nếu người dùng tiếp tục bằng các câu như “làm rồi” hoặc “vậy tiếp theo sao”, hãy nhớ những gì  run người dùng vừa làm để trả lời theo ngữ cảnh.

Trả lời thật kỹ, rõ ràng, đầy đủ thông tin cần thiết, không quá dài dòng cũng không ngắn gọn.

Luôn thân thiện, tránh nói năng cứng nhắc hoặc quá máy móc.
Các câu trả lời cần được trình bày một cách có hệ thống, ví dụ như sử dụng các biểu tượng cảm xúc để làm nổi bật ý chính, hoặc sử dụng các tiêu đề phụ để phân chia nội dung.
📌 Khi người dùng hỏi cách sử dụng app:
Trả lời như sau (chỉ một lần duy nhất):

Bước 1: Tải ảnh MRI não của bạn lên trang phân tích khối u
Bước 2: Chờ một chút để hệ thống nhận diện loại khối u (nếu có)
Bước 3: Nhận kết quả và hỏi mình nếu cần giải thích thêm nhé!
Bước 4: Quay lại báo cho mình biết kết quả để mình hướng dẫn tiếp nhé!
Bước 5: Nếu là khối u cần điều trị, bạn có thể tiến đến trang đề xuất phương pháp điều trị phù hợp
Bước 6: Nếu là khối u lành tính như pituitary, bạn không cần gấp gáp xem phần điều trị. Hãy tìm hiểu qua mình trước nhé!
Thêm các icon ở mỗi câu, format các bước dễ nhìn, align đều nhau và tách hàng ra.
Sau đó, nếu người dùng hỏi lại kiểu như:

“vậy giờ làm gì tiếp theo?”
→ Trả lời:

"Bạn chỉ cần tải ảnh MRI lên rồi quay lại đây với mình khi có kết quả nhé!"

“làm rồi”
→ Trả lời:

"Tuyệt! Bạn nhận được kết quả gì vậy? Mình sẽ giải thích giúp bạn hiểu rõ hơn 💡"

📌 Khi người dùng hỏi “vậy tôi làm tiếp gì”, sau khi đã làm bước trước:
Nếu bạn đã nói rằng người dùng nên tải ảnh lên, và giờ họ nói "làm rồi" hoặc "xong rồi", thì bạn cần trả lời dựa vào logic:

Nếu chưa nhận kết quả:

"Bạn có thấy kết quả nào hiện ra không? Nếu có, gửi mình tên loại u hoặc kết quả để mình hỗ trợ nhé!"

Nếu nhận được kết quả "Normal":

Chúc mừng người dùng trước, thể hiện sự vui mừng, cũng như khuyên họ duy trì lối sống lành mạnh

Nếu nhận được kết quả khối u cụ thể:

"Bạn đang bị dạng u nào vậy? Gửi mình biết tên u để mình nói rõ thêm và hướng dẫn tiếp nhé."

Nếu người dùng đã nhận kết quả khối u, hãy hỏi họ về tên khối u hoặc kết quả cụ thể để bạn có thể giải thích rõ hơn.

Kết thúc nhẹ nhàng:

"📌 Bạn nên xác nhận với bác sĩ trước khi áp dụng nhé."


Nếu là pituitary, nói rằng lành tính và không cần điều trị vội

🔁 Luôn phản hồi theo trạng thái hiện tại:
Ví dụ, nếu user nói:

“Tôi upload rồi, giờ sao?” → Xác nhận và yêu cầu họ cho biết kết quả.


Lưu ý: Nếu kết quả khối u là "pituitary", bạn hãy nói rằng đây là một khối u lành tính, không cần gấp gáp xem phần điều trị. Gợi ý người dùng tìm hiểu qua bạn trước.


Ví dụ: "Glioma là một loại khối u hình thành từ tế bào thần kinh đệm. Nó có thể tiến triển nhanh hoặc chậm tuỳ loại."
Sau khi giải thích, nếu là khối u lành tính như pituitary, hãy nói rõ:
"Pituitary là khối u lành tính, thường không cần điều trị gấp. Vậy nên, hãy duy trì lối sống lành mạnh và theo dõi sức khỏe định kỳ nhé!"
Nếu là hai khối u còn lại (glioma, meningioma), hãy giải thích rõ ràng về chúng. Sau đó, hướng dẫn họ sang phần hệ thống đề xuất điều trị phù hợp.
Trả lời là : Tiếp theo, hãy sang phần đề xuất điều trị để biết phương pháp phù hợp nhé!
Kết thúc bằng:

"💡 Những thông tin này mang tính tham khảo. Bạn nên thảo luận thêm với bác sĩ để có kết luận chính xác nhé!"

Luôn diễn giải nhẹ nhàng:

"Radiation therapy là phương pháp sử dụng tia năng lượng cao để tiêu diệt tế bào ung thư. Thường được dùng sau phẫu thuật để ngăn u tái phát."

Kết thúc bằng:

"📌 Bạn hãy trao đổi với bác sĩ để xác nhận liệu pháp này có phù hợp với tình trạng sức khỏe hiện tại của bạn không nhé."

🌿 Khi kết quả là Normal:
Thể hiện sự vui mừng một cách ấm áp:

"🎉 Tuyệt vời! Hình ảnh MRI của bạn không phát hiện khối u bất thường.
🧘‍♀️ Đừng quên duy trì lối sống lành mạnh như ăn uống khoa học, vận động thường xuyên và ngủ đủ giấc nhé!"

Không cần chuyển hướng đến đề xuất điều trị

🧑‍⚕️ Khi được hỏi kiến thức y khoa:
Cung cấp thông tin ngắn gọn, dễ hiểu trước

KHÔNG lặp đi lặp lại "Tôi không phải bác sĩ", mà chỉ cần 1 câu ngắn sau cùng:

"🧠 Đây là thông tin bạn có thể tham khảo. Tuy nhiên, mình vẫn khuyên bạn xác nhận lại với bác sĩ chuyên khoa để có chẩn đoán chính xác nhất nhé!"

🤖 Cách trả lời tổng quát:
Luôn thân thiện, chào người dùng nếu là câu đầu tiên

Dùng icon nhẹ nhàng (😊, 🔍, 💡, 🎯, ✅)

Tránh từ ngữ như “tôi không thể”, “tôi không được phép”, trừ khi thực sự cần thiết

Khi không biết thông tin: nói nhẹ nhàng là bạn đang cập nhật thêm, và sẵn sàng giúp với nội dung khác


Trích thông tin theo từ khóa (tên điều trị, tên loại u)

Hiển thị thông tin gọn, rõ, tránh trích nguyên văn dài dòng

🔁 Khi người dùng đã chẩn đoán xong:
Hãy khuyến khích họ quay lại chatbot trước khi xem phần điều trị

Vì có một số khối u lành tính (như pituitary) không cần xử lý gấp

Ví dụ:

"📢 Hãy cho tôi biết kết quả chẩn đoán để tôi tư vấn nhé!"
Với các khối u như pituitary, hãy nói rõ đây là khối u lành tính, không cần gấp gáp xem phần điều trị. Giải thích với họ không cần lo lắng quá. Với các khối u khác, hãy hướng dẫn họ đến phần điều trị.

🎯 Mục tiêu tổng thể:
Giúp người dùng cảm thấy:

1.Được hướng dẫn tận tình

2.Hiểu rõ tình trạng bản thân

3.Biết bước tiếp theo cần làm

4.Không bị hoang mang hay lo sợ""")

# ------------- LOAD MODEL -------------
loaded_model = load_model(MODEL_PATH)

# ------------- SIDEBAR MENU -------------
with st.sidebar:
    selected = option_menu(
        menu_title="🧠 MedPred",
        options=["🏠 Home", "💬 Chatbot Gemini", "🧪 Dự đoán ảnh MRI","💉 Dự đoán phương pháp điều trị","🔐 Sign In"],
        icons=["house", "robot", "activity"],
        default_index=0
    )

# ------------- TRANG CHỦ -------------
if selected == "🏠 Home":
    st.markdown(
        """
        <style>
        .big-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #4db8ff;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #f0f0f0;
            margin-bottom: 1.2rem;
        }
        .feature-card {
            background: #222c3c;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 8px #0002;
            margin-bottom: 18px;
            color: #f0f0f0;
        }
        .feature-icon {
            font-size: 2rem;
            margin-right: 10px;
            vertical-align: middle;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<div class='big-title'>🧠 MedPredict - Trợ lý AI Y tế toàn diện</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ứng dụng AI hỗ trợ phân tích ảnh MRI, gợi ý điều trị và tư vấn sức khỏe cá nhân hóa.</div>", unsafe_allow_html=True)

    st.image("https://base.vn/wp-content/uploads/2025/04/Ai-trong-y-te.webp", 
              caption="AI & Y học - Sức mạnh của công nghệ cho sức khỏe cộng đồng")

    st.divider()

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(
            "<div class='feature-card'><span class='feature-icon'>🧬</span><b>Phân loại khối u MRI</b><br>Nhận diện nhanh chóng, chính xác các loại khối u não từ ảnh MRI chỉ với một cú nhấp chuột.</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            "<div class='feature-card'><span class='feature-icon'>💉</span><b>Gợi ý điều trị thông minh</b><br>Đưa ra phương pháp điều trị phù hợp dựa trên dữ liệu cá nhân và y học hiện đại.</div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            "<div class='feature-card'><span class='feature-icon'>🤖</span><b>Chatbot AI y tế</b><br>Hỏi đáp mọi thắc mắc về sức khỏe, kết quả MRI, kiến thức y học và hướng dẫn sử dụng app.</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.info("🔒 Bảo mật dữ liệu: MedPredict cam kết bảo vệ tuyệt đối thông tin cá nhân và kết quả y tế của bạn.", icon="🔒")

    st.success("🚀 Bắt đầu ngay: Hãy chọn chức năng từ menu bên trái để trải nghiệm sức mạnh của AI trong y học!", icon="🚀")

    with st.expander("📋 Hướng dẫn sử dụng nhanh"):
        st.markdown("""
        **1. Phân tích ảnh MRI:**  
        - Chọn mục **Dự đoán ảnh MRI** ở menu bên trái  
        - Tải ảnh MRI não lên và nhận kết quả phân loại khối u

        **2. Gợi ý điều trị:**  
        - Chọn mục **Dự đoán phương pháp điều trị**  
        - Nhập thông tin bệnh nhân để nhận gợi ý điều trị phù hợp

        **3. Chatbot AI:**  
        - Chọn mục **Chatbot Gemini**  
        - Đặt câu hỏi về sức khỏe, kết quả MRI hoặc cách sử dụng app

        **4. Đánh giá trải nghiệm:**  
        - Đừng quên gửi đánh giá để MedPredict ngày càng hoàn thiện hơn!
        """)

    st.warning("💡 Lưu ý: Kết quả từ MedPredict chỉ mang tính chất tham khảo. Hãy luôn tham khảo ý kiến bác sĩ chuyên khoa!", icon="💡")
   
    
   

# ------------- CHATBOT GEMINI -------------


elif selected == "💬 Chatbot Gemini":
    import re

    st.title(":rainbow[AI Health Chatbot]")

    st.markdown("""
        <style>
        @keyframes popIn {
            0% { transform: scale(0.85) translateY(20px); opacity: 0; }
            80% { transform: scale(1.05) translateY(-2px); opacity: 1; }
            100% { transform: scale(1) translateY(0); opacity: 1; }
        }
        .chat-bubble {
            border-radius: 16px;
            padding: 14px 20px;
            margin-bottom: 10px;
            max-width: 80%;
            font-size: 17px;
            line-height: 1.5;
            box-shadow: 0 2px 8px #0002;
            animation: popIn 0.38s cubic-bezier(.68,-0.55,.27,1.55);
            word-break: break-word;
        }
        .chat-user {
            background: linear-gradient(90deg, #4db8ff 60%, #1e2a38 100%);
            color: #fff !important;
            margin-left: auto;
            text-align: right;
        }
        .chat-assistant {
            background: #222c3c;
            color: #f0f0f0 !important;
            margin-right: auto;
            text-align: left;
        }
        .chat-avatar {
            width: 36px; height: 36px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 8px;
        }
        .chat-row {
            display: flex; align-items: flex-end; margin-bottom: 6px;
        }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.assistant { justify-content: flex-start; }
        .loading-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            margin: 0 2px;
            background: #4db8ff;
            border-radius: 50%;
            animation: blink 1.2s infinite both;
        }
        .loading-dot:nth-child(2) { animation-delay: 0.2s;}
        .loading-dot:nth-child(3) { animation-delay: 0.4s;}
        @keyframes blink {
            0%, 80%, 100% { opacity: 0.2; }
            40% { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Trợ lý sức khỏe AI MedPredict")
    st.markdown("Bạn có thể hỏi về kết quả MRI, hướng dẫn sử dụng, kiến thức y học, hoặc các thắc mắc khác.")

    if "last_mri_result" in st.session_state:
        last_result = st.session_state["last_mri_result"]
        st.markdown(
            f"""
            <div style="background: #2b3e50; border-radius: 12px; padding: 12px 16px; margin-bottom:12px;">
                <b>🧠 Kết quả MRI gần nhất:</b> <span style="color:#4db8ff;">{last_result['label']}</span>
                <span style="color:#aaa;">({last_result['confidence']:.2%})</span>
            </div>
            """, unsafe_allow_html=True
        )

    if 'history_log' not in st.session_state or not isinstance(st.session_state.history_log, list):
        st.session_state.history_log = [
            {"role":"assistant",
             "content":"Xin chào, tôi là trợ lý của MedPredict. Bạn cần hỏi gì về sức khỏe, kết quả MRI hoặc cách sử dụng app?"}
        ]
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False

    def send_message():
        if st.session_state.chat_input.strip() != "":
            st.session_state.is_generating = True
            st.session_state.history_log.append({"role": "user", "content": st.session_state.chat_input})
            st.session_state.chat_input = ""

    chat_placeholder = st.container()
    with chat_placeholder:
        for idx, message in enumerate(st.session_state.history_log):
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="chat-row user">
                        <div class="chat-bubble chat-user">
                            <img src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" class="chat-avatar"/>
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )
            elif message["role"] == "assistant":
                content = re.sub(r'(</div>\s*)+$', '', message["content"].strip())
                st.markdown(
                    f"""
                    <div class="chat-row assistant">
                        <div class="chat-bubble chat-assistant">
                            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="chat-avatar"/>
                            {content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

        if st.session_state.is_generating:
            st.markdown(
                """
                <div class="chat-row assistant">
                    <div class="chat-bubble chat-assistant">
                        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" class="chat-avatar"/>
                        <span>Đang soạn trả lời</span>
                        <span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    st.text_input(
        "Nhập câu hỏi cho trợ lý...",
        key="chat_input",
        label_visibility="collapsed",
        placeholder="Nhập nội dung và nhấn Enter...",
        help="Bạn có thể hỏi về kết quả MRI, hướng dẫn sử dụng, kiến thức y học,...",
        on_change=send_message
    )

    if st.session_state.is_generating:
        user_message = None
        for i in range(len(st.session_state.history_log)-1, -1, -1):
            if st.session_state.history_log[i]["role"] == "user":
                if i == len(st.session_state.history_log)-1 or st.session_state.history_log[i+1]["role"] != "assistant":
                    user_message = st.session_state.history_log[i]["content"]
                    break
        if user_message:
            context = ""
            if "last_mri_result" in st.session_state:
                context = f"Kết quả MRI gần nhất của tôi là: {st.session_state['last_mri_result']['label']} ({st.session_state['last_mri_result']['confidence']:.2%})"
            full_prompt = user_message + ("\n" + context if context else "")
            response = Gemini_model.generate_content(full_prompt)
            bot_reply = response.text
            bot_reply = re.sub(r'(</div>\s*)+$', '', bot_reply.strip())
            st.session_state.history_log.append({"role":"assistant", "content": bot_reply})
        st.session_state.is_generating = False
        st.rerun()

# ------------- DỰ ĐOÁN MRI -------------
elif selected == "🧪 Dự đoán ảnh MRI":
    st.title("🧠 Phân loại khối u não từ ảnh MRI")
    st.markdown(
        "<h4 style='color:#4db8ff;'>Tải ảnh MRI não lên để hệ thống phân tích và dự đoán loại khối u</h4>",
        unsafe_allow_html=True
    )

    # Kiểm tra đăng nhập
    if "user" not in st.session_state:
        st.warning("🔐 Vui lòng đăng nhập để sử dụng chức năng dự đoán ảnh MRI.", icon="🔒")
        st.info("Bạn có thể đăng nhập tại mục 'Sign In' trong menu bên trái.")
        st.stop()

    with st.expander("❓ Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        1. Nhấn **Tải ảnh MRI não lên** và chọn file ảnh.
        2. Ảnh sẽ hiển thị bên trái, nhấn **🔍 Dự đoán** để xem kết quả.
        3. Kết quả và xác suất từng loại khối u sẽ hiển thị bên phải.
        """)

    def preprocess_PIL_keep_aspect(pil_img, input_size=(128,128)):
        img = pil_img.convert("RGB")
        img.thumbnail(input_size, Image.LANCZOS)
        new_img = Image.new("RGB", input_size, (0,0,0))
        left = (input_size[0] - img.width) // 2
        top = (input_size[1] - img.height) // 2
        new_img.paste(img, (left, top))
        img_array = keras_image.img_to_array(new_img)
        img_array = np.expand_dims(img_array, axis=0)
        test_datagen = keras_image.ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True
        )
        img_generator = test_datagen.flow(img_array, batch_size=1)
        return img_generator

    uploaded_file = st.file_uploader("📤 Tải ảnh MRI não lên", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            img_pil = Image.open(uploaded_file)
            st.image(img_pil, caption="🖼️ Ảnh MRI đã tải lên", use_column_width=True)
        with col2:
            st.markdown("<h5 style='color:#4db8ff;'>Kết quả phân tích</h5>", unsafe_allow_html=True)
            if st.button("🔍 Dự đoán", use_container_width=True):
                img_gen = preprocess_PIL_keep_aspect(img_pil, INPUT_SIZE)
                predictions = loaded_model.predict(next(img_gen))
                prediction_idx = np.argmax(predictions)
                predicted_label = CLASS_LABELS[prediction_idx]
                confidence = float(np.max(predictions))

                st.markdown(
                    f"""
                    <div style="background: #222c3c; border-radius: 12px; padding: 18px 16px; box-shadow: 0 2px 8px #0002; margin-bottom:16px;">
                        <h3 style="color:#4db8ff; margin-bottom:0;">✅ Kết quả dự đoán:</h3>
                        <p style="font-size:22px; font-weight:bold; color:#fff;">{predicted_label} <span style="color:#4db8ff;">({confidence:.2%})</span></p>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown("### 📊 Xác suất từng loại:")
                for i, prob in enumerate(predictions[0]):
                    st.markdown(
                        f"<b>{CLASS_LABELS[i]}</b>",
                        unsafe_allow_html=True
                    )
                    st.progress(float(prob), text=f"{prob:.2%}")

# ------------- DỰ ĐOÁN PHƯƠNG PHÁP ĐIỀU TRỊ -------------
elif selected == "💉 Dự đoán phương pháp điều trị":
    st.title("🩺 Dự đoán phương pháp điều trị cho bệnh nhân")

    # Kiểm tra đăng nhập
    if "user" not in st.session_state:
        st.warning("🔐 Vui lòng đăng nhập để sử dụng chức năng dự đoán phương pháp điều trị.", icon="🔒")
        st.info("Bạn có thể đăng nhập tại mục 'Sign In' trong menu bên trái.")
        st.stop()
                    

    else:
    

        df = pd.read_csv('BrainTumor.csv')
        df['Tumor Type'] = df['Tumor Type'].replace({
            'Glioblastoma': 'Glioma',   
            'Astrocytoma': 'Glioma'
        })
        df.drop(columns=['Patient ID','Tumor Grade','Tumor Location','Treatment Outcome','Time to Recurrence (months)','Recurrence Site','Survival Time (months)'], inplace=True)
        le_gender = LabelEncoder()
        le_tumor = LabelEncoder()
        le_treatment = LabelEncoder()
        df['Gender'] = le_gender.fit_transform(df['Gender'])
        df['Tumor Type'] = le_tumor.fit_transform(df['Tumor Type'])
        df['Treatment'] = le_treatment.fit_transform(df['Treatment'])
        x = df[['Age','Gender','Tumor Type']]
        y = df['Treatment']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        col1, col2 = st.columns([1,2], gap="large")
        with col1:
            st.markdown("<div class='treat-card'>", unsafe_allow_html=True)
            st.markdown("<div class='treat-label'><span class='treat-icon'>🧑‍⚕️</span>Thông tin bệnh nhân</div>", unsafe_allow_html=True)
            age = st.number_input("Tuổi", min_value=0, max_value=72, value=40, step=1, key="age_input2")
            gender_map = {"👨 Nam": "Male", "👩 Nữ": "Female"}
            gender_input = st.selectbox("Giới tính", list(gender_map.keys()), key="gender_input2")
            gender_clean = gender_map[gender_input]
            gender_encoded = le_gender.transform([gender_clean])[0]
            tumor_input = st.selectbox("Loại khối u", le_tumor.classes_, key="tumor_input2")
            tumor_encoded = le_tumor.transform([tumor_input])[0]
            if st.button("🎯 Dự đoán", use_container_width=True):
                prediction = model.predict([[age, gender_encoded, tumor_encoded]])
                treatment = le_treatment.inverse_transform(prediction)[0]
                st.session_state["last_treatment"] = treatment
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='treat-card'>", unsafe_allow_html=True)
            st.markdown("<div class='treat-label'><span class='treat-icon'>💡</span>Kết quả gợi ý điều trị</div>", unsafe_allow_html=True)
            if "last_treatment" in st.session_state:
                st.markdown(f"<div class='treat-value'>✅ {st.session_state['last_treatment']}</div>", unsafe_allow_html=True)
                st.success("📌 Bạn nên xác nhận với bác sĩ trước khi áp dụng phương pháp này.")
            else:
                st.info("Vui lòng nhập thông tin và nhấn nút dự đoán để nhận gợi ý điều trị.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='treat-card'>", unsafe_allow_html=True)
            st.markdown("<div class='treat-label'><span class='treat-icon'>🌟</span>Đánh giá trải nghiệm</div>", unsafe_allow_html=True)
            rating_emoji = {
                1: "😡 Rất tệ",
                2: "😞 Không hài lòng",
                3: "😐 Bình thường",
                4: "🙂 Hài lòng",
                5: "🤩 Tuyệt vời"
            }
            rating = st.selectbox("Chọn mức độ hài lòng", list(rating_emoji.values()), key="rating_select2")
            if st.button("📤 Gửi đánh giá", key="rating_send2", use_container_width=True):
                new_row = pd.DataFrame([[rating]], columns=["rating"])
                new_row.to_csv("ratings.csv", mode="a", header=False, index=False)
                st.success("🎉 Cảm ơn bạn đã đánh giá!")
                st.balloons()
            st.markdown("</div>", unsafe_allow_html=True)
# ------------- ĐĂNG NHẬP -------------
elif selected == "🔐 Sign In":
    if "user" in st.session_state:
        st.success(f"👋 Chào mừng, bạn đã đăng nhập!")
        if st.button("Đăng xuất", use_container_width=True):
            del st.session_state["user"]
            st.rerun()
    else:
        login_ui()


    
