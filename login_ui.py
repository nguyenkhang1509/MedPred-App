import streamlit as st
from firebase_auth import auth
import requests
import pyrebase
def show_login_success(email):
    st.markdown("""
        <style>
        .success-box {
            background: linear-gradient(to right, #4db8ff, #1e2a38);
            padding: 20px;
            border-radius: 14px;
            color: white;
            font-size: 17px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .avatar {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid white;
            margin-right: 20px;
        }
        .user-row {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='success-box'>
            <div class='user-row'>
                <img src='https://cdn-icons-png.flaticon.com/512/747/747376.png' class='avatar' />
                <div>
                    <b>🎉 Đăng nhập thành công!</b><br>
                    👤 <b>{email}</b><br>
                    ✅ Chào mừng bạn đến với MedPredict!
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.button("🏠 Vào trang chủ", on_click=lambda: st.session_state.update({"menu": "🏠 Home"}), use_container_width=True)



def login_ui():
    st.markdown("## 🔐 Đăng nhập vào MedPredict")
    tab1, tab2 = st.tabs(["🚪 Đăng nhập", "📝 Đăng ký"])

    # ----- ĐĂNG NHẬP -----
    with tab1:
        st.subheader("🔑 Đăng nhập")
        login_email = st.text_input("📧 Email", key="login_email")
        login_password = st.text_input("🔒 Mật khẩu", type="password", key="login_password")

        login_email_error = ""
        login_pass_error = ""
        login_success = False

        if st.button("✅ Đăng nhập", use_container_width=True):
            if not login_email.strip():
                login_email_error = "❗ Vui lòng nhập email."
            if not login_password.strip():
                login_pass_error = "❗ Vui lòng nhập mật khẩu."
            if not login_email_error and not login_pass_error:
                try:
                    user = auth.sign_in_with_email_and_password(login_email.strip(), login_password.strip())
                    st.session_state["user"] = user
                    login_success = True
                except requests.exceptions.HTTPError as e:
                    code = ""
                    if hasattr(e, "response") and e.response is not None:
                        try:
                            code = e.response.json().get("error", {}).get("message", "")
                            
                        except Exception:
                            code = "UNKNOWN_JSON"
                    else:
                        code = "NO_RESPONSE"
                    if  "EMAIL_NOT_FOUND" in code:
                        login_email_error = "❌ Email chưa được đăng ký."
                    elif "INVALID_PASSWORD" in code:
                        login_pass_error = "❌ Mật khẩu không đúng."
                    elif "INVALID_EMAIL" in code:
                        login_email_error = "❌ Email không hợp lệ."
                    elif "MISSING_EMAIL" in code:
                        login_email_error = "❗ Vui lòng nhập email."
                    elif "MISSING_PASSWORD" in code:
                        login_pass_error = "❗ Vui lòng nhập mật khẩu."
                    elif code == "":
                        st.error("⚠️ Firebase trả về lỗi trống.")
                    else:
                        st.error(f"⚠️ Firebase trả về mã lỗi không rõ: `{code}`")

        if login_success:
            st.success(f"✅ Đăng nhập thành công với {login_email}")
            st.balloons()
            st.stop()

        if login_email_error:
            st.markdown(f"<div style='color:red'>{login_email_error}</div>", unsafe_allow_html=True)
        if login_pass_error:
            st.markdown(f"<div style='color:red'>{login_pass_error}</div>", unsafe_allow_html=True)

    # ----- ĐĂNG KÝ -----
    with tab2:
        st.subheader("📝 Đăng ký")
        signup_email = st.text_input("📧 Email mới", key="signup_email")
        signup_password = st.text_input("🔐 Mật khẩu mới (≥6 ký tự)", type="password", key="signup_password")

        signup_email_error = ""
        signup_pass_error = ""
        signup_success = False

        if st.button("🚀 Đăng ký", use_container_width=True):
            if not signup_email.strip():
                signup_email_error = "❗ Vui lòng nhập email."
            if not signup_password.strip():
                signup_pass_error = "❗ Vui lòng nhập mật khẩu."
            elif len(signup_password.strip()) < 6:
                signup_pass_error = "❌ Mật khẩu quá yếu. Phải ≥ 6 ký tự."
            if not signup_email_error and not signup_pass_error:
                try:
                    auth.create_user_with_email_and_password(signup_email.strip(), signup_password.strip())
                    signup_success = True
                except requests.exceptions.HTTPError as e:
                    code = ""
                    # Debug toàn bộ exception
                    st.write("EXCEPTION:", e)
                    if hasattr(e, "response") and e.response is not None:
                        try:
                            st.write("RESPONSE TEXT:", e.response.text)
                            code = e.response.json().get("error", {}).get("message", "")
                        except Exception as ex:
                            st.write("EXCEPTION PARSE JSON:", ex)
                            code = "UNKNOWN_JSON"
                    else:
                        code = "NO_RESPONSE"
                    st.write("FIREBASE ERROR CODE:", code)
                    if "EMAIL_EXISTS" in code:
                        signup_email_error = "❌ Email đã được đăng ký."
                    elif "INVALID_EMAIL" in code:
                        signup_email_error = "❌ Email không hợp lệ."
                    elif "WEAK_PASSWORD" in code:
                        signup_pass_error = "❌ Mật khẩu quá yếu. Phải ≥ 6 ký tự."
                    elif "MISSING_EMAIL" in code:
                        signup_email_error = "❗ Vui lòng nhập email."
                    elif "MISSING_PASSWORD" in code:
                        signup_pass_error = "❗ Vui lòng nhập mật khẩu."
                    elif code == "":
                        st.error("⚠️ Firebase trả về lỗi trống.")
                    else:
                        st.error(f"⚠️ Firebase trả về mã lỗi không rõ: `{code}`")

        if signup_success:
            st.success("✅ Đăng ký thành công! Bạn có thể đăng nhập ngay.")
            st.balloons()

        if signup_email_error:
            st.markdown(f"<div style='color:red'>{signup_email_error}</div>", unsafe_allow_html=True)
        if signup_pass_error:
            st.markdown(f"<div style='color:red'>{signup_pass_error}</div>", unsafe_allow_html=True)