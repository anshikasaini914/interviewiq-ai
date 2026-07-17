import streamlit as st
import requests
import uuid

BACKEND_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="InterviewIQ AI",
    page_icon="🎯",
    layout="centered"
)

# --- Custom CSS (same as before) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 10px; padding: 10px; }
    h1 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    .subtitle { color: #9ca3af; font-size: 15px; margin-top: -10px; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.title("InterviewIQ AI")
st.markdown('<p class="subtitle">AI-Powered Data Science Interview Simulator</p>', unsafe_allow_html=True)

# --- Session state init: session_id + messages ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.subheader("Interview Session")
    st.write("Domain: Data Science")
    st.write("Level: Beginner–Intermediate")
    st.divider()
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())  # naya session bhi
        st.rerun()

AVATARS = {"user": "👤", "assistant": "💼"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.write(msg["content"])

user_input = st.chat_input("Type your answer here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(user_input)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Interviewer is typing..."):
            try:
                response = requests.post(BACKEND_URL, json={
                    "session_id": st.session_state.session_id,
                    "message": user_input
                })
                response.raise_for_status()
                ai_reply = response.json()["reply"]
            except requests.exceptions.RequestException:
                ai_reply = "Sorry, I couldn't connect to the server. Please try again."
        st.write(ai_reply)

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})