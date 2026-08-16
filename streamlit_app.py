import streamlit as st
import requests
import uuid

BACKEND_URL = "http://127.0.0.1:8000/chat"
REPORT_URL = "http://127.0.0.1:8000/report"

st.set_page_config(page_title="InterviewIQ AI", page_icon="🎯", layout="centered")

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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_phase" not in st.session_state:
    st.session_state.interview_phase = "greeting"

with st.sidebar:
    st.subheader("Interview Session")
    st.write("Domain: Data Science")
    st.write("Level: Beginner–Intermediate")
    st.divider()
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.interview_phase = "greeting"
        st.rerun()

AVATARS = {"user": "👤", "assistant": "💼"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.write(msg["content"])

if st.session_state.interview_phase == "wrapup":
    st.divider()
    if st.button("📊 Get Feedback Report"):
        with st.spinner("Generating your report..."):
            try:
                response = requests.get(f"{REPORT_URL}/{st.session_state.session_id}")
                response.raise_for_status()
                report = response.json()

                st.subheader("Interview Feedback Report")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Questions", report["total_questions"])
                col2.metric("Correct", report["correct_count"])
                col3.metric("Incorrect", report["incorrect_count"])

                st.write("**Strong Topics:**", ", ".join(report["strong_topics"]) or "None identified")
                st.write("**Weak Topics:**", ", ".join(report["weak_topics"]) or "None identified")
                st.write("**Summary:**")
                st.info(report["summary"])

            except requests.exceptions.RequestException:
                st.error("Could not fetch report. Please try again.")

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
                data = response.json()
                ai_reply = data["reply"]
                new_phase = data["state"]["phase"]
            except requests.exceptions.RequestException:
                ai_reply = "Sorry, I couldn't connect to the server. Please try again."
                new_phase = st.session_state.interview_phase
        st.write(ai_reply)

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

    
    if new_phase == "wrapup" and st.session_state.interview_phase != "wrapup":
        st.session_state.interview_phase = new_phase
        st.rerun()
    else:
        st.session_state.interview_phase = new_phase