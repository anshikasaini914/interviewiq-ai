import streamlit as st
import requests

# Backend URL 
BACKEND_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="InterviewIQ AI", page_icon="🎤")
st.title("🎤 InterviewIQ AI — Data Science Interview Bot")

# To store conversation history (inside session)
if "messages" not in st.session_state:
    st.session_state.messages = []

# for past conversation chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User new input
user_input = st.chat_input("Type your answer here...")

if user_input:
    # Add user message in history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call backned
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(BACKEND_URL, json={"message": user_input})
                response.raise_for_status()
                ai_reply = response.json()["reply"]
            except requests.exceptions.RequestException:
                ai_reply = "Sorry, I couldn't connect to the server. Please try again."

        st.write(ai_reply)

    # To add AI reply in history
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})