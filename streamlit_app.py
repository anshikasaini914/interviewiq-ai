import streamlit as st
import requests
import uuid


# ============================================================
# BACKEND URLS
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000/chat"
REPORT_URL = "http://127.0.0.1:8000/report"
TRANSCRIBE_URL = "http://127.0.0.1:8000/transcribe"
TTS_URL = "http://127.0.0.1:8000/text-to-speech"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InterviewIQ AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE  (everything MUST live in st.session_state,
# not directly on st, or it resets every rerun)
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_phase" not in st.session_state:
    st.session_state.interview_phase = "greeting"

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "text_key" not in st.session_state:
    st.session_state.text_key = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎯 InterviewIQ AI")

    st.caption(
        "AI-Powered Data Science Interview Simulator"
    )

    st.divider()

    st.subheader("Interview Session")

    st.write("📊 **Domain**")
    st.caption("Data Science")

    st.write("📈 **Level**")
    st.caption("Beginner – Intermediate")

    st.write("🤖 **Mode**")
    st.caption("AI Adaptive Interview")

    st.divider()

    # Question count
    assistant_messages = [
        message
        for message in st.session_state.messages
        if message["role"] == "assistant"
    ]

    question_count = len(assistant_messages)

    st.write("### Progress")

    if question_count == 0:
        st.caption("Interview is starting")
    else:
        st.caption(
            f"{question_count} question(s) completed"
        )

    st.divider()

    if st.button(
        "🔄 Reset Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.interview_phase = "greeting"
        st.session_state.audio_key += 1
        st.session_state.text_key += 1

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🎯 InterviewIQ AI")

st.caption(
    "Practice Data Science interviews with an AI interviewer"
)


# ============================================================
# INTERVIEW STATUS
# ============================================================

if st.session_state.interview_phase == "greeting":
    st.info("🟢 Ready to begin your interview")
elif st.session_state.interview_phase == "wrapup":
    st.success("✅ Interview completed")
else:
    st.info("🎤 Interview in progress")


# ============================================================
# CHAT
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])


# ============================================================
# FEEDBACK REPORT
# ============================================================

if st.session_state.interview_phase == "wrapup":

    st.divider()
    st.subheader("📊 Interview Feedback")

    if st.button("Generate Feedback Report", use_container_width=True):

        with st.spinner("Analyzing your interview..."):

            try:
                response = requests.get(
                    f"{REPORT_URL}/{st.session_state.session_id}"
                )
                response.raise_for_status()
                report = response.json()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions", report.get("total_questions", 0))
                with col2:
                    st.metric("Correct", report.get("correct_count", 0))
                with col3:
                    st.metric("Incorrect", report.get("incorrect_count", 0))

                st.divider()

                st.subheader("💪 Strong Topics")
                strong_topics = report.get("strong_topics", [])
                if strong_topics:
                    for topic in strong_topics:
                        st.success(topic)
                else:
                    st.caption("No strong topics identified.")

                st.subheader("📚 Topics to Improve")
                weak_topics = report.get("weak_topics", [])
                if weak_topics:
                    for topic in weak_topics:
                        st.warning(topic)
                else:
                    st.caption("No weak topics identified.")

                st.subheader("📝 Interview Summary")
                st.write(report.get("summary", "No summary available."))

            except requests.exceptions.RequestException:
                st.error(
                    "Could not fetch the feedback report. "
                    "Please make sure the backend is running."
                )


# ============================================================
# ANSWER AREA
# ============================================================

if st.session_state.interview_phase != "wrapup":

    st.divider()

    input_col, mic_col, send_col = st.columns(
        [7, 1, 1],
        vertical_alignment="bottom"
    )

    with input_col:
        user_input = st.text_input(
            "answer",
            placeholder="Type your answer here...",
            label_visibility="collapsed",
            key=f"text_{st.session_state.text_key}"
        )

    with mic_col:
        with st.popover("🎙️", use_container_width=True):
            st.write("### Voice Answer")
            st.caption("Click the microphone below and speak your answer.")
            audio_value = st.audio_input(
                "Record",
                label_visibility="collapsed",
                key=f"audio_{st.session_state.audio_key}"
            )

    with send_col:
        send_clicked = st.button("➤", use_container_width=True)

    final_input = None

    # ------------------------------------------------------
    # VOICE ANSWER
    # ------------------------------------------------------
    if audio_value is not None:

        with st.spinner("🎙️ Transcribing..."):

            try:
                response = requests.post(
                    TRANSCRIBE_URL,
                    files={
                        "audio_file": ("recording.wav", audio_value, "audio/wav")
                    }
                )
                response.raise_for_status()
                transcribed_text = response.json().get("text", "").strip()

                if transcribed_text:
                    final_input = transcribed_text
                else:
                    st.warning("I couldn't hear your answer clearly. Please try again.")

            except requests.exceptions.RequestException:
                st.error("Could not transcribe your audio. Please try typing your answer.")

            finally:
                st.session_state.audio_key += 1

    # ------------------------------------------------------
    # TEXT ANSWER
    # ------------------------------------------------------
    elif send_clicked:
        if user_input.strip():
            final_input = user_input.strip()
        else:
            st.warning("Please type your answer first.")

    # ------------------------------------------------------
    # SEND TO BACKEND
    # ------------------------------------------------------
    if final_input:

        st.session_state.messages.append({"role": "user", "content": final_input})

        with st.spinner("🤖 Interviewer is thinking..."):

            try:
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "session_id": st.session_state.session_id,
                        "message": final_input
                    }
                )
                response.raise_for_status()
                data = response.json()
                ai_reply = data["reply"]
                new_phase = data["state"]["phase"]

            except requests.exceptions.RequestException:
                ai_reply = (
                    "Sorry, I couldn't connect to "
                    "the interview server. Please try again."
                )
                new_phase = st.session_state.interview_phase

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
        st.session_state.interview_phase = new_phase
        st.session_state.audio_key += 1
        st.session_state.text_key += 1

        try:
            tts_response = requests.post(TTS_URL, params={"text": ai_reply})
            tts_response.raise_for_status()
            st.audio(tts_response.content, format="audio/mp3", autoplay=True)
        except requests.exceptions.RequestException:
            pass

        st.rerun()