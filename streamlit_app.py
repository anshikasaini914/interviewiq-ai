import streamlit as st
import time
import io
import uuid
import requests
import matplotlib.pyplot as plt

try:
    from mutagen.mp3 import MP3
except ImportError:
    MP3 = None


st.set_page_config(
    page_title="InterviewIQ AI",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
<style>
    /* Centered, narrow column — ChatGPT/Claude style */
    .block-container {
        max-width: 760px;
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Minimal header */
    h1 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.2rem !important;
    }

    /* Chat bubbles — soft, rounded, low visual weight */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0.4rem 0;
    }
    [data-testid="stChatMessageContent"] {
        background: rgba(130,130,150,0.08);
        border-radius: 14px;
        padding: 10px 14px;
    }

    /* Slim timer status line */
    .status-line {
        font-size: 0.85rem;
        color: rgba(200,200,210,0.75);
        padding: 4px 2px 10px 2px;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        margin-bottom: 0.6rem;
    }

    /* Reduce default widget clutter */
    [data-testid="stExpander"] {
        border-radius: 12px;
    }

    /* Smaller, quieter captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        opacity: 0.65;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND URLS
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000/chat"
REPORT_URL = "http://127.0.0.1:8000/report"
TRANSCRIBE_URL = "http://127.0.0.1:8000/transcribe"
TTS_URL = "http://127.0.0.1:8000/text-to-speech"
RESUME_SKILLS_URL = "http://127.0.0.1:8000/extract-resume-skills"


# ============================================================
# TIMER SETTINGS
# ============================================================

OVERALL_TIME = 20 * 60
ANSWER_TIME = 90
TECHNICAL_BUFFER = 10
QUESTION_TIME = ANSWER_TIME + TECHNICAL_BUFFER
LOW_TIME_BUFFER = 120


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_phase" not in st.session_state:
    st.session_state.interview_phase = "intro"

if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = "🎙️ Voice Interview"

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

if "text_key" not in st.session_state:
    st.session_state.text_key = 0

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "interview_start_time" not in st.session_state:
    st.session_state.interview_start_time = None

if "question_start_time" not in st.session_state:
    st.session_state.question_start_time = None

if "question_number" not in st.session_state:
    st.session_state.question_number = 0

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False

if "report" not in st.session_state:
    st.session_state.report = None

if "report_error" not in st.session_state:
    st.session_state.report_error = None

if "resume_skills" not in st.session_state:
    st.session_state.resume_skills = []

if "resume_processed" not in st.session_state:
    st.session_state.resume_processed = False

if "suggested_role" not in st.session_state:
    st.session_state.suggested_role = ""

if "confirmed_role" not in st.session_state:
    st.session_state.confirmed_role = ""

if "intro_sent" not in st.session_state:
    st.session_state.intro_sent = False

if "new_question_pending" not in st.session_state:
    st.session_state.new_question_pending = False

if "question_speaking_until" not in st.session_state:
    st.session_state.question_speaking_until = None

if "answer_times" not in st.session_state:
    st.session_state.answer_times = []

if "evaluated_answers" not in st.session_state:
    st.session_state.evaluated_answers = 0

if "overall_end_time" not in st.session_state:
    st.session_state.overall_end_time = None

if "adaptive_note" not in st.session_state:
    st.session_state.adaptive_note = None


# ============================================================
# TIMER FUNCTIONS
# ============================================================

def start_interview_timer():
    if st.session_state.interview_start_time is None:
        st.session_state.interview_start_time = time.time()


def start_question_timer(delay=0):
    st.session_state.question_start_time = time.time() + max(0, delay)
    st.session_state.question_number += 1
    st.session_state.question_speaking_until = (
        time.time() + delay if delay > 0 else None
    )
    st.session_state.new_question_pending = False


def mark_question_pending():
    st.session_state.new_question_pending = True


# ============================================================
# GET FINAL REPORT
# ============================================================

def load_report():
    try:
        response = requests.get(
            f"{REPORT_URL}/{st.session_state.session_id}",
            timeout=60
        )
        if response.status_code == 200:
            st.session_state.report = response.json()
            st.session_state.report_error = None
            return True
        else:
            st.session_state.report_error = (
                f"Report API returned {response.status_code}: "
                f"{response.text}"
            )
            return False
    except requests.exceptions.RequestException as e:
        st.session_state.report_error = (
            f"Could not connect to report API: {e}"
        )
        return False


def get_audio_duration(audio_bytes):
    if not audio_bytes or MP3 is None:
        return 0
    try:
        return float(MP3(io.BytesIO(audio_bytes)).info.length)
    except Exception:
        return 0


# ============================================================
# SAVE SESSION METRICS
# ============================================================

def save_session_metrics():
    try:
        if st.session_state.interview_start_time is None:
            return

        end_time = st.session_state.overall_end_time or time.time()
        overall_seconds = max(0.0, end_time - st.session_state.interview_start_time)

        requests.post(
            f"http://127.0.0.1:8000/metrics/{st.session_state.session_id}",
            json={
                "overall_seconds": overall_seconds,
                "answer_times": st.session_state.answer_times,
                "evaluated_answers": st.session_state.evaluated_answers,
            },
            timeout=10,
        )
    except requests.exceptions.RequestException:
        pass


# ============================================================
# SEND MESSAGE
# ============================================================

def send_message(message, target_role=""):

    if not message or not message.strip():
        return None

    try:

        response = requests.post(
            BACKEND_URL,
            json={
                "session_id": st.session_state.session_id,
                "message": message,
                "target_role": target_role
            },
            timeout=60
        )

        if response.status_code != 200:
            st.error(
                f"Backend error: {response.status_code}\n\n"
                f"{response.text}"
            )
            return None

        data = response.json()

        if data.get("answer_evaluated") is True:
            start = st.session_state.question_start_time
            if start is not None:
                elapsed = max(0.0, time.time() - start)
                elapsed = min(elapsed, QUESTION_TIME)
                st.session_state.answer_times.append(elapsed)
            st.session_state.evaluated_answers += 1

        st.session_state.messages.append(
            {"role": "user", "content": message}
        )

        ai_reply = data.get("reply", "")

        st.session_state.messages.append(
            {"role": "assistant", "content": ai_reply}
        )

        state = data.get("state", {})

        if data.get("answer_evaluated") is True:
            if data.get("same_question") is True:
                st.session_state.adaptive_note = "Let's clarify that a bit more."
            elif data.get("new_question") is True:
                st.session_state.adaptive_note = None
        elif data.get("ignored_as_noise") is True:
            st.session_state.adaptive_note = "That response wasn't counted — please answer the question."
        elif data.get("new_question") is True:
            st.session_state.adaptive_note = None

        backend_phase = state.get("phase", "")
        st.session_state.interview_phase = backend_phase

        backend_finished = (backend_phase.lower() == "wrapup")

        completion_message = "Thank you for completing the interview"
        reply_finished = (completion_message.lower() in ai_reply.lower())

        if backend_finished or reply_finished:
            st.session_state.interview_finished = True
            st.session_state.overall_end_time = time.time()
            st.session_state.question_start_time = None
            st.session_state.question_speaking_until = None

            save_session_metrics()
            load_report()

            return ai_reply

        if not st.session_state.intro_sent:
            st.session_state.intro_sent = True
        elif data.get("new_question") is True:
            mark_question_pending()

        save_session_metrics()
        return ai_reply

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to backend:\n{e}")
        return None


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(audio_value):
    try:
        audio_bytes = audio_value.getvalue()

        files = {
            "audio_file": ("recording.wav", audio_bytes, "audio/wav")
        }

        response = requests.post(TRANSCRIBE_URL, files=files, timeout=60)

        if response.status_code != 200:
            st.error(
                f"Transcription error: {response.status_code}\n\n{response.text}"
            )
            return None

        data = response.json()
        return data.get("text", "")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not transcribe audio:\n{e}")
        return None


# ============================================================
# TEXT TO SPEECH
# ============================================================

def generate_speech(text):
    try:
        response = requests.post(TTS_URL, params={"text": text}, timeout=60)
        if response.status_code == 200:
            return response.content
        st.warning("Could not generate AI voice.")
    except requests.exceptions.RequestException as e:
        st.warning(f"TTS error: {e}")
    return None


# ============================================================
# RESET INTERVIEW
# ============================================================

def reset_interview():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.interview_phase = "intro"
    st.session_state.audio_key += 1
    st.session_state.text_key += 1
    st.session_state.last_audio = None
    st.session_state.transcript = ""
    st.session_state.is_processing = False
    st.session_state.interview_start_time = None
    st.session_state.question_start_time = None
    st.session_state.question_number = 0
    st.session_state.interview_finished = False
    st.session_state.report = None
    st.session_state.report_error = None
    st.session_state.resume_skills = []
    st.session_state.resume_processed = False
    st.session_state.suggested_role = ""
    st.session_state.confirmed_role = ""
    st.session_state.intro_sent = False
    st.session_state.new_question_pending = False
    st.session_state.question_speaking_until = None
    st.session_state.answer_times = []
    st.session_state.evaluated_answers = 0
    st.session_state.overall_end_time = None
    st.session_state.adaptive_note = None

    st.rerun()


# ============================================================
# SIDEBAR — kept minimal, only essentials
# ============================================================

with st.sidebar:

    st.markdown("**InterviewIQ**")

    st.session_state.interview_mode = st.radio(
        "Mode",
        ["🎙️ Voice Interview", "⌨️ Text Interview"],
        index=0,
        label_visibility="collapsed"
    )

    if st.session_state.confirmed_role:
        st.caption(f"Role: {st.session_state.confirmed_role}")

    if st.session_state.resume_skills:
        st.caption("Skills: " + ", ".join(st.session_state.resume_skills))

    st.button("🔄 Reset", use_container_width=True, on_click=reset_interview)


# ============================================================
# HEADER — minimal
# ============================================================

st.title("InterviewIQ")


# ============================================================
# RESUME UPLOAD + ROLE SELECTION (sirf interview shuru hone se pehle)
# ============================================================

if len(st.session_state.messages) == 0 and not st.session_state.interview_finished:

    with st.expander("Upload resume (optional)", expanded=True):

        resume_file = st.file_uploader(
            "PDF resume",
            type=["pdf"],
            key="resume_uploader",
            label_visibility="collapsed"
        )

        if resume_file is not None and not st.session_state.resume_processed:

            with st.spinner("Analyzing resume..."):

                try:
                    files = {
                        "resume_file": (resume_file.name, resume_file.getvalue(), "application/pdf")
                    }

                    response = requests.post(RESUME_SKILLS_URL, files=files, timeout=60)

                    if response.status_code == 200:
                        data = response.json()
                        skills = data.get("skills", [])
                        st.session_state.suggested_role = data.get("suggested_role", "")

                        if skills:
                            st.session_state.resume_skills = skills

                except requests.exceptions.RequestException:
                    pass

            st.session_state.resume_processed = True

        role_input = st.text_input(
            "Target role",
            value=st.session_state.suggested_role,
            placeholder="e.g. Backend Developer, Data Analyst, Product Manager",
        )

        if st.button("Start Interview", type="primary", use_container_width=True):

            final_role = role_input.strip()
            st.session_state.confirmed_role = final_role
            skills = st.session_state.resume_skills

            if skills:
                opening_message = (
                    f"Hi, I'm ready to begin. My background includes: {', '.join(skills)}. "
                    f"I'd like the interview to focus on these areas if possible."
                )
            else:
                opening_message = "Hi, I'm ready to begin the interview."

            if final_role:
                opening_message += f" I'm targeting a {final_role} role."

            with st.spinner("Preparing..."):
                ai_reply = send_message(opening_message, target_role=final_role)

            if ai_reply and st.session_state.interview_mode == "🎙️ Voice Interview":
                with st.spinner("..."):
                    audio = generate_speech(ai_reply)
                if audio:
                    st.session_state.last_audio = audio
                    st.session_state.new_question_pending = True
                    duration = get_audio_duration(audio)
                    start_question_timer(delay=duration)

            elif ai_reply:
                start_question_timer()

            st.rerun()

    st.stop()


# ============================================================
# START TIMER
# ============================================================

start_interview_timer()


# ============================================================
# TIMER — slim single-line status bar (not metric cards)
# ============================================================

def _render_timer():
    if st.session_state.interview_finished:
        return

    current_time = time.time()

    total_elapsed = (
        current_time - st.session_state.interview_start_time
        if st.session_state.interview_start_time is not None
        else 0
    )
    total_remaining = max(0, OVERALL_TIME - total_elapsed)
    total_minutes = int(total_remaining // 60)
    total_seconds = int(total_remaining % 60)

    if st.session_state.question_start_time is not None:
        question_elapsed = (current_time - st.session_state.question_start_time)
    else:
        question_elapsed = 0

    speaking = (
        st.session_state.question_speaking_until is not None
        and current_time < st.session_state.question_speaking_until
    )

    if speaking:
        label = "🎙️ speaking..."
    elif question_elapsed <= ANSWER_TIME:
        remaining = max(0, ANSWER_TIME - question_elapsed)
        label = f"⏳ {int(remaining // 60):02d}:{int(remaining % 60):02d} to answer"
    else:
        buffer_remaining = max(0, QUESTION_TIME - question_elapsed)
        label = f"⚙️ {int(buffer_remaining)}s buffer"

    if st.session_state.question_speaking_until is not None and not speaking:
        st.session_state.question_speaking_until = None

    st.markdown(
        f"""<div class="status-line">
        ⏱️ {total_minutes:02d}:{total_seconds:02d} left &nbsp;·&nbsp;
        Q{max(st.session_state.question_number, 1)} &nbsp;·&nbsp;
        {label}
        </div>""",
        unsafe_allow_html=True
    )

    if total_remaining <= LOW_TIME_BUFFER and total_remaining > 0:
        st.caption("⚠️ Less than 2 minutes remaining — wrap up soon.")


if not st.session_state.interview_finished:
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        _render_timer = fragment(run_every=1)(_render_timer)
    _render_timer()


# ============================================================
# ADAPTIVE STATUS — quiet caption, not a big info box
# ============================================================

if st.session_state.adaptive_note and not st.session_state.interview_finished:
    st.caption(st.session_state.adaptive_note)

# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ============================================================
# FINAL REPORT (with charts)
# ============================================================

if st.session_state.interview_finished:

    st.divider()
    st.success("Interview completed")
    st.subheader("Your Report")

    if st.session_state.report is None:
        with st.spinner("Generating report..."):
            load_report()

    if st.session_state.report is not None:

        report = st.session_state.report

        correct = report.get("correct_count", 0)
        incorrect = report.get("incorrect_count", 0)
        total = report.get("total_questions", correct + incorrect)

        m1, m2, m3 = st.columns(3)
        m1.metric("Questions", total)
        m2.metric("Correct", correct)
        m3.metric("Incorrect", incorrect)

        accuracy = (correct / total * 100) if total > 0 else 0
        st.progress(min(int(accuracy), 100), text=f"Performance: {accuracy:.0f}%")

        overall_seconds = report.get("overall_time_seconds", 0)
        avg_answer_seconds = report.get("average_hands_on_time_seconds", 0)
        evaluated = report.get("evaluated_answers", correct + incorrect)

        def fmt_duration(seconds):
            seconds = max(0, int(round(seconds or 0)))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"

        tm1, tm2, tm3 = st.columns(3)
        tm1.metric("Total Time", fmt_duration(overall_seconds))
        tm2.metric("Avg Answer Time", fmt_duration(avg_answer_seconds))
        tm3.metric("Evaluated", evaluated)

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Correct vs Incorrect**")
            if correct + incorrect > 0:
                fig1, ax1 = plt.subplots()
                ax1.pie(
                    [correct, incorrect],
                    labels=["Correct", "Incorrect"],
                    autopct="%1.0f%%",
                    colors=["#2ecc71", "#e74c3c"],
                    startangle=90
                )
                ax1.axis("equal")
                st.pyplot(fig1)
            else:
                st.caption("No data yet.")

        with chart_col2:
            st.markdown("**Strong vs Weak Topics**")
            strong_topics = report.get("strong_topics", [])
            weak_topics = report.get("weak_topics", [])

            if strong_topics or weak_topics:
                fig2, ax2 = plt.subplots()
                ax2.bar(
                    ["Strong", "Weak"],
                    [len(strong_topics), len(weak_topics)],
                    color=["#3498db", "#f39c12"]
                )
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
            else:
                st.caption("No topic data.")

        st.divider()

        if report.get("summary"):
            st.markdown("**Summary**")
            st.write(report["summary"])

        col_s, col_w = st.columns(2)

        with col_s:
            st.markdown("**Strengths**")
            for item in strong_topics:
                st.write(f"• {item}")

        with col_w:
            st.markdown("**Improve**")
            for item in weak_topics:
                st.write(f"• {item}")

        with st.expander("Raw report"):
            st.json(report)

    else:
        st.error("Could not load report.")
        if st.session_state.report_error:
            st.code(st.session_state.report_error)

    st.stop()


# ============================================================
# ANSWER AREA
# ============================================================

if st.session_state.interview_mode == "🎙️ Voice Interview":

    audio_value = st.audio_input(
        "Record your answer",
        key=f"audio_{st.session_state.audio_key}",
        label_visibility="collapsed"
    )

    if audio_value is not None:

        if not st.session_state.is_processing:

            st.session_state.is_processing = True

            audio_bytes = audio_value.getvalue()

            if len(audio_bytes) < 8000:
                st.warning("Recording seems too short. Please try again.")
                st.session_state.is_processing = False
                st.session_state.audio_key += 1
                st.rerun()

            else:
                with st.spinner("Transcribing..."):
                    transcript = transcribe_audio(audio_value)

                if transcript and len(transcript.strip()) > 2:

                    st.session_state.transcript = transcript

                    with st.spinner("Thinking..."):
                        ai_reply = send_message(transcript, target_role=st.session_state.confirmed_role)

                    if ai_reply:
                        with st.spinner("..."):
                            audio = generate_speech(ai_reply)

                        if audio:
                            st.session_state.last_audio = audio

                            if st.session_state.new_question_pending:
                                duration = get_audio_duration(audio)
                                start_question_timer(delay=duration)
                else:
                    st.warning("Couldn't hear anything clearly. Please try again.")

                st.session_state.is_processing = False
                st.session_state.audio_key += 1
                st.rerun()


else:

    text_answer = st.chat_input("Type your answer...")

    if text_answer and text_answer.strip():

        with st.spinner("Thinking..."):
            ai_reply = send_message(text_answer, target_role=st.session_state.confirmed_role)

        if ai_reply and st.session_state.new_question_pending:
            start_question_timer()

        st.rerun()


# ============================================================
# AI VOICE FEEDBACK
# ============================================================

if (
    st.session_state.interview_mode == "🎙️ Voice Interview"
    and st.session_state.last_audio
    and not st.session_state.interview_finished
):
    st.audio(
        st.session_state.last_audio,
        format="audio/mpeg",
        autoplay=True
    )