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
    page_title="AI Data Science Interview",
    page_icon="🎙️",
    layout="wide"
)


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

OVERALL_TIME = 20 * 60       # 20 minutes
ANSWER_TIME = 90              # Candidate gets 90 seconds to answer
TECHNICAL_BUFFER = 10         # Extra 10 seconds for processing/technical delay
QUESTION_TIME = ANSWER_TIME + TECHNICAL_BUFFER
LOW_TIME_BUFFER = 120         # 2 min — warning zone before overall time runs out


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


# ============================================================
# TIMER FUNCTIONS
# ============================================================

def start_interview_timer():
    # Overall interview clock starts once the interview screen is active.
    if st.session_state.interview_start_time is None:
        st.session_state.interview_start_time = time.time()


def start_question_timer(delay=0):
    # IMPORTANT: answer time starts AFTER the interviewer has finished
    # delivering the question. For voice mode, delay is the TTS duration.
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
    """Return MP3 duration in seconds when mutagen is available."""
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

def send_message(message):

    if not message or not message.strip():
        return None

    try:

        response = requests.post(
            BACKEND_URL,
            json={
                "session_id": st.session_state.session_id,
                "message": message
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

        # Record hands-on answer time ONLY for a real evaluated answer.
        # Noise, greetings, repeat requests, and ignored speech do not count.
        if data.get("answer_evaluated") is True:
            start = st.session_state.question_start_time
            if start is not None:
                elapsed = max(0.0, time.time() - start)
                elapsed = min(elapsed, QUESTION_TIME)
                st.session_state.answer_times.append(elapsed)
            st.session_state.evaluated_answers += 1

        # ----------------------------------------------------
        # SAVE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": message
            }
        )

        # ----------------------------------------------------
        # SAVE AI RESPONSE
        # ----------------------------------------------------

        ai_reply = data.get("reply", "")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ai_reply
            }
        )

        # ----------------------------------------------------
        # READ BACKEND STATE
        # ----------------------------------------------------

        state = data.get("state", {})

        backend_phase = state.get(
            "phase",
            ""
        )

        st.session_state.interview_phase = backend_phase

        # ====================================================
        # IMPORTANT FIX
        # ====================================================

        backend_finished = (
            backend_phase.lower() == "wrapup"
        )

        completion_message = (
            "Thank you for completing the interview"
        )

        reply_finished = (
            completion_message.lower()
            in ai_reply.lower()
        )

        if backend_finished or reply_finished:

            st.session_state.interview_finished = True

            st.session_state.overall_end_time = time.time()
            st.session_state.question_start_time = None
            st.session_state.question_speaking_until = None

            save_session_metrics()
            load_report()

            return ai_reply

        # ----------------------------------------------------
        # TIMER CONTROL
        # Reset the question timer ONLY when the backend says a
        # NEW question was generated.
        #
        # This is important:
        # - "no", "hey", "thank you" -> same question -> no reset
        # - wrong answer + hint -> same question -> no reset
        # - repeat request -> same question -> no reset
        # - actual progression -> new question -> reset
        # ----------------------------------------------------

        if not st.session_state.intro_sent:
            st.session_state.intro_sent = True
        elif data.get("new_question") is True:
            # Do NOT start the answer timer here. The interviewer may still
            # be speaking through TTS. The timer is started after TTS ends.
            mark_question_pending()

        save_session_metrics()
        return ai_reply

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not connect to backend:\n{e}"
        )

        return None


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(audio_value):

    try:

        audio_bytes = audio_value.getvalue()

        files = {
            "audio_file": (
                "recording.wav",
                audio_bytes,
                "audio/wav"
            )
        }

        response = requests.post(
            TRANSCRIBE_URL,
            files=files,
            timeout=60
        )

        if response.status_code != 200:

            st.error(
                f"Transcription error: "
                f"{response.status_code}\n\n"
                f"{response.text}"
            )

            return None

        data = response.json()

        return data.get("text", "")

    except requests.exceptions.RequestException as e:

        st.error(
            f"Could not transcribe audio:\n{e}"
        )

        return None


# ============================================================
# TEXT TO SPEECH
# ============================================================

def generate_speech(text):

    try:

        response = requests.post(
            TTS_URL,
            params={
                "text": text
            },
            timeout=60
        )

        if response.status_code == 200:

            return response.content

        st.warning(
            "Could not generate AI voice."
        )

    except requests.exceptions.RequestException as e:

        st.warning(
            f"TTS error: {e}"
        )

    return None


# ============================================================
# RESET INTERVIEW
# ============================================================

def reset_interview():

    st.session_state.session_id = str(
        uuid.uuid4()
    )

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
    st.session_state.intro_sent = False
    st.session_state.new_question_pending = False
    st.session_state.question_speaking_until = None
    st.session_state.answer_times = []
    st.session_state.evaluated_answers = 0
    st.session_state.overall_end_time = None

    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎙️ Interview Settings")

    st.subheader("Interview Mode")

    st.session_state.interview_mode = st.radio(
        "Choose mode",
        [
            "🎙️ Voice Interview",
            "⌨️ Text Interview"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("⏱️ Interview Settings")

    st.write(
        "Overall time: **20 minutes**"
    )

    st.write(
        "Time per question: **90 seconds**"
    )
   
    st.divider()
    st.caption("⏱️ Timer updates live every second")

    if st.button(
        "🔄 Reset Interview",
        use_container_width=True
    ):

        reset_interview()


# ============================================================
# HEADER
# ============================================================

st.title("🎙️ AI Data Science Interview")

st.caption(
    "Practice your Data Science interview with an AI interviewer."
)


# ============================================================
# RESUME UPLOAD (sirf interview shuru hone se pehle)
# ============================================================

if len(st.session_state.messages) == 0 and not st.session_state.interview_finished:

    with st.expander("📄 Upload your resume (optional) — questions will adapt to your skills", expanded=True):

        resume_file = st.file_uploader(
            "Upload PDF resume",
            type=["pdf"],
            key="resume_uploader"
        )

        if resume_file is not None and not st.session_state.resume_processed:

            with st.spinner("📄 Analyzing your resume..."):

                try:
                    files = {
                        "resume_file": (resume_file.name, resume_file.getvalue(), "application/pdf")
                    }

                    response = requests.post(RESUME_SKILLS_URL, files=files, timeout=60)

                    if response.status_code == 200:
                        skills = response.json().get("skills", [])

                        if skills:
                            st.session_state.resume_skills = skills
                            st.success(f"Detected skills: {', '.join(skills)}")
                        else:
                            st.warning("Couldn't detect specific skills — proceeding with general questions.")
                    else:
                        st.warning("Could not analyze resume — proceeding with general questions.")

                except requests.exceptions.RequestException:
                    st.warning("Could not analyze resume — proceeding with general questions.")

            st.session_state.resume_processed = True

        if st.button("🚀 Start Interview", type="primary", use_container_width=True):

            skills = st.session_state.resume_skills

            if skills:
                opening_message = (
                    f"Hi, I'm ready to begin. My background includes: {', '.join(skills)}. "
                    f"I'd like the interview to focus on these areas if possible."
                )
            else:
                opening_message = "Hi, I'm ready to begin the interview."

            with st.spinner("🤖 Interviewer is preparing..."):
                ai_reply = send_message(opening_message)

            # The first interviewer question is also a real question.
            # Generate/play it, then START Q1 timer only after speech ends.
            if ai_reply and st.session_state.interview_mode == "🎙️ Voice Interview":
                with st.spinner("🔊 Interviewer is speaking..."):
                    audio = generate_speech(ai_reply)
                if audio:
                    st.session_state.last_audio = audio
                    st.session_state.new_question_pending = True
                    # Reserve the question number now, but delay the answer clock
                    # until the TTS question has finished playing.
                    duration = get_audio_duration(audio)
                    start_question_timer(delay=duration)

            elif ai_reply:
                # In text mode the question is already displayed, so start immediately.
                start_question_timer()

            st.rerun()

    st.stop()


# ============================================================
# START TIMER (only reaches here once interview has begun)
# ============================================================

start_interview_timer()


# ============================================================
# TIMER DISPLAY
# ============================================================

def _render_timer():
    """Live timer fragment — reruns once per second."""
    if st.session_state.interview_finished:
        return

    current_time = time.time()

    total_elapsed = (
        current_time - st.session_state.interview_start_time
        if st.session_state.interview_start_time is not None
        else 0
    )

    total_remaining = max(0, OVERALL_TIME - total_elapsed)

    if st.session_state.question_start_time is not None:
        question_elapsed = (
            current_time - st.session_state.question_start_time
        )
    else:
        question_elapsed = 0

    # While the interviewer is still speaking, do NOT consume answer time.
    if (
        st.session_state.question_speaking_until is not None
        and current_time < st.session_state.question_speaking_until
    ):
        speaking_remaining = max(0, st.session_state.question_speaking_until - current_time)
        q_minutes = int(speaking_remaining // 60)
        q_seconds = int(speaking_remaining % 60)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Interview Time", f"{total_minutes:02d}:{total_seconds:02d}")
        with col2:
            st.metric("🎙️ Interviewer Speaking", f"{q_minutes:02d}:{q_seconds:02d}")
        with col3:
            st.metric("❓ Question", max(st.session_state.question_number, 1))
        return

    if st.session_state.question_speaking_until is not None:
        st.session_state.question_speaking_until = None

    # First 90 sec = actual answer time.
    if question_elapsed is None:
        answer_remaining = ANSWER_TIME
        buffer_remaining = TECHNICAL_BUFFER
    else:
        answer_remaining = max(0, ANSWER_TIME - question_elapsed)
        buffer_elapsed = max(0, question_elapsed - ANSWER_TIME)
        buffer_remaining = max(0, TECHNICAL_BUFFER - buffer_elapsed)

    total_minutes = int(total_remaining // 60)
    total_seconds = int(total_remaining % 60)

    if question_elapsed is None:
        q_minutes = speaking_remaining // 60
        q_seconds = speaking_remaining % 60
        question_label = "🎙️ Interviewer Speaking"
        question_value = f"{q_minutes:02d}:{q_seconds:02d}"
    elif question_elapsed <= ANSWER_TIME:
        q_minutes = int(answer_remaining // 60)
        q_seconds = int(answer_remaining % 60)
        question_label = "⏳ Answer Time"
        question_value = f"{q_minutes:02d}:{q_seconds:02d}"
    else:
        q_minutes = int(buffer_remaining // 60)
        q_seconds = int(buffer_remaining % 60)
        question_label = "⚙️ Processing Buffer"
        question_value = f"{q_minutes:02d}:{q_seconds:02d}"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏱️ Interview Time",
            f"{total_minutes:02d}:{total_seconds:02d}"
        )

    with col2:
        st.metric(
            question_label,
            question_value
        )

    with col3:
        st.metric(
            "❓ Question",
            max(st.session_state.question_number, 1)
        )

    if question_elapsed is not None and question_elapsed > ANSWER_TIME and buffer_remaining > 0:
        st.warning(
            "⚙️ Your 90-second answer time is over. "
            f"You have {int(buffer_remaining)} seconds of technical buffer."
        )
    elif question_elapsed is not None and question_elapsed > QUESTION_TIME:
        st.error(
            "⏰ Question time is over. Submit your current response "
            "or wait for the interviewer to continue."
        )

    if total_remaining <= LOW_TIME_BUFFER and total_remaining > 0:
        st.warning(
            f"⚠️ Less than {LOW_TIME_BUFFER // 60} minutes remaining — "
            "wrap up soon!"
        )


if not st.session_state.interview_finished:
    # Streamlit fragments rerun this section independently every second,
    # so the timer changes without requiring the user to click/submit.
    fragment = getattr(st, "fragment", None)

    if fragment is not None:
        _render_timer = fragment(run_every=1)(_render_timer)

    _render_timer()


# ============================================================
# CHAT HISTORY
# ============================================================

st.divider()

for message in st.session_state.messages:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])


# ============================================================
# FINAL REPORT (with charts)
# ============================================================

if st.session_state.interview_finished:

    st.divider()
    st.success("🎉 Interview completed!")
    st.subheader("📊 Final Interview Report")

    if st.session_state.report is None:
        with st.spinner("Generating your final feedback report..."):
            load_report()

    if st.session_state.report is not None:

        report = st.session_state.report

        correct = report.get("correct_count", 0)
        incorrect = report.get("incorrect_count", 0)
        total = report.get("total_questions", correct + incorrect)

        # ------------------------------------------------
        # TOP METRICS
        # ------------------------------------------------

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Questions", total)
        m2.metric("Correct", correct)
        m3.metric("Incorrect", incorrect)

        accuracy = (correct / total * 100) if total > 0 else 0
        st.progress(min(int(accuracy), 100), text=f"Average Performance: {accuracy:.0f}%")

        overall_seconds = report.get("overall_time_seconds", 0)
        avg_answer_seconds = report.get("average_hands_on_time_seconds", 0)
        evaluated = report.get("evaluated_answers", correct + incorrect)

        def fmt_duration(seconds):
            seconds = max(0, int(round(seconds or 0)))
            return f"{seconds // 60:02d}:{seconds % 60:02d}"

        tm1, tm2, tm3 = st.columns(3)
        tm1.metric("Total Interview Time", fmt_duration(overall_seconds))
        tm2.metric("Avg. Hands-on Answer Time", fmt_duration(avg_answer_seconds))
        tm3.metric("Evaluated Answers", evaluated)

        st.caption("Performance and timing metrics are calculated only from actual evaluated question answers. Greetings, filler/noise, and repeat requests are excluded.")

        st.divider()

        # ------------------------------------------------
        # CHARTS
        # ------------------------------------------------

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
                st.caption("No data to chart yet.")

        with chart_col2:
            st.markdown("**Strong vs Weak Topics**")
            strong_topics = report.get("strong_topics", [])
            weak_topics = report.get("weak_topics", [])

            if strong_topics or weak_topics:
                fig2, ax2 = plt.subplots()
                ax2.bar(
                    ["Strong Topics", "Weak Topics"],
                    [len(strong_topics), len(weak_topics)],
                    color=["#3498db", "#f39c12"]
                )
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
            else:
                st.caption("No topic data available.")

        st.divider()

        # ------------------------------------------------
        # SUMMARY
        # ------------------------------------------------

        if report.get("summary"):
            st.markdown("### 📝 Overall Summary")
            st.write(report["summary"])

        # ------------------------------------------------
        # STRENGTHS / WEAKNESSES LISTS
        # ------------------------------------------------

        col_s, col_w = st.columns(2)

        with col_s:
            st.markdown("### 💪 Strengths")
            if strong_topics:
                for item in strong_topics:
                    st.write(f"✅ {item}")
            else:
                st.caption("None identified.")

        with col_w:
            st.markdown("### 📌 Areas for Improvement")
            if weak_topics:
                for item in weak_topics:
                    st.write(f"⚠️ {item}")
            else:
                st.caption("None identified.")

        with st.expander("🔍 View complete report (raw JSON)"):
            st.json(report)

    else:

        st.error(
            "The interview finished, but the final "
            "report could not be loaded."
        )

        if st.session_state.report_error:
            st.code(st.session_state.report_error)

    st.stop()


# ============================================================
# ANSWER AREA
# ============================================================

st.divider()
st.subheader("Your Answer")


# ============================================================
# VOICE MODE
# ============================================================
if st.session_state.interview_mode == "🎙️ Voice Interview":

    st.write("🎙️ Record your answer using the microphone.")

    audio_value = st.audio_input(
        "Record your answer",
        key=f"audio_{st.session_state.audio_key}"
    )

    if audio_value is not None:

        if not st.session_state.is_processing:

            st.session_state.is_processing = True

            audio_bytes = audio_value.getvalue()

            # Bahut chhoti recording = likely silence/no speech
            if len(audio_bytes) < 8000:

                st.warning("Recording seems too short. Please try again.")

                st.session_state.is_processing = False
                st.session_state.audio_key += 1

                st.rerun()

            else:

                with st.spinner("🎧 Transcribing your answer..."):
                    transcript = transcribe_audio(audio_value)

                if transcript and len(transcript.strip()) > 2:

                    st.session_state.transcript = transcript
                    st.info(f"📝 You said: {transcript}")

                    with st.spinner("🤖 AI is evaluating your answer..."):
                        ai_reply = send_message(transcript)

                    if ai_reply:

                        with st.spinner("🔊 Generating AI voice..."):
                            audio = generate_speech(ai_reply)

                        if audio:
                            st.session_state.last_audio = audio

                            # Delay answer countdown for the full interviewer speech.
                            if st.session_state.new_question_pending:
                                duration = get_audio_duration(audio)
                                start_question_timer(delay=duration)
                else:
                    st.warning("Couldn't hear anything clearly. Please try again.")

                st.session_state.is_processing = False
                st.session_state.audio_key += 1

                st.rerun()


# ============================================================
# TEXT MODE
# ============================================================

else:

    text_answer = st.text_area(
        "Type your answer",
        key=f"text_{st.session_state.text_key}",
        height=150,
        placeholder="Type your answer here..."
    )

    if st.button(
        "📤 Submit Answer",
        type="primary",
        use_container_width=True
    ):

        if text_answer.strip():

            with st.spinner("🤖 AI is evaluating your answer..."):
                ai_reply = send_message(text_answer)

            if ai_reply and st.session_state.new_question_pending:
                # Text has no speech delay, so the question is already visible.
                start_question_timer()

            st.session_state.text_key += 1
            st.rerun()

        else:
            st.warning("Please enter an answer first.")


# ============================================================
# AI VOICE FEEDBACK
# ============================================================

if (
    st.session_state.interview_mode == "🎙️ Voice Interview"
    and st.session_state.last_audio
    and not st.session_state.interview_finished
):

    st.divider()
    st.subheader("🔊 AI Interviewer")

    st.audio(
        st.session_state.last_audio,
        format="audio/mpeg",
        autoplay=True
    )