
import os
import io
import json
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

import streamlit as st
from openai import OpenAI
#pdf파일 가능하도록 추가 코드 작성
try:
    import pdfplumber
except ImportError:
    os.system('pip install pdfplumber')
    import pdfplumber

# =========================
# 1. 기본 설정
# =========================

st.set_page_config(
    page_title="김 대리 AI 개인비서 에이블",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 김 대리 AI 개인비서 에이블")
st.caption("음성 인식, 업무 답변, 이메일 초안 작성, 음성 답변 재생까지 지원하는 AI 개인비서 에이블 입니다.")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================
# 2. 역할 프롬프트 정의
# =========================

ROLE_PROMPTS = {
    "AI 개인비서": """
당신은 사용자의 업무를 도와주는 전문 AI 개인비서 에이블 입니다.

역할:
- 사용자의 질문을 이해하고 친절하게 답변합니다.
- 일정, 이메일, 회의록, 업무 정리, 아이디어 도출을 도와줍니다.
- 답변은 너무 길지 않게 핵심부터 말합니다.
- 사용자가 초보자일 수 있으므로 어려운 표현은 쉽게 풀어 설명합니다.
- 확실하지 않은 내용은 추측하지 말고 모른다고 말합니다.

응답 방식:
- 먼저 결론을 말합니다.
- 필요하면 단계별로 정리합니다.
- 사용자가 바로 행동할 수 있도록 구체적으로 제안합니다.
""",

    "김 대리 개인비서 에이블": """
당신은 회사 내 '김 대리'의 AI 개인비서 에이블입니다.

상황:
- 김 대리는 회의, 이메일, 일정, 업무 정리를 자주 처리합니다.
- 김 대리는 부재 중일 수 있으며, 그 경우 대신 내용을 정리하고 전달할 수 있어야 합니다.
- 사용자의 업무 맥락을 고려해 실무적인 답변을 제공합니다.
- 너무 가볍지 않고, 전문적인 톤을 유지합니다.
- 확실하지 않은 정보는 임의로 만들지 않습니다.

응답 방식:
- 보고서/메일/회의록에 바로 쓸 수 있는 표현을 우선 사용합니다.
- 필요한 경우 “확인 필요” 항목을 따로 표시합니다.
"""
}


# =========================
# 3. 세션 상태 초기화
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "email_draft" not in st.session_state:
    st.session_state.email_draft = {} # Changed from None to {} for consistency

if "reference_text" not in st.session_state:
    st.session_state.reference_text = ""

if "meeting_summary" not in st.session_state:
    st.session_state.meeting_summary = ""

if "kim_tasks" not in st.session_state:
    st.session_state.kim_tasks = ""

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""


# =========================
# 4. 유틸 함수
# =========================

def keep_recent_messages(limit=10):
    st.session_state.messages = st.session_state.messages[-limit:]


def transcribe_audio(audio_file):
    audio_bytes = audio_file.getvalue()

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", io.BytesIO(audio_bytes), "audio/wav")
    )

    return transcript.text


def generate_answer(role_name, user_text, reference_text=""):
    system_content = ROLE_PROMPTS[role_name]

    if reference_text:
        system_content += f"""

아래 참고 자료가 있다면 답변에 반영하세요.
단, 참고 자료에 없는 내용은 임의로 만들지 마세요.

[참고 자료]
{reference_text}
"""

    input_messages = [
        {"role": "system", "content": system_content}
    ]

    input_messages.extend(st.session_state.messages)

    input_messages.append(
        {"role": "user", "content": user_text}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=input_messages
    )

    return response.choices[0].message.content


def text_to_speech(text):
    tts_model = os.getenv("TTS_MODEL", "tts-1")
    tts_voice = os.getenv("TTS_VOICE", "alloy")

    speech = client.audio.speech.create(
        model=tts_model,
        voice=tts_voice,
        input=text
    )

    return speech.content

# ✅ 번역기 기능 추가
def translate_text(text, target_language):
    system_prompt = """
당신은 전문 번역가입니다.

사용자가 입력한 문장을 자연스럽고 정확하게 번역하세요.
의미를 바꾸지 말고 문맥에 맞게 번역하세요.
불필요한 설명 없이 번역 결과만 출력하세요.
"""

    prompt = f"""
다음 문장을 {target_language}로 번역하세요.

TEXT:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def make_email_draft_from_text(user_text):
    system_prompt = """
당신은 이메일 발송 정보를 추출하는 도우미입니다.

사용자 요청에서 이메일 발송에 필요한 정보를 추출하세요.

반드시 아래 JSON 형식으로만 답하세요.

{
  "is_email_request": true 또는 false,
  "to": "받는 사람 이메일 주소. 없으면 빈 문자열",
  "subject": "메일 제목. 없으면 적절히 작성",
  "body": "메일 본문. 없으면 적절히 작성",
  "missing_info": ["부족한 정보 목록"]
}

주의:
- 이메일 요청이 아니면 is_email_request는 false입니다.
- 받는 사람 이메일 주소가 없으면 절대 임의로 만들지 마세요.
- 사용자가 말하지 않은 사실을 지어내지 마세요.
- JSON 외의 설명 문장은 출력하지 마세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "is_email_request": False,
            "to": "",
            "subject": "",
            "body": "",
            "missing_info": ["이메일 정보를 JSON으로 해석하지 못했습니다."]
        }

    return data


def send_email_sendgrid(to_email, subject, body):
    """
    SendGrid API를 사용해 실제 이메일을 발송하는 함수
    """

    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("EMAIL_ADDRESS")

    if not sendgrid_api_key:
        raise ValueError("SENDGRID_API_KEY가 설정되지 않았습니다.")

    if not from_email:
        raise ValueError("EMAIL_ADDRESS가 설정되지 않았습니다.")

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    sg = SendGridAPIClient(sendgrid_api_key)
    response = sg.send(message)

    return response.status_code

# =========================
# 4-1. 참고 자료 프롬프트 보조 함수
# =========================

def build_reference_block(reference_text):
    if not reference_text.strip():
        return ""

    return f"""
아래 참고 자료가 있다면 답변에 반영하세요.
단, 참고 자료에 없는 내용은 임의로 만들지 마세요.

[참고 자료]
{reference_text}
"""

# =========================
# 4-2. 회의록 요약 함수
# =========================

def summarize_meeting(role_name, meeting_text, reference_text=""):
    system_prompt = f"""
{ROLE_PROMPTS[role_name]}

당신은 회의록 정리 전문가입니다.
주어진 내용을 바탕으로 회의록을 아래 형식으로 정리하세요.

형식:
1. 회의 개요
2. 핵심 논의 내용
3. 결정 사항
4. 후속 액션 아이템
5. 확인 필요 사항

조건:
- 불필요한 수식어 없이 깔끔하게 작성합니다.
- 실행해야 할 항목은 구체적으로 씁니다.
- 회의 내용에 없는 사실은 만들지 않습니다.
""" + build_reference_block(reference_text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": meeting_text}
        ]
    )
    return response.choices[0].message.content

# =========================
# 4-3. 김 대리 담당업무 추출 함수
# =========================

def extract_kim_tasks(role_name, meeting_text, reference_text=""):
    system_prompt = f"""
{ROLE_PROMPTS[role_name]}

당신은 회의 내용에서 '김 대리'의 담당업무를 추출하는 실무 비서입니다.

아래 형식으로만 정리하세요.

- 담당 업무
- 목적
- 우선순위
- 기한
- 협업 대상
- 확인 필요 사항

조건:
- 김 대리가 해야 할 일만 추립니다.
- 기한이 명확하지 않으면 '미정'으로 적습니다.
- 협업 대상이 없으면 '없음'으로 적습니다.
- 불확실한 내용은 '확인 필요'로 분리합니다.
- 회의 내용에 없는 사실은 추가하지 않습니다.
""" + build_reference_block(reference_text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": meeting_text}
        ]
    )
    return response.choices[0].message.content


# =========================
# 4-4. 메일 초안 생성 함수
# =========================

def generate_email_draft(role_name, request_text, meeting_summary="", kim_tasks="", reference_text=""):
    system_prompt = f"""
{ROLE_PROMPTS[role_name]}

당신은 회사 이메일 초안 작성 도우미입니다.
사용자의 요청과 회의 요약, 담당업무 내용을 참고하여 메일 초안을 작성하세요.

반드시 아래 JSON 형식으로만 답하세요.

{{
  "to": "",
  "subject": "",
  "body": ""
}}

조건:
- 받는 사람 주소를 모르면 빈 문자열로 둡니다.
- 제목은 실무적으로 간결하게 작성합니다.
- 본문은 바로 보낼 수 있는 자연스러운 비즈니스 한국어로 작성합니다.
- 회의 요약이나 김 대리 업무 내용이 있으면 반영합니다.
- JSON 외 문장은 출력하지 마세요.
"""

    user_payload = f"""
[사용자 요청]
{request_text}

[회의 요약]
{meeting_summary}

[김 대리 담당업무]
{kim_tasks}

[참고 자료]
{reference_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ]
    )

    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "to": "",
            "subject": "회의 내용 관련 안내",
            "body": raw
        }





# =========================
# 5. 사이드바 설정
# =========================

with st.sidebar:
    st.header("⚙️ 비서 설정")

    role_name = st.radio(
        "비서 역할 선택",
        ["AI 개인비서", "김 대리 개인비서 에이블"]
    )

    use_tts = st.checkbox("AI 답변을 음성으로 재생", value=False)
    use_email_tool = st.checkbox("이메일 발송 도구 사용", value=False)

      # ✅ 번역기 기능 추가
    use_translate_tool = st.checkbox("번역기 기능 사용", value=False)

    st.divider()

    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.session_state.transcript_text = ""
        st.session_state.last_answer = ""
        st.session_state.email_draft = {} # Changed from None to {} for consistency
        st.session_state.reference_text = ""
        st.session_state.meeting_summary = ""
        st.session_state.kim_tasks = ""
        # ✅ 번역기 기능 추가
        st.session_state.translated_text = ""

        st.success("대화 기록이 초기화되었습니다.")

    st.divider()
    st.write("현재 저장된 대화 수:", len(st.session_state.messages))


# =========================
# 6. 입력 영역
# =========================

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("🎙️ 음성 입력")

    with st.form("task_form"):
# pdf 파일 추가 코드 - 허용 확장자에 pdf 추가
        uploaded_file = st.file_uploader(
            "참고할 파일 업로드 (.txt, .pdf)",
            type=["txt", "pdf"]
        )

        audio_value = st.audio_input("무엇을 도와드릴까요?")

        submit = st.form_submit_button("음성 인식 실행")


with right_col:
    st.subheader("📌 사용 방법")
    st.info(
        """
1. 왼쪽 사이드바에서 비서 역할을 선택합니다.
2. 참고할 txt,pdf 파일이 있으면 업로드합니다.
3. 음성으로 질문합니다.
4. 음성 인식 결과를 확인하고 수정합니다.
5. AI 답변 또는 이메일 초안을 생성합니다.
"""
    )

    st.warning(
        """
이메일은 자동 발송되지 않습니다.
반드시 초안을 확인한 뒤, 사용자가 직접 발송 버튼을 눌러야만 전송됩니다.
"""
    )


# =========================
# 6-1. 직접 텍스트 입력 추가
# =========================

manual_text = st.text_area(
    "또는 직접 텍스트 입력",
    height=140,
    placeholder="예: 오늘 회의 내용을 정리해줘. 그리고 김 대리 담당업무를 따로 뽑아줘."
)





# =========================
# 7. 음성 인식 실행
# =========================

if submit:
    if uploaded_file is not None:

#pdf파일인경우
        if uploaded_file.name.endswith(".pdf"):
            try:
                extracted_text = ""
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                st.session_state.reference_text = extracted_text
            except Exception as e:
                st.error("PDF 파일을 읽는 중 오류가 발생했습니다.")
                st.stop()

        # TXT 파일인 경우
        elif uploaded_file.name.endswith(".txt"):
            try:
                st.session_state.reference_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                st.error("텍스트 파일 인코딩을 읽을 수 없습니다. UTF-8 형식을 사용해주세요.")
                st.stop()
    else:
        st.session_state.reference_text = ""

    if audio_value is None:
        st.warning("음성을 먼저 녹음해주세요.")
        st.stop()

    try:
        with st.spinner("음성을 텍스트로 변환하는 중입니다..."):
            transcript_text = transcribe_audio(audio_value)
            st.session_state.transcript_text = transcript_text

        st.success("음성 인식이 완료되었습니다.")

    except Exception as e:
        st.error("음성 인식 중 오류가 발생했습니다.")
        st.code(str(e))


# =========================
# 7-1. 입력 텍스트 통합 처리
# =========================

combined_text = ""

if audio_value is not None:
    try:
        with st.spinner("음성을 텍스트로 변환하는 중입니다..."):
            transcript_text = transcribe_audio(audio_value)
        combined_text += transcript_text.strip()
    except Exception as e:
        st.error("음성 인식 중 오류가 발생했습니다.")
        st.code(str(e))

if manual_text.strip():
    if combined_text:
        combined_text += "\n\n"
    combined_text += manual_text.strip()

st.session_state.transcript_text = combined_text

if combined_text.strip():
    st.success("입력이 반영되었습니다.")
else:
    st.warning("음성 또는 텍스트를 입력해주세요.")



# =========================
# 8. STT 결과 확인 및 수정
# =========================

if st.session_state.transcript_text:
    st.subheader("📝 음성 인식 결과 확인")

    edited_user_text = st.text_area(
        "음성 인식 결과가 맞는지 확인하고, 틀린 부분이 있으면 수정하세요.",
        value=st.session_state.transcript_text,
        height=120
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        run_llm = st.button("이 내용으로 AI 답변 생성")

    with col2:
        check_email = st.button("이 내용으로 이메일 요청 분석")

    if run_llm:
        try:
            with st.spinner("로딩중..."):
                answer = generate_answer(
                    role_name=role_name,
                    user_text=edited_user_text,
                    reference_text=st.session_state.reference_text
                )

            st.session_state.last_answer = answer

            st.session_state.messages.append(
                {"role": "user", "content": edited_user_text}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            keep_recent_messages(limit=10)

        except Exception as e:
            st.error("AI 답변 생성 중 오류가 발생했습니다.")
            st.code(str(e))

    if check_email:
        if not use_email_tool:
            st.warning("사이드바에서 이메일 발송 도구 사용을 켜주세요.")
        else:
            try:
                with st.spinner("이메일 요청인지 분석하는 중입니다..."):
                    email_draft = make_email_draft_from_text(edited_user_text)
                    st.session_state.email_draft = email_draft

            except Exception as e:
                st.error("이메일 요청 분석 중 오류가 발생했습니다.")
                st.code(str(e))


# =========================
# 8-1. 기능별 실행 버튼 확장
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    run_summary = st.button("회의록 요약")
with col2:
    run_tasks = st.button("김 대리 업무 추출")
with col3:
    run_answer = st.button("AI 답변 생성")
with col4:
    run_email = st.button("메일 초안 생성")


# =========================
# 8-2. 회의록 요약 실행
# =========================

if run_summary:
    try:
        with st.spinner("회의록을 요약하는 중입니다..."):
            summary = summarize_meeting(
                role_name=role_name,
                meeting_text=edited_user_text,
                reference_text=st.session_state.reference_text
            )
        st.session_state.meeting_summary = summary
    except Exception as e:
        st.error("회의록 요약 중 오류가 발생했습니다.")
        st.code(str(e))

# =========================
# 8-3. 김 대리 담당업무 추출 실행
# =========================

if run_tasks:
    try:
        with st.spinner("김 대리 담당업무를 추출하는 중입니다..."):
            tasks = extract_kim_tasks(
                role_name=role_name,
                meeting_text=edited_user_text,
                reference_text=st.session_state.reference_text
            )
        st.session_state.kim_tasks = tasks
    except Exception as e:
        st.error("담당업무 추출 중 오류가 발생했습니다.")
        st.code(str(e))


# =========================
# 8-4. 메일 초안 생성 실행
# =========================

if run_email:
    try:
        with st.spinner("메일 초안을 생성하는 중입니다..."):
            email_draft = generate_email_draft(
                role_name=role_name,
                request_text=edited_user_text,
                meeting_summary=st.session_state.meeting_summary,
                kim_tasks=st.session_state.kim_tasks,
                reference_text=st.session_state.reference_text
            )
        st.session_state.email_draft = email_draft
    except Exception as e:
        st.error("메일 초안 생성 중 오류가 발생했습니다.")
        st.code(str(e))





# =========================
# 9. AI 답변 출력 + TTS
# =========================

if st.session_state.last_answer:
    st.subheader("💬 AI 답변")
    st.write(st.session_state.last_answer)

    if use_tts:
        try:
            with st.spinner("AI 답변을 음성으로 변환하는 중입니다..."):
                audio_bytes = text_to_speech(st.session_state.last_answer)

            st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.warning("음성 재생 기능에서 오류가 발생했습니다.")
            st.code(str(e))


# =========================
# 9-1. 회의록 요약 결과 출력
# =========================

if st.session_state["meeting_summary"]:
    st.subheader("회의록 요약")
    st.write(st.session_state["meeting_summary"])


# =========================
# 9-2. 김 대리 담당업무 결과 출력
# =========================

if st.session_state.kim_tasks:
    st.subheader("✅ 김 대리 담당업무")
    st.write(st.session_state.kim_tasks)




# =========================
# 10. 이메일 초안 확인 및 발송
# =========================

if st.session_state.email_draft:
    draft = st.session_state.email_draft

    st.subheader("📧 이메일 도구")

    if not draft.get("is_email_request"):
        st.info("현재 입력은 이메일 발송 요청으로 판단되지 않았습니다.")
    else:
        missing_info = draft.get("missing_info", [])

        to_email = st.text_input("받는 사람 이메일", value=draft.get("to", ""))
        subject = st.text_input("제목", value=draft.get("subject", ""))
        body = st.text_area("본문", value=draft.get("body", ""), height=220)

        if missing_info:
            st.warning(f"확인 필요한 정보: {', '.join(missing_info)}")

        st.error("안전을 위해 이메일은 자동 발송하지 않습니다. 아래 버튼을 눌러야만 발송됩니다.")

        if st.button("이메일 실제 발송"):
            if not to_email:
                st.warning("받는 사람 이메일 주소가 필요합니다.")
            elif not subject:
                st.warning("메일 제목이 필요합니다.")
            elif not body:
                st.warning("메일 본문이 필요합니다.")
            else:
                try:
                    # SendGrid를 사용해 실제 이메일 발송
                    status_code = send_email_sendgrid(
                        to_email=to_email,
                        subject=subject,
                        body=body
                    )

                    if 200 <= status_code < 300:
                        st.success("이메일을 발송했습니다.")
                    else:
                        st.warning(f"이메일 요청은 전송되었지만 상태 코드를 확인하세요: {status_code}")

                except Exception as e:
                    st.error("이메일 발송 중 오류가 발생했습니다.")
                    st.code(str(e))


# =========================
# 10-1. 초기화 항목 추가
# =========================

import streamlit as st

# =========================
# Session State 초기화
# =========================
if "meeting_summary" not in st.session_state:
    st.session_state["meeting_summary"] = ""

if "meeting_text" not in st.session_state:
    st.session_state["meeting_text"] = ""

if "email_draft" not in st.session_state:
    st.session_state["email_draft"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# =========================
# 11. 최근 대화 기록
# =========================

st.divider()
st.subheader("🗂️ 최근 대화 기록")

if not st.session_state.messages:
    st.caption("아직 저장된 대화가 없습니다.")
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])
