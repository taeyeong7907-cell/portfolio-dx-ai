
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "AIVLE School 학습 도우미"
DEFAULT_REPO_FAISS_DIR = "faiss_index"
DEFAULT_COLAB_PATH = "/content/drive/MyDrive/project04/"
DEFAULT_PDF_NAME = "AIVLE School 백서.pdf"

BASE_PATH = os.getenv(
    "AIVLE_BASE_PATH",
    DEFAULT_COLAB_PATH if os.path.exists("/content") else "",
)


# =========================================================
# 1. API KEY 로딩
# =========================================================
def _read_key_file(filepath: str) -> None:
    if not filepath or not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and value and not os.getenv(key):
                os.environ[key] = value


def load_api_keys() -> bool:
    try:
        if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    candidates = [
        os.path.join(BASE_PATH, "api_key.txt") if BASE_PATH else "",
        "api_key.txt",
    ]

    for fp in candidates:
        if fp and os.path.exists(fp):
            _read_key_file(fp)

    return bool(os.getenv("OPENAI_API_KEY"))


HAS_OPENAI_KEY = load_api_keys()


# =========================================================
# 2. CSS 디자인
# =========================================================
st.markdown(
    """
<style>
:root {
    --main-green: #087c73;
    --soft-green: #eaf7f5;
    --line: #d7e6e3;
    --text: #163b3a;
    --muted: #6e8380;
    --bg-card: rgba(255, 255, 255, 0.94);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1760px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4fbfa 0%, #ffffff 100%);
    border-right: 1px solid var(--line);
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 8px 22px 8px;
}

.logo-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #0aa68f, #006d65);
    color: white;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.logo-title {
    font-size: 20px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.05;
}

.logo-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}

.app-title {
    font-size: 34px;
    font-weight: 850;
    color: #143736;
    letter-spacing: -0.03em;
    margin: 4px 0 2px 0;
}

.welcome {
    color: #496763;
    font-size: 15px;
    margin-bottom: 10px;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 8px 24px rgba(7, 72, 68, 0.04);
}

.card-title {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--text);
    font-weight: 850;
    font-size: 18px;
    margin-bottom: 13px;
}

.icon-circle {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0b8e82, #006f68);
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.small-muted {
    color: var(--muted);
    font-size: 13px;
}

.chat-box {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #fbfefe;
    padding: 12px 14px;
    margin: 10px 0;
}

.msg-row {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    margin: 13px 0;
}

.avatar-user, .avatar-ai {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 850;
}

.avatar-user {
    background: #00796f;
    color: white;
}

.avatar-ai {
    background: #cdeee9;
    color: #006b64;
}

.msg-content {
    flex: 1;
    border: 1px solid #e1efec;
    border-radius: 14px;
    padding: 12px 14px;
    background: white;
    color: #193b39;
    line-height: 1.55;
    font-size: 15px;
}

.msg-name {
    color: #00796f;
    font-weight: 850;
    margin-bottom: 4px;
}

mark {
    background: #fff3a3;
    color: #153c39;
    padding: 0 2px;
    border-radius: 3px;
}

.source-item {
    border-bottom: 1px solid #e4efed;
    padding: 8px 0;
    font-size: 14px;
}

.source-badge {
    display: inline-block;
    min-width: 22px;
    text-align: center;
    background: #0a8077;
    color: white;
    border-radius: 999px;
    font-size: 12px;
    padding: 2px 7px;
    margin-right: 7px;
}

.admin-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #d9e8e5;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 7px 0;
    background: #fbfefe;
    color: #244947;
    font-weight: 650;
}

.keyword {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px 4px 4px 0;
    border: 1px solid #cfe2df;
    border-radius: 999px;
    background: #f7fcfb;
    color: #0b6f68;
    font-weight: 650;
    font-size: 13px;
}

.footer-note {
    font-size: 13px;
    color: #5f7774;
    border: 1px solid #dce9e7;
    border-radius: 14px;
    background: #f8fcfb;
    padding: 12px 15px;
    margin-top: 12px;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid #c9dedb;
    color: #0b625d;
    font-weight: 750;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #087c73;
    color: #087c73;
    background: #eef9f7;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3. 유틸 함수
# =========================================================
def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def highlight_text(text: str, keyword: str) -> str:
    safe = html_escape(text)
    keyword = keyword.strip()

    if not keyword:
        return safe.replace("\n", "<br>")

    pattern = re.compile(re.escape(html_escape(keyword)), re.IGNORECASE)
    safe = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)

    return safe.replace("\n", "<br>")


def short_preview(text: str, n: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:n] + ("..." if len(text) > n else "")


def make_summary(answer: str) -> List[str]:
    if not answer:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", answer.strip())
    clean = [
        s.strip("-•* 0123456789.\t")
        for s in sentences
        if len(s.strip()) > 8
    ]

    return clean[:3]


def extract_keywords(text: str, limit: int = 8) -> List[str]:
    if not text:
        return ["AIVLE", "교육과정", "프로젝트", "취업"]

    tokens = re.findall(r"[가-힣A-Za-z0-9_\-]{2,}", text)

    stop = {
        "그리고", "하지만", "대한", "위해", "다음", "있습니다",
        "합니다", "하는", "에서", "으로", "또는", "질문",
        "답변", "내용", "경우", "방법", "사용", "관련",
    }

    freq: Dict[str, int] = {}

    for t in tokens:
        if t in stop or len(t) < 2:
            continue
        freq[t] = freq.get(t, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    return [w for w, _ in ranked[:limit]] or ["AIVLE", "교육과정", "프로젝트", "취업"]


def doc_source_name(doc: Document) -> str:
    meta = doc.metadata or {}
    source = meta.get("source") or meta.get("file_name") or meta.get("filename") or "업로드 문서"
    page = meta.get("page")
    source = os.path.basename(str(source))

    if page is not None:
        try:
            return f"{source} p.{int(page) + 1}"
        except Exception:
            return f"{source} p.{page}"

    return source


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_for_prompt(docs: List[Document]) -> str:
    lines = []

    for i, doc in enumerate(docs, 1):
        source = doc_source_name(doc)
        content = doc.page_content.strip()
        lines.append(f"[출처 {i}: {source}]\n{content}")

    return "\n\n".join(lines)


def deduplicate_docs(docs: List[Document], max_docs: int) -> List[Document]:
    seen = set()
    out = []

    for d in docs:
        key = (doc_source_name(d), short_preview(d.page_content, 120))

        if key in seen:
            continue

        seen.add(key)
        out.append(d)

        if len(out) >= max_docs:
            break

    return out


# =========================================================
# 4. 문서 로딩 및 벡터스토어
# =========================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource(show_spinner=False)
def load_base_vectorstore() -> Optional[FAISS]:
    if not HAS_OPENAI_KEY:
        return None

    embedding = get_embedding_model()

    candidates = [
        DEFAULT_REPO_FAISS_DIR,
        os.path.join(BASE_PATH, DEFAULT_REPO_FAISS_DIR) if BASE_PATH else "",
    ]

    for faiss_dir in candidates:
        if faiss_dir and os.path.exists(faiss_dir):
            try:
                return FAISS.load_local(
                    faiss_dir,
                    embedding,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                st.warning(f"FAISS 인덱스 로드 실패: {faiss_dir} / {e}")

    return None


def build_vectorstore_from_default_pdf() -> Optional[FAISS]:
    if not HAS_OPENAI_KEY:
        return None

    pdf_candidates = [
        os.path.join(BASE_PATH, DEFAULT_PDF_NAME) if BASE_PATH else "",
        DEFAULT_PDF_NAME,
    ]

    pdf_path = next(
        (p for p in pdf_candidates if p and os.path.exists(p)),
        None
    )

    if not pdf_path:
        return None

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, get_embedding_model())


def extract_docs_from_uploads(uploaded_files: List[Any]) -> List[Document]:
    docs: List[Document] = []

    for uf in uploaded_files:
        name = uf.name
        suffix = Path(name).suffix.lower()
        raw = uf.getvalue()

        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            try:
                loaded = PyMuPDFLoader(tmp_path).load()

                for d in loaded:
                    d.metadata["source"] = name

                docs.extend(loaded)

            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        elif suffix in [".txt", ".md", ".csv"]:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp949", errors="ignore")

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": name}
                )
            )

        else:
            st.warning(f"지원하지 않는 파일 형식입니다: {name}")

    return [
        d for d in docs
        if d.page_content and d.page_content.strip()
    ]


def build_vectorstore_from_docs(docs: List[Document]) -> Optional[FAISS]:
    if not docs or not HAS_OPENAI_KEY:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, get_embedding_model())


def retrieve_docs(query: str, k: int, mode: str) -> List[Document]:
    docs: List[Document] = []

    base_vs = st.session_state.get("base_vectorstore")
    uploaded_vs = st.session_state.get("uploaded_vectorstore")

    if mode in ["기본 백서", "기본+업로드"] and base_vs is not None:
        docs.extend(
            base_vs.as_retriever(search_kwargs={"k": k}).invoke(query)
        )

    if mode in ["업로드 파일", "기본+업로드"] and uploaded_vs is not None:
        docs.extend(
            uploaded_vs.as_retriever(search_kwargs={"k": k}).invoke(query)
        )

    if not docs and base_vs is not None:
        docs.extend(
            base_vs.as_retriever(search_kwargs={"k": k}).invoke(query)
        )

    return deduplicate_docs(docs, max_docs=k * 2)


# =========================================================
# 5. 답변 생성 기능
# =========================================================
def answer_with_rag(
    question: str,
    docs: List[Document],
    style: str,
    include_examples: bool
) -> str:

    if not HAS_OPENAI_KEY:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다."

    if not docs:
        return "검색된 문서 근거가 없습니다. 관리자에게 문서 업로드를 요청해 주세요."

    style_guide = {
        "간단 요약": "핵심만 3~5문장으로 짧게 답합니다.",
        "상세 설명": "개념, 이유, 단계, 주의점을 구분해서 자세히 답합니다.",
        "예시 포함": "설명 뒤에 쉬운 예시와 적용 예를 포함합니다.",
    }.get(style, "핵심만 명확히 답합니다.")

    example_guide = (
        "가능하면 짧은 예시를 포함하세요."
        if include_examples
        else "예시가 꼭 필요하지 않으면 생략하세요."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 AIVLE School 학습 도우미입니다.

아래 [문서 근거]만 우선적으로 활용해 답변하세요.
문서에 없는 내용은 단정하지 말고,
필요한 경우 '문서 근거상 확인되지 않습니다'라고 말하세요.

답변은 한국어로 작성하고,
학생이 바로 이해할 수 있도록 친절하고 명확하게 설명하세요.

[답변 스타일]
{style_guide}

{example_guide}

[문서 근거]
{context}
"""),
        ("human", "질문: {question}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": format_docs_for_prompt(docs),
        "question": question,
        "style_guide": style_guide,
        "example_guide": example_guide,
    })


# =========================================================
# 6. 추천 질문 생성 기능
# =========================================================
def generate_recommended_questions(search_mode: str) -> str:
    if not HAS_OPENAI_KEY:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다."

    docs = retrieve_docs(
        "AIVLE School 핵심 내용 교육 과정 프로젝트 역량 지원 진로 주요 개념",
        k=5,
        mode=search_mode
    )

    if not docs:
        return "추천 질문을 만들 문서 근거가 없습니다. 관리자에게 문서 업로드를 요청해 주세요."

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
문서 내용을 바탕으로 사용자가 자주 물어볼 만한 추천 질문 5개를 생성하세요.

조건:
1. 문서 내용과 관련된 질문만 만드세요.
2. 각 질문은 한 줄씩 작성하세요.
3. 질문 앞에 번호를 붙이세요.
4. 너무 어렵지 않고 실제 사용자가 물어볼 만한 질문으로 만드세요.
5. 질문만 출력하세요.
"""),
        ("human", """
문서 내용:
{context}
""")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context
    })


# =========================================================
# 7. 이어지는 질문 / 꼬리질문 생성 기능
# =========================================================
def generate_followup_questions(question: str, answer: str) -> List[str]:
    fallback = [
        "이 내용을 더 쉽게 예시로 설명해줘.",
        "핵심만 표로 정리해줘.",
        "실제 상황에서는 어떻게 활용할 수 있어?",
    ]

    if not HAS_OPENAI_KEY or not answer:
        return fallback

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
사용자의 질문과 답변을 보고 이어서 물어보면 좋은 후속 질문 3개를 생성하세요.

조건:
1. 업로드된 문서 또는 기존 답변 흐름과 관련된 질문이어야 합니다.
2. 각 질문은 한 줄씩 작성하세요.
3. 너무 길지 않게 작성하세요.
4. 번호 없이 질문만 출력하세요.
"""),
        ("human", """
사용자 질문:
{question}

AI 답변:
{answer}
""")
    ])

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

        chain = prompt | llm | StrOutputParser()

        text = chain.invoke({
            "question": question,
            "answer": answer
        })

        items = [
            re.sub(r"^[-*\d.\s]+", "", x).strip()
            for x in text.splitlines()
            if x.strip()
        ]

        return items[:3] or fallback

    except Exception:
        return fallback


# =========================================================
# 8. 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_summary" not in st.session_state:
    st.session_state.last_summary = []

if "followups" not in st.session_state:
    st.session_state.followups = []

if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = None

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "base_vectorstore" not in st.session_state:
    with st.spinner("기본 지식베이스를 불러오는 중입니다..."):
        base = load_base_vectorstore()

        if base is None:
            base = build_vectorstore_from_default_pdf()

        st.session_state.base_vectorstore = base


# =========================================================
# 9. 사이드바
# =========================================================
with st.sidebar:
    st.markdown(
        """
<div class="sidebar-logo">
  <div class="logo-mark">A</div>
  <div>
    <div class="logo-title">AIVLE School</div>
    <div class="logo-sub">학습 도우미</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.header("🔗 바로가기")

    st.link_button(
        "📘 KT AIVLE SCHOOL 공식 사이트",
        "https://aivle.kt.co.kr"
    )

    st.link_button(
        "❓ FAQ 바로가기",
        "https://aivle.kt.co.kr/home/brd/faq/main?mcd=MC00000056"
    )

    st.link_button(
        "📢 모집 안내",
        "https://aivle.kt.co.kr/home/main/applyMain?mcd=MC00000051"
    )

    st.link_button(
        "🎥 AIVLE 소개",
        "https://www.youtube.com/watch?v=HEawaDxw3WM&list=PL_WCuvyChN3giYr6Xa2gWMOvRjDUrfb0a&index=2"
    )

    st.markdown("---")

    st.info("AIVLE과 함께 성장하는 나의 미래\n\n오늘도 화이팅! 💪")


# =========================================================
# 10. 본문 레이아웃
# =========================================================
main_col, right_col = st.columns([3.2, 1.15], gap="large")


# =========================================================
# 11. 메인 영역
# =========================================================
with main_col:
    st.markdown(
        f"<div class='app-title'>{APP_TITLE}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='welcome'>👋 환영합니다. 무엇을 도와드릴까요?</div>",
        unsafe_allow_html=True
    )

    # 답변 설정
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    top1, top2 = st.columns([1.05, 1.0], gap="large")

    with top1:
        st.markdown(
            "<div class='card-title'><span class='icon-circle'>⚙</span>답변 스타일 설정</div>",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            style = st.selectbox(
                "답변 길이",
                ["간단 요약", "상세 설명", "예시 포함"],
                label_visibility="collapsed"
            )

        with c2:
            search_mode = st.selectbox(
                "검색 대상",
                ["기본 백서", "업로드 파일", "기본+업로드"],
                label_visibility="collapsed"
            )

        with c3:
            include_examples = st.toggle(
                "예시 포함",
                value=True
            )

    with top2:
        st.markdown(
            "<div class='card-title'><span class='icon-circle'>⌕</span>답변 내 검색</div>",
            unsafe_allow_html=True
        )

        answer_keyword = st.text_input(
            "답변 내에서 검색",
            placeholder="답변 내에서 검색...",
            label_visibility="collapsed"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 꼬리질문 버튼을 눌렀을 때 질문창에 반영
    if st.session_state.pending_question:
        st.session_state.question_input = st.session_state.pending_question
        st.session_state.pending_question = None

    # 질문 입력
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>?</span>질문하기</div>",
        unsafe_allow_html=True
    )

    q_col, btn_col = st.columns([5, 0.7])

    with q_col:
        user_question = st.text_input(
            "질문",
            placeholder="궁금한 내용을 입력하세요...",
            label_visibility="collapsed",
            key="question_input"
        )

    with btn_col:
        ask_clicked = st.button(
            "🔍 검색",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 질문 처리
    if ask_clicked and user_question.strip():
        with st.spinner("문서를 검색하고 답변을 생성하는 중입니다..."):
            docs = retrieve_docs(
                user_question,
                k=4,
                mode=search_mode
            )

            answer = answer_with_rag(
                user_question,
                docs,
                style,
                include_examples
            )

            summary = make_summary(answer)

            sources = [
                {
                    "name": doc_source_name(d),
                    "preview": short_preview(d.page_content, 260),
                }
                for d in docs
            ]

            followups = generate_followup_questions(
                user_question,
                answer
            )

            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.last_summary = summary
            st.session_state.last_sources = sources
            st.session_state.followups = followups

            st.rerun()

    # 대화 내용
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>…</span>대화 내용</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            "<div class='small-muted'>아직 대화가 없습니다. 위 질문하기 영역에 질문을 입력해 주세요.</div>",
            unsafe_allow_html=True
        )

    else:
        for msg in st.session_state.messages[-10:]:
            if msg["role"] == "user":
                st.markdown(
                    f"""
<div class="msg-row">
  <div class="avatar-user">나</div>
  <div class="msg-content">{highlight_text(msg['content'], answer_keyword)}</div>
</div>
""",
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
<div class="msg-row">
  <div class="avatar-ai">🤖</div>
  <div class="msg-content">
    <div class="msg-name">AIVLE 도우미</div>
    {highlight_text(msg['content'], answer_keyword)}
  </div>
</div>
""",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 핵심 요약 / 출처
    s_col, src_col = st.columns([1.05, 1.35], gap="large")

    with s_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'><span class='icon-circle'>★</span>핵심 요약</div>",
            unsafe_allow_html=True
        )

        if st.session_state.last_summary:
            for item in st.session_state.last_summary:
                st.markdown(
                    f"✅ {html_escape(item)}",
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                "<div class='small-muted'>답변이 생성되면 핵심 요약이 표시됩니다.</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with src_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'><span class='icon-circle'>“</span>출처/인용</div>",
            unsafe_allow_html=True
        )

        if st.session_state.last_sources:
            for i, src in enumerate(st.session_state.last_sources, 1):
                st.markdown(
                    f"""
<div class="source-item">
  <span class="source-badge">{i}</span>
  <b>{html_escape(src['name'])}</b><br>
  <span class="small-muted">{html_escape(src['preview'])}</span>
</div>
""",
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                "<div class='small-muted'>검색된 문서 출처가 여기에 표시됩니다.</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer-note'>ⓘ AIVLE 도우미의 답변은 학습 참고용으로 제공되며, 중요한 내용은 원문 자료와 함께 확인해 주세요.</div>",
        unsafe_allow_html=True
    )


# =========================================================
# 12. 오른쪽 영역: 관리자 / 업로드 / 추천 질문 / 꼬리질문
# =========================================================
with right_col:
    # 관리자 로그인
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>🔐</span>관리자 페이지</div>",
        unsafe_allow_html=True
    )

    admin_password = st.text_input(
        "관리자 비밀번호",
        type="password"
    )

    if st.button("관리자 로그인", use_container_width=True):
        if admin_password == "aivle123":
            st.session_state.admin_mode = True
            st.success("관리자 로그인 성공!")

        else:
            st.error("비밀번호가 틀렸습니다.")

    if st.session_state.admin_mode:
        st.success("관리자 모드 활성화")
    else:
        st.info("관리자만 문서 업로드 가능합니다.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 관리자만 업로드 가능
    if st.session_state.admin_mode:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'><span class='icon-circle'>📂</span>문서 업로드</div>",
            unsafe_allow_html=True
        )

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("업로드 문서 분석", use_container_width=True):
            if not uploads:
                st.warning("먼저 PDF 파일을 업로드해 주세요.")

            elif not HAS_OPENAI_KEY:
                st.error("OPENAI_API_KEY가 필요합니다.")

            else:
                with st.spinner("문서 분석 중..."):
                    docs = extract_docs_from_uploads(uploads)
                    vs = build_vectorstore_from_docs(docs)

                    st.session_state.uploaded_vectorstore = vs

                st.success("문서 업로드 완료!")

        st.markdown("</div>", unsafe_allow_html=True)

    # 추천 질문
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>💡</span>추천 질문</div>",
        unsafe_allow_html=True
    )

    if st.button("추천 질문 생성", use_container_width=True):
        with st.spinner("추천 질문을 생성하고 있습니다..."):
            st.session_state.recommended_questions = generate_recommended_questions(
                search_mode
            )

    if st.session_state.recommended_questions:
        st.markdown(st.session_state.recommended_questions)

    else:
        st.markdown(
            "<div class='small-muted'>문서 기반 추천 질문을 생성할 수 있습니다.</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 주요 키워드
    latest_answer = ""

    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant":
            latest_answer = m.get("content", "")
            break

    keywords = extract_keywords(latest_answer)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>◆</span>주요 키워드</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "".join([
            f"<span class='keyword'>{html_escape(k)}</span>"
            for k in keywords
        ]),
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # 이어지는 질문 / 꼬리질문
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>👉</span>이어지는 질문</div>",
        unsafe_allow_html=True
    )

    if st.session_state.followups:
        for fq in st.session_state.followups:
            if st.button(fq, use_container_width=True):
                st.session_state.pending_question = fq
                st.rerun()

    else:
        default_followups = [
            "AIVLE School의 교육 과정은 어떻게 구성되어 있나요?",
            "프로젝트는 어떤 방식으로 진행되나요?",
            "수료 후 취업 지원은 어떻게 이루어지나요?",
        ]

        for fq in default_followups:
            st.markdown(
                f"<div class='admin-row'><span>{html_escape(fq)}</span><span>›</span></div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # 상태 표시
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'><span class='icon-circle'>i</span>상태</div>",
        unsafe_allow_html=True
    )

    base_ready = st.session_state.get("base_vectorstore") is not None
    upload_ready = st.session_state.get("uploaded_vectorstore") is not None

    st.write(f"OpenAI Key: {'✅' if HAS_OPENAI_KEY else '❌'}")
    st.write(f"기본 FAISS: {'✅' if base_ready else '❌'}")
    st.write(f"업로드 KB: {'✅' if upload_ready else '❌'}")

    st.markdown("</div>", unsafe_allow_html=True)
