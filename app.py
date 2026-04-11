import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import streamlit as st

os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "torch"

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
FAISS_CACHE_DIR = BASE_DIR / ".cache" / "faiss_index"
FAISS_META_PATH = BASE_DIR / ".cache" / "faiss_meta.json"
BACKGROUND_VIDEO_PATH = Path(r"C:\FlutterFinal\Personal_AI\data\mudassar.mp4")

PROFILE_IMAGE_CANDIDATES = (
    "mudassar.jpg",
    "mudassar.jpeg",
    "mudassar.png",
    "profile.jpg",
    "profile.jpeg",
    "profile.png",
)

FALLBACK_TRIGGER_PHRASES = (
    "i don't have any information",
    "i do not have any information",
    "the provided context doesn't mention",
    "doesn't provide any",
    "does not provide any",
    "no specific information",
)

GREETING_INPUTS = {
    "hi",
    "hello",
    "hey",
    "salam",
    "assalamualaikum",
    "assalamu alaikum",
}

QA_PROMPT_TEMPLATE = """
You are Mudassar's AI assistant.
Use only the provided context to answer the question.
If the answer is missing in context, say: "I don't have that in the available profile data yet."
If the user asks for image/photo/picture/profile image, return exactly: SHOW_IMAGE

Formatting requirements:
- Keep tone professional, clear, and concise.
- Use Markdown with short paragraphs and bullet points when listing items.
- Avoid long intros like "As Mudassar's AI assistant...".
- Use **bold** only for important keywords.
- Mention concrete project/skill names from context when relevant.

Context:
{context}

Question: {question}

Answer:
""".strip()


def load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_text_files(data_path: Path) -> list[Path]:
    return sorted(
        [file_path for file_path in data_path.iterdir() if file_path.is_file() and file_path.suffix.lower() == ".txt"],
        key=lambda p: p.name.lower(),
    )


def find_profile_image_path(data_path: Path) -> Path | None:
    for candidate in PROFILE_IMAGE_CANDIDATES:
        candidate_path = data_path / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def normalize_llm_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content

    return str(value)


def has_profile_image(profile_image_path: Path | None, profile_image_url: str) -> bool:
    return bool(profile_image_path) or bool(
        profile_image_url and profile_image_url.lower() not in {"undefined", "none", "null"}
    )


def should_show_profile_image(question: str, answer: str) -> bool:
    normalized_question = question.strip().lower()
    normalized_answer = answer.strip().lower()

    image_keywords = ("image", "photo", "picture", "pic", "show me", "profile")
    return "show_image" in normalized_answer or any(keyword in normalized_question for keyword in image_keywords)


def should_use_full_context_fallback(answer: str) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return True
    return any(phrase in normalized for phrase in FALLBACK_TRIGGER_PHRASES)


def clean_assistant_response(question: str, response: str) -> str:
    cleaned = response.strip()
    if not cleaned:
        return cleaned

    if question.strip().lower() in GREETING_INPUTS:
        return "Hi! Ask me anything about Muhammad Mudassar."

    generic_intro_patterns = (
        r"^hello[^\n]*assistant[^\n]*[.!?]\s*",
        r"^hi[^\n]*assistant[^\n]*[.!?]\s*",
        r"^assalamu[^\n]*assistant[^\n]*[.!?]\s*",
        r"^as[^\n]*assistant[^\n]*[:\-]\s*",
    )

    for pattern in generic_intro_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return normalize_answer_layout(cleaned) or response.strip()


def normalize_answer_layout(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    normalized = re.sub(r"(?<!\n)(\d+\.\s+)", r"\n\1", normalized)
    normalized = re.sub(r"(?<!\n)([-*]\s+)", r"\n\1", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def format_inline_assistant_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    return escaped


def assistant_markdown_to_html(text: str) -> str:
    normalized = normalize_answer_layout(text)
    if not normalized:
        return ""

    lines = normalized.split("\n")
    blocks: list[str] = []
    paragraph_parts: list[str] = []
    list_kind: str | None = None
    list_items: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_parts
        if not paragraph_parts:
            return

        paragraph_text = " ".join(part.strip() for part in paragraph_parts if part.strip())
        if paragraph_text:
            blocks.append(f"<p>{format_inline_assistant_html(paragraph_text)}</p>")
        paragraph_parts = []

    def flush_list() -> None:
        nonlocal list_kind, list_items
        if list_kind and list_items:
            tag = "ol" if list_kind == "ordered" else "ul"
            items_html = "".join(f"<li>{format_inline_assistant_html(item)}</li>" for item in list_items)
            blocks.append(f"<{tag}>{items_html}</{tag}>")
        list_kind = None
        list_items = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1)) + 2
            heading_text = format_inline_assistant_html(heading_match.group(2).strip())
            blocks.append(f"<h{level}>{heading_text}</h{level}>")
            continue

        ordered_match = re.match(r"^\d+\.\s+(.+)$", line)
        if ordered_match:
            flush_paragraph()
            if list_kind != "ordered":
                flush_list()
                list_kind = "ordered"
            list_items.append(ordered_match.group(1).strip())
            continue

        bullet_match = re.match(r"^[-*]\s+(.+)$", line)
        if bullet_match:
            flush_paragraph()
            if list_kind != "unordered":
                flush_list()
                list_kind = "unordered"
            list_items.append(bullet_match.group(1).strip())
            continue

        flush_list()
        paragraph_parts.append(line)

    flush_paragraph()
    flush_list()

    if not blocks:
        return f"<p>{format_inline_assistant_html(normalized)}</p>"

    return "".join(blocks)


def build_data_signature(text_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for file_path in text_files:
        stat = file_path.stat()
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


@st.cache_data(show_spinner=False)
def file_to_data_uri(file_path_str: str, fallback_mime: str) -> str | None:
    file_path = Path(file_path_str)
    if not file_path.exists():
        return None

    mime_type = mimetypes.guess_type(file_path.name)[0] or fallback_mime
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def inject_modern_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --chat-font-size: 15px;
            --chat-line-height: 1.65;
        }

        .stApp {
            background: #0f172a;
            color: #e2e8f0;
            font-family: "Segoe UI", "Inter", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stAppViewContainer"] > .main .block-container {
            background: transparent !important;
            position: relative;
            z-index: 3;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent !important;
        }

        .video-background {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .video-background video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: saturate(1.08) brightness(0.95);
            background: #0f172a;
        }

        .video-background video::-webkit-media-controls {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }

        .video-overlay {
            position: fixed;
            inset: 0;
            z-index: 1;
            background: rgba(15, 23, 42, 0.28);
            pointer-events: none;
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 920px;
            padding-top: 1.2rem;
            padding-bottom: 8rem;
        }

        .fade-in {
            animation: fadeInUp 0.45s ease both;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .profile-wrapper {
            text-align: center;
            margin-bottom: 1.2rem;
        }

        .profile-avatar {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid rgba(255, 255, 255, 0.65);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        }

        .profile-avatar.placeholder {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.8);
            color: #e2e8f0;
            font-size: 3rem;
            font-weight: 700;
        }

        .profile-name {
            margin-top: 0.65rem;
            font-size: 1.65rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .profile-role {
            margin-top: 0.2rem;
            color: #cbd5e1;
            font-size: 1rem;
        }

        .chat-title {
            text-align: center;
            margin: 1rem 0 0.4rem;
            color: #f8fafc;
            font-size: 1.82rem;
            letter-spacing: 0.01em;
        }

        .chat-subtitle {
            text-align: center;
            color: #cbd5e1;
            margin-bottom: 1rem;
        }

        .chat-row {
            display: flex;
            width: 100%;
            margin: 0.54rem 0;
        }

        .chat-row.user { justify-content: flex-end; }
        .chat-row.assistant { justify-content: flex-start; }

        .chat-bubble {
            max-width: min(74%, 760px);
            padding: 0.82rem 1rem;
            border-radius: 14px;
            line-height: var(--chat-line-height);
            font-size: var(--chat-font-size);
            letter-spacing: 0.004em;
            backdrop-filter: blur(8px);
            word-wrap: break-word;
        }

        .chat-bubble.user {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.9), rgba(14, 165, 233, 0.82));
            color: #ffffff;
            border-bottom-right-radius: 6px;
            box-shadow: 0 8px 22px rgba(30, 64, 175, 0.34);
            white-space: pre-wrap;
            font-weight: 500;
        }

        .chat-bubble.assistant {
            background: rgba(255, 255, 255, 0.9);
            color: #0b1220;
            border: 1px solid rgba(148, 163, 184, 0.42);
            border-bottom-left-radius: 6px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.22);
            font-weight: 430;
        }

        .assistant-content {
            text-wrap: pretty;
        }

        .assistant-content p {
            margin: 0 0 0.56rem;
        }

        .assistant-content p:last-child {
            margin-bottom: 0;
        }

        .assistant-content ol,
        .assistant-content ul {
            margin: 0.16rem 0 0.72rem 1.15rem;
            padding-left: 0.8rem;
        }

        .assistant-content li {
            margin: 0.2rem 0;
        }

        .assistant-content li::marker {
            color: #334155;
        }

        .assistant-content strong {
            font-weight: 640;
            color: #0b1220;
        }

        .assistant-content em {
            font-style: italic;
        }

        .assistant-content h3,
        .assistant-content h4,
        .assistant-content h5 {
            margin: 0.2rem 0 0.45rem;
            line-height: 1.35;
            color: #0b1220;
        }

        .assistant-content code {
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.9em;
            padding: 0.08rem 0.35rem;
            border-radius: 6px;
            background: rgba(15, 23, 42, 0.08);
        }

        .chat-image-wrap {
            margin: 0.35rem 0 0.9rem;
        }

        .chat-image-wrap img {
            border-radius: 14px;
            border: 1px solid rgba(226, 232, 240, 0.15);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.38);
        }

        [data-testid="stChatInput"] {
            position: fixed;
            left: 50%;
            bottom: 16px;
            transform: translateX(-50%);
            width: min(920px, calc(100% - 2rem));
            z-index: 1001;
        }

        [data-testid="stChatInput"] > div {
            border-radius: 999px;
            background: rgba(248, 250, 252, 0.97);
            border: 1px solid rgba(148, 163, 184, 0.45);
            box-shadow: 0 12px 24px rgba(2, 6, 23, 0.28);
            backdrop-filter: blur(8px);
        }

        [data-testid="stChatInput"] textarea {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            caret-color: #0f172a !important;
            font-weight: 500 !important;
            font-size: 0.98rem !important;
            line-height: 1.45 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #64748b !important;
        }

        [data-testid="stChatInput"] button {
            color: #0f172a !important;
        }

        [data-testid="stChatInput"] button svg {
            fill: #0f172a !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(10, 14, 24, 0.75) !important;
            border-right: 1px solid rgba(226, 232, 240, 0.08);
            backdrop-filter: blur(8px);
            position: relative;
            z-index: 5;
        }

        @media (max-width: 900px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-bottom: 7rem;
            }

            .chat-title {
                font-size: 1.55rem;
            }

            .chat-bubble {
                max-width: 92%;
                font-size: 14.8px;
                padding: 0.76rem 0.88rem;
            }

            [data-testid="stChatInput"] {
                width: calc(100% - 1rem);
                bottom: 10px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_background_video(video_path: Path) -> bool:
    if not video_path.exists():
        st.markdown('<div class="video-overlay"></div>', unsafe_allow_html=True)
        return False

    video_data_uri = file_to_data_uri(str(video_path), "video/mp4")
    if not video_data_uri:
        st.markdown('<div class="video-overlay"></div>', unsafe_allow_html=True)
        return False

    st.markdown(
        f"""
        <div class="video-background">
            <video autoplay loop muted playsinline webkit-playsinline preload="auto">
                <source src="{video_data_uri}" type="video/mp4" />
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="video-overlay"></div>', unsafe_allow_html=True)
    return True


def build_chat_bubble_html(role: str, text: str) -> str:
    safe_role = "user" if role == "user" else "assistant"

    if safe_role == "assistant":
        bubble_content = f'<div class="assistant-content">{assistant_markdown_to_html(text)}</div>'
    else:
        bubble_content = html.escape(text).replace("\n", "<br>")

    return (
        f'<div class="chat-row {safe_role} fade-in">'
        f'<div class="chat-bubble {safe_role}">{bubble_content}</div>'
        "</div>"
    )


def render_chat_bubble(role: str, text: str) -> None:
    st.markdown(build_chat_bubble_html(role, text), unsafe_allow_html=True)


def render_profile_header(profile_image_path: Path | None, profile_image_url: str) -> None:
    image_src = None
    if profile_image_path:
        image_src = file_to_data_uri(str(profile_image_path), "image/jpeg")

    if not image_src and profile_image_url and profile_image_url.lower() not in {"undefined", "none", "null"}:
        image_src = profile_image_url

    if image_src:
        avatar_html = f'<img class="profile-avatar" src="{image_src}" alt="Muhammad Mudassar" />'
    else:
        avatar_html = '<div class="profile-avatar placeholder">M</div>'

    st.markdown(
        f"""
        <div class="profile-wrapper fade-in">
            {avatar_html}
            <div class="profile-name">Muhammad Mudassar</div>
            <div class="profile-role">AI Developer • Flutter Engineer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_image_reply(profile_image_path: Path | None, profile_image_url: str) -> None:
    st.markdown('<div class="chat-image-wrap fade-in">', unsafe_allow_html=True)
    if profile_image_path:
        st.image(str(profile_image_path), width=280)
    elif profile_image_url and profile_image_url.lower() not in {"undefined", "none", "null"}:
        st.image(profile_image_url, width=280)
    else:
        st.warning("I couldn't find the profile image. Please add `mudassar.jpg` in `data/` or set `PROFILE_IMAGE_URL`.")
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing knowledge base...")
def build_qa_chain(data_path: Path, groq_api_key: str, groq_model: str, data_signature: str):
    try:
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_community.document_loaders import TextLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_groq import ChatGroq
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise RuntimeError(
            f"LangChain dependencies failed to import ({exc.__class__.__name__}: {exc}). "
            "If this is Streamlit Cloud, set Python to 3.11 in App Settings → Advanced settings and reboot the app."
        ) from exc

    text_files = get_text_files(data_path)
    if not text_files:
        raise ValueError("No .txt files found in the data folder.")

    full_context_sections = []
    for file_path in text_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except Exception:
            content = ""

        if content:
            full_context_sections.append(f"[{file_path.name}]\n{content}")

    full_context = "\n\n".join(full_context_sections)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = None
    index_source = "rebuilt"

    if FAISS_CACHE_DIR.exists() and FAISS_META_PATH.exists():
        try:
            meta = json.loads(FAISS_META_PATH.read_text(encoding="utf-8"))
            if (
                meta.get("data_signature") == data_signature
                and meta.get("embedding_model") == EMBEDDING_MODEL_NAME
            ):
                try:
                    db = FAISS.load_local(
                        str(FAISS_CACHE_DIR),
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                except TypeError:
                    db = FAISS.load_local(str(FAISS_CACHE_DIR), embeddings)
                index_source = "cache"
        except Exception:
            db = None

    if db is None:
        documents = []
        for file_path in text_files:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)
        try:
            FAISS_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
            db.save_local(str(FAISS_CACHE_DIR))
            FAISS_META_PATH.write_text(
                json.dumps(
                    {
                        "data_signature": data_signature,
                        "embedding_model": EMBEDDING_MODEL_NAME,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.3},
    )

    llm = ChatGroq(api_key=groq_api_key, model=groq_model)
    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
    )

    return retrieval_qa, index_source, len(text_files), llm, full_context


# =====================
# CONFIG + STYLES
# =====================
st.set_page_config(page_title="Mudassar AI", page_icon="🤖", layout="wide")
inject_modern_styles()

video_path = BACKGROUND_VIDEO_PATH if BACKGROUND_VIDEO_PATH.exists() else (BASE_DIR / "data" / "mudassar.mp4")
video_loaded = render_background_video(video_path)

load_dotenv_file(BASE_DIR / ".env")

# =====================
# PROFILE IMAGE CARD
# =====================
profile_image_url = os.getenv("PROFILE_IMAGE_URL", "").strip()
if not profile_image_url:
    try:
        profile_image_url = str(st.secrets.get("PROFILE_IMAGE_URL", "")).strip()
    except Exception:
        profile_image_url = ""

profile_data_path = BASE_DIR / "data"
profile_image_path = find_profile_image_path(profile_data_path) if profile_data_path.exists() else None

render_profile_header(profile_image_path, profile_image_url)

st.markdown('<div class="chat-title fade-in">💬 Ask anything about Mudassar</div>', unsafe_allow_html=True)

# =====================
# API KEY + DATA CHECKS
# =====================
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_api_key:
    try:
        groq_api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        groq_api_key = ""

groq_model = os.getenv("GROQ_MODEL", "").strip()
if not groq_model:
    try:
        groq_model = str(st.secrets.get("GROQ_MODEL", "")).strip()
    except Exception:
        groq_model = ""

if not groq_model:
    groq_model = DEFAULT_GROQ_MODEL

invalid_key_values = {"", "your_groq_api_key", "undefined", "none", "null"}
if groq_api_key.strip().lower() in invalid_key_values:
    st.error("Please set `GROQ_API_KEY` in `.env` or Streamlit Secrets before running the assistant.")
    st.stop()

data_path = BASE_DIR / "data"
if not data_path.exists():
    st.error("The `data` folder was not found.")
    st.stop()

text_files = get_text_files(data_path)
if not text_files:
    st.error("No `.txt` files were found in the `data` folder.")
    st.stop()

empty_text_files = []
for file_path in text_files:
    try:
        if not file_path.read_text(encoding="utf-8").strip():
            empty_text_files.append(file_path.name)
    except Exception:
        empty_text_files.append(file_path.name)

data_signature = build_data_signature(text_files)

try:
    qa, index_source, indexed_file_count, llm, full_context = build_qa_chain(
        data_path,
        groq_api_key,
        groq_model,
        data_signature,
    )
except Exception as exc:
    st.error("Failed to initialize the RAG pipeline.")
    st.caption(str(exc))
    root_exc = exc.__cause__ if getattr(exc, "__cause__", None) else exc
    with st.expander("Technical details"):
        st.code(
            f"{type(root_exc).__name__}: {root_exc}\n"
            f"Python runtime: {sys.version.split()[0]}"
        )
    st.stop()


# =====================
# CHAT UI
# =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if message.get("type") == "image":
        render_profile_image_reply(profile_image_path, profile_image_url)
    else:
        render_chat_bubble(message.get("role", "assistant"), message.get("content", ""))

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input, "type": "text"})
    render_chat_bubble("user", user_input)

    try:
        raw_response = qa.invoke({"query": user_input})
        response = raw_response.get("result", "") if isinstance(raw_response, dict) else str(raw_response)

        if should_use_full_context_fallback(response) and full_context.strip():
            fallback_prompt = (
                "You are Mudassar's AI assistant. Use only the provided context.\n"
                "If user asks for image/photo/picture/profile image, return exactly SHOW_IMAGE.\n"
                "Write a concise, professional answer using Markdown.\n"
                "Use bullet points when listing multiple items.\n"
                "Avoid intros like \"As Mudassar's AI assistant\".\n"
                "Mention concrete names from context whenever available.\n\n"
                f"Context:\n{full_context}\n\n"
                f"Question: {user_input}\n\n"
                "Answer:"
            )
            fallback_result = llm.invoke(fallback_prompt)
            fallback_text = normalize_llm_text(fallback_result).strip()
            if fallback_text:
                response = fallback_text

        response = clean_assistant_response(user_input, response)

        if should_show_profile_image(user_input, response):
            if response.strip().upper() == "SHOW_IMAGE" or not response.strip():
                response = "Here is Mudassar's image."

            st.session_state.chat_history.append({"role": "assistant", "content": response, "type": "text"})
            render_chat_bubble("assistant", response)

            st.session_state.chat_history.append({"role": "assistant", "content": "profile_image", "type": "image"})
            render_profile_image_reply(profile_image_path, profile_image_url)
        elif not response.strip():
            warning_text = "Please try asking in a different way."
            st.session_state.chat_history.append({"role": "assistant", "content": warning_text, "type": "text"})
            render_chat_bubble("assistant", warning_text)
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": response, "type": "text"})
            render_chat_bubble("assistant", response)

    except Exception as exc:
        error_text = f"The Groq request failed: {type(exc).__name__}: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_text, "type": "text"})
        render_chat_bubble("assistant", error_text)


# =====================
# SIDEBAR
# =====================
st.sidebar.title("📂 Data Preview")
st.sidebar.caption(f"Model: {groq_model}")

if video_loaded:
    st.sidebar.success("🎬 Background video active")
    st.sidebar.caption("If motion is still blocked on mobile, browser autoplay policy is likely preventing playback.")
else:
    st.sidebar.warning("Background video file not found. Add `data/mudassar.mp4`.")

if index_source == "cache":
    st.sidebar.success("⚡ FAISS index loaded from cache")
else:
    st.sidebar.info("🧠 FAISS index rebuilt and cached")

if empty_text_files:
    st.sidebar.warning(f"Empty data files: {', '.join(empty_text_files)}")

st.sidebar.caption(f"Indexed files: {indexed_file_count}")
for file_path in text_files:
    st.sidebar.write("📄", file_path.name)
