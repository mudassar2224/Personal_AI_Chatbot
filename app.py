import os
import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "torch"

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
FAISS_CACHE_DIR = BASE_DIR / ".cache" / "faiss_index"
FAISS_META_PATH = BASE_DIR / ".cache" / "faiss_meta.json"
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

    image_keywords = (
        "image",
        "photo",
        "picture",
        "pic",
        "show me",
        "profile",
    )

    return (
        "show_image" in normalized_answer
        or any(keyword in normalized_question for keyword in image_keywords)
    )


def render_profile_image_reply(profile_image_path: Path | None, profile_image_url: str) -> None:
    if profile_image_path:
        st.image(str(profile_image_path), width=260)
        st.caption("Here is Mudassar's image.")
        return

    if profile_image_url and profile_image_url.lower() not in {"undefined", "none", "null"}:
        st.image(profile_image_url, width=260)
        st.caption("Here is Mudassar's image.")
        return

    st.warning("I couldn't find the profile image. Please add `mudassar.jpg` in `data/` or set `PROFILE_IMAGE_URL`.")


def should_use_full_context_fallback(answer: str) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return True
    return any(phrase in normalized for phrase in FALLBACK_TRIGGER_PHRASES)


def build_data_signature(text_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for file_path in text_files:
        stat = file_path.stat()
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


@st.cache_resource(show_spinner="Preparing knowledge base...")
def build_qa_chain(data_path: Path, groq_api_key: str, groq_model: str, data_signature: str):
    try:
        from langchain.chains import RetrievalQA
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

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever), index_source, len(text_files), llm, full_context


# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Mudassar AI", page_icon="🤖", layout="centered")
st.title("🤖 Mudassar Personal AI Assistant (RAG + GROQ)")

load_dotenv_file(BASE_DIR / ".env")

# =====================
# PROFILE IMAGE
# =====================
profile_image_url = os.getenv("PROFILE_IMAGE_URL", "").strip()
if not profile_image_url:
    try:
        profile_image_url = str(st.secrets.get("PROFILE_IMAGE_URL", "")).strip()
    except Exception:
        profile_image_url = ""

profile_data_path = BASE_DIR / "data"
profile_image_path = find_profile_image_path(profile_data_path) if profile_data_path.exists() else None

if has_profile_image(profile_image_path, profile_image_url):
    if profile_image_path:
        st.image(str(profile_image_path), width=200)
        st.caption("Muhammad Mudassar - AI Developer")
    else:
        st.image(profile_image_url, width=200)
        st.caption("Muhammad Mudassar - AI Developer")
else:
    st.info("Profile image not found. Add one in `data/` (e.g., `mudassar.jpg`) or set `PROFILE_IMAGE_URL` in Secrets.")

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
st.subheader("💬 Ask anything about Mudassar")
user_input = st.text_input("Type your question:")

if user_input:
    with st.spinner("Thinking... 🤔"):
        try:
            raw_response = qa.invoke({"query": user_input})
            response = raw_response.get("result", "") if isinstance(raw_response, dict) else str(raw_response)

            if should_use_full_context_fallback(response) and full_context.strip():
                fallback_prompt = (
                    "You are Mudassar's AI assistant. Use only the provided context. "
                    "If the user asks about projects, list concrete project names and short descriptions from the context. "
                    "If exact details are unavailable, clearly say what is missing and provide the closest relevant information.\n\n"
                    f"Context:\n{full_context}\n\n"
                    f"Question: {user_input}"
                )
                fallback_result = llm.invoke(fallback_prompt)
                fallback_text = normalize_llm_text(fallback_result).strip()
                if fallback_text:
                    response = fallback_text

            if should_show_profile_image(user_input, response):
                render_profile_image_reply(profile_image_path, profile_image_url)
                if response.strip().upper() != "SHOW_IMAGE":
                    st.success(response)
            elif not response.strip():
                st.warning("The model returned an empty response. Please try rephrasing your question.")
            else:
                st.success(response)
        except Exception as exc:
            st.error("The Groq request failed.")
            st.caption(f"{type(exc).__name__}: {exc}")
            st.info(
                "Try setting `GROQ_MODEL` to `llama-3.1-8b-instant` or `llama-3.3-70b-versatile` in Streamlit Secrets."
            )

# =====================
# SIDEBAR
# =====================
st.sidebar.title("📂 Data Preview")
st.sidebar.caption(f"Model: {groq_model}")
if index_source == "cache":
    st.sidebar.success("⚡ FAISS index loaded from cache")
else:
    st.sidebar.info("🧠 FAISS index rebuilt and cached")

if empty_text_files:
    st.sidebar.warning(f"Empty data files: {', '.join(empty_text_files)}")

st.sidebar.caption(f"Indexed files: {indexed_file_count}")
for file_path in text_files:
    st.sidebar.write("📄", file_path.name)
