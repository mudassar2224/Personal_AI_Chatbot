import os
import hashlib
import json
from pathlib import Path

import streamlit as st

os.environ["STREAMLIT_WATCHDOG_IGNORE"] = "torch"

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_CACHE_DIR = BASE_DIR / ".cache" / "faiss_index"
FAISS_META_PATH = BASE_DIR / ".cache" / "faiss_meta.json"


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


def build_data_signature(text_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for file_path in text_files:
        stat = file_path.stat()
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


@st.cache_resource(show_spinner="Preparing knowledge base...")
def build_qa_chain(data_path: Path, groq_api_key: str, data_signature: str):
    try:
        from langchain.chains import RetrievalQA
        from langchain_community.document_loaders import TextLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_groq import ChatGroq
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise RuntimeError(
            "LangChain dependencies failed to import. "
            "If this is Streamlit Cloud, set Python to 3.11 in runtime.txt."
        ) from exc

    text_files = get_text_files(data_path)
    if not text_files:
        raise ValueError("No .txt files found in the data folder.")

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

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever), index_source, len(text_files)


# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Mudassar AI", page_icon="🤖", layout="centered")
st.title("🤖 Mudassar Personal AI Assistant (RAG + GROQ)")

# =====================
# PROFILE IMAGE
# =====================
image_path = BASE_DIR / "data" / "mudassar.jpg"
if image_path.exists():
    st.image(str(image_path), width=200)
    st.caption("Muhammad Mudassar - AI Developer")

# =====================
# API KEY + DATA CHECKS
# =====================
load_dotenv_file(BASE_DIR / ".env")

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_api_key:
    try:
        groq_api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        groq_api_key = ""

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

data_signature = build_data_signature(text_files)

try:
    qa, index_source, indexed_file_count = build_qa_chain(data_path, groq_api_key, data_signature)
except Exception as exc:
    st.error("Failed to initialize the RAG pipeline.")
    st.caption(str(exc))
    st.stop()

# =====================
# CHAT UI
# =====================
st.subheader("💬 Ask anything about Mudassar")
user_input = st.text_input("Type your question:")

if user_input:
    with st.spinner("Thinking... 🤔"):
        try:
            response = qa.invoke({"query": user_input}).get("result", "")
        except Exception:
            response = qa.run(user_input)
        st.success(response)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("📂 Data Preview")
if index_source == "cache":
    st.sidebar.success("⚡ FAISS index loaded from cache")
else:
    st.sidebar.info("🧠 FAISS index rebuilt and cached")

st.sidebar.caption(f"Indexed files: {indexed_file_count}")
for file_path in text_files:
    st.sidebar.write("📄", file_path.name)
