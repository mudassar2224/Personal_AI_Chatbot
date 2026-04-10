import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="Mudassar AI", page_icon="🤖", layout="centered")

st.title("🤖 Mudassar Personal AI Assistant (RAG)")

# =====================
# SHOW IMAGE (PROFILE)
# =====================
image_path = "data/mudassar.jpg"

if os.path.exists(image_path):
    st.image(image_path, width=200)
    st.caption("Muhammad Mudassar - AI Developer")

# =====================
# API KEY (PUT YOUR KEY)
# =====================
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

DATA_PATH = "data"

# =====================
# LOAD FILES
# =====================
documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
        documents.extend(loader.load())

# =====================
# SPLIT TEXT
# =====================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# =====================
# EMBEDDINGS + VECTOR DB
# =====================
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

# =====================
# AI MODEL (BRAIN)
# =====================
llm = ChatOpenAI(model="gpt-4o-mini")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# =====================
# CHAT UI
# =====================
st.subheader("💬 Ask anything about Mudassar")

user_input = st.text_input("Type your question:")

if user_input:
    with st.spinner("Thinking... 🤔"):
        response = qa.run(user_input)
        st.success(response)

# =====================
# SIDEBAR (DATA PREVIEW)
# =====================
st.sidebar.title("📂 Data Preview")

for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        st.sidebar.write("📄", file)
