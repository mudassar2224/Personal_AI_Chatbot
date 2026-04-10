import streamlit as st
import os

st.set_page_config(page_title="Mudassar AI Assistant", layout="centered")

st.title("🤖 Mudassar Personal AI Chatbot")

DATA_FOLDER = "data"

# 📂 Load all files
def load_all_data():
    text = ""
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
                text += f"\n\n" + f.read()
    return text

all_data = load_all_data()

# 🧠 Simple AI logic (upgrade later to RAG)
def get_answer(question):
    q = question.lower()

    if "name" in q:
        return "My name is Muhammad Mudassar."

    if "education" in q:
        return open("data/education.txt", encoding="utf-8").read()

    if "projects" in q:
        return open("data/flutter_AI_Apps.txt", encoding="utf-8").read()

    if "machine learning" in q:
        return open("data/machine_learning_projects.txt", encoding="utf-8").read()

    if "skills" in q:
        return open("data/skills.txt", encoding="utf-8").read()

    if "contact" in q:
        return open("data/contactinfo.txt", encoding="utf-8").read()

    if "achievements" in q:
        return open("data/achievements.txt", encoding="utf-8").read()

    if "who are you" in q:
        return open("data/about.txt", encoding="utf-8").read()

    return "I don't know this yet. Ask about education, skills, projects, or contact info."

# 💬 UI
user_input = st.text_input("Ask about Mudassar:")

if user_input:
    response = get_answer(user_input)
    st.write(response)

st.sidebar.title("📂 Data Preview")
st.sidebar.write(all_data[:1000])