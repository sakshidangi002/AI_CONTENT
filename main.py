import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os, requests, time
from difflib import SequenceMatcher
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

load_dotenv()

#  Connection function with check
@st.cache_resource
def db_con():
    return dict(
        host=os.getenv("POSTGRESQL_HOST"),
        port=os.getenv("POSTGRESQL_PORT"),
        dbname=os.getenv("POSTGRESQL_DB"),
        user=os.getenv("POSTGRESQL_USER"),
        password=os.getenv("POSTGRESQL_PASSWORD"),
        cursor_factory=RealDictCursor
    )


def get_conn():
    try:
        return psycopg2.connect(**db_con())
    except Exception as e:
        st.error(f"DB Connection failed: {e}")
        return None


#  Save content safely
def save_content(topic, content_type, length, prompt, content, plagiarism_score, generatedat=None):
    if generatedat is None:
        generatedat = datetime.now()

    conn = get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO content 
                (topic, contenttype, wordlength, prompt, content, plagiarism_score, generatedat)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (topic, content_type, length, prompt, content, plagiarism_score, generatedat)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Error while saving content: {e}")
    finally:
        conn.close()



#  History fetch
def history(limit=50):
    conn = get_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM content ORDER BY generatedat DESC LIMIT %s", (limit,))
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error while fetching history: {e}")
        return []
    finally:
        conn.close()


# ------------------- Streamlit App -------------------
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

llm = ChatOllama(model="gemma:2b")

st.title("AI content creation")
topic = st.text_input("enter your topic")
content_type = st.selectbox("content type", {"BlogPost", "Social Media", "Blog-Articles", "Product-Description"})
length = st.number_input("word length", min_value=100, max_value=20000)


# ---------- Plagiarism Checker ----------
def search_google(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": OLLAMA_API_KEY, "cx": GOOGLE_CSE_ID, "q": query}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [item["snippet"] for item in data.get("items", [])]

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def check_plagiarism(text, threshold=0.8):
    sentences = text.split(".")
    plagiarized = 0
    total = len(sentences)

    for sent in sentences:
        if len(sent.strip()) < 20:
            continue
        snippets = search_google(sent[:50])
        for snip in snippets:
            if similarity(sent.lower(), snip.lower()) > threshold:
                plagiarized += 1
                break

    score = (plagiarized / total) * 100 if total > 0 else 0
    return score


# ---------- Generate Content ----------
if st.button("generate content"):
    attempt = 1
    plagiarism_score = 100
    content = ""

    while plagiarism_score > 10 and attempt <= 5:  # add safety limit
        st.write(f"Attempt {attempt}: Generating content...")

        prompt = f"Write a unique {content_type} about {topic}, around {length} words. Avoid plagiarism."
        generated = llm.invoke(prompt)
        content = generated.content

        st.write(" Checking plagiarism via Google CSE…")
        plagiarism_score = check_plagiarism(content)
        st.write(f" Plagiarism score: {plagiarism_score:.2f}%")

        if plagiarism_score <= 10:
            st.subheader(" Final Content")
            st.write(content)
            save_content(topic, content_type, length, prompt, content, plagiarism_score)
            break
        else:
            st.warning(" High plagiarism score, regenerating content...")
            attempt += 1
            time.sleep(2)

    if plagiarism_score > 10:
        st.error(" Could not generate clean content within attempts limit.")


# ---------- History Sidebar ----------
if st.sidebar.button(" History"):
    st.subheader("LLM Response History")
    rows = history()

    if rows:
        for row in rows:
            with st.expander(f"{row['topic']} ({row['generatedat']})"):
                st.write(f"**Type:** {row['contenttype']}")
                st.write(f"**Length:** {row['wordlength']}")   
                st.write(f"**Prompt:** {row['prompt']}")
                st.write(f"**Plagiarism Score:** {row['plagiarism_score']}")
                st.write("**Generated Content:**")
                st.write(row['content'])
    else:
        st.info("No history available yet.")
