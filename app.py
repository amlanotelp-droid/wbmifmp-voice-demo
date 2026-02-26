import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="WBMIFMP Voice AI Demo")
st.title("🎙 WBMIFMP Voice AI Assistant (Level 1 Demo)")
st.write("Speak or type. Answers only from official website.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Build Knowledge Base (First Run Only)
# -------------------------
if "db_ready" not in st.session_state:

    urls = [
        "https://wbmifmp.wb.gov.in",
        "https://wbmifmp.wb.gov.in/FAQ.aspx",
        "https://wbmifmp.wb.gov.in/Components.aspx"
    ]

    documents = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        documents.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    vectorstore.persist()
    st.session_state.db_ready = True

# -------------------------
# Load DB
# -------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Answer ONLY using the context provided.
If answer is not available, say:
"Not available on the official website."
Context:
{context}
Question:
{question}
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# -------------------------
# Voice Button (Browser STT)
# -------------------------
st.markdown("""
<script>
function startRecognition() {
    const recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = function(event) {
        const text = event.results[0][0].transcript;
        const inputBox = window.parent.document.querySelector('input');
        inputBox.value = text;
        inputBox.dispatchEvent(new Event('input', { bubbles: true }));
    }
}
</script>
<button onclick="startRecognition()" 
style="padding:8px 16px;font-size:16px;">
🎤 Speak
</button>
""", unsafe_allow_html=True)

# -------------------------
# Text Input
# -------------------------
query = st.text_input("Ask about WBMIFMP:")

if query:
    response = qa.run(query)

    st.subheader("Answer")
    st.write(response)

    # Text-to-Speech
    st.markdown(f"""
    <script>
    const speech = new SpeechSynthesisUtterance("{response}");
    speech.lang = "en-US";
    window.speechSynthesis.speak(speech);
    </script>
    """, unsafe_allow_html=True)