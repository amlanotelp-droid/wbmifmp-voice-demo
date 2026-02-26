import streamlit as st
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
@st.cache_data
def load_website():
    url = "https://wbmifmp.wb.gov.in"
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

st.title("WBMIFMP AI Assistant")

# Load API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Scrape Website
@st.cache_data
def load_website():
    url = "https://wbmifmp.wb.gov.in"
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

text_data = load_website()

# Step 2: Create Chroma DB
chroma_client = chromadb.Client()
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="wbmifmp",
    embedding_function=embedding_function
)

if collection.count() == 0:
    collection.add(
        documents=[text_data],
        ids=["wbmifmp_home"]
    )

# Step 3: Query Input
query = st.text_input("Ask about WBMIFMP:")

if query:
    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    context = results["documents"][0][0]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only from the given context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )

    st.write(response.choices[0].message.content)