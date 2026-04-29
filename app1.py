import streamlit as st
import os
import ssl
import httpx

from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# --- SSL FIX ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Load env ---
load_dotenv()

# --- HTTP client ---
client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "token"


# --- Load TCS Models ---
def load_models():
    api_key = os.getenv("API_KEY") or "PASTE_YOUR_API_KEY_HERE"

    llm = ChatOpenAI(
        openai_api_base="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=api_key,
        http_client=client
    )

    embeddings = OpenAIEmbeddings(
        openai_api_base="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key=api_key,
        http_client=client
    )

    return llm, embeddings

llm, embedding_model = load_models()

# --- UI ---
st.title("📄 Insurance Policy Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Step 2: Extract text
    text = extract_text(uploaded_file)

    st.subheader("📖 Extracted Text Preview:")
    st.write(text[:500])

    # Step 3: Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    #st.write(f"🧩 Total chunks: {len(chunks)}")

    # Step 4: Create embeddings + DB
    db = Chroma.from_texts(chunks, embedding_model)

    st.success("✅ Embeddings created using TCS GenAI Lab!")

    # Save DB
    st.session_state["vector_db"] = db


# =========================
# 🚀 STEP 5: ASK QUESTIONS
# =========================

if "vector_db" in st.session_state:

    st.subheader("💬 Ask Questions")

    user_question = st.text_input("Ask something about your document:")

    if user_question:
        db = st.session_state["vector_db"]

        # Retrieve relevant chunks
        retriever = db.as_retriever()
        docs = retriever.invoke(user_question)[:3]

        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
        You are a helpful assistant.

        Answer in very simple language like explaining to a beginner.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # Call LLM
        response = llm.invoke(prompt)

        # Show answer
        st.subheader("🤖 Answer")
        st.write(response.content)

        # Show sources (important for judges)
        st.subheader("📄 Source from document")
        for i, doc in enumerate(docs):
            st.write(f"Source {i+1}:")
            st.write(doc.page_content[:200])
            st.write("------")