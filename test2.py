# =====================================================
# FINAL STREAMLIT + LANGCHAIN + XAI + HUMAN READABLE UI
# =====================================================

import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO
from dotenv import load_dotenv

import tempfile
import httpx
import os
import re
import json

# =====================================================
# INIT
# =====================================================

load_dotenv()

client = httpx.Client(verify=False)

os.environ["TIKTOKEN_CACHE_DIR"] = "token"

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found")

os.environ["OPENAI_API_KEY"] = api_key

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Business Process Document Generator",
    layout="wide"
)

st.title("Business Process Document Generator")

st.markdown("""
### Upload Documents
""")

# =====================================================
# HELPERS
# =====================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    replacements = {
        r'\bpls\b': 'please',
        r'\bu\b': 'you',
        r'\basap\b': 'as soon as possible'
    }

    for pattern, replacement in replacements.items():

        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def is_business_document(text):

    keywords = [
        "process",
        "approval",
        "workflow",
        "requirement",
        "manager",
        "project",
        "stakeholder",
        "business",
        "meeting",
        "SOP"
    ]

    count = sum(
        word.lower() in text.lower()
        for word in keywords
    )

    return count >= 3


def format_docs(docs):

    formatted = []

    for idx, doc in enumerate(docs):

        formatted.append(
            f"[C{idx+1}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def extract_file_text(uploaded_file):

    suffix = ".pdf"

    if uploaded_file.name.endswith(".txt"):
        suffix = ".txt"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getvalue())

        file_path = tmp.name

    if uploaded_file.name.endswith(".pdf"):

        text = extract_text(file_path)

    elif uploaded_file.name.endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

    else:

        text = ""

    return clean_text(text)

# =====================================================
# XAI HELPERS
# =====================================================

def generate_confidence_score(context_docs, answer):

    context_length = len(context_docs)

    answer_length = len(answer)

    context_score = min(
        context_length * 12,
        40
    )

    answer_score = min(
        answer_length / 18,
        25
    )

    reference_score = 20 if "C1" in answer else 10

    structure_score = 15

    total_score = (
        context_score +
        answer_score +
        reference_score +
        structure_score
    )

    total_score = min(
        round(total_score),
        100
    )

    if total_score >= 90:
        confidence_level = "Highly Reliable"

    elif total_score >= 75:
        confidence_level = "Reliable"

    elif total_score >= 60:
        confidence_level = "Moderate Confidence"

    else:
        confidence_level = "Low Confidence"

    return total_score, confidence_level

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    llm = ChatOpenAI(
        openai_api_base="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=os.getenv("API_KEY"),
        temperature=0.2,
        http_client=client
    )

    embeddings = OpenAIEmbeddings(
        openai_api_base="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key=os.getenv("API_KEY"),
        http_client=client
    )

    return llm, embeddings


llm, embedding_model = load_models()

# =====================================================
# SESSION STATE
# =====================================================

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "workflow_steps" not in st.session_state:
    st.session_state.workflow_steps = []

if "raw_text" not in st.session_state:
    st.session_state.raw_text = None

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "📤 Upload Business Documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# =====================================================
# PROCESS DOCUMENTS
# =====================================================

if uploaded_files:

    all_text = ""

    for uploaded_file in uploaded_files:

        extracted_text = extract_file_text(
            uploaded_file
        )

        all_text += "\n\n" + extracted_text

    st.session_state.raw_text = all_text

    if not all_text.strip():

        st.error("No readable content found")
        st.stop()

    if not is_business_document(all_text):

        st.error(
            "Uploaded documents do not appear to be business documents"
        )

        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_text(all_text)

    vectordb = Chroma.from_texts(
        chunks,
        embedding_model
    )

    st.session_state.vectordb = vectordb

    # =====================================================
    # ANALYSIS PROMPT
    # =====================================================

    analysis_prompt = ChatPromptTemplate.from_template("""

You are an expert Business Analyst and Explainable AI Assistant.

Your task is to analyze ONLY the uploaded business documents.

STRICT RULES:

1. Use ONLY the provided document context.
2. Do NOT make assumptions.
3. Do NOT generate imaginary workflows or objectives.
4. If information is unavailable, return:
"Information not available in provided document."
5. Return human readable content.
6. Return ONLY valid JSON.

JSON FORMAT:

{{
    "executive_summary": {{
        "summary": ""
    }},

    "business_objective": {{
        "objective": ""
    }},

    "workflow_steps": [
        {{
            "step": "",
            "stakeholder": ""
        }}
    ]
}}

DOCUMENT CONTEXT:
{text}

""")

    analysis_chain = (
        analysis_prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Analyzing documents..."):

        result = analysis_chain.invoke({
            "text": all_text
        })

    try:

        cleaned_result = result.strip()

        if cleaned_result.startswith("```json"):

            cleaned_result = cleaned_result.replace(
                "```json",
                ""
            )

            cleaned_result = cleaned_result.replace(
                "```",
                ""
            )

        parsed = json.loads(cleaned_result)

    except Exception:

        st.error("AI returned invalid JSON")

        st.write(result)

        st.stop()

    st.session_state.analysis = parsed

    st.session_state.workflow_steps = parsed.get(
        "workflow_steps",
        []
    )

# =====================================================
# DISPLAY ANALYSIS
# =====================================================

if st.session_state.analysis:

    analysis = st.session_state.analysis

    st.markdown("---")

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================

    st.subheader("📄 Executive Summary")

    executive_summary = st.text_area(
        "Executive Summary",
        value=analysis.get(
            "executive_summary",
            {}
        ).get(
            "summary",
            ""
        ),
        height=180
    )

    # =====================================================
    # BUSINESS OBJECTIVE
    # =====================================================

    st.subheader("🎯 Business Objective")

    business_objective = st.text_area(
        "Business Objective",
        value=analysis.get(
            "business_objective",
            {}
        ).get(
            "objective",
            ""
        ),
        height=120
    )

    # =====================================================
    # WORKFLOW STEPS
    # =====================================================

    st.subheader(
        "🔄 Editable Workflow Steps with Stakeholders"
    )

    workflow_steps = st.session_state.workflow_steps

    edited_steps = []

    for i, item in enumerate(workflow_steps):

        col1, col2 = st.columns([3, 2])

        with col1:

            updated_step = st.text_input(
                f"Workflow Step {i+1}",
                value=item.get("step", ""),
                key=f"step_{i}"
            )

        with col2:

            updated_stakeholder = st.text_input(
                f"Stakeholder {i+1}",
                value=item.get("stakeholder", ""),
                key=f"stakeholder_{i}"
            )

        edited_steps.append({
            "step": updated_step,
            "stakeholder": updated_stakeholder
        })

    # =====================================================
    # ADD NEW STEP
    # =====================================================

    st.markdown("### ➕ Add New Workflow Step")

    col1, col2 = st.columns([3, 2])

    with col1:

        new_step = st.text_input(
            "New Workflow Step",
            key="new_workflow_step"
        )

    with col2:

        new_stakeholder = st.text_input(
            "New Stakeholder",
            key="new_stakeholder"
        )

    if st.button("Add Workflow Step"):

        if new_step.strip():

            st.session_state.workflow_steps.append({
                "step": new_step,
                "stakeholder": new_stakeholder
            })

            st.rerun()

    st.session_state.workflow_steps = edited_steps

# =====================================================
# AI CHATBOT
# =====================================================

if st.session_state.vectordb:

    st.markdown("---")

    st.subheader("💬 Ask Questions")

    user_question = st.text_input(
        "Ask about workflow, process or stakeholders"
    )

    if st.button("Ask AI") and user_question:

        retriever = (
            st.session_state.vectordb.as_retriever()
        )

        docs = retriever.invoke(user_question)

        qa_prompt = ChatPromptTemplate.from_template("""

You are an AI Business Analyst Assistant.

Answer ONLY from provided context.

CONTEXT:
{context}

QUESTION:
{question}

""")

        qa_chain = (
            {
                "context": lambda x: format_docs(
                    x["documents"]
                ),
                "question": RunnablePassthrough()
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        answer = qa_chain.invoke({
            "documents": docs,
            "question": user_question
        })

        st.markdown("### 🤖 AI Answer")

        st.write(answer)

        confidence_score, confidence_level = (
            generate_confidence_score(
                docs,
                answer
            )
        )

        st.markdown("### 📊 Confidence Score")

        st.progress(confidence_score / 100)

        st.write(
            f"Score: {confidence_score}% "
            f"({confidence_level})"
        )

# =====================================================
# PDF EXPORT
# =====================================================

def generate_pdf(
    summary,
    objective,
    steps
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "Business Process Document",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"<b>Executive Summary:</b> {summary}",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            f"<b>Business Objective:</b> {objective}",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "<b>Workflow Steps & Stakeholders</b>",
            styles["Heading2"]
        )
    )

    for item in steps:

        line = (
            f"• <b>Step:</b> {item.get('step', '')}"
            f"<br/><b>Stakeholder:</b> "
            f"{item.get('stakeholder', '')}"
        )

        content.append(
            Paragraph(
                line,
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 10))

    doc.build(content)

    buffer.seek(0)

    return buffer

# =====================================================
# DOWNLOAD REPORT
# =====================================================

if st.session_state.analysis:

    pdf = generate_pdf(
        executive_summary,
        business_objective,
        st.session_state.workflow_steps
    )

    st.download_button(
        label="📥 Download File",
        data=pdf,
        file_name="Business_Process.pdf",
        mime="application/pdf"
    )



# =====================================================
# RAW TEXT VIEW
# =====================================================

if st.session_state.raw_text:

    with st.expander(
        "📜 View Extracted Raw Text"
    ):

        st.write(
            st.session_state.raw_text
        )