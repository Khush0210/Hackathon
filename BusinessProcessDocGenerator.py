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

st.markdown(
    '<div id="top"></div>',
    unsafe_allow_html=True
)

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


def build_context_references(docs):

    references = []

    for idx, doc in enumerate(docs):

        references.append({
            "reference_id": f"C{idx+1}",
            "content": doc.page_content[:500],
            "relevance": "High"
        })

    return references

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
3. Do NOT generate imaginary workflows, stakeholders, approvals, or objectives.
4. If information is unavailable, return:
"Information not available in provided document."
5. Quote exact phrases wherever possible.
6. Every output item must include:
- justification
- source_evidence
- confidence_score
7. confidence_score must be between 0-100.
8. Higher confidence only when exact evidence exists in document.
9. Return ONLY valid JSON.
10. No markdown.
11. No explanation outside JSON.

JSON FORMAT:

{{
"executive_summary": {{
    "summary": "",
    "justification": "",
    "source_evidence": "",
    "confidence_score": 0
}},

"business_objective": {{
    "objective": "",
    "justification": "",
    "source_evidence": "",
    "confidence_score": 0
}},

"workflow_steps": [
    {{
    "step": "",
    "stakeholder": "",
    "justification": "",
    "source_evidence": "",
    "confidence_score": 0
    }}
],

"stakeholders": [
    {{
    "name": "",
    "role": "",
    "source_evidence": "",
    "confidence_score": 0
    }}
],

"approvals": [
    {{
    "approval": "",
    "approver": "",
    "source_evidence": "",
    "confidence_score": 0
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

    executive_summary_data = analysis.get(
        "executive_summary",
        {}
    )

    executive_summary = st.text_area(
        "Executive Summary",
        value=executive_summary_data.get(
            "summary",
            ""
        ),
        height=200
    )

    st.caption(
        f"Confidence Score: "
        f"{executive_summary_data.get('confidence_score', 0)}%"
    )

    with st.expander("View Evidence"):

        st.write("Justification:")

        st.write(
            executive_summary_data.get(
                "justification",
                ""
            )
        )

        st.write("Source Evidence:")

        st.write(
            executive_summary_data.get(
                "source_evidence",
                ""
            )
        )

    # =====================================================
    # BUSINESS OBJECTIVE
    # =====================================================

    st.subheader("🎯 Business Objective")

    objective_data = analysis.get(
        "business_objective",
        {}
    )

    business_objective = st.text_area(
        "Business Objective",
        value=objective_data.get(
            "objective",
            ""
        ),
        height=120
    )

    st.caption(
        f"Confidence Score: "
        f"{objective_data.get('confidence_score', 0)}%"
    )

    with st.expander("View Evidence"):

        st.write("Justification:")

        st.write(
            objective_data.get(
                "justification",
                ""
            )
        )

        st.write("Source Evidence:")

        st.write(
            objective_data.get(
                "source_evidence",
                ""
            )
        )

    # =====================================================
    # WORKFLOW STEPS
    # =====================================================

    st.subheader(
        "🔄 Workflow Steps with Stakeholders"
    )

    workflow_steps = analysis.get(
        "workflow_steps",
        []
    )

    for i, item in enumerate(workflow_steps):

        st.markdown(
            f"### Step {i+1}"
        )

        col1, col2 = st.columns([3, 2])

        with col1:

            st.text_input(
                f"Workflow Step {i+1}",
                value=item.get(
                    "step",
                    ""
                ),
                key=f"step_{i}"
            )

        with col2:

            st.text_input(
                f"Stakeholder {i+1}",
                value=item.get(
                    "stakeholder",
                    ""
                ),
                key=f"stakeholder_{i}"
            )

        st.caption(
            f"Confidence Score: "
            f"{item.get('confidence_score', 0)}%"
        )

        with st.expander(
            f"Evidence for Step {i+1}"
        ):

            st.write("Justification:")

            st.write(
                item.get(
                    "justification",
                    ""
                )
            )

            st.write("Source Evidence:")

            st.write(
                item.get(
                    "source_evidence",
                    ""
                )
            )

    # =====================================================
    # STAKEHOLDERS
    # =====================================================

    st.subheader("👥 Stakeholders")

    stakeholders = analysis.get(
        "stakeholders",
        []
    )

    for idx, stakeholder in enumerate(stakeholders):

        st.markdown(
            f"### Stakeholder {idx+1}"
        )

        st.write(
            f"Name: {stakeholder.get('name', '')}"
        )

        st.write(
            f"Role: {stakeholder.get('role', '')}"
        )

        st.caption(
            f"Confidence Score: "
            f"{stakeholder.get('confidence_score', 0)}%"
        )

        with st.expander(
            f"Evidence for Stakeholder {idx+1}"
        ):

            st.write("Source Evidence:")

            st.write(
                stakeholder.get(
                    "source_evidence",
                    ""
                )
            )

    # =====================================================
    # APPROVALS
    # =====================================================

    st.subheader("✅ Approvals")

    approvals = analysis.get(
        "approvals",
        []
    )

    for idx, approval in enumerate(approvals):

        st.markdown(
            f"### Approval {idx+1}"
        )

        st.write(
            f"Approval: {approval.get('approval', '')}"
        )

        st.write(
            f"Approver: {approval.get('approver', '')}"
        )

        st.caption(
            f"Confidence Score: "
            f"{approval.get('confidence_score', 0)}%"
        )

        with st.expander(
            f"Evidence for Approval {idx+1}"
        ):

            st.write("Source Evidence:")

            st.write(
                approval.get(
                    "source_evidence",
                    ""
                )
            )

# =====================================================
# AI CHATBOT WITH XAI
# =====================================================

if st.session_state.vectordb:

    st.markdown("---")

    st.subheader("💬 Ask Questions")

    user_question = st.text_input(
        "Ask about workflow, approvals, stakeholders, process"
    )

    if st.button("Ask AI") and user_question:

        retriever = (
            st.session_state.vectordb.as_retriever()
        )

        docs = retriever.invoke(user_question)

        qa_prompt = ChatPromptTemplate.from_template("""

You are an Explainable AI Business Analyst Assistant.

Your task is to answer ONLY from the provided context.

STRICT RULES:
1. Do NOT generate information outside the context.
2. If information is unavailable, clearly say:
"Information not available in provided documents."
3. Mention exact reference IDs from context.

OUTPUT FORMAT:

Final Answer:
...

Key Evidence:
...

Context References:
...

Confidence Reason:
...

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

        st.markdown("### 📚 Context References")

        references = build_context_references(docs)

        for ref in references:

            with st.expander(
                f"{ref['reference_id']} "
                f"- {ref['relevance']} Relevance"
            ):

                st.write(ref["content"])

# =====================================================
# PDF EXPORT
# =====================================================

def generate_pdf(summary, objective):

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

    doc.build(content)

    buffer.seek(0)

    return buffer

# =====================================================
# DOWNLOAD REPORT
# =====================================================

if st.session_state.analysis:

    pdf = generate_pdf(
        executive_summary,
        business_objective
    )

    st.download_button(
        label="📥 Download Final BRD Report",
        data=pdf,
        file_name="business_analysis_report.pdf",
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