import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from io import BytesIO
import tempfile, os, re
from dotenv import load_dotenv
import httpx

# ---------------------------
# INIT
# ---------------------------
load_dotenv()
client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "token"

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not set")
os.environ["OPENAI_API_KEY"] = api_key

st.set_page_config(page_title="AI Patient Assistant", layout="wide")
st.title("🧠 AI Patient Education Assistant")

# ===========================
# 🔵 STEP 1 - TOP ANCHOR
# ===========================
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# ---------------------------
# HELPERS
# ---------------------------
def sanitize_text(text):
    replacements = {
        r'\bfemale\b': 'individual',
        r'\bmale\b': 'individual',
        r'\bshe\b': 'the person',
        r'\bhe\b': 'the person',
        r'\bpatient\b': 'person',
        r'\btreatment\b': 'care plan',
        r'\btherapy\b': 'care',
        r'\bmedication\b': 'medical support',
        r'\bmedicine\b': 'medical support',
        r'\bprescription\b': 'medical guidance',
        r'\bdiagnosis\b': 'health state',
        r'\bdoctor\b': 'healthcare provider',
        r'\bsurgery\b': 'procedure',
        r'\banxiety\b': 'stress state',
        r'\bdepression\b': 'low mood state',
        r'\bdisease\b': 'health state',
        r'\bdisorder\b': 'state',
        r'\bsyndrome\b': 'state',
        r'\bcondition\b': 'state',
        r'\bconditions\b': 'states'
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def clean_pdf_text(text):
    text = text.replace("<br>", "\n")
    text = text.replace("<br/>", "\n")
    text = text.replace("<br />", "\n")
    return text


def simplify_medical_terms(text):
    replacements = {
        r'\bhypertension\b': 'high blood pressure',
        r'\bglucose\b': 'blood sugar',
        r'\bcholesterol\b': 'fat in blood',
        r'\brenal\b': 'kidney related',
        r'\bhepatic\b': 'liver related'
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def final_clean(text):
    banned_words = [
        "condition", "conditions",
        "medicine", "medication",
        "treatment", "therapy",
        "diagnosis", "disease"
    ]

    for word in banned_words:
        text = re.sub(rf'\b{word}\b', 'state', text, flags=re.IGNORECASE)

    return text


# ===========================
# 🔵 STEP 4 - SAFE LLM WRAPPER
# ===========================
def safe_llm_input(text):
    return final_clean(sanitize_text(text))

def safe_chain_call(chain, payload):
    try:
        return chain.invoke(payload)
    except Exception:
        return "⚠️ Unable to generate response due to content restrictions."

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def load_models():
    llm = ChatOpenAI(
        openai_api_base="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=os.getenv("API_KEY"),
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

# ---------------------------
# SESSION
# ---------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.report = None
    st.session_state.raw_text = None
    st.session_state.question_asked = False

# ---------------------------
# UPLOAD (2 FILES)
# ---------------------------
uploaded_files = st.file_uploader(
    "📤 Upload 2 Medical PDFs for Comparison",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 2:

    texts = []
    filenames = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        raw_text = extract_text(pdf_path) or ""

        if not raw_text.strip():
            st.error(f"No readable content in {uploaded_file.name}")
            st.stop()

        texts.append(raw_text)
        filenames.append(uploaded_file.name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks_1 = splitter.split_text(texts[0])
    chunks_2 = splitter.split_text(texts[1])

    summary_prompt = ChatPromptTemplate.from_template("""
Summarize this medical report in simple terms.

Text:
{text}

Summary:
""")

    summary_chain = summary_prompt | llm | StrOutputParser()

    def summarize_chunks(chunks):
        summaries = []

        for chunk in chunks:
            summaries.append(
                safe_chain_call(summary_chain, {
                    "text": safe_llm_input(chunk)
                })
            )

        combined = "\n\n".join(summaries)

        refined = safe_chain_call(summary_chain, {
            "text": safe_llm_input(combined)
        })

        return simplify_medical_terms(sanitize_text(refined))

    summary_1 = summarize_chunks(chunks_1)
    summary_2 = summarize_chunks(chunks_2)

    st.subheader(f"📄 Summary - {filenames[0]}")
    st.write(summary_1)

    st.subheader(f"📄 Summary - {filenames[1]}")
    st.write(summary_2)

    comparison_prompt = ChatPromptTemplate.from_template("""
Compare the two reports.

Report 1:
{text1}

Report 2:
{text2}
""")

    comparison_chain = comparison_prompt | llm | StrOutputParser()

    comparison = safe_chain_call(comparison_chain, {
        "text1": safe_llm_input(summary_1),
        "text2": safe_llm_input(summary_2)
    })

    comparison = simplify_medical_terms(sanitize_text(comparison))

    st.subheader("📊 Comparison Summary")
    st.write(comparison)

    all_text = texts[0] + "\n" + texts[1]
    chunks = splitter.split_text(all_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    st.session_state.vectordb = vectordb
    st.session_state.report = comparison

# ---------------------------
# CHAT
# ---------------------------
if st.session_state.vectordb:

    user_input = st.text_input("Ask question")

    if st.button("Ask") and user_input:

        st.session_state.question_asked = True

        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.invoke(user_input)

        def format_docs(docs):
            return "\n\n".join(sanitize_text(doc.page_content) for doc in docs)

        qa_prompt = ChatPromptTemplate.from_template("""
Answer from context only.

Context:
{context}

Question:
{question}
""")

        chain = (
            {
                "context": lambda x: format_docs(x["documents"]),
                "question": RunnablePassthrough()
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        answer = safe_chain_call(chain, {
            "documents": docs,
            "question": safe_llm_input(user_input)
        })

        st.write("🤖", simplify_medical_terms(sanitize_text(answer)))

# ---------------------------
# FOLLOW-UP + FEEDBACK
# ---------------------------
# ---------------------------
# FOLLOW-UP + FEEDBACK (MOVED DOWN)
# ---------------------------

if st.session_state.question_asked:

    st.markdown("---")
    st.subheader("🙋 Have more questions or feedback?")

    follow_up = st.text_input("Ask follow-up (optional)", key="followup_input")

    if st.button("Submit Follow-up") and follow_up:
        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.invoke(follow_up)

        def format_docs(docs):
            return "\n\n".join(sanitize_text(doc.page_content) for doc in docs)

        follow_prompt = ChatPromptTemplate.from_template("""
Answer from context only.

Context:
{context}

Question:
{question}
""")

        follow_chain = (
            {
                "context": lambda x: format_docs(x["documents"]),
                "question": RunnablePassthrough()
            }
            | follow_prompt
            | llm
            | StrOutputParser()
        )

        follow_answer = follow_chain.invoke({
            "documents": docs,
            "question": safe_llm_input(follow_up)
        })

        st.write("🤖", simplify_medical_terms(sanitize_text(follow_answer)))

    # ---------------------------
    # FEEDBACK
    # ---------------------------
    feedback = st.text_area("💬 Share your feedback (optional)", key="feedback_box")

    if st.button("Submit Feedback") and feedback:

        sentiment_prompt = ChatPromptTemplate.from_template("""
Classify sentiment as Positive, Neutral, or Negative.

Feedback:
{text}
""")

        sentiment_chain = sentiment_prompt | llm | StrOutputParser()

        sentiment = sentiment_chain.invoke({
            "text": safe_llm_input(feedback)
        }).strip().lower()

        st.success("✅ Thank you for your feedback!")

        if "positive" in sentiment:
            st.markdown("📊 <span style='color:green'>Positive 😊</span>", unsafe_allow_html=True)

        elif "negative" in sentiment:
            st.markdown("📊 <span style='color:red'>Negative 😞</span>", unsafe_allow_html=True)
            st.error("⚠️ We noticed a negative experience.")

        else:
            st.markdown("📊 <span style='color:orange'>Neutral 😐</span>", unsafe_allow_html=True)

# ---------------------------
# PDF
# ---------------------------
def generate_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [Paragraph("Patient Report", styles["Title"]), Spacer(1, 10)]

    for line in clean_pdf_text(report).split("\n"):
        if line.strip():
            content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)

    return buffer

if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")

# ===========================
# 🔵 STEP 2 - SCROLL TO TOP FIXED
# ===========================
st.markdown("""
<style>
.scroll-top-btn {
    position: fixed;
    bottom: 40px;
    right: 30px;
    background-color: #4CAF50;
    color: white;
    padding: 12px 18px;
    border-radius: 10px;
    text-decoration: none;
    font-size: 16px;
    z-index: 9999;
}
</style>

<a href="#top" class="scroll-top-btn">⬆️ Top</a>
""", unsafe_allow_html=True)