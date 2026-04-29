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

# ✅ STEP 1: TOP ANCHOR
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# ---------------------------
# SESSION STATE (UPDATED)
# ---------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.report = None
    st.session_state.raw_text = None

if "question_asked" not in st.session_state:
    st.session_state.question_asked = False

# 🧠 STEP 3: CHAT MEMORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        r'\bdiagnosis\b': 'medical condition',
        r'\bdoctor\b': 'healthcare provider',
        r'\bsurgery\b': 'procedure',
        r'\banxiety\b': 'stress condition',
        r'\bdepression\b': 'low mood condition',
        r'\bdisease\b': 'health condition',
        r'\bdisorder\b': 'condition',
        r'\bsyndrome\b': 'condition',
        r'\bdrug\b': 'substance',
        r'\bclinical\b': 'medical',
        r'\btrial\b': 'study'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
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


def is_medical_document(text):
    keywords = ["diagnosis", "treatment", "medication", "patient", "blood", "doctor"]
    return sum(word in text.lower() for word in keywords) >= 3

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
# UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Medical PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    raw_text = extract_text(pdf_path) or ""
    st.session_state.raw_text = raw_text

    if not raw_text.strip():
        st.error("No readable content")
        st.stop()

    if not is_medical_document(raw_text):
        st.error("Not a medical document")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(chunks, embedding_model)
    st.session_state.vectordb = vectordb

    # ---------------------------
    # SUMMARY
    # ---------------------------
    summary_prompt = ChatPromptTemplate.from_template("""
Summarize this medical report in simple terms.

Text:
{text}

Summary:
""")

    summary_chain = summary_prompt | llm | StrOutputParser()

    summaries = []
    for chunk in chunks:
        summaries.append(summary_chain.invoke({"text": sanitize_text(chunk)}))

    final_summary = "\n\n".join(summaries)

    final_summary = summary_chain.invoke({
        "text": sanitize_text(final_summary)
    })

    final_summary = simplify_medical_terms(sanitize_text(final_summary))

    st.subheader("📄 Medical Summary")
    st.write(final_summary)

    st.session_state.report = final_summary

# ---------------------------
# CHAT + STEP 3 + STEP 4
# ---------------------------
if st.session_state.vectordb:

    user_input = st.text_input("Ask question")

    if st.button("Ask") and user_input:
        st.session_state.question_asked = True

        retriever = st.session_state.vectordb.as_retriever()
        docs = retriever.invoke(user_input)

        def format_docs(docs):
            return "\n\n".join(sanitize_text(doc.page_content) for doc in docs)

        # 🧠 MEMORY
        history_text = "\n".join(
            [f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history[-5:]]
        )

        qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

History:
{history}

Context:
{context}

Question:
{question}
""")

        chain = (
            {
                "history": lambda x: history_text,
                "context": lambda x: format_docs(x["documents"]),
                "question": RunnablePassthrough()
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "documents": docs,
            "question": sanitize_text(user_input)
        })

        final_answer = simplify_medical_terms(sanitize_text(answer))

        st.write("🤖", final_answer)

        # SAVE MEMORY
        st.session_state.chat_history.append({
            "q": user_input,
            "a": final_answer
        })

        # 💡 SUGGESTIONS
        suggestion_prompt = ChatPromptTemplate.from_template("""
Suggest 3 follow-up questions.

Q: {q}
A: {a}

Return bullet points only.
""")

        suggestion_chain = suggestion_prompt | llm | StrOutputParser()

        suggestions = suggestion_chain.invoke({
            "q": user_input,
            "a": final_answer
        })

        st.markdown("### 💡 Suggested Questions")
        st.write(suggestions)

# ---------------------------
# FOLLOW-UP + FEEDBACK
# ---------------------------
if st.session_state.question_asked:

    st.markdown("---")
    st.subheader("🙋 Have more questions or feedback?")

    follow_up = st.text_input("Ask follow-up question")

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
            "question": sanitize_text(follow_up)
        })

        final_follow = simplify_medical_terms(sanitize_text(follow_answer))

        st.write("🤖", final_follow)

        st.session_state.chat_history.append({
            "q": follow_up,
            "a": final_follow
        })

    feedback = st.text_area("💬 Share your feedback")

    if st.button("Submit Feedback") and feedback:

        sentiment_prompt = ChatPromptTemplate.from_template("""
Classify sentiment: Positive, Neutral, Negative.

Feedback:
{text}
""")

        sentiment_chain = sentiment_prompt | llm | StrOutputParser()

        sentiment = sentiment_chain.invoke({
            "text": sanitize_text(feedback)
        }).strip().lower()

        st.success("Thanks for feedback!")

        if "negative" in sentiment:
            st.error("⚠️ Negative feedback detected")

        elif "positive" in sentiment:
            st.success("😊 Positive feedback")

        else:
            st.info("😐 Neutral feedback")

# ---------------------------
# PDF EXPORT
# ---------------------------
def generate_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [Paragraph("Patient Report", styles["Title"]), Spacer(1, 10)]

    for line in report.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    doc.build(content)
    buffer.seek(0)
    return buffer


if st.session_state.report:
    pdf = generate_pdf(st.session_state.report)
    st.download_button("📥 Download PDF", pdf, file_name="report.pdf")

# ---------------------------
# STEP 2: SCROLL TO TOP
# ---------------------------
st.markdown("""
<style>
.scroll-top-btn {
    position: fixed;
    bottom: 40px;
    right: 30px;
    background: #e0e0e0;
    color: black;
    padding: 12px 18px;
    border-radius: 10px;
    text-decoration: none;
    z-index: 100;
}
</style>

<a href="#top" class="scroll-top-btn">⬆️ Top</a>
""", unsafe_allow_html=True)