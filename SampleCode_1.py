import streamlit as st
from langchain_openai import ChatOpenAI
import httpx

# ==============================
# 🔐 TCS GenAI Config
# ==============================
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-YkM8lBAYyvLk36WC0ubseg",  # replace with your key
    http_client=client
)

# ==============================
# 🧠 SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are an expert Business Analyst AI.

- Identify unclear requirements
- Ask 1-2 clarification questions
- Suggest missing details

When clear:
- Provide structured refined requirements
"""

# ==============================
# UI
# ==============================
st.set_page_config(page_title="AI BA Assistant")
st.title("🤖 Requirement Clarification Bot")

# ==============================
# SESSION
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# SAMPLE INPUTS
# ==============================
samples = {
    "Select sample": "",
    "Basic": "Build a system for managing users and reports.",
    "HR": "We need an application for employees to apply for leave.",
    "Hospital": "We need a system to manage hospital patients and doctors.",
    "E-commerce": "Build an e-commerce platform with cart and payments.",
}

choice = st.selectbox("Choose sample", list(samples.keys()))

user_input = st.text_area(
    "Enter requirement:",
    value=samples.get(choice, "")
)

# ==============================
# LLM FUNCTION
# ==============================
def ask_ai(user_text):
    try:
        prompt = SYSTEM_PROMPT + "\n\nUser: " + user_text
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Error: {e}"

# ==============================
# ANALYZE
# ==============================
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter something")
    else:
        reply = ask_ai(user_input)

        st.session_state.messages.append(("User", user_input))
        st.session_state.messages.append(("AI", reply))

# ==============================
# CHAT DISPLAY
# ==============================
for role, msg in st.session_state.messages:
    if role == "User":
        st.write("🧑:", msg)
    else:
        st.write("🤖:", msg)

# ==============================
# FOLLOW-UP
# ==============================
follow = st.text_input("Your reply:")

if st.button("Send"):
    if follow:
        reply = ask_ai(follow)

        st.session_state.messages.append(("User", follow))
        st.session_state.messages.append(("AI", reply))

        st.rerun()

# ==============================
# RESET
# ==============================
if st.button("Reset"):
    st.session_state.messages = []
    st.rerun()