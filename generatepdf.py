import streamlit as st
import re
import random
from pdfminer.high_level import extract_text
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.title("🩸 Blood Report Generator (Offline - No API)")

uploaded_file = st.file_uploader("Upload Blood Report PDF", type=["pdf"])

# ---------------------------
# FUNCTION: MODIFY VALUES
# ---------------------------
def modify_values(text):
    def change_number(match):
        num = float(match.group())
        
        # vary value by ±10%
        variation = random.uniform(-0.1, 0.1)
        new_num = num + (num * variation)
        
        return str(round(new_num, 2))

    # Replace numeric values
    modified_text = re.sub(r"\b\d+\.?\d*\b", change_number, text)

    return modified_text

# ---------------------------
# FUNCTION: CREATE PDF
# ---------------------------
def create_pdf(content, filename="modified_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    for line in content.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return filename

# ---------------------------
# MAIN FLOW
# ---------------------------
if uploaded_file is not None:
    
    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    text = extract_text("temp.pdf")

    st.subheader("📄 Original Report")
    st.text(text[:2000])

    # Modify report
    modified_text = modify_values(text)

    st.subheader("🧪 Generated Modified Report")
    st.text(modified_text[:2000])

    # Create PDF
    pdf_file = create_pdf(modified_text)

    # Download button
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Modified Report",
            data=f,
            file_name="modified_blood_report.pdf",
            mime="application/pdf"
        )