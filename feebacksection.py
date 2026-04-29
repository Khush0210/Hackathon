# ---------------------------
# FOLLOW-UP + FEEDBACK
# ---------------------------
st.markdown("---")
st.subheader("🙋 Have more questions or feedback?")

# Follow-up question
follow_up = st.text_input("Ask any further question (optional)")

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

    st.write("🤖", simplify_medical_terms(sanitize_text(follow_answer)))

# ---------------------------
# FEEDBACK SECTION
# ---------------------------
feedback = st.text_area("💬 Share your feedback (optional)")

if st.button("Submit Feedback") and feedback:
    st.success("✅ Thank you for your feedback!")