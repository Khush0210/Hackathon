# ---------------------------
# FEEDBACK + SENTIMENT + ALERT
# ---------------------------
feedback = st.text_area("💬 Share your feedback (optional)")

if st.button("Submit Feedback") and feedback:

    # 🔹 Sentiment prompt
    sentiment_prompt = ChatPromptTemplate.from_template("""
Classify the sentiment of the following feedback as:
Positive, Neutral, or Negative.

Feedback:
{text}

Answer with only one word: Positive, Neutral, or Negative.
""")

    sentiment_chain = sentiment_prompt | llm | StrOutputParser()

    sentiment = sentiment_chain.invoke({
        "text": sanitize_text(feedback)
    }).strip().lower()

    st.success("✅ Thank you for your feedback!")

    # ---------------------------
    # 🎯 SENTIMENT DISPLAY
    # ---------------------------
    if "positive" in sentiment:
        st.markdown("📊 Sentiment: <span style='color:green'>Positive 😊</span>", unsafe_allow_html=True)

    elif "negative" in sentiment:
        st.markdown("📊 Sentiment: <span style='color:red'>Negative 😞</span>", unsafe_allow_html=True)

        # 🚨 NEGATIVE ALERT
        st.error("⚠️ We noticed a negative experience. Our team will review this.")

        # (Optional) Save negative feedback
        with open("negative_feedback.txt", "a", encoding="utf-8") as f:
            f.write(feedback + "\n---\n")

    else:
        st.markdown("📊 Sentiment: <span style='color:orange'>Neutral 😐</span>", unsafe_allow_html=True)
