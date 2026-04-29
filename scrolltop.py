# ✅ STEP 2: SCROLL TO TOP BUTTON (WORKING)
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
    z-index: 100;
}
.scroll-top-btn:hover {
    background-color: #45a049;
}
</style>

<a href="#top" class="scroll-top-btn">⬆️ Top</a>
""", unsafe_allow_html=True)