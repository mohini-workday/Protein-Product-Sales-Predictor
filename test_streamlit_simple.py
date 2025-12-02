# Simple test to verify Streamlit is working
import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Streamlit Test Page")
st.write("If you see this, Streamlit is working correctly!")
st.success("✅ Streamlit is rendering properly")
st.info("This is a test page to verify Streamlit functionality")

st.markdown("---")
st.header("Test Components")
st.button("Test Button")
st.slider("Test Slider", 0, 100, 50)
st.text_input("Test Input", "Type something here")

st.markdown("---")
st.write("**If you can see all of the above, Streamlit is working!**")

