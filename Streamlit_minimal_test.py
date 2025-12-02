# Minimal Streamlit test - should always render
import streamlit as st

st.set_page_config(
    page_title="Minimal Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 MINIMAL TEST - If you see this, Streamlit works!")
st.write("This is a minimal test page.")
st.success("✅ Streamlit is working!")

# Test sidebar
st.sidebar.title("Test Sidebar")
st.sidebar.write("Sidebar is working too!")

st.markdown("---")
st.write("**If you can see all of this, the basic Streamlit setup is working correctly.**")

