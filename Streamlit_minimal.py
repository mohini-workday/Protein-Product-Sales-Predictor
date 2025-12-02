"""
Minimal Streamlit test to verify the app works
"""
import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Streamlit Test Page")
st.success("If you see this, Streamlit is working!")

st.markdown("---")
st.header("Testing Model Loading")

# Test loading models
from pathlib import Path
import joblib

OUTPUT_DIR = Path("/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject/ml_outputs")

try:
    st.write("Attempting to load models...")
    rf_reg = joblib.load(OUTPUT_DIR / 'rf_reg.pkl')
    st.success("✅ rf_reg.pkl loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading rf_reg.pkl: {e}")

try:
    xgb_reg = joblib.load(OUTPUT_DIR / 'xgb_reg.pkl')
    st.success("✅ xgb_reg.pkl loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading xgb_reg.pkl: {e}")

try:
    rf_clf = joblib.load(OUTPUT_DIR / 'rf_clf.pkl')
    st.success("✅ rf_clf.pkl loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading rf_clf.pkl: {e}")

try:
    scaler = joblib.load(OUTPUT_DIR / 'scaler.pkl')
    st.success("✅ scaler.pkl loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading scaler.pkl: {e}")

try:
    import pandas as pd
    df = pd.read_csv(OUTPUT_DIR / 'feature_table_with_metadata.csv')
    st.success(f"✅ Data loaded: {len(df)} rows")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

st.markdown("---")
st.info("If all checks passed, the main app should work. If not, check the errors above.")

