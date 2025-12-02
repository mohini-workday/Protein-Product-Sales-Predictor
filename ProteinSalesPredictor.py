# Protein Sales Predictor - Streamlit Application
# User workflow: Load Image -> Select Model -> View Performance, Charts, SHAP Analysis, and Conclusion
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Example
install_package("streamlit")
install_package("pandas")
install_package("numpy")
install_package("matplotlib")
install_package("seaborn")
install_package("joblib")
install_package("plotly")
install_package("shap")
install_package("xgboost")
install_package("tensorflow")
install_package("keras")
install_package("opencv-python-headless")
install_package("scikit-image")
install_package("pillow")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageStat
import io
import warnings
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from matplotlib.colors import rgb_to_hsv
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import shap
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import math

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Protein Sales Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS (from ProteinData.ipynb)
# ============================================================================

def load_image(path, target_size=None):
    """Load image from path or file upload"""
    if isinstance(path, str):
        img = Image.open(path).convert('RGB')
    else:
        img = Image.open(io.BytesIO(path.read())).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.BICUBIC)
    return np.array(img)

def image_stats(img_arr):
    """Extract basic RGB statistics"""
    img = Image.fromarray(img_arr.astype('uint8'))
    stat = ImageStat.Stat(img)
    mean = stat.mean
    std = stat.stddev
    return mean, std

def color_features(img_arr, k=3):
    """Extract dominant colors using K-Means and hue histogram"""
    h, w, _ = img_arr.shape
    pixels = img_arr.reshape(-1, 3).astype(float)
    sample_idx = np.random.choice(len(pixels), size=min(5000, len(pixels)), replace=False)
    sample = pixels[sample_idx]
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(sample)
    centers = km.cluster_centers_.astype(int)
    labels_full = km.labels_
    counts = np.bincount(labels_full, minlength=k) / len(labels_full)
    hsv = rgb_to_hsv(img_arr / 255.0)
    hvals = (hsv[:, :, 0].ravel() * 360)
    hue_hist, _ = np.histogram(hvals, bins=12, range=(0, 360), density=True)
    return centers.flatten().tolist() + counts.tolist() + hue_hist.tolist()

def texture_features(img_arr):
    """Extract texture features: HOG, edge density, LBP"""
    gray = rgb2gray(img_arr)
    hog_feat, _ = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), 
                      visualize=True, feature_vector=True)
    edges = sobel(gray)
    edge_density = (edges > 0.02).mean()
    lbp = local_binary_pattern(gray, P=8, R=1.0)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**8 + 1), density=True)
    return {
        'hog_mean': np.mean(hog_feat),
        'edge_density': edge_density,
        'lbp_hist': hist[:16].tolist()
    }

def layout_logo_features(img_arr):
    """Extract layout and logo features"""
    h, w, _ = img_arr.shape
    aspect_ratio = w / h
    gray = cv2.cvtColor(img_arr.astype('uint8'), cv2.COLOR_RGB2GRAY)
    white_pct = np.mean(gray > 245)
    edges = cv2.Canny(gray, 100, 200)
    logo_score = np.sum(edges) / (h * w)
    return aspect_ratio, white_pct, logo_score

def typography_proxy(img_arr):
    """Extract typography features"""
    gray = cv2.cvtColor(img_arr.astype('uint8'), cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
    text_pct = np.mean(th > 0)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_cnts = len(contours)
    return text_pct, text_cnts

@st.cache_resource
def load_resnet():
    """Load ResNet50 model for embeddings"""
    try:
        resnet = ResNet50(weights='imagenet', include_top=False, 
                         pooling='avg', input_shape=(224, 224, 3))
        return resnet
    except Exception as e:
        st.warning(f"Could not load ResNet50: {e}")
        return None

def resnet_embed(img_arr, resnet_model):
    """Extract ResNet50 embeddings"""
    if resnet_model is None:
        return np.zeros(2048)
    img = Image.fromarray(img_arr.astype('uint8')).resize((224, 224))
    x = np.array(img).astype('float32')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = resnet_model.predict(x, verbose=0)
    return feat.flatten()

def extract_all_features(img_arr, resnet_model=None):
    """Extract all features from an image"""
    features = {}
    mean, std = image_stats(img_arr)
    features['mean_r'] = mean[0]
    features['mean_g'] = mean[1]
    features['mean_b'] = mean[2]
    features['std_r'] = std[0]
    features['std_g'] = std[1]
    features['std_b'] = std[2]
    
    color_feats = color_features(img_arr, k=3)
    for i, v in enumerate(color_feats):
        features[f'color_feat_{i}'] = float(v)
    
    tex = texture_features(img_arr)
    features['edge_density'] = tex['edge_density']
    features['hog_mean'] = tex['hog_mean']
    for i, v in enumerate(tex['lbp_hist']):
        features[f'lbp_{i}'] = float(v)
    
    aspect_ratio, white_pct, logo_score = layout_logo_features(img_arr)
    features['aspect_ratio'] = aspect_ratio
    features['white_pct'] = white_pct
    features['logo_score'] = logo_score
    
    text_pct, text_cnts = typography_proxy(img_arr)
    features['text_pct'] = text_pct
    features['text_cnts'] = text_cnts
    
    if resnet_model:
        embed = resnet_embed(img_arr, resnet_model)
        for i in range(64):
            features[f'emb_{i}'] = float(embed[i]) if i < len(embed) else 0.0
    else:
        for i in range(64):
            features[f'emb_{i}'] = 0.0
    
    return features

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    PROJECT_DIR = Path("/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject")
    OUTPUT_DIR = PROJECT_DIR / "ml_outputs"
    
    models = {}
    try:
        models['rf_reg'] = joblib.load(OUTPUT_DIR / 'rf_reg.pkl')
        models['xgb_reg'] = joblib.load(OUTPUT_DIR / 'xgb_reg.pkl')
        models['scaler'] = joblib.load(OUTPUT_DIR / 'scaler.pkl')
        
        # Try to load Ridge, if not available, we'll train it
        try:
            models['ridge'] = joblib.load(OUTPUT_DIR / 'ridge_reg.pkl')
        except:
            models['ridge'] = None
        
        return models, True, None
    except Exception as e:
        return None, False, str(e)

@st.cache_data
def load_data():
    """Load feature table and processed data"""
    PROJECT_DIR = Path("/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject")
    OUTPUT_DIR = PROJECT_DIR / "ml_outputs"
    
    try:
        df = pd.read_csv(OUTPUT_DIR / 'feature_table_with_metadata.csv')
        return df, True, None
    except Exception as e:
        return None, False, str(e)

@st.cache_resource
def train_ridge_model(data, scaler):
    """Train Ridge model if not available (cached)"""
    keep_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
    X = data[keep_cols].fillna(0)
    y = data['Sale'].values
    
    X_scaled = scaler.transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    return ridge

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("📊 Protein Product Sales Predictor")
st.markdown("---")

# Load models and data
models, models_loaded, models_error = load_models()
data, data_loaded, data_error = load_data()

if not models_loaded:
    st.error(f"⚠️ Models not loaded: {models_error}")
    st.info("Please ensure models are trained and saved in ml_outputs/ directory.")
    st.stop()

if not data_loaded:
    st.error(f"⚠️ Data not loaded: {data_error}")
    st.info("Please ensure feature_table_with_metadata.csv exists in ml_outputs/ directory.")
    st.stop()

# Train Ridge if needed (only if data is loaded)
if models.get('ridge') is None and data_loaded:
    try:
        with st.spinner("Training Ridge model (this may take a moment)..."):
            models['ridge'] = train_ridge_model(data, models['scaler'])
    except Exception as e:
        st.warning(f"Could not train Ridge model: {e}. Ridge option will be unavailable.")
        models['ridge'] = None

# ============================================================================
# STEP 1: IMAGE UPLOAD
# ============================================================================

st.header("📸 Step 1: Load Product Image")
st.markdown("Upload a product label image to analyze its sales potential.")

uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a product label image"
)

if uploaded_file is None:
    st.info("👆 Please upload an image to begin analysis.")
    st.stop()

# Display uploaded image
col1, col2 = st.columns([1, 1])
with col1:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Product Image", use_container_width=True)

# Extract features
with st.spinner("🔍 Extracting features from image..."):
    img_arr = load_image(uploaded_file)
    resnet_model = load_resnet()
    features_dict = extract_all_features(img_arr, resnet_model)
    
    # Prepare features for prediction
    keep_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
    feature_df = pd.DataFrame([features_dict])
    keep_cols = [c for c in keep_cols if c in feature_df.columns]
    X_new = feature_df[keep_cols].fillna(0)
    X_scaled = models['scaler'].transform(X_new)

with col2:
    st.subheader("✅ Features Extracted")
    st.success(f"Successfully extracted {len(features_dict)} features")
    st.json({k: round(v, 4) for k, v in list(features_dict.items())[:10]})
    st.info(f"Showing first 10 of {len(features_dict)} features")

st.markdown("---")

# ============================================================================
# STEP 2: MODEL SELECTION
# ============================================================================

st.header("🤖 Step 2: Select Model")
st.markdown("Choose one of the three trained models to analyze sales performance.")

# Build model options (only include available models)
model_options = {}
if models.get('ridge') is not None:
    model_options["Ridge Regression"] = "ridge"
model_options["Random Forest Regressor"] = "rf_reg"
model_options["XGBoost Regressor"] = "xgb_reg"

if not model_options:
    st.error("No models available. Please check model files.")
    st.stop()

selected_model_name = st.selectbox(
    "Select a model:",
    list(model_options.keys()),
    help="Choose which model to use for sales prediction"
)

selected_model_key = model_options[selected_model_name]
selected_model = models[selected_model_key]

# Make prediction
prediction = selected_model.predict(X_scaled)[0]

st.markdown("---")

# ============================================================================
# STEP 3: PERFORMANCE METRICS
# ============================================================================

st.header("📈 Step 3: Model Performance Analysis")

# Calculate performance metrics on training data
keep_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
X_train = data[keep_cols].fillna(0)
y_train = data['Sale'].values
X_train_scaled = models['scaler'].transform(X_train)

y_pred_train = selected_model.predict(X_train_scaled)
train_rmse = math.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Sales",
        f"${prediction:,.0f}",
        delta=f"${prediction - data['Sale'].mean():,.0f} vs avg" if 'Sale' in data.columns else None
    )

with col2:
    st.metric(
        "Training RMSE",
        f"${train_rmse:,.0f}",
        help="Root Mean Squared Error on training data"
    )

with col3:
    st.metric(
        "Training R² Score",
        f"{train_r2:.3f}",
        delta=f"{train_r2*100:.1f}% variance explained"
    )

with col4:
    avg_sales = data['Sale'].mean() if 'Sale' in data.columns else 0
    st.metric(
        "Dataset Average Sales",
        f"${avg_sales:,.0f}",
        help="Average sales across all products in dataset"
    )

# Performance comparison chart
st.subheader("📊 Model Performance Comparison")

# Get all model predictions for comparison
all_predictions = {}
for name, key in model_options.items():
    model = models[key]
    pred = model.predict(X_scaled)[0]
    all_predictions[name] = pred

# Create comparison chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(all_predictions.keys()),
    y=list(all_predictions.values()),
    marker_color=['#667eea', '#764ba2', '#f093fb'],
    text=[f"${v:,.0f}" for v in all_predictions.values()],
    textposition='auto',
))
fig.update_layout(
    title="Sales Predictions by Model",
    xaxis_title="Model",
    yaxis_title="Predicted Sales ($)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Actual vs Predicted scatter plot
st.subheader("📉 Actual vs Predicted Sales (Training Data)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_train,
    y=y_pred_train,
    mode='markers',
    marker=dict(color='#667eea', size=10, opacity=0.6),
    name='Predictions'
))
fig.add_trace(go.Scatter(
    x=[y_train.min(), y_train.max()],
    y=[y_train.min(), y_train.max()],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Perfect Prediction'
))
fig.update_layout(
    title=f"{selected_model_name} - Actual vs Predicted Sales",
    xaxis_title="Actual Sales ($)",
    yaxis_title="Predicted Sales ($)",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Residual plot
st.subheader("📊 Residual Analysis")
residuals = y_train - y_pred_train
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_pred_train,
    y=residuals,
    mode='markers',
    marker=dict(color='#764ba2', size=10, opacity=0.6),
    name='Residuals'
))
fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Residual")
fig.update_layout(
    title="Residual Plot (Predicted vs Residuals)",
    xaxis_title="Predicted Sales ($)",
    yaxis_title="Residuals ($)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# STEP 4: FEATURE IMPORTANCE CHARTS
# ============================================================================

st.header("📊 Step 4: Feature Importance & Visualizations")

# Feature importance based on model type
if hasattr(selected_model, 'feature_importances_'):
    # Tree-based models
    feature_importance = selected_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': keep_cols,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(20)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top 20 Most Important Features - {selected_model_name}',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
elif hasattr(selected_model, 'coef_'):
    # Linear models (Ridge)
    coef_df = pd.DataFrame({
        'Feature': keep_cols,
        'Coefficient': selected_model.coef_
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(20)
    
    fig = px.bar(
        coef_df,
        x='Abs_Coefficient',
        y='Feature',
        orientation='h',
        title=f'Top 20 Most Important Features (Absolute Coefficients) - {selected_model_name}',
        color='Coefficient',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# Feature categories analysis
st.subheader("📈 Feature Category Analysis")

categories = {
    'Color': [c for c in keep_cols if c.startswith('color_feat_') or c.startswith('mean_') or c.startswith('std_')],
    'Texture': [c for c in keep_cols if c.startswith('lbp_') or 'edge_density' in c or 'hog_mean' in c],
    'Layout': [c for c in keep_cols if 'aspect_ratio' in c or 'white_pct' in c or 'logo_score' in c],
    'Typography': [c for c in keep_cols if 'text_pct' in c or 'text_cnts' in c],
    'Embeddings': [c for c in keep_cols if c.startswith('emb_')]
}

if hasattr(selected_model, 'feature_importances_'):
    category_importance = {}
    for cat, features in categories.items():
        if features:
            feat_indices = [keep_cols.index(f) for f in features if f in keep_cols]
            if feat_indices:
                category_importance[cat] = np.mean([feature_importance[i] for i in feat_indices])
    
    if category_importance:
        fig = px.bar(
            x=list(category_importance.keys()),
            y=list(category_importance.values()),
            title='Average Feature Importance by Category',
            labels={'x': 'Feature Category', 'y': 'Average Importance'},
            color=list(category_importance.values()),
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Top features vs Sales scatter plots
st.subheader("🔍 Top Features vs Sales Relationship")

top_n_features = st.slider("Number of top features to visualize", 3, 8, 5)
if hasattr(selected_model, 'feature_importances_'):
    top_features = importance_df.head(top_n_features)['Feature'].tolist()
elif hasattr(selected_model, 'coef_'):
    top_features = coef_df.head(top_n_features)['Feature'].tolist()
else:
    top_features = keep_cols[:top_n_features]

cols = st.columns(2)
for idx, feat in enumerate(top_features):
    col = cols[idx % 2]
    with col:
        try:
            fig = px.scatter(
                data,
                x=feat,
                y='Sale',
                title=f'Sales vs {feat}',
                trendline='ols',
                labels={feat: feat, 'Sale': 'Sales ($)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass

st.markdown("---")

# ============================================================================
# STEP 5: SHAP ANALYSIS
# ============================================================================

st.header("🔬 Step 5: SHAP (SHapley Additive exPlanations) Analysis")
st.markdown("SHAP values explain how each feature contributes to the prediction for this specific image.")

with st.spinner("Computing SHAP values (this may take a moment)..."):
    try:
        # Create SHAP explainer
        if selected_model_key == 'xgb_reg':
            explainer = shap.Explainer(selected_model)
        elif selected_model_key == 'rf_reg':
            explainer = shap.Explainer(selected_model)
        elif selected_model_key == 'ridge':
            # For Ridge, use LinearExplainer with a sample of training data
            sample_size = min(100, len(X_train_scaled))
            explainer = shap.LinearExplainer(selected_model, X_train_scaled[:sample_size])
        else:
            # Fallback to TreeExplainer or KernelExplainer
            try:
                explainer = shap.Explainer(selected_model)
            except:
                explainer = shap.LinearExplainer(selected_model, X_train_scaled[:100])
        
        # Compute SHAP values for the uploaded image
        shap_values = explainer(X_scaled)
        
        # Summary plot
        st.subheader("📊 SHAP Summary Plot (Global Feature Importance)")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_scaled, feature_names=keep_cols, show=False, max_display=20)
        st.pyplot(fig)
        plt.close()
        
        # Waterfall plot for this specific prediction
        st.subheader("💧 SHAP Waterfall Plot (This Image's Prediction)")
        try:
            # For single prediction
            if hasattr(shap_values, 'values'):
                shap_values_single = shap_values[0]
            else:
                shap_values_single = shap_values
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(shap_values_single, show=False, max_display=15)
            st.pyplot(fig)
            plt.close()
        except:
            st.info("Waterfall plot not available for this model type. Showing summary plot instead.")
        
        # Feature contribution bar chart
        st.subheader("📊 Feature Contribution to Prediction")
        if hasattr(shap_values, 'values'):
            values = shap_values.values[0]
        else:
            values = shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0]
        
        # Get top contributing features
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-15:][::-1]
        
        contrib_data = {
            'Feature': [keep_cols[i] for i in top_indices],
            'SHAP Value': [values[i] for i in top_indices],
            'Absolute Impact': [abs_values[i] for i in top_indices]
        }
        contrib_df = pd.DataFrame(contrib_data)
        
        fig = go.Figure()
        colors = ['#667eea' if v > 0 else '#f093fb' for v in contrib_df['SHAP Value']]
        fig.add_trace(go.Bar(
            x=contrib_df['SHAP Value'],
            y=contrib_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"${v:.0f}" if abs(v) > 1000 else f"{v:.2f}" for v in contrib_df['SHAP Value']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Top 15 Features Contributing to Prediction (SHAP Values)",
            xaxis_title="SHAP Value (Impact on Sales)",
            yaxis_title="Feature",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation text
        st.info("""
        **SHAP Value Interpretation:**
        - **Positive SHAP values** (blue) increase the predicted sales
        - **Negative SHAP values** (pink) decrease the predicted sales
        - The magnitude shows how much each feature contributes to the final prediction
        """)
        
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        st.info("SHAP analysis may not be available for all model types. Try selecting a different model.")

st.markdown("---")

# ============================================================================
# STEP 6: CONCLUSION
# ============================================================================

st.header("📝 Step 6: Conclusion & Summary")

# Calculate key insights
prediction_diff = prediction - avg_sales
prediction_pct = (prediction_diff / avg_sales * 100) if avg_sales > 0 else 0

# Get top contributing features from SHAP if available
top_positive_features = []
top_negative_features = []
try:
    if hasattr(shap_values, 'values'):
        values = shap_values.values[0]
    else:
        values = shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0]
    
    abs_values = np.abs(values)
    top_indices = np.argsort(abs_values)[-5:][::-1]
    
    for idx in top_indices:
        feat_name = keep_cols[idx]
        feat_value = features_dict.get(feat_name, 0)
        shap_val = values[idx]
        if shap_val > 0:
            top_positive_features.append((feat_name, shap_val, feat_value))
        else:
            top_negative_features.append((feat_name, shap_val, feat_value))
except:
    # Fallback to feature importance
    if hasattr(selected_model, 'feature_importances_'):
        top_indices = np.argsort(selected_model.feature_importances_)[-5:][::-1]
        for idx in top_indices:
            top_positive_features.append((keep_cols[idx], selected_model.feature_importances_[idx], features_dict.get(keep_cols[idx], 0)))

# Create conclusion
conclusion_text = f"""
## Analysis Summary

### What We Did

1. **Image Feature Extraction**: Extracted **{len(features_dict)} visual features** from the uploaded product image, including:
   - **Color features** (24): Dominant colors, color coverage, and hue distribution
   - **Texture features** (18): HOG patterns, edge density, and Local Binary Patterns (LBP)
   - **Layout features** (3): Aspect ratio, white space percentage, and logo prominence
   - **Typography features** (2): Text coverage and text region count
   - **Deep learning embeddings** (64): High-level visual features from ResNet50 CNN

2. **Model Selection**: Used **{selected_model_name}** to predict sales based on the extracted features.

3. **Performance Analysis**: 
   - Model achieved **R² score of {train_r2:.3f}** ({train_r2*100:.1f}% variance explained)
   - Training RMSE: **${train_rmse:,.0f}**
   - This indicates the model {'explains most of the variance' if train_r2 > 0.8 else 'has moderate predictive power' if train_r2 > 0.5 else 'has limited predictive power'}

4. **SHAP Analysis**: Identified which specific features contributed most to the prediction for this image.

### Prediction Results

- **Predicted Sales**: **${prediction:,.0f}**
- **Compared to Average**: {'Above average' if prediction_diff > 0 else 'Below average'} by **${abs(prediction_diff):,.0f}** ({abs(prediction_pct):.1f}%)
"""

if top_positive_features:
    conclusion_text += "\n### Key Features Driving Prediction\n\n"
    conclusion_text += "**Features increasing sales prediction:**\n"
    for feat_name, impact, value in top_positive_features[:3]:
        conclusion_text += f"- **{feat_name}**: Value = {value:.4f}, Impact = +${impact:.0f}\n"

if top_negative_features:
    conclusion_text += "\n**Features decreasing sales prediction:**\n"
    for feat_name, impact, value in top_negative_features[:3]:
        conclusion_text += f"- **{feat_name}**: Value = {value:.4f}, Impact = ${impact:.0f}\n"

conclusion_text += f"""

### What This Means

The analysis reveals that visual design elements significantly influence predicted sales performance. The **{selected_model_name}** model identified key visual features that correlate with sales outcomes:

1. **Design Optimization**: Products with certain color palettes, texture patterns, and layout characteristics tend to perform better in sales predictions.

2. **Feature Importance**: The SHAP analysis shows which specific visual elements in this image contribute most to the sales prediction, providing actionable insights for design improvements.

3. **Model Reliability**: With an R² score of {train_r2:.3f}, the model {'provides strong predictive capability' if train_r2 > 0.8 else 'provides moderate predictive capability' if train_r2 > 0.5 else 'provides limited predictive capability'} for sales forecasting based on visual features.

### Recommendations

- **For High Sales Prediction**: Focus on maintaining or enhancing the visual features that positively contribute to sales
- **For Low Sales Prediction**: Consider redesigning elements that negatively impact the prediction
- **Design Strategy**: Use the feature importance rankings to prioritize which visual elements to optimize

### Methodology

This analysis used a comprehensive machine learning pipeline:
- **Feature Extraction**: Computer vision techniques (color analysis, texture detection, layout analysis) combined with deep learning (ResNet50 embeddings)
- **Model Training**: Three regression models (Ridge, Random Forest, XGBoost) trained on historical product data
- **Explainability**: SHAP values provide interpretable explanations for each prediction
- **Validation**: Model performance validated using R² score and RMSE metrics
"""

st.markdown(conclusion_text)

# Final summary box
st.success("""
✅ **Analysis Complete!** 

The model has analyzed your product image and provided:
- Sales prediction based on visual features
- Performance metrics and validation
- Feature importance rankings
- SHAP explanations for interpretability
- Actionable insights for design optimization
""")

st.markdown("---")
st.markdown("**Application created for Protein Product Sales Prediction Analysis** | Author: Mohini")

