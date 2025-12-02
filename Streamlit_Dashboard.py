# Streamlit Dashboard - Display Charts from ProteinData.ipynb
# No model loading - just visualizations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageStat
import warnings
import joblib
import io
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel
from matplotlib.colors import rgb_to_hsv

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Protein Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GUARANTEED INITIAL RENDER
# ============================================================================
st.title("📊 Protein Product Sales Analysis Dashboard")
st.markdown("---")
st.write("**Visualization Dashboard** - Charts and analysis from ProteinData.ipynb")

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================
def find_project_root(start_path=None):
    """Find the MainProject directory by walking up from start_path.
    Works in both local development and deployed environments (Streamlit Cloud).
    """
    if start_path is None:
        # Try to get the script's directory, fallback to current working directory
        try:
            start_path = Path(__file__).parent.resolve()
        except NameError:
            # __file__ not available (e.g., in some deployment environments like Streamlit Cloud)
            start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    # First, check if we're already in a directory with ml_outputs
    # This is the most reliable indicator and works in deployment
    test_path = start_path / "ml_outputs"
    if test_path.exists():
        return start_path
    
    # Check parent directories for ml_outputs (more reliable than directory name)
    current = start_path
    while current != current.parent:
        test_path = current / "ml_outputs"
        if test_path.exists():
            return current
        current = current.parent
    
    # Fallback: Check if directory is named "MainProject"
    current = start_path
    while current != current.parent:
        if current.name == "MainProject":
            return current
        current = current.parent
    
    # Final fallback: return the original start_path
    return start_path

# Get the MainProject root directory
PROJECT_DIR = find_project_root()
OUTPUT_DIR = PROJECT_DIR / "ml_outputs"

@st.cache_data
def load_data():
    """Load feature table data"""
    feature_file = OUTPUT_DIR / 'feature_table_with_metadata.csv'
    
    # Check if file exists first
    if not feature_file.exists():
        # Provide helpful debugging information
        debug_info = f"""
**File not found:** `{feature_file}`

**Debugging Information:**
- PROJECT_DIR: `{PROJECT_DIR}`
- OUTPUT_DIR: `{OUTPUT_DIR}`
- File path: `{feature_file}`
- OUTPUT_DIR exists: `{OUTPUT_DIR.exists()}`
- OUTPUT_DIR is directory: `{OUTPUT_DIR.is_dir() if OUTPUT_DIR.exists() else 'N/A'}`

**Solution:**
1. Ensure `feature_table_with_metadata.csv` exists in the `ml_outputs/` directory
2. Run `ProteinData.ipynb` to generate the file
3. Verify the file is committed to your repository (for deployed apps)
"""
        return None, False, debug_info
    
    try:
        df = pd.read_csv(feature_file)
        if df.empty:
            return None, False, "File exists but is empty"
        return df, True, None
    except Exception as e:
        return None, False, f"Error reading file: {str(e)}"

@st.cache_resource
def load_all_models():
    """Load all available models and scaler"""
    scaler_file = OUTPUT_DIR / 'scaler.pkl'
    
    models = {}
    missing_files = []
    
    # Check scaler first (required for all models)
    if not scaler_file.exists():
        missing_files.append(f"scaler.pkl (path: {scaler_file})")
    else:
        try:
            models['scaler'] = joblib.load(scaler_file)
        except Exception as e:
            return {}, False, f"Error loading scaler: {str(e)}"
    
    # Try to load each model
    model_files = {
        'rf_clf': OUTPUT_DIR / 'rf_clf.pkl',
        'rf_reg': OUTPUT_DIR / 'rf_reg.pkl',
        'xgb_reg': OUTPUT_DIR / 'xgb_reg.pkl'
    }
    
    for model_name, model_file in model_files.items():
        if model_file.exists():
            try:
                models[model_name] = joblib.load(model_file)
            except Exception as e:
                # Continue loading other models even if one fails
                missing_files.append(f"{model_name}.pkl (error: {str(e)})")
        else:
            missing_files.append(f"{model_name}.pkl (not found)")
    
    if not models.get('scaler'):
        debug_info = f"""
**Scaler file not found:**

Missing file: `scaler.pkl` (path: {scaler_file})

**Debugging Information:**
- PROJECT_DIR: `{PROJECT_DIR}`
- OUTPUT_DIR: `{OUTPUT_DIR}`
- OUTPUT_DIR exists: `{OUTPUT_DIR.exists()}`

**Solution:**
1. Run `ProteinData.ipynb` to generate the model files
2. Or run `save_models_as_pkl.py` to convert models from the notebook
3. Verify the files are committed to your repository (for deployed apps)
"""
        return {}, False, debug_info
    
    # Return models dict, success status, and any warnings
    warnings = None
    if missing_files and len(models) > 1:  # Only warn if scaler is loaded but some models are missing
        warnings = f"Some models not available: {', '.join(missing_files)}"
    
    return models, True, warnings

# Load data and models
data, data_loaded, data_error = load_data()
models, models_loaded, model_warnings = load_all_models()
scaler = models.get('scaler') if models_loaded else None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📈 Saved Charts", "📊 Interactive Charts", "📋 Data Explorer", "🧪 Testing"]
)

# Show data and model status
st.sidebar.markdown("---")
st.sidebar.write("**Status:**")
if data_loaded:
    st.sidebar.success(f"✅ Data loaded ({len(data)} products)")
else:
    st.sidebar.error("⚠️ Data not loaded")
    if data_error:
        st.sidebar.caption(f"{data_error[:50]}...")

if models_loaded:
    available_models = [name for name in ['rf_clf', 'rf_reg', 'xgb_reg'] if name in models]
    model_count = len(available_models)
    scaler_status = "✅" if scaler else "❌"
    st.sidebar.success(f"✅ {model_count} model(s) loaded")
    st.sidebar.info(f"{scaler_status} Scaler: {'Loaded' if scaler else 'Missing'}")
    if model_warnings:
        st.sidebar.warning(f"⚠️ {model_warnings[:40]}...")
else:
    st.sidebar.error("⚠️ Models not loaded")
    st.sidebar.caption("Check error details in Testing page")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.header("📋 Dashboard Overview")
    
    st.markdown("""
    This dashboard displays visualizations and charts created from the ProteinData.ipynb analysis.
    
    ### Available Sections:
    
    **📈 Saved Charts**: View the 6 comprehensive charts saved from the notebook:
    - Basic Image Statistics
    - Color Features Analysis
    - Texture Features Analysis
    - Layout & Logo Features
    - Typography Features
    - Comprehensive Summary
    
    **📊 Interactive Charts**: Explore the data with interactive Plotly visualizations
    
    **📋 Data Explorer**: Browse and filter the dataset
    """)
    
    if data_loaded:
        st.markdown("---")
        st.header("📁 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(data))
        
        with col2:
            feature_count = len([c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_'))])
            st.metric("Features Extracted", feature_count)
        
        with col3:
            if 'Sale' in data.columns:
                avg_sales = data['Sale'].mean()
                st.metric("Average Sales", f"${avg_sales:,.0f}")
            else:
                st.metric("Average Sales", "N/A")
        
        with col4:
            st.metric("Chart Files", "6")
    else:
        st.error("⚠️ Data not loaded")
        if data_error:
            with st.expander("🔍 View Error Details", expanded=True):
                st.markdown(data_error)
        st.info("💡 **Tip:** Run `ProteinData.ipynb` to generate `feature_table_with_metadata.csv` in the `ml_outputs/` directory.")

# ============================================================================
# PAGE 2: SAVED CHARTS
# ============================================================================
elif page == "📈 Saved Charts":
    st.header("📈 Saved Visualization Charts")
    st.markdown("These charts were generated from ProteinData.ipynb and saved as PNG files.")
    
    chart_files = {
        "1. Basic Image Statistics": "01_basic_image_statistics.png",
        "2. Color Features": "02_color_features.png",
        "3. Texture Features": "03_texture_features.png",
        "4. Layout & Logo Features": "04_layout_logo_features.png",
        "5. Typography Features": "05_typography_features.png",
        "6. Comprehensive Summary": "06_comprehensive_summary.png"
    }
    
    # Display all charts
    for chart_name, filename in chart_files.items():
        chart_path = OUTPUT_DIR / filename
        
        if chart_path.exists():
            st.subheader(chart_name)
            try:
                img = Image.open(chart_path)
                st.image(img, use_container_width=True, caption=chart_name)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
            st.markdown("---")
        else:
            st.warning(f"Chart file not found: {filename}")
            st.info(f"Expected location: {chart_path}")
            # Debug information (only shown in development)
            if st.sidebar.checkbox("Show debug info", key=f"debug_{filename}"):
                st.code(f"PROJECT_DIR: {PROJECT_DIR}\nOUTPUT_DIR: {OUTPUT_DIR}\nChart path: {chart_path}")

# ============================================================================
# PAGE 3: INTERACTIVE CHARTS
# ============================================================================
elif page == "📊 Interactive Charts":
    st.header("📊 Interactive Data Visualizations")
    
    if not data_loaded:
        st.error("⚠️ Data not loaded. Cannot display interactive charts.")
        if data_error:
            with st.expander("🔍 View Error Details"):
                st.markdown(data_error)
        st.info("💡 **Tip:** Run `ProteinData.ipynb` to generate `feature_table_with_metadata.csv` in the `ml_outputs/` directory.")
    else:
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Sales Distribution",
                "Feature vs Sales",
                "Feature Correlation Heatmap",
                "RGB Channel Analysis",
                "Feature Categories Overview"
            ]
        )
        
        st.markdown("---")
        
        if chart_type == "Sales Distribution":
            st.subheader("📊 Sales Distribution")
            if 'Sale' in data.columns:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    bins = st.slider("Number of Bins", 5, 50, 20)
                    fig = px.histogram(
                        data, 
                        x='Sale', 
                        nbins=bins,
                        title='Distribution of Product Sales',
                        labels={'Sale': 'Sales ($)', 'count': 'Number of Products'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Statistics")
                    st.metric("Mean", f"${data['Sale'].mean():,.0f}")
                    st.metric("Median", f"${data['Sale'].median():,.0f}")
                    st.metric("Std Dev", f"${data['Sale'].std():,.0f}")
                    st.metric("Min", f"${data['Sale'].min():,.0f}")
                    st.metric("Max", f"${data['Sale'].max():,.0f}")
            else:
                st.warning("Sales column not found in data")
        
        elif chart_type == "Feature vs Sales":
            st.subheader("📈 Feature vs Sales Relationships")
            if 'Sale' in data.columns:
                feature_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
                
                selected_feature = st.selectbox("Select Feature", feature_cols[:30])
                show_trendline = st.checkbox("Show Trendline", True)
                
                if selected_feature:
                    try:
                        fig = px.scatter(
                            data,
                            x=selected_feature,
                            y='Sale',
                            title=f'Sales vs {selected_feature}',
                            trendline="ols" if show_trendline else None,
                            labels={selected_feature: selected_feature, 'Sale': 'Sales ($)'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        corr = data[[selected_feature, 'Sale']].corr().iloc[0, 1]
                        st.metric("Correlation with Sales", f"{corr:.3f}")
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {e}")
            else:
                st.warning("Sales column not found in data")
        
        elif chart_type == "Feature Correlation Heatmap":
            st.subheader("🔥 Feature Correlation Heatmap")
            if 'Sale' in data.columns:
                feature_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
                
                selected_features = st.multiselect(
                    "Select Features (max 15 for performance)",
                    feature_cols,
                    default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols
                )
                
                if len(selected_features) > 0 and len(selected_features) <= 15:
                    try:
                        corr_matrix = data[selected_features + ['Sale']].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Feature Correlation Heatmap",
                            color_continuous_scale="RdBu"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {e}")
            else:
                st.warning("Sales column not found in data")
        
        elif chart_type == "RGB Channel Analysis":
            st.subheader("🎨 RGB Channel Analysis")
            if all(col in data.columns for col in ['mean_r', 'mean_g', 'mean_b']):
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    x_pos = np.arange(len(data))
                    width = 0.25
                    fig.add_trace(go.Bar(x=x_pos, y=data['mean_r'], name='Red', marker_color='#FF6B6B', width=width))
                    fig.add_trace(go.Bar(x=x_pos, y=data['mean_g'], name='Green', marker_color='#4ECDC4', width=width))
                    fig.add_trace(go.Bar(x=x_pos, y=data['mean_b'], name='Blue', marker_color='#45B7D1', width=width))
                    fig.update_layout(
                        title='RGB Mean Values Across Products',
                        xaxis_title='Product Index',
                        yaxis_title='Mean RGB Value (0-255)',
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data['mean_r'], name='Red', marker_color='#FF6B6B', opacity=0.6))
                    fig.add_trace(go.Histogram(x=data['mean_g'], name='Green', marker_color='#4ECDC4', opacity=0.6))
                    fig.add_trace(go.Histogram(x=data['mean_b'], name='Blue', marker_color='#45B7D1', opacity=0.6))
                    fig.update_layout(
                        title='Distribution of RGB Mean Values',
                        xaxis_title='Mean RGB Value',
                        yaxis_title='Frequency',
                        barmode='overlay',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("RGB columns (mean_r, mean_g, mean_b) not found in data")
        
        elif chart_type == "Feature Categories Overview":
            st.subheader("📊 Feature Categories Overview")
            if 'Sale' in data.columns:
                categories = {
                    'Color': [c for c in data.columns if c.startswith('color_feat_') or c.startswith('mean_') or c.startswith('std_')],
                    'Texture': [c for c in data.columns if c.startswith('lbp_') or 'edge_density' in c or 'hog_mean' in c],
                    'Layout': [c for c in data.columns if 'aspect_ratio' in c or 'white_pct' in c or 'logo_score' in c],
                    'Typography': [c for c in data.columns if 'text_pct' in c or 'text_cnts' in c],
                    'Embeddings': [c for c in data.columns if c.startswith('emb_')]
                }
                
                category_corrs = {}
                for cat, features in categories.items():
                    if features:
                        try:
                            corrs = [abs(data[[f, 'Sale']].corr().iloc[0, 1]) for f in features if f in data.columns]
                            category_corrs[cat] = np.mean(corrs) if corrs else 0
                        except:
                            category_corrs[cat] = 0
                
                if len(category_corrs) > 0:
                    fig = px.bar(
                        x=list(category_corrs.keys()),
                        y=list(category_corrs.values()),
                        title='Average Feature Importance by Category',
                        labels={'x': 'Feature Category', 'y': 'Average Absolute Correlation with Sales'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sales column not found in data")

# ============================================================================
# PAGE 4: DATA EXPLORER
# ============================================================================
elif page == "📋 Data Explorer":
    st.header("📋 Data Explorer")
    
    if not data_loaded:
        st.error("⚠️ Data not loaded.")
        if data_error:
            with st.expander("🔍 View Error Details"):
                st.markdown(data_error)
        st.info("💡 **Tip:** Run `ProteinData.ipynb` to generate `feature_table_with_metadata.csv` in the `ml_outputs/` directory.")
    else:
        st.markdown("Explore the dataset interactively. Filter, sort, and analyze the data.")
        
        # Filters
        st.subheader("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Sale' in data.columns:
                min_sales = st.number_input("Min Sales", value=float(data['Sale'].min()))
                max_sales = st.number_input("Max Sales", value=float(data['Sale'].max()))
            else:
                min_sales = max_sales = None
        
        with col2:
            if 'Product Name' in data.columns:
                product_filter = st.text_input("Filter by Product Name", "")
            else:
                product_filter = ""
        
        with col3:
            show_cols = st.multiselect(
                "Select Columns to Display",
                data.columns.tolist(),
                default=['Product Name', 'Sale'] if 'Product Name' in data.columns and 'Sale' in data.columns else data.columns[:5].tolist()
            )
        
        # Apply filters
        filtered_data = data.copy()
        
        if 'Sale' in data.columns and min_sales is not None and max_sales is not None:
            filtered_data = filtered_data[(filtered_data['Sale'] >= min_sales) & 
                                         (filtered_data['Sale'] <= max_sales)]
        
        if 'Product Name' in data.columns and product_filter:
            filtered_data = filtered_data[data['Product Name'].str.contains(product_filter, case=False, na=False)]
        
        # Display data
        st.subheader("📊 Filtered Data")
        st.dataframe(
            filtered_data[show_cols] if show_cols else filtered_data,
            use_container_width=True,
            height=400
        )
        
        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(filtered_data))
        with col2:
            if 'Sale' in filtered_data.columns:
                st.metric("Avg Sales", f"${filtered_data['Sale'].mean():,.0f}")
        with col3:
            if 'Sale' in filtered_data.columns:
                st.metric("Min Sales", f"${filtered_data['Sale'].min():,.0f}")
        with col4:
            if 'Sale' in filtered_data.columns:
                st.metric("Max Sales", f"${filtered_data['Sale'].max():,.0f}")

# ============================================================================
# PAGE 5: TESTING (Image Classification)
# ============================================================================
elif page == "🧪 Testing":
    st.header("🧪 Product Label Testing & Prediction")
    
    if not models_loaded or not scaler:
        st.error("⚠️ Models not loaded")
        st.info("💡 **Tip:** Run `ProteinData.ipynb` to generate the required model files (`rf_clf.pkl`, `rf_reg.pkl`, `xgb_reg.pkl`, and `scaler.pkl`).")
        st.info("📝 **For deployed apps:** Ensure the model files are committed to your repository and pushed to GitHub.")
    else:
        st.markdown("""
        Upload a product label image and select a model to predict sales performance.
        The application will extract visual features from the image and make a prediction.
        """)
        
        # Model selection
        available_models_list = []
        model_descriptions = {
            'rf_clf': 'Random Forest Classifier - Predicts High/Low Sales Category',
            'rf_reg': 'Random Forest Regressor - Predicts Sales Value',
            'xgb_reg': 'XGBoost Regressor - Predicts Sales Value (Advanced)'
        }
        
        for model_name in ['rf_clf', 'rf_reg', 'xgb_reg']:
            if model_name in models:
                available_models_list.append(model_name)
        
        if not available_models_list:
            st.error("⚠️ No models available. Please ensure model files exist in ml_outputs/")
            st.stop()
        
        # Model selection dropdown
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model_name = st.selectbox(
                "Select Model",
                options=available_models_list,
                format_func=lambda x: {
                    'rf_clf': '🎯 Random Forest Classifier (High/Low Sales)',
                    'rf_reg': '📊 Random Forest Regressor (Sales Value)',
                    'xgb_reg': '🚀 XGBoost Regressor (Sales Value)'
                }.get(x, x),
                help="Choose which model to use for prediction"
            )
        
        with col2:
            st.markdown("### Model Info")
            st.caption(model_descriptions.get(selected_model_name, "Model description"))
        
        selected_model = models[selected_model_name]
        is_classification = selected_model_name == 'rf_clf'
        
        st.markdown("---")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a product label image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a product label image for prediction"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Product Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Load and convert image
                    status_text.text("Loading image...")
                    progress_bar.progress(10)
                    img_arr = np.array(image.convert('RGB'))
                    
                    # Extract features
                    status_text.text("Extracting features...")
                    progress_bar.progress(30)
                    
                    # Feature extraction functions (simplified - no ResNet50)
                    def extract_features_simple(img_arr):
                        """Extract features from image (without ResNet50 for speed)"""
                        features = {}
                        
                        # Basic stats
                        img = Image.fromarray(img_arr.astype('uint8'))
                        stat = ImageStat.Stat(img)
                        mean = stat.mean
                        std = stat.stddev
                        features['mean_r'] = mean[0]
                        features['mean_g'] = mean[1]
                        features['mean_b'] = mean[2]
                        features['std_r'] = std[0]
                        features['std_g'] = std[1]
                        features['std_b'] = std[2]
                        
                        # Color features
                        h, w, _ = img_arr.shape
                        pixels = img_arr.reshape(-1, 3).astype(float)
                        sample_idx = np.random.choice(len(pixels), size=min(5000, len(pixels)), replace=False)
                        sample = pixels[sample_idx]
                        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sample)
                        centers = km.cluster_centers_.astype(int)
                        labels_full = km.labels_
                        counts = np.bincount(labels_full, minlength=3) / len(labels_full)
                        hsv = rgb_to_hsv(img_arr / 255.0)
                        hvals = (hsv[:, :, 0].ravel() * 360)
                        hue_hist, _ = np.histogram(hvals, bins=12, range=(0, 360), density=True)
                        
                        color_feats = centers.flatten().tolist() + counts.tolist() + hue_hist.tolist()
                        for i, v in enumerate(color_feats):
                            features[f'color_feat_{i}'] = float(v)
                        
                        # Texture features
                        gray = rgb2gray(img_arr)
                        hog_feat, _ = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), 
                                          visualize=True, feature_vector=True)
                        edges = sobel(gray)
                        edge_density = (edges > 0.02).mean()
                        lbp = local_binary_pattern(gray, P=8, R=1.0)
                        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**8 + 1), density=True)
                        
                        features['edge_density'] = edge_density
                        features['hog_mean'] = np.mean(hog_feat)
                        for i, v in enumerate(hist[:16]):
                            features[f'lbp_{i}'] = float(v)
                        
                        # Layout/Logo
                        h, w, _ = img_arr.shape
                        aspect_ratio = w / h
                        gray_cv = cv2.cvtColor(img_arr.astype('uint8'), cv2.COLOR_RGB2GRAY)
                        white_pct = np.mean(gray_cv > 245)
                        edges_cv = cv2.Canny(gray_cv, 100, 200)
                        logo_score = np.sum(edges_cv) / (h * w)
                        
                        features['aspect_ratio'] = aspect_ratio
                        features['white_pct'] = white_pct
                        features['logo_score'] = logo_score
                        
                        # Typography
                        th = cv2.adaptiveThreshold(gray_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
                        text_pct = np.mean(th > 0)
                        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        text_cnts = len(contours)
                        
                        features['text_pct'] = text_pct
                        features['text_cnts'] = text_cnts
                        
                        # Embeddings (set to zero - not using ResNet50 for speed)
                        for i in range(64):
                            features[f'emb_{i}'] = 0.0
                        
                        return features
                    
                    features_dict = extract_features_simple(img_arr)
                    progress_bar.progress(70)
                    status_text.text("Preparing features for model...")
                    
                    # Prepare features in correct order
                    if data_loaded:
                        keep_cols = [c for c in data.columns if c.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
                    else:
                        # Fallback: use feature names from the dict
                        keep_cols = [k for k in features_dict.keys() if k.startswith(('mean_','std_','color_feat_','lbp_','emb_','edge_density','hog_mean','aspect_ratio','white_pct','logo_score','text_pct','text_cnts'))]
                    
                    # Create feature DataFrame
                    feature_df = pd.DataFrame([features_dict])
                    keep_cols = [c for c in keep_cols if c in feature_df.columns]
                    X_new = feature_df[keep_cols].fillna(0)
                    
                    # Scale features
                    X_scaled = scaler.transform(X_new)
                    
                    progress_bar.progress(90)
                    status_text.text(f"Running {selected_model_name} prediction...")
                    
                    # Make prediction based on model type
                    if is_classification:
                        prediction = selected_model.predict(X_scaled)[0]
                        probabilities = selected_model.predict_proba(X_scaled)[0]
                    else:
                        prediction = selected_model.predict(X_scaled)[0]
                        probabilities = None
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    if is_classification:
                        # Classification results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sales_category = "High Sales" if prediction == 1 else "Low Sales"
                            category_color = "🟢" if prediction == 1 else "🔴"
                            st.metric(
                                "Predicted Category",
                                f"{category_color} {sales_category}",
                                delta=f"{probabilities[prediction]*100:.1f}% confidence"
                            )
                        
                        with col2:
                            high_prob = probabilities[1] * 100
                            st.metric(
                                "High Sales Probability",
                                f"{high_prob:.1f}%"
                            )
                        
                        with col3:
                            low_prob = probabilities[0] * 100
                            st.metric(
                                "Low Sales Probability",
                                f"{low_prob:.1f}%"
                            )
                        
                        # Probability visualization
                        st.markdown("---")
                        st.subheader("📈 Prediction Confidence")
                        
                        prob_data = pd.DataFrame({
                            'Category': ['Low Sales', 'High Sales'],
                            'Probability': [probabilities[0] * 100, probabilities[1] * 100]
                        })
                        
                        fig = px.bar(
                            prob_data,
                            x='Category',
                            y='Probability',
                            title='Sales Category Probabilities',
                            color='Category',
                            color_discrete_map={'Low Sales': '#FF6B6B', 'High Sales': '#4ECDC4'},
                            text='Probability',
                            texttemplate='%{text:.1f}%'
                        )
                        fig.update_layout(height=400, yaxis_title='Probability (%)', yaxis_range=[0, 100])
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Regression results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Predicted Sales",
                                f"${prediction:,.0f}",
                                help="Predicted sales value based on image features"
                            )
                        
                        with col2:
                            # Show model name
                            model_display_name = {
                                'rf_reg': 'Random Forest Regressor',
                                'xgb_reg': 'XGBoost Regressor'
                            }.get(selected_model_name, selected_model_name)
                            st.metric(
                                "Model Used",
                                model_display_name
                            )
                        
                        with col3:
                            # Show if prediction is above/below average (if data is loaded)
                            if data_loaded and 'Sale' in data.columns:
                                avg_sales = data['Sale'].mean()
                                diff = prediction - avg_sales
                                diff_pct = (diff / avg_sales * 100) if avg_sales > 0 else 0
                                st.metric(
                                    "vs Average",
                                    f"{diff_pct:+.1f}%",
                                    delta=f"${diff:+,.0f}"
                                )
                        
                        # Sales value visualization
                        st.markdown("---")
                        st.subheader("📈 Sales Prediction")
                        
                        # Create a comparison chart if data is available
                        if data_loaded and 'Sale' in data.columns:
                            comparison_data = pd.DataFrame({
                                'Type': ['Predicted', 'Average (Training Data)'],
                                'Sales': [prediction, data['Sale'].mean()]
                            })
                            
                            fig = px.bar(
                                comparison_data,
                                x='Type',
                                y='Sales',
                                title='Predicted Sales vs Average Training Sales',
                                color='Type',
                                color_discrete_map={
                                    'Predicted': '#4ECDC4',
                                    'Average (Training Data)': '#95A5A6'
                                },
                                text='Sales',
                                texttemplate='$%{text:,.0f}'
                            )
                            fig.update_layout(height=400, yaxis_title='Sales ($)')
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Simple display if no comparison data
                            st.info(f"**Predicted Sales Value:** ${prediction:,.2f}")
                    
                    # Feature summary
                    st.markdown("---")
                    st.subheader("🔍 Feature Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("📋 View Raw Features"):
                            st.json({k: round(v, 4) for k, v in list(features_dict.items())[:20]})
                            st.caption(f"Total features extracted: {len(features_dict)}")
                    
                    with col2:
                        with st.expander("📊 View Scaled Features (After Scaler)"):
                            # Show scaled features
                            scaled_features_dict = {keep_cols[i]: round(X_scaled[0][i], 4) for i in range(min(20, len(keep_cols)))}
                            st.json(scaled_features_dict)
                            st.caption(f"Features scaled using StandardScaler (scaler.pkl)")
                            st.info("💡 Features are normalized (mean=0, std=1) before model prediction")
                    
                    # Scaler information
                    st.markdown("---")
                    with st.expander("⚙️ Scaler Information"):
                        st.markdown("""
                        **StandardScaler (scaler.pkl)**
                        
                        The scaler normalizes features before feeding them to the models:
                        - **Mean normalization**: Centers features around 0
                        - **Standard deviation scaling**: Scales features to unit variance
                        - **Formula**: `scaled = (value - mean) / std`
                        
                        This ensures all features are on the same scale, which is important for:
                        - Linear models (Ridge, Logistic Regression)
                        - Neural networks
                        - Better model performance and convergence
                        """)
                        
                        if scaler:
                            st.success("✅ Scaler loaded successfully")
                            st.caption(f"Scaler type: {type(scaler).__name__}")
                            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                                st.caption(f"Number of features scaled: {len(scaler.mean_)}")
                        else:
                            st.error("❌ Scaler not available")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.exception(e)
                    progress_bar.progress(0)
                    status_text.text("❌ Error occurred")

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
**Protein Sales Analysis Dashboard**

Visualization dashboard displaying charts and analysis from ProteinData.ipynb

**Author**: Mohini  
**Project**: ML PostGrad - Main Project
""")

