# Protein Product Sales Predictor

A machine learning application that predicts product sales based on visual features extracted from product label images.

## 📋 Project Overview

This project uses **Machine Learning** to analyze visual features of protein product labels and predict their sales performance. The system extracts 117 visual features including colors, textures, layout, typography, and deep learning embeddings to understand what visual elements drive sales.

## 🔑 Key Features

- **🎨 Visual Feature Extraction**
  - Color analysis (dominant colors, hue distribution)
  - Texture features (HOG, LBP, edge detection)
  - Layout metrics (aspect ratio, white space, logo prominence)
  - Typography analysis (text density, text regions)
  - Deep learning embeddings (ResNet50)

- **🤖 Machine Learning Models**
  - Regression Models: Ridge, Random Forest, XGBoost
  - Classification Models: Random Forest, XGBoost, Logistic Regression

- **📊 Analysis Tools**
  - Feature importance analysis
  - SHAP value explanations
  - Permutation importance
  - Interactive Streamlit dashboards (2 apps)
  - Model selection interface
  - Real-time image classification testing
  - Validation accuracy visualizations

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Create and activate virtual environment
python3.12 -m venv protein_env
source protein_env/bin/activate  # On macOS/Linux
# or
protein_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook

```bash
jupyter notebook ProteinData.ipynb
```

### 3. Run Streamlit Apps

The project includes two Streamlit applications:

**A. Visualization Dashboard** (`Streamlit_Dashboard.py`)
- Displays saved charts and visualizations from analysis
- Interactive data exploration
- Product label classification testing page
- Model selection and performance metrics

```bash
streamlit run Streamlit_Dashboard.py
```

**B. Sales Predictor** (`ProteinSalesPredictor.py`)
- Upload product images for sales prediction
- Real-time feature extraction
- Model selection (Ridge, Random Forest, XGBoost)
- SHAP analysis and feature importance
- Comprehensive performance metrics

```bash
streamlit run ProteinSalesPredictor.py
```

**Deployed App**: https://mohini-workday-protein-product-sales-streamlit-dashboard-hvldbs.streamlit.app/

**Or use the setup script:**

```bash
./setup_venv.sh
```

## 📁 Project Structure

```
MainProject/
├── ProteinData.ipynb              # Main analysis notebook
├── Streamlit_Dashboard.py         # Visualization dashboard app
├── ProteinSalesPredictor.py       # Sales prediction app
├── requirements.txt               # Python dependencies
├── setup_venv.sh                  # Setup script
├── save_models_as_pkl.py          # Model serialization script
├── ml_outputs/                    # Trained models and outputs
│   ├── rf_reg.pkl                 # Random Forest Regressor
│   ├── xgb_reg.pkl                # XGBoost Regressor
│   ├── rf_clf.pkl                 # Random Forest Classifier
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_table_with_metadata.csv
│   ├── merged_embeddings.csv
│   └── *.png                      # Visualization charts
├── ProteinProductImages/          # Product label images
├── ProteinProducts.xlsx           # Product metadata
├── ValidationAccuracy.png          # Model validation visualization
└── Documentation/
    ├── APP_INSTRUCTIONS.md         # Application usage guide
    ├── STREAMLIT_EXPLANATION.md    # Streamlit app details
    ├── FEATURE_VISUALIZATION_GUIDE.md
    ├── GRAPH_INTERPRETATION_GUIDE.md
    └── QUICK_GRAPH_SUMMARY.md
```

## 📊 Features Extracted

- **Basic Stats** (6): RGB mean and standard deviation
- **Color Features** (24): Dominant colors, coverage, hue histogram
- **Texture Features** (18): HOG, edge density, LBP patterns
- **Layout Features** (3): Aspect ratio, white space, logo score
- **Typography Features** (2): Text percentage, text regions
- **Deep Embeddings** (64): ResNet50 CNN features

**Total: ~117 features per image**

## 🎯 Model Performance

- **Random Forest Regressor**: Best R² score for sales prediction
- **XGBoost Regressor**: Excellent performance with gradient boosting
- **Ridge Regression**: Linear baseline model (auto-trained if needed)
- **Random Forest Classifier**: High accuracy for high/low sales classification
- **XGBoost Classifier**: Advanced classification performance
- **Logistic Regression**: Binary classification baseline

See `ValidationAccuracy.png` for detailed validation metrics.

## 📚 Documentation

- `APP_INSTRUCTIONS.md` - Detailed guide for using the Streamlit applications
- `STREAMLIT_EXPLANATION.md` - Technical details about the Streamlit apps
- `FEATURE_VISUALIZATION_GUIDE.md` - Guide to feature extraction visualizations
- `GRAPH_INTERPRETATION_GUIDE.md` - Detailed interpretation of all graphs
- `QUICK_GRAPH_SUMMARY.md` - Quick reference for graph conclusions

## 🔄 Recent Updates

- ✅ Added model selection feature to Testing page
- ✅ Enhanced error handling for missing model files
- ✅ Added scaler.pkl information display
- ✅ Fixed matplotlib import error for Streamlit Cloud deployment
- ✅ Added ValidationAccuracy.png visualization
- ✅ Improved dynamic path resolution for deployment
- ✅ Added comprehensive feature extraction pipeline

## 👤 Author

Mohini - ML PostGrad Main Project

## 📝 License

This project is part of academic research.

