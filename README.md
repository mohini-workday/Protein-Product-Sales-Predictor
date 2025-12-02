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
  - Interactive Streamlit dashboard

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Create and activate virtual environment
python3.12 -m venv protein_env
source protein_env/bin/activate  # On macOS/Linux
# or
protein_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements_protein.txt
```

### 2. Run Jupyter Notebook

```bash
jupyter notebook ProteinData.ipynb
```

### 3. Run Streamlit App

```bash
streamlit run Streamlit_Dashboard.py
```
Or Use the deployed app - https://mohini-workday-protein-product-sales-streamlit-dashboard-hvldbs.streamlit.app/

Or use the setup script:

```bash
./setup_venv.sh
```

## 📁 Project Structure

```
MainProject/
├── ProteinData.ipynb          # Main analysis notebook
├── Streamlit.py               # Interactive web application
├── requirements_protein.txt   # Python dependencies
├── setup_venv.sh             # Setup script
├── ml_outputs/               # Trained models and outputs
│   ├── rf_reg.py             # Random Forest Regressor
│   ├── xgb_reg.py            # XGBoost Regressor
│   ├── rf_clf.py             # Random Forest Classifier
│   ├── scaler.py             # Feature scaler
│   └── feature_table_with_metadata.csv
├── ProteinProductImages/      # Product label images
├── ProteinProducts.xlsx       # Product metadata
└── Documentation/
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

- **Random Forest Regressor**: Best R² score
- **XGBoost Regressor**: Excellent performance
- **Random Forest Classifier**: High accuracy for high/low sales prediction

## 📚 Documentation

- `FEATURE_VISUALIZATION_GUIDE.md` - Guide to feature extraction visualizations
- `GRAPH_INTERPRETATION_GUIDE.md` - Detailed interpretation of all graphs
- `QUICK_GRAPH_SUMMARY.md` - Quick reference for graph conclusions

## 👤 Author

Mohini - ML PostGrad Main Project

## 📝 License

This project is part of academic research.

