# Protein Sales Predictor - Application Instructions

## Overview

This Streamlit application allows you to:
1. **Load a product image** - Upload a protein product label image
2. **Select a model** - Choose from 3 trained regression models (Ridge, Random Forest, or XGBoost)
3. **View performance metrics** - See model performance, predictions, and validation metrics
4. **Explore visualizations** - View feature importance charts, actual vs predicted plots, and more
5. **SHAP analysis** - Understand how each feature contributes to the prediction
6. **Read conclusions** - Get a comprehensive summary of the analysis

## Prerequisites

1. **Models and Data**: Ensure you have run `ProteinData.ipynb` to generate:
   - `rf_reg.pkl` - Random Forest Regressor
   - `xgb_reg.pkl` - XGBoost Regressor
   - `scaler.pkl` - StandardScaler
   - `feature_table_with_metadata.csv` - Feature data
   - `ridge_reg.pkl` - Ridge Regressor (optional, will be trained automatically if missing)

2. **Python Environment**: Activate your virtual environment:
   ```bash
   source protein_env/bin/activate  # On macOS/Linux
   # or
   protein_env\Scripts\activate  # On Windows
   ```

3. **Dependencies**: Install required packages:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn shap xgboost tensorflow opencv-python-headless scikit-image pillow joblib
   ```

## Running the Application

1. **Navigate to project directory**:
   ```bash
   cd /Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject
   ```

2. **Run Streamlit app**:
   ```bash
   streamlit run ProteinSalesPredictor.py
   ```

3. **Open in browser**: The app will automatically open in your default browser at `http://localhost:8501`

## Application Workflow

### Step 1: Load Image
- Click "Choose an image file" button
- Select a product label image (PNG, JPG, or JPEG format)
- The app will automatically extract 117 visual features from the image

### Step 2: Select Model
- Choose one of three models:
  - **Ridge Regression**: Linear model with L2 regularization
  - **Random Forest Regressor**: Ensemble of decision trees
  - **XGBoost Regressor**: Gradient boosting model

### Step 3: View Performance
- **Predicted Sales**: The model's sales prediction for your image
- **Training Metrics**: R² score and RMSE on training data
- **Comparison Charts**: See how all three models compare
- **Actual vs Predicted**: Scatter plot showing model accuracy
- **Residual Analysis**: Check for prediction patterns

### Step 4: Feature Importance
- **Top Features**: Bar chart showing most important features
- **Category Analysis**: Feature importance by category (Color, Texture, Layout, etc.)
- **Feature vs Sales**: Scatter plots showing relationships

### Step 5: SHAP Analysis
- **Summary Plot**: Global feature importance across all predictions
- **Waterfall Plot**: Detailed breakdown for your specific image
- **Feature Contribution**: Bar chart showing how each feature affects the prediction

### Step 6: Conclusion
- **Analysis Summary**: What was done and how
- **Key Insights**: Top features driving the prediction
- **Interpretation**: What the results mean
- **Recommendations**: Actionable insights for design optimization

## Features Extracted

The application extracts 117 features from each image:

- **Basic Stats (6)**: RGB mean and standard deviation
- **Color Features (24)**: Dominant colors, coverage, hue distribution
- **Texture Features (18)**: HOG, edge density, LBP patterns
- **Layout Features (3)**: Aspect ratio, white space, logo score
- **Typography Features (2)**: Text coverage, text regions
- **Deep Learning Embeddings (64)**: ResNet50 CNN features

## Troubleshooting

### Models Not Loading
- Ensure `ProteinData.ipynb` has been executed
- Check that `.pkl` files exist in `ml_outputs/` directory
- Run `save_models_as_pkl.py` if needed

### Data Not Loading
- Verify `feature_table_with_metadata.csv` exists in `ml_outputs/`
- Check file permissions

### SHAP Analysis Errors
- Some model types may have limited SHAP support
- Try selecting a different model if SHAP fails
- Ensure SHAP library is installed: `pip install shap`

### ResNet50 Loading Issues
- First run may take time to download weights
- Ensure internet connection for initial download
- Model will be cached for subsequent runs

## File Structure

```
MainProject/
├── ProteinSalesPredictor.py          # Main Streamlit application
├── ProteinData.ipynb                 # Notebook with model training
├── ml_outputs/
│   ├── rf_reg.pkl                    # Random Forest model
│   ├── xgb_reg.pkl                   # XGBoost model
│   ├── scaler.pkl                    # Feature scaler
│   └── feature_table_with_metadata.csv  # Training data
└── ProteinProductImages/             # Sample product images
```

## Notes

- **Ridge Model**: If `ridge_reg.pkl` doesn't exist, the app will automatically train it on first use (cached for subsequent runs)
- **Performance**: Feature extraction may take 10-30 seconds depending on image size
- **SHAP Analysis**: May take 30-60 seconds to compute, especially for tree models
- **Browser**: Works best in Chrome, Firefox, or Safari

## Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify model files exist in `ml_outputs/`
3. Ensure virtual environment is activated
4. Check Streamlit logs for error messages

---

**Author**: Mohini  
**Project**: ML PostGrad - Main Project  
**Date**: 2024

