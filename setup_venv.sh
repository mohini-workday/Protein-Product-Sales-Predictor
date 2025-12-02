#!/bin/bash

# Navigate to project directory
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject"

# Create virtual environment if it doesn't exist
if [ ! -d "protein_env" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv protein_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source protein_env/bin/activate

# Check Python version
echo "Python version:"
python --version

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements_protein.txt --quiet
echo "Dependencies installed!"
echo ""

# Check if models need to be trained
echo ""
echo "======================================"
echo " Training Models"
echo "======================================"
echo ""

MODELS_DIR="ml_outputs"
MODELS_EXIST=true

if [ ! -f "$MODELS_DIR/rf_reg.pkl" ] || [ ! -f "$MODELS_DIR/xgb_reg.pkl" ] || [ ! -f "$MODELS_DIR/rf_clf.pkl" ] || [ ! -f "$MODELS_DIR/scaler.pkl" ]; then
    MODELS_EXIST=false
fi

if [ "$MODELS_EXIST" = false ]; then
    echo "Models not found. Training all three models..."
    echo "This may take several minutes..."
    echo ""
    
    # Execute the notebook to train models
    python -m jupyter nbconvert --to notebook --execute ProteinData.ipynb --output ProteinData_executed.ipynb --ExecutePreprocessor.timeout=3600
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Models trained successfully!"
        echo ""
        echo "Converting models to .pkl format for Streamlit..."
        python save_models_as_pkl.py
        echo ""
    else
        echo ""
        echo "⚠️  Warning: Model training encountered issues. Continuing anyway..."
        echo ""
    fi
else
    echo "✅ Models already exist. Skipping training."
    echo "To retrain models, delete .pkl files from ml_outputs/ directory."
    echo ""
fi

echo ""
echo "======================================"
echo " Setup complete!"
echo "======================================"
echo ""
echo " Starting Streamlit application..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Streamlit..."
    kill $STREAMLIT_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Start Streamlit server in background
echo "Starting Streamlit UI "
protein_env/bin/streamlit run Streamlit.py --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

echo ""
echo "======================================"
echo "Application is running!"
echo "======================================"
echo ""
echo "📱 Streamlit UI: http://localhost:8504"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for the process
wait