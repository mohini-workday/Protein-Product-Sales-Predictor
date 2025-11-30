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
echo "Starting Streamlit UI on http://localhost:8501..."
protein_env/bin/streamlit run Streamlit.py --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

echo ""
echo "======================================"
echo "Application is running!"
echo "======================================"
echo ""
echo "📱 Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for the process
wait