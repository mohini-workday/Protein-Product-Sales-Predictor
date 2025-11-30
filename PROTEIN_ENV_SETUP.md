# ProteinData.ipynb Virtual Environment Setup

## ✅ Environment Created Successfully!

A new Python virtual environment has been created at:
```
/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject/protein_env
```

## 📦 Installed Libraries

All required libraries for `ProteinData.ipynb` have been installed:
- ✅ numpy, pandas, matplotlib, seaborn
- ✅ Pillow (PIL), scikit-image, opencv-python-headless
- ✅ scikit-learn, xgboost, tensorflow, keras
- ✅ shap, joblib, openpyxl
- ✅ ipykernel, jupyter (for notebook support)

## 🔄 How to Switch Python Kernel in Jupyter

### Option 1: In Jupyter Notebook/Lab (Recommended)
1. Open `ProteinData.ipynb` in Jupyter Notebook or JupyterLab
2. Click on the kernel name in the top-right corner (usually shows "Python 3" or similar)
3. Select **"Python (protein_env)"** from the dropdown menu
4. The kernel will switch and you're ready to run the notebook!

### Option 2: Using Command Line
If you prefer to activate the environment in terminal:
```bash
cd /Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/MainProject
source protein_env/bin/activate
jupyter notebook  # or jupyter lab
```

## 🧪 Testing the Environment

To verify everything works, you can run this in a Python cell:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import cv2
from PIL import Image

print("✓ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {__import__('sys').version}")
```

## 📝 Requirements File

A requirements file has been created at:
- `requirements_protein.txt`

To recreate this environment elsewhere:
```bash
python3.12 -m venv protein_env
source protein_env/bin/activate
pip install -r requirements_protein.txt
python -m ipykernel install --user --name=protein_env --display-name="Python (protein_env)"
```

## 🎯 Next Steps

1. Open `ProteinData.ipynb` in Jupyter
2. Switch kernel to "Python (protein_env)" (top-right dropdown)
3. Run the notebook cells - everything should work!

---

**Note:** This environment uses Python 3.12.8 (compatible with TensorFlow and all required packages).

