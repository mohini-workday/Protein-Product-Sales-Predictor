# How Streamlit Applications Work

## Important: Streamlit Doesn't Use a `main()` Function!

Unlike traditional Python scripts, **Streamlit applications don't have a `main()` function**. Instead:

### How Streamlit Executes Code

1. **Top-to-Bottom Execution**: Streamlit runs your entire script from **line 1 to the end** every time:
   - The page loads
   - A user interacts with a widget (button, slider, etc.)
   - The page refreshes

2. **Entry Point**: The entry point is simply **the first line of your Python file**

3. **Conditional Rendering**: You control what appears on screen using `if` statements based on user selections

## Structure of Streamlit.py

Looking at your `Streamlit.py` file:

```python
# Line 1-30: Imports
import streamlit as st
import pandas as pd
# ... other imports

# Line 32-86: Page configuration and CSS
st.set_page_config(...)
st.markdown("""<style>...</style>""")

# Line 88-255: Function definitions
def load_image(...):
    ...
def extract_all_features(...):
    ...

# Line 260-317: Load models and data (using @st.cache)
@st.cache_resource
def load_models():
    ...

# Line 340-361: Sidebar navigation
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🔮 Model Testing", ...]
)

# Line 373+: Conditional page rendering
if page == "🏠 Home":
    st.title("Protein Product Sales Predictor")
    # ... home page content

elif page == "🔮 Model Testing":
    st.header("🔮 Test Model with New Product Image")
    # ... model testing content

# ... more pages
```

## Execution Flow

```
START (Line 1)
    ↓
Import all libraries
    ↓
Set page configuration
    ↓
Define all functions (load_image, extract_features, etc.)
    ↓
Load models/data (cached - only runs once)
    ↓
Create sidebar navigation
    ↓
Check which page user selected
    ↓
Render the selected page content
    ↓
END (Last line)
```

## Key Points

1. **No `if __name__ == "__main__"` needed** - Streamlit handles this automatically

2. **Caching**: Use `@st.cache_resource` or `@st.cache_data` to prevent re-running expensive operations:
   ```python
   @st.cache_resource
   def load_models():
       # This only runs once, even if script re-executes
       return joblib.load('model.pkl')
   ```

3. **State Management**: Streamlit automatically tracks widget states (what user selected, uploaded, etc.)

4. **Re-execution**: Every interaction causes the script to re-run from the top, but cached functions skip execution

## Running the Application

Simply run:
```bash
streamlit run Streamlit.py
```

Streamlit will:
1. Execute the entire script
2. Create a web interface
3. Re-execute on user interactions
4. Handle all the web server, routing, and state management automatically

## Example: Adding a Main Function (Optional)

If you want to organize code with a main function (though not required), you could do:

```python
import streamlit as st

def main():
    st.title("My App")
    st.write("Hello World")

# Call it at the end
if __name__ == "__main__":
    main()
```

But this is **not necessary** - you can just put the code directly:

```python
import streamlit as st

st.title("My App")
st.write("Hello World")
```

Both work the same way in Streamlit!

## Your Files

- **Streamlit.py**: Multi-page application with sidebar navigation
- **ProteinSalesPredictor.py**: Single-page application with step-by-step workflow

Both work the same way - no main function needed!

