# Project Setup Guide

This guide provides step-by-step instructions to set up and run the application.

---

## Prerequisites

Ensure you have the following installed:
- **Python** (>=3.8)
- **pip** (Python package manager)

---

## Setup Instructions

### 1. Create a Virtual Environment
Create a virtual environment for the project dependencies:
```python -m venv myenv```

### 2. Activate virtual environment
```
cd myenv\Scripts\
Activate.ps1
cd ../..
```

### 3. Install required packages
```pip install -r requirements.txt```

### 4. Install the package seperately in the virtual environment
```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

### 5. Generate the model first by running setup/generate_model.python, this will generate the model in the models directory(about 2GB in size)
```Python .\setup\generate_model.python```

### 6. Generate the vector embedding and store it as a csv (This step is not necessary as the text_chunks_and_embeddings_soros_df.csv is already generated)
```Python .\setup\generate_embedding.python```

### 7. Start the server 
```python .\server\server.py```

### 8. Start the client
```streamlit run .\chatbot_ui.py```


