
# RAG Model - Query Guide

## Running a Query

Follow these steps to query against the model:

### 1. Clone the Repository
```bash
git clone "https://github.com/ChiefDelphi-GPT/RAG_Model.git"
```

### 2. Navigate to the Project Directory
```bash
cd RAG_Model
```

### 3. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Navigate to JSON Parser
```bash
cd JsonParser
```

### 6. Run the Query Script
```bash
python3 user_input.py
```

### 7. Enter the Required Data

When prompted, enter the following information:

**Question:**
```
I am one of the programmers on an FRC team and I have a couple questions about how to use Advantage Scope. My team and I have no idea how to use it. Can you help me?
```

**Topic Slug:**
```
Advantage-Scope
```

The model will process your query and return relevant results based on the provided context.

