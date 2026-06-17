# Multi-Agent Fake News Verification System

A production-quality AI-powered fact-checking system using **6 specialized agents**, **LangGraph workflow**, and **FastAPI + Streamlit**.

## 🎯 What This System Does

Verifies factual claims by:
1. Normalizing the claim
2. Generating search queries
3. Retrieving evidence from web sources
4. Ranking evidence by relevance
5. Determining verdict with confidence
6. Generating human-readable explanations

**Example:**
- Input: "The Earth is flat"
- Output: REFUTE (98% confidence) + evidence + explanation

---

## 🚀 Quick Start

### Step 1: Setup (First Time Only)
```bash
cd "D:\Semester8\NLP\NLP PROJECT"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Step 2: Run Backend (Terminal 1)
```bash
cd "D:\Semester8\NLP\NLP PROJECT"
venv\Scripts\activate.bat
python -m backend.main
```

### Step 3: Run Frontend (Terminal 2)
```bash
cd "D:\Semester8\NLP\NLP PROJECT"
venv\Scripts\activate.bat
streamlit run frontend/app.py
```

### Step 4: Access
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 🏗️ Architecture

### 6 Specialized Agents:

```
User Claim → Claim Agent → Query Agent → Retrieval Agent 
  → Ranking Agent → Reasoning Agent → Explanation Agent → Results
```

1. **Claim Agent** - Normalizes and cleans claims
2. **Query Agent** - Generates 7-10 search queries
3. **Retrieval Agent** - Searches Tavily + Wikipedia APIs
4. **Ranking Agent** - Ranks evidence using FAISS vector search
5. **Reasoning Agent** - Determines verdict (SUPPORT/REFUTE/NOT_ENOUGH_INFO)
6. **Explanation Agent** - Generates human-readable explanations

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Agent Orchestration | LangGraph |
| LLM | Gemini API / OpenAI |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| Search APIs | Tavily + Wikipedia |
| Frontend | Streamlit |
| Visualization | Plotly |

---

## 📁 Project Structure

```
NLP PROJECT/
├── venv/                    # Virtual environment
├── backend/
│   ├── agents/              # 6 specialized agents
│   │   ├── claim_agent.py
│   │   ├── query_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── ranking_agent.py
│   │   ├── reasoning_agent.py
│   │   └── explanation_agent.py
│   ├── graph/               # LangGraph workflow
│   │   ├── workflow.py
│   │   └── state.py
│   ├── services/            # Core services
│   │   ├── llm_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── ranking_service.py
│   └── main.py              # FastAPI application
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── data/
│   └── sample_claims.json   # Example claims
│
├── .env                     # API keys (configured)
├── .gitignore               # Git exclusions
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔑 Configuration

Your `.env` file contains:
```env
GEMINI_API_KEY=your_key          # Gemini AI
OPENAI_API_KEY=your_key          # OpenAI (optional)
TAVILY_API_KEY=your_key          # Web search
LLM_PROVIDER=gemini              # Which LLM to use
```

To switch to OpenAI: Change `LLM_PROVIDER=openai`

---

## 🧪 Example Test Claims

Try these in the frontend:

### Should SUPPORT:
- "Water boils at 100 degrees Celsius at sea level"
- "Mount Everest is the tallest mountain on Earth"

### Should REFUTE:
- "The Earth is flat"
- "COVID-19 vaccines contain microchips"
- "Humans only use 10% of their brains"

### Should be NOT_ENOUGH_INFO:
- "Aliens visited Earth in 1950"
- "The Loch Ness Monster exists"

---

## 🐛 Troubleshooting

### ModuleNotFoundError
```bash
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Find process on port 8000
netstat -ano | findstr :8000
# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Reinstall Everything
```bash
# Delete venv folder manually, then run:
cd "D:\Semester8\NLP\NLP PROJECT"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## 📊 System Performance

- **Processing Time:** 10-15 seconds per claim
- **LLM Calls:** 5 per verification
- **Search Calls:** 14-20 per verification
- **Evidence Retrieved:** 20-30 documents
- **Top Evidence Used:** 5 most relevant

---

## 🎨 Frontend Features

- ✅ Clean and intuitive interface
- ✅ Real-time verification
- ✅ Confidence gauge visualization
- ✅ Evidence cards with sources
- ✅ Detailed reasoning panel
- ✅ Agent execution trace
- ✅ Source attribution with URLs

---

## 📄 License

This project is for educational purposes.

---

**Version:** 1.0.0  
**Status:** ✅ Production Ready
