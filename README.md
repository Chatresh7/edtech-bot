# 🎓 EduBot — EdTech Platform Course & Learning Workflow Explainer Bot
### Project 42 | Gemini Flash + FAISS RAG + Streamlit

---

## 🏗️ Project Structure

```
edtech_bot/
├── app.py                     ← Streamlit main app
├── requirements.txt
├── .env                       ← API key (create this)
├── components/
│   ├── __init__.py
│   ├── safety.py              ← Intent detection + safety filter
│   ├── retriever.py           ← FAISS RAG retriever
│   ├── prompts.py             ← Prompt config & builder
│   ├── gemini.py              ← Gemini Flash API wrapper
│   └── logger.py              ← Anonymized interaction logging
├── data/
│   └── knowledge_base.json    ← 50 KB articles
└── logs/
    └── interactions.jsonl     ← Auto-generated logs
```

---

## ⚙️ Setup Instructions

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API Key
Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```
Get your key from: https://aistudio.google.com/app/apikey

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🧠 LLM Selection

| Parameter        | Value                  | Reason                                        |
|------------------|------------------------|-----------------------------------------------|
| Model            | `gemini-2.0-flash`     | Fast (<2s), cost-efficient, long context      |
| Temperature      | `0.3`                  | Factual, consistent answers                   |
| Top-P            | `0.85`                 | Balanced sampling                             |
| Max Tokens       | `512`                  | Concise responses                             |
| Stop Sequences   | `["User:", "Human:"]`  | Prevent prompt injection                      |

---

## 📚 Knowledge Base (50 Statements)

| Category        | Count | Articles Cover                                      |
|-----------------|-------|-----------------------------------------------------|
| Course          | 14    | Enrollment, structure, video, forums, refund, notes |
| Assessment      | 12    | Quiz types, grading, attempts, plagiarism, proctoring|
| Certification   | 10    | Eligibility, issuance, sharing, verification, renewal|
| Progress        | 10    | Dashboard, completion, sync, streaks, activity log  |
| **Total**       | **50**|                                                     |

---

## 🔍 RAG Strategy

- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, fast, accurate)
- **Vector DB:** FAISS `IndexFlatIP` (cosine similarity)
- **Top-K:** 4 chunks per query
- **Category filter:** Intent-based pre-filtering
- **Confidence threshold:** 0.35 — below this, bot asks for clarification

---

## 🛡️ Safety System

**Blocked patterns (examples):**
- "solve question 3"
- "give me the correct answer"
- "answer this MCQ"
- "which option is correct"

**Post-response validation:** Scans LLM output for answer leakage before displaying.

---

## 📊 Test Cases

| Query                                  | Expected Behavior         |
|----------------------------------------|--------------------------|
| "How is a course structured?"          | Informational response    |
| "How are assessments conducted?"       | Policy explanation        |
| "Give me the answer to question 3"     | ❌ Blocked                |
| "What is the passing score?"           | Explains criteria         |
| "How do I get a certificate?"          | Step-by-step workflow     |
| "Solve this MCQ for me"               | ❌ Blocked                |
| "How does peer review work?"           | Process explanation       |

---

## 🔒 Non-Functional Requirements Met

| Requirement          | Implementation                              |
|----------------------|---------------------------------------------|
| < 2s response time   | Gemini Flash + pre-built FAISS index        |
| Retrieval < 200ms    | FAISS IndexFlatIP in-memory search          |
| Rate limit           | 10 req/min per session (in-app enforcement) |
| No PII storage       | SHA-256 hashed session IDs in logs          |
| GDPR alignment       | Anonymized logs, no raw queries stored      |
| Mobile responsive    | Streamlit native                            |
