# AI Content Detector (Human vs AI Text) — 95% Accuracy

A web that able to detect AI-generated text using a **fine-tuned BERT model (Hugging Face + PyTorch)**.  

---

## Problem Statement
With the rise of generative AI tools, it is increasingly difficult to distinguish human vs AI-written text.  
This project builds an automated detector to support **academic integrity**, **content moderation**, and **authorship verification**.

---

## Results
- **Accuracy:** ~**95%**
- Output includes:
  - predicted label (Human / AI)
  - probability score

---

## Techniques Involved
- **Transformer-based NLP:** Fine-tuned **BERT** for binary text classification
- **Backend API:** **FastAPI** + **Uvicorn** (PyTorch inference)
- **Frontend UI:** HTML + JavaScript (`fetch`) for text/file detection
- **Preprocessing:** filtering short/empty texts, normalization

---

## Dataset
Kaggle: *LLM Detect AI Generated Text Dataset*  
https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset

--

## Project Workflow (Steps)
1) Dataset Preparation
- Clean data (remove empty/short text, normalize)
- Convert to Pandas / Hugging Face Dataset for training

2) Model Training (Hugging Face BERT)
- Use `bert-base-uncased` tokenizer & model
- Train & save checkpoint (e.g., `results/checkpoint-2915`)

3) Backend API (FastAPI + PyTorch)
- Enable CORS
- Load model + tokenizer from checkpoint
- Serve `/predict-text` and `/predict-file`

4) Frontend Interface (HTML + JavaScript)
- Paste text / upload file
- Show predicted label + probability bar

---

## How to Run (Local)

### 1) Backend (FastAPI)
```bash
python -m uvicorn AIContentDetector.app:app --reload --port 8000
