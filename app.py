from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch, os, re, io
from PyPDF2 import PdfReader
from docx import Document

# === Initialize FastAPI ===
# Create a FastAPI app with a custom title
app = FastAPI(title="AI vs Human Detector API")

# Enable CORS (Cross-Origin Resource Sharing) so that the frontend
# running in a browser can call this backend API without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow requests from all domains (frontend access)
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# === Load trained model checkpoint ===
MODEL_DIR = r"C:\Users\Shin Yee\PyCharmMiscProject\AIContentDetector\backend\results\checkpoint-2915"
print("📂 Loading model from", MODEL_DIR)

# Load fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()
print("✅ Model loaded successfully!")

# Define input schema for JSON requests
class TextRequest(BaseModel):
    text: str

# === Prediction helper function ===
def predict_proba(text: str) -> float:
    """
    Runs inference on the given text using the fine-tuned BERT model.
    Returns the probability that the text is AI-generated.
    """
    # Encode text into tokens for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
    return float(probs[1])  # index 1 = AI 概率

# === Text prediction endpoint ===
@app.post("/predict-text")
def predict_text(req: TextRequest):
    text = (req.text or "").strip()
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="Text too short. Please provide at least 20 characters.")
    proba = predict_proba(text)
    label = "AI" if proba >= 0.5 else "Human"
    return {"label": label, "prob_ai": round(proba, 4), "chars": len(text)}

# === File upload prediction endpoint ===
@app.post("/predict-file")
def predict_file(file: UploadFile = File(...)):
    fname = (file.filename or "").lower()
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file.")

    # Handle different file types
    if fname.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif fname.endswith(".docx"):
        bio = io.BytesIO(content)
        doc = Document(bio)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif fname.endswith(".pdf"):
        bio = io.BytesIO(content)
        reader = PdfReader(bio)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use .txt, .docx, or .pdf")

    text = text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="File has too little extractable text.")

    # Run prediction
    proba = predict_proba(text)
    label = "AI" if proba >= 0.5 else "Human"
    return {"label": label, "prob_ai": round(proba, 4), "chars": len(text), "filename": file.filename}
