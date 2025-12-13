# scripts/app.py
import os, torch, logging
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient
import numpy as np
from ollama import chat

logging.basicConfig(level=logging.INFO)

# Load Models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
IMG_MODEL = "openai/clip-vit-base-patch32"

logging.info("Loading text model...")
text_model = SentenceTransformer(TEXT_MODEL, device=DEVICE)

logging.info("Loading image model...")
clip_model = CLIPModel.from_pretrained(IMG_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(IMG_MODEL)

# Load ChromaDB collections
client = PersistentClient(path="../output_db")
text_col = client.get_collection("medical_text_collection")
img_col = client.get_collection("medical_image_collection")

def embed_text(query: str):
    return text_model.encode([query], convert_to_numpy=True)[0]

def embed_image(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.cpu().numpy()[0]

def retrieve(query_text=None, query_image=None, top_k=5):
    """Hybrid retrieval from text + image collections."""
    results = {"text": [], "images": []}
    if query_text:
        text_emb = embed_text(query_text)
        text_hits = text_col.query(query_embeddings=[text_emb.tolist()], n_results=top_k)
        results["text"] = list(zip(text_hits["metadatas"][0], text_hits["documents"][0]))

        # text->image search via CLIP text encoder
        inputs = clip_processor(text=query_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_feat = clip_model.get_text_features(**inputs)
        img_hits = img_col.query(query_embeddings=[text_feat.cpu().numpy()[0].tolist()], n_results=top_k)
        results["images"] = img_hits["metadatas"][0]

    elif query_image:
        img_emb = embed_image(query_image)
        img_hits = img_col.query(query_embeddings=[img_emb.tolist()], n_results=top_k)
        results["images"] = img_hits["metadatas"][0]

    return results

def reason_with_ollama(query, context):
    """Send structured prompt to IDEFICS2 via Ollama."""
    prompt = f"""
You are an expert AI medical assistant for qualified doctors.
Using ONLY the following trusted textbook context, answer comprehensively and clearly.

User Query:
{query}

Retrieved Context:
{context}
"""
    response = chat(model="idefics2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def run_pipeline(query_text=None, query_image=None):
    retrieved = retrieve(query_text, query_image)
    # Assemble context
    ctx_texts = "\n".join([m.get("section_title", "") + ": " + d[:500] for m, d in retrieved["text"]])
    answer = reason_with_ollama(query_text, ctx_texts)
    return answer, retrieved
