#!/usr/bin/env python3
"""
idefics_pipeline_v2.py
Final upgraded, robust IDEFICS2 multimodal retrieval+reasoning pipeline.

Features:
- Loads IDEFICS2 from a local HuggingFace snapshot (local_files_only=True)
- Uses BioBERT (SentenceTransformer) for text retrieval (768-dim)
- Uses OpenCLIP (ViT-B-32) for image retrieval (512-dim)
- Robust retrieval from ChromaDB collections: medical_text_collection & medical_image_collection
- Interactive, clinician-style follow-up questions to obtain patient context
- Structured prompt engineering and final output formatting
- Prints retrieved image file paths and metadata so you can inspect sources
- Graceful fallbacks and good logging
"""

import os
import sys
import time
import torch
from loguru import logger
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import chromadb
import open_clip
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True)

# IMPORTANT: point this at the exact snapshot folder that contains the safetensors shards
LOCAL_IDEFICS_SNAPSHOT = (
    "/teamspace/studios/this_studio/.cache/huggingface/hub/"
    "models--HuggingFaceM4--idefics2-8b/snapshots/2c42686c57fe21cf0348c9ce1077d094b72e7698"
)

# If you prefer to use a different local path, change above.
# Collections in ChromaDB:
TEXT_COLLECTION_NAME = "medical_text_collection"
IMAGE_COLLECTION_NAME = "medical_image_collection"

# Retrieval settings
TEXT_TOP_K = 6
IMAGE_TOP_K = 6

# Prompting settings
MAX_NEW_TOKENS = 600

# ---------------- HELPERS ----------------
def safe_load_idefics(local_path: str):
    """Load IDEFICS2 model + processor from local snapshot if available, else try remote fallback."""
    logger.info("Loading IDEFICS2 model and processor (local_files_only=True)...")
    try:
        processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            local_path, local_files_only=True, torch_dtype=torch.float16, device_map="auto"
        )
        logger.info("✅ Loaded IDEFICS2 from local snapshot (no network required).")
        return processor, model
    except Exception as e_local:
        logger.warning("Local-only load failed: {}", e_local)
        logger.warning("Attempting to load from remote (requires network). This will re-download if needed.")
        # fallback to remote (network)
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceM4/idefics2-8b", torch_dtype=torch.float16, device_map="auto"
        )
        logger.info("✅ Loaded IDEFICS2 from remote HF hub.")
        return processor, model

def load_retrieval_models():
    logger.info("Loading text & image retrieval encoders...")
    text_encoder = SentenceTransformer(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device=DEVICE
    )
    # OpenCLIP create model and transforms
    clip_model, clip_preprocess_train, clip_preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(DEVICE).eval()
    logger.info("✅ Text encoder (BioBERT) and CLIP loaded.")
    return text_encoder, clip_model, clip_tokenizer, clip_preprocess_train

def connect_chroma_db(db_path: str = "../output_db"):
    logger.info("Connecting to ChromaDB persistent client at: {}", db_path)
    client = chromadb.PersistentClient(path=db_path)
    available = [c.name for c in client.list_collections()]
    logger.info("Collections found: {}", available)
    if TEXT_COLLECTION_NAME not in available or IMAGE_COLLECTION_NAME not in available:
        logger.error("Expected collections '{}' and '{}' not present. Found: {}", TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME, available)
        raise RuntimeError("ChromaDB collections missing. Run embed pipeline first.")
    text_col = client.get_collection(TEXT_COLLECTION_NAME)
    image_col = client.get_collection(IMAGE_COLLECTION_NAME)
    return client, text_col, image_col

def norm_np(x: np.ndarray) -> np.ndarray:
    """Normalize numpy vector along last dim"""
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom

# ---------------- RETRIEVAL ----------------
def retrieve_context(
    query: str,
    text_encoder: SentenceTransformer,
    clip_model,
    clip_tokenizer,
    text_col,
    image_col,
    text_k: int = TEXT_TOP_K,
    image_k: int = IMAGE_TOP_K,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns:
      text_hits: list of dicts: {"document": text, "metadata": {...}, "score": float}
      image_hits: list of dicts: {"image_path": path, "metadata": {...}, "score": float}
    """
    logger.info("Retrieving context for query: {}", query)

    # 1) Text retrieval (BioBERT)
    q_emb_text = text_encoder.encode([query], convert_to_numpy=True)
    text_res = text_col.query(query_embeddings=q_emb_text.tolist(), n_results=text_k)
    # text_res structure: dict with keys 'ids', 'metadatas', 'documents', 'distances' (depending on db impl)
    text_docs = text_res.get("documents", [[]])[0]
    text_metas = text_res.get("metadatas", [[]])[0]
    # distances might be missing depending on chroma version
    text_distances = text_res.get("distances", [[]])[0] if "distances" in text_res else [None] * len(text_docs)

    text_hits = []
    for doc, meta, dist in zip(text_docs, text_metas, text_distances):
        text_hits.append({"document": doc, "metadata": meta, "score": float(dist) if dist is not None else None})

    # 2) Image retrieval (via CLIP text encoder)
    with torch.no_grad():
        tokens = clip_tokenizer([query]).to(DEVICE)  # tokenization returns tensors (open_clip)
        clip_text_emb = clip_model.encode_text(tokens)  # (1, dim)
        clip_text_emb = clip_text_emb / clip_text_emb.norm(dim=-1, keepdim=True)
        clip_text_emb_np = clip_text_emb.cpu().numpy()

    image_res = image_col.query(query_embeddings=clip_text_emb_np.tolist(), n_results=image_k)
    image_metas = image_res.get("metadatas", [[]])[0]
    image_ids = image_res.get("ids", [[]])[0]
    image_dist = image_res.get("distances", [[]])[0] if "distances" in image_res else [None] * len(image_ids)

    image_hits = []
    for meta, iid, dist in zip(image_metas, image_ids, image_dist):
        # The metadata should contain the 'image_path' saved earlier by embed pipeline
        image_path = meta.get("image_path") or meta.get("image_path_ref") or meta.get("image_path_local") or None
        image_hits.append({"image_path": image_path, "metadata": meta, "id": iid, "score": float(dist) if dist is not None else None})

    logger.info("Retrieved {} text items and {} image items.", len(text_hits), len(image_hits))
    return text_hits, image_hits

# ---------------- PROMPT / FOLLOW-UP QUESTIONS ----------------
def clinical_followup_dialog() -> dict:
    """
    Ask a short set of clinically relevant follow-up questions (interactive).
    Returns a dict with keys: age, sex, smoking_pack_years, symptom_duration_days,
    main_symptoms (list/string), prior_cancer_history (bool/str), comorbidities (str).
    """
    print("\n--- Quick clinical context (these help the model be accurate). Press Enter to skip any ---")
    age = input("Patient age (years): ").strip()
    sex = input("Patient sex (M/F/Other): ").strip()
    smoking = input("Smoking history (e.g., '40 pack-years' or 'never'): ").strip()
    duration = input("Symptom duration (days / weeks / months): ").strip()
    main_symptoms = input("Key symptoms (comma separated): ").strip()
    prior_cancer = input("Prior cancer history (Yes / No — brief details if Yes): ").strip()
    comorbid = input("Important comorbidities (e.g., COPD, heart disease) or medications: ").strip()

    clinical = {
        "age": age or None,
        "sex": sex or None,
        "smoking": smoking or None,
        "duration": duration or None,
        "main_symptoms": main_symptoms or None,
        "prior_cancer": prior_cancer or None,
        "comorbidities": comorbid or None,
    }
    return clinical

def build_structured_prompt(query: str, clinical_ctx: dict, text_hits: List[dict], image_hits: List[dict]) -> str:
    """
    Build a strong prompt for IDEFICS2:
    - Provide user query
    - Provide short, bullet clinical context
    - Provide numbered retrieved text snippets with source metadata
    - Provide list of image references (paths + metadata)
    - Ask the model to ask clarifying q's first if more needed, otherwise produce structured answer
    """
    header = "You are an expert AI medical assistant for qualified physicians. Use ONLY the retrieved textbook context and images to answer. Cite sources (book filename and page_number) for statements that come from a textbook. Do NOT invent references.\n\n"

    clinical_lines = ["Clinical Context:"]
    for k, v in clinical_ctx.items():
        clinical_lines.append(f"- {k}: {v if v else 'Unknown'}")
    clinical_block = "\n".join(clinical_lines)

    # text snippets: number them and include short metadata
    snippets = []
    for idx, hit in enumerate(text_hits, start=1):
        doc = hit["document"] or ""
        meta = hit.get("metadata", {}) or {}
        book = meta.get("book_filename", meta.get("filename", "unknown"))
        page = meta.get("page_number", meta.get("page", "unknown"))
        short = doc.strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400].rsplit(" ", 1)[0] + "..."
        snippets.append(f"[{idx}] ({book} | page {page}) {short}")

    snippets_block = "Retrieved Text Snippets:\n" + "\n".join(snippets) if snippets else "No text snippets retrieved."

    # image refs
    img_lines = []
    for idx, hit in enumerate(image_hits, start=1):
        path = hit.get("image_path") or "<no-path>"
        meta = hit.get("metadata") or {}
        book = meta.get("book_filename", meta.get("filename", "unknown"))
        page = meta.get("page_number", meta.get("page_number", "unknown"))
        img_lines.append(f"[{idx}] {path}  ({book} | page {page})")
    imgs_block = "Retrieved Images:\n" + "\n".join(img_lines) if img_lines else "No images retrieved."

    instruction = (
        "\n\nTask:\n"
        "1) If you require additional clinical clarifying questions to answer this safely and correctly, list them (short). "
        "2) Otherwise, produce a structured answer with sections: Summary, Likely differential diagnoses (with reasoning), "
        "Recommended next diagnostic steps (imaging / labs), Immediate management suggestions (brief), and References (cite the retrieved sources by number).\n"
        "Keep answers concise, clinically actionable, and cite sources using the [n] reference number from the 'Retrieved Text Snippets' section.\n"
    )

    prompt = (
        header
        + f"User Query: {query}\n\n"
        + clinical_block
        + "\n\n"
        + snippets_block
        + "\n\n"
        + imgs_block
        + instruction
    )
    return prompt

# ---------------- GENERATION ----------------
def generate_with_idefics(processor, model, prompt_text: str, images: Optional[List[Image.Image]] = None) -> str:
    """
    Send multimodal request to IDEFICS2. If images provided, prompt_text must contain <image> token
    for each image. We will only pass up to 2 images to keep memory manageable.
    """
    # Limit images to at most 2 (configurable)
    if images:
        images = images[:2]
        # ensure prompt has exactly as many <image> tokens as images
        # if not present, prepend one <image> token
        n_images_needed = len(images)
        # count tokens already in prompt
        cur_count = prompt_text.count("<image>")
        if cur_count < n_images_needed:
            # prepend required number of tokens
            prompt_text = ("<image>\n" * (n_images_needed - cur_count)) + prompt_text

    # Prepare inputs
    inputs = processor(text=prompt_text, images=images if images else None, return_tensors="pt")
    inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    # Generate
    logger.info("Sending inputs to IDEFICS2 (generate)...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # Decode
    if isinstance(out, (list, tuple)):
        out_ids = out[0]
    else:
        out_ids = out
    decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return decoded

# ---------------- MAIN INTERACTIVE FLOW ----------------
def main():
    # 1) Load models
    processor, idefics_model = safe_load_idefics(LOCAL_IDEFICS_SNAPSHOT)
    text_encoder, clip_model, clip_tokenizer, clip_preproc = load_retrieval_models()
    # 2) Connect DB
    client, text_col, image_col = connect_chroma_db("../output_db")

    # 3) Get user query
    print("\n=== IDEFICS2 Multimodal Medical Assistant (v2) ===")
    user_query = input("Enter the clinical question (short): ").strip()
    if not user_query:
        logger.error("No query provided. Exiting.")
        return

    # 4) Optional interactive clinical follow-ups to enrich context
    clinical_ctx = clinical_followup_dialog()

    # 5) Ask if user wants to upload a patient image to include
    img_path = input("Optional: Enter a local image path to include (full path), or press Enter to skip: ").strip() or None
    user_images = []
    if img_path:
        if os.path.exists(img_path):
            try:
                user_images.append(Image.open(img_path).convert("RGB"))
                logger.info("Loaded user image: {}", img_path)
            except Exception as e:
                logger.warning("Could not open provided image path: {}  (error: {}). Ignoring.", img_path, e)
        else:
            logger.warning("Image path does not exist: {}. Ignoring.", img_path)

    # 6) Retrieve context from DB
    text_hits, image_hits = retrieve_context(user_query, text_encoder, clip_model, clip_tokenizer, text_col, image_col, text_k=TEXT_TOP_K, image_k=IMAGE_TOP_K)

    # Print retrieved sources (for user to inspect)
    print("\n--- Retrieved Text Snippets (short) ---")
    for i, hit in enumerate(text_hits, start=1):
        meta = hit.get("metadata", {}) or {}
        book = meta.get("book_filename", meta.get("filename", "unknown"))
        page = meta.get("page_number", meta.get("page", "unknown"))
        print(f"[{i}] ({book} | page {page})  score={hit.get('score')}")
        # print first 200 chars of document
        txt = (hit.get("document") or "").replace("\n", " ")
        print("   ", txt[:200].strip(), "..." if len(txt) > 200 else "")

    print("\n--- Retrieved Images (paths + metadata) ---")
    for i, ih in enumerate(image_hits, start=1):
        print(f"[{i}] path: {ih.get('image_path')}  score={ih.get('score')}  book={ih.get('metadata', {}).get('book_filename', ih.get('metadata', {}).get('filename', 'unknown'))} page={ih.get('metadata', {}).get('page_number')}")

    # 7) Build structured prompt
    prompt_text = build_structured_prompt(user_query, clinical_ctx, text_hits, image_hits)

    # 8) Assemble images to send to model: user image(s) first (if any), then top retrieved image (if exists)
    images_to_send = []
    if user_images:
        images_to_send.extend(user_images)
    # add up to 1 retrieved image for context (optional)
    for ih in image_hits[:1]:
        p = ih.get("image_path")
        if p and os.path.exists(p):
            try:
                images_to_send.append(Image.open(p).convert("RGB"))
            except Exception as e:
                logger.warning("Could not open retrieved image {}: {}", p, e)

    # 9) Generate (first check: ask clarifying q's?)
    # We let the model itself decide in the prompt whether clarifying Qs are needed; if user wants manual QA loop, add here.
    answer = generate_with_idefics(processor, idefics_model, prompt_text, images=images_to_send if images_to_send else None)

    # 10) Final output
    print("\n\n=== Generated Answer (IDEFICS2) ===\n")
    print(answer)
    print("\n\n--- End of answer ---\n")
    # Always print the image file locations for reproducibility / auditing
    print("Retrieved image paths (for audit):")
    for ih in image_hits:
        print("-", ih.get("image_path"))

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled error in pipeline: {}", e)
    finally:
        logger.info("Run time: {:.1f}s", time.time() - t0)
