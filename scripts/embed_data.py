#!/usr/bin/env python3
"""
embed_data.py - Robust hybrid embedding v9
- Text: BioBERT (sentence-transformers)
- Images: OpenAI CLIP via transformers (CLIPModel + CLIPProcessor)
- Two separate persistent Chroma collections:
    - medical_text_collection  (768-D)
    - medical_image_collection (512-D)
- Robust image path resolution, batching, OOM backoff
- Chunked DB writes (safe for large books)
- Hybrid retrieval helper functions
"""

import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from loguru import logger

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

import chromadb

# allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

INPUT_DATA_PATH = "../input_data"     # folder containing per-book directories
OUTPUT_DB_PATH = "../output_db"       # persistent chroma directory
TEXT_COLLECTION_NAME = "medical_text_collection"
IMAGE_COLLECTION_NAME = "medical_image_collection"

# batching tuning
TEXT_BATCH = 32
IMG_BATCH_START = 16   # initial image batch size; will reduce if OOM
DB_CHUNK = 2048        # how many records to write per collection.add call

# behavior flags
RESUME_IF_COLLECTION_EXISTS = False  # if False, deletes any existing collections with same name and starts fresh
VERBOSE = True

# ---------------- LOGGING ----------------
logger.remove()
logger.add(lambda m: print(m, end=""), level="INFO")


# ---------------- Utilities ----------------
def load_models():
    logger.info(f"Loading text model: {TEXT_MODEL_NAME} (SentenceTransformer) ...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    logger.info("Loading CLIP (transformers) ...")
    clip_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    logger.info(f"Models loaded on {DEVICE}")
    return text_model, clip_model, clip_processor


def resolve_image_path(img_meta_path: str, book_path: str) -> Optional[str]:
    """
    Try multiple candidate locations and return first existing absolute path or None.
    """
    if not img_meta_path:
        return None

    candidates = []
    p = Path(img_meta_path)

    # direct absolute
    if p.is_absolute():
        candidates.append(p)

    # try as-is relative to book_path
    candidates.append(Path(book_path) / img_meta_path)

    # basename inside book_path
    candidates.append(Path(book_path) / p.name)

    # common subfolders in book folder
    candidates.append(Path(book_path) / "images" / p.name)
    candidates.append(Path(book_path) / "figures" / p.name)
    candidates.append(Path(book_path) / "figures" / p)
    candidates.append(Path(book_path) / Path(img_meta_path).relative_to("./") if img_meta_path.startswith("./") else Path(book_path) / img_meta_path)

    # try central input_data folder
    candidates.append(Path(INPUT_DATA_PATH) / p.name)
    candidates.append(Path(INPUT_DATA_PATH) / img_meta_path)

    # normalize and test
    for c in candidates:
        try:
            if c.exists():
                return str(c.resolve())
        except Exception:
            continue
    return None


def embed_images_paths(image_paths: List[str], clip_model: CLIPModel, clip_processor: CLIPProcessor,
                       start_batch_size: int = IMG_BATCH_START) -> np.ndarray:
    """
    Encode images from file paths into CLIP image embeddings (normalized).
    Handles CUDA OOM by backing off batch_size.
    Returns numpy array shape (n_valid_images, dim).
    """
    device = next(clip_model.parameters()).device
    embeddings = []
    n = len(image_paths)
    if n == 0:
        return np.zeros((0, clip_model.config.projection_dim if hasattr(clip_model.config, "projection_dim") else 512), dtype=np.float32)

    batch_size = start_batch_size
    i = 0
    while i < n:
        end = min(i + batch_size, n)
        batch_paths = image_paths[i:end]
        pil_images = []
        valid_indices = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                pil_images.append(img)
                valid_indices.append(i + j)
            except Exception as e:
                logger.warning(f"Failed to open image {p}: {e}")

        if len(pil_images) == 0:
            i = end
            continue

        try:
            inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                out_np = out.cpu().numpy()
            # append each embedding in order
            for emb in out_np:
                embeddings.append(emb)
            # close PIL images
            for img in pil_images:
                try:
                    img.close()
                except Exception:
                    pass
            i = end
        except torch.cuda.OutOfMemoryError as oom:
            logger.warning(f"CUDA OOM with batch_size={batch_size}. Reducing batch size and retrying.")
            torch.cuda.empty_cache()
            for img in pil_images:
                try:
                    img.close()
                except Exception:
                    pass
            if batch_size <= 1:
                logger.error("Batch size 1 still OOM. Aborting image embedding.")
                raise oom
            batch_size = max(1, batch_size // 2)
            time.sleep(0.5)
            continue
        except Exception as e:
            logger.error(f"Unexpected error during image batch encoding: {e}")
            # fallback: try single-image encoding for this batch
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    inputs = clip_processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = clip_model.get_image_features(**inputs)
                        out = out / out.norm(p=2, dim=-1, keepdim=True)
                        embeddings.append(out.cpu().numpy()[0])
                    img.close()
                except Exception as e2:
                    logger.warning(f"Single-image fallback failed for {p}: {e2}")
            i = end

    if len(embeddings) == 0:
        return np.zeros((0, clip_model.config.projection_dim if hasattr(clip_model.config, "projection_dim") else 512), dtype=np.float32)
    return np.vstack(embeddings)


# ---------------- Chroma init ----------------
def init_chroma_collections(persist_path: str):
    # Use PersistentClient; this will create DB files under persist_path
    client = chromadb.PersistentClient(path=persist_path)
    # handle existing collections
    existing = [c.name for c in client.list_collections()]
    if not RESUME_IF_COLLECTION_EXISTS:
        # delete if exist
        for nm in [TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME]:
            if nm in existing:
                logger.info(f"Deleting existing collection: {nm}")
                client.delete_collection(name=nm)

    # create/get collections
    text_col = client.get_or_create_collection(name=TEXT_COLLECTION_NAME)
    image_col = client.get_or_create_collection(name=IMAGE_COLLECTION_NAME)
    logger.info(f"Chroma collections ready: {TEXT_COLLECTION_NAME} (text), {IMAGE_COLLECTION_NAME} (image)")
    return client, text_col, image_col


# ---------------- Main processing ----------------
def process_and_embed():
    logger.info("=== Starting Hybrid embedding (final v9) ===")
    text_model, clip_model, clip_processor = load_models()
    client, text_collection, image_collection = init_chroma_collections(OUTPUT_DB_PATH)

    total_added = 0
    book_folders = [d for d in sorted(os.listdir(INPUT_DATA_PATH)) if os.path.isdir(os.path.join(INPUT_DATA_PATH, d))]
    for book_folder in book_folders:
        book_path = os.path.join(INPUT_DATA_PATH, book_folder)
        jsonl_path = os.path.join(book_path, "structured_output.jsonl")
        if not os.path.exists(jsonl_path):
            logger.warning(f"No structured_output.jsonl in {book_folder}; skipping.")
            continue

        logger.info(f"--- Processing book: {book_folder} ---")

        text_items = []
        text_meta = []
        text_ids = []

        image_paths = []
        image_meta = []
        image_ids = []

        total_image_entries = 0
        missing_images = []

        last_title = "General"

        # Read JSONL and collect items
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                etype = data.get("type", "")
                txt = (data.get("text") or "").strip()
                meta = data.get("metadata") or {}

                if etype == "Title" and txt:
                    last_title = txt

                if etype in ["NarrativeText", "ListItem", "UncategorizedText", "Table", "FigureCaption", "Title"]:
                    if txt:
                        text_items.append(txt)
                        text_meta.append({
                            "content_type": "text",
                            "element_type": etype,
                            "section_title": last_title,
                            "book_filename": book_folder,
                            "page_number": int(meta.get("page_number", 0)) if meta.get("page_number") is not None else 0,
                        })
                        text_ids.append(f"txt_{uuid.uuid4()}")
                elif etype == "Image":
                    total_image_entries += 1
                    meta_img_path = meta.get("image_path") or meta.get("path") or ""
                    resolved = resolve_image_path(meta_img_path, book_path)
                    if not resolved:
                        missing_images.append(meta_img_path)
                        logger.debug(f"Could not resolve image path: {meta_img_path}")
                        continue
                    image_paths.append(resolved)
                    image_meta.append({
                        "content_type": "image",
                        "section_title": last_title,
                        "book_filename": book_folder,
                        "page_number": int(meta.get("page_number", 0)) if meta.get("page_number") is not None else 0,
                        "text_description": txt,
                        "image_path": resolved
                    })
                    image_ids.append(f"img_{uuid.uuid4()}")

        logger.info(f"Collected: {len(text_items)} text items, {total_image_entries} image entries (found {len(image_paths)} valid, missing {len(missing_images)})")

        # ---------- Text embedding & DB insertion ----------
        if text_items:
            logger.info(f"Encoding {len(text_items)} text items ...")
            text_embeddings = text_model.encode(text_items, batch_size=TEXT_BATCH, show_progress_bar=True, convert_to_numpy=True)
            # chunked add
            for i in range(0, len(text_ids), DB_CHUNK):
                c_ids = text_ids[i:i+DB_CHUNK]
                c_emb = text_embeddings[i:i+DB_CHUNK].tolist()
                c_meta = text_meta[i:i+DB_CHUNK]
                c_docs = text_items[i:i+DB_CHUNK]
                text_collection.add(ids=c_ids, embeddings=c_emb, metadatas=c_meta, documents=c_docs)
                total_added += len(c_ids)
            logger.info(f"Added {len(text_ids)} text embeddings to collection.")

        # ---------- Image embedding & DB insertion ----------
        if image_paths:
            logger.info(f"Encoding {len(image_paths)} images (start batch {IMG_BATCH_START}) ...")
            try:
                img_embeddings = embed_images_paths(image_paths, clip_model, clip_processor, start_batch_size=IMG_BATCH_START)
            except Exception as e:
                logger.error(f"Fatal image embedding error: {e}")
                img_embeddings = np.zeros((0, 512), dtype=np.float32)

            # If embeddings count differs from ids, align
            if img_embeddings.shape[0] != len(image_ids):
                logger.warning(f"Image embeddings count ({img_embeddings.shape[0]}) != image ids ({len(image_ids)}). Truncating to min length.")
                minlen = min(img_embeddings.shape[0], len(image_ids))
                img_embeddings = img_embeddings[:minlen]
                image_ids = image_ids[:minlen]
                image_meta = image_meta[:minlen]
                image_paths = image_paths[:minlen]

            # add in chunks
            for i in range(0, len(image_ids), DB_CHUNK):
                c_ids = image_ids[i:i+DB_CHUNK]
                c_emb = img_embeddings[i:i+DB_CHUNK].tolist()
                c_meta = image_meta[i:i+DB_CHUNK]
                c_docs = [m.get("text_description") or "[IMAGE]" for m in c_meta]
                image_collection.add(ids=c_ids, embeddings=c_emb, metadatas=c_meta, documents=c_docs)
                total_added += len(c_ids)
            logger.info(f"Added {len(image_ids)} image embeddings to collection.")

        logger.info(f"Finished book {book_folder}. Total added so far: {total_added}")

    logger.success(f"=== All books processed. Total items added: {total_added} ===")
    try:
        logger.info(f"Text collection count: {text_collection.count()}, Image collection count: {image_collection.count()}")
    except Exception:
        pass

    # return handles for optional interactive usage
    return text_model, clip_model, clip_processor, client, text_collection, image_collection


# ---------------- Hybrid retrieval helpers ----------------
def hybrid_query_text_and_images(query_text: str, text_model: SentenceTransformer, clip_model: CLIPModel,
                                 clip_processor: CLIPProcessor, text_collection, image_collection,
                                 top_k_text: int = 5, top_k_image: int = 5):
    """
    Performs:
      - text search in text_collection using sentence-transformers text_model
      - image search in image_collection using CLIP text encoder
    Returns both raw Chroma results.
    """
    # Text retrieval
    q_emb_text = text_model.encode([query_text], convert_to_numpy=True)
    text_res = text_collection.query(query_embeddings=q_emb_text, n_results=top_k_text)

    # CLIP text -> image retrieval
    device = next(clip_model.parameters()).device
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        q_emb_clip = clip_model.get_text_features(**inputs)
        q_emb_clip = q_emb_clip / q_emb_clip.norm(p=2, dim=-1, keepdim=True)
        q_emb_clip_np = q_emb_clip.cpu().numpy()
    image_res = image_collection.query(query_embeddings=q_emb_clip_np, n_results=top_k_image)

    return {
        "text_results": text_res,
        "image_results": image_res
    }


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    start_time = time.time()
    try:
        text_model, clip_model, clip_processor, client, text_col, image_col = process_and_embed()
    except Exception as e:
        logger.error(f"Critical failure during embedding run: {e}")
        raise

    elapsed = time.time() - start_time
    logger.info(f"Embedding run finished in {elapsed:.1f}s")

    # Optional quick test query (uncomment to run small verification)
    # sample_q = "mesangiocapillary glomerulonephritis"
    # res = hybrid_query_text_and_images(sample_q, text_model, clip_model, clip_processor, text_col, image_col, 5, 5)
    # logger.info(f"Sample query text results: {res['text_results']}")
    # logger.info(f"Sample query image results: {res['image_results']}")

    logger.info("Done.")
