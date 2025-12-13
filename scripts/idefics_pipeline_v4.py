#!/usr/bin/env python3
"""
idefics_pipeline_v4.py
Robust multimodal clinical assistant (IDEFICS2 + retrieval + iterative clarifying Qs + persistent session memory)

Save as ~/Medical_Bot/scripts/idefics_pipeline_v4.py
Run: python idefics_pipeline_v4.py
"""

import os, sys, time, json, uuid, re
from typing import List, Optional, Tuple, Dict
import torch
from loguru import logger
from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import chromadb
import numpy as np

# New imports for fetching remote images
import urllib.request
import tempfile
from statistics import mean

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True)

# Put exact snapshot (the folder that contains model-00001-of-00007.safetensors ... model-00007-of-00007.safetensors)
LOCAL_IDEFICS_SNAPSHOT = "/teamspace/studios/this_studio/.cache/huggingface/hub/models--HuggingFaceM4--idefics2-8b/snapshots/2c42686c57fe21cf0348c9ce1077d094b72e7698"

# Chroma DB path and collections
CHROMA_DB_PATH = "../output_db"
TEXT_COLLECTION = "medical_text_collection"
IMAGE_COLLECTION = "medical_image_collection"

# Retrieval & generation params
TEXT_TOP_K = 6
IMAGE_TOP_K = 6
MAX_NEW_TOKENS = 512
MAX_CLARIFY_ROUNDS = 3

# Session storage
SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ---------------- UTIL ----------------
def check_snapshot_ok(path: str) -> Tuple[bool, List[str]]:
    """Check that the snapshot folder exists and that safetensors shards exist."""
    if not os.path.isdir(path):
        return False, [f"Path not found: {path}"]
    files = os.listdir(path)
    # look for model-00001-of-00007.safetensors pattern
    shards = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]
    if len(shards) == 0:
        return False, ["No safetensors shards found in snapshot folder: " + path]
    return True, shards

def ensure_chroma_collections(client):
    names = [c.name for c in client.list_collections()]
    if TEXT_COLLECTION not in names or IMAGE_COLLECTION not in names:
        raise RuntimeError(f"Chroma collections missing. Found: {names}. Expected: {TEXT_COLLECTION}, {IMAGE_COLLECTION}")

def save_session(session_id: str, data: dict):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_session(session_id: str) -> dict:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"session_id": session_id, "history": [], "patient": {}}

def norm_np(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x, axis=-1, keepdims=True)
    d[d==0]=1.0
    return x/d

# ---------------- Helper: fetch image from local path or URL ----------------
def fetch_image(path_or_url: Optional[str]) -> Optional[Image.Image]:
    """
    Accepts a local filesystem path or an http/https URL and returns a PIL RGB Image
    or None if it cannot be opened. Uses a temporary file for URLs.
    """
    if not path_or_url:
        return None
    try:
        # Local file
        if os.path.exists(path_or_url):
            try:
                return Image.open(path_or_url).convert("RGB")
            except Exception as e:
                logger.warning("fetch_image: failed to open local image '{}' : {}", path_or_url, e)
                return None
        # URL handling
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            tmp_fd, tmp_path = None, None
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(path_or_url)[-1] or ".jpg")
                os.close(tmp_fd)
                urllib.request.urlretrieve(path_or_url, tmp_path)
                img = Image.open(tmp_path).convert("RGB")
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return img
            except Exception as e:
                logger.warning("fetch_image: failed to download/open URL '{}' : {}", path_or_url, e)
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return None
        # Not a path we can handle
        logger.debug("fetch_image: path_or_url not a file or http(s): {}", path_or_url)
        return None
    except Exception as e:
        logger.warning("fetch_image: unexpected error for '{}': {}", path_or_url, e)
        return None

# ---------------- LOAD MODELS ----------------
def load_idefics(local_path: str):
    ok, info = check_snapshot_ok(local_path)
    if not ok:
        logger.warning("Local IDEFICS snapshot check failed: {}", info)
        logger.warning("Will attempt remote load (this will re-download if necessary). To avoid downloads, place snapshot at the path above.")
        # fallback remote — still set to fp16 and device_map auto
        proc = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        model = AutoModelForImageTextToText.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.float16, device_map="auto")
        return proc, model, False
    # local_files_only True avoids re-download. If anything missing, HF will raise.
    proc = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(local_path, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
    return proc, model, True

def load_retrieval():
    logger.info("Loading retrieval encoders (BioBERT + OpenCLIP)...")
    text_enc = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device=DEVICE)
    clip_model, clip_transforms, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(DEVICE).eval()
    return text_enc, clip_model, clip_tok, clip_transforms

# ---------------- CHROMA CONNECT ----------------
def connect_chroma(db_path=CHROMA_DB_PATH):
    logger.info("Connecting to ChromaDB at {}", db_path)
    client = chromadb.PersistentClient(path=db_path)
    ensure_chroma_collections(client)
    text_col = client.get_collection(TEXT_COLLECTION)
    image_col = client.get_collection(IMAGE_COLLECTION)
    return client, text_col, image_col

# ---------------- RETRIEVAL ----------------
def retrieve(query: str, text_enc, clip_model, clip_tok, text_col, image_col, text_k=TEXT_TOP_K, image_k=IMAGE_TOP_K):
    logger.info("Retrieving for query: {}", query)
    # text embedding
    text_emb = text_enc.encode([query], convert_to_numpy=True)
    text_res = text_col.query(query_embeddings=text_emb.tolist(), n_results=text_k)
    # extract documents and metadata safely
    docs = text_res.get("documents", [[]])[0]
    metas = text_res.get("metadatas", [[]])[0]
    dists = text_res.get("distances", [[]])[0] if "distances" in text_res else [None]*len(docs)
    text_hits = [{"document": docs[i], "metadata": metas[i], "score": (dists[i] if dists else None)} for i in range(len(docs))]

    # clip text embed
    with torch.no_grad():
        tokens = clip_tok([query]).to(DEVICE)
        clip_emb = clip_model.encode_text(tokens)
        clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
        clip_emb_np = clip_emb.cpu().numpy()
    image_res = image_col.query(query_embeddings=clip_emb_np.tolist(), n_results=image_k)
    image_metas = image_res.get("metadatas", [[]])[0]
    image_ids = image_res.get("ids", [[]])[0]
    image_dists = image_res.get("distances", [[]])[0] if "distances" in image_res else [None]*len(image_ids)
    image_hits = []
    for i, iid in enumerate(image_ids):
        meta = image_metas[i] if i < len(image_metas) else {}
        # Try multiple metadata keys that might contain a path or URL
        path = meta.get("image_path") or meta.get("image_path_ref") or meta.get("image_path_local") or meta.get("path") or meta.get("img_path") or meta.get("image") or meta.get("url")
        image_hits.append({"id": iid, "image_path": path, "metadata": meta, "score": (image_dists[i] if image_dists else None)})

    # Compute image score statistics if available
    img_scores = [ih["score"] for ih in image_hits if ih.get("score") is not None]
    image_avg = float(mean(img_scores)) if len(img_scores) > 0 else None

    # Choose sort direction intelligently:
    # - If avg score is large (>1.0) -> likely a distance metric (lower better) -> sort ascending
    # - Else assume similarity (higher better) -> sort descending
    if image_avg is not None:
        if image_avg > 1.0:
            image_hits = sorted(image_hits, key=lambda x: (x.get("score") if x.get("score") is not None else float("inf")))
        else:
            image_hits = sorted(image_hits, key=lambda x: (-(x.get("score") if x.get("score") is not None else 0)))
    else:
        # no scores — keep in returned order
        pass

    # Keep top-k images (soft cap)
    image_hits = image_hits[:image_k]

    # Logging retrieval confidence approx for text (safe)
    try:
        txt_scores = [t.get("score") for t in text_hits if t.get("score") is not None]
        txt_avg = float(mean(txt_scores)) if len(txt_scores) > 0 else None
        if txt_avg is not None:
            logger.info("Retrieval confidence (approx): {}", txt_avg)
    except Exception as e:
        logger.warning("Confidence calc failed: {}", e)

    logger.info("Retrieved {} text and {} images", len(text_hits), len(image_hits))
    return text_hits, image_hits

# ---------------- PROMPT + CLARIFY PROTOCOL ----------------
SYSTEM_PROMPT = """
You are an expert clinical assistant for physicians.

Before answering, ALWAYS verify the following critical patient information is available:
- Age
- Sex
- Major comorbidities (diabetes, hypertension, heart disease, etc.)
- Duration and nature of key symptoms
- Key labs or imaging findings (if relevant)

If any of these are missing, you MUST return:
{"clarify":["question 1?","question 2?",...]}

If you have enough info, return:
{"final_answer":"<structured clinical answer>"}

The final_answer SHOULD include:
- SUMMARY
- DIFFERENTIAL
- RECOMMENDED TESTS
- MANAGEMENT
- PROCEDURE_INDICATIONS
- MEDS_AND_LIFESTYLE
- UNCERTAINTY
- REFERENCES

Always be concise, evidence-based, and cite retrieved snippets like [1], [2].
Only return valid JSON.
"""

def build_prompt(user_query, clinical_ctx, text_hits, image_hits):
    """
    Constructs a structured, evidence-grounded prompt for IDEFICS2.
    """
    # Context (text)
    if text_hits:
        context_text = "\n".join(
            [f"[{i+1}] {hit['document']}" for i, hit in enumerate(text_hits)]
        )
    else:
        context_text = "No relevant medical text found."

    # Context (images)
    if image_hits:
        image_refs = "\n".join(
            [f"Image {i+1}: {hit.get('image_path') or hit.get('metadata', {}).get('path') or 'No path available'}" for i, hit in enumerate(image_hits)]
        )
    else:
        image_refs = "No relevant images found."

    patient_context = "\n".join([f"- {k}: {v}" for k, v in clinical_ctx.items()]) or "No patient-specific data available."

    system_prompt = (
        "You are an expert AI clinical assistant trained on authoritative medical textbooks. "
        "Your task is to synthesize context from both text and image evidence to provide a professional, "
        "evidence-based medical explanation. "
        "If critical clinical details are missing, ask only relevant clarifying questions."
    )

    prompt_text = f"""
{system_prompt}

Patient Context:
{patient_context}

User Query:
{user_query}

Retrieved Text Context:
{context_text}

Retrieved Images:
{image_refs}

Instructions:
- Use logical clinical reasoning.
- Output sections: Summary, Mechanism/Pathophysiology, Clinical Features, Investigations, Management, References.
- Cite each reference clearly using the source book and page number if available.
- Always interpret both the text and retrieved images and make use of image evidence where relevant.
- Each reference [n] must map to its textbook title and figure if known.
- Include a full "REFERENCES" section listing the sources and, where appropriate, the figure numbers for any images used.
"""
    return prompt_text

# ✅ Fix C: Smarter JSON Extraction
def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None

# ---------------- GENERATE WITH IDEFICS ----------------
def generate_with_idefics(processor, model, prompt_text: str, images: Optional[List[Image.Image]] = None) -> str:
    images = images or []
    if images:
        n_needed = len(images)
        cur = prompt_text.count("<image>")
        if cur < n_needed:
            prompt_text = ("<image>\n" * (n_needed - cur)) + prompt_text
    inputs = processor(text=prompt_text, images=images if images else None, return_tensors="pt")
    inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return decoded

# ---------------- MAIN INTERACTIVE ----------------
def main():
    logger.info("Device: {}", DEVICE)
    proc, idef_model, local_ok = load_idefics(LOCAL_IDEFICS_SNAPSHOT)
    text_enc, clip_model, clip_tok, clip_transforms = load_retrieval()
    client, text_col, image_col = connect_chroma()
    session_id = input("Session ID (leave blank to create new): ").strip() or str(uuid.uuid4())[:8]
    session = load_session(session_id)
    logger.info("Session: {}", session_id)

    while True:
        user_query = input("\nEnter clinical question (or 'exit' to quit): ").strip()
        if not user_query:
            print("Please enter a question or 'exit'.")
            continue
        if user_query.lower() in ("exit", "quit"):
            save_session(session_id, session)
            logger.info("Saved session {}", session_id)
            break

        print("\nCurrent patient memory snapshot (patient fields):")
        for k, v in session.get("patient", {}).items():
            print(f"- {k}: {v}")

        text_hits, image_hits = retrieve(user_query, text_enc, clip_model, clip_tok, text_col, image_col)
        session["history"].append({"role": "user", "query": user_query, "time": time.time()})

        imgpath = input("Optional: local image path to include in this query (press Enter to skip): ").strip() or None
        user_images = []
        if imgpath:
            if os.path.exists(imgpath):
                try:
                    user_images.append(Image.open(imgpath).convert("RGB"))
                except Exception as e:
                    logger.warning("Cannot open image {} : {}", imgpath, e)
            else:
                logger.warning("Image path does not exist: {}", imgpath)

        clinical_ctx = session.get("patient", {})
        prompt = build_prompt(user_query, clinical_ctx, text_hits, image_hits)
        rounds = 0
        last_model_text = None
        while rounds < MAX_CLARIFY_ROUNDS:
            rounds += 1
            logger.info("Round {} generate...", rounds)
            images_to_send = user_images[:1]
            if image_hits:
                # Attempt to fetch the top retrieved image robustly (local path or URL or alternate meta keys)
                p = image_hits[0].get("image_path")
                if not p:
                    meta = image_hits[0].get("metadata", {}) or {}
                    p = meta.get("image_path") or meta.get("path") or meta.get("img_path") or meta.get("image") or meta.get("url")
                if p:
                    fetched = fetch_image(p)
                    if fetched:
                        images_to_send.append(fetched)
                    else:
                        logger.warning("Could not fetch retrieved image from path/url: {}", p)
                else:
                    # Detailed debug for missing path in metadata
                    logger.debug("No image path found in image_hits[0] metadata. Full metadata: {}", image_hits[0].get("metadata"))
            raw = generate_with_idefics(proc, idef_model, prompt, images=images_to_send if images_to_send else None)
            last_model_text = raw.strip()

            parsed = extract_json(last_model_text)
            if not parsed:
                logger.warning("Could not parse JSON — treating as final answer.")
                parsed = {"final_answer": last_model_text}

            if "clarify" in parsed and isinstance(parsed["clarify"], list) and len(parsed["clarify"]) > 0:
                clar_qs = parsed["clarify"]
                print("\nModel requests clarifying questions (only the ones it deemed necessary):")
                answers = {}
                for q in clar_qs:
                    a = input(q + " ").strip()
                    answers[q] = a or None
                for q, a in answers.items():
                    key = q.strip().lower().replace(" ", "_")[:60]
                    session.setdefault("patient", {})
                    session["patient"][key] = a
                session["patient"]["_last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_session(session_id, session)
                clinical_ctx = session.get("patient", {})
                prompt = build_prompt(user_query, clinical_ctx, text_hits, image_hits)
                continue
            else:
                final_text = parsed.get("final_answer") if isinstance(parsed, dict) else last_model_text
                session["history"].append({
                    "role": "assistant",
                    "answer": final_text,
                    "time": time.time(),
                    "retrieved_text_ids": [h.get("metadata", {}).get("filename") for h in text_hits]
                })
                save_session(session_id, session)
                print("\n--- Retrieved image paths & metadata (audit):")
                for ih in image_hits:
                    print("-", ih.get("image_path"), "   metadata keys:", list(ih.get("metadata", {}).keys()))
                print("\n=== FINAL ANSWER (structured) ===\n")
                print(final_text)
                print("\n--- End of answer ---\n")
                break

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        logger.exception("Fatal pipeline error: {}", e)
    finally:
        logger.info("Completed in {:.1f}s", time.time() - t0)
