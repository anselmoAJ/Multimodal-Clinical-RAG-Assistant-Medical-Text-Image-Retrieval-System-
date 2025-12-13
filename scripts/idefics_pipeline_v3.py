#!/usr/bin/env python3
"""
idefics_pipeline_v3.py
Robust multimodal clinical assistant (IDEFICS2 + retrieval + iterative clarifying Qs + persistent session memory)

Save as ~/Medical_Bot/scripts/idefics_pipeline_v3.py
Run: python idefics_pipeline_v3.py
"""

import os, sys, time, json, uuid
from typing import List, Optional, Tuple, Dict
import torch
from loguru import logger
from transformers import AutoProcessor, AutoModelForImageTextToText
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
import chromadb
import numpy as np

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
    """
    Load retrieval encoders. By default we *load them to CPU* for stability in Streamlit
    and to avoid GPU OOM/device mismatch during retrieval.
    """
    logger.info("Loading retrieval encoders (BioBERT + OpenCLIP) on CPU ...")
    retrieval_device = torch.device("cpu")

    # sentence-transformers supports device parameter
    text_enc = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device=str(retrieval_device))

    # open_clip: create model and transforms, then move to CPU
    clip_model, clip_transforms, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-B-32")
    try:
        clip_model = clip_model.to(retrieval_device).eval()
    except Exception as e:
        logger.warning("Could not move CLIP to CPU cleanly: %s", e)

    return text_enc, clip_model, clip_tok, clip_transforms

# ---------------- CHROMA CONNECT ----------------
def connect_chroma(db_path=CHROMA_DB_PATH):
    db_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), db_path))
    logger.info("Connecting to ChromaDB at %s", db_abs)

    # Try legacy PersistentClient first (many projects still use it),
    # if that fails, try new chromadb.Client with Settings.
    try:
        client = chromadb.PersistentClient(path=db_abs)
        ensure_chroma_collections(client)
        text_col = client.get_collection(TEXT_COLLECTION)
        image_col = client.get_collection(IMAGE_COLLECTION)
        logger.info("Connected to ChromaDB via PersistentClient.")
        return client, text_col, image_col
    except Exception as e_p:
        logger.warning("PersistentClient connect failed: %s", e_p)

    # fallback to new client API
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl=os.environ.get("CHROMA_DB_IMPL", "sqlite"),
                                          persist_directory=db_abs,
                                          anonymized_telemetry=False))
        ensure_chroma_collections(client)
        text_col = client.get_collection(TEXT_COLLECTION)
        image_col = client.get_collection(IMAGE_COLLECTION)
        logger.info("Connected to ChromaDB via new Client API.")
        return client, text_col, image_col
    except Exception as e_new:
        logger.error("Could not connect to ChromaDB with either client: %s | %s", e_p, e_new)
        raise RuntimeError(f"ChromaDB connection failed: {e_new}")

# ---------------- RETRIEVAL ----------------
def _to_2d_float_list(arr) -> List[List[float]]:
    """
    Convert a numpy array or torch tensor to a 2D list of float32.
    Ensures shape (n, d).
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    # if >2 dims, squeeze leading singleton dims until 2-D
    while arr.ndim > 2:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    arr = arr.astype(np.float32)
    return arr.tolist()

def retrieve(query: str, text_enc, clip_model, clip_tok, text_col, image_col, text_k=TEXT_TOP_K, image_k=IMAGE_TOP_K):
    """
    Robust retrieval:
      - Ensures text & clip embeddings are CPU numpy 2-D lists (shape (1, D))
      - Logs shapes/devices for debugging
      - Returns text_hits (list of dict) and image_hits (list of dict)
    """
    logger.info("Retrieval started for query: %s", query)
    text_hits, image_hits = [], []

    # -------- TEXT EMBEDDING (SentenceTransformer) --------
    try:
        # Ensure text_enc returns numpy array
        # use convert_to_numpy=True (SentenceTransformer API)
        text_emb = text_enc.encode([query], convert_to_numpy=True)
        text_emb = np.asarray(text_emb)
        # force 2-D shape (n, d)
        if text_emb.ndim == 1:
            text_emb = np.expand_dims(text_emb, axis=0)
        elif text_emb.ndim > 2:
            text_emb = np.squeeze(text_emb)
            if text_emb.ndim == 1:
                text_emb = np.expand_dims(text_emb, axis=0)
        logger.info("Text emb shape: %s", text_emb.shape)

        text_emb_list = _to_2d_float_list(text_emb)  # safe conversion
        # Query Chroma
        try:
            text_res = text_col.query(query_embeddings=text_emb_list, n_results=text_k)
            docs = text_res.get("documents", [[]])[0]
            metas = text_res.get("metadatas", [[]])[0]
            dists = text_res.get("distances", [[]])[0] if "distances" in text_res else [None] * len(docs)
            for i, doc in enumerate(docs):
                text_hits.append({"document": doc, "metadata": metas[i] if i < len(metas) else {}, "score": dists[i] if dists else None})
            logger.info("Text retrieval returned %d docs", len(text_hits))
        except Exception as e:
            logger.warning("Chroma text query error: %s", e)
    except Exception as e:
        logger.warning("Text embedding failed: %s", e)

    # -------- CLIP TEXT EMBEDDING (for image retrieval) --------
    try:
        # determine model device and ensure tokens on same device
        try:
            model_device = next(clip_model.parameters()).device
        except Exception:
            model_device = torch.device("cpu")

        # Tokenize - open_clip tokenizer usually returns torch.LongTensor on CPU
        tokens = clip_tok([query])
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(model_device)
        else:
            try:
                tokens = torch.tensor(tokens).to(model_device)
            except Exception:
                # As last resort, try encoding via clip_tok.encode if available
                try:
                    tokens = clip_tok.encode([query]).to(model_device)
                except Exception:
                    logger.warning("Could not build token tensor for CLIP")
                    tokens = None

        if tokens is None:
            raise RuntimeError("No tokens available for CLIP encoding")

        with torch.no_grad():
            clip_emb = clip_model.encode_text(tokens)
            # debug info
            try:
                logger.info("clip_emb dtype/device/shape: %s / %s / %s", clip_emb.dtype, clip_emb.device, tuple(clip_emb.shape))
            except Exception:
                pass
            # L2-normalize safely
            try:
                clip_emb = clip_emb / (clip_emb.norm(dim=-1, keepdim=True) + 1e-12)
            except Exception:
                pass

        # convert to 2D float list for Chroma
        clip_emb_list = _to_2d_float_list(clip_emb)
        logger.info("CLIP emb converted to list: len=%d dim=%d", len(clip_emb_list), len(clip_emb_list[0]) if len(clip_emb_list) else 0)

        try:
            image_res = image_col.query(query_embeddings=clip_emb_list, n_results=image_k)
            image_ids = image_res.get("ids", [[]])[0]
            image_metas = image_res.get("metadatas", [[]])[0]
            image_dists = image_res.get("distances", [[]])[0] if "distances" in image_res else [None] * len(image_ids)
            for i, iid in enumerate(image_ids):
                meta = image_metas[i] if i < len(image_metas) else {}
                path = meta.get("image_path") or meta.get("path") or meta.get("uri")
                if path and not os.path.isabs(path):
                    cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
                    if os.path.exists(cand):
                        path = cand
                image_hits.append({"id": iid, "image_path": path, "metadata": meta, "score": image_dists[i] if image_dists else None})
            logger.info("Image retrieval returned %d results", len(image_hits))
        except Exception as e:
            logger.warning("Chroma image query error: %s", e)

    except Exception as e:
        logger.warning("Clip embedding / image retrieval failed: %s", e)

    logger.info("retrieve() finished: %d text_hits, %d image_hits", len(text_hits), len(image_hits))
    return text_hits, image_hits

# ---------------- PROMPT + CLARIFY PROTOCOL ----------------
SYSTEM_PROMPT = """
You are an expert, evidence-first clinical assistant for qualified physicians.
Use ONLY the retrieved textbook context (numbered snippets) and optionally images.
Output must be JSON in one of two formats ONLY.

If you require any clarifying clinical information to answer safely and accurately, respond:
{"clarify":["short question 1?","short question 2?", ...]}

If you have enough to answer, respond:
{"final_answer":"<full multi-section clinical answer here>"}

The final_answer SHOULD be a structured clinical note with these sections:
- SUMMARY: one-line clinical gist.
- DIFFERENTIAL: bullet list with brief reasoning; cite retrieved snippet numbers like [1], [2].
- RECOMMENDED TESTS: specific imaging / labs (priority order), cite sources if textbook supports.
- MANAGEMENT: immediate actions, when to refer, supportive measures.
- PROCEDURE_INDICATIONS: when surgical / interventional therapy is indicated.
- MEDS_AND_LIFESTYLE: general (do NOT provide exact controlled-dose prescriptions; say "consider X class of drugs" unless citation exists).
- UNCERTAINTY: what would change the plan if additional info found.
- REFERENCES: map the [n] numbers to book + page (derived from snippets).

Only return the required JSON (no extra commentary). Keep all clarifying questions concise and clinically relevant.
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
            [f"Image {i+1}: {hit['image_path']}" for i, hit in enumerate(image_hits)]
        )
    else:
        image_refs = "No relevant images found."

    system_prompt = (
        "You are an expert AI clinical assistant trained on authoritative medical textbooks. "
        "Your task is to synthesize context from both text and image evidence to provide a professional, "
        "evidence-based medical explanation. "
        "If critical clinical details are missing, ask only relevant clarifying questions."
    )

    prompt_text = f"""
{system_prompt}

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
"""

    return prompt_text

# ---------------- GENERATE WITH IDEFICS ----------------
def generate_with_idefics(processor, model, prompt_text: str, images: Optional[List[Image.Image]] = None) -> str:
    # ensure <image> tokens match number of images
    images = images or []
    if images:
        n_needed = len(images)
        cur = prompt_text.count("<image>")
        if cur < n_needed:
            prompt_text = ("<image>\n" * (n_needed - cur)) + prompt_text
    inputs = processor(text=prompt_text, images=images if images else None, return_tensors="pt")
    # move to device (model may be on GPU)
    inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return decoded

# ---------------- MAIN INTERACTIVE ----------------
def main():
    logger.info("Device: %s", DEVICE)
    # load models
    proc, idef_model, local_ok = load_idefics(LOCAL_IDEFICS_SNAPSHOT)
    text_enc, clip_model, clip_tok, clip_transforms = load_retrieval()
    client, text_col, image_col = connect_chroma()
    # session
    session_id = input("Session ID (leave blank to create new): ").strip() or str(uuid.uuid4())[:8]
    session = load_session(session_id)
    logger.info("Session: %s", session_id)

    # main loop
    while True:
        user_query = input("\nEnter clinical question (or 'exit' to quit): ").strip()
        if not user_query:
            print("Please enter a question or 'exit'.")
            continue
        if user_query.lower() in ("exit", "quit"):
            save_session(session_id, session)
            logger.info("Saved session %s", session_id)
            break

        # show current patient memory (brief)
        print("\nCurrent patient memory snapshot (patient fields):")
        for k, v in session.get("patient", {}).items():
            print(f"- {k}: {v}")

        # retrieve context
        text_hits, image_hits = retrieve(user_query, text_enc, clip_model, clip_tok, text_col, image_col)
        # add to session history
        session["history"].append({"role": "user", "query": user_query, "time": time.time()})

        # ask if user wants to include a patient image now
        imgpath = input("Optional: local image path to include in this query (press Enter to skip): ").strip() or None
        user_images = []
        if imgpath:
            if os.path.exists(imgpath):
                try:
                    user_images.append(Image.open(imgpath).convert("RGB"))
                except Exception as e:
                    logger.warning("Cannot open image %s: %s", imgpath, e)
            else:
                logger.warning("Image path does not exist: %s", imgpath)

        # build prompt and run iterative clarify loop
        clinical_ctx = session.get("patient", {})
        prompt = build_prompt(user_query, clinical_ctx, text_hits, image_hits)
        rounds = 0
        last_model_text = None
        while rounds < MAX_CLARIFY_ROUNDS:
            rounds += 1
            logger.info("Round %d generate...", rounds)
            images_to_send = user_images[:1]  # include user image first if present
            # include top retrieved image for context if available
            if image_hits:
                p = image_hits[0].get("image_path")
                if p and os.path.exists(p):
                    try:
                        images_to_send.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        logger.warning("Cannot open retrieved image %s: %s", p, e)
            raw = generate_with_idefics(proc, idef_model, prompt, images=images_to_send if images_to_send else None)
            last_model_text = raw.strip()
            # Try parse JSON — model was instructed to return JSON only
            parsed = None
            try:
                parsed = json.loads(last_model_text)
            except Exception as e:
                # sometimes model returns newline then JSON; try to find first '{'
                try:
                    s = last_model_text[last_model_text.find("{"):]
                    parsed = json.loads(s)
                except Exception:
                    logger.warning("Model did not return valid JSON. Printing raw output and treating as final_answer.")
                    parsed = {"final_answer": last_model_text}
            # If it's clarify, ask those questions to user and add to patient memory
            if "clarify" in parsed and isinstance(parsed["clarify"], list) and len(parsed["clarify"])>0:
                clar_qs = parsed["clarify"]
                print("\nModel requests clarifying questions (only the ones it deemed necessary):")
                answers = {}
                for q in clar_qs:
                    a = input(q + " ").strip()
                    answers[q] = a or None
                # merge answers into patient memory (use keys as sanitized question text)
                for q, a in answers.items():
                    key = q.strip().lower().replace(" ", "_")[:60]
                    session.setdefault("patient", {})
                    session["patient"][key] = a
                save_session(session_id, session)
                # rebuild prompt with updated clinical_ctx
                clinical_ctx = session.get("patient", {})
                prompt = build_prompt(user_query, clinical_ctx, text_hits, image_hits)
                continue  # next round
            else:
                # final answer scenario
                final_text = parsed.get("final_answer") if isinstance(parsed, dict) else last_model_text
                # store in session
                session["history"].append({"role": "assistant", "answer": final_text, "time": time.time(), "retrieved_text_ids": [h.get("metadata",{}).get("filename") for h in text_hits]})
                save_session(session_id, session)
                # Print retrieved images for audit
                print("\n--- Retrieved image paths (audit):")
                for ih in image_hits:
                    print("-", ih.get("image_path"))
                print("\n=== FINAL ANSWER (structured) ===\n")
                print(final_text)
                print("\n--- End of answer ---\n")
                break
        # end iterative rounds

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        logger.exception("Fatal pipeline error: %s", e)
    finally:
        logger.info("Completed in %.1fs", time.time() - t0)
