# idefics_pipeline_v3_fix.py
# Run: python idefics_pipeline_v3_fix.py
import os
import sys
import json
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import open_clip
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Collection names (update if yours are different)
TEXT_COLLECTION_NAME = "medical_text_collection"
IMAGE_COLLECTION_NAME = "medical_image_collection"

# thresholds
HIGH_CONF_THRESH = 0.75
LOW_CONF_THRESH = 0.45

# Limits
MAX_IMAGES_TO_SEND = 2
TOP_K_TEXT = 5
TOP_K_IMAGE = 5

# Load retrieval encoders (your existing BioBERT / CLIP)
def load_retrieval_encoders():
    logging.info("Loading retrieval encoders...")
    # Text encoder (SentenceTransformer)
    text_model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    text_encoder = SentenceTransformer(text_model_name, device=DEVICE)
    # OpenCLIP for image retrieval
    clip_model, _, clip_preproc = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model.to(DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return text_encoder, (clip_model, clip_preproc, tokenizer)

# Load IDEFICS2 (already done in your environment)
def load_idefics(model_id="HuggingFaceM4/idefics2-8b", cache_dir=None):
    # if your previous script uses AutoModelForVision2Seq, use the same call
    logging.info("Loading IDEFICS2 model and processor...")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    # remove any 'chat_template' in processor config to avoid leakage
    try:
        if hasattr(processor, 'config') and isinstance(processor.config, dict):
            processor.config.pop('chat_template', None)
    except Exception:
        pass

    model = AutoModelForVision2Seq.from_pretrained(model_id, local_files_only=True, device_map="auto", torch_dtype=torch.float16)
    model.to(DEVICE)
    return model, processor

# Connect ChromaDB
def connect_chroma(path="../output_db"):
    logging.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=path)
    cols = {c.name: c for c in client.list_collections()}
    text_col = client.get_collection(TEXT_COLLECTION_NAME) if TEXT_COLLECTION_NAME in cols else client.create_collection(TEXT_COLLECTION_NAME)
    img_col = client.get_collection(IMAGE_COLLECTION_NAME) if IMAGE_COLLECTION_NAME in cols else client.create_collection(IMAGE_COLLECTION_NAME)
    logging.info(f"Collections found: {list(cols.keys())}")
    return client, text_col, img_col

# Safely compute similarity/confidence from retrieval result
def compute_confidence(text_hits):
    # text_hits expected to have 'distances' or 'scores' depending on Chroma schema
    # We'll compute an average normalized score (if distances -> convert to similarity)
    scores = []
    for hit in text_hits:
        # Chroma's query result might have 'distances' or 'scores'
        if 'score' in hit:
            scores.append(float(hit['score']))
        elif 'distance' in hit:
            # convert distance to similarity (example: sim = 1/(1+distance))
            d = float(hit['distance'])
            scores.append(1.0/(1.0 + d))
        else:
            # fallback: if hit has 'similarity' or no numeric, skip
            v = hit.get('similarity') or hit.get('score')
            if v is not None:
                scores.append(float(v))
    if not scores:
        return 0.0
    return float(np.mean(scores))

# Build the structured prompt
def build_prompt(user_query, clinical_ctx, text_hits, image_hits, include_images=0):
    # clinical_ctx = dict from patient memory (may be empty)
    # text_hits and image_hits = lists of retrieved items (each item has 'document' and 'metadata')
    # include separators to avoid instruction leakage
    sep1 = "\n--- RETRIEVED CONTEXT ---\n"
    sep2 = "\n--- END CONTEXT ---\n"
    sb = []
    sb.append("You are an expert medical assistant for qualified clinicians.")
    sb.append("Answer ONLY using the retrieved context; cite sources (book + page).")
    sb.append("--- BEGIN TASK ---")
    sb.append(f"User Query: {user_query}")
    # patient memory summary
    if clinical_ctx:
        sb.append("\nPatient memory:\n")
        sb.append(json.dumps(clinical_ctx, indent=2))
    # include text hits
    sb.append(sep1)
    for i, t in enumerate(text_hits[:TOP_K_TEXT]):
        doc = t.get('document') or t.get('text') or ""
        meta = t.get('metadata', {})
        sb.append(f"[Text Source {i+1}] ({meta.get('book_filename','unknown')}, Page: {meta.get('page_number','?')}): {doc}\n")
    # image hits
    for i, im in enumerate(image_hits[:TOP_K_IMAGE]):
        meta = im.get('metadata', {})
        img_path = meta.get('image_path') or im.get('image_path') or meta.get('path') or None
        sb.append(f"[Image Source {i+1}] ({meta.get('book_filename','unknown')}, Page: {meta.get('page_number','?')}) Path: {img_path} Description: {meta.get('text_description','')}\n")
    sb.append(sep2)
    sb.append("Tasks:")
    sb.append("1) Provide a concise structured answer: Summary; Differential diagnosis (ranked); Rationale; Recommended immediate tests; Management steps (short-term and long-term); References.")
    sb.append("2) If the evidence is insufficient, return JSON with a key 'clarify' containing a short list of 1-2 concise clinical follow-up questions to ask the user.")
    sb.append("3) Keep the answer professional and cite the retrieved sources (book filename + page number) next to each statement when used.")
    sb.append("--- END TASK ---")
    # If images will be sent, instruct where to place image tokens:
    if include_images > 0:
        # insert <image> tokens placeholders equal to images (processor requires matching tokens)
        img_tokens = " ".join(["<image>"] * include_images)
        sb.insert(3, f"Images: {img_tokens}")
    prompt = "\n".join(sb)
    return prompt

# Defensive JSON parse of model output
def try_parse_json(text):
    # Try to find JSON object in text
    try:
        # sometimes model wraps JSON in ```json ... ```
        start = text.find("{")
        if start == -1:
            return None
        candidate = text[start:]
        parsed = json.loads(candidate)
        return parsed
    except Exception:
        # fallback: attempt naive repair using regex or simple heuristics
        try:
            # remove trailing non-json and parse until balanced braces
            braces = 0
            s = ""
            for ch in text:
                if ch == "{":
                    braces += 1
                if braces > 0:
                    s += ch
                if ch == "}":
                    braces -= 1
                    if braces == 0:
                        break
            if s:
                return json.loads(s)
        except Exception:
            return None
    return None

# Build follow-up questions when confidence is medium/low
def build_followups(user_query, text_hits, clinical_ctx):
    # Simple heuristics: look for keywords and missing critical fields
    followups = []
    q = user_query.lower()
    # duration question for symptom words
    symptoms = ["fever", "pain", "headache", "cough", "breathless", "dyspnea", "shortness of breath", "swelling", "edema", "vomiting", "diarrhoea"]
    for s in symptoms:
        if s in q:
            followups.append(f"How long has the {s} been present (hours/days)?")
            break
    # meds / prior tests
    if "fever" in q or "infection" in q:
        followups.append("Have you taken any antibiotics, antipyretics or other medications for this episode? If yes, which and when?")
    # red flag: chest pain -> ask radiating/ exertional
    if "chest pain" in q or "chest" in q:
        followups.append("Is the chest pain exertional or associated with sweating, nausea, or radiation to the arm/jaw?")
    # add one general
    if len(followups) == 0:
        followups.append("Please state the patient's age, known chronic illnesses (diabetes, hypertension, heart disease), and current medications.")
    # remove duplicates & limit to 2
    uniq = []
    for fq in followups:
        if fq not in uniq:
            uniq.append(fq)
        if len(uniq) >= 2:
            break
    return uniq

# Main generate function
def generate_answer(idefics_model, processor, text_encoder, clip_tuple, text_col, img_col, user_query, local_image_path=None, clinical_ctx=None):
    # Step 1: retrieval (text)
    logging.info("Retrieving text hits from ChromaDB...")
    # Encode query for text retrieval
    q_emb = text_encoder.encode([user_query], convert_to_numpy=True)
    text_res = text_col.query(query_embeddings=q_emb.tolist(), n_results=TOP_K_TEXT)
    # Chroma returns documents in result['documents'] etc. Normalize to list of dicts
    text_hits = []
    if text_res and 'documents' in text_res:
        docs = text_res['documents'][0]
        metadatas = text_res.get('metadatas', [[]])[0]
        ids = text_res.get('ids', [[]])[0]
        scores = text_res.get('distances', [[]])[0]  # distances -> convert
        for d, m, _id, s in zip(docs, metadatas, ids, scores):
            text_hits.append({'document': d, 'metadata': m or {}, 'distance': s})
    # Step 2: retrieval (images) using CLIP
    logging.info("Retrieving image hits from ChromaDB...")
    clip_model, clip_preproc, clip_tokenizer = clip_tuple
    # encode text with open_clip tokenize & encode
    try:
        tokens = open_clip.tokenize([user_query]).to(DEVICE)
        with torch.no_grad():
            clip_text_emb = clip_model.encode_text(tokens).cpu().numpy()
    except Exception:
        # fallback: use text_encoder embeddings (less accurate)
        clip_text_emb = q_emb
    img_res = img_col.query(query_embeddings=clip_text_emb.tolist(), n_results=TOP_K_IMAGE)
    image_hits = []
    if img_res and 'metadatas' in img_res:
        docs = img_res.get('documents', [[]])[0]  # might be empty
        metadatas = img_res.get('metadatas', [[]])[0]
        ids = img_res.get('ids', [[]])[0]
        scores = img_res.get('distances', [[]])[0]
        for d, m, _id, s in zip(docs, metadatas, ids, scores):
            image_hits.append({'document': d, 'metadata': m or {}, 'distance': s})
    # compute confidence from text_hits distances
    confidence = compute_confidence(text_hits)
    logging.info(f"Retrieval confidence (approx): {confidence:.3f}")
    # Decide whether to ask follow-ups
    if confidence >= HIGH_CONF_THRESH:
        # high confidence -> generate final answer
        action = "generate"
    elif confidence < LOW_CONF_THRESH:
        action = "clarify_low"
    else:
        action = "clarify_medium"
    # if clarify -> build followups
    if action.startswith("clarify"):
        followups = build_followups(user_query, text_hits, clinical_ctx or {})
        # return follow-ups without calling heavy generator
        return {"type": "clarify", "questions": followups, "confidence": confidence, "text_hits": text_hits, "image_hits": image_hits}
    # action == generate -> prepare prompt and call idefics
    # Limit images and log them
    send_images = []
    if local_image_path:
        if os.path.exists(local_image_path):
            send_images.append(local_image_path)
    # add top image hits for visual context but limit to MAX_IMAGES_TO_SEND
    for im in image_hits:
        if len(send_images) >= MAX_IMAGES_TO_SEND:
            break
        p = im.get('metadata', {}).get('image_path') or im.get('image_path')
        if p and os.path.exists(p):
            send_images.append(p)
    # log image paths
    logging.info("Images to be sent to model (paths):")
    for p in send_images:
        logging.info("  " + str(p))
    # Build prompt with include_images = len(send_images)
    prompt = build_prompt(user_query, clinical_ctx or {}, text_hits, image_hits, include_images=len(send_images))
    # prepare images list for processor
    images = [Image.open(p).convert("RGB") for p in send_images] if send_images else None
    # call processor + model
    try:
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE, torch.float16)
    except ValueError as e:
        # Common problem: mismatch between <image> tokens and images passed
        # Fix: if processor complains, regenerate prompt without <image> tokens and rely on model instruction
        logging.warning("Processor image token mismatch, retrying without explicit <image> tokens.")
        prompt = build_prompt(user_query, clinical_ctx or {}, text_hits, image_hits, include_images=0)
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        outputs = idefics_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
    # try structured parse
    parsed = try_parse_json(answer)
    if parsed and isinstance(parsed, dict) and 'clarify' in parsed and isinstance(parsed['clarify'], list):
        # model itself returned clarifying questions
        return {"type": "clarify", "questions": parsed['clarify'], "raw": answer, "confidence": confidence}
    # else final answer text
    return {"type": "answer", "text": answer, "raw": answer, "confidence": confidence, "text_hits": text_hits, "image_hits": image_hits}

# Example main loop
def main():
    logging.info("=== IDEFICS2 Multimodal Pipeline (fixed v3) ===")
    text_encoder, clip_tuple = load_retrieval_encoders()
    # load idefics (use local files)
    idefics_model, processor = load_idefics()
    client, text_col, img_col = connect_chroma()
    # simple REPL
    patient_memory = {}
    while True:
        q = input("\nEnter clinical question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        img = input("Optional: local image path (press Enter to skip): ").strip()
        img = img if img else None
        try:
            res = generate_answer(idefics_model, processor, text_encoder, clip_tuple, text_col, img_col, q, local_image_path=img, clinical_ctx=patient_memory)
        except Exception as e:
            logging.exception("Fatal pipeline error:")
            continue
        if res['type'] == 'clarify':
            print("\nAssistant requests clarification (confidence {:.2f}):".format(res.get('confidence',0.0)))
            for i, qq in enumerate(res['questions'], 1):
                print(f" Q{i}. {qq}")
            # ask user answers and store minimal into patient_memory (simple demo)
            for i, qq in enumerate(res['questions'], 1):
                a = input(f"Answer Q{i}: ").strip()
                # naive store
                patient_memory[f"clarify_q{i}"] = a
            # after clarifications, call generate again (simple)
            print("Re-running generation with clarifications...")
            res2 = generate_answer(idefics_model, processor, text_encoder, clip_tuple, text_col, img_col, q, local_image_path=img, clinical_ctx=patient_memory)
            if res2['type'] == 'answer':
                print("\n=== Generated Answer ===\n")
                print(res2['text'])
            else:
                print("Still clarification needed:", res2)
        else:
            print("\n=== Generated Answer ===\n")
            print(res['text'])
    logging.info("Session ended.")

if __name__ == "__main__":
    main()
