import streamlit as st
from PIL import Image
import os, json, time, uuid, traceback
from idefics_pipeline_v3 import (
    load_idefics, load_retrieval, retrieve,
    build_prompt, generate_with_idefics,
    load_session, save_session, DEVICE
)

st.set_page_config(page_title="IDEFICS Clinical Assistant", layout="wide")
st.title("🧠 IDEFICS Clinical Assistant (Multimodal)")
st.markdown(
    "Ask a clinical question, optionally upload an image, and view both text and image evidence along with the model’s structured answer."
)

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def safe_json_parse(raw_output: str):
    """Tries to parse JSON part if exists, else returns full text."""
    try:
        return json.loads(raw_output)
    except Exception:
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end != -1:
                json_candidate = raw_output[start:end]
                return json.loads(json_candidate)
        except Exception:
            pass
    return {"final_answer": raw_output}


# ----------------------------------------------------
# Model Initialization (cached)
# ----------------------------------------------------
@st.cache_resource
def init_models():
    st.info("🔄 Loading models... please wait (first load can take minutes)")
    proc, idef_model, local_ok = load_idefics(
        "/teamspace/studios/this_studio/.cache/huggingface/hub/"
        "models--HuggingFaceM4--idefics2-8b/snapshots/"
        "2c42686c57fe21cf0348c9ce1077d094b72e7698"
    )

    # Load retrieval models
    text_enc, clip_model, clip_tok, clip_transforms = load_retrieval()

    # ✅ Force both encoders to CPU (avoid CUDA mismatch)
    import torch
    try:
        clip_model = clip_model.to("cpu").eval()
        text_enc.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        st.warning(f"⚠️ Could not move encoders to CPU: {e}")

    # Connect to ChromaDB
    import chromadb
    try:
        DB_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../output_db")
        )
        st.write("📁 Using Chroma path:", DB_PATH)
        client = chromadb.PersistentClient(path=DB_PATH)
        names = [c.name for c in client.list_collections()]
        text_col = (
            client.get_collection("medical_text_collection")
            if "medical_text_collection" in names
            else None
        )
        image_col = (
            client.get_collection("medical_image_collection")
            if "medical_image_collection" in names
            else None
        )
        st.success("✅ Connected to ChromaDB successfully")
    except Exception as e:
        st.warning("⚠️ Could not connect to ChromaDB. Retrieval will be skipped.")
        st.text(f"Reason: {e}")
        client, text_col, image_col = None, None, None

    return proc, idef_model, text_enc, clip_model, clip_tok, clip_transforms, text_col, image_col


# ----------------------------------------------------
# Load models
# ----------------------------------------------------
try:
    proc, idef_model, text_enc, clip_model, clip_tok, clip_transforms, text_col, image_col = init_models()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------------------------------
# Session
# ----------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
session_id = st.session_state.session_id
session = load_session(session_id)
st.sidebar.success(f"Session ID: {session_id}")

# ----------------------------------------------------
# User Input
# ----------------------------------------------------
st.subheader("🩺 Ask a clinical question")
user_query = st.text_area("Enter your clinical question:", height=120)
uploaded_image = st.file_uploader("Optional: Upload a patient image", type=["jpg", "jpeg", "png"])
run_btn = st.button("🚀 Run Inference")

# ----------------------------------------------------
# Inference
# ----------------------------------------------------
if run_btn:
    if not user_query.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Running retrieval and generating structured answer..."):
        # Retrieval
        if not text_col or not image_col:
            text_hits, image_hits = [], []
        else:
            try:
                text_hits, image_hits = retrieve(
                    user_query, text_enc, clip_model, clip_tok, text_col, image_col
                )
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.code(traceback.format_exc())
                text_hits, image_hits = [], []

        # Show retrieved text
        st.subheader("📚 Retrieved Text Context")
        if text_hits:
            for i, hit in enumerate(text_hits):
                st.markdown(f"**[{i+1}]** {hit.get('document', '')[:1000]}...")
        else:
            st.info("No relevant medical text found.")

        # Show retrieved images
        st.subheader("🖼️ Retrieved Images")
        if image_hits:
            cols = st.columns(3)
            for i, hit in enumerate(image_hits):
                path = hit.get("image_path") or hit.get("path")
                if path and os.path.exists(path):
                    with cols[i % 3]:
                        st.image(Image.open(path), caption=os.path.basename(path), use_container_width=True)
                else:
                    with cols[i % 3]:
                        st.warning(f"Image missing: {path}")
        else:
            st.info("No relevant images found.")

        # Include uploaded image
        user_images = []
        if uploaded_image:
            try:
                img = Image.open(uploaded_image).convert("RGB")
                user_images.append(img)
                st.image(img, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Image load error: {e}")

        # Build prompt and generate output
        clinical_ctx = session.get("patient", {})
        prompt = build_prompt(user_query, clinical_ctx, text_hits, image_hits)

        try:
            raw_output = generate_with_idefics(proc, idef_model, prompt, images=user_images or None)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.code(traceback.format_exc())
            raw_output = "ERROR: generation failed"

        # ------------------------------------------------
        # Show Full Model Output (Uncut)
        # ------------------------------------------------
        st.subheader("🧾 Full Model Output (Complete Response)")
        st.markdown(
            f"```text\n{raw_output.strip()}\n```"
        )

        # Optional: extracted sections
        if "--- Retrieved image paths" in raw_output or "Retrieved Text Context:" in raw_output:
            pre_json = raw_output.split("=== FINAL ANSWER")[0]
            st.subheader("📘 Retrieved Context + Instructions (From Model Output)")
            st.markdown(f"```\n{pre_json.strip()}\n```")

        # ------------------------------------------------
        # Parse JSON (Non-destructive)
        # ------------------------------------------------
        parsed = safe_json_parse(raw_output)

        # ------------------------------------------------
        # Display final structured answer
        # ------------------------------------------------
        st.subheader("✅ Structured Final Answer")
        st.markdown(parsed.get("final_answer", raw_output))

        # Clarifications if present
        if "clarify" in parsed:
            st.warning("Model requests clarification:")
            for q in parsed["clarify"]:
                st.write("- " + q)

        # Save session
        session.setdefault("history", []).append({
            "query": user_query,
            "answer": parsed.get("final_answer", raw_output),
            "time": time.time()
        })
        try:
            save_session(session_id, session)
        except Exception:
            st.warning("Could not save session history.")

else:
    st.info("Type your question and click **Run Inference**.")
