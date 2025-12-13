# scripts/streamlit_app.py
import streamlit as st
from PIL import Image
from app import run_pipeline

st.set_page_config(page_title="🩺 MedAI Clinical Assistant", layout="wide")

st.title("🩺 MedAI Clinical Assistant")
st.markdown("Ask questions or upload an image for medical reasoning from textbooks.")

query_text = st.text_area("🧠 Enter your medical query:")
uploaded_image = st.file_uploader("📷 Optional: Upload an image (X-ray, lesion, etc.)", type=["png", "jpg", "jpeg"])

if st.button("🔍 Run Analysis"):
    img = None
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

    with st.spinner("Retrieving and reasoning... please wait ⏳"):
        answer, retrieved = run_pipeline(query_text=query_text, query_image=img)

    st.subheader("💡 AI Response")
    st.write(answer)

    with st.expander("📚 View Retrieved Context"):
        for meta, doc in retrieved["text"]:
            st.markdown(f"**{meta.get('book_filename','')} | Page {meta.get('page_number')}**")
            st.write(doc[:800] + "...")
        st.markdown("---")
        for meta in retrieved["images"]:
            st.image(meta.get("image_path"), caption=f"{meta.get('book_filename')} | Page {meta.get('page_number')}")

st.markdown("---")
st.caption("Powered by BioBERT + CLIP + IDEFICS2 + ChromaDB")
