import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import logging
from PIL import Image
import requests
from io import BytesIO

# ======================= CONFIG ==========================
TEXT_COLLECTION_NAME = "medical_text_collection"
IMAGE_COLLECTION_NAME = "medical_image_collection"
CHROMA_DB_PATH = "../output_db"
TEXT_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================= LOGGER ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================= LOAD MODELS =====================
def load_models():
    logging.info(f"Loading text model: {TEXT_MODEL_NAME}")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)

    logging.info(f"Loading image model: {IMAGE_MODEL_NAME}")
    image_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(DEVICE)
    image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)

    return text_model, image_model, image_processor

# ======================= VALIDATION ======================
def validate_collections(client):
    collections = [c.name for c in client.list_collections()]
    logging.info(f"Collections found: {collections}")

    if TEXT_COLLECTION_NAME not in collections:
        raise ValueError(f"❌ Text collection '{TEXT_COLLECTION_NAME}' not found in DB!")
    if IMAGE_COLLECTION_NAME not in collections:
        raise ValueError(f"❌ Image collection '{IMAGE_COLLECTION_NAME}' not found in DB!")

    text_col = client.get_collection(TEXT_COLLECTION_NAME)
    img_col = client.get_collection(IMAGE_COLLECTION_NAME)

    logging.info(f"✅ Text collection count: {text_col.count()}")
    logging.info(f"✅ Image collection count: {img_col.count()}")

   # Sample text
    sample_text = text_col.get(limit=1, include=["embeddings", "metadatas", "documents"])
    if sample_text.get('embeddings') is not None and len(sample_text['embeddings']) > 0:
        emb = sample_text['embeddings'][0]
        logging.info(f"Sample Text Metadata: {sample_text['metadatas'][0]}")
        logging.info(f"Sample Text Embedding Dimension: {len(emb)}")
    else:
        logging.warning("⚠️ No embeddings found in sample text item.")


# Sample image
    sample_img = img_col.get(limit=1, include=["embeddings", "metadatas", "documents"])
    if sample_img.get('embeddings') is not None and len(sample_img['embeddings']) > 0:
        emb = sample_img['embeddings'][0]
        logging.info(f"Sample Image Metadata: {sample_img['metadatas'][0]}")
        logging.info(f"Sample Image Embedding Dimension: {len(emb)}")
    else:
        logging.warning("⚠️ No embeddings found in sample image item.")


    return text_col, img_col

# ======================= RETRIEVAL TEST ======================
def retrieval_test(text_model, image_model, image_processor, text_col, img_col):
    logging.info("=== Running Retrieval Test ===")
    query = "lung cancer diagnosis and imaging features"
    logging.info(f"Test Query: '{query}'")

    # Encode text query (for text collection search)
    text_query_emb = text_model.encode([query], convert_to_numpy=True)[0]

    # Text search
    text_results = text_col.query(
        query_embeddings=[text_query_emb],
        n_results=5
    )
    logging.info(f"Top 5 Text Matches for Query:")
    for i, doc in enumerate(text_results['documents'][0]):
        meta = text_results['metadatas'][0][i]
        logging.info(f"[{i+1}] Page {meta.get('page_number')} | Section: {meta.get('section_title')}")
        logging.info(f"Text: {doc[:200]} ...\n")

    # Encode text query as CLIP text (for image search)
    inputs = image_processor(text=[query], images=None, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features_clip = image_model.get_text_features(**inputs)
    text_features_clip = text_features_clip.cpu().numpy()[0]

    # Image search
    image_results = img_col.query(
        query_embeddings=[text_features_clip],
        n_results=3
    )

    logging.info("Top 3 Image Matches for Query:")
    for i, meta in enumerate(image_results['metadatas'][0]):
        path = meta.get("image_path")
        section = meta.get("section_title")
        page = meta.get("page_number")
        logging.info(f"[{i+1}] Page {page} | Section: {section} | Path: {path}")

# ======================= MAIN ======================
def main():
    logging.info("=== VALIDATION SCRIPT STARTED ===")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    text_model, image_model, image_processor = load_models()

    text_col, img_col = validate_collections(client)

    retrieval_test(text_model, image_model, image_processor, text_col, img_col)

    logging.info("✅ Validation completed successfully.")

if __name__ == "__main__":
    main()
