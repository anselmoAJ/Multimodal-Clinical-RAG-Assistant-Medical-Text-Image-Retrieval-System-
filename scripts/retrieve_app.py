import os
import torch
import logging
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DB_PATH = "../output_db"
TEXT_COLLECTION = "medical_text_collection"
IMAGE_COLLECTION = "medical_image_collection"

TEXT_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_TEXT = 5
TOP_K_IMAGE = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------
def load_models():
    logging.info("Loading text model (BioBERT)...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)

    logging.info("Loading image model (CLIP)...")
    clip_model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)

    return text_model, clip_model, clip_processor


# -------------------------------------------------------------
# LOAD COLLECTIONS
# -------------------------------------------------------------
def load_collections():
    client = chromadb.PersistentClient(path=DB_PATH)
    text_col = client.get_collection(TEXT_COLLECTION)
    image_col = client.get_collection(IMAGE_COLLECTION)
    logging.info("✅ Collections loaded successfully.")
    return text_col, image_col


# -------------------------------------------------------------
# SEARCH FUNCTIONS
# -------------------------------------------------------------
def search_text(query, text_col, text_model):
    logging.info(f"Running text search for: '{query}'")
    query_emb = text_model.encode([query], convert_to_numpy=True)
    results = text_col.query(query_embeddings=query_emb, n_results=TOP_K_TEXT)
    return results


def search_images(query, image_col, clip_model, clip_processor):
    logging.info(f"Running image-based search for: '{query}'")
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        query_emb = clip_model.get_text_features(**inputs).cpu().numpy()
    results = image_col.query(query_embeddings=query_emb, n_results=TOP_K_IMAGE)
    return results


def search_by_image_path(image_path, image_col, clip_model, clip_processor):
    logging.info(f"Running similarity search for image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        query_emb = clip_model.get_image_features(**inputs).cpu().numpy()
    results = image_col.query(query_embeddings=query_emb, n_results=TOP_K_IMAGE)
    return results


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def main():
    text_model, clip_model, clip_processor = load_models()
    text_col, image_col = load_collections()

    print("\n=== HYBRID RETRIEVAL INTERFACE ===")
    print("Options:")
    print("1. Text Query")
    print("2. Image Query")
    print("3. Exit")

    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        if choice == "3":
            print("Exiting...")
            break

        elif choice == "1":
            query = input("Enter your medical text query: ").strip()
            if not query:
                continue

            # --- Text Search ---
            text_results = search_text(query, text_col, text_model)
            print("\nTop Text Matches:")
            for i, (meta, doc) in enumerate(zip(text_results["metadatas"][0], text_results["documents"][0])):
                print(f"[{i+1}] Book: {meta.get('book_filename')}, Page: {meta.get('page_number')}, Section: {meta.get('section_title')}")
                print(f"     Text: {doc[:300]}...\n")

            # --- Image Search ---
            img_results = search_images(query, image_col, clip_model, clip_processor)
            print("\nTop Image Matches:")
            for i, meta in enumerate(img_results["metadatas"][0]):
                print(f"[{i+1}] Book: {meta.get('book_filename')}, Page: {meta.get('page_number')}, Section: {meta.get('section_title')}")
                print(f"     Path: {meta.get('image_path')}")
            print("\n---------------------------------------------")

        elif choice == "2":
            image_path = input("Enter full image path: ").strip()
            if not os.path.exists(image_path):
                print("❌ Invalid path.")
                continue

            img_results = search_by_image_path(image_path, image_col, clip_model, clip_processor)
            print("\nTop Matching Images:")
            for i, meta in enumerate(img_results["metadatas"][0]):
                print(f"[{i+1}] Book: {meta.get('book_filename')}, Page: {meta.get('page_number')}, Section: {meta.get('section_title')}")
                print(f"     Path: {meta.get('image_path')}")
            print("\n---------------------------------------------")

        else:
            print("Invalid choice, try again.")


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
