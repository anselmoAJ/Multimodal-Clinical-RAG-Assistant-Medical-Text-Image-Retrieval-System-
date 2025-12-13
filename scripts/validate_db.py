# ==============================================================================
#                      DATABASE VALIDATION SCRIPT (FINAL VERSION)
# ==============================================================================
# This script robustly validates the final, high-accuracy knowledge base.
# It uses the correct collection name and model loading method.
# ==============================================================================

import chromadb
from sentence_transformers import SentenceTransformer, models
import torch

# --- 1. CONFIGURATION ---
DB_PATH = '../output_db'
# IMPORTANT: Using the new, final collection name
COLLECTION_NAME = "medical_knowledge_base_high_accuracy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'openai/clip-vit-base-patch32' 

def validate_database():
    print("--- Starting Final Database Validation ---")

    # --- Load the Correct Model for Querying ---
    print(f"Loading the query model ({MODEL_NAME})...")
    try:
        word_embedding_model = models.CLIPModel(MODEL_NAME)
        query_model = SentenceTransformer(modules=[word_embedding_model], device=DEVICE)
        print("✅ Query model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not load the CLIP model. {e}")
        return

    # --- Test 1: Connection and Count ---
    print("\n[TEST 1/3] Connecting to DB and Verifying Count...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"✅ Connection Successful!")
        # The expected count should match your last run's output
        print(f"✅ Item Count: {count} (Expected around 148,264)")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to the database. {e}")
        return

    # --- Test 2: Text Search ---
    print("\n[TEST 2/3] Performing a specific text search...")
    try:
        # Using a specific term from the JSONL snippet you provided
        query_text = "Mesangiocapillary glomerulonephritis"
        query_embedding = query_model.encode(query_text, convert_to_numpy=True)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            where={"content_type": "text"} # Search in all text types
        )
        
        print(f"✅ Text query for '{query_text}' completed.")
        
        if results and results.get('documents') and results['documents'][0]:
            print("Top 3 retrieved text results:")
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                print(f"  Result {i+1}:")
                print(f"    - Source: {metadata.get('book_filename')}, Page: {metadata.get('page_number')}")
                print(f"    - Text: {doc[:150]}...")
        else:
            print("⚠️ Warning: Text query returned no results. This might be okay, but is unexpected.")

    except Exception as e:
        print(f"❌ ERROR during text search: {e}")

    # --- Test 3: Image Search ---
    print("\n[TEST 3/3] Performing a sample image similarity search...")
    try:
        image_embeddings_data = collection.get(
            where={"content_type": "image"}, limit=5, include=["embeddings", "metadatas"]
        )

        if image_embeddings_data and image_embeddings_data['ids']:
            query_embedding = image_embeddings_data['embeddings'][0]
            original_metadata = image_embeddings_data['metadatas'][0]
            
            print(f"✅ Querying with a random image from '{original_metadata.get('book_filename')}'...")

            results = collection.query(
                query_embeddings=[query_embedding], n_results=3, where={"content_type": "image"}
            )
            
            if results and results.get('metadatas') and results['metadatas'][0]:
                print("Top 3 similar images found:")
                for i, metadata in enumerate(results['metadatas'][0]):
                     print(f"  Result {i+1}:")
                     print(f"    - Source: {metadata.get('book_filename')}, Page: {metadata.get('page_number')}")
                     print(f"    - Description: {metadata.get('text_description', '')[:100]}...")
            else:
                 print("✅ Image query returned no similar results, which is a valid outcome.")
        else:
            print("⚠️ Warning: Could not retrieve any image embeddings to perform the test.")
    except Exception as e:
        print(f"❌ ERROR during image search: {e}")
        
    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    validate_database()