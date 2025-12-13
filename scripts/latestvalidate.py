# ==============================================================================
#                      DATABASE VALIDATION SCRIPT (DEBUG VERSION)
# ==============================================================================
# This version is filled with print statements to find the exact point of failure.
# ==============================================================================

print("--- Script Starting ---")

print("Step 1: Importing libraries...")
import chromadb
from sentence_transformers import SentenceTransformer, models
import torch
print("✅ Libraries imported successfully.")

# --- 1. CONFIGURATION ---
print("Step 2: Defining configuration...")
DB_PATH = '../output_db'
COLLECTION_NAME = "medical_knowledge_base_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'openai/clip-vit-base-patch32' 
print("✅ Configuration defined.")

def validate_database():
    print("\n---> Inside validate_database function...")

    # --- Load the Correct Model for Querying ---
    print("---> Step A: Loading the query model...")
    try:
        word_embedding_model = models.CLIPModel(MODEL_NAME)
        query_model = SentenceTransformer(modules=[word_embedding_model], device=DEVICE)
        print("✅ Query model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not load the CLIP model. {e}")
        return

    # --- Test 1: Connection and Count ---
    print("---> Step B: Connecting to DB and Verifying Count...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        print("    - ChromaDB client created.")
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
        print("    - Collection retrieved.")
        count = collection.count()
        print(f"✅ Connection & Count Successful! Total items: {count}")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to the database. {e}")
        return

    # --- Test 2: Text Search ---
    print("---> Step C: Performing a sample text search...")
    try:
        query_text = "symptoms of arrhythmia"
        print(f"    - Encoding text: '{query_text}'...")
        query_embedding = query_model.encode(query_text, convert_to_numpy=True)
        print("    - Text encoded. Querying the database...")
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            where={"content_type": "text_chunk"}
        )
        
        print("    - Database query finished.")
        
        if results and results.get('documents') and results['documents'][0]:
            print("✅ Text query successful with results.")
        else:
            print("✅ Text query successful with no results.")

    except Exception as e:
        print(f"❌ ERROR during text search: {e}")

    print("---> Function finished.")

# --- Main execution block ---
if __name__ == "__main__":
    print("Step 3: Starting main execution block...")
    validate_database()
    print("--- Script Finished ---")
else:
    print("Script was imported, not run directly.")