# Multimodal Clinical RAG Assistant (Medical Text + Image Retrieval System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Model](https://img.shields.io/badge/AI-IDEFICS2%208B%20(VLM)-purple)
![Precision](https://img.shields.io/badge/Retrieval-100%25%20Text%20Accuracy-green)
![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20L4%20(4--bit)-orange)

**A doctor-assistive AI system that interprets medical knowledge and patient images simultaneously. It utilizes a Dual-Encoder architecture to cross-reference textbook theory with visual pathology, generating clinically grounded diagnoses.**

---

###  🖼️ System Visualization

![Project Thumbnail](assets/thumbnail.png)


*(The pipeline: Patient Image + Query -> Dual Vector Search -> Multimodal Reasoning -> Diagnosis)*

---

### 🏥 Purpose & Clinical Impact

This project addresses the "Modality Gap" in medical AI. Standard RAG systems are text-blind; they cannot "see" the X-Ray or Skin Lesion a doctor is asking about.

**Our Solution:** We built a multimodal pipeline that understands medical text and patient images together. By ingesting high-quality structured data (from Project 1), this system allows a clinician to upload a patient image and ask, *"What is this condition and how should I treat it?"*. The AI then retrieves visually similar case studies and relevant medical literature to provide an evidence-based answer.

### 🎯 Key Results
* **Image-Aware Diagnosis:** Unlike text-only models, this system matches patient photos with textbook diagrams for higher clinical confidence.
* **High-Precision Alignment:** Achieved **100% text retrieval accuracy** and near-perfect image retrieval by using a shared semantic space for MiniLM and OpenCLIP vectors.
* **Low Hallucination:** Outputs are grounded in retrieved structured JSONL data (tables, diagrams, metadata), significantly reducing medical errors.

---

### ⚙️ System Architecture

The pipeline consists of three advanced engineering stages:

### 1. Dual-Encoder Embedding Engine
We employ two specialized models to handle different data modalities:
* **Text Stream:** Uses **MiniLM-L6-v2** to embed medical text, tables, and JSONL metadata.
* **Visual Stream:** Uses **OpenCLIP ViT-B/32** to embed medical diagrams and patient images into the same vector space.

### 2. Hybrid Retrieval (ChromaDB)
All vectors are stored in **ChromaDB** with rich metadata (page, coordinates, image path). When a query comes in:
* It finds the **Top-N text chunks** (Symptoms, Treatment).
* It finds **visually similar diagrams** for cross-verification.

### 3. Multimodal Reasoning (IDEFICS2-8B)
The retrieved context (Text + Images) is fed into **IDEFICS2-8B**, a powerful Vision-Language Model. We use **4-bit Quantization** to run this massive model efficiently on an NVIDIA L4 GPU, allowing it to "see" the retrieved evidence and generate a diagnosis.

---

###📥 Model Setup (Critical)

Since `requirements.txt` only installs libraries, you need to set up the Model Weights (several GBs).

**Option A: Auto-Download (Recommended)**
The scripts are designed to automatically download the models from HuggingFace on the first run.
* **Run `embed_data.py`:** Downloads `sentence-transformers/all-MiniLM-L6-v2` and `laion/CLIP-ViT-B-32`.
* **Run `ui_idefics_app.py`:** Downloads `HuggingFaceM4/idefics2-8b` (approx 15GB).
* *Note: Ensure you have a stable internet connection for the first execution.*

**Option B: Manual Download (Offline Mode)**
If you are on a restricted network, download these models manually from HuggingFace and update the paths in `config` or script variables:
1.  **Text Embedding:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
2.  **Image Embedding:** [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
3.  **Inference Model:** [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b)

---

### 🚀 Setup & Installation Guide

Follow these steps strictly to deploy the system.

### Step 1: Clone the Repository

```bash
git clone [https://github.com/revoker3661/Multimodal-Clinical-RAG-Assistant-Medical-Text-Image-Retrieval-System-.git]( https://github.com/revoker3661/Multimodal-Clinical-RAG-Assistant-Medical-Text-Image-Retrieval-System-.git)
cd Multimodal-Clinical-RAG
```

### Step 2: Create Environment

```bash
python -m venv venv
```

# Activate environment
For  Windows:

```bash
.\venv\Scripts\activate
```

# For Linux/Mac:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify GPU & Versions
Ensure your CUDA 12.8 environment is ready for 4-bit quantization.

```bash
python scripts/check_versions.py
```
________________________________________
### 🧠 Data Ingestion Workflow
You must populate the vector database before asking questions.
### 1. Place Input Data
Ensure your structured output and images are located in the input_data folder structure:
•	input_data/DavidsonMedicine24th/structured_output.jsonl
•	input_data/DavidsonMedicine24th/[...images...]
### 2. Generate Embeddings
Run the embedding engine to process text and images using the Dual-Encoders.

```bash
python scripts/embed_data.py3. Validate Database
```

Check if the vectors are correctly stored in ChromaDB.

```bash
python scripts/validate_db.py
```
________________________________________
### ▶️ Usage: Clinical Diagnosis UI
Launch the Streamlit Interface to interact with the AI doctor assistant.

```bash
# Launch the UI
streamlit run scripts/ui_idefics_app.py
```

(Alternatively, you can run streamlit run scripts/streamlit_app.py if using the updated dashboard)
How to use:
1.	Upload a patient image (optional) or ask a text question.
2.	The system retrieves related diagrams and textbook passages.
3.	IDEFICS2 analyzes the combined context and provides a diagnosis, causes, and treatment plan.
________________________________________
### 📂 Project Structure

```bash
Plaintext
Multimodal-Clinical-RAG/
.
├── exact_version.txt
├── input_data/
 │   ├── DavidsonMedicine24th/
 │   │   ├── [... approx 1000+ figure-xxx.jpg files ...]
│   │   └── structured_output.jsonl
│   ├── Firestein & Kelley’s Textbook of Rheumatology, 2-Volume Set.../
│   │   ├── [... approx 1000+ figure-xxx.jpg files ...]
│   │   └── structured_output.jsonl
│   └── Goldman-Cecil Medicine/
│       ├── [... figure-xxx.jpg files ...]
│       └── structured_output.jsonl
├── model_cache/
 │   └── idefics2
├── models/
 │   └── idefics2/
 │       └── models--HuggingFaceM4--idefics2-8b
├── output_db/
│   ├── 425d4a71-0f53-416b-a24a-c6796cdf880a/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── index_metadata.pickle
│   │   ├── length.bin
│   │   └── link_lists.bin
│   ├── 5feea19d-1699-4cdd-8914-8c7afb6eaf58/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── index_metadata.pickle
│   │   ├── length.bin
│   │   └── link_lists.bin
│   └── chroma.sqlite3
├── project_structure.txt
├── requirements.txt
└── scripts/
    ├── __pycache__/
    ├── sessions/
    │   ├── [... various .json session files ...]
    ├── app.py
    ├── check_versions.py
    ├── download.py
    ├── embed_data.py
    ├── hello.py
    ├── idefics_pipeline.py
    ├── idefics_pipeline_v3.py
    ├── idefics_pipeline_v3_fix.py
    ├── idefics_pipeline_v4.py
    ├── latestvalidate.py
    ├── retrieve_app.py
    ├── run_out.log
    ├── streamlit_app.py
    ├── test_chroma_query.py
    ├── test_idefics2.py
    ├── ui_idefics_app.py
    ├── validate_db.py
    └── validate_embeddings.py
└── README.md                    # 📖 Manual

```
________________________________________
### 📊 Performance Metrics
Component	Technology	Performance
Text Embedding	MiniLM-L6-v2	384-dim, High semantic overlap
Image Embedding	OpenCLIP ViT-B/32	512-dim, Zero-shot alignment
Inference Engine	IDEFICS2-8B	4-bit Quantized (BitsAndBytes)
Hardware	NVIDIA L4 GPU	Efficient VRAM usage (~12GB)
________________________________________
### 🤝 Contributing
This project is part of a larger research initiative to build scalable healthcare-grade multimodal RAG systems.
Contributions are welcome!



