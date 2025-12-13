from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

print("🚀 Loading IDEFICS2 model...")

model_id = "HuggingFaceM4/idefics2-8b"
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

print("✅ Model and processor loaded successfully.")
