from transformers import AutoModelForImageTextToText, AutoProcessor
import torch, os

model_dir = "/teamspace/studios/this_studio/Medical_Bot/models/idefics2"

os.makedirs(model_dir, exist_ok=True)

model_id = "HuggingFaceM4/idefics2-8b"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=model_dir
)

processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_dir)
