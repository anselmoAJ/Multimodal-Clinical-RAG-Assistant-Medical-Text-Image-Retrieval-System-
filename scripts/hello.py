from open_clip import create_model_and_transforms, get_tokenizer
import torch
clip_model, clip_transform, clip_tok = create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.eval().to("cuda")
clip_tokenizer = get_tokenizer("ViT-B-32")

text = ["Explain the pathophysiology and clinical features of acute myocardial infarction"]
tokens = clip_tokenizer(text).to("cuda")
with torch.no_grad():
    emb = clip_model.encode_text(tokens)
print(emb.shape)
