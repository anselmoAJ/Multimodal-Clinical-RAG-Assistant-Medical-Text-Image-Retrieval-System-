# test_chroma_query.py
from idefics_pipeline_v3 import load_retrieval, connect_chroma, _to_2d_float_list
enc, clip, tok, transforms = load_retrieval()
client, text_col, image_col = connect_chroma()
q = "acute kidney injury elderly"
emb = enc.encode([q], convert_to_numpy=True)
emb_list = _to_2d_float_list(emb)
print("Emb shape:", len(emb_list), len(emb_list[0]))
res = text_col.query(query_embeddings=emb_list, n_results=5)
print("Query keys:", res.keys())
print("Docs:", res.get("documents", [[]])[0][:3])

