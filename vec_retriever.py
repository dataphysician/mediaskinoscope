import faiss
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionModel,
    SiglipTextModel,
    SiglipModel
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models and processor once
def load_models(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    vision_model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    text_model   = SiglipTextModel.from_pretrained(model_id).to(device).eval()
    multi_model  = SiglipModel.from_pretrained(model_id).to(device).eval()
    return processor, vision_model, text_model, multi_model

# Normalize numpy embeddings

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms

# Load index + labels
def load_index_and_labels(index_dir: Path, index_name: str):
    idx = faiss.read_index(str(index_dir / f"{index_name}.index"))
    labels = [l.strip() for l in (index_dir / f"{index_name}_labels.txt").read_text().splitlines() if l.strip()]
    return idx, labels

# Embed a single image

def embed_image(path: Path, processor, vision_model) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = vision_model(**inputs).pooler_output.cpu().numpy().astype("float32")
    return l2_normalize(emb)

# Embed a single text
def embed_text(text: str, processor, text_model) -> np.ndarray:
    inputs = processor(text=[text], padding="max_length", truncation=True,
                       max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        out = text_model(**inputs)
        emb = out.pooler_output.cpu().numpy().astype("float32")
    return l2_normalize(emb)

# Embed both image + text as combined entries

def embed_image_and_text(path: Path, text: str, processor, multi_model) -> tuple[np.ndarray, list[str]]:
    # Preprocess both
    inputs = processor(images=Image.open(path).convert("RGB"),
                       text=[text], padding="max_length",
                       truncation=True, max_length=64,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        out = multi_model(**inputs)
        img_emb = out.image_embeds.cpu().numpy().astype("float32")
        txt_emb = out.text_embeds.cpu().numpy().astype("float32")
    all_emb = np.vstack([l2_normalize(img_emb), l2_normalize(txt_emb)])
    labels = [path.name, text]
    return all_emb, labels

# Retrieve top-k neighbors from index
def retrieve(vec: np.ndarray, index: faiss.Index, labels: list[str], top_k: int):
    D, I = index.search(vec, top_k)
    return [(labels[i], float(D[0, rank])) for rank, i in enumerate(I[0])]

# High-level API
def query_index(
    mode: str,
    query: str,
    index_dir: str,
    index_name: str,
    model_id: str = "google/medsiglip-448",
    top_k: int = 3
):
    processor, vision_model, text_model, multi_model = load_models(model_id)
    idx, labels = load_index_and_labels(Path(index_dir), index_name)
    if mode == "image":
        vec = embed_image(Path(query), processor, vision_model)
        results = retrieve(vec, idx, labels, top_k)
    elif mode == "text":
        vec = embed_text(query, processor, text_model)
        results = retrieve(vec, idx, labels, top_k)
    elif mode == "both":
        vecs, labs = embed_image_and_text(Path(query), query, processor, multi_model)
        # expand labels array for combined mode
        # here we assume index contains both image and text entries
        results = retrieve(vecs, idx, labels, top_k)
    else:
        raise ValueError("Mode must be 'image', 'text', or 'both'.")
    return results



# Sample:
# vec_retriever.query_index(mode="image", query="./demo/skin_lesion.png", index_dir="./vector_index", index_name="derma", model_id="google/medsiglip-448", top_k=3)