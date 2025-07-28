import os
import faiss
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionModel,
    SiglipTextModel,
    SiglipModel
)
from pydantic import FilePath, DirectoryPath

def resize_image_file(file_path: FilePath) -> Image.Image:
    """
    Resize a PIL Image to 448×448 using bilinear interpolation.
    """
    file_path = Path(file_path)
    image = Image.open(file_path).convert("RGB")
    return image.resize((448, 448), resample=Image.BILINEAR)

def build_index(
    index_name: str,
    texts: list[str],
    images: list[FilePath],
    embedding_model: str,
    output_dir: str
):
    """
    Builds one FAISS IndexFlatIP over all items at once.
    Args:
      index_name:       e.g. "derma"
      items:            List of N label strings
      embedding_model:  HF model ID (e.g. "google/medsiglip-448")
      output_dir:       Directory to write `{index_name}.index` and `{index_name}_labels.txt`
    """

    if not embedding_model:
        embedding_model = "google/medsiglip-448"

    os.makedirs(output_dir, exist_ok=True)

    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    if images and all(img_file.suffix.lower() in img_exts for img_file in images):
        resized_images = [resize_image_file(img_path) for img_path in images]
    else:
        if images:
            invalid = [str(img_file) for img_file in images if img_file.suffix.lower() not in img_exts]
            raise ValueError(f"Unsupported image extensions in: {invalid}")
        resized_images = []

    # 1. Load model + processor once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision_model = SiglipVisionModel.from_pretrained(embedding_model).to(device).eval()
    text_model   = SiglipTextModel.from_pretrained(embedding_model).to(device).eval()
    multi_model  = SiglipModel.from_pretrained(embedding_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(embedding_model)

    # 2. AutoProcessor kwargs
    proc_kwargs = {"images":None, "text":None, "padding":"max_length", "return_tensors":"pt"}

    if resized_images and texts:
        proc_kwargs["text"] = texts
        proc_kwargs["images"] = resized_images
        model = multi_model

    if texts:
        proc_kwargs["text"] = texts
        model = text_model
        
    if resized_images:
        proc_kwargs["images"] = resized_images
        model = vision_model

    if not ("text" in proc_kwargs or "images" in proc_kwargs):
        raise ValueError("Must supply at least one of `texts` or `images` to embed")

    inputs = processor(**proc_kwargs).to(device)

    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Gather Embeddings
    image_labels = [Path(p).name for p in images]
    embs, labs = [], []
    image_embed_count = 0
    txt_embed_count = 0
    if texts and images:
        # dual-tower/encode → use text_embeds and image_embeds
        embs.extend([outputs.image_embeds, outputs.text_embeds])
        image_embed_count = len(outputs.image_embeds)
        txt_embed_count = len(outputs.pooler_output)
        labs.extend(image_labels)
        labs.extend(texts)
    elif texts:
        # text only → use pooler_output
        embs.append(outputs.pooler_output)
        txt_embed_count = len(outputs.pooler_output)
        labs.extend(texts)
    else:
        # images only → use pooler_output
        embs.append(outputs.pooler_output)
        image_embed_count = len(outputs.pooler_output)
        labs.extend(image_labels)

    all_embeds = torch.cat(embs, dim=0)

    print(f"Successfully encoded {image_embed_count} images and {txt_embed_count} texts.")

    embeds_np = all_embeds.cpu().numpy().astype("float32") # Send to CPU and convert to Numpy array for FAISS indexing

    # 5. Normalize for cosine similarity (inner-product search)
    norms = np.linalg.norm(embeds_np, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1
    embeds_np = embeds_np / norms

    # 6. Build a single flat IP index
    d     = embeds_np.shape[1]
    index = faiss.IndexFlatIP(d) # Flat, CPU-based exact inner-product index
    index.add(embeds_np)

    # 7. Persist index and labels
    idx_path = os.path.join(output_dir, f"{index_name}.index")
    label_path = os.path.join(output_dir, f"{index_name}_labels.txt")
    faiss.write_index(index, idx_path)

    image_labels = [img_path.name for img_path in images]
    text_labels  = texts
    all_labels   = image_labels + text_labels

    with open(label_path, "w") as f:
        for label in all_labels:
            f.write(label + "\n")
    print(f"{len(all_labels)} items successfully indexed in {index_name} under {label_path}")


def build_index_from_path(
        index_name: str = None,
        text_file: FilePath|None = None,
        images_folder: DirectoryPath|None = None,
        embedding_model: str = None,
        output_dir: str = None
):
    texts = []
    if text_file is not None and Path(text_file).suffix.lower() == ".txt":
        print(f"Loading {text_file}")
        with open(text_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    # else:
    #     raise ValueError(f"{text_file} is not a .txt file")

    if images_folder is not None and Path(images_folder).is_dir():
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        images = [
            img_file for img_file in Path(images_folder).iterdir()
            if img_file.suffix.lower() in img_exts
        ]
        if not images:
            raise ValueError(f"No supported images found in {images_folder}")
    else:
        images = []

    build_index(
        index_name = index_name, 
        texts = texts or [], 
        images = images or [],
        embedding_model = embedding_model, 
        output_dir = output_dir
    )

def batch_index(
    index_map: dict[str, dict[FilePath|DirectoryPath]],
    embedding_model: str,
    output_dir: str
):
    for index_name, path in index_map.items():
        build_index_from_path(
            index_name = index_name,
            text_file = path["texts"],
            images_folder = path["images"],
            embedding_model = embedding_model,
            output_dir = output_dir
        )

# Sample:
# vec_indexer.build_index_from_path(index_name="derma", text_file="./vector_index/derma.txt", embedding_model="google/medsiglip-448", output_dir="./vector_index")