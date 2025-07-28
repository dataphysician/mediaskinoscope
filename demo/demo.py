import re
import faiss, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import gradio as gr
from fastrtc import WebRTC, VideoStreamHandler, AdditionalOutputs
from transformers import AutoProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 1) Load model & FAISS index
device      = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = "google/medsiglip-448"

processor = AutoProcessor.from_pretrained(embed_model, use_fast=True)
vision_model = (SiglipVisionModel.from_pretrained(embed_model).to(device).eval()# .half()
)
# vision_model = torch.compile(vision_model, mode="reduce-overhead")#"max-autotune")

idx_dir = Path("./demo/vector_index")
index   = faiss.read_index(str(idx_dir/"derma.index"))
labels  = [l.strip() for l in (idx_dir/"derma_labels.txt").read_text().splitlines() if l.strip()]

# 2) Embedding + FAISS lookup
def query_top_k(image: Image.Image, top_k: int = 3):
    img = image.convert("RGB")  # client already sends 448×448
    inp = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = vision_model(**inp).pooler_output.cpu().numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb   = emb / np.where(norms==0, 1, norms)
    D, I  = index.search(emb.astype("float32"), top_k)
    return [(labels[i], float(D[0,rank])) for rank,i in enumerate(I[0])]

_ = query_top_k(Image.new("RGB",(448,448)), top_k=1)  # warm‑up


# 3) Frame handler with throttling
def make_handler(skip_every: int = 3):
    state = {"count": 0, "last_txt": "Waiting for inference…"}
    
    def handler(frame: np.ndarray):
        state["count"] += 1

        # run only every Nth frame
        if state["count"] % skip_every == 0:
            try:
                with torch.no_grad():#, torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    label_score = query_top_k(Image.fromarray(frame), top_k=1)
                lbl, sc = label_score[0]
                if lbl:
                    state["last_txt"] = lbl
                torch.cuda.synchronize()
            except Exception as e:
                state["last_txt"] = f"Inference error: {e}"
                torch.cuda.empty_cache()

        # return the frame plus a clean string payload
        return frame, AdditionalOutputs(state["last_txt"])

    return handler


old_chat = ""
from openai import OpenAI
def summarize_condition(condition):
    result = OpenAI().chat.completions.create(
        model = "gpt-4o",
        temperature = 0.1,
        messages = [
            {"role":"user", "content":f"Provide a concise summary of the dermatologic condition: {condition}."}
        ]
    )
    if valid_response:=result.choices[0].message.content:
        return valid_response
    return

# 4) Build and return the Gradio Blocks app
def build_ui():
    handler = make_handler(skip_every=3)

    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("## MedSigLIP Live Webcam Classification")

        with gr.Row():

            with gr.Column(scale=0.6):
                camera = WebRTC(
                    mode="send-receive",
                    modality="video",
                )
            with gr.Column(scale=1):
                textbox = gr.Textbox(label="Livestream Label", interactive=False)
                chatbox = gr.Textbox(label="Detected Skin Conditions", lines=5, interactive=False)
                diagnosis = gr.Textbox(label="Diagnosis Info", lines=5, interactive=False)

        camera.stream(
            fn=VideoStreamHandler(handler, skip_frames=True),
            inputs=[camera],
            outputs=[camera],
            concurrency_limit=1,
        )

        camera.on_additional_outputs(
            fn=lambda label: label,
            outputs=[textbox],
            queue=False
        )


        suppress = {"Normal Skin","UNKNOWN IMAGE", "[]"}

        def append_if_new(label: str, current_chat: str) -> str:
            # 1) Extract the true label from any patch syntax
            tokens = re.findall(r"'([^']*)'", label)
            clean = tokens[-1] if tokens else label

            # 2) Only append if it's in your known labels *and* not suppressed
            if clean in labels and clean not in suppress:
                lines = current_chat.splitlines()
                if clean not in lines:
                    global old_chat
                    old_chat = current_chat
                    return current_chat + ("\n" if current_chat else "") + clean

            return current_chat

        textbox.change(
            fn=append_if_new,
            inputs=[textbox, chatbox],
            outputs=[chatbox],
            queue=False
        )


        def diagnosis_info(chat):
            if (old_chat is None or old_chat=="") and chat is not None:
                dx_info = summarize_condition(chat)
                return chat.upper() + "\n" + str(dx_info)
        
        chatbox.change(
            fn=diagnosis_info,
            inputs=[chatbox],
            outputs=[diagnosis],
            queue=False
        )


    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch()