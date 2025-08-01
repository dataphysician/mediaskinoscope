import asyncio
import re, os, faiss, torch, numpy as np
from pathlib import Path
from PIL import Image

import gradio as gr
from fastrtc import WebRTC, VideoStreamHandler, AdditionalOutputs
from transformers import AutoProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from openai import AsyncOpenAI

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ———————————————————————————————
# 1) Model & Index Setup
# ———————————————————————————————
device      = "cuda" if torch.cuda.is_available() else "cpu"
processor   = AutoProcessor.from_pretrained("google/medsiglip-448", use_fast=True)
vision_model= SiglipVisionModel.from_pretrained("google/medsiglip-448").to(device).eval()

idx = faiss.read_index("demo/vector_index/derma.index")
labels = [l.strip() for l in open("demo/vector_index/derma_labels.txt") if l.strip()]

def query_top_k(img: Image.Image, top_k=1):
    inp = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = vision_model(**inp).pooler_output.cpu().numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-6)
    D, I = idx.search(emb.astype("float32"), top_k)
    i0 = I[0][0]
    return (labels[i0], float(D[0,0])) if i0 >= 0 else ("", 0.0)

# ———————————————————————————————
# 2) Frame Handler (synchronous)
# ———————————————————————————————
def make_handler(skip_every=3):
    state = {"count": 0, "last_label": "Waiting for inference…"}
    def handler(frame: np.ndarray):
        state["count"] += 1
        if state["count"] % skip_every == 0:
            pil = Image.fromarray(frame).resize((448,448), Image.BILINEAR)
            try:
                lbl, _ = query_top_k(pil)
                if lbl:
                    state["last_label"] = lbl
                torch.cuda.synchronize()
            except:
                torch.cuda.empty_cache()
        return frame, AdditionalOutputs(state["last_label"])
    return handler

# ———————————————————————————————
# 3) Async GPT Summarization
# ———————————————————————————————
async def summarize_condition(condition: str) -> str:
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": f"Provide a concise summary of the dermatologic condition: {condition}."
        }]
    )
    return resp.choices[0].message.content or ""

# ———————————————————————————————
# 4) Build the Gradio App
# ———————————————————————————————
label_buffer: list[str] = []

def build_ui():
    handler  = make_handler(skip_every=3)
    suppress = {"Normal Skin", "UNKNOWN IMAGE", ""}

    with gr.Blocks() as demo:
        gr.Markdown("## MedSigLIP Live Webcam Classification")

        with gr.Row():
            with gr.Column(scale=0.6):
                camera   = WebRTC(mode="send-receive", modality="video")
                live_box = gr.Textbox(label="🔁 Live Label",
                                     value="Waiting for inference…",
                                     interactive=False)
            with gr.Column(scale=1):
                alert_box = gr.Textbox(label="🚨 Alert",
                                       interactive=False, lines=1)
                dx_info   = gr.Textbox(label="📘 Diagnosis Summary",
                                       interactive=False, lines=5)
                reset_btn = gr.Button("Reset Buffer")

        # 1) Capture frame labels into buffer and update live label
        camera.stream(
            fn=VideoStreamHandler(handler, skip_frames=True),
            inputs=[camera], outputs=[camera], concurrency_limit=1
        )
        def capture_label(label: str):
            if label:
                label_buffer.append(label)
            return gr.update(value=label)
        camera.on_additional_outputs(
            fn=capture_label, inputs=[], outputs=[live_box], queue=False
        )

        # 2) Poll for new alerts (only on change, otherwise no‐op)
        def detect_alert():
            if not label_buffer:
                return gr.update()      # no change
            last = label_buffer[-1]
            if last not in suppress and last != detect_alert.prev:
                detect_alert.prev = last
                return gr.update(value=last)
            return gr.update()          # preserve existing alert_box
        detect_alert.prev = ""

        timer = gr.Timer(0.5, active=True)
        timer.tick(fn=detect_alert, outputs=[alert_box])

        # 3) Only summarize once per new alert, and pause 3s
        async def on_alert(_alert, prev):
            if not label_buffer:
                return prev
            last = label_buffer[-1]
            if last != on_alert.prev:
                on_alert.prev = last
                summary = await summarize_condition(last)
                await asyncio.sleep(3)
                return f"{last.upper()}\n{summary}"
            return prev
        on_alert.prev = ""

        alert_box.change(
            fn=on_alert,
            inputs=[alert_box, dx_info],
            outputs=[dx_info],
            queue=False,
        )

        # 4) Reset: clear everything *including* on_alert.prev
        def reset_all():
            label_buffer.clear()
            detect_alert.prev = ""
            on_alert.prev = ""
            # explicitly clear both boxes
            return gr.update(value=""), gr.update(value="")
        reset_btn.click(
            fn=reset_all,
            outputs=[alert_box, dx_info]
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(share=False)
