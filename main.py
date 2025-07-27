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

idx_dir = Path("./vector_index")
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
                    raw = query_top_k(Image.fromarray(frame), top_k=1)
                # raw might contain entries like (label, score) or (-1) hits
                lines = []
                for lbl, sc in raw:
                    # skip any missing labels (e.g. FAISS returned -1)
                    if lbl not in labels:
                        continue
                    lines.append(f"{lbl}")# (Cosine similarity:{sc:.2f})")
                if lines:
                    state["last_txt"] = "\n".join(lines)
                # sync/cleanup
                torch.cuda.synchronize()
            except Exception as e:
                state["last_txt"] = f"Inference error: {e}"
                torch.cuda.empty_cache()

        # return the frame plus a clean string payload
        return frame, AdditionalOutputs(state["last_txt"])

    return handler


import re
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from vapi import Vapi
import requests

# 1. Configure the global LLM
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# 2. Define your summarization prompt
def summarize_condition(condition: str) -> str:
    prompt = f"Provide a concise summary of the dermatologic condition: {condition}."
    # 3. Call the LLM directly
    response = Settings.llm.complete(prompt)
    return response.text

VAPI_TOKEN = Vapi(token=os.getenv("VAPI_TOKEN"))
ASSISTANT_ID= os.getenv("VAPI_ASSISTANT_ID")
SESSION_ID = "derma"
def vapi_tts(text: str) -> bytes:
    """
    Sends `text` to Vapi's chat endpoint (with TTS enabled)
    and returns the raw MP3 bytes of the Rime AI–synthesized speech.
    """
    url = f"https://api.vapi.ai/assistants/{ASSISTANT_ID}/chat"
    headers = {
        "Authorization": f"Bearer {VAPI_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "sessionId": SESSION_ID,
        "messages": [
            {"role": "assistant", "text": text}
        ]
    }

    # 1) Request chat + TTS—Vapi returns both text and audioUrl
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    audio_url = data["audioUrl"]  # URL pointing to Rime AI TTS output :contentReference[oaicite:0]{index=0}

    # 2) Download the MP3 in one go
    audio_resp = requests.get(audio_url)
    audio_resp.raise_for_status()
    return audio_resp.content

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
                textbox = gr.Textbox(label="Latest Label", interactive=False)
                chatbox = gr.Textbox(label="Abnormal Skin Conditions", lines=5, interactive=False)
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
        
        # camera.on_additional_outputs(fn=lambda labels: labels,
        #                              outputs=[textbox],
        #                              queue=False)
        # if textbox.value not in ["Normal Skin", "UNKNOWN IMAGE"]:
        #     chatbox.value == textbox.value

        # # When textbox (latest label) changes, check it and update chatbox
        # # Append to chatbox whenever label is “interesting”
        # def maybe_alert(label: str, current_chat: str) -> str:
        #     suppress = {
        #         "Normal Skin",
        #         "UNKNOWN IMAGE"
        #     }
        #     if not label or label in suppress:
        #         return current_chat
        #     return current_chat + ("\n" if current_chat else "") + label

        # textbox.change(
        #     fn=maybe_alert,
        #     inputs=[textbox, chatbox],
        #     outputs=[chatbox],
        #     queue=False
        # )

    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch()

    # print("Initiating query...")
    # start = time.time()
    # results = query_top_k(image_path="./sample/eczema_herpetica.png", top_k=3)
    # t1 = time.time()
    # results2 = query_top_k(image_path="./sample/AK.png", top_k=3)
    # t2 = time.time()
    # results3 = query_top_k(image_path="./sample/eczema_herpetica.png", top_k=3)
    # t3 = time.time()
    # results4 = query_top_k(image_path="./sample/AK.png", top_k=3)
    # t4 = time.time()
    # results5 = query_top_k(image_path="./sample/ak_upclose.png", top_k=3)
    # t5 = time.time()
    # results6 = query_top_k(image_path="./sample/ak_upclose.png", top_k=3)
    # end = time.time()

    # print(results)
    # print(results2)
    # print(results3)
    # print(results4)
    # print(results5)
    # print(results6)
    # print(f"First Run: {(t1 - start)*1000:.1f} ms")
    # print(f"Second Run: {(t2 - t1)*1000:.1f} ms")
    # print(f"Third Run: {(t3 - t2)*1000:.1f} ms")
    # print(f"Fourth Run: {(t4 - t3)*1000:.1f} ms")
    # print(f"Fifth Run: {(t5 - t4)*1000:.1f} ms")
    # print(f"Sixth Run: {(end - t5)*1000:.1f} ms")
    # print(f"Total time: {(end - start)*1000:.1f} ms")

