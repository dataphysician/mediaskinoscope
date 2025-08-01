## ✨ What is MediaSKINoscope?  
MediaSKINoscope harnesses real‑time WebRTC streaming and Google’s MedSigLIP encoder to run vector‑based, zero‑shot analysis of skin conditions directly on live video feeds. It identifies and highlights dermatologic anomalies, such as rashes, lesions, or unusual pigmentation - for both clinicians and content creators.  

---

## 🔍 What It Does
**MedSigLIP Encoder**  
A lightweight, 400 M‑parameter dual‑tower model from Google HAI-DEF that brings medical images and clinical texts into a single embedding space, enabling robust zero‑shot classification and retrieval on edge devices. Trained on paired image–text data covering chest x‑rays, dermatology photos, histopathology slides, ophthalmologic images, and CT/MRI slices, MedSigLIP is designed for strong out‑of‑the‑box performance across modalities in a wide variety of clinical domains.

---

## 🚀 Features
- **Zero‑Shot Flexibility:**  Broad clinical coverage out-of-the-box, with optional domain-specific fine‑tuning. 
- **Live‑Stream Speed:** Analyze every frame in under 100 ms to keep pace with real‑time video.  
- **Human‑Friendly Insights:** Produce concise, natural‑language explanations linked to medical ontologies.  
- **Privacy‑Preserving Inference:** Optional on‑device processing to keep patient data local and secure.
- **Adjustable Sensitivity:** Enable clinicians to tweak alert thresholds per condition for optimal precision and recall. 

---

## 🛠 Tech Stack
- **Live Streaming:** FastRTC for WebRTC‑based video capture and transport, with built‑in Gradio UI for seamless demos.  
- **Embeddings:** MedSigLIP encoder model transforms each frame into a unified medical embedding for zero‑shot tasks.  
- **LLM Inference:** LlamaIndex orchestrates Retrieval‑Augmented Generation (RAG) workflow, and medical data analyses.  
- **TTS Orchestration:** VAPI handles text‑to‑speech generation for spoken alerts, playing LlamaIndex outputs.  
- **Frontend:** Gradio powers a lightweight dashboard and alert overlay, for instant deployment without custom UI code.  
- **Backend:** FastAPI with Uvicorn manages the async pipeline—frame ingestion → MedSigLIP → LlamaIndex → VAPI TTS—for high‑throughput, event‑driven processing.


## Note:
LlamaIndex and VAPI have been removed after the hackathon to limit library dependencies.