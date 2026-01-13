# Chest X-ray AI (Research Demo)

This repository contains a research-oriented AI system for chest X-ray analysis,
developed as part of a diploma / thesis project.

The system performs:
- image upload
- multi-class classification (Negative / Typical / Indeterminate / Atypical)
- confidence estimation
- Grad-CAM visual explanation

⚠️ This tool is intended for **research and educational purposes only** and does **not**
provide a medical diagnosis.

---

## Project Structure

backend/ – FastAPI backend with trained CNN model
frontend/ – Lightweight Vue.js frontend


---

## Model Weights

Due to file size limitations, trained model weights are **not included**.

To run the backend:
1. Place the trained model file  
   `resnet50_siim_cw_es_best.h5`  
   into:
backend/models/


---

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at:
```bash
http://127.0.0.1:8000
```

Frontend Setup
```bash
cd frontend
python -m http.server 5500
```

Open in browser:
```bash
http://127.0.0.1:5500
```

---
*Features*

Uncertainty-aware predictions

Grad-CAM visualization for interpretability

Minimal and transparent UI

Honest handling of ambiguous cases

---

*License*

This project is released for academic and non-commercial use.

