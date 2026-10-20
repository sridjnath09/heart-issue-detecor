# AI Heart Issue Detector

Heart-sound screening: signal analysis + multi-condition classifier + LangGraph + Gemini agent with RAG that writes a readable report and next steps. **Not a medical device.**

- `backend/` FastAPI (`/analyze`, `/health`), LangGraph agent (`app/agent`), TF-IDF RAG (`app/rag`, knowledge in `data/knowledge`), ONNX classifier (`app/model`), `train.py`.
- `frontend/` Vite + React UI (deploy to Netlify).

## Run locally
```
cd backend && python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env   # add GOOGLE_API_KEY (Gemini) for LLM reports (template report used without it)
.venv/bin/uvicorn app.main:app --reload
cd ../frontend && cp .env.example .env && npm install && npm run dev
```
Without `backend/models/heart_model.onnx` the API returns signal metrics + a template report and **no condition prediction**. Train with `backend/train.py` (see docstring) and drop the ONNX file in `backend/models/`.

## Deploy
1. Backend: push repo, create a Render Blueprint from `render.yaml`; set `GOOGLE_API_KEY` and `ALLOWED_ORIGIN` (your Netlify URL).
2. Frontend: Netlify, base dir `frontend`, set `VITE_API_URL` to the Render URL.
