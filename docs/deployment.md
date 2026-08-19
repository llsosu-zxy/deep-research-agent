# Deployment Guide

## Local API

```powershell
python scripts\seed_corpus.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /api/health`
- `POST /api/research` with `{"question": "..."}`
- `GET /api/traces`
- `WS /ws/research`

## Docker Compose

```powershell
docker compose -f docker\docker-compose.yml up --build
```

API: `http://localhost:8000`
Gradio UI: `http://localhost:7860`

## GitHub

```powershell
git remote add origin https://github.com/<user>/deep-research-agent.git
git push -u origin main
```

The included `.github/workflows/ci.yml` runs tests and a golden regression smoke
on push and pull requests.

## Hugging Face Space

1. Create a Gradio Space with Python 3.12.
2. Push this repository to the Space.
3. Keep `hf_space_app.py` at the Space root.
4. Add Space secrets for `LLM_BASE_URL` and `LLM_API_KEY` when using a real
   OpenAI-compatible provider.
5. The default mock mode runs on the free CPU tier; GPU accelerators are only
   needed for BGE-M3/reranker or LoRA fine-tuning.

See `hf_space/README.md` for the file checklist.

## Real LLM / RAGAS

Copy `.env.example` to `.env` and set `LLM_PROVIDER=openai_compatible`,
`LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. Then run:

```powershell
python scripts\run_demo.py
python scripts\run_eval.py
$env:RAGAS_ENABLED="1"
python scripts\run_ragas.py
```
