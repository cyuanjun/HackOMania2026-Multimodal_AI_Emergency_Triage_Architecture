# HackOMania 2026 - PAB Operator Dashboard

This repo is scaffolded as a single repo with:

- `backend/` FastAPI API
- `frontend/` Next.js app
- `data/seed/` demo seed data (profiles)

## Run backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend default URL: `http://127.0.0.1:8000`

## Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: `http://localhost:3000`

Set `NEXT_PUBLIC_API_BASE_URL` if backend is not at `http://127.0.0.1:8000`.
