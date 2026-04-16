# Gary

Personal gym coach app with:
- FastAPI backend in `src/gary/api`
- Next.js frontend in `web`
- PostgreSQL for persistence

## 1) Local Setup

### Prerequisites
- Python 3.10+
- Node 22+
- Docker

### Environment
Copy `.env.example` to `.env` and adjust values if needed:

```powershell
Copy-Item .env.example .env
```

### Start database
```bash
docker compose up -d
```

### Initialize schema
Windows (PowerShell):
```powershell
.\scripts\init_db.ps1
```

Linux/macOS:
```bash
bash scripts/init_db.sh
```

### Start backend
```bash
pip install .
uvicorn gary.api.main:app --reload
```

Backend endpoints:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`

### Start frontend
```bash
cd web
npm install
npm run dev
```

Frontend:
- `http://127.0.0.1:3000`

## 2) Docker Build

Build API image:
```bash
docker build -f Dockerfile.api -t gary-api .
```

Build web image:
```bash
docker build -f web/Dockerfile -t gary-web ./web
```

## 3) CI

GitHub Actions CI is defined in `.github/workflows/ci.yml` and runs:
- backend install + compile checks + `pytest`
- frontend lint + build

Local tests:

```bash
pip install ".[dev]"
pytest
```

## 4) Dynamic workout flow

1. Open the web app and sign in/sign up first (`/login` is the public entry).
2. **Generate program** (Gemini when `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set; otherwise a safe template).
   - A global LLM call budget is enforced when Gemini is configured (`LLM_MAX_CALLS`, default 10).
3. **Edit** days and exercises (modify/delete), **Save**, or **Refine** with natural-language instructions.
4. From the program hub, **Start this day** to open the workout runner; **Start workout** → **Workout complete** with an optional note.
5. Suggestions are reviewable and require explicit accept/decline/modify decisions.
6. Legacy paths: session exceptions, apply-adjustment, and workout-feedback endpoints remain available for tooling and tests.

## 5) Deployment

Staging deployment guidance and smoke tests are documented in:
- `docs/gcp-staging.md`
