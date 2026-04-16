# GCP Staging Runbook

This runbook describes a baseline staging deploy for Gary on GCP.

## Target Architecture
- `gary-api` on Cloud Run
- `gary-web` on Cloud Run
- Cloud SQL PostgreSQL
- Secret Manager for sensitive env vars

## Prerequisites
- `gcloud` CLI configured
- GCP project selected
- Artifact Registry repository created
- Cloud SQL PostgreSQL instance created

## 1) Build and push images

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
docker build -f Dockerfile.api -t REGION-docker.pkg.dev/PROJECT_ID/gary/gary-api:staging .
docker push REGION-docker.pkg.dev/PROJECT_ID/gary/gary-api:staging

docker build -f web/Dockerfile -t REGION-docker.pkg.dev/PROJECT_ID/gary/gary-web:staging ./web
docker push REGION-docker.pkg.dev/PROJECT_ID/gary/gary-web:staging
```

## 2) Deploy API

Set secrets/env:
- `DATABASE_URL` (Cloud SQL connection details)
- `ALLOWED_ORIGINS` (include your web staging URL)
- `AUTH_MAX_USERS` (default 5)
- `LLM_MAX_CALLS` (default 10)
- `SESSION_TTL_HOURS` (default 24)
- `COOKIE_SECURE=true` for HTTPS
- Optional: `GEMINI_API_KEY` or `GOOGLE_API_KEY`, and `GEMINI_MODEL` (defaults to `gemini-1.5-flash`) for AI program generation. If omitted, the API uses a deterministic template fallback.

Deploy:
```bash
gcloud run deploy gary-api-staging \
  --image REGION-docker.pkg.dev/PROJECT_ID/gary/gary-api:staging \
  --region REGION \
  --allow-unauthenticated
```

## 3) Deploy Web

Set env:
- `NEXT_PUBLIC_API_BASE_URL=https://<gary-api-staging-url>` (browser calls to the API; must match CORS `ALLOWED_ORIGINS`)

Deploy:
```bash
gcloud run deploy gary-web-staging \
  --image REGION-docker.pkg.dev/PROJECT_ID/gary/gary-web:staging \
  --region REGION \
  --allow-unauthenticated
```

## 4) Smoke test checklist

Use these checks after each staging deploy:
- `GET /health` returns `{"status":"ok"}`
- `GET /ready` returns database ready
- `/login` is shown for unauthenticated browser sessions
- Signup blocks once `AUTH_MAX_USERS` is reached
- Generate/refine is blocked once global LLM limit `LLM_MAX_CALLS` is reached (when Gemini is configured)
- Create program -> create session -> create exception -> apply adjustment flow succeeds from web

Local helper scripts:
- PowerShell: `scripts/smoke_test.ps1`
- Bash: `scripts/smoke_test.sh`

## 5) Promotion to production

Before promoting:
- Run CI green on main branch
- Run smoke test checklist on staging
- Confirm Cloud Logging has no critical errors
