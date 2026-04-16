param(
  [string]$ApiBaseUrl = "http://127.0.0.1:8000"
)

Write-Host "Health check..."
Invoke-RestMethod -Uri "$ApiBaseUrl/health" -Method Get

Write-Host "Readiness check..."
Invoke-RestMethod -Uri "$ApiBaseUrl/ready" -Method Get

Write-Host "Smoke checks passed (health + ready)."
