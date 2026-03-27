# 경찰청 습득물 AI 검색 서버 시작 스크립트
# 실행: powershell -ExecutionPolicy Bypass -File start_server.ps1

Set-Location $PSScriptRoot

$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_XET   = "1"
$env:TQDM_DISABLE          = "1"
$env:PYTHONIOENCODING      = "utf-8"

Write-Host "=== 경찰청 습득물 AI 검색 서버 시작 ===" -ForegroundColor Cyan
Write-Host "Swagger UI : http://localhost:8000/docs"    -ForegroundColor Green
Write-Host "Qdrant     : http://localhost:6333/dashboard" -ForegroundColor Green

.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
