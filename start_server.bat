@echo off
cd /d E:\projects\lostitem

set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_XET=1
set TQDM_DISABLE=1
set PYTHONIOENCODING=utf-8

echo Starting server...
.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 > uvicorn_new.log 2>&1
