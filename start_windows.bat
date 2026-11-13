@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [CaRSEL] Criando ambiente Python...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Falha ao criar ambiente. Instale Python 3.11 e tente novamente.
    pause
    exit /b 1
  )
)
echo [CaRSEL] Instalando dependencias...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
  echo Falha na instalacao.
  pause
  exit /b 1
)
echo.
echo Acesse http://127.0.0.1:8000 no navegador.
echo Para parar o servidor, pressione Ctrl+C.
echo.
".venv\Scripts\python.exe" -m uvicorn carsel.api:app --host 127.0.0.1 --port 8000
pause
