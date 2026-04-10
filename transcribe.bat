@echo off
cd /d %~dp0
if "%~1"=="" (
    echo Uso: transcribe.bat "audio.m4a" [--voices_folder "voices"] [--language es] [opciones]
    echo.
    echo Opciones: --compress --skip_enhance --verbose --model --token --output_format --grouping
    pause
    exit /b 1
)
python -m speechlib %*
pause
