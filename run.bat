@echo off
title He Thong Giam Sat Tai Xe - DMS
echo ===================================================================
echo   KHOI DONG HE THONG PHAT HIEN BUON NGU REAL-TIME
echo ===================================================================
echo.

:: Xoa anh xa o Z: cu neu co va anh xa thu muc hien tai thanh Z:
if exist Z:\ (
    subst Z: /d >nul 2>&1
)
subst Z: "%~dp0"

if not exist Z:\ (
    echo [Canh bao] Khong the tao o dia ao Z:, chay truc tiep...
    .\venv\Scripts\python.exe DrowsinessDetector.py
) else (
    echo [Thanh cong] Da chuyen thu muc sang o dia sach: Z:\
    cd /d Z:\
    Z:\venv\Scripts\python.exe DrowsinessDetector.py
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [Thong bao] Dang chuyen sang che do nhe run_opencv.py...
        Z:\venv\Scripts\python.exe run_opencv.py
    )
    cd /d "%~dp0"
    subst Z: /d >nul 2>&1
)

echo.
echo Ung dung da ket thuc.
pause
