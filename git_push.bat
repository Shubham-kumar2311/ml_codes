@echo off
setlocal

set REPO_URL=https://github.com/Shubham-kumar2311/ml_codes.git

echo 🚀 Initializing Git...
git init

echo 📂 Adding all files...
git add .

echo 💬 Committing changes...
git commit -m "Force upload on %date% %time%"

echo 🌐 Connecting to remote repo...
git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%

echo ⬆️ Pushing files to GitHub (forcefully)...
git branch -M main
git push -u origin main --force

echo 🧹 Removing .git folder...
rmdir /s /q .git

echo.
echo ✅ Upload complete and .git folder removed.
echo.
pause
