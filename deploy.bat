@echo off
git add .
git commit -m "Update: %date% %time%"
git pull origin main --rebase
git push origin main
