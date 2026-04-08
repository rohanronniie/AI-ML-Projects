Write-Host "Cloning repository..."
git clone https://github.com/rohanronniie/Applied-AI-ML-Projects.git _temp_upload
Write-Host "Copying files..."
robocopy . _temp_upload\Fake-News-Detection /MIR /XD .git _temp_upload .gemini
Set-Location _temp_upload
Write-Host "Committing..."
git add .
git commit -m "feat: complete rewrite of Neurolingual Fake News Detection with Premium UI"
Write-Host "Pushing..."
git push origin main
Set-Location ..
Write-Host "Cleaning up..."
Remove-Item -Recurse -Force _temp_upload
Write-Host "Done!"
