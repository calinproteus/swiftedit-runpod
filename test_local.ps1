# Local test script for download_models.py
# This lets you test the download logic WITHOUT waiting for RunPod builds

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing download_models.py locally" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will download ~7GB to your machine." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to cancel, or Enter to continue..." -ForegroundColor Yellow
Read-Host

Write-Host ""
Write-Host "Running download_models.py..." -ForegroundColor Green
python download_models.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Script works locally" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Safe to push and rebuild on RunPod!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ FAILED! Fix errors before pushing" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

