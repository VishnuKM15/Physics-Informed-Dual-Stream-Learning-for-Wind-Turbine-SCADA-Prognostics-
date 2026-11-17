# ==============================================================================
# PowerShell Script to Deploy CARE Edge Model & Dashboard to Raspberry Pi
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   CARE Wind Turbine Prognostics - Raspberry Pi Deployment Utility    " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$sourceDir = Join-Path $PSScriptRoot "edge_pi"
if (-not (Test-Path $sourceDir)) {
    Write-Host "[!] Error: edge_pi folder not found at $sourceDir" -ForegroundColor Red
    Exit
}

$pkgSizeMb = [math]::Round(((Get-ChildItem -Recurse $sourceDir | Measure-Object -Property Length -Sum).Sum / 1MB), 2)
Write-Host "[*] Found deployment package in: edge_pi" -ForegroundColor Green
Write-Host "[*] Total package size: $pkgSizeMb MB (Lightweight & Pi-ready!)" -ForegroundColor Green
Write-Host ""

Write-Host "Choose deployment method:" -ForegroundColor Yellow
Write-Host "  1. Direct Transfer via SSH / SCP (Over Wi-Fi or Ethernet)"
Write-Host "  2. Package into ZIP file for USB drive or manual transfer"
Write-Host ""
$choice = Read-Host "Select option [1 or 2] (Default: 1)"
if (-not $choice) { $choice = "1" }

if ($choice -eq "2") {
    $zipPath = Join-Path $PSScriptRoot "edge_pi_deployment.zip"
    Write-Host "[*] Compressing edge_pi into $zipPath ..." -ForegroundColor Yellow
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path "$sourceDir\*" -DestinationPath $zipPath
    Write-Host "[+] Successfully created ZIP package: $zipPath" -ForegroundColor Green
    Write-Host "[+] Copy this ZIP file to your Raspberry Pi and unzip it." -ForegroundColor Green
    Exit
}

# SSH / SCP Deployment
$piHost = Read-Host "Enter Raspberry Pi IP address or hostname [e.g. 192.168.1.100 or raspberrypi.local]"
if (-not $piHost) { $piHost = "raspberrypi.local" }

$piUser = Read-Host "Enter Raspberry Pi username [Default: pi]"
if (-not $piUser) { $piUser = "pi" }

$remoteDir = Read-Host "Enter destination directory on Raspberry Pi [Default: /home/$piUser/renewable_project]"
if (-not $remoteDir) { $remoteDir = "/home/$piUser/renewable_project" }

Write-Host ""
Write-Host "[*] Creating remote directory on Raspberry Pi..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no "${piUser}@${piHost}" "mkdir -p ${remoteDir}"

Write-Host "[*] Transferring files via SCP..." -ForegroundColor Yellow
scp -r -o StrictHostKeyChecking=no "$sourceDir/*" "${piUser}@${piHost}:${remoteDir}/"

Write-Host ""
Write-Host "[*] Setting execution permissions on run_dashboard.sh..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no "${piUser}@${piHost}" "chmod +x ${remoteDir}/run_dashboard.sh"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   DEPLOYMENT TO RASPBERRY PI COMPLETED SUCCESSFULLY!                 " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To launch the dashboard on your Raspberry Pi, run:"
Write-Host "   ssh ${piUser}@${piHost}" -ForegroundColor Cyan
Write-Host "   cd ${remoteDir}" -ForegroundColor Cyan
Write-Host "   ./run_dashboard.sh" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then open your browser to:" -ForegroundColor Yellow
Write-Host "   http://${piHost}:5000" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
