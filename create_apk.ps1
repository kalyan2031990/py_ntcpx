# PowerShell script to create APK for py_ntcpx_v1.1.0
# Requires: Python, buildozer (or alternative APK creation tool)

Write-Host "Creating APK for py_ntcpx_v1.1.0..." -ForegroundColor Green

# Check if buildozer is installed
$buildozerInstalled = pip show buildozer 2>$null
if (-not $buildozerInstalled) {
    Write-Host "Installing buildozer..." -ForegroundColor Yellow
    pip install buildozer
}

# Navigate to project directory
$projectDir = $PSScriptRoot
Set-Location $projectDir

# Create mobile app directory if it doesn't exist
$mobileDir = Join-Path $projectDir "mobile_app"
if (-not (Test-Path $mobileDir)) {
    New-Item -ItemType Directory -Path $mobileDir
}

# Create buildozer.spec if it doesn't exist
$specFile = Join-Path $projectDir "buildozer.spec"
if (-not (Test-Path $specFile)) {
    Write-Host "Creating buildozer.spec..." -ForegroundColor Yellow
    
    $specContent = @"
[app]
title = NTCP Analysis
package.name = ntcpanalysis
package.domain = org.ntcp
version = 1.1.0
requirements = python3,kivy,numpy,pandas,scipy,scikit-learn,matplotlib,xgboost
orientation = portrait

[buildozer]
log_level = 2
"@
    
    Set-Content -Path $specFile -Value $specContent
}

Write-Host "Building APK..." -ForegroundColor Yellow
Write-Host "Note: This may take a long time on first build" -ForegroundColor Yellow

# Build APK
try {
    buildozer android debug
    Write-Host "APK created successfully!" -ForegroundColor Green
    Write-Host "APK location: bin\ntcpanalysis-debug.apk" -ForegroundColor Green
} catch {
    Write-Host "APK build failed: $_" -ForegroundColor Red
    Write-Host "Alternative: Use web-based approach (see APK_PACKAGING_GUIDE.md)" -ForegroundColor Yellow
}

Set-Location $projectDir
