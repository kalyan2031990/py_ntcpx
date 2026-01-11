# PowerShell Script: Create GitHub Release for Zenodo DOI (Private Repository)
# For repository: kalyan2031990/py_ntcpx

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Zenodo DOI Setup - Private Repository" -ForegroundColor Cyan
Write-Host "Repository: kalyan2031990/py_ntcpx" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Configuration
$repo = "kalyan2031990/py_ntcpx"
$version = "v1.0.0"
$title = "py_ntcpx $version - Initial Release"

# Step 1: Check GitHub CLI Installation
Write-Host "[Step 1] Checking GitHub CLI installation..." -ForegroundColor Yellow
try {
    $ghVersion = gh --version 2>&1 | Select-Object -First 1
    Write-Host "✓ GitHub CLI is installed: $ghVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ GitHub CLI is NOT installed" -ForegroundColor Red
    Write-Host "`nPlease install GitHub CLI:" -ForegroundColor Yellow
    Write-Host "  Option 1: winget install --id GitHub.cli" -ForegroundColor White
    Write-Host "  Option 2: Download from https://cli.github.com/" -ForegroundColor White
    Write-Host "`nAfter installation, restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Authentication
Write-Host "`n[Step 2] Checking GitHub authentication..." -ForegroundColor Yellow
try {
    gh auth status 2>&1 | Out-Null
    Write-Host "✓ GitHub CLI is authenticated" -ForegroundColor Green
} catch {
    Write-Host "✗ Not authenticated" -ForegroundColor Red
    Write-Host "`nPlease authenticate:" -ForegroundColor Yellow
    Write-Host "  Run: gh auth login" -ForegroundColor White
    Write-Host "  Follow the prompts to authenticate via browser" -ForegroundColor White
    exit 1
}

# Step 3: Verify Repository Access
Write-Host "`n[Step 3] Verifying repository access..." -ForegroundColor Yellow
try {
    $repoInfo = gh repo view $repo 2>&1
    Write-Host "✓ Repository access confirmed" -ForegroundColor Green
    $visibility = gh api repos/$repo --jq '.private'
    if ($visibility -eq "true") {
        Write-Host "  Repository is PRIVATE" -ForegroundColor Cyan
        Write-Host "  ⚠ Make sure Zenodo has access to private repos in GitHub settings" -ForegroundColor Yellow
    } else {
        Write-Host "  Repository is PUBLIC" -ForegroundColor Cyan
    }
} catch {
    Write-Host "✗ Cannot access repository: $repo" -ForegroundColor Red
    Write-Host "  Make sure the repository exists and you have access" -ForegroundColor Yellow
    exit 1
}

# Step 4: Check Existing Releases
Write-Host "`n[Step 4] Checking for existing releases..." -ForegroundColor Yellow
$existingReleases = gh release list --repo $repo --limit 5 2>&1
if ($existingReleases -match $version) {
    Write-Host "⚠ Release $version already exists!" -ForegroundColor Yellow
    $overwrite = Read-Host "  Do you want to delete and recreate it? (y/n)"
    if ($overwrite -eq "y") {
        Write-Host "  Deleting existing release..." -ForegroundColor Yellow
        gh release delete $version --repo $repo --yes
        git tag -d $version 2>&1 | Out-Null
        git push origin :refs/tags/$version 2>&1 | Out-Null
        Write-Host "  ✓ Existing release deleted" -ForegroundColor Green
    } else {
        Write-Host "  Aborted. Using existing release." -ForegroundColor Yellow
        exit 0
    }
} else {
    Write-Host "✓ No existing release with tag $version" -ForegroundColor Green
}

# Step 5: Prepare Release Notes
Write-Host "`n[Step 5] Preparing release notes..." -ForegroundColor Yellow
$releaseNotes = @"
## Initial Release - py_ntcpx $version

First release of the redesigned NTCP Analysis and Machine Learning Pipeline for Head & Neck Cancer.

### Features
- Complete pipeline redesign with enhanced features
- Biological DVH (bDVH) generation
- Enhanced QA modules with uncertainty quantification
- Improved code organization and documentation

### Citation
If you use this software, please cite it using the DOI provided by Zenodo.
"@

# Step 6: Create Release
Write-Host "`n[Step 6] Creating GitHub release..." -ForegroundColor Yellow
try {
    gh release create $version `
        --repo $repo `
        --title $title `
        --notes $releaseNotes
    
    Write-Host "✓ Release created successfully!" -ForegroundColor Green
    Write-Host "  Release URL: https://github.com/$repo/releases/tag/$version" -ForegroundColor Cyan
} catch {
    Write-Host "✗ Failed to create release: $_" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure you have write access to the repository" -ForegroundColor White
    Write-Host "  2. Try manually creating the release on GitHub website" -ForegroundColor White
    exit 1
}

# Step 7: Display Next Steps
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✓ Release Created Successfully!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "NEXT STEPS (Do these via web browser):" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Grant Zenodo Access to Private Repositories:" -ForegroundColor Yellow
Write-Host "   https://github.com/settings/applications" -ForegroundColor White
Write-Host "   → Find 'Zenodo' → Grant private repo access" -ForegroundColor White
Write-Host ""
Write-Host "2. Enable Repository in Zenodo:" -ForegroundColor Yellow
Write-Host "   https://zenodo.org/account/settings/github/" -ForegroundColor White
Write-Host "   → Click 'Sync now'" -ForegroundColor White
Write-Host "   → Find '$repo' and toggle ON" -ForegroundColor White
Write-Host ""
Write-Host "3. Wait 5-10 minutes for Zenodo to archive your release" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Get Your DOI:" -ForegroundColor Yellow
Write-Host "   https://zenodo.org/deposit" -ForegroundColor White
Write-Host "   → Find your archived release" -ForegroundColor White
Write-Host "   → Copy the DOI (format: 10.5281/zenodo.XXXXXXXX)" -ForegroundColor White
Write-Host ""
Write-Host "5. After getting DOI, run this command to update files:" -ForegroundColor Yellow
Write-Host "   # I can help you update CITATION.cff and README.md with the new DOI" -ForegroundColor White
Write-Host ""

# Open release page in browser
$releaseUrl = "https://github.com/$repo/releases/tag/$version"
$openBrowser = Read-Host "Open release page in browser? (y/n)"
if ($openBrowser -eq "y") {
    Start-Process $releaseUrl
}

Write-Host "`nDone! ✓" -ForegroundColor Green

