# PowerShell Commands for Zenodo DOI Setup (Private Repository)

This guide provides PowerShell commands to automate parts of the Zenodo DOI setup process for your **private** GitHub repository.

## ⚠️ Important Notes for Private Repositories

- Zenodo **CAN** archive private repositories, but:
  1. You must grant Zenodo access to your private repositories in GitHub
  2. The Zenodo archive itself will be public (the archived code), but your GitHub repo remains private
  3. You need to authorize Zenodo via the web interface first

## Step 1: Install GitHub CLI (if not installed)

Check if GitHub CLI is installed:

```powershell
# Check if gh CLI is installed
gh --version

# If not installed, install via winget (Windows 10/11)
winget install --id GitHub.cli

# Or download from: https://cli.github.com/
```

## Step 2: Authenticate with GitHub CLI

```powershell
# Login to GitHub (opens browser for authentication)
gh auth login

# Follow the prompts:
# - Choose GitHub.com
# - Choose HTTPS
# - Authenticate via web browser
# - Choose your preferred authentication method
```

## Step 3: Verify Repository Access

```powershell
# Check if you can access the repository
gh repo view kalyan2031990/py_ntcpx

# List current releases
gh release list --repo kalyan2031990/py_ntcpx
```

## Step 4: Create GitHub Release (PowerShell Commands)

### Option A: Using GitHub CLI (Recommended)

```powershell
# Create a release with tag v1.0.0
gh release create v1.0.0 `
  --repo kalyan2031990/py_ntcpx `
  --title "py_ntcpx v1.0.0 - Initial Release" `
  --notes @release_notes.md

# Or with inline notes:
gh release create v1.0.0 `
  --repo kalyan2031990/py_ntcpx `
  --title "py_ntcpx v1.0.0 - Initial Release" `
  --notes "## Initial Release - py_ntcpx v1.0.0

First release of the redesigned NTCP Analysis and Machine Learning Pipeline for Head & Neck Cancer.

### Features
- Complete pipeline redesign with enhanced features
- Biological DVH (bDVH) generation
- Enhanced QA modules with uncertainty quantification
- Improved code organization and documentation

### Citation
If you use this software, please cite it using the DOI provided by Zenodo."

# Verify release was created
gh release view v1.0.0 --repo kalyan2031990/py_ntcpx
```

### Option B: Using Git Commands (Alternative)

```powershell
# Ensure you're on the main branch
git checkout main

# Create an annotated tag
git tag -a v1.0.0 -m "py_ntcpx v1.0.0 - Initial Release"

# Push the tag to GitHub
git push origin v1.0.0

# Note: After pushing tag, you'll need to create the release manually on GitHub website
# Or use GitHub CLI to convert tag to release:
gh release create v1.0.0 --repo kalyan2031990/py_ntcpx --title "py_ntcpx v1.0.0" --notes "Initial release"
```

## Step 5: Check Release Status

```powershell
# View all releases
gh release list --repo kalyan2031990/py_ntcpx

# View specific release details
gh release view v1.0.0 --repo kalyan2031990/py_ntcpx

# View release in browser
gh release view v1.0.0 --repo kalyan2031990/py_ntcpx --web
```

## Step 6: Complete Setup Script

Here's a complete PowerShell script you can run:

```powershell
# Complete Zenodo Setup Script for Private Repository
# Run this script to create a GitHub release

$repo = "kalyan2031990/py_ntcpx"
$version = "v1.0.0"
$title = "py_ntcpx $version - Initial Release"

# Check if gh CLI is installed
try {
    $ghVersion = gh --version 2>&1
    Write-Host "✓ GitHub CLI is installed" -ForegroundColor Green
} catch {
    Write-Host "✗ GitHub CLI is not installed" -ForegroundColor Red
    Write-Host "Install with: winget install --id GitHub.cli" -ForegroundColor Yellow
    exit 1
}

# Check authentication
try {
    gh auth status 2>&1 | Out-Null
    Write-Host "✓ GitHub CLI is authenticated" -ForegroundColor Green
} catch {
    Write-Host "✗ Not authenticated. Run: gh auth login" -ForegroundColor Red
    exit 1
}

# Verify repository access
Write-Host "`nVerifying repository access..." -ForegroundColor Cyan
try {
    gh repo view $repo 2>&1 | Out-Null
    Write-Host "✓ Repository access confirmed" -ForegroundColor Green
} catch {
    Write-Host "✗ Cannot access repository: $repo" -ForegroundColor Red
    Write-Host "Make sure the repository exists and you have access" -ForegroundColor Yellow
    exit 1
}

# Check if release already exists
Write-Host "`nChecking for existing releases..." -ForegroundColor Cyan
$existingRelease = gh release list --repo $repo --limit 1 2>&1
if ($existingRelease -match $version) {
    Write-Host "⚠ Release $version already exists" -ForegroundColor Yellow
    $overwrite = Read-Host "Overwrite? (y/n)"
    if ($overwrite -ne "y") {
        Write-Host "Aborted." -ForegroundColor Yellow
        exit 0
    }
}

# Create release
Write-Host "`nCreating release $version..." -ForegroundColor Cyan
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

try {
    gh release create $version `
        --repo $repo `
        --title $title `
        --notes $releaseNotes
    
    Write-Host "✓ Release created successfully!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://zenodo.org/account/settings/github/" -ForegroundColor White
    Write-Host "2. Make sure '$repo' is enabled (toggle ON)" -ForegroundColor White
    Write-Host "3. Wait 5-10 minutes for Zenodo to archive your release" -ForegroundColor White
    Write-Host "4. Check your Zenodo dashboard: https://zenodo.org/deposit" -ForegroundColor White
    
    # Open release page
    Start-Process "https://github.com/$repo/releases/tag/$version"
    
} catch {
    Write-Host "✗ Failed to create release: $_" -ForegroundColor Red
    exit 1
}
```

## Step 7: Web Interface Steps (Required for Private Repos)

**You MUST do these steps via web browser** (cannot be automated):

1. **Authorize Zenodo to access private repositories:**
   - Go to: https://github.com/settings/applications
   - Find "Zenodo" in authorized OAuth apps
   - Click "Grant" for private repository access (if not already granted)

2. **Enable repository in Zenodo:**
   - Go to: https://zenodo.org/account/settings/github/
   - Click "Sync now"
   - Find `kalyan2031990/py_ntcpx`
   - Toggle it **ON**

3. **Check Zenodo after release (5-10 minutes):**
   - Go to: https://zenodo.org/deposit
   - Your archived release should appear
   - Copy the DOI

## Step 8: Verify Zenodo Archival (Check Script)

```powershell
# After creating release, wait 5-10 minutes, then check:
Write-Host "Checking Zenodo archive status..." -ForegroundColor Cyan
Write-Host "Go to: https://zenodo.org/deposit" -ForegroundColor Yellow
Write-Host "Or check your email for Zenodo notification" -ForegroundColor Yellow

# Get release URL
$releaseUrl = "https://github.com/kalyan2031990/py_ntcpx/releases/tag/v1.0.0"
Write-Host "`nRelease URL: $releaseUrl" -ForegroundColor Cyan
Start-Process $releaseUrl
```

## Troubleshooting Commands

```powershell
# Check GitHub CLI authentication
gh auth status

# Re-authenticate if needed
gh auth login

# Check repository permissions
gh api repos/kalyan2031990/py_ntcpx --jq '.permissions'

# List all tags
git tag -l

# Delete a tag (if needed to recreate)
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0

# Check release details via API
gh api repos/kalyan2031990/py_ntcpx/releases/tags/v1.0.0
```

## Quick One-Liner (After Setup)

Once GitHub CLI is configured, you can create a release with:

```powershell
gh release create v1.0.0 --repo kalyan2031990/py_ntcpx --title "py_ntcpx v1.0.0" --notes "Initial release"
```

