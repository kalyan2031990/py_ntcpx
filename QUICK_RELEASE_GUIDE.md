# Quick Guide: Create GitHub Release (No GitHub CLI Required)

Since GitHub CLI is not installed, here's the **easiest way** to create a release:

## Option 1: GitHub Website (RECOMMENDED - Easiest)

1. **Go to your repository releases page:**
   - https://github.com/kalyan2031990/py_ntcpx/releases

2. **Click "Draft a new release"**

3. **Fill in the details:**
   - **Choose a tag**: Type `v1.0.0` (will create new tag)
   - **Release title**: `py_ntcpx v1.0.0 - Initial Release`
   - **Description**:
     ```
     ## Initial Release - py_ntcpx v1.0.0
     
     First release of the redesigned NTCP Analysis and Machine Learning Pipeline for Head & Neck Cancer.
     
     ### Features
     - Complete pipeline redesign with enhanced features
     - Biological DVH (bDVH) generation
     - Enhanced QA modules with uncertainty quantification
     - Improved code organization and documentation
     
     ### Citation
     If you use this software, please cite it using the DOI provided by Zenodo.
     ```

4. **Click "Publish release"**

✅ **Done!** This will create the release and trigger Zenodo to archive it.

---

## Option 2: Using Git Commands + GitHub Website

If you prefer to create the tag via command line:

```powershell
# Create an annotated tag
git tag -a v1.0.0 -m "py_ntcpx v1.0.0 - Initial Release"

# Push the tag to GitHub
git push origin v1.0.0

# Then go to GitHub website:
# https://github.com/kalyan2031990/py_ntcpx/releases
# Click "Draft a new release"
# Select tag: v1.0.0
# Add description and publish
```

---

## Option 3: Install GitHub CLI (For Future Use)

If you want to use the PowerShell script in the future:

```powershell
# Install GitHub CLI
winget install --id GitHub.cli

# Or download from: https://cli.github.com/

# Authenticate
gh auth login

# Then you can use the script
.\create_zenodo_release.ps1
```

---

## After Creating the Release

1. **Enable Zenodo** (if not already done):
   - Go to: https://zenodo.org/account/settings/github/
   - Click "Sync now"
   - Find `kalyan2031990/py_ntcpx` and toggle **ON**

2. **Wait 5-10 minutes** for Zenodo to archive your release

3. **Get your NEW DOI:**
   - Go to: https://zenodo.org/deposit
   - Find your archived release
   - Copy the DOI (format: `10.5281/zenodo.XXXXXXXX`)

4. **Update CITATION.cff and README.md** with the new DOI (I can help with this!)

---

**Recommendation:** Use **Option 1** (GitHub website) - it's the simplest and fastest! 🚀

