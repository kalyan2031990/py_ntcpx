# DOI Status for py_ntcpx Repository

## Current Status

✅ **GitHub is correctly displaying your CITATION.cff file** - This is automatic and working as expected!

⚠️ **However, the DOI in CITATION.cff is still from the OLD repository:**
- Current DOI: `10.5281/zenodo.16786956` (from NTCP_Analysis_Pipeline)
- This needs to be updated with a NEW DOI for py_ntcpx

## What You Need to Do

### Step 1: Create GitHub Release
Create a release to trigger Zenodo to generate a NEW DOI for py_ntcpx:

**Option A: Use the PowerShell script (easiest)**
```powershell
.\create_zenodo_release.ps1
```

**Option B: Use GitHub CLI manually**
```powershell
gh release create v1.0.0 `
  --repo kalyan2031990/py_ntcpx `
  --title "py_ntcpx v1.0.0 - Initial Release" `
  --notes "Initial release of py_ntcpx - NTCP Analysis Pipeline"
```

**Option C: Use GitHub website**
1. Go to: https://github.com/kalyan2031990/py_ntcpx/releases
2. Click "Draft a new release"
3. Tag: `v1.0.0`
4. Title: `py_ntcpx v1.0.0 - Initial Release`
5. Publish release

### Step 2: Enable Zenodo (Web Interface - Required)

1. **Grant Zenodo access to private repos:**
   - https://github.com/settings/applications
   - Find "Zenodo" → Grant private repo access

2. **Enable repository in Zenodo:**
   - https://zenodo.org/account/settings/github/
   - Click "Sync now"
   - Find `kalyan2031990/py_ntcpx` and toggle ON

### Step 3: Wait for Zenodo Processing
- Wait 5-10 minutes after publishing release
- Check: https://zenodo.org/deposit
- You'll see a NEW DOI (format: `10.5281/zenodo.XXXXXXXX`)

### Step 4: Update Files with New DOI

Once you have the NEW DOI, I can help you update:
- `CITATION.cff` - Update `doi` and `url` fields
- `README.md` - Update DOI badge and citation

**Just share the new DOI with me and I'll update both files!**

## Files That Need Updating

These files currently have the OLD DOI and need to be updated:

1. **CITATION.cff** (Line 11-12):
   ```
   doi: "10.5281/zenodo.16786956"  ← OLD
   url: "https://doi.org/10.5281/zenodo.16786956"  ← OLD
   ```

2. **README.md** (Line 347, 354):
   - Citation example has placeholder
   - DOI badge has OLD DOI

## Summary

- ✅ CITATION.cff is displayed correctly on GitHub
- ✅ Repository URL is correct (py_ntcpx)
- ⚠️ DOI needs to be updated (currently OLD DOI)
- 📋 Next: Create release → Get new DOI → Update files

