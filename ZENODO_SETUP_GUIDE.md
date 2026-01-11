# Creating a Zenodo DOI for py_ntcpx Repository

This guide will help you create a new Zenodo DOI for your GitHub repository: **https://github.com/kalyan2031990/py_ntcpx**

## Step 1: Connect GitHub Repository to Zenodo

1. **Go to Zenodo**: https://zenodo.org/
2. **Log in** using your GitHub account (click "Log in with GitHub")
3. **Go to GitHub Integration Settings**:
   - Click on your profile (top right)
   - Go to **Settings** → **GitHub**
   - Or directly: https://zenodo.org/account/settings/github/
4. **Enable Repository**:
   - Click "Sync now" to refresh your repository list
   - Find `kalyan2031990/py_ntcpx` in the list
   - Toggle the switch to **ON** (enable it)
   - This will allow Zenodo to automatically archive releases

## Step 2: Create a GitHub Release

1. **Go to your GitHub repository**: https://github.com/kalyan2031990/py_ntcpx
2. **Click on "Releases"** (right sidebar, or go to: https://github.com/kalyan2031990/py_ntcpx/releases)
3. **Click "Draft a new release"**
4. **Fill in the release details**:
   - **Tag version**: `v1.0.0` (or `v1.0` - must start with 'v')
   - **Release title**: `py_ntcpx v1.0.0 - Initial Release` (or similar)
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
5. **Click "Publish release"**

## Step 3: Wait for Zenodo Processing

- After publishing the release, Zenodo will automatically detect it (usually within 5-10 minutes)
- Zenodo will archive your release and assign a DOI
- You'll receive an email notification when it's ready

## Step 4: Get Your New DOI

1. **Check Zenodo Dashboard**:
   - Go to: https://zenodo.org/deposit
   - Or check your email for the notification
   - The new deposit will appear in your "Uploads" section

2. **Copy the DOI**:
   - The DOI will be in format: `10.5281/zenodo.XXXXXXXX`
   - Copy this DOI

## Step 5: Update Repository Files with New DOI

After you get the new DOI, update these files:

1. **CITATION.cff**: Update the `doi` and `url` fields
2. **README.md**: Update the DOI badge and citation
3. **.zenodo.json**: Update if needed (optional, Zenodo reads from GitHub release)

I can help you update these files once you have the new DOI!

## Important Notes

- ✅ Each GitHub release gets its own DOI
- ✅ The DOI is permanent and citable
- ✅ Make sure to publish releases (not just create tags) for Zenodo to archive them
- ✅ You can create new releases for each version (v1.1.0, v2.0.0, etc.) and each gets a new DOI

## Citation Format (After Getting DOI)

Once you have the DOI, you can cite your repository as:

```
Mondal, K. (2025). py_ntcpx: NTCP Analysis and Machine Learning Pipeline (v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXXX
```

## Troubleshooting

- **Repository not showing in Zenodo?**: Make sure it's set to "Public" (or your Zenodo account has access)
- **Release not archived?**: Check that the release was published (not just a tag)
- **Need to update metadata?**: Edit the release on GitHub and Zenodo will update

