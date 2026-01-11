# Troubleshooting: No Repository List in Zenodo

If you don't see any repositories in Zenodo GitHub settings, follow these steps:

## Step 1: Make Sure You're Logged In

1. **Go to Zenodo homepage:** https://zenodo.org/
2. **Click "Sign in"** (top right)
3. **Choose "Sign in with GitHub"**
4. **Authorize Zenodo** to access your GitHub account

## Step 2: Grant Zenodo Access to Your GitHub Account

1. **Go to GitHub Settings:**
   - https://github.com/settings/applications
   
2. **Authorize Zenodo:**
   - Look for "Zenodo" in "Authorized OAuth Apps"
   - If you see it, click on it to check permissions
   - Make sure it has access to:
     - ✅ Public repositories
     - ✅ Private repositories (if your repo is private)
   
3. **If Zenodo is NOT in the list:**
   - Go back to Zenodo: https://zenodo.org/account/settings/github/
   - It should prompt you to authorize GitHub access
   - Click "Authorize" or "Connect GitHub account"

## Step 3: Re-authorize Zenodo (If Needed)

1. **Disconnect and reconnect:**
   - Go to: https://zenodo.org/account/settings/github/
   - Look for "Disconnect" or "Revoke access" button
   - Click it to disconnect
   - Then click "Connect GitHub account" or "Authorize"
   - Follow the prompts to authorize

## Step 4: Grant Private Repository Access (If Repository is Private)

If your repository is **private**, you need to grant special access:

1. **Go to GitHub OAuth Apps:**
   - https://github.com/settings/applications
   
2. **Find "Zenodo"** and click on it
   
3. **Grant access:**
   - Look for "Organization access" or "Repository access"
   - Click "Grant" or "Configure" next to private repository access
   - Authorize Zenodo to access your private repositories

## Step 5: Sync Repository List

1. **Go back to Zenodo:**
   - https://zenodo.org/account/settings/github/
   
2. **Click "Sync now"** button (should be visible after authorization)
   
3. **Wait a few seconds** for the list to refresh
   
4. **Look for `kalyan2031990/py_ntcpx`** in the list

## Alternative: Manual Upload to Zenodo

If the GitHub integration doesn't work, you can manually upload to Zenodo:

1. **Go to Zenodo:**
   - https://zenodo.org/deposit/new
   
2. **Click "New Upload"**
   
3. **Fill in metadata:**
   - Title: "py_ntcpx: NTCP Analysis and Machine Learning Pipeline"
   - Version: "v1.0.0"
   - Authors: Your name
   - Description: Copy from your README
   - License: MIT
   
4. **Upload files:**
   - Download your repository as ZIP from GitHub
   - Upload the ZIP file
   
5. **Publish** to get a DOI

## Still Not Working?

If repositories still don't show up after these steps:

1. **Check if you're using the correct GitHub account**
2. **Try logging out and logging back in to Zenodo**
3. **Clear browser cache and try again**
4. **Check Zenodo status:** https://status.zenodo.org/
5. **Contact Zenodo support:** https://zenodo.org/support

