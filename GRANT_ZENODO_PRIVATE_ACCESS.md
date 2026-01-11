# Grant Zenodo Access to Private Repositories

Your Zenodo has these permissions:
- ✅ Full control of repository hooks
- ✅ Read org and team membership
- ✅ Read all user profile data
- ✅ Access user email addresses

**But it's MISSING:**
- ❌ **Access to private repositories** (needed for your private repo!)

## How to Grant Private Repository Access

### Option 1: Through GitHub OAuth Apps (Recommended)

1. **Go to GitHub OAuth Apps:**
   - https://github.com/settings/applications
   
2. **Find "Zenodo"** in "Authorized OAuth Apps"
   
3. **Click on "Zenodo"**
   
4. **Look for "Repository access" section**
   
5. **Click "Grant" or "Configure"** next to:
   - "Access private repositories"
   - Or "Repository access" 
   - Or "Organization access" (if applicable)
   
6. **Authorize the additional permission**
   
7. **Go back to Zenodo:**
   - https://zenodo.org/account/settings/github/
   - Click "Sync now"
   - Your repositories should now appear!

### Option 2: Re-authorize Zenodo (If Option 1 Doesn't Work)

1. **Go to Zenodo:**
   - https://zenodo.org/account/settings/github/
   
2. **Look for "Disconnect" or "Revoke access"** button
   
3. **Click it to disconnect**
   
4. **Click "Connect GitHub account" or "Authorize" again**
   
5. **This time, make sure to:**
   - Check the box for "Private repository access" (if shown)
   - Authorize all requested permissions
   
6. **Go back to Zenodo settings:**
   - Click "Sync now"
   - Your repositories should appear!

### Option 3: Through GitHub Organization Settings (If Using Org)

If your repository belongs to an organization:

1. **Go to your organization settings:**
   - https://github.com/organizations/YOUR_ORG/settings/applications
   
2. **Find "Third-party access"**
   
3. **Look for "Zenodo"**
   
4. **Grant organization access**

## After Granting Access

1. **Go to Zenodo:**
   - https://zenodo.org/account/settings/github/
   
2. **Click "Sync now"** (button should be visible)
   
3. **Wait a few seconds** for sync
   
4. **Look for `kalyan2031990/py_ntcpx`** in the list
   
5. **Toggle it ON** (enable it)

## Verify It's Working

After enabling:
- You should see `kalyan2031990/py_ntcpx` in the list
- The toggle switch should be visible
- When you turn it ON, it should show as enabled
- Your release `v1.0.0` should be archived within 5-10 minutes

