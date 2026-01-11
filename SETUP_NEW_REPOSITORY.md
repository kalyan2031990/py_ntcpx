# Setting Up New Repository for py_ntcpx

This document guides you through setting up a new GitHub repository for the redesigned py_ntcpx pipeline.

## Step 1: Create New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: **`py_ntcpx`** (recommended) or `NTCP_Analysis_Pipeline_v2`
3. Description: "NTCP Analysis and Machine Learning Pipeline for Head & Neck Cancer - Redesigned version (py_ntcpx)"
4. Visibility: Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Remove Current Remote and Add New One

Run these commands in your terminal:

```bash
# Remove the old remote (pointing to original repository)
git remote remove origin

# Add your new repository as remote
git remote add origin https://github.com/kalyan2031990/py_ntcpx.git
# (Replace with your actual new repository URL)

# Verify the remote
git remote -v
```

## Step 3: Stage All Files

```bash
# Add all files
git add .

# Check what will be committed
git status
```

## Step 4: Make Initial Commit

```bash
git commit -m "Initial commit: py_ntcpx v1.0 - Redesigned NTCP Analysis Pipeline

- Complete pipeline redesign with enhanced features
- Added biological DVH (bDVH) generation
- Enhanced QA modules with uncertainty quantification
- Updated architecture and code organization
- Cleaned up repository structure"
```

## Step 5: Push to New Repository

```bash
# Push to main branch (or master, depending on GitHub default)
git branch -M main
git push -u origin main
```

## Step 6: Update Repository References (After Push)

After pushing, update these files with the new repository URL:

1. **CITATION.cff**: Update `repository-code` field
2. **README.md**: Update any repository links if needed
3. **.zenodo.json**: Update if needed for Zenodo integration

## Alternative: Keep Both Repositories Linked

If you want to acknowledge the original repository, you can add a note in README.md:

```markdown
## Version History

This repository contains the redesigned version (v2.0) of the NTCP Analysis Pipeline.

- **Original version**: [NTCP_Analysis_Pipeline](https://github.com/kalyan2031990/NTCP_Analysis_Pipeline)
- **This version (py_ntcpx)**: Current repository
```

## Why New Repository Instead of Branch?

1. **Clear separation**: Original version remains stable and accessible
2. **Better branding**: New software name (py_ntcpx) deserves its own identity
3. **Citation clarity**: Easier to cite specific versions
4. **User clarity**: Users know they're using the redesigned version
5. **Maintenance**: Easier to maintain separate versions independently

