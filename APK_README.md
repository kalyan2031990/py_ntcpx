# APK Creation Guide for py_ntcpx_v1.1.0

## Quick Start

### Option 1: Using PowerShell Script (Easiest)
```powershell
.\create_apk.ps1
```

### Option 2: Manual Build with Buildozer
```bash
pip install buildozer
buildozer init
buildozer android debug
```

## APK Location

After successful build, the APK will be located at:
```
bin/ntcpanalysis-debug.apk
```

## Installation on Android

```bash
adb install bin/ntcpanalysis-debug.apk
```

## Requirements

- Python 3.9+
- Buildozer
- Android SDK (for signing release APK)
- Java JDK (for Android build tools)

## Troubleshooting

See `APK_PACKAGING_GUIDE.md` in the root directory for detailed troubleshooting and alternative methods.

## Alternative: Web-Based Deployment

For easier deployment and updates, consider deploying as a web service:
- Flask/FastAPI backend
- Simple Android app that calls web API
- Benefits: Smaller APK, centralized updates

See `APK_PACKAGING_GUIDE.md` for web deployment options.
