# 📱 Quick Installation Instructions for iPhone

## Step-by-Step Guide

### 1. Prepare the App

Before installing, you need to:
- Add icon files to the `icons/` folder (see README.md for sizes)
- Host the app over HTTPS or use localhost

### 2. Open on iPhone

**Option A: Local File**
1. Transfer the `iphone app` folder to your iPhone
2. Open Files app
3. Navigate to the folder
4. Tap `index.html`
5. Tap "Share" → "Open in Safari"

**Option B: Web Server**
1. Host the app on a web server (see README.md)
2. Open Safari on iPhone
3. Visit the URL

**Option C: Local Network**
1. On your computer, run: `python -m http.server 8000` (in the iphone app folder)
2. Find your computer's IP address
3. On iPhone Safari, visit: `http://[your-ip]:8000/index.html`

### 3. Install to Home Screen

1. In Safari, tap the **Share button** (square with up arrow)
2. Scroll down and tap **"Add to Home Screen"**
3. Optionally change the name
4. Tap **"Add"**

### 4. Use the App

1. Tap the app icon on your home screen
2. Allow microphone access when prompted
3. Follow the on-screen instructions

## ⚠️ Important Notes

- **Must use Safari** (not Chrome or Firefox on iOS)
- **HTTPS required** for production (localhost OK for testing)
- **Microphone permission** must be granted
- **Icons** should be added for best experience

## 🎨 Creating Icons

Create a 512x512px icon with your design, then:

1. Use https://realfavicongenerator.net/
2. Upload your 512x512 icon
3. Download the generated icons
4. Place them in the `icons/` folder
5. Make sure filenames match manifest.json

Or manually create these sizes and save to `icons/`:
- icon-72x72.png
- icon-96x96.png
- icon-128x128.png
- icon-144x144.png
- icon-152x152.png
- icon-192x192.png
- icon-384x384.png
- icon-512x512.png
- apple-touch-icon.png (180x180)

## ✅ Verification

After installation:
- ✅ App icon appears on home screen
- ✅ App opens in standalone mode (no Safari UI)
- ✅ Full screen experience
- ✅ Microphone works
- ✅ Works offline (after first load)

Enjoy your AI Music Conductor on iPhone! 🎵

