# Running God Music Locally

## Quick Start

1. **Open Terminal/Command Prompt**
2. **Navigate to the god music folder:**
   ```bash
   cd "path/to/god music"
   ```

3. **Start local server:**
   ```bash
   # Option A: Python (recommended)
   python -m http.server 8080

   # Option B: Node.js (if installed)
   npx serve . -p 8080

   # Option C: PHP (if installed)
   php -S localhost:8080
   ```

4. **Open in browser:**
   - Go to: `http://localhost:8080/index.html`
   - Or just open `index.html` directly in Chrome/Firefox

## What You'll See

- **Hero section** with app description
- **Bio-Signature card** for voice analysis
- **Band controls** (Start/Stop/Test buttons)
- **Instrument mix** (volume sliders for each instrument)
- **Visualizers** (spectrum and waveform)
- **System log** (real-time activity)

## Using the App

1. **Allow microphone access** when prompted
2. **Click "🎤 Calibrate"** and speak/hum/sing for 3 seconds
3. **Click "🎼 START BAND"** to begin
4. **The AI band will harmonize with you!**

## Troubleshooting

**"Microphone blocked"**
- Click the microphone icon in address bar
- Allow microphone access
- Refresh page

**"Page not loading"**
- Make sure you're using a modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- Try a different browser
- Check if the server is running

**"Audio not working"**
- Check system volume
- Make sure no other apps are using microphone
- Try refreshing the page

**"Instruments not playing"**
- Make sure you've calibrated first
- Check if instrument volumes are up
- Look at the log for error messages

## Browser Requirements

- **Chrome 90+** (recommended)
- **Firefox 88+**
- **Safari 14+**
- **Edge 90+**
- **Microphone access** required
- **HTTPS or localhost** for production use

## Development Mode

For development with hot reload:
```bash
npm install
npm run dev
```

This starts Vite dev server with automatic reloading.

## Local Server Commands

### Windows
```batch
cd "god music"
python -m http.server 8080
```

### macOS/Linux
```bash
cd "god music"
python3 -m http.server 8080
```

### Alternative: Node.js
```bash
npx serve . -p 8080
```

## URL to Access

Once server is running, open:
```
http://localhost:8080/index.html
```

The app should load and be fully functional!
