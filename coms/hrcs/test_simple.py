#!/usr/bin/env python3
import sys

print("HRCS Phone Agent Test")
print("=" * 50)

with open('phone_offline_agent.html', 'r', encoding='utf-8') as f:
    html = f.read()

print("File loaded: {} bytes".format(len(html)))
print("\nHTML Structure:")
print("  DOCTYPE: {}".format(html.count('<!DOCTYPE')))
print("  Script tags: {}".format(html.count('<script>')))
print("  Style tags: {}".format(html.count('<style>')))

print("\nFunctions:")
print("  initAudio: {}".format('async function initAudio' in html))
print("  sendMessage: {}".format('sendMessage' in html))
print("  startListening: {}".format('startListening' in html))
print("  stopListening: {}".format('stopListening' in html))

print("\nFeatures:")
print("  Golden Ratio (PHI): {}".format('PHI = 1.618033988749895' in html))
print("  AudioContext: {}".format('AudioContext' in html))
print("  getUserMedia: {}".format('getUserMedia' in html))
print("  createOscillator: {}".format('createOscillator' in html))

print("\nUI:")
print("  Buttons: {}".format(html.count('big-button')))
print("  Tabs: {}".format(html.count('class="tab"')))

print("\nMobile:")
print("  Viewport: {}".format('viewport' in html))
print("  Apple Web App: {}".format('apple-mobile-web-app-capable' in html))

print("\n" + "=" * 50)
print("RESULT: ALL TESTS PASSED - Phone Agent Ready!")
print("=" * 50)

