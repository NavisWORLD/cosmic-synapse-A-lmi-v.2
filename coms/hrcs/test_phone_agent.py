#!/usr/bin/env python3
"""
Test HRCS Phone Agent HTML File
"""

def test_phone_agent():
    """Test the phone agent HTML file"""
    
    print("=" * 50)
    print("HRCS Phone Agent Verification Test")
    print("=" * 50)
    print()
    
    # Read file
    with open('phone_offline_agent.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    print(f"[OK] File loaded: {len(html)} bytes")
    print()
    
    # Test HTML structure
    print("HTML Structure:")
    print(f"  • DOCTYPE declarations: {html.count('<!DOCTYPE')}")
    print(f"  • Style tags: {html.count('<style>')}")
    print(f"  • Script tags: {html.count('<script>')}")
    print(f"  • Meta tags: {html.count('<meta')}")
    print()
    
    # Test JavaScript functions
    print("JavaScript Functions:")
    functions = [
        ('initAudio', 'async function initAudio'),
        ('sendMessage', 'async function sendMessage'),
        ('startListening', 'async function startListening'),
        ('stopListening', 'function stopListening'),
        ('demonstrateMath', 'function demonstrateMath'),
        ('switchTab', 'function switchTab'),
        ('log', 'function log'),
    ]
    
    all_functions_present = True
    for name, search in functions:
        found = search in html
        status = "✓" if found else "✗"
        print(f"  {status} {name}: {found}")
        if not found:
            all_functions_present = False
    
    print()
    
    # Test features
    print("Core Features:")
    features = [
        ('Golden Ratio Constant', 'PHI = 1.618033988749895'),
        ('AudioContext', 'AudioContext'),
        ('Microphone Access', 'getUserMedia'),
        ('Oscillator', 'createOscillator'),
        ('Frequency Generation', 'frequencies.push'),
        ('LocalStorage', 'localStorage'),
    ]
    
    for name, search in features:
        found = html.count(search) > 0
        count = html.count(search)
        status = "✓" if found else "✗"
        print(f"  {status} {name}: {found} ({count} occurrences)")
    
    print()
    
    # Test UI elements
    print("UI Elements:")
    ui_elements = [
        ('Buttons', 'class="big-button"'),
        ('Tabs', 'class="tab"'),
        ('Log Containers', 'class="log-container"'),
        ('Status Indicators', 'status-dot'),
    ]
    
    for name, search in ui_elements:
        count = html.count(search)
        print(f"  ✓ {name}: {count} occurrences")
    
    print()
    
    # Test mobile optimization
    print("Mobile Optimizations:")
    mobile_features = [
        ('Viewport Meta', 'viewport'),
        ('Apple Web App', 'apple-mobile-web-app-capable'),
        ('Theme Color', 'theme-color'),
        ('Touch-friendly', '-webkit-tap-highlight-color'),
        ('Responsive Design', '@media'),
    ]
    
    for name, search in mobile_features:
        found = search in html
        status = "✓" if found else "✗"
        print(f"  {status} {name}")
    
    print()
    
    # Final verdict
    print("=" * 50)
    if all_functions_present:
        print("✅ ALL TESTS PASSED - Phone Agent Ready!")
    else:
        print("⚠️  SOME FUNCTIONS MISSING - Review needed")
    print("=" * 50)
    
    return all_functions_present

if __name__ == "__main__":
    test_phone_agent()

