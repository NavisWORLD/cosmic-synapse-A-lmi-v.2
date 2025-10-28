#!/usr/bin/env python3
"""
Generate secure network keys for HRCS
"""

import secrets
import sys


def generate_key():
    """Generate a secure random key"""
    return secrets.token_urlsafe(32)


def main():
    """Generate and print a new key"""
    count = 1
    
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print("Usage: generate_key.py [count]")
            sys.exit(1)
    
    print("HRCS Network Keys")
    print("=" * 50)
    print()
    
    for i in range(count):
        key = generate_key()
        print(f"Key {i+1}: {key}")
    
    print()
    print("⚠️  WARNING: Keep these keys secret!")
    print("   Use the same key on all nodes in your network.")


if __name__ == "__main__":
    main()

