"""
Simplest possible test - just print Python version and check files
"""

import sys
import os

print("Python version:", sys.version)
print("\nCurrent directory:", os.getcwd())
print("\nFiles in current directory:")
for item in os.listdir('.'):
    print(f"  - {item}")

print("\n✓ Python is working!")
print("\nIf you see this message, Python is installed correctly.")
print("Now tell me what error you're getting and I'll help fix it!")
