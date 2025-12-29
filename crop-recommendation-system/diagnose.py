"""
Diagnostic Script - Check Project Status
"""

import sys
import os

print("\n" + "="*60)
print("PROJECT DIAGNOSTIC CHECK")
print("="*60)

# 1. Check Python version
print("\n1. Python Version:")
print(f"   {sys.version}")

# 2. Check current directory
print("\n2. Current Directory:")
print(f"   {os.getcwd()}")

# 3. Check if .env exists
print("\n3. Environment File (.env):")
if os.path.exists('.env'):
    print("   ✓ .env file exists")
    with open('.env', 'r') as f:
        content = f.read()
        if 'OPENWEATHER_API_KEY' in content:
            print("   ✓ API key configured")
        else:
            print("   ✗ API key not found in .env")
else:
    print("   ✗ .env file not found")

# 4. Check project structure
print("\n4. Project Structure:")
required_dirs = ['src', 'app', 'data', 'models']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing")
        os.makedirs(dir_name, exist_ok=True)
        print(f"      → Created {dir_name}/")

# 5. Check key files
print("\n5. Key Files:")
key_files = [
    'config.py',
    'requirements.txt',
    'src/data_preprocessing.py',
    'src/model_training.py',
    'app/streamlit_app.py'
]
for file_path in key_files:
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} missing")

# 6. Check installed packages
print("\n6. Checking Required Packages:")
required_packages = [
    'requests',
    'pandas',
    'numpy',
    'sklearn',
    'streamlit',
    'plotly'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package} installed")
    except ImportError:
        print(f"   ✗ {package} NOT installed")

# 7. Test API if requests is available
print("\n7. Testing Weather API:")
try:
    import requests
    
    # Read API key
    api_key = None
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('OPENWEATHER_API_KEY'):
                    api_key = line.split('=')[1].strip()
                    break
    
    if api_key:
        print(f"   API Key: {api_key[:10]}...")
        
        # Quick test
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {'q': 'Mumbai,IN', 'appid': api_key, 'units': 'metric'}
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ API Working! Temperature in Mumbai: {data['main']['temp']}°C")
        else:
            print(f"   ✗ API Error: Status {response.status_code}")
    else:
        print("   ✗ API key not found")
        
except ImportError:
    print("   ⚠ requests package not installed - cannot test API")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")

# Summary
print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

print("\n📋 RECOMMENDED ACTIONS:")
print("\n1. Install missing packages:")
print("   py -m pip install pandas numpy scikit-learn streamlit plotly")

print("\n2. Download dataset:")
print("   py download_dataset.py")

print("\n3. Train model:")
print("   py quick_train.py")

print("\n4. Run application:")
print("   streamlit run app/streamlit_app.py")

print("\n" + "="*60)
