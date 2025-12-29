# 🔧 Installation & Setup Guide

## Step-by-Step Instructions to Get Your Project Running

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Module not found" or "No module named 'X'"

**Solution:** Install the missing package
```bash
py -m pip install package_name
```

### Issue 2: "Python was not found"

**Solution:** Use `py` instead of `python`
```bash
py your_script.py
```

### Issue 3: pip installation fails

**Solution:** Upgrade pip first
```bash
py -m pip install --upgrade pip
```

---

## 📦 Complete Installation Process

### Step 1: Install Core Dependencies (One by One)

Instead of installing all at once, let's install packages individually:

```bash
# Essential packages
py -m pip install requests
py -m pip install python-dotenv
py -m pip install pandas
py -m pip install numpy
```

### Step 2: Install ML Packages

```bash
py -m pip install scikit-learn
py -m pip install joblib
```

### Step 3: Install Web Framework

```bash
py -m pip install streamlit
py -m pip install plotly
```

### Step 4: Install Optional Packages

```bash
py -m pip install matplotlib
py -m pip install seaborn
```

---

## 🚀 Quick Start (Minimal Setup)

If you want to test quickly without all packages:

### Option A: Test API Only

```bash
# Already done - your API is working!
py test_api_simple.py
```

### Option B: Create Sample Dataset

```bash
# Install pandas first
py -m pip install pandas

# Create sample data
py download_dataset.py
```

### Option C: Train Basic Model

```bash
# Install ML packages
py -m pip install pandas numpy scikit-learn joblib

# Train model
py quick_train.py
```

### Option D: Run Full Application

```bash
# Install all packages
py -m pip install streamlit plotly pandas numpy scikit-learn joblib requests python-dotenv

# Run app
streamlit run app/streamlit_app.py
```

---

## 🐛 Debugging Steps

### 1. Check Python Installation

```bash
py --version
```

Should show Python 3.8 or higher.

### 2. Check pip

```bash
py -m pip --version
```

### 3. List Installed Packages

```bash
py -m pip list
```

### 4. Run Diagnostic

```bash
py diagnose.py
```

---

## 📝 What Error Are You Getting?

Please share the error message and I'll provide specific solution:

### Common Error Patterns:

**Error 1:** `ModuleNotFoundError: No module named 'X'`
- **Fix:** `py -m pip install X`

**Error 2:** `ImportError: cannot import name 'X'`
- **Fix:** Package version mismatch, reinstall: `py -m pip install --upgrade X`

**Error 3:** `SyntaxError` or `IndentationError`
- **Fix:** Check Python version (need 3.8+)

**Error 4:** `FileNotFoundError`
- **Fix:** Check you're in correct directory

**Error 5:** API errors
- **Fix:** Already solved - your API key is working!

---

## 🎯 Minimal Working Version

If you want to get something running ASAP:

### Create a Simple Test Script

```python
# test_basic.py
import requests

# Test API
api_key = "5db2dc4ca89771c98c26b82e3503f2e2"
url = "http://api.openweathermap.org/data/2.5/weather"
params = {'q': 'Mumbai,IN', 'appid': api_key, 'units': 'metric'}

response = requests.get(url, params=params)
data = response.json()

print(f"Temperature in Mumbai: {data['main']['temp']}°C")
print(f"Humidity: {data['main']['humidity']}%")
print("\n✅ API is working!")
```

Run it:
```bash
py test_basic.py
```

---

## 💡 Next Steps Based on Your Error

**Please tell me:**
1. What command did you run?
2. What error message did you see?
3. What were you trying to do?

Then I can give you the exact fix!

---

## 🆘 Emergency Fallback

If nothing works, we can:
1. Create a simpler version without some dependencies
2. Use Google Colab (online, no installation needed)
3. Create a Docker container
4. Use a different approach

**Just let me know what error you're seeing!** 🔍
