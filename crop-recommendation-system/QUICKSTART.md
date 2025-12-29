# 🚀 Quick Start Guide

## Crop Recommendation System - Setup & Usage

This guide will help you set up and run the Crop Recommendation System in minutes.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for downloading dependencies and weather data)

---

## ⚡ Quick Setup (5 Minutes)

### Step 1: Navigate to Project Directory

```bash
cd crop-recommendation-system
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- scikit-learn (ML models)
- xgboost (Gradient boosting)
- streamlit (Web interface)
- pandas, numpy (Data processing)
- plotly (Visualizations)
- shap (Explainable AI)
- requests (Weather API)

### Step 4: Set Up Dataset

**Option A: Create Sample Dataset (Quick Testing)**
```bash
python download_dataset.py
```
Then press 'y' to create a sample dataset.

**Option B: Download Full Dataset (Recommended for Production)**
1. Visit: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Download the dataset
3. Extract and place `Crop_recommendation.csv` in the `data/` folder

### Step 5: Train the Model

```bash
python src/model_training.py
```

This will:
- Load and preprocess the data
- Train multiple ML models (Random Forest, XGBoost, etc.)
- Evaluate and compare models
- Save the best model to `models/` folder

**Expected output:**
```
✓ Data loaded successfully
✓ Training Random Forest...
✓ Training XGBoost...
✓ Best Model: Random Forest (99.32% accuracy)
✓ Models saved successfully
```

### Step 6: Run the Web Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 🌐 Optional: Set Up Weather Integration

### Get Free API Key

1. Visit: https://openweathermap.org/api
2. Sign up for a free account
3. Generate an API key (free tier: 1000 calls/day)

### Configure API Key

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` file and add your API key:
   ```
   OPENWEATHER_API_KEY=your_actual_api_key_here
   ```

3. Restart the Streamlit app to use live weather data

---

## 📱 Using the Application

### Manual Input Mode

1. **Adjust Soil Parameters:**
   - Nitrogen (N): 0-140
   - Phosphorous (P): 5-145
   - Potassium (K): 5-205
   - pH: 3.5-9.5

2. **Set Environmental Conditions:**
   - Temperature: 8-44°C
   - Humidity: 14-100%
   - Rainfall: 20-300mm

3. **Click "Get Crop Recommendation"**

4. **View Results:**
   - Recommended crop
   - Confidence score
   - Crop information
   - Fertilizer recommendations
   - Visual analytics

### Live Weather Mode

1. Select "Live Weather Integration" in sidebar
2. Enter your city name (e.g., "Mumbai")
3. Enter country code (e.g., "IN" for India)
4. Click "Fetch Weather Data"
5. Temperature and humidity will be auto-filled
6. Enter remaining parameters
7. Click "Get Crop Recommendation"

---

## 🎯 Example Use Cases

### Example 1: Rice Cultivation Check

**Input:**
- N: 90, P: 40, K: 40
- Temperature: 25°C
- Humidity: 80%
- pH: 6.5
- Rainfall: 200mm

**Expected Output:** Rice (with high confidence)

### Example 2: Wheat Cultivation Check

**Input:**
- N: 80, P: 40, K: 20
- Temperature: 20°C
- Humidity: 65%
- pH: 6.0
- Rainfall: 100mm

**Expected Output:** Wheat or Maize

### Example 3: Coffee Plantation Check

**Input:**
- N: 100, P: 50, K: 100
- Temperature: 22°C
- Humidity: 75%
- pH: 6.5
- Rainfall: 180mm

**Expected Output:** Coffee

---

## 🔧 Troubleshooting

### Issue: "Model not found"

**Solution:**
```bash
python src/model_training.py
```
Make sure model training completes successfully.

### Issue: "Dataset not found"

**Solution:**
```bash
python download_dataset.py
```
Or manually download from Kaggle and place in `data/` folder.

### Issue: "Weather API not working"

**Possible causes:**
1. API key not configured → Set up `.env` file
2. Invalid API key → Check your OpenWeatherMap account
3. No internet connection → Check connectivity
4. City name incorrect → Try different spelling or add country code

**Workaround:** Use "Manual Input" mode instead.

### Issue: "Module not found"

**Solution:**
```bash
pip install -r requirements.txt
```
Make sure all dependencies are installed.

### Issue: Streamlit app not opening

**Solution:**
1. Check if port 8501 is available
2. Try: `streamlit run app/streamlit_app.py --server.port 8502`
3. Manually open: `http://localhost:8501` in browser

---

## 📊 Project Structure Overview

```
crop-recommendation-system/
│
├── app/
│   └── streamlit_app.py          # Main web application
│
├── src/
│   ├── data_preprocessing.py     # Data handling
│   ├── model_training.py         # ML model training
│   ├── fertilizer_recommendation.py  # Fertilizer logic
│   ├── weather_integration.py    # Weather API
│   └── explainable_ai.py         # SHAP explanations
│
├── data/
│   └── Crop_recommendation.csv   # Dataset (download required)
│
├── models/
│   ├── random_forest_model.pkl   # Trained model (generated)
│   └── scaler.pkl                # Feature scaler (generated)
│
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🎓 For Academic Presentation

### Demo Flow

1. **Introduction (2 min)**
   - Explain precision agriculture
   - Show problem statement
   - Highlight features

2. **Live Demo (5 min)**
   - Show manual input prediction
   - Demonstrate weather integration
   - Display fertilizer recommendations
   - Show visualizations

3. **Technical Details (3 min)**
   - Explain ML models used
   - Show accuracy metrics
   - Demonstrate explainable AI (SHAP)

4. **Q&A (5 min)**
   - Be ready to explain:
     - Why Random Forest?
     - How SHAP works?
     - Real-world applications
     - Future enhancements

### Key Points to Mention

✅ **99%+ accuracy** with Random Forest
✅ **7 input features** for prediction
✅ **22 crop categories** supported
✅ **Live weather integration** via API
✅ **Intelligent fertilizer recommendations**
✅ **Explainable AI** using SHAP
✅ **Production-ready web interface**

---

## 🚀 Next Steps

### Immediate
- [ ] Download full dataset from Kaggle
- [ ] Train models on full dataset
- [ ] Set up weather API key
- [ ] Test all features

### For Enhancement
- [ ] Add more crops to database
- [ ] Implement crop rotation suggestions
- [ ] Add disease prediction module
- [ ] Create mobile app version
- [ ] Add multi-language support
- [ ] Integrate with IoT sensors

---

## 📞 Support

If you encounter any issues:

1. Check this guide first
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify dataset is in correct location
5. Check Python version (3.8+)

---

## 🎉 You're Ready!

Your Crop Recommendation System is now set up and ready to use.

**Start the app:**
```bash
streamlit run app/streamlit_app.py
```

**Happy Farming! 🌾**
