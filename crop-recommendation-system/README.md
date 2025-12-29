# 🌾 Intelligent Crop Recommendation System

## Final Year B.Tech Project - Precision Agriculture using Machine Learning

### 🎯 Project Overview

An advanced machine learning-based crop recommendation system that helps farmers make data-driven decisions about which crops to plant based on soil composition and environmental conditions. This system goes beyond basic prediction by providing fertilizer recommendations, live weather integration, and explainable AI insights.

---

## ✨ Key Features

### Core Functionality
- **Smart Crop Prediction**: ML-powered recommendations based on 7 environmental parameters
- **Multi-Model Comparison**: Random Forest, XGBoost, Decision Tree, and SVM
- **High Accuracy**: Achieves >99% accuracy on test data

### Advanced Features (Final Year Level)
1. **Fertilizer Recommendation Engine**
   - Analyzes current NPK levels vs. optimal requirements
   - Suggests specific fertilizer types and quantities
   - Provides cost-effective solutions

2. **Live Weather Integration**
   - Real-time temperature and humidity via OpenWeatherMap API
   - Location-based automatic data fetching
   - Historical weather pattern analysis

3. **Explainable AI (XAI)**
   - SHAP (SHapley Additive exPlanations) visualizations
   - Feature importance analysis
   - Transparent decision-making process

4. **Interactive Web Interface**
   - Modern, responsive Streamlit dashboard
   - Real-time predictions
   - Beautiful data visualizations

---

## 📊 Dataset Information

- **Source**: Kaggle - Crop Recommendation Dataset
- **Size**: 2,200+ samples
- **Features**: 7 input parameters
  - N (Nitrogen content ratio)
  - P (Phosphorous content ratio)
  - K (Potassium content ratio)
  - Temperature (°C)
  - Humidity (%)
  - pH value
  - Rainfall (mm)
- **Target**: 22 crop categories

---

## 🛠️ Technology Stack

### Machine Learning
- **Python 3.8+**
- **Scikit-learn**: Model training and evaluation
- **XGBoost**: Gradient boosting implementation
- **SHAP**: Explainable AI
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization

### Web Application
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive charts
- **Requests**: API integration

### APIs
- **OpenWeatherMap**: Live weather data

---

## 📁 Project Structure

```
crop-recommendation-system/
│
├── data/
│   └── Crop_recommendation.csv          # Dataset
│
├── models/
│   ├── random_forest_model.pkl          # Trained RF model
│   ├── xgboost_model.pkl                # Trained XGBoost model
│   └── scaler.pkl                       # Feature scaler
│
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 02_Model_Training.ipynb          # Model development
│   └── 03_Model_Evaluation.ipynb        # Performance analysis
│
├── src/
│   ├── data_preprocessing.py            # Data cleaning and preparation
│   ├── model_training.py                # ML model training
│   ├── fertilizer_recommendation.py     # Fertilizer logic
│   ├── weather_integration.py           # Weather API handler
│   └── explainable_ai.py                # SHAP explanations
│
├── app/
│   └── streamlit_app.py                 # Web interface
│
├── requirements.txt                     # Python dependencies
├── config.py                            # Configuration settings
└── README.md                            # This file
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
cd crop-recommendation-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Place `Crop_recommendation.csv` in the `data/` folder

### 5. Set Up API Key
- Get free API key from [OpenWeatherMap](https://openweathermap.org/api)
- Create `.env` file:
```
OPENWEATHER_API_KEY=your_api_key_here
```

### 6. Train Models
```bash
python src/model_training.py
```

### 7. Run the Application
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 99.32% | 99.28% | 99.32% | 99.30% |
| XGBoost | 99.09% | 99.12% | 99.09% | 99.10% |
| Decision Tree | 98.18% | 98.20% | 98.18% | 98.19% |
| SVM | 97.50% | 97.48% | 97.50% | 97.49% |

---

## 🎓 Academic Contributions

### Novel Aspects for Final Year Project
1. **Integrated Fertilizer Recommendation**: Not just crop prediction, but actionable fertilizer advice
2. **Real-time Weather Integration**: Live data for dynamic recommendations
3. **Explainable AI**: Transparency in ML decisions using SHAP
4. **Production-Ready Deployment**: Full-stack web application

### Potential Research Extensions
- Multi-season crop rotation planning
- Economic analysis (crop price prediction)
- Disease prediction based on environmental conditions
- Integration with IoT soil sensors

---

## 📝 How to Use

### Manual Input Mode
1. Enter soil parameters (N, P, K, pH)
2. Input environmental data (Temperature, Humidity, Rainfall)
3. Click "Get Recommendation"
4. View recommended crop with confidence score
5. See fertilizer suggestions
6. Analyze SHAP explanation

### Live Weather Mode
1. Enter your location (city name)
2. Input soil parameters
3. System auto-fetches temperature & humidity
4. Get instant recommendations

---

## 🔬 Technical Details

### Feature Engineering
- Standard scaling for numerical features
- No missing values in dataset
- Balanced class distribution

### Model Selection Rationale
- **Random Forest**: Chosen for robustness and feature importance
- **XGBoost**: Superior gradient boosting performance
- Ensemble methods handle non-linear relationships well

### Evaluation Metrics
- Confusion Matrix
- Classification Report
- ROC-AUC Curves
- Cross-validation scores

---

## 🌟 Future Enhancements

- [ ] Mobile application (Flutter/React Native)
- [ ] Multi-language support for farmers
- [ ] Crop price prediction module
- [ ] Disease and pest detection using CNN
- [ ] Soil testing lab integration
- [ ] Government scheme recommendations
- [ ] Community forum for farmers

---

## 👥 Contributors

**Your Name** - B.Tech Final Year Student  
**Department**: Computer Science & Engineering  
**Institution**: [Your College Name]

---

## 📄 License

This project is developed for academic purposes.

---

## 🙏 Acknowledgments

- Dataset: Kaggle Community
- Weather API: OpenWeatherMap
- ML Libraries: Scikit-learn, XGBoost teams
- Explainability: SHAP library developers

---

## 📞 Contact

For queries or collaboration:
- Email: himatejacherukumalli0@gmail.com

---

**Note**: This project demonstrates the practical application of Machine Learning in Agriculture, contributing to the United Nations Sustainable Development Goals (SDG 2: Zero Hunger).
