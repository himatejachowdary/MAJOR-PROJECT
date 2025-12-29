"""
Streamlit Web Application for Crop Recommendation System
A modern, interactive interface for crop prediction with advanced features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (APP_TITLE, APP_SUBTITLE, PAGE_ICON, CROP_INFO, 
                   MODEL_PATH, COLORS)
from src.fertilizer_recommendation import FertilizerRecommender
from src.weather_integration import WeatherIntegration
from src.explainable_ai import ExplainableAI

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #66BB6A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


class CropRecommendationApp:
    """
    Main application class for Crop Recommendation System
    """
    
    def __init__(self):
        """Initialize the application"""
        self.model = None
        self.scaler = None
        self.fertilizer_recommender = FertilizerRecommender()
        self.weather = WeatherIntegration()
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Initialize session state
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
    
    def load_models(self):
        """Load trained models"""
        try:
            model_path = os.path.join('..', MODEL_PATH, 'random_forest_model.pkl')
            scaler_path = os.path.join('..', MODEL_PATH, 'scaler.pkl')
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.info("Please train the model first by running: `python src/model_training.py`")
            return False
    
    def render_header(self):
        """Render application header"""
        st.markdown(f'<div class="main-header">{PAGE_ICON} {APP_TITLE}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{APP_SUBTITLE}</div>', 
                   unsafe_allow_html=True)
        
        # Info banner
        st.info("🎓 **Final Year B.Tech Project** | Precision Agriculture using Machine Learning")
    
    def render_sidebar(self):
        """Render sidebar with input options"""
        st.sidebar.header("📊 Input Parameters")
        
        # Input mode selection
        input_mode = st.sidebar.radio(
            "Select Input Mode",
            ["Manual Input", "Live Weather Integration"],
            help="Choose how to input environmental data"
        )
        
        st.sidebar.markdown("---")
        
        # Soil parameters (always manual)
        st.sidebar.subheader("🌱 Soil Parameters")
        
        n = st.sidebar.slider("Nitrogen (N)", 0, 140, 90, 
                             help="Nitrogen content ratio in soil")
        p = st.sidebar.slider("Phosphorous (P)", 5, 145, 42,
                             help="Phosphorous content ratio in soil")
        k = st.sidebar.slider("Potassium (K)", 5, 205, 43,
                             help="Potassium content ratio in soil")
        ph = st.sidebar.slider("pH Value", 3.5, 9.5, 6.5, 0.1,
                              help="pH value of the soil")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🌤️ Environmental Parameters")
        
        if input_mode == "Manual Input":
            temperature = st.sidebar.slider("Temperature (°C)", 8.0, 44.0, 20.8, 0.1)
            humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 82.0, 0.1)
            rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 202.9, 0.1)
            
            weather_data = None
            location = None
        
        else:  # Live Weather Integration
            st.sidebar.info("🌍 Enter your location to fetch live weather data")
            
            city = st.sidebar.text_input("City Name", "Mumbai")
            country = st.sidebar.text_input("Country Code (optional)", "IN",
                                           help="e.g., IN for India, US for USA")
            
            if st.sidebar.button("🔄 Fetch Weather Data"):
                with st.spinner("Fetching live weather data..."):
                    weather_data = self.weather.get_crop_relevant_data(
                        city, country if country else None
                    )
                    
                    if weather_data['status'] == 'success':
                        st.sidebar.success(f"✅ Weather data fetched for {weather_data['location']}")
                        temperature = weather_data['temperature']
                        humidity = weather_data['humidity']
                        st.session_state.weather_data = weather_data
                    else:
                        st.sidebar.error(f"❌ {weather_data['message']}")
                        temperature = 25.0
                        humidity = 70.0
                        weather_data = None
            else:
                temperature = st.sidebar.slider("Temperature (°C)", 8.0, 44.0, 25.0, 0.1)
                humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 70.0, 0.1)
                weather_data = None
            
            rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 202.9, 0.1,
                                        help="Average annual rainfall")
            location = f"{city}, {country}" if country else city
        
        # Area input for fertilizer calculation
        st.sidebar.markdown("---")
        area = st.sidebar.number_input("Farm Area (hectares)", 0.1, 100.0, 1.0, 0.1,
                                      help="Area of land for fertilizer calculation")
        
        return {
            'N': n, 'P': p, 'K': k,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall,
            'area': area,
            'weather_data': weather_data,
            'location': location
        }
    
    def predict_crop(self, input_data):
        """Make crop prediction"""
        # Prepare features
        features = np.array([[
            input_data['N'],
            input_data['P'],
            input_data['K'],
            input_data['temperature'],
            input_data['humidity'],
            input_data['ph'],
            input_data['rainfall']
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None
        
        return prediction, confidence, features
    
    def render_prediction_result(self, prediction, confidence, input_data):
        """Render prediction results"""
        st.markdown("---")
        st.header("🎯 Recommendation Result")
        
        # Main recommendation box
        st.markdown(
            f'<div class="recommendation-box">Recommended Crop: {prediction.upper()}</div>',
            unsafe_allow_html=True
        )
        
        # Confidence and details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if confidence:
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                st.metric("Confidence", "N/A")
        
        with col2:
            if prediction.lower() in CROP_INFO:
                season = CROP_INFO[prediction.lower()]['season']
                st.metric("Season", season)
            else:
                st.metric("Season", "N/A")
        
        with col3:
            if prediction.lower() in CROP_INFO:
                duration = CROP_INFO[prediction.lower()]['duration']
                st.metric("Duration", duration)
            else:
                st.metric("Duration", "N/A")
        
        # Crop information
        if prediction.lower() in CROP_INFO:
            st.markdown("---")
            st.subheader("📖 Crop Information")
            
            crop_data = CROP_INFO[prediction.lower()]
            
            st.markdown(f"""
            <div class="info-box">
                <h4>{prediction.title()}</h4>
                <p>{crop_data['description']}</p>
                <p><strong>Optimal NPK Requirements:</strong></p>
                <ul>
                    <li>Nitrogen (N): {crop_data['optimal_npk']['N']} units</li>
                    <li>Phosphorous (P): {crop_data['optimal_npk']['P']} units</li>
                    <li>Potassium (K): {crop_data['optimal_npk']['K']} units</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def render_fertilizer_recommendation(self, prediction, input_data):
        """Render fertilizer recommendations"""
        st.markdown("---")
        st.header("🌿 Fertilizer Recommendation")
        
        current_npk = {
            'N': input_data['N'],
            'P': input_data['P'],
            'K': input_data['K']
        }
        
        recommendation = self.fertilizer_recommender.recommend_fertilizer(
            current_npk, prediction, input_data['area']
        )
        
        if recommendation['status'] == 'success':
            # NPK Status
            col1, col2, col3 = st.columns(3)
            
            deficit = recommendation['deficit']
            
            with col1:
                n_status = "✅ Adequate" if abs(deficit['N']) <= 5 else ("⬆️ Deficit" if deficit['N'] > 0 else "⬇️ Surplus")
                st.metric("Nitrogen Status", n_status, f"{deficit['N']:+.1f} units")
            
            with col2:
                p_status = "✅ Adequate" if abs(deficit['P']) <= 5 else ("⬆️ Deficit" if deficit['P'] > 0 else "⬇️ Surplus")
                st.metric("Phosphorous Status", p_status, f"{deficit['P']:+.1f} units")
            
            with col3:
                k_status = "✅ Adequate" if abs(deficit['K']) <= 5 else ("⬆️ Deficit" if deficit['K'] > 0 else "⬇️ Surplus")
                st.metric("Potassium Status", k_status, f"{deficit['K']:+.1f} units")
            
            # Recommendations
            if recommendation['recommendations']:
                st.subheader("💊 Recommended Fertilizers")
                
                for i, rec in enumerate(recommendation['recommendations'], 1):
                    if rec.get('type') != 'alternative':
                        with st.expander(f"Option {i}: {rec['fertilizer']}", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Quantity:** {rec['quantity_kg']} kg")
                                st.write(f"**Cost:** ₹{rec['cost']:.2f}")
                            with col2:
                                st.write(f"**Nutrient:** {rec['nutrient_content']}")
                                st.write(f"**Purpose:** {rec['reason']}")
                
                # Total cost
                st.success(f"**Total Estimated Cost:** ₹{recommendation['total_cost']:.2f}")
            else:
                st.success("✅ Your soil nutrient levels are adequate for this crop!")
                st.info("No additional fertilizer required at this time.")
        else:
            st.error(f"Error: {recommendation['message']}")
    
    def render_input_visualization(self, input_data):
        """Visualize input parameters"""
        st.markdown("---")
        st.header("📊 Input Parameters Visualization")
        
        # Radar chart for all parameters
        categories = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        # Normalize values for radar chart (0-100 scale)
        values = [
            (input_data['N'] / 140) * 100,
            (input_data['P'] / 145) * 100,
            (input_data['K'] / 205) * 100,
            (input_data['temperature'] / 44) * 100,
            (input_data['humidity'] / 100) * 100,
            ((input_data['ph'] - 3.5) / 6) * 100,
            (input_data['rainfall'] / 300) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Input',
            line_color='#2E7D32'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Input Parameters (Normalized)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart for NPK
        col1, col2 = st.columns(2)
        
        with col1:
            npk_fig = go.Figure(data=[
                go.Bar(
                    x=['Nitrogen', 'Phosphorous', 'Potassium'],
                    y=[input_data['N'], input_data['P'], input_data['K']],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
            ])
            npk_fig.update_layout(
                title="Soil NPK Levels",
                yaxis_title="Units",
                height=400
            )
            st.plotly_chart(npk_fig, use_container_width=True)
        
        with col2:
            env_fig = go.Figure(data=[
                go.Bar(
                    x=['Temperature (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)'],
                    y=[input_data['temperature'], input_data['humidity'], 
                       input_data['ph'], input_data['rainfall'] / 10],  # Scale rainfall for visibility
                    marker_color=['#FFA07A', '#98D8C8', '#F7DC6F', '#85C1E2']
                )
            ])
            env_fig.update_layout(
                title="Environmental Parameters",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(env_fig, use_container_width=True)
    
    def render_weather_info(self, weather_data):
        """Display weather information"""
        if weather_data and weather_data['status'] == 'success':
            st.markdown("---")
            st.header("🌤️ Live Weather Data")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Location", weather_data['location'])
            with col2:
                st.metric("Temperature", f"{weather_data['temperature']}°C")
            with col3:
                st.metric("Humidity", f"{weather_data['humidity']}%")
            with col4:
                st.metric("Conditions", weather_data['description'])
            
            st.caption(f"Last updated: {weather_data['timestamp']}")
    
    def run(self):
        """Main application loop"""
        self.render_header()
        
        # Load models
        if not self.load_models():
            st.stop()
        
        # Sidebar inputs
        input_data = self.render_sidebar()
        
        # Predict button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🌾 Get Crop Recommendation", use_container_width=True)
        
        if predict_button:
            with st.spinner("Analyzing soil and environmental conditions..."):
                prediction, confidence, features = self.predict_crop(input_data)
                
                st.session_state.prediction_made = True
                st.session_state.last_prediction = {
                    'crop': prediction,
                    'confidence': confidence,
                    'input_data': input_data,
                    'features': features
                }
        
        # Display results if prediction was made
        if st.session_state.prediction_made and st.session_state.last_prediction:
            pred_data = st.session_state.last_prediction
            
            # Weather info
            if input_data['weather_data']:
                self.render_weather_info(input_data['weather_data'])
            
            # Prediction result
            self.render_prediction_result(
                pred_data['crop'],
                pred_data['confidence'],
                pred_data['input_data']
            )
            
            # Fertilizer recommendation
            self.render_fertilizer_recommendation(
                pred_data['crop'],
                pred_data['input_data']
            )
            
            # Input visualization
            self.render_input_visualization(pred_data['input_data'])
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🎓 <strong>Final Year B.Tech Project</strong></p>
            <p>Developed with ❤️ using Machine Learning & Streamlit</p>
            <p>© 2025 Crop Recommendation System | Precision Agriculture</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Entry point"""
    app = CropRecommendationApp()
    app.run()


if __name__ == "__main__":
    main()
