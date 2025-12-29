"""
Configuration settings for Crop Recommendation System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Model Configuration
MODEL_PATH = "models/"
DATA_PATH = "data/"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Crop Information Database
CROP_INFO = {
    'rice': {
        'optimal_npk': {'N': 80, 'P': 40, 'K': 40},
        'description': 'Rice is a staple food crop that requires flooded conditions',
        'season': 'Kharif (Monsoon)',
        'duration': '120-150 days'
    },
    'maize': {
        'optimal_npk': {'N': 80, 'P': 40, 'K': 20},
        'description': 'Maize is a versatile cereal crop',
        'season': 'Kharif & Rabi',
        'duration': '80-110 days'
    },
    'chickpea': {
        'optimal_npk': {'N': 20, 'P': 60, 'K': 40},
        'description': 'Chickpea is a protein-rich pulse crop',
        'season': 'Rabi (Winter)',
        'duration': '100-120 days'
    },
    'kidneybeans': {
        'optimal_npk': {'N': 20, 'P': 60, 'K': 20},
        'description': 'Kidney beans are nitrogen-fixing legumes',
        'season': 'Kharif',
        'duration': '90-120 days'
    },
    'pigeonpeas': {
        'optimal_npk': {'N': 20, 'P': 50, 'K': 20},
        'description': 'Pigeon peas are drought-resistant pulse crops',
        'season': 'Kharif',
        'duration': '150-180 days'
    },
    'mothbeans': {
        'optimal_npk': {'N': 20, 'P': 40, 'K': 20},
        'description': 'Moth beans are drought-tolerant legumes',
        'season': 'Kharif',
        'duration': '75-90 days'
    },
    'mungbean': {
        'optimal_npk': {'N': 20, 'P': 50, 'K': 20},
        'description': 'Mung beans are quick-growing pulse crops',
        'season': 'Kharif & Summer',
        'duration': '60-75 days'
    },
    'blackgram': {
        'optimal_npk': {'N': 20, 'P': 60, 'K': 40},
        'description': 'Black gram is a protein-rich pulse',
        'season': 'Kharif & Rabi',
        'duration': '70-90 days'
    },
    'lentil': {
        'optimal_npk': {'N': 20, 'P': 60, 'K': 20},
        'description': 'Lentils are cool-season pulse crops',
        'season': 'Rabi',
        'duration': '100-110 days'
    },
    'pomegranate': {
        'optimal_npk': {'N': 40, 'P': 40, 'K': 40},
        'description': 'Pomegranate is a fruit crop requiring well-drained soil',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'banana': {
        'optimal_npk': {'N': 100, 'P': 75, 'K': 175},
        'description': 'Banana requires high potassium for fruit development',
        'season': 'Year-round',
        'duration': '9-12 months'
    },
    'mango': {
        'optimal_npk': {'N': 50, 'P': 50, 'K': 50},
        'description': 'Mango is a tropical fruit tree',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'grapes': {
        'optimal_npk': {'N': 40, 'P': 40, 'K': 80},
        'description': 'Grapes require well-drained soil and moderate climate',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'watermelon': {
        'optimal_npk': {'N': 100, 'P': 50, 'K': 50},
        'description': 'Watermelon is a summer fruit crop',
        'season': 'Summer',
        'duration': '80-100 days'
    },
    'muskmelon': {
        'optimal_npk': {'N': 100, 'P': 50, 'K': 50},
        'description': 'Muskmelon thrives in warm weather',
        'season': 'Summer',
        'duration': '70-90 days'
    },
    'apple': {
        'optimal_npk': {'N': 40, 'P': 40, 'K': 40},
        'description': 'Apple requires cool climate for fruit development',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'orange': {
        'optimal_npk': {'N': 60, 'P': 40, 'K': 60},
        'description': 'Orange is a citrus fruit requiring warm climate',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'papaya': {
        'optimal_npk': {'N': 100, 'P': 100, 'K': 100},
        'description': 'Papaya is a fast-growing tropical fruit',
        'season': 'Year-round',
        'duration': '9-12 months'
    },
    'coconut': {
        'optimal_npk': {'N': 60, 'P': 40, 'K': 120},
        'description': 'Coconut requires coastal tropical climate',
        'season': 'Perennial',
        'duration': 'Year-round'
    },
    'cotton': {
        'optimal_npk': {'N': 120, 'P': 60, 'K': 60},
        'description': 'Cotton is a fiber crop requiring warm climate',
        'season': 'Kharif',
        'duration': '150-180 days'
    },
    'jute': {
        'optimal_npk': {'N': 60, 'P': 30, 'K': 30},
        'description': 'Jute is a fiber crop requiring high humidity',
        'season': 'Kharif',
        'duration': '120-150 days'
    },
    'coffee': {
        'optimal_npk': {'N': 100, 'P': 50, 'K': 100},
        'description': 'Coffee requires shade and moderate rainfall',
        'season': 'Perennial',
        'duration': 'Year-round'
    }
}

# Fertilizer Types and NPK Ratios
FERTILIZER_DATABASE = {
    'Urea': {'N': 46, 'P': 0, 'K': 0, 'cost_per_kg': 6},
    'DAP': {'N': 18, 'P': 46, 'K': 0, 'cost_per_kg': 27},
    'MOP': {'N': 0, 'P': 0, 'K': 60, 'cost_per_kg': 17},
    'NPK 10:26:26': {'N': 10, 'P': 26, 'K': 26, 'cost_per_kg': 24},
    'NPK 12:32:16': {'N': 12, 'P': 32, 'K': 16, 'cost_per_kg': 22},
    'NPK 20:20:20': {'N': 20, 'P': 20, 'K': 20, 'cost_per_kg': 25},
    'Ammonium Sulphate': {'N': 21, 'P': 0, 'K': 0, 'cost_per_kg': 8},
    'SSP': {'N': 0, 'P': 16, 'K': 0, 'cost_per_kg': 6},
    'Potash': {'N': 0, 'P': 0, 'K': 50, 'cost_per_kg': 15}
}

# Streamlit UI Configuration
APP_TITLE = "🌾 Intelligent Crop Recommendation System"
APP_SUBTITLE = "AI-Powered Precision Agriculture"
PAGE_ICON = "🌾"

# Color scheme
COLORS = {
    'primary': '#2E7D32',
    'secondary': '#66BB6A',
    'accent': '#FFA726',
    'background': '#F5F5F5',
    'text': '#212121'
}
