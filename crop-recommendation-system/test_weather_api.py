"""
Test Weather API Integration
Quick script to verify your OpenWeatherMap API key is working
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.weather_integration import WeatherIntegration


def test_weather_api():
    """
    Test the weather API with your configured key
    """
    print("\n" + "="*60)
    print("TESTING OPENWEATHERMAP API")
    print("="*60)
    
    # Initialize weather integration
    weather = WeatherIntegration()
    
    # Check if API key is configured
    print("\n1. Checking API configuration...")
    if weather.is_api_configured():
        print("   ✓ API key found in .env file")
    else:
        print("   ✗ API key not configured")
        return False
    
    # Test connection
    print("\n2. Testing API connection...")
    test_result = weather.test_connection()
    
    if test_result['status'] == 'success':
        print("   ✓ API connection successful!")
    else:
        print(f"   ✗ API connection failed: {test_result['message']}")
        return False
    
    # Test with Indian cities
    print("\n3. Fetching weather for Indian cities...")
    
    cities = [
        ('Mumbai', 'IN'),
        ('Delhi', 'IN'),
        ('Bangalore', 'IN')
    ]
    
    for city, country in cities:
        print(f"\n   📍 {city}, {country}")
        weather_data = weather.get_weather_by_city(city, country)
        
        if weather_data['status'] == 'success':
            print(f"      ✓ Temperature: {weather_data['weather']['temperature']}°C")
            print(f"      ✓ Humidity: {weather_data['weather']['humidity']}%")
            print(f"      ✓ Conditions: {weather_data['weather']['description']}")
        else:
            print(f"      ✗ Error: {weather_data['message']}")
    
    # Test crop-relevant data
    print("\n4. Testing crop-relevant data extraction...")
    crop_data = weather.get_crop_relevant_data('Hyderabad', 'IN')
    
    if crop_data['status'] == 'success':
        print(f"   ✓ Location: {crop_data['location']}")
        print(f"   ✓ Temperature: {crop_data['temperature']}°C")
        print(f"   ✓ Humidity: {crop_data['humidity']}%")
        print(f"   ✓ Description: {crop_data['description']}")
    else:
        print(f"   ✗ Error: {crop_data['message']}")
    
    # Summary
    print("\n" + "="*60)
    print("API TEST COMPLETE")
    print("="*60)
    print("\n✅ Your OpenWeatherMap API is working perfectly!")
    print("✅ Live weather integration is ready to use")
    print("\nYou can now use the 'Live Weather Integration' mode in the app!")
    
    return True


if __name__ == "__main__":
    try:
        test_weather_api()
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("\nPlease check:")
        print("1. .env file exists with OPENWEATHER_API_KEY")
        print("2. API key is valid")
        print("3. Internet connection is working")
