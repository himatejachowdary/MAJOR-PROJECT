"""
Simple Weather API Test
Tests OpenWeatherMap API without heavy dependencies
"""

import requests
import json


def test_api_simple():
    """Simple API test"""
    
    print("\n" + "="*60)
    print("TESTING OPENWEATHERMAP API")
    print("="*60)
    
    # Read API key from .env file
    api_key = None
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('OPENWEATHER_API_KEY'):
                    api_key = line.split('=')[1].strip()
                    break
    except FileNotFoundError:
        print("\n✗ .env file not found")
        return
    
    if not api_key:
        print("\n✗ API key not found in .env file")
        return
    
    print(f"\n✓ API Key found: {api_key[:10]}...")
    
    # Test API call
    print("\n📡 Testing API connection with Mumbai, India...")
    
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': 'Mumbai,IN',
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ API CONNECTION SUCCESSFUL!")
            print("\n" + "="*60)
            print("WEATHER DATA FOR MUMBAI")
            print("="*60)
            print(f"\nCity: {data['name']}, {data['sys']['country']}")
            print(f"Temperature: {data['main']['temp']}°C")
            print(f"Feels Like: {data['main']['feels_like']}°C")
            print(f"Humidity: {data['main']['humidity']}%")
            print(f"Weather: {data['weather'][0]['description'].title()}")
            print(f"Wind Speed: {data['wind']['speed']} m/s")
            
            print("\n" + "="*60)
            print("✅ YOUR API KEY IS WORKING PERFECTLY!")
            print("="*60)
            print("\n✓ You can now use live weather integration in the app")
            print("✓ The system will fetch real-time temperature and humidity")
            
        elif response.status_code == 401:
            print("\n✗ API Key is invalid")
            print("Please check your OpenWeatherMap API key")
            
        elif response.status_code == 404:
            print("\n✗ City not found")
            
        else:
            print(f"\n✗ API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n✗ Request timeout - check your internet connection")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Connection error - check your internet connection")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    test_api_simple()
