"""
Weather Integration Module
Fetches live weather data from OpenWeatherMap API
"""

import requests
from datetime import datetime
from config import OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL


class WeatherIntegration:
    """
    Handles live weather data fetching from OpenWeatherMap API
    """
    
    def __init__(self, api_key=None):
        """
        Initialize weather integration
        
        Args:
            api_key (str): OpenWeatherMap API key
        """
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.base_url = OPENWEATHER_BASE_URL
    
    def get_weather_by_city(self, city_name, country_code=None):
        """
        Get current weather data for a city
        
        Args:
            city_name (str): Name of the city
            country_code (str): Optional country code (e.g., 'IN' for India)
            
        Returns:
            dict: Weather data or error message
        """
        if not self.api_key:
            return {
                'status': 'error',
                'message': 'API key not configured. Please set OPENWEATHER_API_KEY in .env file'
            }
        
        # Construct query
        if country_code:
            query = f"{city_name},{country_code}"
        else:
            query = city_name
        
        # API parameters
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'  # Celsius
        }
        
        try:
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            elif response.status_code == 401:
                return {
                    'status': 'error',
                    'message': 'Invalid API key. Please check your OpenWeatherMap API key'
                }
            elif response.status_code == 404:
                return {
                    'status': 'error',
                    'message': f'City "{city_name}" not found. Please check the spelling'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'API error: {response.status_code}'
                }
        
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'message': 'Request timeout. Please check your internet connection'
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'message': 'Connection error. Please check your internet connection'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}'
            }
    
    def get_weather_by_coordinates(self, latitude, longitude):
        """
        Get current weather data by coordinates
        
        Args:
            latitude (float): Latitude
            longitude (float): Longitude
            
        Returns:
            dict: Weather data or error message
        """
        if not self.api_key:
            return {
                'status': 'error',
                'message': 'API key not configured'
            }
        
        params = {
            'lat': latitude,
            'lon': longitude,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                return {
                    'status': 'error',
                    'message': f'API error: {response.status_code}'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error: {str(e)}'
            }
    
    def _parse_weather_data(self, raw_data):
        """
        Parse raw API response into structured format
        
        Args:
            raw_data (dict): Raw API response
            
        Returns:
            dict: Parsed weather data
        """
        try:
            parsed = {
                'status': 'success',
                'location': {
                    'city': raw_data['name'],
                    'country': raw_data['sys']['country'],
                    'coordinates': {
                        'latitude': raw_data['coord']['lat'],
                        'longitude': raw_data['coord']['lon']
                    }
                },
                'weather': {
                    'temperature': round(raw_data['main']['temp'], 1),
                    'feels_like': round(raw_data['main']['feels_like'], 1),
                    'temp_min': round(raw_data['main']['temp_min'], 1),
                    'temp_max': round(raw_data['main']['temp_max'], 1),
                    'humidity': raw_data['main']['humidity'],
                    'pressure': raw_data['main']['pressure'],
                    'description': raw_data['weather'][0]['description'].title(),
                    'main': raw_data['weather'][0]['main']
                },
                'wind': {
                    'speed': raw_data['wind']['speed'],
                    'direction': raw_data['wind'].get('deg', 0)
                },
                'clouds': raw_data['clouds']['all'],
                'timestamp': datetime.fromtimestamp(raw_data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
                'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).strftime('%H:%M:%S')
            }
            
            # Add rainfall if available
            if 'rain' in raw_data:
                parsed['rainfall'] = {
                    '1h': raw_data['rain'].get('1h', 0),
                    '3h': raw_data['rain'].get('3h', 0)
                }
            else:
                parsed['rainfall'] = {'1h': 0, '3h': 0}
            
            return parsed
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error parsing weather data: {str(e)}'
            }
    
    def get_crop_relevant_data(self, city_name, country_code=None):
        """
        Get weather data relevant for crop recommendation
        
        Args:
            city_name (str): Name of the city
            country_code (str): Optional country code
            
        Returns:
            dict: Simplified weather data for crop prediction
        """
        weather_data = self.get_weather_by_city(city_name, country_code)
        
        if weather_data['status'] == 'error':
            return weather_data
        
        # Extract only crop-relevant parameters
        crop_data = {
            'status': 'success',
            'location': f"{weather_data['location']['city']}, {weather_data['location']['country']}",
            'temperature': weather_data['weather']['temperature'],
            'humidity': weather_data['weather']['humidity'],
            'description': weather_data['weather']['description'],
            'timestamp': weather_data['timestamp']
        }
        
        return crop_data
    
    def format_weather_report(self, weather_data):
        """
        Format weather data as a readable report
        
        Args:
            weather_data (dict): Weather data from get_weather_by_city()
            
        Returns:
            str: Formatted report
        """
        if weather_data['status'] == 'error':
            return f"Error: {weather_data['message']}"
        
        report = []
        report.append("="*60)
        report.append("LIVE WEATHER DATA")
        report.append("="*60)
        
        report.append(f"\nLocation: {weather_data['location']['city']}, {weather_data['location']['country']}")
        report.append(f"Coordinates: {weather_data['location']['coordinates']['latitude']}°N, "
                     f"{weather_data['location']['coordinates']['longitude']}°E")
        report.append(f"Time: {weather_data['timestamp']}")
        
        report.append("\n--- Current Conditions ---")
        report.append(f"Weather: {weather_data['weather']['description']}")
        report.append(f"Temperature: {weather_data['weather']['temperature']}°C")
        report.append(f"Feels Like: {weather_data['weather']['feels_like']}°C")
        report.append(f"Humidity: {weather_data['weather']['humidity']}%")
        report.append(f"Pressure: {weather_data['weather']['pressure']} hPa")
        
        report.append("\n--- Additional Info ---")
        report.append(f"Wind Speed: {weather_data['wind']['speed']} m/s")
        report.append(f"Cloud Cover: {weather_data['clouds']}%")
        report.append(f"Sunrise: {weather_data['sunrise']}")
        report.append(f"Sunset: {weather_data['sunset']}")
        
        if weather_data['rainfall']['1h'] > 0:
            report.append(f"\nRainfall (last 1h): {weather_data['rainfall']['1h']} mm")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def is_api_configured(self):
        """
        Check if API key is configured
        
        Returns:
            bool: True if API key is set
        """
        return bool(self.api_key and self.api_key.strip())
    
    def test_connection(self):
        """
        Test API connection with a sample request
        
        Returns:
            dict: Test result
        """
        if not self.is_api_configured():
            return {
                'status': 'error',
                'message': 'API key not configured'
            }
        
        # Test with a known city
        result = self.get_weather_by_city('London', 'GB')
        
        if result['status'] == 'success':
            return {
                'status': 'success',
                'message': 'API connection successful'
            }
        else:
            return result


def main():
    """
    Demonstration of weather integration
    """
    weather = WeatherIntegration()
    
    # Check if API is configured
    if not weather.is_api_configured():
        print("⚠ Warning: OpenWeatherMap API key not configured")
        print("Please set OPENWEATHER_API_KEY in your .env file")
        print("\nTo get a free API key:")
        print("1. Visit https://openweathermap.org/api")
        print("2. Sign up for a free account")
        print("3. Generate an API key")
        print("4. Create a .env file with: OPENWEATHER_API_KEY=your_key_here")
        return
    
    # Test connection
    print("Testing API connection...")
    test_result = weather.test_connection()
    print(f"Status: {test_result['message']}\n")
    
    if test_result['status'] == 'success':
        # Example: Get weather for Indian cities
        cities = [
            ('Mumbai', 'IN'),
            ('Delhi', 'IN'),
            ('Bangalore', 'IN')
        ]
        
        for city, country in cities:
            print(f"\nFetching weather for {city}...")
            weather_data = weather.get_weather_by_city(city, country)
            
            if weather_data['status'] == 'success':
                print(weather.format_weather_report(weather_data))
                
                # Show crop-relevant data
                crop_data = weather.get_crop_relevant_data(city, country)
                print(f"\nCrop-Relevant Data:")
                print(f"  Temperature: {crop_data['temperature']}°C")
                print(f"  Humidity: {crop_data['humidity']}%")
            else:
                print(f"Error: {weather_data['message']}")


if __name__ == "__main__":
    main()
