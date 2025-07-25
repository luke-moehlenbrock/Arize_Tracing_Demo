"""
Simple weather tool that returns hardcoded example data for demonstration purposes.
"""

import json
from typing import Dict, Any


def get_weather(location: str) -> str:
    """
    Get weather information for a given location.
    
    Args:
        location (str): The location to get weather for
        
    Returns:
        str: JSON string containing weather information
    """

    #TODO - add tool call instrumentation here (OPTIONAL)
    # Hardcoded weather data for demonstration
    weather_data = {
        "san francisco": {
            "location": "San Francisco, CA",
            "temperature": "68°F (20°C)",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "wind": "12 mph NW",
            "forecast": "Mild and pleasant with some clouds"
        },
        "new york": {
            "location": "New York, NY",
            "temperature": "72°F (22°C)",
            "condition": "Sunny",
            "humidity": "58%",
            "wind": "8 mph SW",
            "forecast": "Clear skies and comfortable temperatures"
        },
        "london": {
            "location": "London, UK",
            "temperature": "59°F (15°C)",
            "condition": "Light rain",
            "humidity": "78%",
            "wind": "15 mph W",
            "forecast": "Typical London weather with light showers"
        },
        "tokyo": {
            "location": "Tokyo, Japan",
            "temperature": "75°F (24°C)",
            "condition": "Clear",
            "humidity": "62%",
            "wind": "6 mph E",
            "forecast": "Beautiful clear day with mild temperatures"
        }
    }
    
    # Normalize location to lowercase for lookup
    location_key = location.lower().strip()
    
    # Check if we have data for this location
    if location_key in weather_data:
        return json.dumps(weather_data[location_key], indent=2)
    else:
        # Default response for unknown locations
        return json.dumps({
            "location": location,
            "temperature": "72°F (22°C)",
            "condition": "Unknown",
            "humidity": "60%",
            "wind": "10 mph",
            "forecast": f"Weather data not available for {location}, but it's probably nice!"
        }, indent=2)


# Tool definition for OpenAI function calling
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for (e.g., 'San Francisco', 'New York', 'London')"
                }
            },
            "required": ["location"]
        }
    }
} 