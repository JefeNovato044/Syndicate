from  .base_tool import BaseTool
import requests


from pydantic import BaseModel, Field

class WeatherArgs(BaseModel):
    city: str = Field(..., description="Nombre de la ciudad de la cuál se quiere saber el clima")
    unit: str = Field(default="metric", description="Sistema de unidades para los valores de los campos. Puede ser 'metric' o 'imperial'")


class CurrentWeatherTool(BaseTool):
    """
    Tool to get current weather information for a specific city.
    """
    name = "get_current_weather"
    description = "Get the current weather information for a specific city. Returns temperature, conditions, and other weather data."
    args_schema = WeatherArgs

    def __init__(self, api_key: str):
        self.api_key = api_key
        super().__init__()

    def run(self, **kwargs) -> str:
        """
        Fetch current weather for a city.
        
        Args:
            **kwargs: Arguments validated against WeatherArgs schema
                - city: Name of the city
                - unit: Unit system ('metric' or 'imperial')
            
        Returns:
            JSON string with weather data
        """
        # Validate arguments using Pydantic schema
        args = self.args_schema(**kwargs)

        
        # Use validated arguments
        city_encoded = args.city.replace(" ", "%20").lower()
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={city_encoded}&apikey={self.api_key}"
        
        headers = {
            "accept": "application/json",
            "accept-encoding": "deflate, gzip, br"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.text