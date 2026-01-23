import requests

API_KEY = "892e9461d30e3702e6976bfe327d69f7"
lat = 6.9271
lon = 79.8612

url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
response = requests.get(url)

print(response.status_code)  # Should print: 200
print(response.json())  # Should show weather data