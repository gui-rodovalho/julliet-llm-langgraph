from config import OPEN_WEATHER_KEY
import requests

def get_weather(cidade: str)-> str: 
    #cidade = "Campo Grande"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={OPEN_WEATHER_KEY}"

    

    response = requests.get(url, timeout=5)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Erro: {response.status_code} - {response.text}")
    
    
    cidade = data.get("name", "desconhecida")
   
    temp_k = data.get("main", {}).get("temp")
    temp_min_k = data.get("main", {}).get("temp_min")
    temp_max_k = data.get("main", {}).get("temp_max")
    umidade = data.get("main", {}).get("humidity")
    pressao = data.get("main", {}).get("pressure")
    vento = data.get("wind", {})
    descricao = data.get("weather", [{}])[0].get("description", "").capitalize()
    nuvens = data.get("clouds", {}).get("all", 0)
   
    
    # Converte temps para Celsius
    def k_to_c(k):
        return k - 273.15 if k else None

    temp = k_to_c(temp_k)
    temp_min = k_to_c(temp_min_k)
    temp_max = k_to_c(temp_max_k)

    # Data e hora local
    #dt_local = datetime.utcfromtimestamp(timestamp + timezone).strftime("%Y-%m-%d %H:%M:%S")

    resumo = f"""
    Previsão de tempo:

    - Condição: {descricao}
    - Temperatura: {temp:.1f}°C (min: {temp_min:.1f}°C, max: {temp_max:.1f}°C)
    - Umidade: {umidade}%
    - Pressão atmosférica: {pressao} hPa
    - Cobertura de nuvens: {nuvens}%
    - Vento: {vento.get('speed', 'N/A')} m/s, direção {vento.get('deg', 'N/A')}°
    """
    print(resumo)
    return resumo

#get_weather("-20.55855749608114", "-54.57646163676827", "1234")
"""
def get_loc(cidade: str):
    print("iniciando")
    resultado = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={OPEN_WEATHER_KEY}")
    if resultado.status_code == 200:
        data = resultado.json()
        print(data)
    else:
        print(f"Erro: {resultado.status_code} - {resultado.text}")


get_loc("campo grande")
"""