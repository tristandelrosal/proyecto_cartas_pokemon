import os
from dotenv import load_dotenv
from pokemontcgsdk import Card
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Cargar el modelo
model_path = 'model/pokemon_card_predictor.keras'
if not os.path.exists(model_path):
    st.error(f"The model file '{model_path}' does not exist.")
else:
    try:
        model = load_model(model_path)
    except ValueError as e:
        st.error(f"Error loading model: {e}")

# Función para cargar y procesar una imagen
def load_and_preprocess_image(image_path, image_size=(128, 128)):
    if isinstance(image_path, np.ndarray):
        image_path = io.BytesIO(image_path)
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image)
    if image.shape[-1] == 4:  # Si la imagen tiene un canal alfa, eliminarlo
        image = image[..., :3]
    image = image / 255.0  # Normalizar la imagen
    return image


# Función para predecir el ID de una carta
def predict_card_id(image_path, model, image_size=(128, 128)):
    image = load_and_preprocess_image(image_path, image_size)
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Clase personalizada para manejar errores
class CardMarketError(Exception):
    pass

# Función para buscar una carta en el buscador de Cardmarket
def buscar_carta_cardmarket(nombre_carta):
    url_busqueda = "https://www.cardmarket.com/en/Pokemon/MainPage/search"
    params = {"searchString": nombre_carta}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url_busqueda, params=params, headers=headers)
    print(f"Searching for card: {nombre_carta}")
    print(f"Response status code: {response.status_code}")

    if response.status_code != 200:
        raise CardMarketError(f"Error al realizar la búsqueda: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar el primer resultado en los resultados de búsqueda
    resultado = soup.find("a", class_="product-link", href=True)
    if resultado:
        url_carta = "https://www.cardmarket.com" + resultado["href"]
        print(f"Found card URL: {url_carta}")
        return url_carta
    else:
        raise CardMarketError("No se encontró ningún resultado para la carta.")

# Función para extraer los precios de la gráfica de una carta específica
def obtener_precios_cardmarket(url_carta):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url_carta, headers=headers)
    print(f"Fetching URL: {url_carta}")
    print(f"Response status code: {response.status_code}")

    if response.status_code != 200:
        raise CardMarketError(f"Error al acceder a la página de la carta: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar el script con los datos de la gráfica
    script_tag = soup.find("script", text=re.compile("chartData"))
    if not script_tag:
        raise CardMarketError("No se encontró información de la gráfica en la página.")
    print("Found script tag with chartData")

    # Extraer los datos JSON de la gráfica
    json_data_match = re.search(r'chartData = (\[.*?\]);', script_tag.string)
    if json_data_match:
        print("Extracted chartData from script tag")
        chart_data = json.loads(json_data_match.group(1))
        print(f"Chart data: {chart_data}")
        return chart_data  # Convertir a lista Python
    else:
        raise CardMarketError("No se pudo extraer `chartData` del script.")

# Función para guardar los datos en un archivo JSON
def guardar_datos_json(datos, nombre_archivo="precios.json"):
    with open(nombre_archivo, "w") as archivo:
        json.dump(datos, archivo, indent=4)
    print(f"Datos guardados en {nombre_archivo}")

# Función para graficar los datos desde el archivo JSON
def graficar_datos_json(nombre_archivo="precios.json"):
    with open(nombre_archivo, "r") as archivo:
        datos = json.load(archivo)

    fechas = [datetime.strptime(fecha, "%Y-%m-%d") for fecha, _ in datos]
    precios = [precio for _, precio in datos]

    plt.figure(figsize=(10, 6))
    plt.plot(fechas, precios, marker='o', linestyle='-', color='b', label="Precio (€)")
    plt.title("Evolución de Precios de la Carta", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Precio (€)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Set the API key for the Pokémon TCG SDK
api_key = os.getenv('POKEMONTCG_IO_API_KEY')
if api_key:
    os.environ['POKEMONTCG_IO_API_KEY'] = api_key
    
# Set up the Streamlit app
st.title("Pokémon Card Finder")

if __name__ == "__main__":
    # Cargar el DataFrame para obtener el mapeo de IDs
    df = pd.read_csv('./data/cards_with_variations.csv')
    id_to_label = {i: label for i, label in enumerate(df['id'].astype('category').cat.categories)}

    # Probar el modelo con una nueva imagen
    test_image_path = './base3-2_original.png'  # Reemplazar con la ruta de la imagen
    predicted_class = predict_card_id(test_image_path, model)
    predicted_label = id_to_label[predicted_class]
    st.write(f'Predicted Label: {predicted_label}')

# Input for uploading an image
uploaded_image = st.file_uploader("Sube una imagen de tu carta pokemon", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    predicted_class = predict_card_id(uploaded_image, model)
    predicted_label = id_to_label[predicted_class]
    st.write(f'Predicted Label: {predicted_label}')
    
    card_id = predicted_label

    # Fetch and display the card details
    if card_id:
        try:
            card = Card.find(card_id)
            st.image(card.images.small)
            st.write(f"**Name:** {card.name}")
            st.write(f"**Set:** {card.set.name}")
            st.write(f"**Type:** {', '.join(card.types)}")
            st.write(f"**Rarity:** {card.rarity}")
            st.write(f"**HP:** {card.hp}")
            st.write(f"**Supertype:** {card.supertype}")
            st.write(f"**Subtype:** {', '.join(card.subtypes)}")
            
            # Display market price from TCGPlayer
            market_price = None
            if hasattr(card, 'tcgplayer') and card.tcgplayer:
                if hasattr(card.tcgplayer, 'prices') and card.tcgplayer.prices:
                    if hasattr(card.tcgplayer.prices, 'normal') and card.tcgplayer.prices.normal:
                        market_price = card.tcgplayer.prices.normal.market
            
            # If market price not found in TCGPlayer, check Cardmarket
            if market_price is None and hasattr(card, 'cardmarket') and card.cardmarket:
                if hasattr(card.cardmarket, 'prices') and card.cardmarket.prices:
                    market_price = card.cardmarket.prices.averageSellPrice
            
            try:
                precios = obtener_precios_cardmarket(card_id)
                if precios:
                    guardar_datos_json(precios)
                    graficar_datos_json()
                else:
                    print("No precios data found.")
            except CardMarketError as e:
                st.error(f"Error fetching card prices: {e}")
            
            st.write(f"precios: {precios}")

            if market_price:
                st.write(f"**Market Price:** ${market_price}")
            else:
                st.write("Market price not available.")
        except Exception as e:
            st.error(f"Error fetching card: {e}")