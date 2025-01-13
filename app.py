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



# Función para obtener los datos de la gráfica de precios
def obtener_precios_cardmarket(id_carta):
    # URL base con el ID de la carta
    url = f"https://www.cardmarket.com/en/Pokemon/Products/Singles/{id_carta}"
    
    # Enviar la solicitud GET
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error al acceder a la página: {response.status_code}")
        return None

    # Analizar el HTML con BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extraer datos de la gráfica (normalmente en formato JSON o en etiquetas <script>)
    script_tag = soup.find("script", text=re.compile("chartData"))  # Busca el script que contiene "chartData"
    
    if not script_tag:
        print("No se encontró información de la gráfica en la página.")
        return None
    
    # Extraer el JSON de la gráfica
    json_data_match = re.search(r'chartData = (\[.*?\]);', script_tag.string)
    if json_data_match:
        precios_json = json_data_match.group(1)
        return precios_json
    else:
        print("No se pudo extraer el JSON de precios.")
        return None

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
            
            datos_json = obtener_precios_cardmarket(card_id)
            
            
            # Convertir los datos JSON a una lista de pares (fecha, precio)
            datos = json.loads(datos_json)

            # Separar las fechas y precios
            fechas = [datetime.strptime(fecha, "%Y-%m-%d") for fecha, _ in datos]
            precios = [precio for _, precio in datos]

            # Crear la gráfica
            plt.figure(figsize=(10, 6))
            plt.plot(fechas, precios, marker='o', linestyle='-', color='b', label="Precio (€)")

            # Configurar etiquetas y título
            plt.title("Evolución de Precios de la Carta", fontsize=16)
            plt.xlabel("Fecha", fontsize=12)
            plt.ylabel("Precio (€)", fontsize=12)
            plt.xticks(rotation=45)  # Rotar etiquetas del eje X
            plt.grid(alpha=0.5)  # Agregar una cuadrícula suave
            plt.legend(fontsize=12)

            # Mostrar la gráfica
            plt.tight_layout()  # Ajustar los márgenes
            plt.show()
            
            if market_price:
                st.write(f"**Market Price:** ${market_price}")
            else:
                st.write("Market price not available.")
        except Exception as e:
            st.error(f"Error fetching card: {e}")
            
            