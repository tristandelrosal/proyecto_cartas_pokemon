import os
from dotenv import load_dotenv
from pokemontcgsdk import Card
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import load as joblib_load
from streamlit_cropperjs import st_cropperjs

# Load environment variables from .env file
load_dotenv()

# Set the title and icon for the Streamlit app
st.set_page_config(page_title="Pokémon Card Finder", page_icon="./img/favicon.ico")

# Display the header image
header_image_path = "./img/Pokémon_Trading_Card_Game_logo.png"
st.image(header_image_path, use_container_width=True)

def load_model(model_path):
    """Load and validate the model using joblib"""
    if not os.path.exists(model_path):
        st.error(f"The model file '{model_path}' does not exist.")
        return None
    
    try:
        model = joblib_load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("The model file appears to be corrupted. Please ensure it was saved correctly.")
        return None


def load_and_preprocess_image(image_path, image_size=(256, 256)):
    try:
        image = Image.open(image_path).convert('RGB')  # Convertir a RGB
        image = image.resize(image_size)
        image = np.array(image)
        image = image.reshape(1, -1)  # Flatten to 2D array (1, width*height*channels)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict_card_id(image_path, model, image_size=(256, 256)):
    """Predict card ID with proper error handling"""
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    
    try:
        image = load_and_preprocess_image(image_path, image_size)
        predictions = model.predict(image)
        return predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Función para extraer los precios de la gráfica
def obtener_precios_cardmarket(url_carta):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url_carta, headers=headers)

    if response.status_code != 200:
        print(f"Error al acceder a la página de la carta: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar el script con los datos de la gráfica
    script_tag = soup.find("script", text=re.compile("chartData"))
    if not script_tag:
        print("No se encontró información de la gráfica en la página.")
        return None

    # Extraer los datos JSON de la gráfica
    json_data_match = re.search(r'chartData = (\[.*?\]);', script_tag.string)
    if json_data_match:
        return json.loads(json_data_match.group(1))  # Convertir a lista Python
    else:
        print("No se pudo extraer `chartData` del script.")
        return None

# Función para guardar los datos en un archivo JSON
def guardar_datos_json(datos, nombre_archivo="precios.json"):
    with open(nombre_archivo, "w") as archivo:
        json.dump(datos, archivo, indent=4)
    print(f"Datos guardados en {nombre_archivo}")

# Función para graficar los datos desde el JSON
def graficar_datos_json(nombre_archivo="precios.json"):
    if not os.path.exists(nombre_archivo):
        st.error(f"The file '{nombre_archivo}' does not exist.")
        return

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
    
# Load the model
model_path = './model/pokemon_card_classifier_shuffled_256.pkl'
model = load_model(model_path)

# Cargar el DataFrame para obtener el mapeo de IDs
df = pd.read_csv('./data/cards_with_variations.csv')
id_to_label = {i: label for i, label in enumerate(df['id'].astype('category').cat.categories)}

# Input for uploading an image
uploaded_image = st.file_uploader("Sube una imagen de tu carta pokemon", type=["png", "jpg", "jpeg"])
cropped_image = None

if uploaded_image is not None:
    uploaded_image = uploaded_image.read()
    cropped_image = st_cropperjs(uploaded_image, btn_text="Cortar imagen")
    if cropped_image is not None:
        cropped_image = Image.open(io.BytesIO(cropped_image))

# Use cropped_image if available, otherwise use uploaded_image
image_to_predict = cropped_image if cropped_image is not None else uploaded_image

if image_to_predict is not None:
    predicted_class = predict_card_id(image_to_predict, model)
    predicted_label = id_to_label[predicted_class]

    card_id = predicted_label
    
    st.write(f"**Predicted Card ID:** {card_id}")

    # Fetch and display the card details
    if card_id:
        try:
            card = Card.find(card_id)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Carta subida")
                st.image(uploaded_image, use_container_width=True)
                
            with col2:
                st.write(f"Carta encontrada | id: {card_id}")
                st.image(card.images.large, use_container_width=True)
            
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

            # Example base URL for Cardmarket
            base_url = "https://www.cardmarket.com/en/Pokemon/Cards/"

            # Replace spaces with hyphens and remove special characters
            formatted_set_name = re.sub(r'[^A-Za-z0-9-]', '', card.set.name.replace(' ', '-'))
            url_carta = f"{base_url}{formatted_set_name}/{card_id}"

            precios = obtener_precios_cardmarket(url_carta)
            if precios:
                guardar_datos_json(precios)
                graficar_datos_json()
            
            st.write(f"precios: {precios}")

            if market_price:
                st.write(f"**Market Price:** ${market_price}")
            else:
                st.write("Market price not available.")
        except Exception as e:
            st.error(f"Error fetching card: {e}")