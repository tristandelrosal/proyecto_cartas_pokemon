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
import plotly.express as px

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

def load_and_preprocess_image(image_data, image_size=(256, 256)):
    try:
        if isinstance(image_data, Image.Image):
            # Convert PIL image to bytes
            buf = io.BytesIO()
            image_data.save(buf, format='PNG')
            image_data = buf.getvalue()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Convert to RGB
        image = image.resize(image_size)
        image = np.array(image)
        image = image.reshape(1, -1)  # Flatten to 2D array (1, width*height*channels)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def predict_card_id(image_data, model, image_size=(256, 256)):
    """Predict card ID with proper error handling"""
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    
    try:
        image = load_and_preprocess_image(image_data, image_size)
        if image is None:
            return None
        predictions = model.predict(image)
        return predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None
    
    
def get_card_prices_by_id(card_id):
    url = f"https://api.pokemontcg.io/v2/cards/{card_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error al obtener la carta: {response.status_code}")
        return None
    
    data = response.json()
    
    if "data" not in data:
        print("Carta no encontrada.")
        return None

    card = data["data"]  # Datos de la carta
    prices = card.get("cardmarket", {}).get("prices", {})
    
    # Filtrar y renombrar los valores
    relevant_prices = {
        "Desde": f"{prices.get('lowPrice', 0):.2f} €",
        "Tendencia de precio": f"{prices.get('trendPrice', 0):.2f} €",
        "Precio medio 30 días": f"{prices.get('avg30', 0):.2f} €",
        "Precio medio 7 días": f"{prices.get('avg7', 0):.2f} €",
        "Precio medio 1 día": f"{prices.get('avg1', 0):.2f} €"
    }
    
    # Filtrar valores con 0 €
    filtered_prices = {key: value for key, value in relevant_prices.items() if not value.startswith("0.00")}
    return filtered_prices
        

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

# Inicializar variables
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'card_id' not in st.session_state:
    st.session_state.card_id = None
if 'card' not in st.session_state:
    st.session_state.card = None

# Input for uploading an image
uploaded_image = st.file_uploader("Sube una imagen de tu carta pokemon", type=["png", "jpg", "jpeg"])
cropped_image = None

# Reset session state if no image is uploaded
if uploaded_image is None:
    st.session_state.predicted_class = None
    st.session_state.card_id = None
    st.session_state.card_found = False
    st.session_state.cropped_image = None
    st.session_state.card = None
    
if uploaded_image is not None:
    uploaded_image = uploaded_image.read()
    
    if 'card_found' not in st.session_state:
        st.session_state.card_found = False
    
    # Inicializa el estado de la sesión para cropped_image
    if 'cropped_image' not in st.session_state:
        st.session_state.cropped_image = None

    @st.dialog("Recorta tu carta")
    def cropper(uploaded_image):
        if uploaded_image is not None:
            cropped_image = st_cropperjs(uploaded_image, btn_text="Cortar imagen")
            if cropped_image is not None:
                st.session_state.cropped_image = cropped_image
                st.rerun()
            if st.button("Cancelar recorte"):
                st.session_state.cropped_image = None
                st.rerun()
        else:
            st.warning("Por favor, sube una imagen primero.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Recortar imagen", use_container_width=True):
            cropper(uploaded_image)
        if st.session_state.cropped_image is not None:
            try:
                cropped_image = Image.open(io.BytesIO(st.session_state.cropped_image))
                st.image(cropped_image, use_container_width=True, caption="Imagen recortada")
                image_to_predict = cropped_image
            except Exception as e:
                st.error(f"Error al cargar la imagen recortada: {e}")
        elif uploaded_image is not None:
            st.image(uploaded_image, use_container_width=True, caption="Imagen subida")
            image_to_predict = Image.open(io.BytesIO(uploaded_image))
        

    with col2:
        if st.button("Predecir carta", use_container_width=True):
            try:
                if image_to_predict is not None:
                    image_data = io.BytesIO()
                    image_to_predict.save(image_data, format='PNG')
                    image_data = image_data.getvalue()
                    st.session_state.predicted_class = predict_card_id(image_data, model)
                    if st.session_state.predicted_class is not None:
                        st.session_state.card_id = id_to_label[st.session_state.predicted_class]
                    
                        # Fetch and display the card details
                        if st.session_state.card_id:
                            try:
                                card = Card.find(st.session_state.card_id)
                                st.image(card.images.large, use_container_width=True, caption="Carta encontrada")
                                st.session_state.card_found = True
                            except Exception as e:
                                st.error(f"Error fetching card: {e}")
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")
            
    # Fetch and display the card details
if st.session_state.card_id:
    try:
        st.session_state.card = Card.find(st.session_state.card_id)
        st.session_state.card_found = True
    except Exception as e:
        st.error(f"Error fetching card: {e}")

    if st.session_state.card_found:
        if st.session_state.predicted_class is not None and st.session_state.card is not None:
            
            tab, tab2 = st.tabs(["Información de la carta", "Gráfica de precio"])

            # Agregar contenido a cada pestaña
            with tab:
                col1, col2 = st.columns(2)

                

                with col1:
                    
                    st.write(f"**ID de la carta predecida:** {st.session_state.card_id}") 
                    st.write(f"**Nombre:** {st.session_state.card.name}")
                    st.write(f"**Set:** {st.session_state.card.set.name}")
                    st.write(f"**Tipo:** {', '.join(st.session_state.card.types)}")
                    st.write(f"**Rareza:** {st.session_state.card.rarity}")

                    
                with col2:
                    st.write(f"**HP:** {st.session_state.card.hp}")
                    st.write(f"**Tipo de carta:** {st.session_state.card.supertype}")
                    st.write(f"**Categoria:** {', '.join(st.session_state.card.subtypes)}")

                    # Display market price from TCGPlayer
                    market_price = None
                    if hasattr(st.session_state.card, 'tcgplayer') and st.session_state.card.tcgplayer:
                        if hasattr(st.session_state.card.tcgplayer, 'prices') and st.session_state.card.tcgplayer.prices:
                            if hasattr(st.session_state.card.tcgplayer.prices, 'normal') and st.session_state.card.tcgplayer.prices.normal:
                                market_price = st.session_state.card.tcgplayer.prices.normal.market

                    # If market price not found in TCGPlayer, check Cardmarket
                    if market_price is None and hasattr(st.session_state.card, 'cardmarket') and st.session_state.card.cardmarket:
                        if hasattr(st.session_state.card.cardmarket, 'prices') and st.session_state.card.cardmarket.prices:
                            market_price = st.session_state.card.cardmarket.prices.averageSellPrice
                            
                    if market_price:
                        st.write(f"**Market Price:** ${market_price}")
                    else:
                        st.write("Market price not available.")
                    
            with tab2:
                prices = get_card_prices_by_id(st.session_state.card_id)
                
                if prices:                
                    # Crear un DataFrame para la gráfica
                    df = pd.DataFrame({
                    "Tipo de precio": list(prices.keys()),
                    "Precio (€)": list(prices.values())
                })

                    # Convertir los valores de la columna "Precio (€)" a números
                    df["Precio (€)"] = df["Precio (€)"].str.replace('€', '').astype(float)

                    # Ordenar los datos por 'Precio (€)' en orden ascendente
                    df = df.sort_values(by="Precio (€)", ascending=True)

                    # Crear la gráfica de barras con Plotly
                    fig = px.bar(
                        df,
                        x="Tipo de precio",
                        y="Precio (€)",
                        text="Precio (€)",
                        title="Precios de la carta",
                        labels={"Precio (€)": "Precio en €", "Tipo de precio": "Tipo de Precio"},
                        color="Precio (€)",  # Colorear barras según su valor
                        color_continuous_scale="Blues"
                    )

                    # Ajustar la posición del texto y formato
                    fig.update_traces(
                        texttemplate="€ %{y:.2f}",
                        textposition="outside",
                    )

                    # Asegurarse de que el eje Y empiece desde 0
                    fig.update_layout(
                        xaxis_title="Tipo de Precio",
                        yaxis_title="Precio (€)",
                        uniformtext_minsize=8,
                        uniformtext_mode="hide",
                        yaxis=dict(
                            showgrid=True,
                            range=[0, df["Precio (€)"].max() * 1.1]  # Establecer el rango desde 0 hasta un poco más del máximo
                        ),
                        xaxis=dict(showgrid=False)
                    )

                    # Mostrar la gráfica en Streamlit con un key único
                    st.plotly_chart(fig, use_container_width=True, key="plotly_chart_1")

                else:
                    st.write("No se encontraron precios.")
        else:
            st.error("Prediction failed. Please try again with a different image.")