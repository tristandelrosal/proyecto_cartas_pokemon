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

# Input for uploading an image
uploaded_image = st.file_uploader("Sube una imagen de tu carta pokemon", type=["png", "jpg", "jpeg"])
cropped_image = None

@st.dialog("", width="large")
def cropper(uploaded_image): 
     cropped_image = st_cropperjs(uploaded_image, btn_text="Cortar imagen")
     if st.button("Recortar"):
         st.session_state.cropped_image = cropped_image
         st.rerun()
         
if st.button("Recortar imagen"):
    cropper(uploaded_image)
    
if 'cropped_image' in st.session_state:
    st.image(st.session_state.cropped_image)
else:
    st.info("No hay imagen recortada")

# Add a checkbox to control the visibility of the cropper
show_cropper = st.toggle("Recortar imagen")

if uploaded_image is not None:
    uploaded_image = uploaded_image.read()
    
    if show_cropper:
        cropped_image = st_cropperjs(uploaded_image, btn_text="Cortar imagen")
        if cropped_image is not None:
            # Convert cropped image to bytes if it's not already in bytes format
            if isinstance(cropped_image, Image.Image):
                buf = io.BytesIO()
                cropped_image.save(buf, format='PNG')
                cropped_image = buf.getvalue()

# Use cropped_image if available, otherwise use uploaded_image
image_to_predict = cropped_image if cropped_image is not None else uploaded_image

if image_to_predict is not None:
    predicted_class = predict_card_id(image_to_predict, model)
    if predicted_class is not None:
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
                    st.image(cropped_image if cropped_image is not None else uploaded_image, use_container_width=True)
                
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
                        
                        
                prices = get_card_prices_by_id(card_id)
                
                if prices:
                    st.success("Precios obtenidos:")
                    
                    # Crear un DataFrame para la gráfica
                    df = pd.DataFrame({
                    "Tipo de precio": list(prices.keys()),
                    "Precio (€)": list(prices.values())
                })
                    st.write(df["Precio (€)"])
                    # Ordenar los datos por 'Precio (€)' en orden ascendente
                    df = df.sort_values(by="Precio (€)", ascending=True)

                    # Crear la gráfica de barras con Plotly
                    fig = px.bar(
                        df,
                        x="Tipo de precio",
                        y="Precio (€)",
                        title="Precios de la carta según el tipo",
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

                    # Mostrar la gráfica en Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    df = pd.DataFrame({
                        "Tipo de precio": list(prices.keys()),
                        "Precio (€)": list(prices.values())
                    })

                    # Ordenar los datos por 'Precio (€)' en orden ascendente
                    df = df.sort_values(by="Precio (€)", ascending=True)

                    # Crear la gráfica de barras con Plotly
                    fig = px.bar(
                        df,
                        x="Tipo de precio",
                        y="Precio (€)",
                        text="Precio (€)",
                        title="Precios de la carta según el tipo",
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

                    # Mostrar la gráfica en Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                        # Mostrar la gráfica en Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No se encontraron precios.")


                if market_price:
                        st.write(f"**Market Price:** ${market_price}")
                else:
                        st.write("Market price not available.")
            except Exception as e:
                
                st.error(f"Error fetching card: {e}")
    else:
        st.error("Prediction failed. Please try again with a different image.")