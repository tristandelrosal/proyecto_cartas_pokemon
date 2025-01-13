import os
from dotenv import load_dotenv
from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import io

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
def load_and_preprocess_image(uploaded_file, image_size=(128, 128)):
    try:
        image = Image.open(uploaded_file)
        image = image.resize(image_size)
        image = np.array(image)
        if image.shape[-1] == 4:  # Si la imagen tiene un canal alfa, eliminarlo
            image = image[..., :3]
        image = image / 255.0  # Normalizar la imagen
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

# Función para predecir el ID de una carta
def predict_card_id(image, model, image_size=(128, 128)):
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

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

# Main Streamlit app
uploaded_image = st.file_uploader("Sube una imagen de tu carta pokemon", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    try:
        image = load_and_preprocess_image(uploaded_image)
        predicted_class = predict_card_id(image, model)
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
                
                if market_price:
                    st.write(f"**Market Price:** ${market_price}")
                else:
                    st.write("Market price not available.")
            except Exception as e:
                st.error(f"Error fetching card: {e}")
    except ValueError as e:
        st.error(e)
