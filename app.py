import os
from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity
import streamlit as st

os.environ['POKEMONTCG_IO_API_KEY'] = 'e7054604-7781-4e38-b164-0adc8552a081'

# Set up the Streamlit app
st.title("Pokémon Card Finder")

# Input for card ID
card_id = st.text_input("Enter the Pokémon Card ID:")

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