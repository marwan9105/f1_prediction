import streamlit as st
import pandas as pd

# Charger les données
pilote_data = pd.read_csv('merged_data_2023.csv')

# Filtrer pour avoir une seule ligne par pilote
pilotes_unique = pilote_data.drop_duplicates(subset=['pilote'])

# Titre du dashboard
st.title("Simulation de course F1 - Sélection des positions de départ")

# Créer un dictionnaire pour stocker les positions choisies pour chaque pilote
positions_depart = {}

# Affichage des pilotes avec sélecteurs pour la position de départ
for pilote in pilotes_unique['pilote']:
    positions_depart[pilote] = st.selectbox(f"Sélectionnez la position de départ pour {pilote}", 
                                            options=range(1, 21), 
                                            index=0)

# Afficher les positions sélectionnées
st.write("Positions de départ choisies :")
st.write(positions_depart)

# Calculer et afficher le classement
classement = pilote_data.groupby('pilote').agg({'classement': 'min'}).reset_index()

st.write("Classement des pilotes:")
st.write(classement)
