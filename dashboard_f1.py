import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    # Charger le fichier CSV pré-traité
    file_path = 'preprocessed_data.csv'
    df = pd.read_csv(file_path)
    
    # Filtrer les résultats uniquement pour Singapour (en utilisant le circuitId de Singapour)
    singapore_circuit_id = 32  # Remplacez par l'ID correct si nécessaire
    df_singapore = df[df['circuitId'] == singapore_circuit_id]
    
    return df_singapore

def main():
    st.title("Dashboard des Performances des Pilotes à Singapour")
    
    # Charger les données filtrées pour Singapour
    df = load_data()
    
    # Sélectionner un pilote
    selected_driver = st.selectbox("Sélectionnez un pilote", df['surname'].unique())
    
    # Filtrer les données pour le pilote sélectionné
    df_driver = df[df['surname'] == selected_driver]
    
    # Afficher les informations du pilote sélectionné
    st.subheader(f"Performances du pilote {selected_driver} à Singapour")
    
    # Afficher les statistiques clés
    avg_position = df_driver['position'].mean()
    avg_start_position = df_driver['start_position'].mean()
    st.metric(label="Position Moyenne", value=f"{avg_position:.2f}")
    st.metric(label="Position de Départ Moyenne", value=f"{avg_start_position:.2f}")
    
    # Graphique des positions du pilote sur Singapour
    st.subheader("Graphique des positions")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df_driver['raceId'], y=df_driver['position'], marker="o")
    plt.title(f"Positions du pilote {selected_driver} à Singapour")
    plt.xlabel('ID de la course')
    plt.ylabel('Position finale')
    st.pyplot(plt)

    # Si vous souhaitez ajouter plus de graphiques, vous pouvez en ajouter ici
    st.subheader("Comparaison avec les autres pilotes à Singapour")
    plt.figure(figsize=(10, 6))
    avg_position_by_driver = df.groupby('surname')['position'].mean().sort_values()
    sns.barplot(x=avg_position_by_driver.index, y=avg_position_by_driver.values)
    plt.title("Position Moyenne par Pilote à Singapour")
    plt.xticks(rotation=90)
    plt.xlabel("Pilote")
    plt.ylabel("Position Moyenne")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
