import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

circuits_info = {
    1: "Albert Park Grand Prix Circuit",
    2: "Sepang International Circuit",
    3: "Bahrain International Circuit",
    4: "Circuit de Barcelona-Catalunya",
    5: "Istanbul Park",
    6: "Circuit de Monaco",
    7: "Circuit Gilles Villeneuve",
    8: "Circuit de Nevers Magny-Cours",
    9: "Silverstone Circuit",
    10: "Hockenheimring",
    11: "Hungaroring",
    12: "Valencia Street Circuit",
    13: "Circuit de Spa-Francorchamps",
    14: "Autodromo Nazionale di Monza",
    15: "Marina Bay Street Circuit",
    16: "Fuji Speedway",
    17: "Shanghai International Circuit",
    18: "Autódromo José Carlos Pace",
    19: "Indianapolis Motor Speedway",
    20: "Nürburgring",
    21: "Autodromo Enzo e Dino Ferrari",
    22: "Suzuka Circuit",
    80: "Las Vegas Strip Street Circuit",
    24: "Yas Marina Circuit",
    25: "Autódromo Juan y Oscar Gálvez",
    26: "Circuito de Jerez",
    27: "Autódromo do Estoril",
    28: "Okayama International Circuit",
    29: "Adelaide Street Circuit",
    30: "Kyalami",
    31: "Donington Park",
    32: "Autódromo Hermanos Rodríguez",
    33: "Phoenix street circuit",
    34: "Circuit Paul Ricard",
    35: "Korean International Circuit",
    36: "Autódromo Internacional Nelson Piquet",
    37: "Detroit Street Circuit",
    38: "Brands Hatch",
    39: "Circuit Park Zandvoort",
    40: "Zolder",
    41: "Dijon-Prenois",
    42: "Fair Park",
    43: "Long Beach",
    44: "Las Vegas Street Circuit",
    45: "Jarama",
    46: "Watkins Glen",
    47: "Scandinavian Raceway",
    48: "Mosport International Raceway",
    49: "Montjuïc",
    50: "Nivelles-Baulers",
    51: "Charade Circuit",
    52: "Circuit Mont-Tremblant",
    53: "Rouen-Les-Essarts",
    54: "Le Mans",
    55: "Reims-Gueux",
    56: "Prince George Circuit",
    57: "Zeltweg",
    58: "Aintree",
    59: "Circuito da Boavista",
    60: "Riverside International Raceway",
    61: "AVUS",
    62: "Monsanto Park Circuit",
    63: "Sebring International Raceway",
    64: "Ain Diab",
    65: "Pescara Circuit",
    66: "Circuit Bremgarten",
    67: "Circuit de Pedralbes",
    68: "Buddh International Circuit",
    69: "Circuit of the Americas",
    70: "Red Bull Ring",
    71: "Sochi Autodrom",
    73: "Baku City Circuit",
    75: "Autódromo Internacional do Algarve",
    76: "Autodromo Internazionale del Mugello",
    77: "Jeddah Corniche Circuit",
    78: "Losail International Circuit",
    79: "Miami International Autodrome"
}

def load_data():

    file_path = 'merged_data_2023.csv'
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['surname'])
    
    return df

def load_weather_data():

    weather_path = 'weather.parquet'
    weather_df = pd.read_parquet(weather_path)
    
    return weather_df

def merge_data_with_weather(df, weather_df, selected_circuit_id):
    weather_for_circuit = weather_df[weather_df['circuitId'] == selected_circuit_id]
    
    merged_df = pd.merge(df, weather_for_circuit, on='circuitId', how='left')
    
    return merged_df

def train_model(df):
    # Inclure les données météo dans les caractéristiques d'entrée
    X = df[['start_position', 'avg_driver_position', 'avg_constructor_position', 'temperature', 'humidity', 'wind_speed']]
    y = df['position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modèle de RandomForest pour la classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def train_lap_time_model(df):
    # Vérifier si la colonne 'laps' est présente
    if 'laps' not in df.columns:
        st.error("La colonne 'laps' est manquante dans les données.")
        return None
    
    # Inclure les données météo dans les caractéristiques d'entrée
    X = df[['start_position', 'avg_driver_position', 'avg_constructor_position', 'temperature', 'humidity', 'wind_speed']]
    y = df['laps']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modèle de RandomForest pour la régression (temps de tour)
    lap_time_model = RandomForestRegressor(n_estimators=100, random_state=42)
    lap_time_model.fit(X_train, y_train)
    
    return lap_time_model

def simulate_race(df, model, lap_time_model, selected_circuit_id):
    # Sélection des pilotes disponibles
    drivers = df['surname'].dropna().unique()
    
    selected_positions = []
    simulation_results = []
    
    st.subheader("Sélectionnez une position de départ unique pour chaque pilote")
    
    # Filtrer les données par le circuit sélectionné
    df_circuit = df[df['circuitId'] == selected_circuit_id]
    
    # Créer un DataFrame avec les moyennes par pilote pour le circuit sélectionné
    df_driver_avg = df_circuit.groupby('surname').agg({
        'avg_driver_position': 'mean',
        'avg_constructor_position': 'mean'
    }).reset_index()
    
    df_driver_avg['start_position'] = df_driver_avg['avg_driver_position'].rank(ascending=True).astype(int)
    
    for _, row in df_driver_avg.iterrows():
        driver = row['surname']
        start_position = row['start_position']
        
        available_positions = [i for i in range(1, 21) if i not in selected_positions]
        
        st.markdown(f"<h3 style='font-size:20px'><b>{driver}</b></h3>", unsafe_allow_html=True)
        start_position = st.selectbox(f"Position de départ pour {driver} (suggérée : {start_position})",
                                      available_positions, 
                                      index=available_positions.index(start_position) if start_position in available_positions else 0,
                                      key=driver)
        
        selected_positions.append(start_position)
        
        avg_driver_position = row['avg_driver_position']
        avg_constructor_position = row['avg_constructor_position']
        
        input_data = pd.DataFrame({
            'start_position': [start_position],
            'avg_driver_position': [avg_driver_position],
            'avg_constructor_position': [avg_constructor_position],
            # Inclure les données météo correspondantes pour le circuit
            'temperature': [df_circuit['temperature'].mean()],
            'humidity': [df_circuit['humidity'].mean()],
            'wind_speed': [df_circuit['wind_speed'].mean()]
        })
        
        predicted_score = model.predict(input_data)[0]
        if lap_time_model:
            predicted_best_lap_time = lap_time_model.predict(input_data)[0]
        else:
            predicted_best_lap_time = None
        
        simulation_results.append((driver, predicted_score, predicted_best_lap_time))
    
    simulation_results.sort(key=lambda x: x[1])
    
    final_positions = pd.DataFrame(simulation_results, columns=['Pilote', 'Score', 'Temps du Meilleur Tour'])
    final_positions['Position Finale'] = range(1, len(final_positions) + 1)
    
    return final_positions

def display_podium(df):
    podium = df.head(3)
    
    st.subheader("Podium des Trois Premiers")
    for i, row in podium.iterrows():
        position = row['Position Finale']
        driver = row['Pilote']
        st.markdown(f"<h2 style='font-size:24px'>{position}ème Place : <b>{driver}</b></h2>", unsafe_allow_html=True)

def main():
    st.title("Simulation de Course de F1 avec Données Météo")

    df = load_data()
    weather_df = load_weather_data()

    selected_circuit_id = st.selectbox("Sélectionnez un circuit", options=circuits_info.keys(), format_func=lambda x: circuits_info[x])
    
    st.write(f"Circuit sélectionné : {circuits_info[selected_circuit_id]}")

    df_with_weather = merge_data_with_weather(df, weather_df, selected_circuit_id)
    
    model = train_model(df_with_weather)
    lap_time_model = train_lap_time_model(df_with_weather)
    
    if model and lap_time_model:
        final_positions = simulate_race(df_with_weather, model, lap_time_model, selected_circuit_id)
        
        st.subheader("Positions Finales Simulées")
        st.write(final_positions)
        
        display_podium(final_positions)

if __name__ == "__main__":
    main()
