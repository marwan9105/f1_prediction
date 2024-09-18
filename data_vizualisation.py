import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    file_path = 'preprocessed_data.csv'
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas. Veuillez vérifier le chemin et la présence du fichier.")
    
    return pd.read_csv(file_path)

def plot_avg_position_singapore(df):
    # Filtrer les données pour le circuit de Singapour (circuitId spécifique à Singapour)
    singapore_circuit_id = 32  # Remplacez par l'ID correct du circuit de Singapour si différent
    df_singapore = df[df['circuitId'] == singapore_circuit_id]
    
    plt.figure(figsize=(12, 6))
    avg_position_singapore = df_singapore.groupby('surname')['position'].mean().sort_values()
    sns.barplot(x=avg_position_singapore.index, y=avg_position_singapore.values)
    plt.title('Position Moyenne des Pilotes sur le Circuit de Singapour')
    plt.xlabel('Pilote')
    plt.ylabel('Position Moyenne')
    plt.xticks(rotation=90)
    plt.show()

def plot_start_vs_final_singapore(df):
    # Filtrer les données pour le circuit de Singapour
    singapore_circuit_id = 32  # Remplacez par l'ID correct du circuit de Singapour si différent
    df_singapore = df[df['circuitId'] == singapore_circuit_id]
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='start_position', y='position', data=df_singapore)
    plt.title('Position de Départ vs Position Finale sur le Circuit de Singapour')
    plt.xlabel('Position de Départ')
    plt.ylabel('Position Finale')
    plt.show()

def plot_driver_times_singapore(df):
    # Filtrer les données pour le circuit de Singapour
    singapore_circuit_id = 32  # Remplacez par l'ID correct du circuit de Singapour si différent
    df_singapore = df[df['circuitId'] == singapore_circuit_id]
    
    # Assurez-vous que la colonne 'time' est au format numérique
    df_singapore['fastestLapTime'] = pd.to_numeric(df_singapore['fastestLapTime'], errors='coerce')
    df_singapore = df_singapore.dropna(subset=['fastestLapTime'])
    
    plt.figure(figsize=(12, 6))
    avg_time_singapore = df_singapore.groupby('surname')['fastestLapTime'].mean().sort_values()
    sns.barplot(x=avg_time_singapore.index, y=avg_time_singapore.values)
    plt.title('Temps Moyens des Pilotes sur le Circuit de Singapour')
    plt.xlabel('Pilote')
    plt.ylabel('Temps Moyen')
    plt.xticks(rotation=90)
    plt.show()

def train_predictive_model(df):
    # Exemple simplifié de modèle de prédiction
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    # Filtrer les données pour le circuit de Singapour
    singapore_circuit_id = 15  # Remplacez par l'ID correct du circuit de Singapour si différent
    df_singapore = df[df['circuitId'] == singapore_circuit_id]
    
    # Préparer les données pour la modélisation
    X = df_singapore[['start_position', 'avg_driver_position', 'avg_constructor_position']]
    y = df_singapore['position']
    
    X = X.fillna(0)  # Remplacer les valeurs NaN par 0 pour cet exemple
    
    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédire les résultats
    y_pred = model.predict(X_test)
    
    # Afficher le rapport de classification
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    df = load_data()
    plot_avg_position_singapore(df)
    plot_start_vs_final_singapore(df)
    plot_driver_times_singapore(df)  # Ajouter le graphique des temps des pilotes
    train_predictive_model(df)

if __name__ == "__main__":
    main()
