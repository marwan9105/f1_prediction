import os
import pandas as pd

def load_and_preprocess_data():
    # Vérifier la présence des fichiers
    file_paths = {
        'results': 'new_csv/results.csv',
        'drivers': 'new_csv/filtered_drivers.csv',
        'constructors': 'new_csv/constructors.csv',
        'lap_times': 'new_csv/lap_times.csv',
        'qualifying': 'new_csv/qualifying.csv',
        'races': 'new_csv/races.csv'
    }
    
    for key, path in file_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Le fichier {path} est manquant. Veuillez vérifier le chemin et la présence du fichier.")
    
    # Charger les fichiers CSV nettoyés
    results = pd.read_csv(file_paths['results'])
    drivers = pd.read_csv(file_paths['drivers'])
    constructors = pd.read_csv(file_paths['constructors'])
    lap_times = pd.read_csv(file_paths['lap_times'])
    qualifying = pd.read_csv(file_paths['qualifying'])
    races = pd.read_csv(file_paths['races'])

    # Filtrer les courses de l'année 2023 uniquement
    circuits_2023 = [
        "Bahrain International Circuit", "Jeddah Corniche Circuit", "Albert Park Grand Prix Circuit",
        "Baku City Circuit", "Miami International Autodrome", "Circuit de Monaco",
        "Circuit de Barcelona-Catalunya", "Circuit Gilles Villeneuve", "Red Bull Ring",
        "Silverstone Circuit", "Hungaroring", "Circuit de Spa-Francorchamps",
        "Circuit Park Zandvoort", "Autodromo Nazionale di Monza", "Marina Bay Street Circuit",
        "Suzuka Circuit", "Losail International Circuit", "Circuit of the Americas",
        "Autódromo Hermanos Rodríguez", "Autódromo José Carlos Pace", "Las Vegas Strip Street Circuit",
        "Yas Marina Circuit"
    ]

    # Filtrer les courses de 2023
    races = races[races['year'] == 2023]
    races = races[races['name'].isin(circuits_2023)]

    # Fusionner les données
    results = results.merge(drivers, on='driverId', how='left')
    results = results.merge(constructors, on='constructorId', how='left')
    results = results.merge(races, on='raceId', how='left')
    results = results.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left', suffixes=('', '_qualifying'))

    # Vérifier les types de données
    print("Types de données des colonnes :")
    print(results.dtypes)

    # Convertir les colonnes nécessaires en numériques, en forçant les erreurs à NaN
    results['position'] = pd.to_numeric(results['position'], errors='coerce')
    results['position_qualifying'] = pd.to_numeric(results['position_qualifying'], errors='coerce')

    # Supprimer les lignes avec des valeurs NaN dans les colonnes critiques
    results = results.dropna(subset=['position', 'position_qualifying'])

    # Créer des caractéristiques
    results['start_position'] = results['position_qualifying']

    # Calculer les positions moyennes
    results['avg_driver_position'] = results.groupby('driverId')['position'].transform('mean')
    results['avg_constructor_position'] = results.groupby('constructorId')['position'].transform('mean')

    # Enregistrer les données prétraitées
    results.to_csv('preprocessed_data.csv', index=False)
    print("Données prétraitées sauvegardées dans 'preprocessed_data.csv'.")

    return results

if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("\nDataFrame prétraité :")
    print(data.head())
