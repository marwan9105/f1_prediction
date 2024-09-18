import pandas as pd

def merge_csv_files(file_paths, output_file='merged_data_2023.csv'):
    """
    Fusionne les fichiers CSV en ne conservant que les données de 2023 et les colonnes nécessaires pour la prédiction.

    :param file_paths: Dictionnaire contenant les chemins vers les fichiers CSV.
    :param output_file: Nom du fichier CSV de sortie.
    """
    # Lire les fichiers CSV
    results_df = pd.read_csv(file_paths['results'])
    drivers_df = pd.read_csv(file_paths['drivers'])
    constructors_df = pd.read_csv(file_paths['constructors'])
    lap_times_df = pd.read_csv(file_paths['lap_times'])
    qualifying_df = pd.read_csv(file_paths['qualifying'])
    races_df = pd.read_csv(file_paths['races'])

    # Afficher les colonnes de chaque DataFrame pour diagnostic
    print("Colonnes dans 'results_df':", results_df.columns)
    print("Colonnes dans 'drivers_df':", drivers_df.columns)
    print("Colonnes dans 'constructors_df':", constructors_df.columns)
    print("Colonnes dans 'lap_times_df':", lap_times_df.columns)
    print("Colonnes dans 'qualifying_df':", qualifying_df.columns)
    print("Colonnes dans 'races_df':", races_df.columns)

    # Filtrer les données de 2023
    races_df = races_df[races_df['year'] == 2023]

    # Fusionner les DataFrames
    merged_df = results_df.merge(races_df, on='raceId', how='inner')
    merged_df = merged_df.merge(drivers_df[['driverId', 'driverRef', 'forename', 'surname', 'dob', 'nationality']], on='driverId', how='inner')
    merged_df = merged_df.merge(constructors_df[['constructorId', 'constructorRef', 'name']], on='constructorId', how='inner')
    merged_df = merged_df.merge(lap_times_df[['raceId', 'driverId', 'time']], on=['raceId', 'driverId'], how='inner')

    # Vérifier si la colonne 'grid' existe dans 'results_df' et 'qualifying_df'
    if 'grid' in results_df.columns:
        qualifying_df = qualifying_df[['raceId', 'driverId', 'position']]  # Utiliser 'position' si 'grid' n'est pas disponible
    else:
        qualifying_df = qualifying_df[['raceId', 'driverId', 'position']]
        print("Colonne 'grid' non trouvée. Utilisation de 'position' à la place.")

    # Fusionner avec qualifying_df
    merged_df = merged_df.merge(qualifying_df, on=['raceId', 'driverId'], how='inner')

    # Afficher les colonnes après la fusion pour diagnostic
    print("Colonnes après fusion:", merged_df.columns)

    # Sélectionner les colonnes nécessaires
    columns_to_keep = [
        'resultId', 'raceId', 'driverId', 'constructorId', 'circuitId', 
        'grid', 'position', 'points', 'laps', 'time', 'fastestLapTime', 
        'fastestLapSpeed', 'driverRef', 'forename', 'surname', 'dob', 
        'nationality', 'constructorRef', 'name'
    ]
    
    # Ajuster la liste des colonnes à conserver si certaines colonnes sont manquantes
    columns_to_keep = [col for col in columns_to_keep if col in merged_df.columns]

    # Filtrer le DataFrame pour garder seulement les colonnes nécessaires
    filtered_df = merged_df[columns_to_keep]

    # Sauvegarder le fichier CSV fusionné
    filtered_df.to_csv(output_file, index=False)

    print(f"Fichier CSV fusionné créé avec succès : '{output_file}'")

# Exemple d'utilisation
file_paths = {
    'results': 'new_csv/results.csv',
    'drivers': 'new_csv/filtered_drivers.csv',
    'constructors': 'new_csv/constructors.csv',
    'lap_times': 'new_csv/lap_times.csv',
    'qualifying': 'new_csv/qualifying.csv',
    'races': 'new_csv/races.csv'
}

merge_csv_files(file_paths)
