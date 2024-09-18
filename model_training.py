import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate_model():
    # Charger et prétraiter les données
    data = load_and_preprocess_data()
    
    # Définir les caractéristiques (features) et la variable cible (target)
    features = ['start_position', 'avg_driver_position', 'avg_constructor_position']
    target = 'position'  # Par exemple, vous pouvez prédire la position finale

    # Supprimer les lignes avec des valeurs manquantes pour les caractéristiques et la cible
    data = data.dropna(subset=features + [target])

    X = data[features]
    y = data[target]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_and_evaluate_model()
