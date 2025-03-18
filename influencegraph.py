import pandas as pd
import os

def extract_data_for_date_from_folder(folder_path, date):
    """
    Parcourt tous les fichiers CSV dans un dossier, extrait les données pour une date donnée,
    et stocke les valeurs dans un dictionnaire avec les noms des colonnes comme clés.

    :param folder_path: Chemin du dossier contenant les fichiers CSV.
    :param date: Date pour laquelle extraire les données (format identique à celui des fichiers CSV).
    :return: Un dictionnaire avec les noms des colonnes comme clés et les listes de valeurs comme valeurs.
    """
    data_dict = {}

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith('.csv'):  # Vérifier que le fichier est un CSV
            file_path = os.path.join(folder_path, filename)

            # Charger le fichier CSV dans un DataFrame
            df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)

            # Vérifier si la date existe dans l'index
            if date in df.index:
                # Extraire les données pour la date donnée
                row = df.loc[date]

                # Ajouter les valeurs de la ligne au dictionnaire
                for col_name, value in row.items():
                    if col_name not in data_dict:
                        data_dict[col_name] = []  # Initialiser une liste pour chaque colonne
                    data_dict[col_name].append(value)

    return data_dict

# Exemple d'utilisation
folder_path = "PressureData_with_rendements"  # Chemin vers le dossier contenant les fichiers CSV
date_t = "2010"  # Remplacez par la date souhaitée

data_for_date = extract_data_for_date_from_folder(folder_path, date_t)
print(f"Données pour la date {date_t} :\n{data_for_date}")