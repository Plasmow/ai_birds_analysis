import pandas as pd
import os
import matplotlib.pyplot as plt


#The goal is to create a dictionary for a specific date of dataframes, where the keys are the countries
#and the values are tuples of the form (pressure_type,value)

#This allows us to plot birds at time t as a function of pressure at time tthen birds at time t+1 as a function of pressure at time t

def extract_data_for_date_from_folder(folder_path, date):
    """
    Parcourt tous les fichiers CSV dans un dossier, extrait les données pour une date donnée,
    et stocke les valeurs dans un dictionnaire avec les pays comme clés.
    Les valeurs sont des dictionnaires avec le type de pression comme clé et la valeur comme valeur.

    :param folder_path: Chemin du dossier contenant les fichiers CSV.
    :param date: Date pour laquelle extraire les données (format identique à celui des fichiers CSV).
    :return: Un dictionnaire avec les pays comme clés et des dictionnaires (type de pression: valeur).
    """
    data_dict = {}

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Vérifier que le fichier est un CSV
            file_path = os.path.join(folder_path, filename)

            # Charger le fichier CSV avec les en-têtes
            df = pd.read_csv(file_path, sep=";", decimal=",", header=0)

            # Définir la première colonne comme index
            df.set_index(df.columns[0], inplace=True)

            # Vérifier si la date existe dans l'index
            if date in df.index:
                # Extraire les données pour la date donnée
                row = df.loc[date]

                # Ajouter les valeurs de la ligne au dictionnaire
                for col_name, value in row.items():
                    country = col_name  # Les clés sont les pays
                    pressure_type = filename[:3].capitalize()  # Type de pression basé sur le nom du fichier
                    if country not in data_dict:
                        data_dict[country] = {}  # Initialiser un dictionnaire pour chaque pays
                    # Ajouter une entrée (type de pression: valeur)
                    data_dict[country][pressure_type] = value

    return data_dict

def extract_bird_data_for_date(folder_path, date, country_codes_path):
    """
    Parcourt tous les fichiers CSV dans un dossier, extrait les données pour une date donnée,
    et stocke les valeurs dans un dictionnaire avec les pays comme clés.
    Les valeurs sont des dictionnaires avec le numéro de l'oiseau comme clé et un tuple (count, count_with_rendements) comme valeur.

    :param folder_path: Chemin du dossier contenant les fichiers CSV.
    :param date: Date pour laquelle extraire les données (format identique à celui des fichiers CSV).
    :param country_codes_path: Chemin vers le fichier des codes pays.
    :return: Un dictionnaire avec les pays comme clés et des dictionnaires (numéro de l'oiseau: (count, count_with_rendements)).
    """
    # Charger les codes pays
    country_codes = pd.read_csv(country_codes_path, sep=";", index_col=0, header=0, names=["code", "country"])
    bird_data_dict = {}

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Vérifier que le fichier est un CSV
            file_path = os.path.join(folder_path, filename)

            # Charger le fichier CSV avec les en-têtes
            df = pd.read_csv(file_path, sep=";", decimal=",", header=0)

            # Vérifier si la date existe dans les données
            if date in df['year'].values:
                # Filtrer les données pour la date donnée
                filtered_data = df[df['year'] == date]

                # Ajouter les valeurs au dictionnaire
                for _, row in filtered_data.iterrows():
                    country_code = row['site']
                    country_name = country_codes.loc[country_code, 'country'] if country_code in country_codes.index else f"Unknown_{country_code}"
                    bird_number = filename.split('_')[1]  # Extraire le numéro de l'oiseau depuis le nom du fichier
                    if country_name not in bird_data_dict:
                        bird_data_dict[country_name] = {}  # Initialiser un dictionnaire pour chaque pays
                    # Ajouter une entrée (numéro de l'oiseau: (count, count_with_rendements))
                    bird_data_dict[country_name][bird_number] = (row['count'], row['count_rendements'])

    return bird_data_dict

# Exemple d'utilisation
folder_path = "Pressure_with_rendements"  # Chemin vers le dossier contenant les fichiers CSV
date_t = 2010  # Remplacez par la date souhaitée

pressure_for_date = extract_data_for_date_from_folder(folder_path, date_t)


bird_folder_path = "BirdCounts_with_rendements"  # Chemin vers le dossier contenant les fichiers CSV
country_codes_path = "country codes.csv"  # Chemin vers le fichier des codes pays
date_t = 2010  # Remplacez par la date souhaitée

bird_data_for_date = extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)


def correlation(country, bird, pressure):
    X = [extract_data_for_date_from_folder(folder_path, date_t)[country+"_rendements"][pressure] for date_t in range(1980, 2017)][:-1]
    Y = [extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)[country][bird][1] for date_t in range(1980, 2017)][1:]
    
    plt.scatter(X, Y)
    plt.xlabel(f"{pressure} (Pressure)")
    plt.ylabel(f"Bird rendements ({bird})")
    plt.title(f"Correlation between {pressure} and Bird rendements in {country}")
    plt.legend([f"{country} - {bird} vs {pressure}"])
    plt.show()
    

correlation("Spain", '2870', "Tem")
