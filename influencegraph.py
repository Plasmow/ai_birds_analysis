import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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
    :return: Un dictionnaire avec les pays comme clés et comme valeurs des dictionnaires (numéro de l'oiseau: (count, count_with_rendements)).
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
date_t = 1980 

pressure_for_date = extract_data_for_date_from_folder(folder_path, date_t)


bird_folder_path = "BirdCounts_with_rendements"  # Chemin vers le dossier contenant les fichiers CSV
country_codes_path = "country codes.csv"  # Chemin vers le fichier des codes pays


bird_data_for_date = extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)




def correlation(country, bird, pressure):
    #plot correlation between bird rendements and pressure rendements earlier in time for a specific country and bird
    X=[]
    Y=[]
    for date_t in range(1980,2017):
        value=extract_bird_data_for_date(bird_folder_path,date_t,country_codes_path)[country][bird][1] #replace "Germany East" by country
        if not np.isnan(value) and value!=0:
            X.append(extract_data_for_date_from_folder(folder_path, date_t)[country+"Germany_rendements"][pressure]) #replace "Germany" by country
            Y.append(value)
    X=X[:-1]
    Y=Y[1:]

    plt.scatter(X, Y, label= pressure )

#plot 
'''
correlation("Poland","10190","Hic")
plt.xlabel(f"(Pressure)")
plt.ylabel(f"Bird rendements")
plt.legend(loc="upper right", fontsize="small")
plt.title(f"Correlation between HIC pressure rendement and Bird 10190 rendement in Poland")
plt.show()
'''
country="Germany East"
bird='5190'

def correlation_all(country,bird):
    for pressure in ["Far","For","Hic","Nat","Pla","Tem","Uaa","Urb"]:
        correlation(country,bird, pressure)
    plt.xlabel(f"(Pressure)")
    plt.ylabel(f"Bird rendements")
    plt.legend(loc="upper right", fontsize="small")
    plt.title(f"Correlation between pressure rendements and Bird rendements in {country}")
    plt.show()

correlation_all(country,bird)

def correlation_coefficient(country, bird, pressure):
    #compute correlation coeff between bird rendements and pressure rendements earlier in time for a specific country and bird
    X=[]
    Y=[]
    for date_t in range(1980,2017):
        if bird in extract_bird_data_for_date(bird_folder_path,date_t,country_codes_path)[country]:
            value=extract_bird_data_for_date(bird_folder_path,date_t,country_codes_path)[country][bird][1]
            if not np.isnan(value) and value!=0:
                X.append(extract_data_for_date_from_folder(folder_path, date_t)[country+"_rendements"][pressure]) #replace "Germany" by country
                Y.append(value)

    X=X[:-1]
    Y=Y[1:]
    if len(X)<2 or len(Y)<2:
        r=-1
        p_value=1
    else:
        r, p_value = pearsonr(X, Y)
        if np.isnan(p_value):
                p_value=1
    
    return(p_value)
    
    """"
    print(f"Coefficient de corrélation de Pearson : {r:.3f}")
    print(f"Valeur p associée : {p_value:.3e}")
    print(p_value<0.05)
    """
#correlation_coefficient(country,bird,'Tem')


pressures=["Far","For","Hic","Nat","Pla","Tem","Uaa","Urb"]
def find_correlation(country,folder_path="BirdCounts_with_rendements"):
    bird_numbers = []
    p_values={}
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("BIRD_") and filename.endswith(".csv"):
            # Extract the bird number from the filename
            bird_number = filename.split('_')[1].split('.')[0]
            bird_numbers.append(bird_number)
    np.random.shuffle(bird_numbers)
    for bird_number in bird_numbers:
            pressure_dict={}
            for pressure in pressures:
                p_value = correlation_coefficient(country,bird_number,pressure)
                if p_value<0.05:
                    return("country"+str(country)+"bird_number:"+str(bird_number))
                else:
                    pressure_dict[pressure]=p_value
            p_values[bird_number]=pressure_dict
    print(bird_numbers)

#find_correlation(country)

def population(country,bird):
    Y=[extract_bird_data_for_date(bird_folder_path,date_t,country_codes_path)[country][bird][0] for date_t in range(1980,2017)]
    X=[i for i in range(1980,2017)]
    
    
            
    plt.plot(X, Y,marker='o')
    plt.xlabel(f"time(year)")
    plt.ylabel(f"Bird count ({bird})")
    plt.title(f"Evolution of birds in time")
    plt.show()

#population("Germany East","5190")

def population_all_countries(bird):
    """
    Superpose the evolution of a bird's population in all countries on a single graph.
    
    :param bird: Bird number to analyze.
    """
    X = [i for i in range(1980, 2017)]
    plt.figure(figsize=(10, 6))
    
    for country in extract_bird_data_for_date(bird_folder_path, 1980, country_codes_path).keys():
        # Correctly retrieve the population (count) instead of rendements
        Y = [extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)
             .get(country, {})
             .get(bird, (0,))[0]  # Use index [0] to get 'count' (population)
             for date_t in range(1980, 2017)]
        plt.plot(X, Y, label=country)
    
    plt.xlabel("Time (year)")
    plt.ylabel(f"Bird count ({bird})")
    plt.title(f"Evolution of bird population ({bird}) across countries")
    plt.legend(loc="upper right", fontsize="small")
    plt.show()

# Example usage:
#population_all_countries('10190')

import random

def get_random_bird_number(folder_path):
    """
    Returns a random bird number from the files in the specified folder.

    :param folder_path: Path to the folder containing bird count files.
    :return: A random bird number as a string.
    """
    bird_numbers = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith("BIRD_") and filename.endswith(".csv"):
            # Extract the bird number from the filename
            bird_number = filename.split('_')[1].split('.')[0]
            bird_numbers.append(bird_number)

    # Return a random bird number
    if bird_numbers:
        return random.choice(bird_numbers)
    else:
        raise ValueError("No bird files found in the specified folder.")

# Example usage
random_bird = get_random_bird_number("BirdCounts_with_rendements")


#population_all_countries('10090')


def plot_test():
    Y = [extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)[country][bird][0] for date_t in range(1980, 2017)]
    X = [i for i in range(1980, 2017)]
    
    plt.plot(X, Y, marker='o')  # Use plt.plot to connect points with lines and add markers
    plt.xlabel(f"time(year)")
    plt.ylabel(f"Bird count ({bird})")
    plt.title(f"Evolution of birds in time")
    plt.show()

bird_folder_path = "BirdCounts_with_rendements"  
date_t = 2016
bird='5190'
country="Lithuania"

#plot_test()

country_names = [
    "Austria", "Belgium-Brussels", "Belgium-Wallonia", "Bulgaria", "Cyprus",
    "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany East",
    "Germany West", "Greece", "Hungary", "Italy", "Latvia", "Lithuania",
    "Luxembourg", "Netherlands", "Norway", "Poland", "Portugal",
    "Republic of Ireland", "Romania", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "United Kingdom"
]

def sum_pop(bird):
    Y = [
        sum([
            extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path)
            .get(country, {})
            .get(bird, (0,))[0]  # Get population (count) or 0 if bird is not present
            for country in country_names
            if bird in extract_bird_data_for_date(bird_folder_path, date_t, country_codes_path).get(country, {})
        ])
        for date_t in range(1980, 2017)
    ]
    X = [i for i in range(1980, 2017)]
    
    plt.plot(X, Y, marker='o')  # Use plt.plot to connect points with lines and add markers
    plt.xlabel(f"time(year)")
    plt.ylabel(f"Bird count ({bird})")
    plt.title(f"Evolution of bird {bird} in time")
    plt.show()

#sum_pop(bird)
