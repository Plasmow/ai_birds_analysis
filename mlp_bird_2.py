import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# 📥 Fonctions d’extraction
# ----------------------------------

def extract_bird_data_for_date(folder_path, date, country_codes_path):
    country_codes = pd.read_csv(country_codes_path, sep=";", index_col=0, names=["code", "country"])
    bird_data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=";", decimal=",")
            if date in df['year'].values:
                filtered_data = df[df['year'] == date]
                for _, row in filtered_data.iterrows():
                    country_code = row['site']
                    country_name = country_codes.loc[country_code, 'country'] if country_code in country_codes.index else f"Unknown_{country_code}"
                    bird_number = filename.split('_')[1]
                    if country_name not in bird_data_dict:
                        bird_data_dict[country_name] = {}
                    bird_data_dict[country_name][bird_number] = (row['count'], row['count_rendements'])
    return bird_data_dict

def extract_data_for_date_from_folder(folder_path, date):
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=";", decimal=",")
            df.set_index(df.columns[0], inplace=True)
            if date in df.index:
                row = df.loc[date]
                for col_name, value in row.items():
                    country = col_name
                    pressure_type = filename[:3].capitalize()
                    if country not in data_dict:
                        data_dict[country] = {}
                    data_dict[country][pressure_type] = value
    return data_dict

# ----------------------------------
# 📊 Dataset par espèce
# ----------------------------------

def build_dataset_for_species(bird_id, bird_folder, pressure_folder, country_codes_path, years, epsilon=0.05):
    X_all, Y_all = [], []

    for t in years:
        try:
            bird_data_t = extract_bird_data_for_date(bird_folder, t, country_codes_path)
            bird_data_tp1 = extract_bird_data_for_date(bird_folder, t + 1, country_codes_path)
            pressure_t = extract_data_for_date_from_folder(pressure_folder, t)
            pressure_tm1 = extract_data_for_date_from_folder(pressure_folder, t - 1)
        except:
            continue

        for country in pressure_t:
            if country in pressure_tm1 and country in bird_data_t and country in bird_data_tp1:
                if bird_id in bird_data_t[country] and bird_id in bird_data_tp1[country]:
                    pt = pressure_t[country]
                    ptm1 = pressure_tm1[country]
                    common = sorted(set(pt.keys()) & set(ptm1.keys()))
                    if len(common) < 4:
                        continue
                    diff = []
                    for p in common:
                        v1, v0 = pt.get(p, np.nan), ptm1.get(p, np.nan)
                        if np.isnan(v1) or np.isnan(v0):
                            break
                        diff.append(v1 - v0)
                    else:
                        ct = bird_data_t[country][bird_id][0]
                        ctp1 = bird_data_tp1[country][bird_id][0]
                        if ct > 0:
                            r = (ctp1 - ct) / ct
                            X_all.append(diff)
                            if r < -epsilon:
                                Y_all.append(0)  # baisse
                            elif r > epsilon:
                                Y_all.append(2)  # hausse
                            else:
                                Y_all.append(1)  # stagnation

    return np.array(X_all), np.array(Y_all),common

# ----------------------------------
# 🧠 Réseau de neurones multiclasses
# ----------------------------------

class MLPMultiClass(nn.Module):
    def __init__(self, input_dim, n_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_classifier(X, Y, epochs=200):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42, stratify=Y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

    model = MLPMultiClass(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = criterion(val_preds, Y_val_tensor).item()
            print(f"Epoch {epoch:03d} | Val Loss = {val_loss:.4f}")

    return model, X_val_tensor, Y_val_tensor

def evaluate_multiclass(model, X_val, Y_val):
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        preds = torch.argmax(logits, dim=1).numpy()
        y_true = Y_val.numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds)
    return acc, f1, cm

import matplotlib.pyplot as plt

def plot_predictions_multiclass(model, X_val_tensor, Y_val_tensor, title="Comparaison classes réelles vs prédites"):
    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
        true = Y_val_tensor.numpy()

    sorted_idx = np.argsort(true)  # trie par classes réelles
    true_sorted = true[sorted_idx]
    preds_sorted = preds[sorted_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(true_sorted, label="Classe réelle", marker='o', linestyle='--')
    plt.plot(preds_sorted, label="Classe prédite", marker='x', linestyle='-')
    plt.xlabel("Exemples triés")
    plt.ylabel("Classe (0=Baisse, 1=Stagne, 2=Hausse)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ----------------------------------
# 🔍 Scan auto des 10 espèces
# ----------------------------------

def scan_top_species(bird_folder, country_codes_path, years, max_species=10):
    all_species = Counter()
    for year in years:
        data = extract_bird_data_for_date(bird_folder, year, country_codes_path)
        for country in data:
            all_species.update(data[country].keys())
    return [s for s, _ in all_species.most_common(max_species)]

# ----------------------------------
# ✅ Lancement complet
# ----------------------------------
"""
if __name__ == "__main__":
    bird_csv_folder = "BirdCounts_with_rendements"
    pressure_csv_folder = "Pressure_with_rendements"
    country_codes_path = "country codes.csv"
    years = list(range(2000, 2014))
    epsilon = 0.5

    best = None
    print("🔍 Test des 10 espèces les plus fréquentes...\n")

    for bird_id in scan_top_species(bird_csv_folder, country_codes_path, years, max_species=10):
        print(f"\n🕊️ Espèce {bird_id}")
        X, Y ,pressure_names= build_dataset_for_species(bird_id, bird_csv_folder, pressure_csv_folder, country_codes_path, years, epsilon)

        if len(X) < 50 or len(set(Y)) < 2:
            print("⚠️ Pas assez de données ou classe unique. Skip.")
            continue

        try:
            model, X_val_tensor, Y_val_tensor = train_classifier(X, Y)
            acc, f1, cm = evaluate_multiclass(model, X_val_tensor, Y_val_tensor)
            plot_predictions_multiclass(model, X_val_tensor, Y_val_tensor)

            print(f"✅ Accuracy: {acc:.3f} | F1 macro: {f1:.3f}")
            print("Matrice de confusion :\n", cm)

            if best is None or f1 > best["f1"]:
                best = {"id": bird_id, "acc": acc, "f1": f1, "cm": cm}
        except Exception as e:
            print("❌ Erreur:", e)

    print("\n🏆 Meilleur résultat :")
    if best:
        print(f"Espèce {best['id']} | Accuracy = {best['acc']:.3f} | F1 = {best['f1']:.3f}")
        print("Matrice confusion finale :\n", best['cm'])
    else:
        print("Aucun modèle concluant.")

import matplotlib.pyplot as plt
import seaborn as sns

def plot_pressure_effects_by_class(model, pressure_names, class_labels=["Baisse", "Stagnation", "Hausse"]):
   # Affiche l'effet signé de chaque pression sur chaque classe (par propagation des poids).

    with torch.no_grad():
        # Récupération des matrices de poids des couches linéaires
        W1 = model.model[0].weight.numpy()  # (64, n_features)
        W2 = model.model[3].weight.numpy()  # (32, 64)
        W3 = model.model[6].weight.numpy()  # (3, 32)

        # Propagation linéaire des poids jusqu'à la sortie
        W = W3 @ W2 @ W1  # Résultat final : (3, n_features)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.heatmap(W, annot=True, xticklabels=pressure_names, yticklabels=class_labels, center=0, cmap="coolwarm")
    plt.title("Effet directionnel des pressions sur chaque classe (poids signés)")
    plt.tight_layout()
    plt.show()




plot_pressure_effects_by_class(model, pressure_names)

"""

###CLUSTERING
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def build_bird_response_matrix(bird_folder, pressure_folder, country_codes_path, date_tm1, date_t, date_tp1):
    """
    Construit une matrice (oiseau × pression) où chaque case est la corrélation entre pression et rendement.
    """
    bird_data_t = extract_bird_data_for_date(bird_folder, date_t, country_codes_path)
    bird_data_tp1 = extract_bird_data_for_date(bird_folder, date_tp1, country_codes_path)
    pressure_data_t = extract_data_for_date_from_folder(pressure_folder, date_t)
    pressure_data_tm1 = extract_data_for_date_from_folder(pressure_folder, date_tm1)

    bird_vectors = {}

    for country in pressure_data_t.keys():
        if country in pressure_data_tm1 and country in bird_data_t and country in bird_data_tp1:
            pressures_t = pressure_data_t[country]
            pressures_tm1 = pressure_data_tm1[country]

            if set(pressures_t.keys()) != set(pressures_tm1.keys()):
                continue

            pressure_diff = [pressures_t[p] - pressures_tm1[p] for p in sorted(pressures_t.keys())]

            birds_t = bird_data_t[country]
            birds_tp1 = bird_data_tp1[country]
            common_birds = set(birds_t.keys()) & set(birds_tp1.keys())

            for b in common_birds:
                count_t = birds_t[b][0]
                count_tp1 = birds_tp1[b][0]
                if count_t > 0:
                    rendement = (count_tp1 - count_t) / count_t
                    if b not in bird_vectors:
                        bird_vectors[b] = {'X': [], 'Y': []}
                    bird_vectors[b]['X'].append(pressure_diff)
                    bird_vectors[b]['Y'].append(rendement)

    # Calculer le vecteur de corrélation par oiseau
    bird_corrs = {}
    for b in bird_vectors:
        X = bird_vectors[b]['X']
        Y = bird_vectors[b]['Y']
        if len(Y) < 4:
            continue  # trop peu de données
        corr_vector = []
        X = np.array(X)
        for i in range(X.shape[1]):
            try:
                r, _ = pearsonr(X[:, i], Y)
            except:
                r = 0
            corr_vector.append(r)
        bird_corrs[b] = corr_vector

    return bird_corrs, sorted(pressures_t.keys())  # vecteurs + noms des pressions
def cluster_birds(bird_corrs, n_clusters=4, plot=True):
    bird_ids = list(bird_corrs.keys())
    X_corr = np.array([bird_corrs[b] for b in bird_ids])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_corr)

    bird_clusters = dict(zip(bird_ids, labels))

    if plot:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_corr)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50)
        plt.title("Clustering des oiseaux selon leur réponse aux pressions")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return bird_clusters, labels

date_tm1 = 2000
date_tm1 = 2001
date_t = 2002
date_tp1 = 2003

bird_csv_folder = "BirdCounts_with_rendements"
pressure_csv_folder = "Pressure_with_rendements"
country_codes_path = "country codes.csv"

# --- Clusterisation des oiseaux ---
bird_corrs, pressure_names = build_bird_response_matrix(
    bird_csv_folder,
    pressure_csv_folder,
    country_codes_path,
    date_tm1,
    date_t,
    date_tp1
)

bird_clusters, cluster_labels = cluster_birds(bird_corrs, n_clusters=4, plot=True)

# Exemple : afficher un résumé
from collections import Counter
print("Répartition des oiseaux par cluster :")
print(Counter(bird_clusters.values()))
