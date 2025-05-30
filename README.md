# ai_birds_analysis
Voici un **README.md** clair et structuré pour présenter ton projet d’analyse de l’impact des pressions anthropiques sur les populations d’oiseaux à l’aide de méthodes de machine learning, y compris un MLP multiclasses.

---

### 📝 **README.md**

```markdown
# 🕊️ BirdPopML : Analyse des Pressions Anthropiques sur les Oiseaux

Ce projet explore l'impact des pressions anthropiques (urbanisation, température, agriculture...) sur les populations d’oiseaux européennes à l’aide de données EBCC (European Bird Census Council) et de techniques de machine learning.

---

## 📁 Structure du projet

```

.
├── BirdCounts\_with\_rendements/        # Données oiseaux par espèce (CSV)
├── Pressure\_with\_rendements/          # Données de pressions normalisées (CSV)
├── country codes.csv                  # Dictionnaire code pays → nom
├── mlp\_bird.py                        # Version MLP simple (régression)
├── mlp\_bird\_2.py                      # Version MLP multiclasses (hausse/stagnation/baisse)
├── clusters.py                        # Clustering des oiseaux selon leur réponse aux pressions
├── utils.py                           # Fonctions d’extraction et de prétraitement
└── README.md                          # Ce fichier

````

---

## 🧪 Objectifs

- Étudier la corrélation entre l’évolution des pressions anthropiques et les rendements des populations d’oiseaux.
- Prédire les dynamiques de population (hausse, stagnation, baisse) à l’aide d’un réseau de neurones multiclasses.
- Identifier les pressions les plus influentes selon les espèces à l’aide des poids du modèle.

---

## ⚙️ Lancement rapide

### 1. 📦 Dépendances

Installe les bibliothèques nécessaires :

```bash
pip install torch pandas scikit-learn matplotlib seaborn
````

### 2. ▶️ Lancer un MLP multiclasses

Pour entraîner le modèle sur les 10 espèces les plus fréquentes :

```bash
python mlp_bird_2.py
```

Ce script :

* Entraîne un modèle pour chaque espèce
* Affiche la matrice de confusion
* Affiche l’**importance directionnelle** des pressions sur chaque classe (via un heatmap)

---

## 🧠 Modèles utilisés

* **MLP (réseau de neurones feedforward)** avec classification en 3 classes :

  * 0 → Baisse
  * 1 → Stagnation
  * 2 → Hausse
* **Clustering KMeans** pour regrouper les espèces par réponse aux pressions (optionnel)
* Mesures d’évaluation :

  * Accuracy
  * F1-score macro
  * Matrice de confusion

---

## 📊 Résultats

* Le modèle détecte certaines corrélations faibles à modérées entre les pressions et les dynamiques d’espèces.
* Les poids du réseau permettent d’**interpréter les effets directionnels** de chaque pression.
* Les performances varient selon les espèces ; certaines réagissent mieux (ex : espèce 9760, Accuracy ≈ 70%).

---

## 📌 Auteurs & Contexte

Projet réalisé dans le cadre d’un travail universitaire (niveau Bac+3) — visant à croiser analyse écologique et méthodes d’intelligence artificielle.

---

## 📎 À faire

* [ ] Ajouter une interface simple pour choisir une espèce et afficher les résultats
* [ ] Ajouter un mode "prédiction interactive"
* [ ] Tester d'autres modèles : XGBoost, LSTM, régression logistique

```

---

Souhaites-tu une version `.md` téléchargeable directement ? Ou bien un deuxième README simplifié pour usage en présentation orale ?
```
