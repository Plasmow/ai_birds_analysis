
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Jeu de données fictif
X = [[100, 3, 10, 5], [150, 4, 5, 3], [80, 2, 20, 8], [200, 5, 2, 2]]
y = [300000, 450000, 200000, 600000]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Application de la régression Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Prédiction
y_pred = lasso.predict(X_test)

print("Coefficients de régression Lasso :", lasso.coef_)
































