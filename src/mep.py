import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import sys

# Charger le modèle MLflow
model_uri = "runs:/4f5116206ecc48b1bec74ec7501d30a4/iris_logistic_regression_model"  # Remplacer <RUN_ID> par l'ID de votre exécution MLflow
model = mlflow.sklearn.load_model(model_uri)

# Charger les données de test
iris = load_iris()
X_test = iris.data
y_test = iris.target

# Effectuer des prédictions avec le modèle
predictions = model.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
