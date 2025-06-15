
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Cargar el dataset real
dataset = load_dataset("sst2")

# 2. Separar en entrenamiento y prueba
train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]
test_texts = dataset["validation"]["sentence"]
test_labels = dataset["validation"]["label"]

# 3. Vectorizar los textos
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 4. Entrenar el modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 5. Evaluar
y_pred = model.predict(X_test)
print("📊 REPORTE DE CLASIFICACIÓN:")
print(classification_report(test_labels, y_pred))

# 6. Guardar modelo y vectorizador
joblib.dump(model, "modelo_sentimientos.pkl")
joblib.dump(vectorizer, "vectorizador.pkl")

print("✅ Modelo y vectorizador guardados con éxito.")
