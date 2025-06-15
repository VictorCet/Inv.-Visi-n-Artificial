import joblib

# Cargar modelo y vectorizador
model = joblib.load("modelo_sentimientos.pkl")
vectorizer = joblib.load("vectorizador.pkl")

print("Análisis de Sentimientos (escribe 'salir' para terminar):")

while True:
    frase = input("> ")
    if frase.lower() == "salir":
        break
    X = vectorizer.transform([frase])
    pred = model.predict(X)[0]
    sentimiento = "Positivo" if pred == 1 else "Negativo"
    print(f"'{frase}' => {sentimiento}\n")
