from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Загружаем модель и векторизатор
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Преобразуем текст в вектор
        email_vector = vectorizer.transform([email_text])

        # Предсказываем (1 - фишинг, 0 - безопасно)
        prediction = model.predict(email_vector)[0]
        result = "Phishing!" if prediction == 1 else "Safe"

        return render_template("index.html", result=result, email_text=email_text)

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
