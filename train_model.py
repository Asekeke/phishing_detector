import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Загружаем датасет
df = pd.read_csv("phishing_emails.csv")

# Переименовываем колонки
df = df.rename(columns={"EmailText": "text", "Label": "label"})

# Проверяем, есть ли нужные столбцы
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Ошибка: в датасете должны быть столбцы 'text' и 'label'.")

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Преобразуем текст в векторы
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучаем модель
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Сохраняем модель и векторизатор
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Модель обучена и сохранена!")
