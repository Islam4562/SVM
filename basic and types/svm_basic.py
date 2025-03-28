from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Загружаем датасет
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем линейный SVM
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
