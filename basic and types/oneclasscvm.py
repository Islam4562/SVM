import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# Генерируем нормальные данные
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Добавляем выбросы
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# Обучаем One-Class SVM
model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.1)
model.fit(X_train)

# Предсказание (1 – нормальные, -1 – аномалии)
y_pred_train = model.predict(X_train)
y_pred_outliers = model.predict(X_outliers)

# Визуализация
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label="Нормальные данные")
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', label="Аномалии", marker='x')
plt.legend()
plt.title("One-Class SVM (Обнаружение аномалий)")
plt.show()
