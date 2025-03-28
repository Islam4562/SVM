import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Генерируем данные
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=X.shape[0])

# Обучаем SVR
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X, y)

# Предсказание
y_pred = model.predict(X)

# Визуализация
plt.scatter(X, y, label="Исходные данные")
plt.plot(X, y_pred, color='red', label="SVR-предсказание")
plt.legend()
plt.show()
