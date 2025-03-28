import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Генерация сложного многомерного набора данных
def generate_complex_data(n_samples=2000, noise=0.2):
    np.random.seed(42)
    X1 = np.random.rand(n_samples) * 2 - 1  # Диапазон от -1 до 1
    X2 = np.random.rand(n_samples) * 2 - 1  # Диапазон от -1 до 1
    
    # Сложное нелинейное правило для классов
    Y = (np.sin(3 * np.pi * X1) + np.cos(3 * np.pi * X2) + noise * np.random.randn(n_samples)) > 0
    Y = Y.astype(int)

    X = np.column_stack((X1, X2))
    return X, Y

# Создание данных
X, Y = generate_complex_data(3000, noise=0.3)

# Разделение данных
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Оптимизация гиперпараметров SVM с помощью GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # Регуляризация
    'gamma': [0.001, 0.01, 0.1, 1, 10],  # Параметр для RBF-ядра
    'kernel': ['rbf']  # Нелинейное ядро
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Лучшие параметры
best_svm = grid_search.best_estimator_
print("Лучшие параметры SVM:", grid_search.best_params_)

# Оценка модели на тестовых данных
Y_pred = best_svm.predict(X_test)

# Вывод матрицы ошибок
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# Вывод отчета о классификации
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# Визуализация границы принятия решений
def plot_decision_boundary(model, X, Y):
    h = 0.02  # Шаг сетки
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("SVM Decision Boundary")
    plt.show()

plot_decision_boundary(best_svm, X_test, Y_test)
