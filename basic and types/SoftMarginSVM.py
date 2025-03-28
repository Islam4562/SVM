from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Генерируем данные с пересечением классов
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=0.5, random_state=42)

# Обучаем SVM с мягким разделением
model = SVC(kernel='linear', C=0.1)
model.fit(X, y)

# Визуализация
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Soft Margin SVM (C=0.1)")
plt.show()
