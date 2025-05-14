import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных
data_dev = pd.DataFrame({
    'task_time_hours': [10, 15, 12, 20, 18, 8, 14, 16, 22, 19],
    'bugs_count': [5, 3, 4, 2, 1, 6, 3, 2, 1, 2],
    'client_nps': [7, 8, 6, 9, 8, 5, 7, 8, 9, 8]
})

# Определение предикторов и целевой переменной
X = data_dev[['task_time_hours', 'bugs_count']]
y = data_dev['client_nps']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R^2): {r2:.2f}")

print("Коэффициенты модели:")
print(f"task_time_hours: {model.coef_[0]:.2f}")
print(f"bugs_count: {model.coef_[1]:.2f}")
print(f"Свободный член: {model.intercept_:.2f}")

# Визуализация многомерной линейной регрессии
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Точки данных
ax.scatter(data_dev['task_time_hours'], data_dev['bugs_count'], 
           data_dev['client_nps'], color='blue', label='Фактические данные')

# Создаем сетку для плоскости регрессии
x_surf, y_surf = np.meshgrid(
    np.linspace(data_dev['task_time_hours'].min(), data_dev['task_time_hours'].max(), 20),
    np.linspace(data_dev['bugs_count'].min(), data_dev['bugs_count'].max(), 20)
)

# Предсказанные значения плоскости
z_surf = (model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf)

# Отрисовка плоскости регрессии
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Плоскость регрессии')

# Подписи осей
ax.set_xlabel('Время на задачу (часы)')
ax.set_ylabel('Количество багов')
ax.set_zlabel('NPS клиента')
ax.set_title('Многомерная линейная регрессия')

plt.legend()
plt.tight_layout()

# Сохраняем фигуру в файл
plt.savefig('dev_multivariate_linear_regression_dev_3d.png')
