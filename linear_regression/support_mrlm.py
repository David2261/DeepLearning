import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных
data_support = pd.DataFrame({
    'response_time_min': [5, 10, 7, 15, 12, 8, 6, 9, 11, 14],
    'resolved_tickets': [20, 15, 18, 10, 12, 17, 19, 16, 14, 11],
    'user_satisfaction': [4.5, 3.8, 4.0, 3.5, 3.7, 4.2, 4.3, 4.1, 3.9, 3.6]
})

# Определение предикторов и целевой переменной
X = data_support[['response_time_min', 'resolved_tickets']]
y = data_support['user_satisfaction']

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
print(f"response_time_min: {model.coef_[0]:.2f}")
print(f"resolved_tickets: {model.coef_[1]:.2f}")
print(f"Свободный член: {model.intercept_:.2f}")

# Визуализация многомерной линейной регрессии
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Точки данных
ax.scatter(data_support['response_time_min'], data_support['resolved_tickets'], 
           data_support['user_satisfaction'], color='blue', label='Фактические данные')

# Создаем сетку для плоскости регрессии
x_surf, y_surf = np.meshgrid(
    np.linspace(data_support['response_time_min'].min(), data_support['response_time_min'].max(), 20),
    np.linspace(data_support['resolved_tickets'].min(), data_support['resolved_tickets'].max(), 20)
)

# Предсказанные значения плоскости
z_surf = (model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf)

# Отрисовка плоскости регрессии
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Плоскость регрессии')

# Подписи осей
ax.set_xlabel('Время ответа (минуты)')
ax.set_ylabel('Решенные тикеты')
ax.set_zlabel('Удовлетворенность пользователей')
ax.set_title('Многомерная линейная регрессия')

plt.legend()
plt.tight_layout()

# Сохраняем фигуру в файл
plt.savefig('support_multivariate_linear_regression_3d.png')

plt.close()

