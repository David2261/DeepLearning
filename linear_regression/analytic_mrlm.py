import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных
data_analytics = pd.DataFrame({
    'analysis_time_hours': [30, 40, 35, 50, 45, 25, 38, 42, 48, 36],
    'prediction_accuracy': [85, 90, 88, 92, 87, 80, 89, 91, 93, 86],
    'revenue_thousands': [500, 600, 550, 700, 650, 450, 580, 620, 680, 560]
})

# Определение предикторов и целевой переменной
X = data_analytics[['analysis_time_hours', 'prediction_accuracy']]
y = data_analytics['revenue_thousands']

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
print(f"analysis_time_hours: {model.coef_[0]:.2f}")
print(f"prediction_accuracy: {model.coef_[1]:.2f}")
print(f"Свободный член: {model.intercept_:.2f}")

# Визуализация многомерной линейной регрессии
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Точки данных
ax.scatter(data_analytics['analysis_time_hours'], data_analytics['prediction_accuracy'], 
           data_analytics['revenue_thousands'], color='blue', label='Фактические данные')

# Создаем сетку для плоскости регрессии
x_surf, y_surf = np.meshgrid(
    np.linspace(data_analytics['analysis_time_hours'].min(), data_analytics['analysis_time_hours'].max(), 20),
    np.linspace(data_analytics['prediction_accuracy'].min(), data_analytics['prediction_accuracy'].max(), 20)
)

# Предсказанные значения плоскости
z_surf = (model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf)

# Отрисовка плоскости регрессии
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, label='Плоскость регрессии')

# Подписи осей
ax.set_xlabel('Время на анализ (часы)')
ax.set_ylabel('Точность прогнозов (%)')
ax.set_zlabel('Выручка (тыс. долларов)')
ax.set_title('Многомерная линейная регрессия')

plt.legend()
plt.tight_layout()

# Сохраняем фигуру в файл
plt.savefig('multivariate_linear_regression_analytics_3d.png')

plt.show()
