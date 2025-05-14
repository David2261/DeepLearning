import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Генерация синтетических данных
data_analytics = pd.DataFrame({
    'analysis_time_hours': [30, 40, 35, 50, 45, 25, 38, 42, 48, 36],
    'prediction_accuracy': [85, 90, 88, 92, 87, 80, 89, 91, 93, 86],
    'revenue_thousands': [500, 600, 550, 700, 650, 450, 580, 620, 680, 560]
})

# Расчет корреляционной матрицы (Пирсон)
corr_matrix_analytics = data_analytics.corr(method='pearson')

# Вывод корреляционной матрицы
print("Корреляционная матрица (Аналитика):")
print(corr_matrix_analytics)

# Визуализация корреляционной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_analytics, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляционная матрица (Аналитика)')
plt.savefig('./correlation_matrix.png')  # Сохранение в файл
plt.close()  # Закрытие фигуры

# Диаграмма рассеяния для пары переменных (analysis_time_hours vs prediction_accuracy)
plt.figure(figsize=(8, 6))
plt.scatter(data_analytics['analysis_time_hours'], data_analytics['prediction_accuracy'])
plt.xlabel('Время на анализ (часы)')
plt.ylabel('Точность прогнозов (%)')
plt.title('Диаграмма рассеяния: Время анализа vs Точность прогнозов')
plt.grid(True)
plt.savefig('./scatter_plot.png')  # Сохранение в файл
plt.close()  # Закрытие фигуры
