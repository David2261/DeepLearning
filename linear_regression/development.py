import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Генерация синтетических данных
data_dev = pd.DataFrame({
    'task_time_hours': [10, 15, 12, 20, 18, 8, 14, 16, 22, 19],
    'bugs_count': [5, 3, 4, 2, 1, 6, 3, 2, 1, 2],
    'client_nps': [7, 8, 6, 9, 8, 5, 7, 8, 9, 8]
})

# Расчет корреляционной матрицы (Пирсон)
corr_matrix_dev = data_dev.corr(method='pearson')

# Вывод корреляционной матрицы
print("Корреляционная матрица (Разработка):")
print(corr_matrix_dev)

# Визуализация корреляционной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_dev, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляционная матрица (Разработка)')

# Сохранение в файл
try:
    plt.savefig('./correlation_matrix_dev.png')  # Замените на ваш путь
except Exception as e:
    print(f"Ошибка при сохранении графика: {e}")

plt.close()  # Закрытие фигуры

# Диаграмма рассеяния для пары переменных (task_time_hours vs bugs_count)
plt.figure(figsize=(8, 6))
plt.scatter(data_dev['task_time_hours'], data_dev['bugs_count'])
plt.xlabel('Время на задачу (часы)')
plt.ylabel('Количество багов')
plt.title('Диаграмма рассеяния: Время на задачу vs Количество багов')
plt.grid(True)

# Сохранение в файл
try:
    plt.savefig('./scatter_plot_dev.png')  # Замените на ваш путь
except Exception as e:
    print(f"Ошибка при сохранении графика: {e}")

plt.close()  # Закрытие фигуры
