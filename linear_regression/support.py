import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Генерация синтетических данных
data_support = pd.DataFrame({
	'response_time_min': [5, 10, 7, 15, 12, 8, 6, 9, 11, 14],
	'resolved_tickets': [20, 15, 18, 10, 12, 17, 19, 16, 14, 11],
	'user_satisfaction': [4.5, 3.8, 4.0, 3.5, 3.7, 4.2, 4.3, 4.1, 3.9, 3.6]
})

# Расчет корреляционной матрицы (Пирсон)
corr_matrix_support = data_support.corr(method='pearson')

# Вывод корреляционной матрицы
print("Корреляционная матрица (Поддержка):")
print(corr_matrix_support)

# Визуализация корреляционной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_support, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Корреляционная матрица (Поддержка)')
# Сохранение в файл
try:
    plt.savefig('./correlation_matrix_support.png')  # Замените на ваш путь
except Exception as e:
    print(f"Ошибка при сохранении графика: {e}")

plt.close()  # Закрытие фигуры


# Диаграмма рассеяния для пары переменных (response_time_min vs user_satisfaction)
plt.figure(figsize=(8, 6))
plt.scatter(data_support['response_time_min'], data_support['user_satisfaction'])
plt.xlabel('Время ответа (минуты)')
plt.ylabel('Удовлетворенность пользователей')
plt.title('Диаграмма рассеяния: Время ответа vs Удовлетворенность')
plt.grid(True)
try:
    plt.savefig('./scatter_plot_support.png')  # Замените на ваш путь
except Exception as e:
    print(f"Ошибка при сохранении графика: {e}")

plt.close()  # Закрытие фигуры
