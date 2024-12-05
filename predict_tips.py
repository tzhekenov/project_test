import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

# Пример данных: суммы заказов и соответствующие чаевые с добавлением случайного шума
np.random.seed(42)  # Для воспроизводимости результатов
order_amounts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
tips = np.array([1, 3, 3, 4, 5, 6, 7, 8, 9, 10])

# Добавление случайного шума
order_amounts = order_amounts + np.random.normal(0, 5, order_amounts.shape)
tips = tips + np.random.normal(0, 0.5, tips.shape)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(order_amounts, tips)

# Прогнозирование чаевых для диапазона сумм заказов
predicted_tips = model.predict(order_amounts)

# Построение графика данных и линии регрессии
plt.scatter(order_amounts, tips, color='blue', label='Фактические чаевые')
plt.plot(order_amounts, predicted_tips, color='red', label='Прогнозируемые чаевые')
plt.xlabel('Сумма заказа')
plt.ylabel('Сумма чаевых')
plt.title('Прогнозирование чаевых на основе суммы заказа')
plt.legend()
plt.show()

# Функция для прогнозирования чаевых на основе суммы заказа
def predict_tip(order_amount):
    return model.predict(np.array([[order_amount]]))[0]

# Пример использования
order_amount = 150  # Стоимость заказа
predicted_tip = predict_tip(order_amount)
print(f'Прогнозируемые чаевые для суммы заказа {order_amount} составляют {predicted_tip:.2f}')
# Заголовок приложения
st.title('Прогнозирование чаевых на основе суммы заказа')

# Поле ввода для суммы заказа
order_amount = st.number_input('Введите сумму заказа', min_value=0, max_value=200, value=75)

# Прогнозирование чаевых
predicted_tip = predict_tip(order_amount)

# Отображение прогнозируемых чаевых
st.write(f'Прогнозируемые чаевые для суммы заказа {order_amount} составляют {predicted_tip:.2f}')

# Построение графика данных и линии регрессии
fig, ax = plt.subplots()
ax.scatter(order_amounts, tips, color='blue', label='Фактические чаевые')
ax.plot(order_amounts, predicted_tips, color='red', label='Прогнозируемые чаевые')
ax.set_xlabel('Сумма заказа')
ax.set_ylabel('Сумма чаевых')
ax.set_title('Прогнозирование чаевых на основе суммы заказа')
ax.legend()

# Отображение графика в Streamlit
st.pyplot(fig)