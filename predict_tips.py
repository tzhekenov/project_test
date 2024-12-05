import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

# Пример данных: суммы заказов, время суток и соответствующие чаевые с добавлением случайного шума
np.random.seed(42)  # Для воспроизводимости результатов
order_amounts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
times_of_day = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]).reshape(-1, 1)  # 0 - утро, 1 - день, 2 - вечер, 3 - ночь
tips = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Добавление случайного шума
order_amounts = order_amounts + np.random.normal(0, 5, order_amounts.shape)
tips = tips + np.random.normal(0, 0.5, tips.shape)

# Объединение признаков в одну матрицу
features = np.hstack((order_amounts, times_of_day))

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(features, tips)

# Прогнозирование чаевых для диапазона сумм заказов и времени суток
predicted_tips = model.predict(features)

# Построение графика данных и линии регрессии
plt.scatter(order_amounts, tips, color='blue', label='Фактические чаевые')
plt.plot(order_amounts, predicted_tips, color='red', label='Прогнозируемые чаевые')
plt.xlabel('Сумма заказа')
plt.ylabel('Сумма чаевых')
plt.title('Прогнозирование чаевых на основе суммы заказа и времени суток')
plt.legend()
plt.show()

# Функция для прогнозирования чаевых на основе суммы заказа и времени суток
def predict_tip(order_amount, time_of_day):
    return model.predict(np.array([[order_amount, time_of_day]]))[0]

# Пример использования
order_amount = 75  # Стоимость заказа
time_of_day = 1  # Время суток (1 - день)
predicted_tip = predict_tip(order_amount, time_of_day)
print(f'Прогнозируемые чаевые для суммы заказа {order_amount} и времени суток {time_of_day} составляют {predicted_tip:.2f}')

# Заголовок приложения
st.title('Прогнозирование чаевых на основе суммы заказа и времени суток')

# Поле ввода для суммы заказа
order_amount = st.number_input('Введите сумму заказа', min_value=0, max_value=200, value=75)

# Поле ввода для времени суток
time_of_day = st.selectbox('Выберите время суток', options=[0, 1, 2, 3], format_func=lambda x: ['Утро', 'День', 'Вечер', 'Ночь'][x])

# Прогнозирование чаевых
predicted_tip = predict_tip(order_amount, time_of_day)

# Отображение прогнозируемых чаевых
st.write(f'Прогнозируемые чаевые для суммы заказа {order_amount} и времени суток {["Утро", "День", "Вечер", "Ночь"][time_of_day]} составляют {predicted_tip:.2f}')

# Построение графика данных и линии регрессии
fig, ax = plt.subplots()
ax.scatter(order_amounts, tips, color='blue', label='Фактические чаевые')
ax.plot(order_amounts, predicted_tips, color='red', label='Прогнозируемые чаевые')
ax.set_xlabel('Сумма заказа')
ax.set_ylabel('Сумма чаевых')
ax.set_title('Прогнозирование чаевых на основе суммы заказа и времени суток')
ax.legend()

# Отображение графика в Streamlit
st.pyplot(fig)