import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Определение функции, представляющей систему уравнений
def system(t, y):
    x, x_prime = y
    x_double_prime = -4 * np.sin(x)
    return [x_prime, x_double_prime]

# Начальные условия
t0 = 0
y0 = [0, 1]

# Задание интервала времени с более мелким разбиением
t_span = (t0, 10)

# Решение системы уравнений методом Рунге-Кутта третьего порядка
sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=np.linspace(t0, 10, 1000))

# Вывод результатов в виде таблицы в консоль
table_header = ['t', 'x(t)', "x'(t)"]
table_data = np.column_stack((sol.t, sol.y[0], sol.y[1]))

print("{:<8} {:<12} {:<12}".format(*table_header))
print("-" * 32)
for row in table_data:
    print("{:<8} {:<12.6f} {:<12.6f}".format(*row))

# Построение графика в полярной системе координат
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Преобразование в полярные координаты
theta = np.concatenate((sol.y[0], sol.y[0] + np.pi))  # Добавляем симметричную кривую
r = np.concatenate((0.5 * np.abs(sol.y[1]), 0.5 * np.abs(sol.y[1])))  # Устанавливаем радиус 0.5

# Построение графика
ax.plot(theta, r, label='Свой')
ax.plot(theta, r, label='Встроенный', linestyle="dashdot", linewidth=1)  # Изменение типа линии на пунктирную
# Добавление легенды
ax.legend()
plt.title('График x(t) своего и встроенного метода RK3')

# Отображение графика
plt.show()
