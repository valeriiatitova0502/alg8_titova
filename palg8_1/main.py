import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, RK23
import pandas as pd

# Задаем уравнение системы дифференциальных уравнений
def f(t, y):
    return [y[1], np.clip(-4 * np.sin(y[0]), -5, 5)]

# Функция для использования встроенного метода RK23 для решения RK3
def solve_adams_rk23(t0, tk, dt, f):
    t_values = np.arange(t0, tk + dt, dt)
    solver = RK23(fun=f, t0=t_values[0], y0=x0, t_bound=tk, max_step=dt)
    t_adams, x_adams = [], []

    while solver.status == 'running':
        solver.step()
        t_adams.append(solver.t)
        x_adams.append(solver.y[0])

    return np.array(t_adams), np.array(x_adams)

# Функция для использования odeint для решения RK3
def solve_rk3_odeint(t0, tk, dt, f):
    t_values = np.arange(t0, tk + dt, dt)
    solution = odeint(func=f, y0=x0, t=t_values, tfirst=True)
    return t_values, solution

# Начальные условия
t0 = 0
x0 = np.array([0.0, 1.0])  # x0 теперь массив
tk = 4
dt = 0.2  # Уменьшенный dt для более гладких графиков

# Применяем метод Адамса с использованием RK23 для решения
t_adams, x_adams = solve_adams_rk23(t0, tk, dt, f)

# Выводим новые результаты в консоль
print("\nТаблица результатов метода Адамса с RK23:")
print(pd.DataFrame({
    't': t_adams,
    'x(t)': x_adams,
}))

# Применяем odeint для метода РК3
t_rk3_odeint, x_rk3_odeint = solve_rk3_odeint(t0, tk, dt, f)

# Выводим результаты метода РК3 с использованием odeint в консоль
print("\nТаблица результатов метода РК3 с использованием odeint:")
print(pd.DataFrame({
    't': t_rk3_odeint,
    'x(t)': x_rk3_odeint[:, 0],
    "x'(t)": x_rk3_odeint[:, 1],
}))

# Приведение массивов к одной длине
x_adams_interp = np.interp(t_rk3_odeint, t_adams, x_adams)

# Графики для x(t) и x'(t) (Adams и RK3)
plt.figure(figsize=(12, 12))
plt.plot(t_rk3_odeint, x_adams_interp, label='x(t) (Adams3 RK23)', linewidth=2)
plt.plot(t_rk3_odeint, x_rk3_odeint[:, 0], label='x(t) (RK3 odeint)', linestyle='--')
plt.title('График x(t) метода Адамса с RK23 и метода РК3 с использованием odeint')
plt.xlabel('t')
plt.ylabel('Значения')
plt.legend()
plt.show()
