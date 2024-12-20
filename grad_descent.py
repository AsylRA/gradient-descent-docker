import numpy as np


def gradient_descent(
    func, grad_func, start_point, learning_rate=0.01, max_iter=1000, tol=1e-6
):
    """
    Алгоритм градиентного спуска для минимизации функции.
    """
    point = np.array(start_point)
    for i in range(max_iter):
        grad = np.array(grad_func(point))
        new_point = point - learning_rate * grad
        if np.linalg.norm(new_point - point) < tol:
            return new_point, func(new_point), i
        point = new_point
    return point, func(point), max_iter


# Тестовая функция для проверки
if __name__ == "__main__":
    # Целевая функция (например, квадратичная функция)
    def test_func(x):
        return x[0] ** 2 + x[1] ** 2

    # Градиент целевой функции
    def grad_test_func(x):
        return np.array([2 * x[0], 2 * x[1]])

    # Начальная точка
    start = [1.0, 1.0]

    # Вызываем градиентный спуск
    result, value, iterations = gradient_descent(test_func, grad_test_func, start)

    # Выводим результаты
    print(
        f"Минимум найден в точке: {result}, значение функции: {value}, итераций: {iterations}"
    )
