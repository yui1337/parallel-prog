import numpy as np
import subprocess
import os
import json
import matplotlib.pyplot as plt

def load_config(config_path="config.json"):
    """Загружает настройки из JSON файла."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден.")
    with open(config_path, "r") as f:
        return json.load(f)

def setup_environment(dirs_dict):
    """Создает необходимые директории."""
    for path in dirs_dict.values():
        os.makedirs(path, exist_ok=True)

def generate_matrix_files(n, path_a, path_b):
    """Генерирует случайные матрицы и сохраняет их в файлы."""
    a = np.random.randint(0, 100, size=(n, n), dtype=np.int64)
    b = np.random.randint(0, 100, size=(n, n), dtype=np.int64)
    np.savetxt(path_a, a, fmt='%d')
    np.savetxt(path_b, b, fmt='%d')
    return a, b

def verify_correctness(path_res, a, b):
    """Проверяет результат C++ с помощью NumPy."""
    c_cpp = np.loadtxt(path_res, dtype=np.int64)
    c_py = np.dot(a, b)
    return np.array_equal(c_cpp, c_py)

def run_cpp_benchmark(n, paths, exe_path):
    """Запускает C++ программу и возвращает время выполнения."""
    try:
        result = subprocess.run(
            [exe_path, str(n), paths['a'], paths['b'], paths['res']],
            capture_output=True, 
            text=True, 
            check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении C++ программы: {e.stderr}")
        return None

def plot_results(stats, save_path):
    """Строит график производительности C++."""
    if not stats:
        return
    
    sizes, times = zip(*stats)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', color='tab:blue', linewidth=2)
    plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    print(f"\nГрафик сохранен в файл: {save_path}")
    plt.show()

def main():
    try:
        config = load_config()
        setup_environment(config['directories'])
        
        if not os.path.exists(config['cpp_exe']):
            print(f"Ошибка: Исполняемый файл {config['cpp_exe']} не найден.")
            return

        stats = []

        for n in config['sizes']:
            print(f"--- Тест: {n}x{n} ---")
            
            paths = {
                'a': os.path.join(config['directories']['matrix_a'], f"A_{n}.txt"),
                'b': os.path.join(config['directories']['matrix_b'], f"B_{n}.txt"),
                'res': os.path.join(config['directories']['results'], f"C_{n}.txt")
            }

            matrix_a, matrix_b = generate_matrix_files(n, paths['a'], paths['b'])

            cpp_time = run_cpp_benchmark(n, paths, config['cpp_exe'])
            
            if cpp_time is not None:
                if verify_correctness(paths['res'], matrix_a, matrix_b):
                    print(f"Результат корректен. Время выполнения: {cpp_time} мс")
                    stats.append((n, cpp_time))
                else:
                    print(f"Результат: ОШИБКА в вычислениях для N = {n}")
                    break

        plot_results(stats, config['plot_filename'])

    except Exception as e:
        print(f"Произошла ошибка: {e.what()}")

if __name__ == "__main__":
    main()