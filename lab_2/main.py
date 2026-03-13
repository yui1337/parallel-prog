import numpy as np
import subprocess
import os
import json
import matplotlib.pyplot as plt

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def setup_environment(dirs_dict):
    for path in dirs_dict.values():
        os.makedirs(path, exist_ok=True)

def generate_data(n, path_a, path_b):
    a = np.random.randint(0, 100, size=(n, n), dtype=np.int64)
    b = np.random.randint(0, 100, size=(n, n), dtype=np.int64)
    np.savetxt(path_a, a, fmt='%d')
    np.savetxt(path_b, b, fmt='%d')
    return a, b

def run_cpp(n, paths, exe_path, threads):
    try:
        cmd = [exe_path, str(n), paths['a'], paths['b'], paths['res'], str(threads)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception:
        return None

def plot_results(all_stats, save_path):
    """
    all_stats: словарь вида {кол-во_потоков: [(size, time), ...]}
    """
    plt.figure(figsize=(12, 7))
    
    for threads, points in all_stats.items():
        sizes, times = zip(*points)
        line, = plt.plot(sizes, times, 'o-', label=f'{threads} потоков')

    plt.title('Зависимость времени выполнения от количества потоков OpenMP', fontsize=14)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время (мс)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig(save_path)
    print(f"График сохранен в {save_path}")
    plt.show()

def main():
    config = load_config()
    setup_environment(config['directories'])
    
    all_stats = {t: [] for t in config['threads']}

    for n in config['sizes']:
        print(f"\n--- Размер матриц {n}x{n} ---")
        paths = {
            'a': os.path.join(config['directories']['matrix_a'], f"A_{n}.txt"),
            'b': os.path.join(config['directories']['matrix_b'], f"B_{n}.txt"),
            'res': os.path.join(config['directories']['results'], f"C_{n}.txt")
        }
        generate_data(n, paths['a'], paths['b'])

        for t in config['threads']:
            print(f"Запуск threads={t}...", end=" ", flush=True)
            cpp_time = run_cpp(n, paths, config['cpp_exe'], t)
            
            if cpp_time is not None:
                print(f"Время выполнения: {cpp_time} мс")
                all_stats[t].append((n, cpp_time))

    plot_results(all_stats, config['plot_filename'])

if __name__ == "__main__":
    main()