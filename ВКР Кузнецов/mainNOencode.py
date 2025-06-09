import json
import os
from typing import List, Dict, Tuple
from visualization import plot_pattern_weight, visualize_pattern_network, cluster_patterns, analyze_pattern_statistics, cluster_patterns_by_levenshtein, create_output_dir
from prefixspan import PrefixSpan
import time
from collections import Counter



CUT_PREFIX_LEN = 0
CUT_UNTIL_ACTION = None
MIN_SEQUENCE_LENGTH = 5


PATTERN_MIN_LENGTH = 3
PATTERN_MAX_LENGTH = 7


COL = 1000


TOPN = 10
COEF_NET = 0.001
N_clusters_KMeans = 3
MAX_DISTANCE_LEVENSHTEIN = None
N_clusters_Levenshtein = 3

def ensure_output_dir():
    if not os.path.exists('output'):
        os.makedirs('output')
        print("Создана директория output/")

def load_data(sequences_file: str = 'data/data.json',
              cut_prefix_len: int = 0,
              cut_until_action: str = None,
              min_length: int = 1):
    """
    Загружает последовательности и применяет фильтрацию
    """
    print("Загрузка данных...")
    start_time = time.time()
    
    with open(sequences_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sequences = []

    for player_data in data.values():
        if 'telemetry_data' in player_data:
            sequence = []
            for line in player_data['telemetry_data'].split('\n'):
                if line.strip():
                    part = line.split('; ')[0]
                    if ':' in part:
                        _, action = part.split(':', 1)
                        sequence.append(action.strip())

            if cut_until_action is not None:
                try:
                    idx = sequence.index(cut_until_action)
                    sequence = sequence[idx:]
                except ValueError:
                    continue
            elif cut_prefix_len > 0:
                sequence = sequence[cut_prefix_len:]

            if len(sequence) >= min_length:
                sequences.append(sequence)
    
    print(f"Загрузка и обработка завершены за {time.time() - start_time:.2f} секунд")
    print(f"Загружено {len(sequences)} последовательностей (после фильтрации)")
    
    return sequences

def main():
    output_dir = create_output_dir()
    print(f"Результаты будут сохранены в директории: {output_dir}")

    sequences = load_data(
        cut_prefix_len=CUT_PREFIX_LEN,
        cut_until_action=CUT_UNTIL_ACTION,
        min_length=MIN_SEQUENCE_LENGTH
    )
    
    print("\nПоиск паттернов с помощью библиотеки PrefixSpan:")
    start_time = time.time()

    ps = PrefixSpan(sequences)
    
    flat = [a for seq in sequences for a in seq]
    total_freq = Counter(flat)

    sequence_freq = Counter()
    for seq in sequences:
        for action in set(seq):
            sequence_freq[action] += 1
    
    def action_potential(action):
        return total_freq[action] * (sequence_freq[action] / len(sequences))

    def predictive_key(patt, matches):
        if not patt:
            return 0
        start = patt[0]
        potential = action_potential(start)
        return potential * len(matches)

    def predictive_bound(patt, matches):
        if not patt:
            return 0
        start = patt[0]
        potential = action_potential(start)
        return potential * len(matches)

    ps.minlen = PATTERN_MIN_LENGTH
    ps.maxlen = PATTERN_MAX_LENGTH

    patterns = ps.topk(
        COL,  
        key=predictive_key,
        bound=predictive_bound,
    )
    
    duration = time.time() - start_time
    print(f"Поиск паттернов завершен за {duration:.2f} секунд ({duration/60:.2f} минут)")
    print(f"Найдено {len(patterns)} паттернов")

    formatted_patterns = []
    for support, pattern in patterns:
        formatted_patterns.append({
            'pattern': pattern,
            'weight': support
        })

    formatted_patterns.sort(key=lambda x: (-x['weight'], len(x['pattern'])))

    print("\nТоп-10 найденных паттернов:")
    for pattern in formatted_patterns[:10]:
        print(f"Паттерн: {pattern['pattern']}, Вес: {pattern['weight']}")

    patterns_json_path = os.path.join(output_dir, 'patterns.json')
    with open(patterns_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_patterns, f, ensure_ascii=False, indent=2)
    print(f"Все найденные паттерны сохранены в {patterns_json_path}")

    if not formatted_patterns:
        print("Не найдено паттернов, удовлетворяющих критериям. Попробуйте изменить параметры.")
        return

    print("\nСоздание визуализаций...")
    start_time = time.time()

    freq_plot_path = plot_pattern_weight(formatted_patterns, top_n=TOPN, output_dir=output_dir)
    print(f"Создан график веса паттернов ({freq_plot_path})")

    network_plot_path = visualize_pattern_network(formatted_patterns, coef_weight=COEF_NET, output_dir=output_dir)
    print(f"Создана визуализация сети паттернов ({network_plot_path})")

    clustered_patterns, clusters_plot_path = cluster_patterns(formatted_patterns, n_clusters=N_clusters_KMeans, output_dir=output_dir)
    print(f"Выполнена кластеризация паттернов методом K-means ({clusters_plot_path})")

    clustered_patterns_lev, dist_matrix, clusters_lev_plot_path = cluster_patterns_by_levenshtein(
        formatted_patterns, 
        max_distance=MAX_DISTANCE_LEVENSHTEIN,
        COL_clusters=N_clusters_Levenshtein,
        output_dir=output_dir
    )
    print(f"Выполнена кластеризация паттернов по расстоянию Левенштейна ({clusters_lev_plot_path})")

    stats, stats_path = analyze_pattern_statistics(formatted_patterns, output_dir=output_dir)
    print("\nСтатистика паттернов:")
    print(stats)
    print(f"Статистика сохранена в {stats_path}")
    
    print(f"Визуализация завершена за {time.time() - start_time:.2f} секунд")
    print(f"\nВсе результаты сохранены в директории: {output_dir}")

if __name__ == "__main__":
    main()
