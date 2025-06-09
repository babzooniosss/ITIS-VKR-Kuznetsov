import json
from typing import List, Dict, Tuple, Set
import time
from collections import defaultdict


CONFIG = {
    'min_action_frequency': 1,
    'min_sequence_coverage': 1,
    'action_types': {'Interaction', 'Quest', 'Transition', 'Shop', 'Event', 'QuestAlgorithm', 'QuestBoardState'},
}

def preprocess_sequences(input_file: str = 'data/data.json', 
                        output_file: str = 'data/encoded_sequences.json',
                        mapping_file: str = 'data/action_mapping.json',
                        stats_file: str = 'data/sequence_stats.json') -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    Предобрабатывает и кодирует последовательности действий
    """
    print("Загрузка данных...")
    start_time = time.time()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    action_counts = defaultdict(int)
    action_sequences = defaultdict(set)
    sequence_stats = {
        'total_sequences': 0,
        'total_actions': 0,
        'action_frequencies': {},
        'action_coverages': {},
        'sequence_lengths': []
    }
    

    print("Сбор статистики по действиям...")
    for player_id, player_data in data.items():
        if 'telemetry_data' in player_data:
            sequence_length = 0
            sequence_actions = set()
            
            for line in player_data['telemetry_data'].split('\n'):
                if line.strip():
                    parts = line.split('; ')[0]
                    if ':' in parts:
                        action_type = parts.split(':')[0]
                        if action_type in CONFIG['action_types']:
                            action = parts.split(':', 1)[1].strip()
                            action_counts[action] += 1
                            sequence_actions.add(action)
                            sequence_length += 1
            
            if sequence_length > 0:
                sequence_stats['total_sequences'] += 1
                sequence_stats['total_actions'] += sequence_length
                sequence_stats['sequence_lengths'].append(sequence_length)

                for action in sequence_actions:
                    action_sequences[action].add(player_id)

    sequence_stats['action_frequencies'] = dict(action_counts)
    sequence_stats['action_coverages'] = {action: len(sequences) 
                                        for action, sequences in action_sequences.items()}

    frequent_actions = {
        action for action, count in action_counts.items() 
        if (count >= CONFIG['min_action_frequency'] and 
            len(action_sequences[action]) >= CONFIG['min_sequence_coverage'])
    }
    
    print(f"\nСтатистика до фильтрации:")
    print(f"  Всего уникальных действий: {len(action_counts)}")
    print(f"  Действий с частотой >= {CONFIG['min_action_frequency']}: "
          f"{sum(1 for count in action_counts.values() if count >= CONFIG['min_action_frequency'])}")
    print(f"  Действий в >= {CONFIG['min_sequence_coverage']} последовательностях: "
          f"{sum(1 for sequences in action_sequences.values() if len(sequences) >= CONFIG['min_sequence_coverage'])}")
    print(f"  Действий, удовлетворяющих обоим критериям: {len(frequent_actions)}")
    print(f"  Всего последовательностей: {sequence_stats['total_sequences']}")
    print(f"  Всего действий: {sequence_stats['total_actions']}")
    print(f"  Средняя длина последовательности: {sum(sequence_stats['sequence_lengths'])/len(sequence_stats['sequence_lengths']):.1f}")
    print(f"  Мин. длина: {min(sequence_stats['sequence_lengths'])}")
    print(f"  Макс. длина: {max(sequence_stats['sequence_lengths'])}")

    action_to_id = {}
    id_to_action = {}
    current_id = 0

    for action in sorted(frequent_actions):
        action_to_id[action] = current_id
        id_to_action[current_id] = action
        current_id += 1

    encoded_sequences = []
    print("\nКодирование последовательностей...")
    for player_data in data.values():
        if 'telemetry_data' in player_data:
            sequence = []
            for line in player_data['telemetry_data'].split('\n'):
                if line.strip():
                    parts = line.split('; ')[0]
                    if ':' in parts:
                        action_type = parts.split(':')[0]
                        if action_type in CONFIG['action_types']:
                            action = parts.split(':', 1)[1].strip()
                            if action in frequent_actions:
                                sequence.append(action_to_id[action])
            
            if sequence:
                encoded_sequences.append(sequence)

    sequence_stats['filtered_sequences'] = len(encoded_sequences)
    sequence_stats['filtered_actions'] = sum(len(seq) for seq in encoded_sequences)
    sequence_stats['filtered_unique_actions'] = len(frequent_actions)
    
    print(f"\nСтатистика после фильтрации:")
    print(f"  Оставлено последовательностей: {sequence_stats['filtered_sequences']}")
    print(f"  Оставлено действий: {sequence_stats['filtered_actions']}")
    print(f"  Уникальных действий после фильтрации: {sequence_stats['filtered_unique_actions']}")

    print(f"\nСохранение закодированных последовательностей в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(encoded_sequences, f, ensure_ascii=False, indent=2)

    print(f"Сохранение маппинга в {mapping_file}...")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'action_to_id': action_to_id,
            'id_to_action': id_to_action
        }, f, ensure_ascii=False, indent=2)

    print(f"Сохранение статистики в {stats_file}...")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(sequence_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nПредобработка завершена за {time.time() - start_time:.2f} секунд")
    
    return encoded_sequences, action_to_id, id_to_action

if __name__ == "__main__":
    preprocess_sequences() 