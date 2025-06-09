#!/usr/bin/env python3
import os
import sys
import sysconfig

def patch__mine():
    """Заменяет функцию _mine в библиотеке prefixspan"""
    site_packages = sysconfig.get_paths()["purelib"]
    target_file = os.path.join(site_packages, 'prefixspan', 'prefixspan.py')
    
    if not os.path.exists(target_file):
        print(f"Файл {target_file} не найден!")
        return False

    new_function = '''def _mine(self, func):
        # type: (Callable[[Pattern, Matches], None]) -> Any
        self._results.clear()

        # Создаем совпадения для всех позиций во всех последовательностях
        all_matches = []
        for seq_id, seq in enumerate(self._db):
            for pos in range(len(seq)):
                all_matches.append((seq_id, pos-1))  # pos-1 потому что nextentries будет искать с pos+1

        func([], all_matches)

        return self._results

    
'''

    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    func_start = content.find("def _mine(")
    if func_start == -1:
        print("Не удалось найти функцию _mine!")
        return False

    next_func = content.find("def ", func_start + 1)
    if next_func == -1:
        next_class = content.find("class ", func_start + 1)
        if next_class == -1:
            func_end = content.find("    frequent ", func_start + 1)
        else:
            func_end = next_class
    else:
        func_end = next_func

    new_content = content[:func_start] + new_function + content[func_end:]

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Функция _mine успешно заменена в {target_file}")
    return True

def main():
    """Основная функция"""
    print("Применение патча для PrefixSpan...")
    


    nextentries_patched = patch__mine()
    
    if nextentries_patched:
        print("Патч успешно примененен!")
        return 0
    else:
        print("Не удалось применить патч!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
