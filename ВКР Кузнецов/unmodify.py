#!/usr/bin/env python3
import os
import sys
import sysconfig


def patch_nextentries():
    """Заменяет функцию nextentries в библиотеке extratools"""
    site_packages = sysconfig.get_paths()["purelib"]
    target_file = os.path.join(site_packages, 'extratools', 'dicttools.py')
    
    if not os.path.exists(target_file):
        print(f"Файл {target_file} не найден!")
        return False

    new_function = '''def nextentries(data: Sequence[Sequence[T]], entries: Entries) -> Mapping[T, Entries]:
    return invertedindex(
        (data[i][lastpos + 1:] for i, lastpos in entries),
        entries
    )

    
'''

    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    func_start = content.find("def nextentries(")
    if func_start == -1:
        print("Не удалось найти функцию nextentries!")
        return False

    next_func = content.find("def ", func_start + 1)
    if next_func == -1:
        next_class = content.find("class ", func_start + 1)
        if next_class == -1:
            func_end = len(content)
        else:
            func_end = next_class
    else:
        func_end = next_func

    new_content = content[:func_start] + new_function + content[func_end:]

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Функция nextentries успешно заменена в {target_file}")
    return True

def main():
    """Основная функция"""
    print("Отмена модификации для PrefixSpan...")
    


    nextentries_patched = patch_nextentries()
    
    if nextentries_patched:
        print("Модификация успешно отменена!")
        return 0
    else:
        print("Не удалось отменить модификацию!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
