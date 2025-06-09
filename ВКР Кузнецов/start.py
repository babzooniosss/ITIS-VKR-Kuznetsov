import subprocess
import sys
import os

def main():
    print("Выберите опцию:")
    print("1. Запустить стандартный вариант скрипта")
    print("2. Запустить модифицированный вариант скрипта (последовательности без пропусков)")
    print("3. Запустить вариант скрипта без кодирования")
    print("4. Запустить кодирование данных")
    print("5. Выйти из программы")

    choice = input("Введите номер опции: ")

    if choice == "1":
        subprocess.check_call([sys.executable, "main.py"])
    elif choice == "2":
        subprocess.check_call([sys.executable, "modify.py"])
        subprocess.check_call([sys.executable, "main.py"])
        subprocess.check_call([sys.executable, "unmodify.py"])
    elif choice == "3":
        subprocess.check_call([sys.executable, "mainNOencode.py"])
    elif choice == "4":
        subprocess.check_call([sys.executable, "encode_actions.py"])
    elif choice == "5":
        print("Выход из программы")
        sys.exit()
    else:
        print("Неверный номер опции")

if __name__ == "__main__":
    sys.exit(main())