import subprocess
import sys
import os

def main():
    print("Установка основных зависимостей...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--prefer-binary", 
                              "numpy", "matplotlib", "pandas", "networkx", "scikit-learn", "python-Levenshtein"])

        print("Установка extratools и prefixspan...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "extratools==0.8.1", "prefixspan==0.5.2"])

        print("Применение патча...")
        if os.path.exists("patch.py"):
            subprocess.check_call([sys.executable, "patch.py"])
        else:
            print("Файл patch.py не найден! Патч не применен.")
        
        print("Установка завершена успешно!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
