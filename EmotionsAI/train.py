import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm  # для красивого progress bar
from datetime import datetime

# Импортируем наши модули
from dataset import get_dataloaders
from model import create_model

# --- 1. Гиперпараметры и настройки ---
# Эти параметры можно менять, чтобы влиять на качество и скорость обучения
DATA_DIR = '../Dataset/images'  # Путь к папке с датасетом (train/valid)
CHECKPOINT_DIR = '../EmotionsAI/checkpoints'  # Папка для сохранения моделей
BATCH_SIZE = 64
EPOCHS = 25  # Количество раз, которое модель "посмотрит" на весь датасет
LEARNING_RATE = 0.001  # Скорость обучения модели


def train_model():
    # Создаем папку для сохранения, если её нет
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(CHECKPOINT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)  # Создаем эту папку
    print(f"Веса для этого запуска будут сохраняться в: {run_dir}")

    # --- 2. Подготовка ---
    # Выбираем устройство (видеокарта, если доступна, иначе процессор)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nОбучение будет на устройстве: {device}")

    # Загружаем данные
    train_loader, valid_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    # Создаем модель
    # В файле train.py

    # ... (код создания модели) ...
    model = create_model(num_classes).to(device)
    # Функция потерь (Loss Function) и Оптимизатор
    criterion = nn.CrossEntropyLoss()
    # Оптимизируем только веса "головы" (model.fc), т.к. остальное заморожено
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 3. Цикл обучения ---
    best_accuracy = 0.0  # Для отслеживания лучшей модели

    for epoch in range(EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{EPOCHS}")
        print('-' * 10)

        # --- Фаза обучения ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Обучение"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Обнуляем градиенты

            outputs = model(inputs)  # Прямой проход
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # Считаем ошибку

            loss.backward()  # Обратное распространение ошибки
            optimizer.step()  # Обновляем веса

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"\nLoss на обучении: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Фаза валидации ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():  # Не считаем градиенты на валидации
            for inputs, labels in tqdm(valid_loader, desc="Валидация"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(valid_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(valid_loader.dataset)
        print(f"\nLoss на валидации: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Сохраняем модель, если она показала лучшую точность
        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print("\nНовая лучшая модель сохранена!")
        scheduler.step()
    print(f"\nОбучение завершено. Лучшая точность на валидации: {best_accuracy:.4f}")


if __name__ == '__main__':
    train_model()