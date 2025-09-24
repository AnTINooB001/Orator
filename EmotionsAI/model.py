import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=6):
    """
    Создает модель ResNet18, адаптированную для нашей задачи.
    1. Загружает предобученную на ImageNet модель ResNet18.
    2. Модифицирует первый слой для приема одноканальных (ЧБ) изображений.
    3. Заменяет последний слой для классификации на num_classes.
    """
    # Загружаем модель с предобученными весами
    model = models.resnet18()

    # --- 1. Модифицируем ПЕРВЫЙ слой ---
    # Оригинальный слой model.conv1 принимает 3-канальные изображения (RGB)
    # Нам нужно, чтобы он принимал 1-канальные (Grayscale)

    # Сохраняем веса оригинального слоя
    original_weights = model.conv1.weight.clone()

    # Создаем новый сверточный слой
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Адаптируем веса: усредняем по трем каналам RGB, чтобы получить веса для одного канала
    with torch.no_grad():
        new_conv1.weight[:] = torch.mean(original_weights, dim=1, keepdim=True)

    # Заменяем старый слой на новый
    model.conv1 = new_conv1
    # --- 2. Модифицируем ПОСЛЕДНИЙ слой (классификатор) ---
    # model.fc - это последний полносвязный слой в ResNet
    ct = 0
    for child in model.children():
        ct += 1
        # Заморозим первые 6 дочерних блоков ResNet18
        # Оставим обучаемыми layer3, layer4, avgpool и fc
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

    # Заменяем последний слой, как и раньше
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # ДОБАВЛЯЕМ DROPOUT. p=0.5 - "выключаем" 50% нейронов
        nn.Linear(num_ftrs, num_classes)
    )

    print("Модель ResNet18 успешно создана и модифицирована.")
    return model


if __name__ == '__main__':
    # Блок для проверки, что функция работает

    # Создаем модель
    model = create_model(num_classes=7)

    # Создаем фейковый тензор, имитирующий наш батч данных
    # (batch_size=1, channels=1, height=48, width=48)
    dummy_input = torch.randn(1, 1, 48, 48)

    # Прогоняем его через модель
    output = model(dummy_input)

    # Печатаем результат
    print(f"\nТестовый запуск модели...")
    print(f"Размер входного тензора: {dummy_input.shape}")
    print(f"Размер выходного тензора: {output.shape}")  # Ожидаем [1, 7]
    print("Это значит, что для 1 картинки модель выдала 7 вероятностей (для каждого класса).")