import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os


def get_dataloaders(data_dir, batch_size=64):
    """
    Создает и возвращает загрузчики данных для тренировки и валидации.
    """
    train_path = os.path.join(data_dir, 'train')
    valid_path = os.path.join(data_dir, 'valid')

    # --- 2. Задаем НОВЫЕ трансформации под ЧБ 48x48 ---

    # Нормализация для одноканальных (grayscale) изображений
    # Переводим пиксели в диапазон [-1, 1]
    mean_gray = [0.5]
    std_gray = [0.5]

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Указываем, что картинка - ЧБ
        transforms.RandomHorizontalFlip(),  # Случайное отражение
        transforms.RandomRotation(10),  # Случайный поворот
        transforms.ToTensor(),  # Превращаем в тензор
        transforms.Normalize(torch.Tensor(mean_gray), torch.Tensor(std_gray))  # Нормализуем
    ])

    valid_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean_gray), torch.Tensor(std_gray))
    ])

    # --- 3. Создаем объекты Dataset ---
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=valid_path, transform=valid_transforms)

    # --- 4. Создаем DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, train_dataset.classes


if __name__ == '__main__':
    # Этот блок кода выполнится, только если запустить dataset.py напрямую
    # Он нужен для быстрой проверки, что все работает

    # Укажи путь к корневой папке с датасетом ('Dataset' в твоем случае)
    dataset_root_path = '../Dataset/images'

    print("Проверка загрузчика данных...")
    try:
        # Получаем загрузчики
        train_loader, valid_loader, class_names = get_dataloaders(data_dir=dataset_root_path, batch_size=32)

        # Берем один батч
        images, labels = next(iter(train_loader))

        print(f"Размер батча с картинками: {images.shape}")  # Ожидаем [32, 1, 48, 48]
        print(f"Размер батча с метками: {labels.shape}")  # Ожидаем [32]
        print(f"Названия классов: {class_names}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Проверь, что путь 'dataset_root_path' указан верно.")