import torch
from torchvision import transforms
from PIL import Image
import argparse
import mediapipe

# Импортируем нашу функцию для создания модели
from model import create_model

def predict(image_path, model_path, class_names):
    """
    Предсказывает эмоцию на одном изображении.
    """
    # --- 1. Подготовка ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем модель с той же архитектурой
    model = create_model(num_classes=len(class_names)).to(device)
    # Загружаем сохраненные веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Переводим модель в режим оценки
    model.eval()

    # --- 2. Трансформации для изображения ---
    # Важно: они должны быть ТАКИМИ ЖЕ, как для валидационного набора
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)), # Приводим к размеру 48x48
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    # --- 3. Предсказание ---
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
        return

    # Применяем трансформации и добавляем "batch" измерение
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        # Применяем Softmax, чтобы получить вероятности
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Находим класс с максимальной вероятностью
        top_prob, top_catid = torch.topk(probabilities, 1)

    prediction = class_names[top_catid]
    confidence = top_prob.item()

    print(f"🤖 Предсказанная эмоция: {prediction.upper()}")
    print(f"🎯 Уверенность: {confidence:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Предсказание эмоции на изображении.')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению для предсказания.')
    args = parser.parse_args()

    # Список классов (убедись, что порядок верный, как при обучении)
    # Ты убрал 'disgust', поэтому проверь порядок
    classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model_checkpoint = '../EmotionsAI/checkpoints/run_2025-09-24_19-22-25/best_model.pth'

    predict(args.image, model_checkpoint, classes)