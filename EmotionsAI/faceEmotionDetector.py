import torch
import torchvision.transforms as transforms
import cv2  # Для работы с изображениями и рисования
import mediapipe as mp  # Для детекции лиц
from PIL import Image
import numpy as np
import argparse
import os

from model import create_model  # Наша функция для создания модели эмоций


def load_emotion_model(model_path, num_classes, device):
    """Загружает обученную модель классификации эмоций."""
    model = create_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Переводим модель в режим оценки
    return model


def detect_and_predict_emotions(image_path, emotion_model, class_names, device):
    """
    Принимает путь к изображению, находит лица, предсказывает эмоции
    и возвращает изображение с нарисованными рамками и эмоциями.
    """
    # Загружаем изображение с помощью OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MediaPipe ожидает RGB

    # Инициализируем MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Трансформации для лица, перед подачей в модель эмоций
    # Должны быть такими же, как для валидационного набора
    emotion_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),  # Наша модель обучена на 48x48
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7) as face_detection:

        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                # Получаем координаты лица
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                    int(bbox_c.width * iw), int(bbox_c.height * ih)

                # Обрезаем лицо с изображения
                # Убедимся, что координаты в пределах изображения
                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)

                face_crop = img_rgb[y:y + h, x:x + w]

                if face_crop.size == 0:  # Пропускаем, если обрезка пустая
                    continue

                # Преобразуем обрезанное лицо для нашей модели эмоций
                face_pil = Image.fromarray(face_crop)
                face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)

                # Предсказываем эмоцию
                with torch.no_grad():
                    outputs = emotion_model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_catid = torch.topk(probabilities, 1)

                emotion_label = class_names[top_catid.item()]
                confidence = top_prob.item()

                # Рисуем рамку и текст на оригинальном изображении
                color = (0, 255, 0)  # Зеленая рамка
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)

                text = f"{emotion_label.upper()} ({confidence:.2%})"

                # Определяем положение текста
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

                # Размещаем текст над рамкой, если есть место, иначе под
                text_x = x
                text_y = y - 10 if y - 10 > text_size[1] else y + h + text_size[1] + 10

                # Добавляем фон для текста, чтобы он был более читаемым
                # cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),
                #               (text_x + text_size[0], text_y + 5), color, -1)

                cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness,
                            cv2.LINE_AA)  # Белый текст

        else:
            print("Лица не обнаружены.")

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Детекция лиц и определение эмоций на изображении.')
    parser.add_argument('--image', type=str, required=True, help='Путь к входному изображению.')
    parser.add_argument('--output', type=str, default='output_emotions.jpg',
                        help='Путь для сохранения выходного изображения.')
    args = parser.parse_args()

    # Пути к модели и классам
    model_checkpoint = '../EmotionsAI/checkpoints/run_2025-09-24_19-22-25/best_model.pth'
    # Убедись, что порядок классов точно соответствует тому, что был при обучении
    # После удаления 'disgust' у нас осталось 6 классов
    emotion_class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Загружаем модель эмоций
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_model = load_emotion_model(model_checkpoint, len(emotion_class_names), device)

    print(f"Обработка изображения: {args.image}")
    output_image = detect_and_predict_emotions(args.image, emotion_model, emotion_class_names, device)

    if output_image is not None:
        cv2.imwrite(args.output, output_image)
        print(f"Результат сохранен в: {args.output}")
        # Опционально: показать изображение
        # cv2.imshow("Emotions Detected", output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()