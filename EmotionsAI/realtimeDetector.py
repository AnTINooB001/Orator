import torch
import torchvision.transforms as transforms
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

from model import create_model


def load_emotion_model(model_path, num_classes, device):
    """Загружает обученную модель классификации эмоций."""
    model = create_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    # --- 1. Настройки и загрузка модели ---
    model_checkpoint = '../EmotionsAI/checkpoints/run_2025-09-24_19-22-25/best_model.pth'
    emotion_class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Загрузка модели эмоций...")
    emotion_model = load_emotion_model(model_checkpoint, len(emotion_class_names), device)
    print("Модель загружена. Инициализация камеры...")

    # Инициализация MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6)

    # Трансформации для лица
    emotion_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    # --- 2. Захват видео с веб-камеры ---
    cap = cv2.VideoCapture(0)  # 0 - это ID твоей основной веб-камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось получить доступ к веб-камере.")
        return

    print("Камера активна. Для выхода нажмите 'q'.")

    while True:
        # Читаем кадр с камеры
        success, frame = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры. Завершение работы.")
            break

        # Переворачиваем кадр по горизонтали для "зеркального" вида
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 3. Детекция и предсказание (та же логика, что и раньше) ---
        results = mp_face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                    int(bbox_c.width * iw), int(bbox_c.height * ih)

                x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
                face_crop = frame_rgb[y:y + h, x:x + w]

                if face_crop.size == 0:
                    continue

                face_pil = Image.fromarray(face_crop)
                face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = emotion_model(face_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_catid = torch.topk(probabilities, 1)

                emotion_label = emotion_class_names[top_catid.item()]
                confidence = top_prob.item()

                # --- 4. Рисуем результат на кадре ---
                text = f"{emotion_label.upper()} ({confidence:.1%})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Показываем результат в окне
        cv2.imshow("Real-time Emotion Detector (Press 'q' to exit)", frame)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 5. Освобождаем ресурсы ---
    cap.release()
    cv2.destroyAllWindows()
    print("Работа завершена.")


if __name__ == '__main__':
    main()