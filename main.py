import cv2
import numpy as np
import os
from datetime import datetime


class FaceRecognizer:
    def __init__(self, reference_images_paths):

        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.reference_faces = []
        self.load_reference_faces(reference_images_paths)

    def load_reference_faces(self, paths):

        for path in paths:
            try:
                img = cv2.imread(path)
                if img is not None:

                    faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face = img[y:y + h, x:x + w]
                        # Изменяем размер для единообразия
                        face = cv2.resize(face, (100, 100))
                        self.reference_faces.append(face)
                        print(f"Загружено эталонное фото: {path}")
                    else:
                        print(f"Лицо не найдено на фото: {path}")
            except Exception as e:
                print(f"Ошибка загрузки {path}: {e}")

    def compare_faces(self, face1, face2):

        face1_hsv = cv2.cvtColor(face1, cv2.COLOR_BGR2HSV)
        face2_hsv = cv2.cvtColor(face2, cv2.COLOR_BGR2HSV)


        hist1 = cv2.calcHist([face1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([face2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])


        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)


        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity * 100

    def recognize(self, frame):


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return False, 0, None


        x, y, w, h = faces[0]
        current_face = frame[y:y + h, x:x + w]
        current_face = cv2.resize(current_face, (100, 100))


        best_match = 0
        for ref_face in self.reference_faces:
            similarity = self.compare_faces(current_face, ref_face)
            if similarity > best_match:
                best_match = similarity


        is_recognized = best_match > 60

        return is_recognized, best_match, (x, y, w, h)



reference_photos = [
    "IMG_9291.jpeg", "IMG_9288.jpeg", "IMG_9285.jpeg", "IMG_9284.jpeg",
    "IMG_9282.jpeg", "IMG_9281.jpeg", "IMG_9275.jpeg", "IMG_9274.jpeg",
    "IMG_9272.jpeg", "IMG_9270.jpeg", "IMG_9269.jpeg"
]


recognizer = FaceRecognizer(reference_photos)


cap = cv2.VideoCapture(0)

print("Программа запущена. Нажмите 'q' для выхода")
recognition_cooldown = 0  # Задержка для уменьшения нагрузки

while True:
    ret, frame = cap.read()
    if not ret:
        break


    if recognition_cooldown == 0:
        is_owner, similarity, face_coords = recognizer.recognize(frame)
        recognition_cooldown = 10  # Пауза на 10 кадров
    else:
        recognition_cooldown -= 1


    if face_coords:
        x, y, w, h = face_coords
        if is_owner:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"Welcome! ({similarity:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Здравствуйте, хозяин! Сходство: {similarity:.1f}%")
        else:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f"Unknown ({similarity:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('Face Recognition - OpenCV', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
