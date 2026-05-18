import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
from PIL import Image, ImageTk
import cv2
import numpy as np



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
        return similarity * 150

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


class ImageApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр изображений + Распознавание лиц")
        self.root.geometry("900x650")

        # Создаем папку data
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Список для хранения путей к эталонным фото
        self.reference_photos = []
        self.image_labels = []
        self.recognizer = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Фрейм для изображений
        images_frame = tk.LabelFrame(main_frame, text="Эталонные изображения", font=("Arial", 10, "bold"))
        images_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Контейнер для 3 изображений
        container = tk.Frame(images_frame)
        container.pack(fill=tk.BOTH, expand=True)

        # Создаем 3 слота для изображений
        for i in range(3):
            frame = tk.Frame(container, relief="solid", borderwidth=2)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Заголовок
            label = tk.Label(frame, text=f"Изображение {i + 1}", font=("Arial", 10, "bold"))
            label.pack(pady=5)

            # Место для картинки
            img_label = tk.Label(frame, text="Нет изображения", bg="lightgray")
            img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.image_labels.append(img_label)

        # Информация о загруженных фото
        info_frame = tk.LabelFrame(main_frame, text="Информация", font=("Arial", 10, "bold"))
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_var = tk.StringVar()
        self.info_var.set("Загружено эталонов: 0")
        info_label = tk.Label(info_frame, textvariable=self.info_var, anchor=tk.W)
        info_label.pack(fill=tk.X, padx=10, pady=5)

        # Кнопки
        buttons_frame = tk.Frame(main_frame)
        buttons_frame.pack(pady=10)

        load_btn = tk.Button(buttons_frame, text="Загрузить эталоны", command=self.load_images, width=18)
        load_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(buttons_frame, text="Очистить эталоны", command=self.clear_images, width=18)
        clear_btn.pack(side=tk.LEFT, padx=5)

        start_btn = tk.Button(buttons_frame, text="🔍 Запустить распознавание", command=self.start_recognition,
                              bg="green", fg="white", width=20)
        start_btn.pack(side=tk.LEFT, padx=5)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе. Загрузите эталонные фото")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def load_images(self):
        """Загрузка эталонных изображений"""
        files = filedialog.askopenfilenames(
            title="Выберите эталонные изображения (лица для распознавания)",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )

        if files:
            self.process_files(list(files))

    def process_files(self, files):
        """Обработка загруженных файлов"""
        self.reference_photos = []

   
        for i in range(min(3, len(files))):
            self.load_image_to_slot(i, files[i])
            self.reference_photos.append(files[i])

       
        for i in range(3, len(files)):
            self.reference_photos.append(files[i])

       
        

        self.info_var.set(f"Загружено эталонов: {len(self.reference_photos)}")
        self.status_var.set(f"Загружено {len(files)} изображений. Теперь можно запустить распознавание")

    
        self.create_recognizer()

    def load_image_to_slot(self, slot, file_path):
        """Загрузка изображения в слот"""
        try:
            img = Image.open(file_path)
            img.thumbnail((180, 180), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.image_labels[slot].config(image=photo, text="")
            self.image_labels[slot].image = photo

        except Exception as e:
            print(f"Ошибка загрузки: {e}")


    def create_recognizer(self):
        """Создание распознавателя лиц"""
        if len(self.reference_photos) > 0:
            self.recognizer = FaceRecognizer(self.reference_photos)
            self.status_var.set(f"Распознаватель создан. Загружено {len(self.reference_photos)} эталонов")

    def clear_images(self):
        """Очистка всех изображений"""
        for i in range(3):
            self.image_labels[i].config(image="", text="Нет изображения")
            self.image_labels[i].image = None

        self.reference_photos = []
        self.recognizer = None
        self.info_var.set("Загружено эталонов: 0")
        self.status_var.set("Изображения очищены. Загрузите новые эталоны")

    def start_recognition(self):
        """Запуск распознавания с камеры"""
        if self.recognizer is None or len(self.reference_photos) == 0:
            messagebox.showwarning("Предупреждение", "Сначала загрузите эталонные изображения!")
            return

        self.run_face_recognition()

    def run_face_recognition(self):
        """Запуск камеры и распознавания"""
        if self.recognizer is None:
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
            return

        self.status_var.set("Распознавание запущено. Нажмите 'q' в окне камеры для выхода")

        recognition_cooldown = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if recognition_cooldown == 0:
                is_owner, similarity, face_coords = self.recognizer.recognize(frame)
                recognition_cooldown = 10
            else:
                recognition_cooldown -= 1

            if face_coords:
                x, y, w, h = face_coords
                if is_owner:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(frame, f"Welcome!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"Здравствуйте, хозяин! Сходство: {similarity:.1f}%")
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, f"Unknown ", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Recognition - Камера', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Распознавание завершено")



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
