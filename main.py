import cv2
import numpy as np
from deepface import DeepFace


owner_face = "owner.jpg"

result = DeepFace.verify(img1_path=owner_face,
                         img2_path="face_from_camera.jpg",
                         model_name="Facenet")

if result["verified"]:
    print("Здравствуйте, хозяин")
else:
    print("Лицо не распознано")

