import cv2
import numpy as np
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imwrite("temp_frame.jpg", frame)
    
    try:
        result = DeepFace.verify(img1_path=[ "IMG_9291.jpeg","IMG_9288.jpeg","IMG_9285.jpeg","IMG_9284.jpeg","IMG_9282.jpeg""IMG_9281.jpeg",
              "IMG_9275.jpeg","IMG_9274.jpeg","IMG_9272.jpeg","IMG_9270.jpeg","IMG_9269.jpeg"],
                                 img2_path="temp_frame.jpg",
                                 model_name="Facenet")
        
        if result["verified"]:
            print("Здравствуйте, хозяин")
            cv2.putText(frame, "Welcome!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Лицо не распознано")
            cv2.putText(frame, "Unknown", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except Exception as e:
        print(f"Ошибка: {e}")
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
