import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\user\Documents\GitHub\Kaggle\Facial Expression\facial_expression.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
class_label = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        input_frame = cv2.resize(face_roi, (224, 224))
        input_frame = input_frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)
        
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions[0])
        label = class_label[predicted_class]
        

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Predicted: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    

    cv2.imshow('Real-Time Capture', frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
