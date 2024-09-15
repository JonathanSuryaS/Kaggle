import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\user\Documents\GitHub\Kaggle\Age detection\facial_expression.h5')
cap = cv2.VideoCapture(0)

class_label = [
    'Abyssinian',
    'Bombay',
    'Egyptian Mau',
    'Exotic Shorthair',
    'Himalayan',
    'Maine Coon',
    'Regdoll',
    'Russian Blue',
    'Scottish Fold',
    'Siamese',
    'Sphynx'
]

while True:
    ret, frame = cap.read()
    
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis = 0)
    
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions[0])
    
    label = class_label[predicted_class]
    
    cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Capture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
