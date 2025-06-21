import cv2
import numpy as np
from tensorflow.keras.models import load_model
from .preprocessor import ImagePreprocessor

class RealTimePredictor:
    def __init__(self, model_path, class_names, img_size=(80,80)):
        self.model = load_model(model_path)
        self.class_names = class_names
        self.preprocessor = ImagePreprocessor(img_size)
    
    def predict_frame(self, frame):
        input_img, processed_img = self.preprocessor.prepare(frame)
        prediction = self.model.predict(input_img, verbose=0)[0]
        label_idx = np.argmax(prediction)
        return self.class_names[label_idx], np.max(prediction) * 100, processed_img
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            label, confidence, processed_img = self.predict_frame(frame)
            
            cv2.putText(frame, f"{label} ({confidence:.1f}%)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            processed_display = cv2.resize(processed_img, (200, 200))
            cv2.imshow("Processed", processed_display)
            cv2.imshow("3D Shape Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()