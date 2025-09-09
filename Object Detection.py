import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
import threading
import os

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        root.title("🔍 Real-Time Object Detection")
        root.geometry("1000x800")
        root.configure(bg='#2c3e50')
        
        self.setup_gui()
        self.check_files()
        
    def check_files(self):
        required_files = ["yolov3.weights", "yolov3.cfg", "coco.names"]
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.status_label.config(text=f"Missing files: {', '.join(missing_files)}")
            messagebox.showerror("Error", 
                f"Please download these files:\n{', '.join(missing_files)}\n\n"
                "1. yolov3.weights: https://pjreddie.com/media/files/yolov3.weights\n"
                "2. yolov3.cfg: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg\n"
                "3. coco.names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
            return
        
        self.setup_yolo()

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        header = tk.Label(main_frame, text="🔍 REAL-TIME OBJECT DETECTION", 
                         font=("Arial", 20, "bold"), bg='#2c3e50', fg='#ffd700')
        header.pack(pady=(0, 15))

        self.video_label = Label(main_frame, bg='#34495e', relief='sunken')
        self.video_label.pack(pady=10, fill='both', expand=True)

        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame, text="🎥 START DETECTION", 
                                 command=self.start_detection,
                                 font=('Arial', 14, 'bold'),
                                 bg='#27ae60', fg='white',
                                 padx=25, pady=12,
                                 cursor='hand2')
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ STOP", 
                                command=self.stop_detection,
                                font=('Arial', 14, 'bold'),
                                bg='#e74c3c', fg='white',
                                padx=25, pady=12,
                                cursor='hand2')
        self.stop_btn.pack(side='left', padx=10)
        self.stop_btn.config(state='disabled')
        
        self.status_label = tk.Label(main_frame, text="✅ Files found! Ready to start...", 
                                    font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#2ecc71')
        self.status_label.pack(pady=10)
        
        info_text = "Detects: persons, cars, animals, phones, laptops, and 80+ objects!"
        info_label = tk.Label(main_frame, text=info_text, font=('Arial', 10), 
                             bg='#2c3e50', fg='#bdc3c7')
        info_label.pack(pady=5)

    def setup_yolo(self):
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam")
            return
            
        self.is_running = True
        self.start_btn.config(state='disabled', bg='#95a5a6')
        self.stop_btn.config(state='normal', bg='#e74c3c')
        self.status_label.config(text="🔍 Detecting objects in real-time...", fg='#3498db')
        
        self.detection_thread = threading.Thread(target=self.detect_objects)
        self.detection_thread.daemon = True
        self.detection_thread.start()

    def stop_detection(self):
        self.is_running = False
        self.start_btn.config(state='normal', bg='#27ae60')
        self.stop_btn.config(state='disabled', bg='#95a5a6')
        self.status_label.config(text="✅ Detection stopped. Ready to start again.", fg='#2ecc71')
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

    def detect_objects(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids, confidences, boxes = [], [], []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = self.colors[class_ids[i]]
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                               font, 0.7, color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk

        self.stop_detection()

    def __del__(self):
        self.is_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_detection(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()