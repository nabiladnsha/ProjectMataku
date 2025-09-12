from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import time

app = Flask(__name__)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dataset warna dalam format HSV
color_dataset = np.array([
    # Merah
    [0, 255, 255], [5, 255, 255], [180, 255, 255], [175, 255, 255],
    # Orange
    [15, 255, 255], [20, 255, 255],
    # Kuning
    [30, 255, 255], [25, 255, 255],
    # Hijau
    [60, 255, 255], [70, 255, 255],
    # Biru
    [120, 255, 255], [110, 255, 255],
    # Violet
    [145, 255, 255], [150, 255, 255],
    # Hitam 
    [0, 0, 30], [180, 30, 30],
    # Putih 
    [0, 0, 255], [180, 30, 255],
])

color_labels = [
    'Merah', 'Merah', 'Merah', 'Merah',
    'Orange', 'Orange',
    'Kuning', 'Kuning',
    'Hijau', 'Hijau',
    'Biru', 'Biru',
    'Violet', 'Violet',
    'Hitam', 'Hitam',
    'Putih', 'Putih'
]

# Inisialisasi KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(color_dataset)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, color_labels)

class ColorStabilizer:
    def __init__(self, buffer_size=10, stability_threshold=0.8):
        self.color_buffer = deque(maxlen=buffer_size)
        self.hex_buffer = deque(maxlen=buffer_size)
        self.stable_color = None
        self.stable_hex = None
        self.threshold = stability_threshold
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 500ms

    def update(self, color, hex_code):
        current_time = time.time()
        
        # Tambahkan warna baru ke buffer
        self.color_buffer.append(color)
        self.hex_buffer.append(hex_code)
        
        # Hanya update jika sudah melewati interval waktu
        if current_time - self.last_update_time > self.update_interval:
            # Cek stabilitas warna
            if len(self.color_buffer) == self.color_buffer.maxlen:
                color_counts = {}
                for c in self.color_buffer:
                    color_counts[c] = color_counts.get(c, 0) + 1
                
                # Ambil warna yang paling sering muncul
                most_common_color = max(color_counts.items(), key=lambda x: x[1])
                if most_common_color[1] / len(self.color_buffer) >= self.threshold:
                    self.stable_color = most_common_color[0]
                    
                    # Kalkulasi rata-rata hex untuk warna yang stabil
                    hex_values = [int(h[1:], 16) for h in self.hex_buffer]
                    avg_hex = sum(hex_values) // len(hex_values)
                    self.stable_hex = f"#{avg_hex:06X}"
            
            self.last_update_time = current_time
        
        return self.stable_color, self.stable_hex

# Inisialisasi stabilizer
color_stabilizer = ColorStabilizer()

def bgr_to_hex(bgr_color):
    """Mengonversi warna BGR ke kode Hex."""
    b, g, r = bgr_color
    return f'{int(r):02X}{int(g):02X}{int(b):02X}'

def get_average_color(frame, hsv_frame, cx, cy, area_size=20):
    """Mengambil rata-rata warna dari area tertentu."""
    x1 = max(0, cx - area_size)
    x2 = min(frame.shape[1], cx + area_size)
    y1 = max(0, cy - area_size)
    y2 = min(frame.shape[0], cy + area_size)
    
    area_bgr = frame[y1:y2, x1:x2]
    area_hsv = hsv_frame[y1:y2, x1:x2]
    
    avg_bgr = np.mean(area_bgr, axis=(0, 1))
    avg_hsv = np.mean(area_hsv, axis=(0, 1))
    
    return avg_bgr, avg_hsv

def predict_color(hsv_values):
    """Memprediksi warna menggunakan model KNN."""
    hsv_array = np.array([hsv_values])
    hsv_scaled = scaler.transform(hsv_array)
    prediction = knn.predict(hsv_scaled)
    return prediction[0]

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        cx, cy = int(width / 2), int(height / 2)
        
        # Area sampling
        area_size = 20
        cv2.rectangle(frame, 
                     (cx - area_size, cy - area_size),
                     (cx + area_size, cy + area_size),
                     (0, 0, 0), 4)
        
        cv2.rectangle(frame, 
                     (cx - area_size + 2, cy - area_size + 2),
                     (cx + area_size - 2, cy + area_size - 2),
                     (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/color_info')
def color_info():
    if not cap.isOpened():
        return jsonify({"color": "N/A", "hex": "N/A"})
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"color": "N/A", "hex": "N/A"})
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape
    cx, cy = int(width / 2), int(height / 2)
    
    avg_bgr, avg_hsv = get_average_color(frame, hsv_frame, cx, cy)
    predicted_color = predict_color(avg_hsv)
    hex_color = bgr_to_hex(avg_bgr)
    
    # Stabilisasi warna
    stable_color, stable_hex = color_stabilizer.update(predicted_color, hex_color)
    
    if stable_color and stable_hex:
        return jsonify({"color": stable_color, "hex": stable_hex})
    return jsonify({"color": predicted_color, "hex": f"#{hex_color}"})

if __name__ == "__main__":
    app.run(debug=True)