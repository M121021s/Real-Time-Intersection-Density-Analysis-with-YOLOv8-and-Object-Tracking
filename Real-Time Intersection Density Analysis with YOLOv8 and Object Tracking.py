from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
import os

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture("arabalar.mp4")

# Video kayıt için
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
success, img = cap.read()
size = list(img.shape)
del size[2]
size.reverse()
video = cv2.VideoWriter("kaydedilen_video.mp4", cv2_fourcc, 24, size)

model = YOLO("yolov8n.pt")

# Poligon noktalarını yükle veya varsayılan değerlerle başlat
polygon_file = "polygon_points.npy"
if os.path.exists(polygon_file):
    polygon_points = np.load(polygon_file)
else:
    polygon_points = np.array([[50, 500], [1200, 450], [1850, 500], [1250, 1050], [50, 850]], np.int32)

# Araç takibi için
vehicle_tracking = {}
vehicle_counter = 0
vehicles_in_polygon = set()

# Görüntüleme ayarları
show_grid = True
edit_mode = False
selected_point = -1


def save_polygon_image(img, polygon_pts):
    """Poligonlu görüntüyü kaydet"""
    # Orijinal görüntünün bir kopyasını oluştur
    img_copy = img.copy()

    # Poligonu çiz
    if len(polygon_pts) > 2:
        cv2.polylines(img_copy, [polygon_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        overlay = img_copy.copy()
        cv2.fillPoly(overlay, [polygon_pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)

    # Noktaları işaretle
    for i, point in enumerate(polygon_pts):
        cv2.circle(img_copy, tuple(point), 8, (255, 0, 0), -1)
        cv2.putText(img_copy, f"{i + 1}", (point[0] + 10, point[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Kaydet
    cv2.imwrite("ornek_resim_poligon.jpg", img_copy)
    print("Poligonlu görüntü kaydedildi: ornek_resim_poligon.jpg")


def draw_grid(img):
    """Ekranda ızgara çizen fonksiyon"""
    h, w = img.shape[:2]
    for x in range(0, w, 50):
        cv2.line(img, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w, y), (50, 50, 50), 1)

    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h), (100, 100, 100), 1)
    for y in range(0, h, 100):
        cv2.line(img, (0, y), (w, y), (100, 100, 100), 1)

    for x in range(0, w, 100):
        cv2.putText(img, str(x), (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for y in range(0, h, 100):
        cv2.putText(img, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def draw_polygon(img):
    """Poligon bölgeyi çizen fonksiyon"""
    if len(polygon_points) > 1:
        cv2.polylines(img, [polygon_points], isClosed=True, color=(0, 255, 255), thickness=2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [polygon_points], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    for i, point in enumerate(polygon_points):
        color = (0, 0, 255) if i == selected_point else (255, 0, 0)
        cv2.circle(img, tuple(point), 8, color, -1)
        cv2.putText(img, f"{i + 1}", (point[0] + 10, point[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def is_in_polygon(x, y):
    """Bir noktanın poligon içinde olup olmadığını kontrol eder"""
    if len(polygon_points) < 3:
        return False
    return cv2.pointPolygonTest(polygon_points, (x, y), False) >= 0


def find_nearest_point(x, y, threshold=20):
    """Belirtilen koordinata en yakın noktayı bulur"""
    for i, point in enumerate(polygon_points):
        if math.hypot(point[0] - x, point[1] - y) < threshold:
            return i
    return -1


def mouse_callback(event, x, y, flags, param):
    """Fare olaylarını işler"""
    global selected_point, polygon_points, edit_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if edit_mode:
            selected_point = find_nearest_point(x, y)
            if selected_point == -1:
                if len(polygon_points) > 1:
                    min_dist = float('inf')
                    insert_index = 0
                    for i in range(len(polygon_points)):
                        next_i = (i + 1) % len(polygon_points)
                        dist = cv2.pointPolygonTest(np.array([polygon_points[i], polygon_points[next_i]]), (x, y), True)
                        if abs(dist) < min_dist:
                            min_dist = abs(dist)
                            insert_index = next_i
                    polygon_points = np.insert(polygon_points, insert_index, [[x, y]], axis=0)
                    selected_point = insert_index
                else:
                    polygon_points = np.append(polygon_points, [[x, y]], axis=0)
                    selected_point = len(polygon_points) - 1

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if edit_mode and selected_point != -1:
            polygon_points[selected_point] = [x, y]

    elif event == cv2.EVENT_RBUTTONDOWN and edit_mode:
        point_idx = find_nearest_point(x, y)
        if point_idx != -1 and len(polygon_points) > 3:
            polygon_points = np.delete(polygon_points, point_idx, axis=0)
            selected_point = -1


# Fare callback fonksiyonu
cv2.namedWindow("Araç Sayma Sistemi")
cv2.setMouseCallback("Araç Sayma Sistemi", mouse_callback)

# İlk kareyi al ve poligonlu görüntüyü kaydet
success, first_frame = cap.read()
if success:
    save_polygon_image(first_frame, polygon_points)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    if show_grid:
        draw_grid(img)

    if edit_mode:
        cv2.putText(img, "EDIT MODE: ON (Press 'E' to exit)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "Left Click: Add/Select Point", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Drag: Move Point", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Right Click: Delete Point", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        results = model(img, stream=True)
        current_frame_vehicles = set()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] not in ['car', 'truck', 'bus', 'motorbike']:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if is_in_polygon(cx, cy):
                    vehicle_id = None
                    min_dist = float('inf')

                    for vid, (prev_cx, prev_cy) in vehicle_tracking.items():
                        dist = math.hypot(cx - prev_cx, cy - prev_cy)
                        if dist < 50 and dist < min_dist:
                            min_dist = dist
                            vehicle_id = vid

                    if vehicle_id is None:
                        vehicle_counter += 1
                        vehicle_id = vehicle_counter

                    vehicle_tracking[vehicle_id] = (cx, cy)
                    current_frame_vehicles.add(vehicle_id)

                    cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=3, rt=1)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf} ID:{vehicle_id}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)

        vehicles_in_polygon = current_frame_vehicles
        cv2.putText(img, f'Araclar: {len(vehicles_in_polygon)}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    draw_polygon(img)
    video.write(img)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Araç Sayma Sistemi", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        edit_mode = not edit_mode
        selected_point = -1
        if not edit_mode:
            # Düzenleme modundan çıkınca poligonu kaydet
            np.save(polygon_file, polygon_points)
            save_polygon_image(first_frame, polygon_points)
    elif key == ord('g'):
        show_grid = not show_grid
    elif key == ord('c'):
        polygon_points = np.array([], np.int32).reshape(0, 2)
    elif key == ord('s'):
        print("Kaydedilen poligon noktaları:")
        print(polygon_points.tolist())
        np.save(polygon_file, polygon_points)
        save_polygon_image(first_frame, polygon_points)

video.release()
cv2.destroyAllWindows()

