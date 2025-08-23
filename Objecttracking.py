from ultralytics import YOLO
import cv2


model = YOLO(r"C:\veri arttirimi\yolo11n.pt")


cap = cv2.VideoCapture(r"C:\veri arttirimi\yavru kedicikler.mp4")


unique_cats = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label == 'cat':  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            track_id = int(box.id[0]) if box.id is not None else -1

           
            unique_cats.add(track_id)

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ID:{track_id} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f'Toplam Kediler: {len(unique_cats)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Kedi Takibi ve Sayimi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Toplam benzersiz kedi sayisi: {len(unique_cats)}")
