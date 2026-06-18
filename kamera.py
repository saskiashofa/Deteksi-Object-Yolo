import cv2  
from ultralytics import YOLO 
from datetime import datetime 

model = YOLO("best.pt") 

cap = cv2.VideoCapture(1) 

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

nama_file = datetime.now().strftime("hasil_tracking_%Y%m%d_%H%M%S.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(nama_file, fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Tracking sampah
    results = model.track(
    frame,
    persist=True,
    conf=0.5
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("YOLO Trash Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video tracking berhasil disimpan sebagai: {nama_file}")