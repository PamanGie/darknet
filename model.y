import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4_new.cfg")
classes = []
with open("yolov4/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()
output_layers = [layer_names.index(name) for name in layer_names]

# Open Webcam
cap = cv2.VideoCapture(0)

# Inisialisasi variabel untuk mengukur FPS
start_time = time.time()
frame_count = 0

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Hitung FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Detect Objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists to store the bounding box information
    boxes = []
    confidences = []
    class_ids = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object Detected
                label = classes[class_id]

                # Get coordinates of the bounding box
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner coordinates
                top_left_x = int(center_x - width / 2)
                top_left_y = int(center_y - height / 2)

                # Store bounding box information
                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels for the remaining detections
    for i in indices:
        i = i[0] if isinstance(i, list) else i  # Handle both cases
        box = boxes[i]
        top_left_x, top_left_y, width, height = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), color, 2)
        cv2.putText(frame, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tampilkan frame dengan deteksi objek
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("YOLO Object Detection", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
