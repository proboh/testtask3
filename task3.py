from inference import get_model
import supervision as sv
import cv2
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=""  # Add your API Key here
)

# Video Path
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)

# Tracking Variables
tracked_paths = {}  # Dictionary to store paths for each tracked car
frame_count = 0  # To keep track of frame number

# Create Supervision Annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Infer cars in the current frame using Roboflow API
    results = CLIENT.infer(frame, model_id="vehicle-object-recognition/1")
    detections = sv.Detections.from_inference(results)

    # Extract detection data
    for idx, box in enumerate(detections.xyxy):
        # Bounding box coordinates
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        center_x, center_y = x1 + width / 2, y1 + height / 2

        # Confidence score and class name
        confidence = detections.confidence[idx]
        class_name = detections.data["class_name"][idx]

        # Only process detections of class "Car"
        if class_name != "Car":
            continue

        # Assign a unique tracker ID (if available) or use index
        car_id = detections.tracker_id[idx] if detections.tracker_id is not None else idx

        # Update tracked paths
        if car_id not in tracked_paths:
            tracked_paths[car_id] = []
        tracked_paths[car_id].append((frame_count, (center_x, center_y)))

        # Draw bounding box and label
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        frame = cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the processed frame
    cv2.imshow("Car Detection and Tracking", frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Paths to File
with open("tracked_paths.txt", "w") as f:
    for car_id, path in tracked_paths.items():
        f.write(f"Car ID: {car_id}, Path: {path}\n")

print("Tracking completed. Paths saved to tracked_paths.txt.")
