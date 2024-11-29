# testtask3

Task 3: Car Path Tracking
  Objective: Track moving cars from a video file (video2 +srt2) and plot their paths on a map.
Steps:
  1. Analyze the video to detect and track moving cars.
  2. You can use methods like background subtraction, object detection models, or Optical Flow.
  3. Extract the paths of the detected cars.
  4. Plot the paths on a map for visualization.

Breakdown of Steps:
  1. Process each video frame dynamically.
  2. Detect cars using the Roboflow API.
   Object Detection:
     Uses Roboflow's pre-trained model for car detection via API.
  3. Track car paths across frames.
     Example: ![Screenshot 2024-11-29 083939](https://github.com/user-attachments/assets/23ed735e-010c-44b3-8d30-8363996d8184)
  4. Store car paths for further mapping.
     WIP
