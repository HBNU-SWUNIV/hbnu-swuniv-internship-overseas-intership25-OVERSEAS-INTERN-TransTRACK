import cv2
import torch
import time
from ultralytics import YOLO
import numpy as np


def main(video_path, model_path, output_path):
    # YOLO Model call
    model = YOLO(model_path)

    # Create objects for capturing video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Create objects for storing video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # setting colors for each classes
    label_colors = {
        0: (255, 0, 0),   # Blue - Eyes closed
        1: (255, 255, 0), # Cyan - Normal
        2: (255, 255, 255) # White - Yawning
    }

    # --------------------------------------------------
    # Frame count variable
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    threshold = int(total_frames * 0.3) # Edit here to change the threshold

    # --------------------------------------------------
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # object detection
        results = model(frame)
        detected = False
        confs = []
        
        # visualize the results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
                conf = box.conf[0].item()  # model detection confidence
                confs.append(float(conf))
                cls = int(box.cls[0].item())  # class ID

                if cls in [0, 2]:
                    detected = True
                
                # visualize bounding box & label
                color = label_colors.get(cls, (0, 255, 0))
                label = f"{model.names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if detected:
                frame_count += 1
        
        # print Alert classification result on the video
        text_x = width - 400 
        text_y = height - 50 

        alertTxt = "Drowsy Alert" if frame_count >= threshold else "No Issues detected"
        cv2.putText(frame, alertTxt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (0,0,255), thickness=2)
        out.write(frame)

        # print the result by a window
        cv2.imshow("YOLO Detection", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return F"Detection Result: {alertTxt}, Mean confidence: {np.mean(confs):.5f}"


if __name__ == "__main__":
    videoName = "1_64_66_3_1740478679"
    video_path = f"./data_Video/{videoName}.mp4"
    model_path = "./train/weights/best.pt" 
    output_path = f"./results1/{videoName}.mp4"
    result = main(video_path, model_path, output_path)

    print(videoName)
    print(result)
