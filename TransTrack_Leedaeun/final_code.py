import cv2
from ultralytics import YOLO
from collections import defaultdict

# 1. 모델 로드 및 동영상 열기
model = YOLO(r"best.pt")
video_path = r"Normal.mp4"
cap = cv2.VideoCapture(video_path)

# 2. 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# 3. 누적 시간과 프레임당 시간 계산
label_time = defaultdict(float)   
frame_time = 1.0 / fps         

# 4. 모델 라벨 매핑 (클래스 id -> 라벨명)
names = model.names if hasattr(model, "names") else model.model.names

# 5. 연속 검출을 위한 변수들
continuous_tracking = {}          
max_continuous = defaultdict(float)  
threshold_continuous = 3.0    

cofidence = []

# 6. 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 현재 비디오 시간 (초)

    # YOLO 모델 추론
    results = model(frame, conf=0.3, iou=0.3)

    # 이번 프레임에서 검출된 라벨을 중복 없이 기록
    detected_labels = set()
    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls)
                label = names[class_id]
                detected_labels.add(label)
                cofidence.append(box.conf.item())
                
    
    # (a) 누적 시간 업데이트: 이번 프레임에서 검출된 각 라벨에 대해 프레임 시간을 누적
    for label in detected_labels:
        label_time[label] += frame_time
        # (b) 연속 검출 tracking 업데이트
        # - 이번 프레임에 검출된 라벨: tracking에 없다면 현재 시간으로 시작 기록
        if label not in continuous_tracking:
            continuous_tracking[label] = video_time

    # - 이번 프레임에 검출되지 않은 라벨은 tracking에서 제거(연속성이 끊긴 것으로 간주)
    labels_to_remove = [label for label in continuous_tracking if label not in detected_labels]
    for label in labels_to_remove:
        del continuous_tracking[label]
    
    # (c) 현재 tracking 중인 라벨에 대해 연속 검출 시간 업데이트 및 최대값 갱신
    for label in continuous_tracking:
        current_duration = video_time - continuous_tracking[label]
        if current_duration > max_continuous[label]:
            max_continuous[label] = current_duration

    # 결과 시각화 및 디스플레이
    annotated_frame = results[0].plot() if results else frame
    out.write(annotated_frame)
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 7. 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

# 8. 최종 결과 결정
# 우선, max_continuous에서 연속 3초 이상 검출된 라벨들을 필터링합니다.
final_label = None
max_label = None
if max_continuous:
    valid_labels = {label: duration for label, duration in max_continuous.items() if duration >= threshold_continuous}
    if valid_labels:
        # valid_labels 중 가장 긴 연속 시간을 가진 라벨을 최종 결과로 선정
        final_label = max(valid_labels, key=valid_labels.get)

# 9. 결과 출력
print("\n****************************************************")
if final_label is not None:
    print("This vedio is {}  (Maximum continuous detection time: {:.2f}sec)".format(final_label, max_continuous[final_label]))
else:
    print("No label detected for more than 3 seconds in a row.")
    print("\nAccumulated detection time by label(sec):")
    for label, t in label_time.items():
        print(f"{label}: {t:.2f}sec")
    if label_time:
        max_label = max(label_time, key=label_time.get)
        max_time = label_time[max_label]
        print(f"Most Detected Labels: {max_label} ({max_time:.2f}sec)")

if max_label == "Eyes Close" or max_label == "Yawning" or final_label == "Eyes Close" or final_label == "Yawning": 
            print("\nThis person is presumed to be sleepy.")   

# (d) 평균 confidence 계산 
if cofidence: 
    avg_cof = sum(cofidence) / len(cofidence)
    print(f"AVG cofidenc : {avg_cof:.4f}")    
else :
    print("Error")
print("\n****************************************************")
