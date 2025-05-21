import cv2
import mediapipe as mp
import time
import math

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 랜드마크 인덱스
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]

# EAR 계산 함수
def eye_aspect_ratio(eye_landmarks):
    A = math.dist(eye_landmarks[1], eye_landmarks[5])
    B = math.dist(eye_landmarks[2], eye_landmarks[4])
    C = math.dist(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

# 목 기울기 각도 계산 함수
def calculate_head_tilt(landmarks):
    nose = landmarks.landmark[1]
    chin = landmarks.landmark[152]
    dx = nose.x - chin.x
    dy = nose.y - chin.y
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# 모의 GPIO 출력 함수
def mock_gpio_output(pin, state):
    print(f"GPIO {pin} -> {state}")

# 기준값 설정
EAR_THRESHOLD = 0.2              # 눈 감김 기준
EAR_FRAMES = 30                  # 약 1초 (30fps 기준)
TILT_LOWER = -95                 # 기울기 하한
TILT_UPPER = -85                 # 기울기 상한
TILT_FRAMES = 30                 # 약 1초 기준 (30fps 기준)

# 상태 변수
eye_closed_counter = 0
tilt_counter = 0
drowsy_detected = False

# 카메라 연결
cap = cv2.VideoCapture(0)

# Mediapipe 모델 시작
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for landmarks in result.multi_face_landmarks:
                h, w = frame.shape[:2]

                # 눈 좌표 추출
                left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE_INDEXES]
                right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE_INDEXES]

                # EAR 계산
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # 목 기울기 계산
                head_tilt_angle = calculate_head_tilt(landmarks)

                # EAR 판단
                if ear < EAR_THRESHOLD:
                    eye_closed_counter += 1
                else:
                    eye_closed_counter = 0

                # 목 기울기 판단 (정상 범위: -95도 ~ -85도)
                if not (TILT_LOWER <= head_tilt_angle <= TILT_UPPER):
                    tilt_counter += 1
                else:
                    tilt_counter = 0

                # 졸음 판별 조건
                if (eye_closed_counter > EAR_FRAMES or tilt_counter > TILT_FRAMES) and not drowsy_detected:
                    print("🚨 Drowsiness Detected! 🚨")
                    mock_gpio_output(18, "HIGH")
                    time.sleep(1)
                    mock_gpio_output(18, "LOW")
                    drowsy_detected = True
                    cv2.putText(frame, "Drowsy Driver!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                elif eye_closed_counter == 0 and tilt_counter == 0:
                    drowsy_detected = False

                # 디버깅 텍스트 출력
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Head Tilt: {head_tilt_angle:.2f}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

        # 프레임 출력
        cv2.imshow("Drowsiness Detection", frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()