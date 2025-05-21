import cv2
import mediapipe as mp
import time

# 얼굴/눈 탐지 모델 준비
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Windows에서 모의 출력으로 대체 (GPIO 핀 제어는 실제로 Jetson Nano에서만 사용)
def mock_gpio_output(pin, state):
    print(f"GPIO Pin {pin} set to {state}")  # 실제 GPIO 핀 대신 출력 메시지

# 카메라 연결
cap = cv2.VideoCapture(0)

# Mediapipe 모델 초기화
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

# 눈 감김 여부를 확인하는 함수 (눈 비율 계산)
def eye_aspect_ratio(eye):
    A = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5
    B = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5
    C = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 비율 임계값 및 타이머 변수
EAR_THRESHOLD = 0.2  # 눈 비율 임계값
eye_closed_frames = 0  # 눈 감김 상태가 지속된 프레임 수
drowsy_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 색상 변환 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 얼굴 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION
            )

            # 왼쪽/오른쪽 눈 랜드마크 추출 (33~133, 133~233는 예시; 필요시 조정)
            left_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y)
                        for i in range(33, 133)]
            right_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y)
                         for i in range(133, 233)]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            print(f"Eye Aspect Ratio: {ear:.3f}")

            # 눈 감김 감지
            if ear < EAR_THRESHOLD:
                eye_closed_frames += 1
            else:
                eye_closed_frames = 0
                drowsy_detected = False

            # 30프레임(약 1초) 이상 지속되면 졸음 판단
            if eye_closed_frames > 30:
                cv2.putText(
                    frame, "Drowsy Driver!",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
                )
                print("Drowsiness detected!")
                drowsy_detected = True

            # 졸음 감지 시 모의 GPIO 출력
            if drowsy_detected:
                mock_gpio_output(18, "HIGH")
                time.sleep(1)
                mock_gpio_output(18, "LOW")
                drowsy_detected = False

    # 화면 표시 및 종료 키 처리
    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()