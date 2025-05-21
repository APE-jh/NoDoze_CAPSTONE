import cv2
import mediapipe as mp

# 얼굴/눈 탐지 모델 준비
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 카메라 연결
cap = cv2.VideoCapture(0)  # 웹캠 연결

# Mediapipe 모델 초기화
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)

# 눈 감김 여부를 확인하는 함수 (눈 비율 계산)
def eye_aspect_ratio(eye):
    # 눈의 좌표들 (두 점 사이의 비율로 눈 감김을 계산)
    A = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5
    B = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5
    C = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5

    ear = (A + B) / (2.0 * C)
    return ear

# 눈 비율이 임계값 이하로 떨어지면 졸음으로 간주
EAR_THRESHOLD = 0.2  # 눈 비율 임계값

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 색상 변환 (RGB -> BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # 얼굴에 대한 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # 눈 부분만 추출 (랜드마크 인덱스를 활용)
            left_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(33, 133)]  # 왼쪽 눈 랜드마크
            right_eye = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in range(133, 233)]  # 오른쪽 눈 랜드마크

            left_eye_ratio = eye_aspect_ratio(left_eye)
            right_eye_ratio = eye_aspect_ratio(right_eye)

            # 눈 비율 계산
            ear = (left_eye_ratio + right_eye_ratio) / 2.0
            print(f"Eye Aspect Ratio: {ear}")

            if ear < EAR_THRESHOLD:
                cv2.putText(frame, "Drowsy Driver!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                print("Drowsiness detected!")  # 졸음 상태 감지

    # 실시간 화면 표시
    cv2.imshow('Drowsiness Detection', frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()