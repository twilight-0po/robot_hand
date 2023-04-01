import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hand_landmarks = 0
begin = time.time()
#X Y z 좌푯값 얻기
def get_landmark_xyz(hand_landmarks, img_width, img_height):
  x_value = str(round(hand_landmarks.landmark[mp_hands.HandLandmark(8).value].x * img_width, 3))
  y_value = str(480 - round(hand_landmarks.landmark[mp_hands.HandLandmark(8).value].y * img_height, 3))
  z_value = str(round(hand_landmarks.landmark[mp_hands.HandLandmark(8).value].z * img_width, 3))
  return x_value, y_value, z_value


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = hands.process(image)
    img_height, img_width, _ = image.shape
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        #0.1초마다 x값 불러와서 표시하기
        end = time.time()
        results = end - begin
        if round(results, 2) >= 0.1:
          x_value, y_value, z_value = get_landmark_xyz(hand_landmarks, img_width,img_height )
          begin = time.time()
        # 지점의 x, y z 값 화면에 표시하기
        cv2.putText(image,f'x:{x_value}', (30, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(image, f'y:{y_value}', (30, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(image, f'z:{z_value}', (30, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 3)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        

  
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

  