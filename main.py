desc = '''
kullanim: python main.py

1. program acilinca ilk olarak labellaranarak kaydedimis imaj dosyalarini alir
2. svm classinda train methodu cagrilarak ogrenme islemi gerceklesir.
3. orneklerin %15 i ogrenme datasi olarak ayrilip cross validation ve confusion matrix hesabi yapilir
4. confusion matrix ekrana plot edilir.
5. confusion matrix kapaninca, kullanici elini kameraya tutup live olarak svm.predict islemini gerceklestirebilir.

'''

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import svm
import utils
import time

svm_util = svm.SVM()
svm_util.train()
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

    image = cv2.flip(image, 1)  # Mirror display
    time.sleep(0.4)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
       
        brect = utils.calc_bounding_rect(image, hand_landmarks)
        landmark_list = utils.calc_landmark_list(image, hand_landmarks)
        image = utils.draw_bounding_rect(True, image, brect)
        image = utils.draw_landmarks(image, landmark_list)
        
        crop_img = image[brect[1]:brect[3], brect[0]:brect[2]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        crop_img = cv2.GaussianBlur(crop_img, (3,3), 0) 
        crop_img = cv2.Canny(image=crop_img, threshold1=100, threshold2=200)        

        up_points = (150, 150)
        crop_img = cv2.resize(crop_img, up_points, interpolation= cv2.INTER_LINEAR)

        prediction = svm_util.predict(crop_img)

        image = utils.draw_info_text(
            image,
            brect,
            handedness,
            prediction,
            "",
        )

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()