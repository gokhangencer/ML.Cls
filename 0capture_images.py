desc = '''
kullanim: python 0capture_images.py

1. program acilinca kullanicidan label ismi istenir.
2. kullanici elini girdigi label a uygun sekilde kameraya gosterir
3. 'a' tusuna basarak kayit islemine baslar
4. 200 ornek olunca program otomatik olarak kapanir

not: 'q' tusu ile de istenilen bir anda cikis saglanabilir.
'''

import cv2
import os
from tkinter import simpledialog
import time
import utils
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

label_name = "1"
num_samples = 200

try:
    label_name = simpledialog.askstring("Capture Label Data", "Enter label name:")
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)

if label_name is None:
    exit(-1);

IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        ret, image = cap.read()
        if not ret:
            continue

        if count == num_samples:
            break

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
                image = utils.draw_info_text(
                    image,
                    brect,
                    handedness,
                    label_name,
                    "",
                )

                crop_img = image[brect[1]:brect[3], brect[0]:brect[2]]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                #crop_img = cv2.GaussianBlur(crop_img, (3,3), 0) 
                crop_img = cv2.Canny(image=crop_img, threshold1=100, threshold2=200)

                up_points = (150, 150)
                crop_img = cv2.resize(crop_img, up_points, interpolation= cv2.INTER_LINEAR)

                if start:
                    save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
                    cv2.imwrite(save_path, crop_img)
                    count += 1  

                cv2.imshow("cropped", crop_img)

        cv2.imshow('MediaPipe Hands', image)

        k = cv2.waitKey(10)
        if k == ord('a'):
            start = not start

        if k == ord('q'):
            break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
