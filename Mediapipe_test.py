import cv2
import mediapipe as mp
import pytesseract
import numpy as np
from scanner import *
# your path may be different
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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
        results = hands.process(image)
        # print(results.multi_hand_landmarks)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Preprocessing image
        # Converting to grayscale
        #image_ocr = cv2.flip(image, 1)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # creating Binary image by selecting proper threshold
        binary_image = cv2.threshold(
            gray_image, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Inverting the image
        inverted_bin = cv2.bitwise_not(binary_image)

        # Some noise reduction
        kernel = np.ones((2, 2), np.uint8)
        processed_img = cv2.erode(inverted_bin, kernel, iterations=1)
        processed_img = cv2.dilate(processed_img, kernel, iterations=1)
        cv2.imshow('Processed', processed_img)
        # Applying image_to_string method
        text = pytesseract.image_to_string(processed_img)
        if text is not None:
            print(f"this is the text({text})")
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                image_hight, image_width, _ = image.shape
                print(f'Index finger tip coordinate: (',
                      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, 'f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})')

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
