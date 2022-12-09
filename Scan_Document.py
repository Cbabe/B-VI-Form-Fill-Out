import cv2
import imutils
from helper_functions import *
import skimage
import mediapipe as mp
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 250)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# For webcam input:
cap = cv2.VideoCapture(0)
prevouis_locations = []
found = False
while cap.isOpened():
    success, image = cap.read()
    image.flags.writeable = True
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    if not found:
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        # show the original image and the edge detected image
        #print("STEP 1: Edge Detection")
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        # loop over the contours
        screenCnt = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        # show the contour (outline) of the piece of paper
        #print("STEP 2: Find contours of paper")
        if screenCnt is not None:
            # print(screenCnt)
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)
            # apply the four point transform to obtain a top-down
            # view of the original image
            warped, maxWidth, maxHeight = four_point_transform(
                orig, screenCnt.reshape(4, 2) * ratio)
            #print(f"maxWidth: {maxWidth} maxHeight: {maxHeight}")
            # convert the warped image to grayscale, then threshold it
            # to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            T = skimage.filters.threshold_local(
                warped, 11, offset=10, method="gaussian")
            warped = (warped > T).astype("uint8") * 255
            # show the original and scanned images
            #print("STEP 3: Apply perspective transform")
            #cv2.imshow("Original", imutils.resize(orig, height=650))
            resized_warped = imutils.resize(warped, height=650)
            cv2.imshow("Scanned", resized_warped)
            # Find the text in the image and return it's position
            location = find_text(resized_warped)
            # If the image contained text
            if location[0] is not None:
                # print(location)
                x_high = 0
                y_low = 600
                # Tries to find the top left coordinate of the paper
                for i in screenCnt:
                    if i[0][0] > x_high and i[0][1] < y_low:
                        x_high = i[0][0]
                        y_low = i[0][1]
                #print(f"Location: {location[0]}, {location[1]} ")
                #print(f"screen: {x_high}, {y_low} ")
                # scale the size of the warped image that was how text discovery and positioning to its size in the original image
                x_scale = maxWidth/(resized_warped.shape)[0]
                y_scale = maxHeight/(resized_warped.shape)[1]
                #print(f"x_scale: {x_scale}")
                #print(f"y_scale: {y_scale}")
                # Converts the position of the signature from the warped image to the original image
                x = x_high+round(location[0] * x_scale)
                y = y_low+round(location[1] * y_scale)
                # Position of the signature
                image_with_point = cv2.circle(image, (x, y), radius=10,
                                              color=[255, 255, 255], thickness=-1)
                # Position top left of the edge of the warped image
                image_with_point = cv2.circle(image, (x_high, y_low), radius=10,
                                              color=[0, 255, 0], thickness=-1)
                cv2.imshow("Points", image_with_point)
                # If this is the first time detecting the signature
                if len(prevouis_locations) == 0:
                    prevouis_locations.append([x, y])
                # Checks if this detection is close to the last detection
                elif prevouis_locations[-1][0]+30 > x and prevouis_locations[-1][0]-30 < x and prevouis_locations[-1][1]+30 > y and prevouis_locations[-1][1]-30 < y:
                    # adds current detection to the list
                    prevouis_locations.append([x, y])
                    print(f"Appended length {len(prevouis_locations)}")
                    if len(prevouis_locations) > 2:  # If the previous position is the same
                        engine.say("Found")  # We found the signature position!
                        engine.runAndWait()
                        print("Found")
                        found = True
                else:
                    # restart, the current position was to far from the last one indicating a misread or movement of the page
                    prevouis_locations = []
    else:
        # Delete windows many windows can cause crashing
        cv2.destroyWindow("Outline")
        cv2.destroyWindow("Scanned")
        cv2.destroyWindow("Points")
        # Creates the signature point
        image_with_point = cv2.circle(image, (x, y), radius=10,
                                      color=[0, 0, 255], thickness=-1)
        cv2.imshow("FinalPoint", image_with_point)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Finds hands in the image
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    image_hight, image_width, _ = image.shape
                    # Finds pointer finger
                    x_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                    y_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight
                    print(f'Index finger tip coordinate: (',
                          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, 'f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})')
                    print(f'point tip coordinate: ({x},{y})')
                # Directs points finger to signature position
                if y_finger-y > 20:
                    print("Go up")
                    engine.say("Up")
                    engine.runAndWait()
                elif y_finger-y < (-20):
                    print("Go down")
                    engine.say("Down")
                    engine.runAndWait()
                if x_finger-x > 20:
                    print("Go Right")
                    engine.say("Right")
                    engine.runAndWait()
                elif x_finger-x < (-20):
                    print("Go Left")
                    engine.say("Left")
                    engine.runAndWait()
                if abs(x_finger-x) < 20 and abs(y_finger-y) < 20:
                    print("right position")
                    engine.say("correct")
                    engine.runAndWait()

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)  # Shows hand
    cv2.waitKey(2)
cv2.destroyAllWindows()  # Kills everything
