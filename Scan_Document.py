import cv2
import imutils
import time
from helper_functions import *
import skimage

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
# For webcam input:
cap = cv2.VideoCapture(0)
prevouis_locations = []
while cap.isOpened():
    success, image = cap.read()
    image.flags.writeable = True
    cv2.imshow("Image", image)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
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
        print(f"maxWidth: {maxWidth} maxHeight: {maxHeight}")
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
        location = find_text(resized_warped)
        if location[0] is not None:
            print(screenCnt)
            x_high = 0
            y_low = 600
            for i in screenCnt:
                if i[0][0] > x_high and i[0][1] < y_low:
                    x_high = i[0][0]
                    y_low = i[0][1]
            print(f"Location: {location[0]}, {location[1]} ")
            print(f"screen: {x_high}, {y_low} ")
            x_scale = maxWidth/(resized_warped.shape)[0]
            y_scale = maxHeight/(resized_warped.shape)[1]
            print(f"x_scale: {x_scale}")
            print(f"y_scale: {y_scale}")
            x = x_high+round(location[0] * x_scale)
            y = y_low+round(location[1] * y_scale)
            image_with_point = cv2.circle(image, (x, y), radius=10,
                                          color=[0, 0, 255], thickness=-1)
            image_with_point = cv2.circle(image, (x_high, y_low), radius=10,
                                          color=[0, 255, 0], thickness=-1)
            if len(prevouis_locations) == 0:
                prevouis_locations.append([x_high, y_low])
            elif prevouis_locations[-1][0]+10 > x_high and prevouis_locations[-1][0]-10 < x_high and prevouis_locations[-1][1]+10 > y_low and prevouis_locations[-1][1]-10 < y_low:
                print("Appended")
                prevouis_locations.append([x_high, y_low])
                if len(prevouis_locations) > 10:
                    print("Found")
                    break
            else:
                prevouis_locations = []
print("Done")


cv2.waitKey(2)
cv2.destroyAllWindows()
