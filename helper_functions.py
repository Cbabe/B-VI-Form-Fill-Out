# import the necessary packages
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped, maxWidth, maxHeight


def find_text(gray_image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(
        gray_image, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Inverting the image
    inverted_bin = cv2.bitwise_not(binary_image)

    # Some noise reduction
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.erode(inverted_bin, kernel, iterations=1)
    processed_img = cv2.dilate(processed_img, kernel, iterations=1)
    results = pytesseract.image_to_data(processed_img, output_type=Output.DICT)
    # loop over each of the individual text localizations
    location = None
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
    # filter out weak confidence text localizations
        location_x = None
        location_y = None
        if conf > 70:
            if ('S' in text) or ('s' in text) or ('i' in text) or ('g' in text) or ('n' in text):
                print("Found text")
                # display the confidence and text to our terminal
                #print("Confidence: {}".format(conf))
                #print("Text: {}".format(text))
                # print("")
                # strip out non-ASCII text so we can draw the text on the image
                # using OpenCV, then draw a bounding box around the text along
                # with the text itself
                text = "".join(
                    [c if ord(c) < 128 else "" for c in text]).strip()
                cv2.rectangle(processed_img, (x, y),
                              (x + w, y + h), 255, 2)
                cv2.putText(processed_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, 255, 3)
                location_x = round(x+w)
                location_y = round(y)
                cv2.circle(processed_img, (location_x, location_y), radius=10,
                           color=255, thickness=-1)

    # show the output image

    cv2.imshow("Image", processed_img)
   # Applying image_to_string method
    return [location_x, location_y]
