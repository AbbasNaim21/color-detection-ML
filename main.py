import cv2
from PIL import Image
from util import get_limits


yellow = [0, 255, 255]  # yellow BGR

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # as the ML or DL read the colors
    # as BGR not RGB we will be using the BGR2HSV converter
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    mask_ = Image.fromarray(mask)

    # draw bounding box surrounding our object draw
    boundingBox = mask_.getbbox()
    if boundingBox is not None:
        x1, y1, x2, y2 = boundingBox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # getting the pixels that contains the color that we want
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
