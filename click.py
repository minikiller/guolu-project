# import the necessary packages
import argparse
import cv2
import setting

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
# videoFile = "20211022102003117.mp4"
# videoFile = "192.168.1.64_01_20211026152332696.mp4"
useVideo = False

videoFile = "./video/02.mp4"
videoUrl = "rtsp://admin:kalix123@192.168.0.64:554/Streaming/channels/101/"


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        print("get point is:", refPt[0], refPt[1])
        # cv2.imshow("image", image)


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
# img_clone = image.copy()
if useVideo:
    cap = cv2.VideoCapture()
    cap.open(videoUrl)
else:
    cap = cv2.VideoCapture(videoFile)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    ret, image = cap.read()
    # cv2.imshow("image", image)
    img_clone = image.copy()
    if(refPt and len(refPt) == 2):
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):  # ???r??????
        # image = img_clone.copy()
        refPt = []
    # if the 'c' key is pressed, break from the loop
    elif key == ord("s"):  # ???s????????????
        if(refPt and len(refPt) == 2):
            x, y = refPt[0]
            w, h = refPt[1]
            result = x, y, w-x, h-y
            print('wriet result to ', result)
            setting.writeScale(*result)
    if key == ord('p'):  # ???p?????????????????????
        cv2.waitKey(-1)
    elif key == 27:  # ???esc?????????
        break
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
    roi = img_clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

    print(roi.shape)

    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
