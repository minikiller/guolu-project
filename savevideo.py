import cv2

# videoName = "20211022102003117.mp4"
videoName = "./video/02.mp4"

targetPath = "guolu02/"
# Opens the Video file
cap = cv2.VideoCapture(videoName)
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(targetPath+'guolu'+str(i)+'.png', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
