import cv2

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error Opening Video File")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

