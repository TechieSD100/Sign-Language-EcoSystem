import cv2
import time
import os
import HandTrackingModule as htm

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load overlay images for numbers
folderPath = "FingerImages"
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in os.listdir(folderPath) if imPath.endswith('.png') or imPath.endswith('.jpg')]

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        try:
            fingers = detector.fingersUp(lmList)
            totalFingers = fingers.count(1)

            # Ensure totalFingers is within range of overlayList
            if 0 <= totalFingers <= len(overlayList):
                h, w, c = overlayList[totalFingers - 1].shape
                img[0:h, 0:w] = overlayList[totalFingers - 1]

                # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
                # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            else:
                print(f"Total fingers {totalFingers} out of range for overlayList")
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
    # else:
    #     print("No hand detected")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
