import cv2
import sys

# Importar arquivo XML


#imagePath = "2.jpg"
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
#image = cv2.imread(imagePath)


#print(image)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.6,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE


    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x, y),(x+w, y+h), (0,255,0), 2)


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 
                "Found {0} faces.".format(len(faces)), 
                (25, 50), 
                font, 0.5, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
#cv2.waitKey(0)

video_capture.release()
cv2.destroyAllWindows()
