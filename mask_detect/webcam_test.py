import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, cam = cap.read()
    
    if(ret):
        cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('camera', cam)
        
        if cv2.waitKey(3000) == 27:
            break

cap.release()
cv2.destroyAllWindows()
