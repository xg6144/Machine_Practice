import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from pygame import mixer
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

mixer.init()
sound = mixer.Sound('audio/alarm.wav')

if not sound:
    print('sound load failed')
    sys.exit()

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')


#노트북에 내장된 카메라 사용
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)

while True:
    ret, cam = cap.read()
    cam = cv2.flip(cam, 1)  # 화면 미러링 끄기

    if ret is None:
        print("Video Load Failed")
        sys.exit()
    h, w = cam.shape[:2]
    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('camera', cam)

    blob = cv2.dnn.blobFromImage(cam, scalefactor=1., size=(300, 300), mean=(104.,177.,123.)) #학습된 파라미터를 넣는다.
    facenet.setInput(blob) #모델에 들어가는 input
    detections = facenet.forward() #결과를 추론한다. model.forward()

    result_img = cam.copy()
    #사진 속 얼굴 개수가 여러 명일 수 있으니 반복문을 사용한다.
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2] #확률값
        if confidence < 0.5: #0.5미만은 넘긴다.
            continue
        x1 = int(detections[0,0,i,3] * w) #
        y1 = int(detections[0,0,i,4] * h)
        x2 = int(detections[0,0,i,5] * w)
        y2 = int(detections[0,0,i,6] * h)
        #바운딩 박스를 계산한다.
        face = cam[y1:y2, x1:x2]

        #이미지 사이즈 변경인데 오류가 발생한다. 그냥 예외처리 한다.
        try:
            face_input = cv2.resize(face, dsize=(224,224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0) #행에 대한 차원을 추가한다.

            mask, nomask = model.predict(face_input).squeeze()

        except Exception as e:
            print(str(e))

        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            sound.play()
            print("Beep")
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

        cv2.rectangle(result_img, pt1=(x1,y1), pt2=(x2,y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1,y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color = color, thickness= 2, lineType=cv2.LINE_AA)

    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
