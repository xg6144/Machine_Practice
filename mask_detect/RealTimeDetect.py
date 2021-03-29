import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel') #미리 학습된 모델을 가져온다.
model = load_model('models/mask_detector.model')

#노트북에 내장된 카메라 사용
cap = cv2.VideoCapture(0)

while True:
    ret, cam = cap.read()
    if ret is None:
        print("Video Load Failed")
   #영상의 크기를 가져온다. 
    h, w = cam.shape[:2]
    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('camera', cam)

    blob = cv2.dnn.blobFromImage(cam, scalefactor=1., size=(300, 300), mean=(104.,177.,123.)) #학습된 파라미터를 넣는다.
    facenet.setInput(blob)
    dets = facenet.forward() #결과를 추론한다. model.forward()

    result_img = cam.copy()

    for i in range(dets.shape[2]): # 모델이 가져오는 최대 박스의 개수
        # detections는 4차원 배열로 이루어져있다.
        # 첫번째 i가 0일 때 detections[0,0]의 첫번째 배열값은 [0.,1.,0.999...,...]을 의미하고 이 중 2, 0.999로 이 박스가 마스크를 썻을 가능성은
        # 99퍼 센트인 것을 의미한다.
        confidence = dets[0,0,i,2]

        if confidence < 0.5: #0.5미만은 넘긴다.
            continue
        x1 = int(dets[0, 0, i, 3] * w)  # i가 0일때 3번째 배열의 값 전체 폭 중 박스 시작점의 x좌표 상대위치
        y1 = int(dets[0, 0, i, 4] * h)  # i가 0일때 4번째 배열의 값 전체 높이 중 박스 시작점의 y좌표 상대위치
        x2 = int(dets[0, 0, i, 5] * w)  # i가 0일때 5번째 배열의 값 전체 폭 중 박스 끝의 x좌표 상대위치
        y2 = int(dets[0, 0, i, 6] * h)  # i가 0일때 6번째 배열의 값 전체 높이 중 박스 끝점의 y좌표 상대위치
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
