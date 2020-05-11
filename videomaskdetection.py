# -*- coding: utf-8 -*-

pip install face_recognition

import face_recognition

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from google.colab import files
from keras.preprocessing import image
from tensorflow import keras

def modelfunc(uploaded):

    path = '/content/' + uploaded
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    xmodel=keras.models.load_model('/content/drive/My Drive/model.h5')
    classes = xmodel.predict(images, batch_size=10)
    print(classes[0][0])
    if classes[0][0]>0.5:
      return 'NO MASK'
    else:
      return 'MASK'

import cv2
from google.colab.patches import cv2_imshow

size = 1
cam = cv2.VideoCapture("/content/drive/My Drive/masktestvid.avi")

import numpy as np
from keras.preprocessing import image

if not cam.isOpened():
    raise IOError("Video not loaded")

cv2.startWindowThread()
res=(int(cam.get(3)),int(cam.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, 20.0, res)

while True:
  (rval, im) = cam.read()
  # predicting 
  if rval == True:
    cv2.imwrite("frame.jpg", im)
    imag = face_recognition.load_image_file("/content/frame.jpg")
    face_locations = face_recognition.face_locations(imag, number_of_times_to_upsample=1, model="cnn")

    for f in face_locations:
      t, r, b, l = f
      sub_face = im[t:b, l:r]
      FaceFileName = "face.jpg" #Saving frame
      if sub_face.size!=0:
        cv2.imwrite(FaceFileName, sub_face)
      text = modelfunc(FaceFileName)
      if text=="MASK":
        cv2.rectangle(im, (l,t), (r,b), (0,255,0), 2) #green
      else:
        cv2.rectangle(im, (l,t), (r,b), (0,0,255), 2) #red

    out.write(im)
  else:
    print("VIDEO FINISHED")
    break
  if cv2.waitKey(1) & 0xFF == 27: #Esc
    break

cam.release()
out.release()
