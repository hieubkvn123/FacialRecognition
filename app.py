# From macbook air
import os
import cv2
import time
import requests
# import tensorflow as tf
import pickle
import dlib
import imutils
import numpy as np
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
import sys, traceback
import gpiozero

from imutils.video import FPS
from imutils.video import WebcamVideoStream
#from imutils.video.pivideostream import PiVideoStream
#from anti_spoofing_pytorch_net import AntiSpoofingNet
#from anti_spoofing_pytorch_net import resnet_model
from thermal import *
#from pytorch_2d_face_embedder_net import Face2DEmbedders, TripletLossNet

import threading
import queue
# import torch

RELAY_PIN = 12
relay = gpiozero.OutputDevice(RELAY_PIN, active_high=False, initial_value=False)
# evo = EvoThermal()

num_iterations = 0
identity = ""
category = ""

img_files = []
num_unknowns = 0
threads = []
users = []
authenticated = []

AUTHENTICATED = False
THRESHOLD = 0.75
break_loop = 0
VERBOSE = True
_BATCH_SIZE = 9
HEIGHT, WIDTH, CHANNELS = 160,160,3
HEIGHT_, WIDTH_, CHANNELS_ = 32, 32, 3
RED = (0,0,255)
GREEN = (0,255,0)

temp_dir = 'temp'
url = 'http://209.97.163.134:5000/img'

def _loss_tensor_(y_true, y_pred, batch_size = _BATCH_SIZE):
   margin = 0.7
   model_anchor = y_pred[0:batch_size:3]
   model_positive = y_pred[1:batch_size:3]
   model_negative = y_pred[2:batch_size:3]
   distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
   distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))

   return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))

def spoof_validation(frame, spoof_detector, hsv_input, yuv_input, x,y,w,h, q):
    spoof = spoof_detector.predict([hsv_input, yuv_input])
    spoof = np.argmax(spoof)
    if(spoof == 0): # Spoof not detected
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
        q.put("Authentic")
    else : # Spoof detected
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        print("[INFO] Spoof detected ...")
        if(VERBOSE):
            cv2.putText(frame, "Spoof!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        q.put("Spoof")

def spoof_validation_ptorch(frame, spoof_detector, face, x,y,w,h,q):
    face = cv2.resize(face, (32, 32))
    hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hsv_faces = np.array([hsv_face])
    hsv_faces = torch.Tensor(hsv_faces)
    hsv_faces = hsv_faces.reshape(-1, 3, 32, 32)

    yuv_face = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
    yuv_faces = np.array([yuv_face])
    yuv_faces = torch.Tensor(yuv_faces)
    yuv_faces = yuv_faces.reshape(-1, 3, 32, 32)

    outputs = spoof_detector(hsv_faces, yuv_faces)
    outputs = outputs.detach().numpy()
    # print("[INFO] Spoof detector output : " + str(outputs))
    spoof = np.argmax(outputs[0])

    if(spoof == 0): # Spoof not detected
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
        q.put("Authentic")
    else : # Spoof detected
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        print("[INFO] Spoof detected ...")
        if(VERBOSE):
            cv2.putText(frame, "Spoof!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        q.put("Spoof")

def generate_embeddings(embedder, faces, q):
    embeddings = embedder.predict(faces)
    q.put(embeddings)

def generate_embeddings_ptorch(embedder, faces, q):
    embeddings = embedder(faces)
    embeddings = embeddings.detach().numpy()
    q.put(embeddings)

def poll_server(url, file_name): #, q):
  global users
  global img_files
  global authenticated

  # if you cannot stop the thread just make it not
  global AUTHENTICATED
  global break_loop

  img = None
  if(not AUTHENTICATED and break_loop == 0):
    try:
      # with open(file_name, 'rb') as img:
      try:
          img = open(file_name, 'rb')
      except:
          print("Thread stopped")
      img_files.append(img)
      name_img= os.path.basename(file_name)
      files= {'image': (name_img,img,'multipart/form-data',{'Expires': '0'}) }
      with requests.Session() as s:
        # print("Posting file")
        try:
          r = s.post(url,files=files)
        except:
          img.close()

        # print(r.text)
        # print(r.content)
        if("Spoof" in str(r.content)):
          category = "Spoof"
        else:
          user, certainty = r.text.split("|")[1].split("-")[0], float(r.text.split("|")[1].split("-")[1])
          # print(user)
          # print(certainty)
          category = "Authentic"
          if(user == "unknown"):
            if(users.count("unknown") == 10):
              num_unknowns += 1

          if(certainty > THRESHOLD and "unknown" not in user):
            users.append(user)
            if(not AUTHENTICATED and break_loop == 0):
              authenticated.append(user)
              print("Authenticated user : " + str(authenticated))
            # print(users)
            #if(users.count(user) == 5):
            #  if(user not in authenticated and user != "unknown"):
            #    if(not AUTHENTICATED and break_loop == 0):
            #      authenticated.append(user)
            #      # send some signal to the main thread
            #      print("Authenticated user : " + str(authenticated))

          # identity = str(r.content).split("|")[1]
      img.close()
      if(os.path.exists(file_name)):  
        os.remove(file_name) 
    except :
      # img.close()
      for file in img_files:
        file.close()
        if(os.path.exists(file.name)):
          os.remove(os.path.abspath(file.name))
        img_files = []
  else: 
    pass
    # img.close()
    # traceback.print_exc(file=sys.stdout)
    # q.put(r.content)

detector = dlib.get_frontal_face_detector()
fps = FPS().start()
vs = WebcamVideoStream(src = 0).start()
# vs = PiVideoStream.start()
# le = pickle.loads(open('models/le.pickle', 'rb').read())

while True:
    try:
        cv2.startWindowThread()
        # check, frame = webcam.read()
        frame = vs.read()
        print("Reading temperature")
        # temps, frame_ = evo.run()
        # frame = cv2.flip(frame, flipCode = -1)
        frame = imutils.resize(frame, width = 400)
        # evo = EvoThermal()
        try:
            labels = []
            probas = []
            # An anti spoofing model will be placed here
            current = time.time()
            # faces = haar.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.05, minNeighbors = 5)
            faces = detector(frame, 1)
            # print("[INFO] Face detected, took " + str(time.time() - current) + " seconds ...")
            if(not AUTHENTICATED and break_loop == 0):
              # print("in loop")
              if(len(faces) > 0): # if face is detected
                evo = EvoThermal()
                temps, frame_ = evo.run()
                for rect in faces:
                  if(len(authenticated) != len(faces) - num_unknowns):
                    # print("[INFO] Face detected ...")
                    iteration_start = time.time()
                    (x,y,w,h) = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                    x_ = int(x/8) # for thermal use
                    y_ = int(y/8) 
                    w_ = int(w/8)
                    h_ = int(h/8)

                    face = frame[y:y+h, x:x+w]
                    temps = temps[y_:y_+h_, x_:x_+w_]
                    _percentile = np.percentile(temps, 95)
                    where = np.where(temps > _percentile)
                    mean = np.mean(temps[where])
                    ratio = formula(50)/formula(0)
                    mean = mean/ratio
                    print(mean)

                    file_name = temp_dir + '/' + str(time.time()) + '.jpg'

                    cv2.imwrite(file_name, face)

                    print("Posting request")
                    t1 = threading.Thread(target = poll_server, args = (url, file_name))# , q1))
                    threads.append(t1)
                    t1.daemon = True
                    t1.start()

                    if(category == "Spoof"):
                      color = RED
                    else:
                      color = GREEN

                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                  else: # if authentication success -> reset users set, authenticated user set and stops all thread
                    AUTHENTICATED = True 
                    relay.on()
                    print("AUTHENTICATED = " + str(AUTHENTICATED))
                    # cv2.putText(frame, "Authenticated", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    break_loop = 10 # 20 frame after authenticated
                    users = []
                    authenticated = []
                    num_unknowns = 0
                    for file in img_files:
                      file.close()
                      if(os.path.exists(file.name)):
                        os.remove(file.name)
                    files = []
                    for thread in threads:
                      thread = None
                    print("[INFO] " + str(len(threads)) + " queued threads stopped ... ")
                    threads = []
                    continue
                  del evo
              else: # if not face detected
                if(len(os.listdir(temp_dir))):
                    pass
                else:
                    os.system("rm " + temp_dir + "/*.jpg")
                #for (dir, dirs, files) in os.walk(temp_dir):
                #  for file in files:
                #    if(os.path.exists(file)):
                #      os.remove(file)
            elif(AUTHENTICATED):
              # print("break " + str(break_loop))
              # AUTHENTICATED = False
              break_loop -= 1  
              if(break_loop == 0):
                relay.off()
                AUTHENTICATED = False       
        #
        except :
            print('[INFO] No face detected ...')
            traceback.print_exc(file=sys.stdout)
        # print(check) #prints true as long as the webcam is running

        cv2.imshow("Capturing", frame)
        fps.update()
        key = cv2.waitKey(1)

        if key == ord('q'):
            # print("Num iterations  = " + str(num_iterations))
            print("Turning off camera.")
            fps.stop()
            vs.stop()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        # webcam.release()
        fps.stop()
        vs.stop()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break


# Note :
# Most time-consuming processes : Face detection, spoof validation and embeddings generation
# For consideration : combine some of the process together or use threading

# Note :
# 3 models to consider : ResNet50, InceptionV3 and VGG19 for antispoofing.
# Speed rank : InceptionV3 - ResNet50 - VGG19
# Accuracy rank : VGG19 - ResNet50 - InceptionV3

# InceptionV3 seems perform better with new data

# Note :
# How to overcome faces with glasses
