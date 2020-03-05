#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np
import crcmod.predefined
import serial
from struct import unpack
import serial.tools.list_ports
import threading

# cam = cv2.VideoCapture(0) # A normal camera
haar_file = "/home/pi/face_recog/models/haarcascade_frontalface_default.xml"
haar = cv2.CascadeClassifier(haar_file)

class EvoThermal():
    def __init__(self):
        ### Search for Evo Thermal port and open it ###
        ports = list(serial.tools.list_ports.comports())
        portname = None
        for p in ports:
            if ":5740" in p[2]:
                # print("EvoThermal found on port " + p[0])
                portname = p[0]
        if portname is None:
            print("Sensor not found. Please Check connections.")
            exit()
        ser = serial.Serial(
                            port=portname,  # To be adapted if using UART backboard
                            baudrate=115200, # 1500000 for UART backboard
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS
                            )
        self.port = ser
        self.serial_lock = threading.Lock()
        ### CRC functions ###
        self.crc32 = crcmod.predefined.mkPredefinedCrcFun('crc-32-mpeg')
        self.crc8 = crcmod.predefined.mkPredefinedCrcFun('crc-8')
        ### Activate sensor USB output ###
        self.activate_command   = (0x00, 0x52, 0x02, 0x01, 0xDF)
        self.deactivate_command = (0x00, 0x52, 0x02, 0x00, 0xD8)
        self.send_command(self.activate_command)

    def get_thermals(self):
        got_frame = False
        frame = None
        while not got_frame:
            with self.serial_lock:
                ### Polls for header ###
                header = self.port.read(2)
                # header = unpack('H', str(header))
                header = unpack('H', header)
                if header[0] == 13:
                    ### Header received, now read rest of frame ###
                    data = self.port.read(2068)
                    # frame = np.reshape(data, (400,400))
                    ### Calculate CRC for frame (except CRC value and header) ###
                    calculatedCRC = self.crc32(data[:2064])
                    data = unpack("H" * 1034, data)
                    receivedCRC = (data[1032] & 0xFFFF ) << 16
                    receivedCRC |= data[1033] & 0xFFFF
                    TA = data[1024]
                    data = data[:1024]

                    #frame = np.reshape(data, (32,32))
                    data = np.reshape(data, (32, 32))
                    frame = data.astype(np.uint8)

                    ### Compare calculated CRC to received CRC ###
                    if calculatedCRC == receivedCRC:
                        got_frame = True
                    else:
                        print("Bad CRC. Dropping frame")
        self.port.flushInput()
        ### Data is sent in dK, this converts it to celsius ###
        data = (data/10.0) - 273.15  # 1428/(np.log(231159.5/(data-6094.284)) + 1) # (data/10.0) - 273.15
        TA = (TA/10.0) - 273.15

        return data, frame

    def send_command(self, command):
        ### This avoid concurrent writes/reads of serial ###
        with self.serial_lock:
            self.port.write(command)
            ack = self.port.read(1)
            ### This loop discards buffered frames until an ACK header is reached ###
            while ord(ack) != 20:
                ack = self.port.read(1)
            else:
                ack += self.port.read(3)
            ### Check ACK crc8 ###
            crc8 = self.crc8(ack[:3])
            if crc8 == ack[3]:
                ### Check if ACK or NACK ###
                if ack[2] == 0:
                    # print("Command acknowledged")
                    return True
                else:
                    print("Command not acknowledged")
                    return False
            else:
                print("Error in ACK checksum")
                return False

    def run(self):
        ### Get frame and print it ###
        temps, frame = self.get_thermals()
        #frame = np.array(frame).astype(np.uint8)
        #frame = cv2.resize(frame, (600,600))
        #cv2.imshow("THERMAlL",frame)
        #key = cv2.waitKey(1)
        return temps, frame

    def stop(self):
        ### Deactivate USB VCP output and close port ###
        self.send_command(self.deactivate_command)
        self.port.close()


detector = dlib.get_frontal_face_detector()
EMISSIVITY = 0.98

def formula(distance):
    return 0.13010564597072907 * np.log(-distance + 200) + 0.3005487500000036
'''
if __name__ == "__main__":
    evo = EvoThermal()
    try:
        while True:
            temps, frame = evo.run()
            ret, frame_ = cam.read()

            frame_ = cv2.resize(cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY), (256,256))
            rects = haar.detectMultiScale(frame_, scaleFactor = 1.05, minNeighbors = 5) # detector(frame_, 1)
            # print(rects)
            if(len(rects) > 0) : print("Face detected")
            for rect in rects:
                (x,y,w,h) = rect  # .left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                x = int(x/8)
                y = int(y/8)
                w = int(w/8)
                h = int(h/8)
                t = temps[y:y+h, x: x +w]
                # get top 5 percentile to filter out the outliers
                _percentile = np.percentile(t, 95)
                where = np.where(t > _percentile)
                # print(t[where])
                mean_ = np.mean(t[where])
                # mean_ = np.max(t)
                # mean = np.mean(np.where(t > np.percentile(t, 25)))
                ratio = formula(50)/formula(0)
                mean = mean_/ratio
                # return mean
                #mean = mean ** 4
                #mean = (mean/EMISSIVITY) **  0.25
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),1)
                print(mean)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            # print(temps)
            if(key == ord("q")):
                evo.stop()
    except KeyboardInterrupt:
        evo.stop()
'''
