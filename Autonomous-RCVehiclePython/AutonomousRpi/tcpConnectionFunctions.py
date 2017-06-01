import socket
import struct
import numpy as np
import cv2

def createRequest(speed = 35,direction =0,steerValue = 6000):
    steering = (steerValue).to_bytes(2, byteorder='big')
    # value = struct.unpack('h',steering)[0]
    req = bytearray()
    req.append(speed)  # first is speed value #default
    req.append(direction)  # second is forward or backwards
    req.extend(steering)
    return  req

def createSocket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.0.106", 1234))
    return s

def getImageFromSocket(s,colorCode = cv2.IMREAD_GRAYSCALE,crop = True): #or cv2.IMREAD_COLOR
    imageSize = s.recv(4)
    imageSize = struct.unpack('i', imageSize)[0]
    imageBytes = b''
    while imageSize > 0:
        chunk = s.recv(imageSize)
        imageBytes += chunk
        imageSize -= len(chunk)

    data = np.fromstring(imageBytes, dtype='uint8')
    image = cv2.imdecode(data, colorCode)
    if crop:
        image = image[230:480, 0:640]
    return image