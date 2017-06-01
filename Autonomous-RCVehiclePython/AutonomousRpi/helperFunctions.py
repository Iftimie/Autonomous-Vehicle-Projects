import cv2
import numpy as np
import math
import random

def translateImage(img, offsetx,offsety):
    translation_matrix = np.float32([[1, 0, offsetx], [0, 1, offsety]])
    return cv2.warpAffine(img, translation_matrix, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

def getLargestRect(cols,rows,angle,type):
    rotatedAngleDeg = math.fmod(angle,180.)
    if rotatedAngleDeg < 0.:
        rotatedAngleDeg+=360.
        rotatedAngleDeg = math.fmod(rotatedAngleDeg,180.)

    if rotatedAngleDeg == 0. or rotatedAngleDeg ==180.:
        return [0.,0.,rows,cols]

    if rotatedAngleDeg > 90.:
        rotatedAngleDeg = 90. - (rotatedAngleDeg-90.)

    rotateAngle = (rotatedAngleDeg*math.pi)/180.
    sinRotAng = math.sin(rotateAngle)
    cosRotAng = math.cos(rotateAngle)
    tanRotAng = math.tan(rotateAngle)

    x1 = sinRotAng*rows
    y1 = 0.

    x2 = cosRotAng*cols + x1
    y2 =sinRotAng*cols

    x3 = x2-x1
    y3 =y2+cosRotAng*rows

    x4 = 0.
    y4 = y3-y2
    midx = x2/2.
    midy = y3/2.

    imgAngle = math.atan(rows/cols)
    imgRotAngle = math.atan(cols/rows)
    tanImgAng = math.tan(imgAngle)
    tanImgRotAng = math.tan(imgRotAngle)

    ibx1 = midy/tanImgAng +midx
    ibx2 = midy * tanImgAng + midx

    a = y2 / x3
    b = tanRotAng * -x1
    c = -rows / cols
    d = tanImgAng * ibx1

    ix1 = (d - b) / (a - c)
    iy1 = a * ix1 + b

    c = -cols / rows
    d = tanImgRotAng * ibx2

    ix2 = (d - b) / (a - c)
    iy2 = a * ix2 + b

    radx1 = math.fabs(midx - ix1)
    rady1 = math.fabs(midy - iy1)
    radx2 = math.fabs(midx - ix2)
    rady2 = math.fabs(midy - iy2)

    area1 = radx1 * rady1
    area2 = radx2 * rady2

    rect1 = [int(midx - radx1),int(midy - rady1),int(radx1 * 2.),int(rady1 * 2.)]
    rect2 = [int(midx - radx2),int(midy - rady2),int(radx2 * 2.),int(rady2 * 2.)]
    if area1 >area2:
        return rect1
    else:
        return rect2


def rotateAndCrop(img,angle):
    if angle == 0:
        return img
    center = tuple(np.array(img.shape)[:2] / 2)
    rot_mat = cv2.getRotationMatrix2D(center=center,angle=angle,scale=1.0)
    img = cv2.warpAffine(src=img, M=rot_mat, dsize=(img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR) #(cols,rows)
    return img

def changeBrighness(img, value):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:, :, 2] =img[:, :, 2] + value
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def equalizeDistribution(X,Y):
    maxim = Y.max()
    minim = Y.min()
    bins = 11
    step_range = (maxim - minim) / bins
    distribution = np.zeros(bins)
    for i in range(Y.shape[0]):
        for x in range(bins):
            if Y[i] >= minim + x * step_range and Y[i] <= minim + (x + 1) * step_range:
                distribution[x] += 1
                break

    maxDistribution = distribution.max()
    for x in range(bins):
        difference = int(maxDistribution - distribution[x])
        for i in range(difference):
            random_index = int(random.uniform(0, len(Y)))
            while Y[random_index] <= minim + x * step_range or Y[random_index] >= minim + (x + 1) * step_range:
                random_index = int(random.uniform(0, len(Y)))

            newY = np.array([Y[random_index]])
            Y = np.concatenate((Y, newY), axis=0)
            newX = np.array([X[random_index]])
            X = np.concatenate((X, newX), axis=0)

    distribution = np.zeros(bins)
    for i in range(Y.shape[0]):
        for x in range(bins):
            if Y[i] >= minim + x * step_range and Y[i] <= minim + (x + 1) * step_range:
                distribution[x] += 1
                break
    return X,Y

def loadDataFromVideo():
    capture = cv2.VideoCapture("/home/iftimie/out" + str(0) + ".avi")
    file = open("/home/iftimie/def" + str(0) + ".txt", "r")
    Y = []
    X = []
    for line in file:
        Y.append([float(line.replace("\n", ""))])
        ret, frame = capture.read()
        if ret == False:
            break;
        frame = frame[230:480, 0:640]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (128, 64))
        X.append(np.squeeze(np.asarray(frame.ravel())))

    Y = np.array(Y)
    X = np.array(X)
    return X,Y

def saveData(X,Y):
    np.save("X.npy",X)
    np.save("Y.npy",Y)

def loadDataFromBinary():
    X = np.load("X.npy")
    Y = np.load("Y.npy")
    return X,Y