import numpy as np
import cv2

class Box:

    def __init__(self):
        x = 640/2
        y = 480-22

        w = 300
        h = 20

        self.lowerLeftCorner = np.array([x - w/2, y + h/2])
        self.upperRightCorner = np.array([x + w/2, y - h/2])

    def ClipLine(self,dimension,v0,v1,f_low,f_high):

        f_dim_low =0
        f_dim_high =0

        f_dim_low = (self.lowerLeftCorner[dimension] - v0[dimension]) / (v1[dimension] - v0[dimension])
        f_dim_high = (self.upperRightCorner[dimension] - v0[dimension]) / (v1[dimension] - v0[dimension])

        if f_dim_high<f_dim_low:
            aux = f_dim_high
            f_dim_high = f_dim_low
            f_dim_low = aux

        if f_dim_high < f_low:
            return False,0,0
        if f_dim_low > f_high:
            return False,0,0

        f_low = max(f_dim_low,f_low)
        f_high = min (f_dim_high,f_high)

        if f_low > f_high:
            return False,0,0

        return True,f_low,f_high

    def LineBoxIntersection(self,v0,v1):
        f_low = 0
        f_high = 1

        valueReturned,f_low,f_high= self.ClipLine (0,v0,v1,f_low,f_high)
        if valueReturned ==False:
            return None
        valueReturned, f_low, f_high = self.ClipLine(1, v0, v1, f_low, f_high)
        if valueReturned == False:
            return None
        b = v1-v0
        intersection = v0 + f_low * b

        return intersection


    def intersection(self,img,v0,v1):

        intersect = self.LineBoxIntersection(v0,v1)
        if intersect!=None:
            img = cv2.circle(img,(int(intersect[0]),int(intersect[1])),3,(0,0,0),10)

        return img