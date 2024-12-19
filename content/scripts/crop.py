import os
import cv2
import subprocess
# import numpy as np

IMGPATH = '../GenTmp/samples/'
TGTPATH = '../Images/tgt.jpg'

if __name__ == '__main__':
    files = [file for file in os.listdir(IMGPATH) if file.endswith('.png')]
    files = files[-6:]
    imgs = []

    for file in files:
        img = cv2.imread(IMGPATH+file)
        img = cv2.resize(img, (224,224))
        cv2.imwrite(IMGPATH+file, img)
        imgs.append(img)
        