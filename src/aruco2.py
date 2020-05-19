import sys
import os
import numpy as np
import cv2
import cv2.aruco as aruco
import time as t
import math
import matplotlib.pyplot as plt

#get markers from https://chev.me/arucogen/
#if video taken on phone reduce resolution

#global variables
cam = 'arucotest2.mp4'      #enter the cam number or the video file here
start_time = t.time()
savetvec = [[0,0,0,0]]
savervec = [[0,0,0,0]]
savethetaR = [0,0]
dist = [0]
x = []
y = []
dx = []
dy = []
fx = open('wldx.txt','a+')
fy = open('wldy.txt','a+')

#******************************************************************************
def marker_pose(marker_id,data):
    #enter all marker pose data here
    x = [0,0,-350,0]
    y = [0,0,270,270]
    theta = [0,0,0,0]
    out = [x[marker_id],y[marker_id],theta[marker_id]]

    return out[data]
#******************************************************************************

def calibrate():

    cap = cv2.VideoCapture(cam)
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # checkerboard of size (9 x 7) is used
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (9,7), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'

        # Display the resulting frame
        cv2.imshow('Calibration',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    #create a file to store data
    from lxml import etree
    from lxml.builder import E

    global fname
    with open(fname, "w") as f:
        f.write("{'ret':"+str(ret)+", 'mtx':"+str(list(mtx))+', "dist":'+str(list(dist))+'}')
        f.close()


#test wheater already calibrated or not
path = os.path.abspath('..')
fname = path + "/res/calibration_parameters.txt"
print(fname)
try:
    f = open(fname, "r")
    f.read()
    f.close()
except:
    calibrate()


cap = cv2.VideoCapture(cam)

#importing aruco dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)    #aruco directory

#calibration parameters
f = open(fname, "r")
ff = [i for i in f.readlines()]
f.close()
from numpy import array
parameters = eval(''.join(ff))
mtx = array(parameters['mtx'])
dist = array(parameters['dist'])

# Create absolute path from this module
file_abspath = os.path.join(os.path.dirname(__file__), 'Samples/box.obj')

tvec = [[[0, 0, 0]]]
rvec = [[[0, 0, 0]]]

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250 )          #aruco directory
markerLength = 9.0   # Here, our measurement unit is centimetre.         lenght of side of marker
parameters = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (480, 640))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if np.all(ids != None):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)


        for i in range(0, ids.size):
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 5)

            # show translation vector on the corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str([round(i,5) for i in tvec[i][0]])
            position = tuple(corners[i][0][0])


            savetvec.append([ids[i][0],tvec[i][0][0],tvec[i][0][1],tvec[i][0][2]])
            savervec.append([ids[i][0],rvec[i][0][0],rvec[i][0][1],rvec[i][0][2]])

            rot_mat,_ = cv2.Rodrigues(rvec[i][:][:])

            c2 = (rot_mat[0][0]**2 + rot_mat[0][1]**2)**(1/2)
            thetaR = math.atan2(-rot_mat[0][2],c2)
            savethetaR.append([ids[i][0],thetaR])

            #print(dist)
            Z = savetvec[-1][3]
            X = savetvec[-1][1]
            thetaR_deg = thetaR*180/math.pi

            #Marker data
            index = savetvec[-1][0]
            Xm = marker_pose(index,0)
            Ym = marker_pose(index,1)

            Yw = -(Z*math.cos(thetaR) - X*math.sin(thetaR)) + Ym
            Xw = -(X*math.cos(thetaR) - Z*math.sin(thetaR)) + Xm
            thetaW = thetaR_deg

            print("World Coordinates|MarkerID:",savetvec[-1][0]," Xw =",Xw, "Yw =",Yw,"phiW =",thetaW,"\n")

            #To plot
            x.append(Xw)
            y.append(Yw)
            
            fx.write('%d\n'% Xw)
            fy.write('%d\n'% Yw)
            
        aruco.drawDetectedMarkers(frame, corners)
    else:
        tvec = [[[0, 0, 0]]]
        rvec = [[[0, 0, 0]]]

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
plt.plot(x, y,'k.-')
plt.axis([-400, 100, 0, 300])    #change axis limits to get a better plot
plt.xlabel('cm')
plt.ylabel('cm')
plt.savefig('botplot.png')

fx.truncate(0)
fy.truncate(0)
fx.close()
fy.close()
cap.release()
cv2.destroyAllWindows()

