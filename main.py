'''
Aluno: Alaf do Nascimento Santos
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 10, 200, markerImage, 1)

cv.imwrite("ArUco.png", markerImage)

#loading the image
img02 = cv.imread('naruto.jpg')
img02_rgb = cv.cvtColor(img02, cv.COLOR_BGR2RGB)

#Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250) 

# Initialize the detector parameters using default values
parameters =  cv.aruco.DetectorParameters_create()

# Get the limits of the image that will be inserted in the original one
[l,c,ch] = np.shape(img02_rgb)

# Source points are the corners of the image that will be warped
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])


capture = cv.VideoCapture(0)

print("Pressione 'espaço' para sair")
while True:
    k = cv.waitKey(30) 
    if k == 32: #sair se espaço for precionado
        capture.release()
        cv.destroyAllWindows()
        break
    elif k == -1:
        _, im_out = capture.read()

        # Detect the markers in the image
        markerCorners, markerIds, _ = cv.aruco.detectMarkers(im_out, dictionary, parameters=parameters)

        detections_img = cv.aruco.drawDetectedMarkers(im_out, markerCorners, markerIds)

        for mark in markerCorners:
            # Define the source and destiny point for calculating the homography
            # Destiny points are the corners of the marker
            pts_dst = np.array(mark[0])

            # Calculate Homography
            h, _ = cv.findHomography(pts_src,pts_dst)

            # Warp source image to destination based on homography
            warped_image = cv.warpPerspective(img02,h,(im_out.shape[1], im_out.shape[0]))

            # Prepare a mask representing region to copy from the warped image into the original frame.
            mask = np.zeros([im_out.shape[0], im_out.shape[1]], dtype=np.uint8)
            cv.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv.LINE_AA)

            # Erode the mask to not copy the boundary effects from the warping
            element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            mask = cv.erode(mask, element, iterations=3)

            # Copy the mask into 3 channels.
            warped_image = warped_image.astype(float)
            mask3 = np.zeros_like(warped_image)

            for i in range(0, ch):
                mask3[:, :, i] = mask / 255


            # Copy the masked warped image into the original frame in the mask region.

            masked_warped_image = cv.multiply(warped_image, mask3)
            masked_frame = cv.multiply(im_out.astype(float), 1-mask3)

            im_out = cv.add(masked_warped_image, masked_frame)

        newFrame = im_out.astype(np.uint8)
        cv.imshow('Press space to exit', newFrame)
    else:
        continue

