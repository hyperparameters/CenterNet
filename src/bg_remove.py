import cv2
import numpy as np
import os

src = "../store2_cam7/"
min_area = 2000
delta_thresh = 100

backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()

fgMask = backSub.apply(bg)

kernel_erode = np.ones((3,3),np.uint8)
kernel_dilate = np.ones((7,7),np.uint8)

def get_fg_mask(frame)
    #frame= cv2.imread(src+frame_name)
    #frame = cv2.resize(frame, (1080,720))
    
    # bg model
    frame = cv2.GaussianBlur(frame,(17,17),0)
    fgMask = backSub.apply(frame,1)


    thresh = cv2.threshold(fgMask, delta_thresh, 255,cv2.THRESH_BINARY)[1]

    thresh_org = thresh.copy()

    eroded = cv2.erode(thresh, kernel_erode, iterations=2)
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=2)

    thresh = dilated
    cnts,he = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    activity = False
    # loop over the contours
    for i,c in enumerate(cnts):
        # if the contour is too small, ignore it
        #print(cv2.contourArea(c))
        if cv2.contourArea(c) < min_area:
            continue
        activity = True
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

    #h,w = frame.shape[:2]
    #canvas = np.zeros((h,w))


    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #canvas[:h//2,:w//2] = cv2.resize(frame,(w//2,h//2))
    #canvas[:h//2,w//2:] = cv2.resize(thresh_org,(w//2,h//2))


    #canvas[h//2:,:w//2] = cv2.resize(eroded,(w//2,(h//2)))
    #canvas[h//2:,w//2:] = cv2.resize(dilated,(w//2,(h//2)))

    #if frame_no%200==0:
#     cv2.imwrite("fg_masks/"+frame_name,thresh)
    return dilated

def sub_bg(frame,fgmask):
    fg_mask_exp  = cv2.dilate(fgmask,np.ones((91,91),dtype="uint8"),10)
    
    bg_deg = frame.copy()
    fg_enh = frame.copy()

    bg_deg  = cv2.convertScaleAbs(bg_deg, alpha=0, beta=100)
    fg_enh  = cv2.convertScaleAbs(fg_enh, alpha=1.5, beta=0)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # fg_enh = clahe.apply(fg_enh)

    merge = cv2.bitwise_and(bg_deg,cv2.bitwise_not(fg_mask_exp)) + cv2.bitwise_and(fg_enh,fg_mask_exp)

    return merge
