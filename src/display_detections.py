import cv2
import json
import numpy as np

#from head_pose_helper import HeadPoseDetector
#headposedetector = HeadPoseDetector()


with open("centernet_det/nike_frames_1_pre_coco.json","r") as f:
#with open("centernet_det/nike_frames_coco_mhp_40.json","r") as f:
    detections_json = json.load(f)

with open("centernet_det/nike_frames_1_wider.json","r") as f:
    detections_face = json.load(f)

with open("nike_frames_yaw_angles_test.json","r") as f:
    gazes = json.load(f)

face_thresh = 0.3
human_thresh = 0.4

cap = cv2.VideoCapture("../../deep_yolo/test_video_1.mp4")
frame_no_vid = 1



def change_thresh(thresh,val):
  return thresh+ val

def draw_thresh(im,frame_no):
    global face_thresh,human_thresh
    #draw thresh:
    cv2.putText(im,"face thres: {:.2f}".format(face_thresh),(25,550),0,1,(0,200,200),2)    
    cv2.putText(im,"human thres: {:.2f}".format(human_thresh),(25,600),0,1,(0,200,200),2) 
    cv2.putText(im,"frame no: {}".format(str(frame_no)),(25,650),0,1,(0,200,200),2) 
    
    return im

#for i in range(200):
#   _,frame = cap.read()
#   frame_no_vid+=1

#for frame_no,det in detections.items():
while 1:

    #im = cv2.imread("nike_out/{}.jpg".format(frame_no))
    ok,im = cap.read()
    
    
    try:
        det = detections_json[str(frame_no_vid)]
        
    except:
        cv2.imshow("",im)
        key= cv2.waitKey(1)
        continue
    
    try :
        det_face = detections_face[str(frame_no_vid)]
        gaze = gazes[str(frame_no_vid)]
    except:
        pass



    frame_no_vid +=1
    if not ok :
        break

    h,w = im.shape[:2]
    humans = det["1"]
           
    for data in humans:
       conf = list(data.keys())[0]
       [x1,y1,x2,y2] = data[conf]
                       
       if float(conf) > human_thresh:
          #boxs.append([x1,y1,x2-x1,y2-y1])
          cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,250),2)
          #cv2.rectangle(OD_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,250,0), 2)
          cv2.putText(im,"{:.2f}".format(float(conf)),(int(x1),int(y2/2)),0,0.8,(0,0,250),2)
                 
    faces = det_face["2"]

    for data in faces:
       conf = list(data.keys())[0]
       [x1,y1,x2,y2] = data[conf]
                       
       if float(conf) > face_thresh:
          #boxs.append([x1,y1,x2-x1,y2-y1])
          x1,y1,x2,y2 = max(0,x1-20),max(0,y1-20),min(w,x2+20),min(h,y2+10)
          cv2.rectangle(im, (int(x1),int(y1)), (int(x2),int(y2)), (0,250,50),2)
          #cv2.rectangle(OD_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,250,0), 2)
          cv2.putText(im,"{:.2f}".format(float(conf)),(int(x1),int(y1+60)),0,0.8,(0,200,10),2)

       

    h2,w2 = int(2*h/3),int(2*w/3)
    im = cv2.resize(im,(w2,h2))
    display = im.copy()

    im = draw_thresh(im,frame_no_vid)
    cv2.imshow("",im)
    key= cv2.waitKey(1)
    

    if key==ord("t"):
       
       k = cv2.waitKey(0)
       #cv2.destroyWindow("")
       if k == ord("f"):
         while 1:
           dis = display.copy()
           dis = draw_thresh(dis,frame_no_vid)
           cv2.imshow("",dis)
           k = cv2.waitKey(0)
           if k==ord("i"):
             face_thresh = change_thresh(face_thresh,0.01)
             print(face_thresh)
           if k==ord("u"):
             face_thresh = change_thresh(face_thresh,-0.01)
             print(face_thresh)
           if k==ord("d"):
             break
           
       if k == ord("h"):
         while 1:
           dis = display.copy()
           dis = draw_thresh(dis,frame_no_vid)
           cv2.imshow("",dis)

           if cv2.waitKey(0)==ord("i"):
             human_thresh = change_thresh(human_thresh,0.01)
             print(human_thresh)
           if cv2.waitKey(0)==ord("u"):
             human_thresh = change_thresh(human_thresh,-0.01)
             print(human_thresh)
           if cv2.waitKey(0)==ord("d"):
             break
              
          
      

    

    

 
    # pause
    if key==ord(" "):
       cv2.waitKey(0)
       
    if key==ord("a"):
       cap.set(1,frame_no_vid-60)
       frame_no_vid = frame_no_vid-60
    
    if key==ord("d"):
       cap.set(1,frame_no_vid+60)
       frame_no_vid = frame_no_vid+60

    if key ==ord("q"):
        cv2.destroyAllWindows()
        break
    
