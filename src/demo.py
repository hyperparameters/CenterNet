from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import numpy as np
from opts import opts
from detectors.detector_factory import detector_factory
import bg_remove
import time

detection_json = dict()
outpath = "results"
show_class =[]

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
debug = True

def print_time(start_time,process):
  
  if debug:
    t2 = time.time()
    print(process + " : " + str(t2-start_time))

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  #opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  global outpath

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    #find input directory
    inpath = os.path.dirname(image_names[0])

    #create output at same directory
    outpath = os.path.join(outpath,inpath)
    if not os.path.isdir(outpath):
      os.makedirs(outpath)
    try:
      image_names = sorted(image_names , key = lambda x : int(os.path.basename(x).split(".")[0]))
    except:
      print("cannot sort")  
    for (image_name) in image_names:
    
      start_time = time.time()
      image = cv2.imread(image_name)
      fgmask = bg_remove.get_fgmask(image)
      image_processed = bg_remove.sub_bg(image,fgmask)
      print_time(start_time,"activity")
      
      start_time = time.time()
      ret = detector.run(image_name)
      print_time(start_time,"detection")

      result = ret["results"]
      image = cv2.imread(image_name)
      name = os.path.basename(image_name).split(".")[0]
      
      create_json(result,name)
      image = create_bbox(result,image)
      #save_image(image,name)
      #show_img(image,"org")
      #show_img(image_processed,"activity")
      
      show_imgs([image],1,1)
      
      time_str = ''
      #for stat in time_stats:
      #  time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      #print(time_str)

    save_json()

def create_bbox(result,image):
  global show_class
  if len(show_class)==0:
    show_class = range(1,opt.num_classes)  

  colors = np.random.randint(0,255,size=(len(show_class),3))
  #print(colors.shape)
  #print(co)
  for i in show_class:
    for bbox in result[i]:
      if bbox[4]>= opt.vis_thresh:
        #print(tuple(colors[i]))
        #color = tuple(colors[i])
        
        cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(217, 1, 250),2)
        cv2.rectangle(image,(int(bbox[0]),int(bbox[1]-10)),(int(bbox[2]),int(bbox[1])),(217, 1, 250),-1)
        cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
  return image
  
def save_image(image,name):
  outpath_image = os.path.join(outpath,name+".jpg")
  cv2.imwrite(outpath_image,image)
  print("imaged saved : {}".format(outpath_image))

def create_json(result,name):
  global show_class,detection_json
  if len(show_class)==0:
    show_class = range(1,opt.num_classes)
  det = []  
  for i in show_class:
    for bbox in result[i]:
      det.append({str(bbox[4]) : bbox[:4].tolist()})

  #add detections to dictionary
  detection_json[str(name)] = det


def show_img(img,win_name = ""):
  h,w = img.shape[:2]
  img = cv2.resize(img,(w//3,h//3))
  cv2.imshow(win_name,img)
  if cv2.waitKey(1)==ord("q"):
    cv2.destroyAllWindows()
    exit()

def show_imgs(list_imgs,rows,cols,size=(720,1080)):
  canvas = np.zeros(size)
  
  h,w = list_imgs[0].shape[:2]
  col_imgs = []
  num_imgs = len(list_imgs)
  #print(rows,cols)
  #import pdb
  #pdb.set_trace()
  tot = rows * cols 
  
  black = [np.zeros((h,w,3))] * (tot - num_imgs)
  list_imgs.extend(black)

  for i in range(rows):
    if i*cols>num_imgs:
      break

    elif (i+1)*cols >= num_imgs:
      
      col_imgs.append(np.hstack(list_imgs[i*cols:]))      
    else:
      col_imgs.append(np.hstack(list_imgs[i*cols:(i+1)*cols]))   
     
  canvas = np.vstack(col_imgs)
  canvas = canvas.astype("uint8")
  show_img(canvas)

def save_json():
  with open("detections.json","w") as f:
    f.write(json.dumps(detection_json))
    print("json results saved")
    
if __name__ == '__main__':

  img1 = np.ones((320,480))
  img2 = np.ones((320,480))
  img3 = np.ones((320,480))
  
  opt = opts().init()
  
  #show_imgs([img1,img2,img3],2,2)
  
  if not os.path.isdir(outpath):
    os.mkdir(outpath)
  demo(opt)


