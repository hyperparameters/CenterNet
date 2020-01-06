from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
detection_json = dict()
outpath = "results"
show_class =[]

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

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
    outpath = outpath + inpath
    if not os.path.isdir(outpath):
      os.makedirs(outpath)
      
    for (image_name) in image_names:
      ret = detector.run(image_name)
      result = ret["results"]
      image = cv2.imread(image_name)
      name = os.path.basename(image_name).split(".")[0]
      
      create_json(result,name)
      image = create_bbox(result,image)
      save_image(image,name)
      
      
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

    save_json()

def create_bbox(result,image):
  if len(show_class)==0:
    show_class = range(1,opt.num_classes)  
  for i in show_class:
    for bbox in result[i]:
      if bbox[4]>= opt.vis_thresh:
        cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color[i],2)
  return image
  
def save_image(image,name):
  cv2.imwrite(os.path.join(outpath,name+".jpg"),image)
  print("imaged saved : {}".format(name))

def create_json(result,name):

  if len(show_class)==0:
    show_class = range(1,opt.num_classes)
  det = []  
  for i in show_class:
    for bbox in result[i]:
      det.append({bbox[4] : bbox[:4].tolist()})

  #add detections to dictionary
  detections_json[name] = det
 
def save_json():
  with open("detections.json","w") as f:
    f.write(json.dumps(detections_json))
    print("json results saved")
    
if __name__ == '__main__':
  opt = opts().init()
  if not os.path.isdir(outpath):
    os.mkdir(outpath)
  demo(opt)
