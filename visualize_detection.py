import os 
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

results_detect_dir="results"
img_dir="datasets/uaic2022_private_test/images"
visualize_dir="visualize2_nms5"
if not os.path.exists(visualize_dir):
    os.makedirs(visualize_dir)
for txt_path in glob.glob(os.path.join(results_detect_dir,"*.txt")):
    img_path=os.path.join(img_dir,os.path.basename(txt_path).replace(".txt",".jpg"))
    img=cv2.imread(img_path)
    with open(txt_path,"r",encoding="utf-8") as f:
        data=f.readlines()
        data=[i.strip() for i in data]
        for line in data:
            line=line.split(",")
            points=[int(i) for i in line[:-1]]
            points=np.array(points).reshape(-1,2)
            cv2.polylines(img,[points],True,(0,255,0),2)
    visualize_path=os.path.join(visualize_dir,os.path.basename(txt_path).replace(".txt",".jpg"))
    print(visualize_path)
    cv2.imwrite(visualize_path,img)
    