import numpy as np
import os
import json
import cv2
from tqdm import tqdm

from vedastr.runners import InferenceRunner
from vedastr.utils import Config
from PIL import Image

def order_points(pts):
    if isinstance(pts, list):
        pts = np.asarray(pts, dtype='float32')
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped






cfg_path = "configs/tps_resnet_bilstm_attn.py"
gpus = "0"
checkpoint="weights/tps_resnet_bilstm_attn/best_norm.pth"
cfg = Config.fromfile(cfg_path)
deploy_cfg = cfg['deploy']
common_cfg = cfg.get('common')
deploy_cfg['gpu_id'] = gpus.replace(" ", "")
runner = InferenceRunner(deploy_cfg, common_cfg)
runner.load_checkpoint(checkpoint)


detect_dir="results"
images_dir="uaic2022_public_valid/uaic2022_public_valid/images"
output_dir="submision"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for file in os.listdir(detect_dir):
    labels_path=os.path.join(detect_dir,file)
    with open(labels_path) as f:
        lines=f.readlines()
        lines=[line.strip() for line in lines]
    image_path=os.path.join(images_dir,file[:-4])
    image=cv2.imread(image_path)
    print(image_path)
    new_lines=[]
    for line in lines:
        #print(line)
        #print(type(line))
        line=line.split(",")
        pts=[[int(line[0]),int(line[1])],[int(line[2]),int(line[3])],[int(line[4]),int(line[5])],[int(line[6]),int(line[7])] ]
        #pts=np.asarray(pts)
        cropped=perspective_transform(image,pts)
        #get size
        print(cropped.shape)
        width,height,dim=cropped.shape
        cropped=cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        #pred_str, probs = runner()
        #cropped=Image.fromarray(cropped)
        text_predict,score=runner(cropped)
        #print(score)
        score=float(score[0])
        text_predict=text_predict[0]
        print(text_predict,score)
        if score<0.4 or width<10 :
            continue
        
        new_line=line[0]+","+line[1]+","+line[2]+","+line[3]+","+line[4]+","+line[5]+","+line[6]+","+line[7]+","+text_predict
        new_lines.append(new_line)
    output_path=os.path.join(output_dir,file.replace(".jpg",""))
    with open(output_path,"w") as f:
        f.write("\n".join(new_lines))
        #exit()

# zip -r -j predicted.zip predicted/
#zip -r -D predicted.zip *