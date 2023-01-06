import numpy as np
import os
import json
import cv2
from tqdm import tqdm
import sys 
sys.path.insert(0, 'vietocr')
sys.path.append('vedastr')
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vedastr.runners import InferenceRunner
from vedastr.utils import Config
from PIL import Image


def check_text(text):
    if len(text)==1:
        if text[0] in["0","1","2","3","4","5","6","7","8","9","/","-"]:
            return True
        else:
            return False
    return True
def remove_space(text):
    text = text.replace(" ", "")
    return text
    
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
def remove_sdt(text):
    count=0
    #remove . - 
    text=text.replace(".","")
    text=text.replace("-","")
    for i in range(len(text)):
        if text[i] in ["0","1","2","3","4","5","6","7","8","9"]:
            count+=1
    if count>=2 and count<=8 and count==len(text) and text[0]=="0":
        return True
    return False
            
vedastr_path="vedastr/"
vietocr_path="vietocr/"
weights_path="weights/"
#cofig vedastr
cfg_path = os.path.join(vedastr_path, 'configs/tps_resnet_bilstm_attn.py')
gpus = "0"
checkpoint=os.path.join(weights_path,"vedastr.pth")
cfg = Config.fromfile(cfg_path)
deploy_cfg = cfg['deploy']
common_cfg = cfg.get('common')
deploy_cfg['gpu_id'] = gpus.replace(" ", "")
model1 = InferenceRunner(deploy_cfg, common_cfg)
model1.load_checkpoint(checkpoint)



#config vietocr

config = Cfg.load_config_from_file("vietocr/config/resnet-transformer.yml")
config['weights'] = os.path.join(weights_path,"vietocr.pth")
config['device'] = 'cuda:0'
config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
#print(config)
model = Predictor(config)

detect_dir="results"
data_dir=sys.argv[1]
images_dir=os.path.join(data_dir,"images")
output_dir="submision"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for file in os.listdir(detect_dir):
    labels_path=os.path.join(detect_dir,file)
    with open(labels_path) as f:
        lines=f.readlines()
        lines=[line.strip() for line in lines]
    image_path=os.path.join(images_dir,file[:-4])+".jpg"
    image=cv2.imread(image_path)
    
    print(image_path)
    new_lines=[]
    for line in lines:

        line=line.split(",")
        pts=[[int(line[0]),int(line[1])],[int(line[2]),int(line[3])],[int(line[4]),int(line[5])],[int(line[6]),int(line[7])] ]
        cropped=perspective_transform(image,pts)
        #print(cropped.shape)
        width,height,dim=cropped.shape
        cropped_vedastr=cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_vietocr=Image.fromarray(cropped)
        text1,score1=model1(cropped_vedastr)
        text2,score2=model.predict(cropped_vietocr,return_prob=True)
        text1,score1=text1[0],score1[0]
        # print("vedastr",text1,score1)
        # print("vietocr",text2,score2)
        if width<10:
            continue
        if max(score1,score2)<0.5:
            continue
        if score1>0.7 and score2<0.5:
            text_predict=text1
        elif score2>0.7 and score1<0.5:
            text_predict=text2
        else:
            if text1==text2:
                text_predict=text1
            if max(score1,score2)>=0.7:
                text_predict=text1 if score1>score2 else text2
            else:
                continue        

        text_predict=text_predict.replace(" ","")
        #print(text_predict)
        if remove_sdt(text_predict)==True:
            continue
        new_line=line[0]+","+line[1]+","+line[2]+","+line[3]+","+line[4]+","+line[5]+","+line[6]+","+line[7]+","+text_predict
        new_lines.append(new_line)
    output_path=os.path.join(output_dir,file.replace(".jpg",""))
    with open(output_path,"w") as f:
        f.write("\n".join(new_lines))
#         #exit()

# # zip -r -j predicted.zip predicted/
# #zip -r -D predicted.zip *