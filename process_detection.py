import os 
from nms_poly import non_max_suppression_poly
import numpy as np
import sys 
#from imutils.object_detection import non_max_suppression
data_dir=sys.argv[1]
img_dir = os.path.join(data_dir, "images")
result_dir="results"


for result_file in os.listdir(result_dir):
    print(result_file)
    result_path=os.path.join(result_dir,result_file)
    with open(result_path,"r",encoding="utf-8") as f:
        data=f.readlines()
        data=[i.strip() for i in data]
        
        data=[i.split(",") for i in data]
        data=[[int(j) for j in i[:-1]]+[round(float(i[-1]),3)] for i in data] 
        #print(len(data))
        conf_score=[i[-1] for i in data]
        rects=[i[:-1] for i in data]
        rects=[[rects[i][j:j+2] for j in range(0,len(rects[i]),2)] for i in range(len(rects))]
        #print(rects)
        #conf_score=np.array(conf_score)
        rects=np.array(rects)
        #print(conf_score)
        keep=non_max_suppression_poly(rects,conf_score,0.1)
        #print(keep)
        #get all the detection results with keep True
        data=[data[i] for i in range(len(data)) if keep[i]==True]
        with open(result_path,"w",encoding="utf-8") as f:
            for line in data:
                f.write(",".join([str(i) for i in line])+"\n")



#print(data)