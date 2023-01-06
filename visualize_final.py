import os
import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# define const
font_path = "./arial.ttf"
prediction_path = "submision/"
input_path = "datasets/uaic2022_private_test/images/"
output_path = "demo/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
max_image = -1
count_image = 0

files = glob.glob(input_path + "*")

def drawBbox(box, image):
    font = ImageFont.truetype(font_path, 16)
    pdraw = ImageDraw.Draw(image)
    polygon_pts = []
    for i in range(0, 8, 2):
        polygon_pts.append((int(box[i]), int(box[i + 1])))
    pdraw.polygon(polygon_pts, fill=None, outline=(36, 255, 12, 255))
    text=box[-1]
    pdraw.text(polygon_pts[0],text , (36,255,12), font=font)

for file in files:
    
    if (count_image == max_image):
        print("Max image exceeded. Stopped!")
        break
    if (count_image != -1):
        count_image += 1
    
    filename = os.path.basename(file)
    
    f = open(prediction_path + filename[:-4] + ".txt", "r", encoding="utf8")
    bboxes1 = f.read()
    f.close()
    bboxes = bboxes1.split("\n")
    
    image = Image.open(file)
    
    for bbox in bboxes:
        box = bbox.split(",")
        if len(box) < 8:
            continue
        drawBbox(box, image)
    #convert to RGB
    image = image.convert("RGB")
    image.save(os.path.join(output_path , filename))