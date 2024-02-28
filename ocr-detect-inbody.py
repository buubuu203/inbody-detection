import cv2
import math
import numpy as np
import pandas as pd
import string
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# Install necessary packages
# !pip install "paddleocr>=2.0.1"
# !pip install paddlepaddle

# Check paddlepaddle version
# !pip show paddlepaddle



# Define functions
def resize_img(img, input_size=600):
    """
    Resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img

def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character, and a single digit
    equal to half the length of Chinese characters.
    """
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    
    for c in s:
        if str(c) in string.ascii_letters or str(c).isdigit() or str(c).isspace():
            en_dg_count += 1
        elif str(c).isalpha():
            count_zh += 1
        else:
            count_pu += 1
    
    return s_len - math.ceil(en_dg_count / 2)

def draw_ocr(image, boxes, txts, scores):
    if scores is None:
        scores = [1] * len(boxes)

    img = np.array(image)
    arrayVal=[]
    for idx in range(len(boxes)):
        if scores[idx] < 0.5 or math.isnan(scores[idx]):
            continue
        coordinates = np.array(boxes[idx][0], dtype=np.int32).reshape(-1, 2)
        img = cv2.polylines(img, [coordinates], isClosed=True, color=(255, 0, 0), thickness=2)
        # img = resize_img(img, input_size=600)  # Placeholder for resize_img
    
    txt_img = []
    for index, txt in enumerate(txts):
        if txt is not None:
            arrayVal.append([index,txt,scores[index]])
    
    # Check if txt_img is not empty before concatenating
    if txt_img:
        txt_img_concatenated = np.concatenate(txt_img, axis=1)
        img = np.concatenate([img, txt_img_concatenated], axis=1)

    return img,arrayVal

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load image
img_path = './inbody_test.png'
image = Image.open(img_path).convert('RGB')

# Perform OCR
result = ocr.ocr(img_path, cls=True)

# Process OCR results
boxes = []
txts = []
scores = []

for value in result[0]:
    scores.append(value[1][1])
    txts.append(value[1][0])
    boxes.append(value)

# Set your desired drop_score value
drop_score = 0.5

# Draw OCR results on image
im_show, arrayValue = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

# Create DataFrame from arrayValue
df = pd.DataFrame(arrayValue, columns=['Index', 'Value', 'Score'])
df.set_index('Index', inplace=True)

# Select specific indices and save to CSV
selected_indices = [6, 7, 8, 9, 15, 20, 24, 30, 36, 152, 159, 170, 153, 160, 172, 195, 202, 208, 196, 203, 209, 171, 179, 186, 154, 161, 168, 155, 162, 174, 197, 204, 210, 198, 205, 211, 173, 180, 187, 16]
filtered_data = df.loc[selected_indices]['Value'].values
print(filtered_data)
df = pd.DataFrame([filtered_data], columns =['Name', 'Height', 'Weight', 'Gender & Test Time', 'Total body water', 'Protein', 'Minerals', 'Body Fat Mass', 'Weight', 'Left hand - Lean (Kg)', 'Left hand - Lean (%)','Left hand - Lean (Status)','Right hand - Lean (Kg)', 'Right hand - Lean (%)', 'Right hand - Lean (Status)','Left leg - Lean (Kg)', 'Left leg - Lean (%)', 'Left leg - Lean (Status)', 'Right leg - Lean (Kg)', 'Right leg - Lean (%)','Right leg - Lean (Status)','Abs - Lean (Kg)', 'Abs - Lean (%)', 'Abs - Lean (Status)','Left hand - Fat (Kg)', 'Left hand - Fat (%)','Left hand - Fat (Status)','Right hand - Fat (Kg)', 'Right hand - Fat (%)', 'Right hand - Fat (Status)','Left leg - Fat (Kg)', 'Left leg - Fat (%)', 'Left leg - Fat (Status)', 'Right leg - Fat (Kg)', 'Right leg - Fat (%)','Right leg - Fat (Status)','Abs - Fat (Kg)', 'Abs - Fat (%)', 'Abs - Fat (Status)', 'Inbody Score'])
df.to_csv('./filtered.csv' )
