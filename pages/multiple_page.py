from PIL import Image
import re
import sys
from Preprocess import preprocess
import json
from operator import attrgetter
import numpy as np
from google.oauth2 import service_account

def merge(dict1, dict2):
    return dict1.update(dict2)

def google_ocr(api_key ,img_path):
    from google.cloud import vision
    import os

    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="pj_key.json"
    
    credentials = service_account.Credentials.from_service_account_info(api_key)

    client = vision.ImageAnnotatorClient(credentials= credentials)

    with open(img_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)


    response = client.text_detection(image=image)
    

  
    return response



class ExtendedAnnotation:
    def __init__(self, annotation):
        self.vertex = annotation.bounding_poly.vertices
        self.text = annotation.description
        self.avg_y = (self.vertex[0].y + self.vertex[1].y + self.vertex[2].y + self.vertex[3].y) / 4
        self.height = ((self.vertex[3].y - self.vertex[1].y) + (self.vertex[2].y - self.vertex[0].y)) / 2
        self.start_x = (self.vertex[0].x + self.vertex[3].x) / 2

    def __repr__(self):
        return '{' + self.text + ', ' + str(self.avg_y) + ', ' + str(self.height) + ', ' + str(self.start_x) + '}'

def get_extended_annotations(response):
    extended_annotations = []
    for annotation in response.text_annotations:
        extended_annotations.append(ExtendedAnnotation(annotation))

    # delete last item, as it is the whole text I guess.
    del extended_annotations[0]
    return extended_annotations

def get_threshold_for_y_difference(annotations):
    annotations.sort(key=attrgetter('avg_y'))
    differences = []
    for i in range(0, len(annotations)):
        if i == 0:
            continue
        differences.append(abs(annotations[i].avg_y - annotations[i - 1].avg_y))
    return np.std(differences)    

def group_annotations(annotations, threshold):
    annotations.sort(key=attrgetter('avg_y'))
    line_index = 0
    text = [[]]
    for i in range(0, len(annotations)):
        if i == 0:
            text[line_index].append(annotations[i])
            continue
        y_difference = abs(annotations[i].avg_y - annotations[i - 1].avg_y)
        if y_difference > threshold:
            line_index = line_index + 1
            text.append([])
        text[line_index].append(annotations[i])
    return text

def sort_and_combine_grouped_annotations(annotation_lists):
    grouped_list = []
    for annotation_group in annotation_lists:
        annotation_group.sort(key=attrgetter('start_x'))
        texts = (o.text for o in annotation_group)
        texts = ' '.join(texts)
        texts = re.sub(r'\s([-;:?.!](?:\s|$))', r'\1', texts)
        grouped_list.append(texts)
    return grouped_list


# print(text_extract('preprocess.jpg'))
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def read_json():
    f = open('field.json', encoding='utf-8')
    data = json.load(f)
    field = []
    for key in data.keys():
        field.append(key+":")
    return data, field

def rearrange_list(content ,field):
    new_list = []
    position = {}
    # print(type(field))
    for i in field: 
        pos = content.find(i)
        temp = {i: pos}
        merge(position, temp)
    position= dict(sorted(position.items(), key=lambda item: item[1]))
    for keys in position.keys():
        new_list.append(keys)
    # print(new_list)


    return new_list

def field_detect(content, field):
    temp = []
    last_item = len(field)
    # print(content)
    for i in content.split('\n'):
        temp.append(i)
    index = index_containing_substring(temp, field[last_item-1])
    # print(index)
    del temp[index+1:len(temp):1]
    temp.append(" end")
    # print(temp)
    real_content = ""
    
    for j in temp:
        real_content += j + " "
    # print(real_content)
    final_result = {}
    for i in range(0, last_item):
        if i < last_item-1:
            reg = re.compile(field[i]+r'(.*?)'+ field[i+1])
            result = reg.findall(real_content)
            # print(type(result))
            merge(final_result, {field[i]: result})
        if i == last_item-1: 
            reg = re.compile(field[i]+r'(.*?)end')
            result = reg.findall(real_content)
            merge(final_result, {field[i]: result})
    return final_result

def inference(api_key, file_path):
    # img = preprocess(file_path)
    data, field = read_json()
    # api_key = os.environ['API_KEY']
    response = google_ocr(api_key, file_path)
    # content = text_extract(img)
    temp_1 = get_extended_annotations(response)
    temp_2 = get_threshold_for_y_difference(temp_1)
    temp_3 = group_annotations(temp_1, temp_2)
    temp_4 = sort_and_combine_grouped_annotations(temp_3)
    content = ""
    for i in temp_4:
        content += i+"\n"
    print(content)
    content = content.replace("..", "")
    content = content.replace("...", "")
    content = content.replace("....", "")
    content = content.replace(".....", "")
    
    for key, val in data.items():
        for i in val:
            content = content.replace(i, key+":")

    list = rearrange_list(content, field)
    temp_result = field_detect(content, list)
    result = {}
    # print(type(result["med_sum:"]))
    for i in list:
        if temp_result[i] != []:
            merge(result, {i:temp_result[i]})
    if "print:" in result:
        del result["print:"]
    if "ommit:" in result:
        del result["ommit:"]
    print(result)
    return result

import contextlib
from io import BytesIO
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
from PIL import Image
import json
import pandas as pd
import imutils
from skimage.filters import threshold_local
import cv2
import os
import mysql.connector
from streamlit_js_eval import streamlit_js_eval
import time
from PIL import ImageOps


st.set_page_config(
    page_title="Nhập liệu bệnh án",
    page_icon="🖼️",
    layout="wide",
)

# ---------- HEADER ----------
st.title("Nhập liệu bệnh án")

ready = False
# ---------- FUNCTIONS ----------
def _reset(key: str) -> None:
    if key == "all":
        st.session_state["rotate_slider"] = 0
        st.session_state["brightness_slider"] = st.session_state[
            "saturation_slider"
        ] = st.session_state["contrast_slider"] = 100
        st.session_state["bg"] = st.session_state["gray_bw"] = 0
    elif key == "rotate_slider":
        st.session_state["rotate_slider"] = 0
    elif key == "checkboxes":
        st.session_state[
            "gray_bw"
        ] = 0
    else:
        st.session_state[key] = 100
        
def scroll_to_top():
    """Scrolls to the top of the page."""
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''
    if "scroll_to_top" not in st.session_state:
        st.session_state["scroll_to_top"] = js
    st.components.v1.html(js)
    st.write(js, unsafe_allow_html=True)

def order_points(pts):
    # initializing the list of coordinates to be ordered
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    # top-left point will have the smallest sum
    rect[0] = pts[np.argmin(s)]
    # bottom-right point will have the largest sum
    rect[2] = pts[np.argmax(s)]

    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # returns ordered coordinates
    return rect


def perspective_transform(image, pts):
    # unpack the ordered coordinates individually
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

    # compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    # Apply the transform matrix
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))

    # return the warped image
    return warped

ready = False
type = st.secrets["type"]
project_id = st.secrets["project_id"]
private_key = st.secrets["private_key"]
private_key_id = st.secrets["private_key_id"]
client_email = st.secrets["client_email"]
client_id = st.secrets["client_id"]
auth_uri = st.secrets["auth_uri"]
token_uri = st.secrets["token_uri"]
auth_provider_x509_cert_url = st.secrets["auth_provider_x509_cert_url"]
client_x509_cert_url = st.secrets["client_x509_cert_url"]
universe_domain = st.secrets["universe_domain"]

api_key = {
    "type": type,
    "project_id": project_id,
    "private_key_id": private_key_id,
    "private_key": private_key,
    "client_email": client_email,
    "client_id": client_id,
    "auth_uri":auth_uri,
    "token_uri":token_uri,
    "auth_provider_x509_cert_url":auth_provider_x509_cert_url,
    "client_x509_cert_url":client_x509_cert_url,
    "universe_domain":universe_domain
}
upload_imgs = st.file_uploader(
        label="Upload an image",
        type=["bmp", "jpg", "jpeg", "png", "svg"],
        accept_multiple_files = True
    )
mode = "upload"
with contextlib.suppress(NameError):
    for img in upload_imgs:
        if img is not None:
            pil_img = (
                upload_img.convert("RGB")
                if mode == "url"
                else Image.open(upload_img).convert("RGB")
            )
            img_arr = np.asarray(pil_img)
            with st.container():
                lcol, rcol = st.columns(2)
                image = Image.fromarray(img_arr)
                image = ImageOps.exif_transpose(image)
                right = 100
                left = 100
                top = 100
                bottom = 100
                width, height = image.size

                new_width = width + right + left
                new_height = height + top + bottom
                    
                result = Image.new(image.mode, (new_width, new_height), (0,0,0))
                result.paste(image, (left, top))
                result = np.asarray(result)
                ratio = result.shape[0] / 500.0
                img_resize = imutils.resize(result, height=500)
                copy = result.copy()
                gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                    
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                edged_img = cv2.Canny(blurred_image, 75, 200)
                cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        doc = approx
                        break
                        
                p = []
                for d in approx:
                    tuple_point = tuple(d[0])
                    cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
                    p.append(tuple_point)
                warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
                warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                    # image = warped_image
                image = Image.fromarray(warped_image)
                image.save(img.name + "png")
                ready = True

# if ready == True:
#     for filename in os.listdir('/src/pages'):
#         if filename.endswith(".png"):







