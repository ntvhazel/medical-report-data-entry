from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import HttpResponseError
from PIL import Image
import re
import sys
from Preprocess import preprocess
import json
from operator import attrgetter
import numpy as np

def merge(dict1, dict2):
    return dict1.update(dict2)

def google_ocr(img_path):
    from google.cloud import vision
    import os

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:\Thesis\LVTN_VanThanhThuan\src\pj_key.json"

    client = vision.ImageAnnotatorClient()

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

def inference(file_path):
    # img = preprocess(file_path)
    data, field = read_json()
    response = google_ocr(file_path)
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
    result = field_detect(content, list)
    del result['print:']
    del result['ommit:']
    
    return result









