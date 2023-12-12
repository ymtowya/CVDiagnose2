# import all the tools we need
import urllib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random
import xml.etree.ElementTree as ET
import time
import requests

# Setting up GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Setting up the model from the previous

num_classes = 3

# loaded = torch.load('./phase3_model.pth')
model2 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model2.roi_heads.box_predictor.cls_score.in_features
model2.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model2 = model2.to(device)

model2.load_state_dict(torch.load('./phase3_model.pth'))
model2.eval()

# helper function for single image prediction
def single_img_predict(img, nm_thrs = 0.2, score_thrs=0.2):
    test_img = transforms.ToTensor()(img)
    model2.eval()
    
    with torch.no_grad():
        predictions = model2(test_img.unsqueeze(0).to(device))
        
    test_img = test_img.permute(1,2,0).numpy()
    
    # non-max supression
    keep_boxes = torchvision.ops.nms(predictions[0]['boxes'].cpu(),predictions[0]['scores'].cpu(),nm_thrs)
    
    # Only display the bounding boxes which higher than the threshold
    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs
 
    if len(keep_boxes) > 1 and score_filter.sum() > 1:
        first_two_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][:3]
        first_two_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][:3]
        return test_img, first_two_boxes, first_two_labels
    
    # get the filtered result
    test_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]
    
    return test_img, test_boxes, test_labels

# path of test images directory
test_dir_path = './runs2'

# List of Image file name 
test_file_list = os.listdir(test_dir_path)

for path in test_file_list:

    test_img = Image.open(os.path.join(test_dir_path, path)).convert('RGB')

    # Prediction
    test_img, test_boxes, test_labels = single_img_predict(test_img)
    test_output = draw_boxes(test_img, test_boxes,test_labels)

    # Display the result
    fig, (ax1) = plt.subplots(1,1,figsize=(15,6))
    ax1.imshow(test_output)
    ax1.set_xlabel('Prediction')
    plt.show()
    plt.savefig('./runs3/res_' + path)