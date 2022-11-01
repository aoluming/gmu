import requests
import sys

import logging

import os
import pdb
# pdb.set_trace()
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from transformers import ViltProcessor

from module.modeling_vilt import ViltModel
from model.ammre import ConditionedVilt
from configs.exp_1 import Exp1

if __name__ == '__main__':
    # prepare image and text
    # import pdb

    pdb.set_trace()
    url = "inf.png"

    image = Image.open(url)

    text = "hello world"
    text2 = "hello world again!"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    inputs = processor([image, image], [text, text2], return_tensors="pt", padding=True)
    config = Exp1()

    # backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm",)
    model = ConditionedVilt(backbone=backbone, training_config=config)
    pdb.set_trace()
    
    # for key,value in inputs.items():
    #     print(value.shape)
    #     print(key)
    #     # pdb.set_trace()
    output, logit = model(**inputs)

    last_hidden_state = output.last_hidden_state

    print(last_hidden_state.shape)
    print(logit.shape)
