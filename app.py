import traceback
from bottle import route, run, request, response, HTTPResponse
from bottle import hook
from json import dumps
import base64
import os
import numpy as np
# import scipy.io
# from sklearn import preprocessing
import json
import pandas as pd
import torch
import torch.nn as nn
# from cnn_train_AlexNet import *
from PIL import Image
import time
from flask import Flask, jsonify
import mysql.connector

app = Flask(__name__)

# 创建应用上下文
app.app_context().push()

connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="root",
    database="data",
    connect_timeout=10,
)
cursor = connection.cursor()


@app.route('/getImages')
def get_images():
    # 查询数据库获取数据
    query = "SELECT imgUrl, description FROM test"
    cursor.execute(query)
    data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

    result = []
    for row in data:
        result.append({
        'imgUrl': row[0],  # 使用列的别名或索引
        'description': row[1]
    })

    return jsonify(result)



if __name__ == '__main__':


    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"
    app.run(host='0.0.0.0', port=7777, debug=False)  # 记得服务器开启8888端口


