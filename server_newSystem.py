#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 1.0
# @Author   : QQ736592720
# @Datetime : 2022/4/9 18:04
# @Project  : 简答题399___baidu_ocr.py
# @File     : 简答题457___轻量级服务器lighthouse搭建.py
import traceback
from bottle import route, run, request, response
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

class AlexNet(nn.Module):
    def __init__(self,num_classes=10):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=3, padding=1, bias=False),
            # nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.feature_extraction2 = nn.Sequential(
            nn.Linear(in_features=1*75*75,out_features=3800),
            nn.Linear(in_features=3800, out_features=2048),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes),
        )
    def forward(self,x,is_feature=False):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),1*75*75)
        x = self.feature_extraction2(x)
        if is_feature:
            return x
        x = self.classifier(x)
        return x

@hook('before_request')
def validate():
    REQUEST_METHOD = request.environ.get('REQUEST_METHOD')

    HTTP_ACCESS_CONTROL_REQUEST_METHOD = request.environ.get('HTTP_ACCESS_CONTROL_REQUEST_METHOD')
    if REQUEST_METHOD == 'OPTIONS' and HTTP_ACCESS_CONTROL_REQUEST_METHOD:
        request.environ['REQUEST_METHOD'] = HTTP_ACCESS_CONTROL_REQUEST_METHOD

@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'

@route('/hello', method='GET')
def hello():
    return "update"



@route('/getImages', method='POST')
def getImages():
    start = int(request.forms.get('start'))
    start = start-1
    qicai = request.forms.get('equipment')
    wenyang = request.forms.get('patterns')
    zhidi = request.forms.get('zhidi')
    pre_wenyang = request.forms.get('pre_wenyang')
    # 还是过不来这个数据
    print("prewenyangw纹样 = ", pre_wenyang)
    res1=None
    # res2=None
    res2 = list()
    res3=None
    if qicai == 'undefined' or qicai == '全部':
        res=all_file_url
        pass
    else:
        res=qicais_dir[qicai]
    if wenyang == 'undefined' or wenyang == '纹样':

        temp = list()
        if pre_wenyang == '动物纹':
            temp = ['二龙戏珠纹', '丹凤纹', '云蝠纹', '虎纹', '云龙纹', '五蝠纹', '凤纹', '双鱼纹', '夔纹', '夔龙纹', '大云龙纹', '游龙纹', '牛纹', '璃螭纹',
                    '白龙纹', '蝉纹', '蟠虯纹', '蟠虺纹', '蟠螭纹', '金龙纹', '龙纹']
        elif pre_wenyang == '植物纹':
            temp = ['串枝花纹', '勾莲纹', '勾莲花卉纹', '四瓣花纹', '团花纹', '宝相花纹', '折枝桃纹', '折枝花卉纹', '折枝花蝶纹',
                                              '折枝莲纹', '朵花纹', '树皮纹', '桃竹纹', '梅兰纹', '梅纹', '梅花纹', '水仙花纹', '海棠纹', '灵芝纹',
                                              '牡丹纹', '百合花纹', '石榴纹', '石榴花纹', '石竹花纹', '胡桃纹', '芍药纹', '芙蓉纹', '芙蓉花纹', '花卉纹',
                                              '花果纹', '花纹', '荔枝纹', '荷叶纹', '荷莲纹', '莲实纹', '莲瓣纹', '莲纹', '莲花纹', '菊瓣纹', '菊纹',
                                              '菊花纹', '萱草纹', '葡萄纹', '葫芦纹', '西番莲纹']
        elif pre_wenyang == '几何纹':
            temp = ['双喜字纹', '回纹', '如意纹', '字纹', '寿字纹', '弦纹', '棱纹', '环带纹', '目纹', '锦纹', '鳞纹', '龟背纹']
        elif pre_wenyang == '自然纹':
            temp = ['云纹', '山水纹', '流水纹', '狩猎纹']
        elif pre_wenyang == '图腾纹':
            temp = ['二龙戏珠纹', '丹凤纹', '五蝠纹', '兽面纹', '凤纹', '夔纹', '夔龙纹', '大云龙纹', '异兽纹', '摩竭纹', '暗八仙纹', '涡纹', '游龙纹', '璃螭纹',
                    '白龙纹', '蟠螭纹', '金龙纹']
        elif pre_wenyang == '组合纹':
            temp = ['云雷纹', '云龙杂宝纹', '八宝纹', '凤衔花纹', '勾莲花卉纹', '双龙捧寿纹', '团花蝴蝶纹', '寿字缠枝莲九龙纹',
                                              '寿字花卉纹', '寿字龙凤纹', '杂宝纹', '梅兰纹', '海水飞兽纹', '牡丹凤凰纹', '瓜蝶纹', '祥云团龙纹',
                                              '花卉双龙捧寿纹', '花卉杂宝纹', '花卉麒麟纹', '花蝶纹', '花鸟纹', '草虫纹', '荷莲缠枝牡丹纹', '荷莲鸭纹',
                                              '莲蝠纹', '菊蝶纹虎皮纹', '蟠螭百花纹', '蟠螭缠枝花卉纹', '蟠螭缠枝花卉莲瓣纹', '鱼藻纹', '鸳鸯鹭莲纹', '鹊梅纹',
                                              '黑花虎纹', '龙凤纹', '龙穿花纹', '龙马纹']

        # 遍历list然后， res2 = wenyangs_dir[wenyang] ，多个list要整合成一个list

        # File "E:/王/server_newSystem/server_newSystem.py", line 113, in getImages
        #     res2 = res2 + wenyangs_dir[wenyang]
        # TypeError: unsupported operand type(s) for +: 'NoneType' and 'list'
        for wenyang in temp:
            print("二级纹样是什么？", wenyang)
            res2 = res2 + wenyangs_dir[wenyang]

        res = [val for val in res if val in res2]
        # pass
    else:
        res2=wenyangs_dir[wenyang]
        print("res2:",res2)
        res=[val for val in res if val in res2]
    if zhidi == 'undefined' or zhidi == '全部':
        pass
    else:
        res3=zhidis_dir[zhidi]
        print("res3:",res3)
        res=[val for val in res if val in res3]
    res=res[start*12:start*12+12]
    labels=list()
    for file_url in res:
        # temp_labels=all_file_dir[file_url]
        # labels.append(temp_labels[0]+'_'+temp_labels[1]+'_'+temp_labels[2])
        labels.append(path_to_name[file_url])
        print("labels:",labels)
    res = {"url": res, "labels": labels}

    return json.dumps(res)


@route('/getLabels', method='POST')
def getLabels():
    # 图片上传保存的路径
    # return json.dumps([list(qicais_dir.keys()),list(wenyangs_dir.keys()),list(zhidis_dir.keys())])
    key = qicais_dir
    list1 = list(qicais_dir.keys())
    list2 = list(zhidis_dir.keys())
    return json.dumps([list(qicais_dir.keys()),['串枝花纹', '勾莲纹', '勾莲花卉纹', '四瓣花纹', '团花纹','宝相花纹','折枝桃纹','折枝花卉纹', '折枝花蝶纹', '折枝莲纹', '朵花纹','树皮纹', '桃竹纹', '梅兰纹', '梅纹', '梅花纹', '水仙花纹', '海棠纹','灵芝纹','牡丹纹','百合花纹','石榴纹', '石榴花纹','石竹花纹','胡桃纹', '芍药纹', '芙蓉纹', '芙蓉花纹', '花卉纹','花果纹', '花纹', '荔枝纹','荷叶纹', '荷莲纹','莲实纹', '莲瓣纹', '莲纹', '莲花纹', '菊瓣纹', '菊纹', '菊花纹','萱草纹', '葡萄纹', '葫芦纹', '西番莲纹'],list(zhidis_dir.keys())])



@route('/classification', method='POST')
def recognition():
    img_base64 = request.forms.get('img_base64')
    img_base64=img_base64.split(',')[1]
    uploadfile = base64.b64decode(img_base64)
    temp_save="./temp_save"
    if os.path.exists(temp_save) is False:
        os.mkdir(temp_save)
    with open(temp_save+"/1.png", 'wb') as f:
        f.write(uploadfile)
    rec_res=cnn_feature(temp_save+"/1.png")
    return rec_res

def load_img_with_path(file_path):
    image = Image.open(file_path)
    x_s = 224
    y_s = 224
    # 修改图片大小
    out = image.resize((x_s, y_s), Image.LANCZOS)
    out = np.asarray(out)
    out = out[:, :, :3]
    return out

def cnn_feature(img_path):
    train_data = list([])
    im_vec = load_img_with_path(img_path)

    im_vec = im_vec.transpose((2, 0, 1))
    im_vec = im_vec / 255.
    train_data.append(im_vec)
    train_data_reg = np.array(train_data)
    res_str=""
    with torch.no_grad():
        x = torch.tensor(train_data_reg)
        x = x.to(torch.float32)
        batch_x = x

        res = qicai_model.forward(batch_x, False).T
        res = res.T[0]
        print("rest0 = ", res)
        res = res.argmax()
        res_str=res_str+list(qicais_dir.keys())[res]
        print("res = ", res_str)

        res = wenyang_model.forward(batch_x, False).T
        res = res.T[0]
        res = res.argmax()
        res_str=res_str+"  "+list(wenyangs_dir.keys())[res]
        print("res = ", res_str)
        
        # res = zhidi_model.forward(batch_x, False).T
        # res = res.T[0]
        # res = res.argmax()
        # res_str=res_str+"  "+list(zhidis_dir.keys())[res]
        # print("res = ", res_str)
    time.sleep(3)
    return res_str



if __name__ == '__main__':

    all_file_url=list()
    all_file_dir={}
    excel_paths = ['故宫博物院数字文物库-1-324 - 副本.xlsx',
                   '故宫博物院数字文物库-325-949 - 副本.xlsx',
                   '故宫博物院数字文物库-950-1399 - 副本.xlsx',
                   '故宫博物院数字文物库-1400-1999 - 副本.xlsx',
                   '故宫博物院数字文物库-2000-2594 - 副本.xlsx',
                   '民族服饰 - 汉族.xlsx']
    pre_url = 'http://10.156.8.23:8086/image/wuyuhui/故宫博物院/'
    preurl = 'http://10.156.8.23:8086/image/liuyaozong/民族服饰 - 汉族/07-11 113638/'

    part_urls = [pre_url + '故宫博物院数字文物库-1-324',
                 pre_url + '故宫博物院数字文物库-325-949',
                 pre_url + '故宫博物院数字文物库-950-1399',
                 pre_url + '故宫博物院数字文物库-1400-1999',
                 pre_url + '故宫博物院数字文物库-2000-2594',
                 preurl]
    # excel_paths=['民族服饰 - 汉族.xlsx']
    # pre_url = 'http://10.156.8.23:8086/image/liuyaozong/民族服饰 - 汉族/07-11 113638/'
    # part_urls=[pre_url]
    # qicais=['织绣', '珐琅器', '漆器', '生活用具', '铜器', '陶瓷', '其他工艺', '外国文物', '玉石器', '石器', '石刻', '玉器', '陶器', '瓷器', '砖瓦', '金银器', '木器', '书法', '绘画', '雕塑造像', '甲骨简牍', '文献文书', '文具', '货币', '印信', '近现代文物', '民俗文物', '少数民族文物', '自然标本']
   
    qicais_dir = {} #用于存储不同类别的图像路径
    
    # for qicai in qicais:
    #     qicais_dir[qicai]=list()
    wenyangs = ['串枝花纹', '丹凤纹', '二龙戏珠纹', '云纹', '云蝠纹', '云雷纹', '云龙杂宝纹', '云龙纹', '五蝠纹', '八宝纹', '兽面纹', '凤纹', '凤衔花纹', '勾莲纹',
                '勾莲花卉纹', '双喜字纹', '双鱼纹', '双龙捧寿纹', '四瓣花纹', '回纹', '团花纹', '团花蝴蝶纹', '夔纹', '夔龙纹', '大云龙纹', '如意纹', '字纹', '宝相花纹',
                '寿字纹', '寿字缠枝莲九龙纹', '寿字花卉纹', '寿字龙凤纹', '山水纹', '异兽纹', '弦纹', '折枝桃纹', '折枝花卉纹', '折枝花蝶纹', '折枝莲纹', '摩竭纹',
                '暗八仙纹', '朵花纹', '杂宝纹', '树皮纹', '桃竹纹', '梅兰纹', '梅纹', '梅花纹', '棱纹', '水仙花纹', '流水纹', '海棠纹', '海水飞兽纹', '涡纹',
                '游龙纹', '灵芝纹', '牛纹', '牡丹凤凰纹', '牡丹纹', '狩猎纹', '环带纹', '璃螭纹', '瓜蝶纹', '白龙纹', '百合花纹', '目纹', '石榴纹', '石榴花纹',
                '石竹花纹', '祥云团龙纹', '福在眼前纹', '福禄寿纹', '穿花纹', '绳纹', '缠枝海石榴纹', '缠枝牡丹纹', '缠枝花卉纹', '缠枝花果纹', '缠枝花纹', '缠枝莲纹',
                '缠枝菊纹', '胡桃纹', '芍药纹', '芙蓉纹', '芙蓉花纹', '花卉双龙捧寿纹', '花卉杂宝纹', '花卉纹', '花卉麒麟纹', '花果纹', '花纹', '花蝶纹', '花鸟纹',
                '草虫纹', '荔枝纹', '荷叶纹', '荷莲纹', '荷莲缠枝牡丹纹', '荷莲鸭纹', '莲实纹', '莲瓣纹', '莲纹', '莲花纹', '莲蝠纹', '菊瓣纹', '菊纹', '菊花纹',
                '菊蝶纹虎皮纹', '萱草纹', '葡萄纹', '葫芦纹', '蝉纹', '蟠虯纹', '蟠虺纹', '蟠螭百花纹', '蟠螭纹', '蟠螭缠枝花卉纹', '蟠螭缠枝花卉莲瓣纹', '西番莲纹',
                '金龙纹', '锦纹', '鱼藻纹', '鳞纹', '鸳鸯鹭莲纹', '鹊梅纹', '黑花虎纹', '龙凤纹', '龙穿花纹', '龙纹', '龙马纹', '龟背纹','虎纹',
                '鱼纹', '蝙蝠纹', '祥云纹', '鸟纹', '蝴蝶纹', '老虎纹', '人物纹', '寿桃纹']
    wenyangs_dir = {}  #用于存储每个纹样名称对应的图像路径
    for wenyang in wenyangs:
        wenyangs_dir[wenyang] = list()#把一个个二级纹样变成一个个的list
    
    #质地
    zhidis = ['石', '宝玉石', '陶', '瓷', '砖瓦', '泥', '玻璃', '金', '银', '铜', '铁', '木', '竹', '纸', '棉麻', '纤维', '毛', '丝', '皮革',
              '骨角牙', '玉石', '金属', '木材']
    
    # {'动物纹': ['二龙戏珠纹','丹凤纹', '云蝠纹','云龙纹','五蝠纹','凤纹','双鱼纹','夔纹', '夔龙纹','大云龙纹','游龙纹','牛纹','璃螭纹','白龙纹', '蝉纹','蟠虯纹','蟠虺纹', '蟠螭纹', '金龙纹','龙纹'],
    #  '植物纹': ['串枝花纹', '勾莲纹', '勾莲花卉纹', '四瓣花纹', '团花纹','宝相花纹','折枝桃纹','折枝花卉纹', '折枝花蝶纹', '折枝莲纹', '朵花纹','树皮纹', '桃竹纹', '梅兰纹', '梅纹', '梅花纹', '水仙花纹', '海棠纹','灵芝纹','牡丹纹','百合花纹','石榴纹', '石榴花纹','石竹花纹','胡桃纹', '芍药纹', '芙蓉纹', '芙蓉花纹', '花卉纹','花果纹', '花纹', '荔枝纹','荷叶纹', '荷莲纹','莲实纹', '莲瓣纹', '莲纹', '莲花纹', '菊瓣纹', '菊纹', '菊花纹','萱草纹', '葡萄纹', '葫芦纹', '西番莲纹'],
    #  '几何纹': ['双喜字纹', '回纹',  '如意纹', '字纹','寿字纹','弦纹', '棱纹','环带纹','目纹', '锦纹','鳞纹','龟背纹'],
    #  '自然纹': ['云纹','山水纹','流水纹','狩猎纹'],
    #  '图腾纹': ['二龙戏珠纹','丹凤纹','五蝠纹','兽面纹','凤纹','夔纹', '夔龙纹','大云龙纹','异兽纹','摩竭纹', '暗八仙纹', '涡纹', '游龙纹','牛纹','璃螭纹','白龙纹', '蟠螭纹', '金龙纹','龙纹'],
    #  '组合纹': ['云雷纹','云龙杂宝纹','八宝纹','凤衔花纹','勾莲花卉纹','双龙捧寿纹','团花蝴蝶纹','寿字缠枝莲九龙纹','寿字花卉纹', '寿字龙凤纹','杂宝纹', '梅兰纹','海水飞兽纹', '牡丹凤凰纹', '瓜蝶纹','祥云团龙纹','花卉双龙捧寿纹', '花卉杂宝纹','花卉麒麟纹','花蝶纹','花鸟纹','草虫纹', '荷莲缠枝牡丹纹','荷莲鸭纹','莲蝠纹', '菊蝶纹虎皮纹','蟠螭百花纹','蟠螭缠枝花卉纹', '蟠螭缠枝花卉莲瓣纹', '鱼藻纹','鸳鸯鹭莲纹', '鹊梅纹', '黑花虎纹', '龙凤纹','龙穿花纹', '龙马纹']}
    zhidis_dir = {}
    for zhidi in zhidis:
        zhidis_dir[zhidi] = list()
    
    #用于存储找不到纹样和质地的图像信息
    unknown_wenyang={}
    unknown_zhidi={}

    path_to_name={} #将图像路径映射到图像名称

    for index in range(len(excel_paths)):
        excel_path = excel_paths[index]
        dataframe = pd.read_excel(excel_path).values  # 打开文件

        #遍历Excel表内容，image_info代表每一行数据的内容
        for image_info in dataframe:
            #image_info[0]表示当前行的第一个元素
            the_class = image_info[0]
            if pd.isnull(the_class) or pd.isnull(image_info[1]) or len(the_class) == 0 or len(image_info[1]) == 0:
                continue

            image_path = part_urls[index] + image_info[1][1:] #image_info[1][1:]表示对image_info[1]进行切片操作，去除第一个字符
            path_to_name[image_path]=image_info[2]
            all_file_url.append(image_path)
            all_file_dir[image_path]=[image_info[0],'未知','未知']

            fine_class_text = image_info[0] + image_info[2]
            temp_qicai = image_info[0]

            if temp_qicai not in qicais_dir.keys():
                qicais_dir[temp_qicai] = list()
            qicais_dir[temp_qicai].append(image_path)

            is_found_wenyang=False
            for wenyang in wenyangs:
                if wenyang in fine_class_text:
                    wenyangs_dir[wenyang].append(image_path)
                    all_file_dir[image_path][1]=wenyang
                    is_found_wenyang=True
            if not is_found_wenyang:
                unknown_wenyang[image_info[2]]=excel_path

            is_found_zhidi=False
            for zhidi in zhidis:
                if zhidi in fine_class_text:
                    zhidis_dir[zhidi].append(image_path)
                    all_file_dir[image_path][2]=zhidi
                    is_found_zhidi=True
            if not is_found_zhidi:
                unknown_zhidi[image_info[2]]=excel_path

    model_path=["./model/qicai.pkl","./model/wenyang.pkl","./model/zhidi.pkl"]
    qicai_model = AlexNet(25)  # 调用模型Model
    qicai_model.load_state_dict(torch.load(model_path[0]))  # 加载模型参数
    qicai_model.eval()
    wenyang_model = AlexNet(140)  # 调用模型Model
    wenyang_model.load_state_dict(torch.load(model_path[1]))  # 加载模型参数
    wenyang_model.eval()
    # zhidi_model = AlexNet(23)  # 调用模型Model
    # zhidi_model.load_state_dict(torch.load(model_path[2]))  # 加载模型参数
    # zhidi_model.eval()

    qicais_dir['服饰']=list()

    run(host='0.0.0.0', port=8888, debug=False)  # 记得服务器开启8888端口


