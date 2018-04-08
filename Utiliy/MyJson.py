# -*- coding: utf-8 -*-

import json
'''
    Json模块提供了四个功能：dumps、dump、loads、load
        - dumps把python数据类型转换成字符串         json_str = json.dumps(test_dict)
        - dump把数据类型转换成字符串并存储在文件中   json.dump(new_dict,file)
        - loads把字符串转换成数据类型               new_dict = json.loads(json_str)
        - load把文件打开从字符串转换成数据类型       load_dict = json.load(load_file)
'''
class MyJson():
    def dump(self,filePath,pyDict):
        '''dump把数据类型转换成字符串并存储在文件中'''
        with open(filePath,"w") as fileJson:
            json.dump(pyDict,fileJson)
            print("加载入文件完成...")
    
    def load(self,filePath):
        '''把文件打开从字符串转换成数据类型'''
        with open(filePath,'r') as load_file:
            load_dict = json.load(load_file)
            return load_dict
'''
写入json文件示例：
    test_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}

    with open("../config/record.json",'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)


    load_dict['smallberg'] = [8200,{1:[['Python',81],['shirt',300]]}]
    print(load_dict)

    with open("../config/record.json","w") as dump_f:
        json.dump(load_dict,dump_f)

'''


