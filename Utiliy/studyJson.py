# -*- coding: utf-8 -*-

import json
'''
    Json模块提供了四个功能：dumps、dump、loads、load
        - dumps把python数据类型转换成字符串         json_str = json.dumps(test_dict)
        - dump把数据类型转换成字符串并存储在文件中  
        - loads把字符串转换成数据类型
        - load把文件打开从字符串转换成数据类型
'''

'''     dumps：将python中的 字典 转换为 字符串  '''
test_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}
print(test_dict)
print(type(test_dict))
#dumps 将数据转换成字符串
json_str = json.dumps(test_dict)
print(json_str)
print(type(json_str))

'''    loads: 将 字符串 转换为 字典    '''
new_dict = json.loads(json_str)
print(new_dict)
print(type(new_dict))

'''    dump: 将数据写入json文件中   '''
with open("../config/record.json","w") as f:
    json.dump(new_dict,f)
    print("加载入文件完成...")

'''     load:把文件打开，并把字符串变换为数据类型   '''
# with open("../config/record.json",'r') as load_f:
with open("record.json",'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)


load_dict['smallberg'] = [8200,{1:[['Python',81],['shirt',300]]}]
print(load_dict)

# with open("../config/record.json","w") as dump_f:
with open("record.json","w") as dump_f:
    json.dump(load_dict,dump_f)


