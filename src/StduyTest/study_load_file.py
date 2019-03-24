# encoding: utf-8
'''
1、读取指定目录下的所有文件
2、读取文件，正则匹配出需要的内容，获取文件名
3、打开此文件(可以选择打开可以选择复制到别的地方去)
'''
import os.path
import sys
import re

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
            print(allDir)
        # child = os.path.join('%s\%s' % (filepath, allDir))
        # if os.path.isfile(child):
        #     readFile(child)
        #     # print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        #     continue
        # eachFile(child)

# # 遍历出结果 返回文件的名字
# def readFile(filenames):
#         fopen = open(filenames, 'r', encoding='UTF-8') # r 代表read
#         fileread = fopen.read()
#         fopen.close()
#         t=re.search(r'clearSpitValve',fileread)
#         if t:
#             # print "匹配到的文件是:"+filenames
#             arr.append(filenames)    

if __name__ == "__main__":
    print('目前系统的编码为：',sys.getdefaultencoding()) 

    filenames = "Data/logos/" # refer root dir
    arr=[]
    eachFile(filenames)
    for i in arr:
        print(i)

