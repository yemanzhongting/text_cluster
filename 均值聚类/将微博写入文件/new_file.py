# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2019/4/9 17:13'
import json
def get_json_list():
    comments=[]
    with open('seg.json','r+',encoding='utf-8') as f:
        one=f.readlines()

    for i in one:
        i=json.loads(i)
        #print(i)
        try:
            comments.append(i['segmentation'])
        except KeyError:
            print('分词为空')
    print(len(comments))
    #总共8708个
    print(comments[1:20])
    return comments

def write_file():
    comments=get_json_list()
    count=0
    for i in comments:
        count=count+1
        path=r'E:\微博文本数据'
        with open((path+'\seg'+str(count)), 'w+', encoding='utf-8') as f:
            f.write(i)

# if __name__=='__main__':
#     write_file()
#     print('写入完成')