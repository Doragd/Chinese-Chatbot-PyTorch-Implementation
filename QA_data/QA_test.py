# -*- coding: UTF-8 -*-

import sqlite3
import jieba
import logging
jieba.setLogLevel(logging.INFO) #设置不输出信息

conn = sqlite3.connect('./QA_data/QA.db')

cursor = conn.cursor()
stop_words = []
with open('./QA_data/stop_words.txt', encoding='gbk') as f:
    for line in f.readlines():
        stop_words.append(line.strip('\n'))

def match(input_question):
    res = []
    cnt = {}
    question = list(jieba.cut(input_question, cut_all=False)) #对查询字符串进行分词
    for word in reversed(question):  #去除停用词
        if word in stop_words:
            question.remove(word)
    for tag in question: #按照每个tag，循环构造查询语句
        keyword = "'%" + tag + "%'"
        result = cursor.execute("select * from QA where tag like " + keyword)
        for row in result:
            if row[0] not in cnt.keys():
                cnt[row[0]]  = 0
            cnt[row[0]] += 1 #统计记录出现的次数
    try:
        res_id = sorted(cnt.items(), key=lambda d:d[1],reverse=True)[0][0] #返回出现次数最高的记录的id
    except:
        return tuple() #若查询不出则返回空
    cursor.execute("select * from QA where id= " + str(res_id))
    res = cursor.fetchone()
    if type(res) == type(tuple()):
        return res #返回元组类型(id, question, answer, tag)
    else:
        return tuple() #若查询不出则返回空

