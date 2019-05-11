# -*- coding: utf-8 -*- 

import re
import sqlite3
import requests
import jieba
import logging
from bs4 import BeautifulSoup as BS
jieba.setLogLevel(logging.INFO) #设置不输出信息

max_length = 100 #爬取的问题数
path = 'QA.db'

stop_words = []
with open('stop_words.txt') as f:
    for line in f.readlines():
        stop_words.append(line.strip('\n'))

print("Start fetching data...")

url = 'https://cloud.tencent.com/document/product/'
r = requests.get(url)
soup = BS(r.text,'html.parser')
doc_item_links = soup.find_all("a", class_="doc-media-panel-item-link")
urls = []
for link in doc_item_links:
    if link['href'].split('/')[-1].isdigit():
        doc_url = url + link['href'].split('/')[-1]
        r = requests.get(doc_url)
        soup_problem = BS(r.text,'html.parser')
        try:
            problem_links = soup_problem.find(title='常见问题').parent.find_all('a',href=re.compile('^/'))                
            for link in problem_links:
                urls.append(url + link['href'][18:])
        except:
            continue
print("URLs has saved successfully")

problems = {}
def save(max_length):
    for problem_url in urls:
        r = requests.get(problem_url)
        soup = BS(r.text,'html.parser')
        problem_list = soup.find_all(name='h3', id=re.compile('^\.'))
        for problem in problem_list:
            problems[problem.string] = ''.join([child.string for child in problem.next_sibling.children if child.string is not None]) + '|Ref:' + problem_url
            if(len(problems) > max_length):
                return

save(max_length)
print("Problems has saved successfully")

conn = sqlite3.connect(path)
print("Opened database successfully")
c = conn.cursor()
try:
    c.execute('''CREATE TABLE QA
           (ID INTEGER PRIMARY KEY    AUTOINCREMENT,
           Q           TEXT    NOT NULL,
           A           TEXT     NOT NULL
           TAG           TEXT     NOT NULL);''')
    print("Table created successfully")
    conn.commit()
except:
    c.execute('drop table QA;')
    c.execute('''CREATE TABLE QA
           (ID INTEGER PRIMARY KEY    AUTOINCREMENT,
           Q           TEXT    NOT NULL,
           A           TEXT     NOT NULL,
           TAG           TEXT     NOT NULL);''')
    print("Table created successfully")
    conn.commit()    

for k,v in problems.items():
    question = list(jieba.cut(k, cut_all=False))
    for word in reversed(question):  #去除停用词
        if word in stop_words:
            question.remove(word)
    tag = "|".join(question) #获取问题的tag
    sql = "insert into QA(Q,A,tag) values('%s','%s','%s')" % (k, v, tag)
    c.execute(sql)
conn.commit()
print("Inserted successfully")
conn.close()