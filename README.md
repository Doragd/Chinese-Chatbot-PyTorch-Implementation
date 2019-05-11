# 🍀 小智，又一个中文聊天机器人:yum:

💖 利用有趣的中文语料库qinyun，由@Doragd同学编写的中文聊天机器人:snowman:

* 尽管她不是那么完善:muscle:，不是那么出色:paw_prints:
* 但她是由我自己coding出来的:sparkling_heart: ，所以![](https://img.shields.io/badge/-It%20means%20everything-ff69b4.svg)

* **希望大家能够多多star支持**:star: 这个NLP初学者:runner:和他的朋友🍀 小智 



## :rainbow:背景

这个项目实际是软件工程课程设计的子模块。我们的目标是开发一个智能客服工单处理系统。

智能客服工单系统实际的工作流程是：当人向系统发出提问时，系统首先去知识库中查找是否存在相关问题，如果有，则返回问题的答案，此时如何人不满意，则可以直接提交工单。如果知识库中不存在，则调用这个聊天机器人进行自动回复。

该系统服务的场景类似腾讯云的客服系统，客户多是来咨询相关问题的(云服务器，域名等)，所以知识库也是有关云服务器，域名等的咨询，故障处理的 (问题,答案) 集合。

系统的前端界面和前后端消息交互由另一个同学完成，主要采用React+Django方式。

@Doragd 同学负责的是知识库的获取和聊天机器人的编写，训练，测试。这个repo的内容也是关于这个的。



## :star2: 测试效果

* 不使用知识库进行回答
<img src="https://i.loli.net/2019/05/11/5cd69c8de54c1.png" width=50%  title="测试效果|不使用知识库" />

* 使用知识库进行回答
<img src="https://i.loli.net/2019/05/11/5cd69dce836a6.png" width=50%  title="测试效果|使用知识库" />


* 整个系统效果：
  <img src="https://i.loli.net/2019/05/11/5cd69aa2eb38d.png" width=100%  title="聊天界面" />



## :floppy_disk:项目结构

```
│  .gitignore
│  config.py               #模型配置参数
│  corpus.pth              #已经过处理的数据集
│  dataload.py             #dataloader
│  datapreprocess.py       #数据预处理
│  LICENSE
│  main.py               
│  model.py       
│  README.md
│  requirements.txt
│  train_eval.py            #训练和验证,测试
│  
├─checkpoints              
│      chatbot_0509_1437   #已经训练好的模型
│      
├─clean_chat_corpus
│      qingyun.tsv         #语料库
│      
├─QA_data
│      QA.db               #知识库
│      QA_test.py          #使用知识库时调用
│      stop_words.txt      #停用词
│      __init__.py
│      
└─utils
        beamsearch.py      #to do 未完工
        greedysearch.py    #贪婪搜索，用于测试
        __init__.py
```



## :couple:依赖库

![torch](https://img.shields.io/badge/torch-1.0.1-orange.svg)
![torchnet](https://img.shields.io/badge/torchnet-0.0.4-brightgreen.svg)
![fire](https://img.shields.io/badge/fire-0.1.3-red.svg)
![jieba](https://img.shields.io/badge/jieba-0.39-blue.svg)

安装依赖

```shell
$ pip install -r requirements.txt
```



## :sparkling_heart:开始使用

### 数据预处理(可省略)

```shell
$ python datapreprocess.py
```

对语料库进行预处理，产生corpus.pth （**这里已经上传好corpus.pth, 故此步可以省略**）

可修改参数:

```
# datapreprocess.py
corpus_file = 'clean_chat_corpus/qingyun.tsv' #未处理的对话数据集
max_voc_length = 10000 #字典最大长度
min_word_appear = 10 #加入字典的词的词频最小值
max_sentence_length = 50 #最大句子长度
save_path = 'corpus.pth' #已处理的对话数据集保存路径
```

### 使用

* 使用知识库

使用知识库时, 需要传入参数`use_QA_first=True` 此时，对于输入的字符串，首先在知识库中匹配最佳的问题和答案，并返回。找不到时，才调用聊天机器人自动生成回复。

这里的知识库是爬取整理的腾讯云官方文档中的常见问题和答案，100条，仅用于测试！

```shell
$ python main.py chat --use_QA_first=True
```

* 不使用知识库

由于课程设计需要，加入了腾讯云的问题答案对，但对于聊天机器人这个项目来说是无关紧要的，所以一般使用时，`use_QA_first=False`  ，该参数默认为`True`

```shell
$ python main.py chat --use_QA_first=False
```

* 使用默认参数

```shell
$ python main.py chat
```

* 退出聊天：输入`exit`, `quit`, `q`  均可

### 其他可配置参数

在`config.py` 文件中说明

需要传入新的参数时，只需要命令行传入即可，形如

```shell
$ python main.py chat --model_ckpt='checkpoints/chatbot_0509_1437' --use_QA_first=False
```

上面的命令指出了加载已训练模型的路径和是否使用知识库



## :cherry_blossom:技术实现

### 语料库

| 语料名称            | 语料数量 | 语料来源说明       | 语料特点         | 语料样例                                  | 是否已分词 |
| ------------------- | -------- | ------------------ | ---------------- | ----------------------------------------- | ---------- |
| qingyun（青云语料） | 10W      | 某聊天机器人交流群 | 相对不错，生活化 | Q:看来你很爱钱 A:噢是吗？那么你也差不多了 | 否         |

* 来源：<https://github.com/codemayq/chinese_chatbot_corpus>

### Seq2Seq

* Encoder：两层双向GRU
* Decoder：双层单向GRU

### Attention

* Global attention，采用dot计算分数
* Ref. https://arxiv.org/abs/1508.04025



## :construction_worker:模型训练与评估

```shell
$ python train_eval.py train [--options]
```

定量评估部分暂时还没写好，应该采用困惑度来衡量，目前只能生成句子，人为评估质量

```shell
$ python train_eval.py eval [--options]
```



## :sob:跳坑记录与总结

* 最深刻的体会就是“深度学习知识的了解和理解之间差了N个编程实现”。虽然理论大家都很清楚，但是真正到编程实现时，总会出这样，那样的问题：从数据集的处理，到许多公式的编程实现，到参数的调节，GPU配置等等各种问题
* 这次实践的过程实际是跟着PyTorch Tutorial先过了一遍Chatbot部分，跑通以后，再更换语料库，处理语料库，再按照类的风格去重构了代码，然后就是无尽的Debug过程，遇到了很多坑，尤其是把张量移到GPU上遇到各种问题，主要是不清楚to(device)时究竟移动了哪些。
  * 通过测试发现，model.to(device)只会把参数移到GPU，不会把类中定义的成员tensor移过去，所以如果在forward方法中定义了新的张量，要记得移动。
  * 还有就是移动的顺序问题：先把模型移动到GPU，再去定义优化器。以及移动的方法：model=model.to(device)，不要忘记赋值。 
  * 很容易出现GPU显存不足的情况，注意写代码时要考虑内存利用率问题，尽量减少重复tensor。
  * 在一开始更换中文语料库后，训练总是不收敛，最后才发现原来是batch_size设置小了，实际上我感觉batch_size在显存足够时要尽量大，其实之前看到过，只是写代码的时候完全忘记这回事了。说明自己当时看mini-batch时还不够理解，还是要真的写代码才能够深入人心，至少bug深入人心
  * 还有一个问题就是误解了torch.long，以为是高精度浮点，结果是int64型，造成了一个bug，找了好久才发现怎么回事。这告诉我们要认真看文档。
  * 最后的收获就是熟悉了如何实际实现一个模型，这很重要。
* 实际上这个模型的效果不是很好，除开模型本身的问题不谈，我发现分词的质量会严重影响句子的质量，但是分词时我连停用词还没设置，会出现一些奇特的结果
* 还有一个问题是处理变长序列时，损失函数如果用自己定义的，很容易出现不稳定情况，现在还在研究官方API
* 本次实践还发现自己对一些参数理解还不够深，不知道怎么调，还要补理论。
* 对模型的评估这部分还要继续做。

## :pray:致谢

* 官方的Chatbot Tutorial
  * <https://pytorch.org/tutorials/beginner/chatbot_tutorial.html>
* 提供中文语料库
  * <https://github.com/codemayq/chinese_chatbot_corpus> 
* 与官方的Chatbot Tutorial内容一致，但是有详尽的代码注释
  * <http://fancyerii.github.io/2019/02/14/chatbot/>
* 模型的写法和习惯均参考
  * <https://github.com/chenyuntc/pytorch-book>