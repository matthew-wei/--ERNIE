#!/usr/bin/env python
# coding: utf-8

# # 千言情感分析比赛  - Gavin
# 本篇是深度学习的自然语言处理课程的大作业，基于paddle高级api的千言情感分析比赛。
# 
# 该比赛分为三种级别情感分析：句子级情感分类、评价对象级情感分类、观点抽取。下面就三种情况进行建模。
# 
# 相关介绍以及比赛可以参考：
# 
# [比赛和数据介绍](https://aistudio.baidu.com/aistudio/competition/detail/50)
# 
# [具体原理与分析](https://aistudio.baidu.com/aistudio/projectdetail/1968542)
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/40bb16c00e944e8a88a8aad62c6742f9734ef048e3824f67882ffd406762fa67)
# 
# 

# In[ ]:


get_ipython().system('pip install --upgrade paddlenlp -i https://pypi.org/simple ')
get_ipython().system('unzip -oq /home/aistudio/data/data94266/nlp_dataset.zip -d /home/aistudio/dataset')


# ## 1. 句子级情感分类
# | 数据集名称 | 训练集大小 | 开发集大小 | 测试集大小
# | -------- | -------- | -------- | -------- | 
# | ChnSentiCorp     | 9,600     |1,200			|1,200
# |NLPCC14-SC 	 |10,000 	 |/ 	 |2,500
# 
# ```
# ChnSentiCorp
# train:
# label	text_a
# 1	选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。
# 
# test:
# qid	text_a
# 0	这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
# 
# dev:
# qid	label	text_a
# 0	1	這間酒店環境和服務態度亦算不錯,但房間空間太小~~不宣容納太大件行李~~且房間格調還可以~~ 
# ```
# 
# ```
# NLPCC14-SC
# train:
# label	text_a
# 1	请问这机不是有个遥控器的吗？
# 
# test:
# qid	text_a
# 0	我终于找到同道中人啦～～～～从初中开始，我就已经喜欢上了michaeljackson.但同学们都用鄙夷的眼光看我。。。。
# ```

# In[ ]:


## 加载数据，预处理
from paddlenlp.datasets import load_dataset
def read(data_path, data_type='train'):
    if data_type=='train':
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                label, text = line.strip().split('\t')
                yield {'label': label, 'text': text}
    elif data_type=='dev':
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                qid, label, text = line.strip().split('\t')
                yield {'qid': qid, 'label': label, 'text': text}
    else:
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                qid, text = line.strip().split('\t')
                yield {'qid': qid, 'text': text}

# 加载两个数据集的数据
data_dict = {'ChnSentiCorp':{'test': load_dataset(read, data_path='dataset/ChnSentiCorp/test.tsv', data_type='test', lazy=False),
                             'train': load_dataset(read, data_path='dataset/ChnSentiCorp/train.tsv', data_type='train', lazy=False),
                             'dev': load_dataset(read, data_path='dataset/ChnSentiCorp/dev.tsv', data_type='dev', lazy=False)},
            'NLPCC14-SC': {'test': load_dataset(read, data_path='dataset/NLPCC14-SC/test.tsv', data_type='test', lazy=False),
                           'train': load_dataset(read, data_path='dataset/NLPCC14-SC/train.tsv', data_type='train', lazy=False)}
                           }
print(data_dict['ChnSentiCorp']['train'][0])
print(data_dict['NLPCC14-SC']['test'][0])


# ## 1.2 构造数据Dataloader（句子级）

# In[ ]:


# 借鉴 【NLP打卡营】
from functools import partial
import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddle.io import DataLoader
import numpy as np

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid



def get_data_loader(data_set, tokenizer, batch_size=32, max_seq_length=128, for_test=False):
    # 将数据处理成模型可读入的数据格式
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_test=for_test)

    data_set = data_set.map(trans_func)
    
    # 将数据组成批量式数据，如
    # 将不同长度的文本序列padding到批量式数据中最大长度
    # 将每条数据label堆叠在一起
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack()  # labels
    ): [data for data in fn(samples)]

    
    shuffle = True if not for_test else False
    if for_test:
        sampler = paddle.io.BatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)

    data_loader = DataLoader(dataset=data_set, batch_sampler=sampler, collate_fn=batchify_fn)

    return data_loader


# ## 1.3 搭建模型（句子级）

# In[ ]:


## 加载预训练模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

## 加载模型，数据标签只有 2种，0 、 1
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=2)
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()


# ## 1.3 训练模型（句子级）

# In[ ]:


import os
import time
import paddle.nn.functional as F
# 参数
batch_size = 64   # 批量数据大小
max_seq_length = 128  # 文本序列最大长度
epochs = 1
learning_rate = 2e-5
# 训练过程中保存模型参数的文件夹
ckpt_dir = "model/sentc"

## 优化
optimizer = paddle.optimizer.AdamW(learning_rate=learning_rate,
    parameters=model.parameters())   # Adam优化器
criterion = paddle.nn.loss.CrossEntropyLoss()     # 交叉熵损失函数
metric = paddle.metric.Accuracy()  # accuracy评价指标

# 数据
data_name = 'NLPCC14-SC'     # ChnSentiCorp      NLPCC14-SC
# print(data_dict[data_name]['train'][0])
## 数据相关
train_data_loader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_seq_length, for_test=False)
if data_name == 'ChnSentiCorp':
    dev_data_loader = get_data_loader(data_dict[data_name]['dev'], tokenizer, batch_size, max_seq_length, for_test=False)
else:
    dev_data_loader = None


# In[ ]:


## 训练
# 开启训练
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 10 == 0:
            save_dir = os.path.join(ckpt_dir, data_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 评估当前训练的模型
            if dev_data_loader:
                evaluate(model, criterion, metric, dev_data_loader)
            # 保存当前模型参数等
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
print('finish train!')


# ## 1.4 预测并保存结果（句子级）

# In[ ]:


import os
# data_name =  'ChnSentiCorp'     # ChnSentiCorp      NLPCC14-SC
test_data_loader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_seq_length, for_test=True)


# 根据实际运行情况，更换加载的参数路径
params_path = os.path.join(ckpt_dir, data_name + '/model_state.pdparams')
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)

label_map = {0: '0', 1: '1'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))



save_dir = {'ChnSentiCorp': './submission/ChnSentiCorp.tsv', 'NLPCC14-SC': './submission/NLPCC14-SC.tsv'}
res_dir = save_dir[data_name]
if not os.path.exists('./submission'):
    os.makedirs('./submission')
# 写入预测结果
with open(res_dir, 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for qid, label in results:
        f.write(str(qid[0])+"\t"+label+"\n")


# # 2 评价对象级情感分类
# 
# | 数据集名称 | 训练集大小 | 开发集大小 | 测试集大小
# | -------- | -------- | -------- | -------- | 
# | SE-ABSA16_PHNS     | 1,336    |/			|529
# |SE-ABSA16_CAME	 |1,317	 |/ 	 |505
# 
# 
# ```
# SE-ABSA16_PHNS
# train:
# label	text_a	text_b
# 1	phone#design_features	今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。
# 
# test:
# qid	text_a	text_b
# 0	software#usability	刚刚入手8600，体会。刚刚从淘宝购买，1635元（包邮）。1、全新，应该是欧版机，配件也是正品全新。2、在三星官网下载了KIES，可用免费软件非常多，绝对够用。3、不到2000元能买到此种手机，知足了。
# ```
# 
# ```
# SE-ABSA16_CAME
# train:
# label	text_a	text_b
# 0	camera#design_features	千呼万唤始出来，尼康的APSC小相机终于发布了，COOLPIX A. 你怎么看呢？我看，尼康是挤牙膏挤惯了啊，1，外观既没有V1时尚，也没P7100专业，反而类似P系列。2，CMOS炒冷饭。3，OVF没有任何提示和显示。（除了框框)4，28MM镜头是不错，可是F2.8定焦也太小气了。5，电池坑爹，用D800和V1的电池很难吗？6，考虑到1100美元的定价，富士X100S表示很欢乐。***好处是，可以确定，尼康会继续大力发展1系列了***另外体积比X100S小也算是A的优势吧***。等2014年年中跌倒1900左右的时候就可以入手了。
# 
# test:
# qid	text_a	text_b
# 0	camera#quality	一直潜水，昨天入d300s +35 1.8g，谈谈感受，dx说，标题一定要长！在我们这尼康一个代理商开的大型体验中心提的货，老板和销售mm都很热情，不欺诈，也没有店大欺客，mm很热情，从d300s到d800，d7000，到d3x配各种镜头，全部把玩了一番，感叹啊，真他妈好东西！尤其d3x，有钱了，一定要他妈买一个，还有，就是d800，一摸心中的神机，顿时凉了半截，可能摸她之前，摸了她们的头牌，d3x的缘故，这手感 真是差了点，样子嘛，之所以喜欢尼康，就是喜欢棱角分明的感觉，d3x方方正正 ，甚是讨喜，d800这丫头，变得圆滑了不少，不喜欢。都说电子产品，买新不买旧，我倒不认为这么看，中低端产品的确如此，但顶级的高端产品，真不是这么回事啊，d3x也是51点对焦，我的d300s也是51点，但明显感觉，对焦就是比d300s 快，准，暗部反差较小时，也很少拉风箱，我的d300s就不行，光线不好反差较小，拉回来拉过去，半天合不上焦，说真的，一分价钱一分货啊，d800电子性能 肯定是先进的，但机械性能 跟d3x还是没可比性，传感器固然先进，但三千多万 像素 和两千多万像素 对我们来说，真的差别这么大吗？d800e3万多，有这钱真的不如加点买 d3x啊，真要是d3x烂，为什么尼康不停产了？人说高像素 是给商业摄影师用，我们的音乐老师，是业余的音乐制作人，也拍摄一些商业广告，平时他玩的时候 都是数码什么的，nc 加起来十几个，大三元全都配齐，但干活的时候，还是120的机器，照他那话说，数码 像素太低，不够用啊！废话说太多了，谈谈感受吧，当初一直在纠结d7000和d300s，都说什么d7000画质超越d300s，我也信，但昨天拿到实机后，我瞬间就决定 d300s了，我的手算小的，握住d300s，我感觉，刚刚好，而且手柄凹槽 ，我觉得还不够深，握感不是十分的充盈，这点要像宾得k5学习，而且d7000小了一点，背部操作空间局促，大拇指没地放，果断d300s，而且试机的时候，我给d300s 换上了24-70，可能我练健身比较久了，没感觉有啥重量，蛮趁手的，现在配35 1.8 感觉轻飘飘的，哈哈，

# ## 2.1 数据提取预处理（评价对象级）

# In[ ]:


import json
import requests
import re
import os
import sys

num=392
def translator(str):
    """
    input : str 需要翻译的字符串
    output：translation 翻译后的字符串
    有每小时1000次访问的限制
    """
    global  num;
    num=num+1
    print("Program has process %d times "%num)
    # API
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数， i为要翻译的内容
    key = {
        'type': "AUTO",
        'i': str,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        # 通过 json.loads 把返回的结果加载成 json 格式
        result = json.loads(response.text)
#         print ("输入的词为：%s" % result['translateResult'][0][0]['src'])
#         print ("翻译结果为：%s" % result['translateResult'][0][0]['tgt'])
        translation = result['translateResult'][0][0]['tgt']
        return translation
    else:
        print("有道词典调用失败")
        # 相应失败就返回空
        return None


# In[ ]:


# tran_dict = {}
# def tranlation_text(texts):
#     if texts in tran_dict:
#         return tran_dict[texts]
#     else:
#         texts_list = texts.strip().split('#')
#         texts_list = [e.replace('_', ' ') for e in texts_list]
#         texts_chs_list = [translator(e) for e in texts_list]
#         texts_chs = '#'.join(texts_chs_list)
#         tran_dict[texts] = texts_chs
#     return texts_chs

# print(tranlation_text('phone#design_features'))


# In[ ]:


## 加载数据，预处理
import os
from paddlenlp.datasets import load_dataset
def read(data_path, data_type='train'):
    if data_type=='train':
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                label, text, text_pair = line.strip().split('\t')
                # text = tranlation_text(text)
                yield {'label': label, 'text': text, 'text_pair': text_pair}
    else:
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                qid, text, text_pair = line.strip().split('\t')
                # text = tranlation_text(text)
                yield {'qid': qid, 'text': text, 'text_pair': text_pair}

# 加载两个数据集的数据
data_dict = {'SE-ABSA16_PHNS':{'test': load_dataset(read, data_path='dataset/SE-ABSA16_PHNS/test.tsv', data_type='test', lazy=False),
                             'train': load_dataset(read, data_path='dataset/SE-ABSA16_PHNS/train.tsv', data_type='train', lazy=False)},
            'SE-ABSA16_CAME': {'test': load_dataset(read, data_path='dataset/SE-ABSA16_CAME/test.tsv', data_type='test', lazy=False),
                           'train': load_dataset(read, data_path='dataset/SE-ABSA16_CAME/train.tsv', data_type='train', lazy=False)}
                           }
print(data_dict['SE-ABSA16_CAME']['train'][0:5])
print(data_dict['SE-ABSA16_CAME']['test'][0])


# In[ ]:


# for name in data_dict.keys():
#     if not os.path.exists(name):
#         os.makedirs(name)
#     for part, a in {'train':'label', 'test':'qid'}.items():
#         res_dir = part + '.tsv'
#         res_dir = os.path.join(name, res_dir)
#         with open(res_dir, 'w', encoding="utf8") as f:
#             f.write(a+"\ttext_a\ttext_b\n")
#             for idx, text in enumerate(data_dict[name][part]):
#                 f.write('\t'.join(text.values())+"\n")
# print('fininsh save ch_text!')


# ## 2.2 准备DataLoader（评价对象级）

# In[ ]:


# 借鉴 【NLP打卡营】
from functools import partial
import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddle.io import DataLoader
import numpy as np

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    dataset_name="SE-ABSA16_PHNS"):
    encoded_inputs = tokenizer(
        text=example["text"],
        text_pair=example["text_pair"],
        max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

def get_data_loader(data_set, tokenizer, batch_size=32, max_seq_length=128, for_test=False):
    # 将数据处理成模型可读入的数据格式
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_test=for_test)
    
    if for_test:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        ): [data for data in fn(samples)]
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Stack(dtype="int64")  # labels
        ): [data for data in fn(samples)]

    data_set = data_set.map(trans_func)    
        
    # 将数据组成批量式数据，如
    # 将不同长度的文本序列padding到批量式数据中最大长度
    # 将每条数据label堆叠在一起
    shuffle = True if not for_test else False
    if for_test:
        sampler = paddle.io.BatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)

    data_loader = DataLoader(dataset=data_set, batch_sampler=sampler, collate_fn=batchify_fn)

    return data_loader


# In[ ]:


## 加载预训练模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
## 加载模型，数据标签只有 2种，0 、 1
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch", num_classes=2)
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_1.0_large_ch")



@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for batch in data_loader:
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


# ## 2.3 训练评价对象级模型

# In[ ]:


import os
import time
import paddle.nn.functional as F
# 参数
batch_size = 16   # 批量数据大小
max_seq_length = 512  # 文本序列最大长度
epochs = 3
learning_rate = 2e-5
# 训练过程中保存模型参数的文件夹
ckpt_dir = "model/tager"

paraparam_ = {'SE-ABSA16_PHNS': [{'batch_size': 8, 'max_seq_length':512, 'epochs':3, 'learning_rate': 2e-5},
                                 {'batch_size': 16, 'max_seq_length':512, 'epochs':3, 'learning_rate': 2e-5}]
                                 }

## 优化
optimizer = paddle.optimizer.AdamW(learning_rate=learning_rate,
    parameters=model.parameters())   # Adam优化器
criterion = paddle.nn.loss.CrossEntropyLoss()     # 交叉熵损失函数
metric = paddle.metric.Accuracy()  # accuracy评价指标

# 数据
data_name = 'SE-ABSA16_CAME'     # SE-ABSA16_PHNS     SE-ABSA16_CAME
# print(data_dict[data_name]['train'][0])
## 数据相关
train_data_loader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_seq_length, for_test=False)
dev_data_loader = None


# In[6]:


global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 50 == 0:
            save_dir = os.path.join(ckpt_dir, data_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)


# ## 2.4 预测和输出（评价对象级）

# In[7]:


label_map = {0: '0', 1: '1'}
# data_name = 'SE-ABSA16_CAME'     # SE-ABSA16_PHNS     SE-ABSA16_CAME
test_data_loader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_seq_length, for_test=True)

params_path = os.path.join(ckpt_dir, data_name + '/model_state.pdparams')
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)

results = predict(model, test_data_loader, label_map)

# 写入预测结果
save_dir = {'SE-ABSA16_PHNS': './submission/SE-ABSA16_PHNS.tsv', 'SE-ABSA16_CAME': './submission/SE-ABSA16_CAME.tsv'}
res_dir = save_dir[data_name]
if not os.path.exists('./submission'):
    os.makedirs('./submission')
# 写入预测结果
with open(res_dir, 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for idx, label in enumerate(results):
        f.write(str(idx)+"\t"+label+"\n")
print('fininsh predict!')


# # 3 观点抽取
# 观点抽取的情感分析参考
# 
# Jordan的项目：[基于Skep模型的情感分析比赛](https://aistudio.baidu.com/aistudio/projectdetail/2099332)
# 
# 『NLP打卡营』实践课5：[情感分析预训练模型SKEP](https://aistudio.baidu.com/aistudio/projectdetail/1968542)
# 
# 『NLP打卡营』实践课3：[使用预训练模型实现快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1329361)
# 
# 
# | 数据集名称 | 训练集大小 | 开发集大小 | 测试集大小
# | -------- | -------- | -------- | -------- | 
# |COTE-BD      | 8,533    |/			|3658
# |COTE-MFW	 |41,253	 |/ 	 |17,681
# |COTE-DP	 |25,258	 |/ 	 |10,825
# 
# 
# ```
# COTE-BD 
# train:
# label	text_a
# 芝罘岛	芝罘岛骑车去过几次，它挺壮观的，毕竟是我国典型的也是最大的陆连岛咯!我喜欢去那儿，反正全岛免费咯啊哈哈哈！风景的确不错而且海水也很干净，有些地方还是军事管理，禁地来着，但是我认识军官。
# 
# test:
# qid	text_a
# 0	毕棚沟的风景早有所闻，尤其以秋季的风景最美，但是这次去晚了，红叶全掉完了，黄叶也看不到了，下了雪只能看看雪山了，还好雪山的雄伟确实值得一看。
# ```
# 
# ```
# COTE-MFW
# train:
# label	text_a
# 恩施大峡谷	秀美恩施大峡谷，因其奇、险让人流连忘返。
# 
# test:
# qid	text_a
# 0	神女溪据说在山峡蓄水前就是条很清澈的小溪，蓄水后很多遗迹都淹没在水底了，这里的水确实和外面黄黄的水不一样。
# ```
# 
# ```
# COTE-DP
# train:
# label	text_a
# 重庆老灶火锅	重庆老灶火锅还是很赞的，有机会可以尝试一下！
# 
# test:
# qid	text_a
# 0	还是第一次进星巴克店里吃东西 那会儿第一次喝咖啡还是外带的
# ```

# ## 3.1 数据提取预处理（观点抽取）

# In[ ]:


## 加载数据，预处理
from paddlenlp.datasets import load_dataset
def read(data_path, data_type='train'):
    if data_type=='train':
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                line_str_tmp = line.strip().split('\t')
                if len(line_str_tmp) == 2:
                    label, text = line_str_tmp
                yield {'label': label, 'text': text}
    else:
        with open(data_path, 'r', encoding='utf8') as f:
            for line in f.readlines()[1:]:
                qid, text = line.strip().split('\t')
                yield {'qid': qid, 'text': text}

# 加载两个数据集的数据
data_dict = {'COTE-BD':{'test': load_dataset(read, data_path='dataset/COTE-BD/test.tsv', data_type='test', lazy=False),
                             'train': load_dataset(read, data_path='dataset/COTE-BD/train.tsv', data_type='train', lazy=False)},
            'COTE-MFW':{'test': load_dataset(read, data_path='dataset/COTE-MFW/test.tsv', data_type='test', lazy=False),
                             'train': load_dataset(read, data_path='dataset/COTE-MFW/train.tsv', data_type='train', lazy=False)},
            'COTE-DP': {'test': load_dataset(read, data_path='dataset/COTE-DP/test.tsv', data_type='test', lazy=False),
                           'train': load_dataset(read, data_path='dataset/COTE-DP/train.tsv', data_type='train', lazy=False)}
                           }
print(data_dict['COTE-BD']['train'][0])
print(data_dict['COTE-BD']['test'][0])


# ## 3.2 准备DataLoader（观点抽取）

# In[ ]:


# 借鉴 【NLP打卡营】
from functools import partial
import paddle
from paddlenlp.data import Stack, Tuple, Pad
from paddle.io import DataLoader
import numpy as np


label_list = {'B': 0, 'I': 1, 'O': 2}
index2label = {0: 'B', 1: 'I', 2: 'O'}
no_entity_label_idx = label_list.get("O", 2)

## 参考
def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
    text = example['text']
    if is_test:
        # qid = example['qid']
        token_res = tokenizer.encode(text, max_seq_len=max_seq_length)
        origin_enc = token_res['input_ids']
        token_type_ids = token_res['token_type_ids']
        # seq_len = token_res['seq_len']
        return np.array(origin_enc, dtype='int64'), np.array(token_type_ids, dtype='int64')
    else:
        label = example['label']
        # 由于并不是每个字都是一个token，这里采用一种简单的处理方法，先编码label，再编码text中除了label以外的词，最后合到一起
        texts = text.split(label)
        label_enc = tokenizer.encode(label)['input_ids']
        cls_enc = label_enc[0]
        sep_enc = label_enc[-1]
        label_enc = label_enc[1:-1]
        # 合并
        origin_enc = []
        label_ids = []
        for index, text in enumerate(texts):
            text_enc = tokenizer.encode(text)['input_ids']
            text_enc = text_enc[1:-1]
            origin_enc += text_enc
            label_ids += [label_list['O']] * len(text_enc)
            if index != len(texts) - 1:
                origin_enc += label_enc
                label_ids += [label_list['B']] + [label_list['I']] * (len(label_enc) - 1)

        origin_enc = [cls_enc] + origin_enc + [sep_enc]
        label_ids = [label_list['O']] + label_ids + [label_list['O']]    
        # 截断
        if len(origin_enc) > max_seq_length:
            origin_enc = origin_enc[:max_seq_length-1] + origin_enc[-1:]
            label_ids = label_ids[:max_seq_length-1] + label_ids[-1:]
        
        token_type_ids = [0] * len(label_ids)

        return np.array(origin_enc, dtype='int64'), np.array(token_type_ids, dtype='int64'), np.array(label_ids, dtype='int64')


def get_data_loader(data_set, tokenizer, batch_size=32, max_len=512, for_test=False):

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_test=for_test)
    
    if for_test:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        ): [data for data in fn(samples)]
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
            Pad(axis=0, pad_val=label_list['O'])  # labels
        ): [data for data in fn(samples)]

    data_set = data_set.map(trans_func)    
        
    # 将数据组成批量式数据，如
    # 将不同长度的文本序列padding到批量式数据中最大长度
    # 将每条数据label堆叠在一起
    shuffle = True if not for_test else False
    if for_test:
        sampler = paddle.io.BatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=data_set, batch_size=batch_size, shuffle=shuffle)

    data_loader = DataLoader(dataset=data_set, batch_sampler=sampler, collate_fn=batchify_fn)

    return data_loader    


# In[ ]:


# 模型和分词
import paddlenlp
from paddlenlp.transformers import SkepForTokenClassification, SkepTokenizer
model = SkepForTokenClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=3)
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

# import paddlenlp
# from paddlenlp.transformers import SkepTokenizer, SkepModel, SkepCrfForTokenClassification
# skep = SkepModel.from_pretrained('skep_ernie_1.0_large_ch')
# model = SkepCrfForTokenClassification(
#     skep, num_classes=3)
# tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')


# ## 3.3 训练观点提取模型

# In[ ]:


import os
import time
import paddle.nn.functional as F
from paddlenlp.metrics import Perplexity
from paddlenlp.metrics import ChunkEvaluator

# 数据
data_name = 'COTE-BD'     # COTE-BD    COTE-MFW    COTE-DP

# 参数
paramers = {'COTE-BD': [{'batch_size': 8, 'max_seq_length': 512, 
                         'epochs': 1, 'learning_rate': 2e-5},
                         {'batch_size': 8, 'max_seq_length': 512, 
                         'epochs': 4, 'learning_rate': 5e-5}],
            'COTE-MFW': {'batch_size': 8, 'max_seq_length': 512, 
                         'epochs': 1, 'learning_rate': 5e-5},
            'COTE-DP': [{'batch_size': 8, 'max_seq_length': 512, 
                         'epochs': 1, 'learning_rate': 5e-5},
                         {'batch_size': 16, 'max_seq_length': 512, 
                         'epochs': 2, 'learning_rate': 2e-5}]
                         }

batch_size = 16   # 批量数据大小
max_seq_length = 512  # 文本序列最大长度
epochs = 1
learning_rate = 2e-5
# 训练过程中保存模型参数的文件夹
ckpt_dir = "model/point"

## 优化
metric = ChunkEvaluator(label_list=label_list.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss()
optimizer = paddle.optimizer.AdamW(learning_rate=learning_rate, parameters=model.parameters())


# print(data_dict[data_name]['train'][0])
## 数据相关
train_data_loader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_seq_length, for_test=False)
dev_data_loader = None


# In[ ]:


global_step = 0 
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, (input_ids, token_type_ids, labels) in enumerate(train_data_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))

    
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()

        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            save_dir = os.path.join(ckpt_dir, data_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)


# ## 3.4 预测和输出（观点提取）

# In[ ]:


# data_name = 'COTE-BD'     # COTE-BD    COTE-MFW    COTE-DP
test_data_loader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_seq_length, for_test=True)


# In[ ]:


params_path = os.path.join(ckpt_dir, data_name + '/model_state.pdparams')
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)


@paddle.no_grad()
def predict(model, data_loader):
    model.eval()
    pred_list = []
    input_ids_list = []
    for batch in data_loader:
        input_ids, token_type_ids = batch
        logits = model(input_ids, token_type_ids)
        pred = paddle.argmax(logits, axis=-1).numpy()
        pred = pred.tolist()
        input_ids = input_ids.numpy().tolist()
        pred_list.extend(pred)
        input_ids_list.extend(input_ids)

    return pred_list, input_ids_list


predicts, input_ids_list = predict(model, test_data_loader)

def find_entity(prediction, input_ids):
    entity = []
    entity_ids = []
    for index, idx in enumerate(prediction):
        if idx == label_list['B']:
            entity_ids = [input_ids[index]]
        elif idx == label_list['I']:
            if entity_ids:
                entity_ids.append(input_ids[index])
        elif idx == label_list['O']:
            if entity_ids:
                entity.append(''.join(tokenizer.convert_ids_to_tokens(entity_ids)))
                entity_ids = []
    return entity

import re
# 写入预测结果
save_dir = {'COTE-BD': './submission/COTE_BD.tsv', 'COTE-MFW': './submission/COTE_MFW.tsv', 'COTE-DP': './submission/COTE_DP.tsv'}
res_dir = save_dir[data_name]
if not os.path.exists('./submission'):
    os.makedirs('./submission')
# 写入预测结果
with open(res_dir, 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for idx, prediction in enumerate(predicts):
        entity = find_entity(prediction, input_ids_list[idx])
        entity = list(set(entity))  # 去重
        entity = [re.sub('##', '', e) for e in entity]  # 去除英文编码时的特殊符号
        entity = [re.sub('[UNK]', '', e) for e in entity]  # 去除未知符号
        entity = [re.sub('\"', '', e) for e in entity]  # 去除引号
        f.write(str(idx) + '\t' + '\x01'.join(entity) + '\n')
print('fininsh predict!')


# In[8]:


get_ipython().run_line_magic('cd', 'submission')
get_ipython().system('zip -r ../submission.zip *')


# 以上实现基于PaddleNLP，开源不易，希望大家多多支持~ 
# 
# **记得给[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)点个小小的Star⭐，及时跟踪最新消息和功能哦**
# 
# GitHub地址：[https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
