import numpy as np
import pandas as pd
import os
import gc #垃圾回收模块
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] #显示中文字符
from sklearn.metrics import log_loss
#from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter
from sklearn.cross_validation import train_test_split
import xgboost as xgb

pal = sns.color_palette()

# print("#文件大小")
# for f in os.listdir('C:/Users/user/Documents/pythonRelated/kaggle/quora question pairs/'):
#     if 'zip' not in f:
#         print(f.ljust(30) + str(round(os.path.getsize('C:/Users/user/Documents/pythonRelated/kaggle/quora question pairs/' + f) / 1000000, 2)) + 'MB')

df_train = pd.read_csv('C:/Users/user/Documents/pythonRelated/kaggle/quora question pairs/train.csv')
# print(df_train[0:5])

#数据可视化
# print("问题总对数：{}".format(len(df_train)))
# print("重复问题对数百分比：{}%".format(round(df_train['is_duplicate'].mean()*100, 2)))
# qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
# print("训练样本中的问题数量：{}".format(len(np.unique(qids))))
# print("出现多次的问题：{}".format(np.sum(qids.value_counts() > 1)))

# plt.figure(figsize = (12, 5))
# plt.hist(qids.value_counts(), bins = 50)
# plt.yscale('log', nonposy = 'clip')
# plt.title("问题出现次数-直方图")
# plt.xlabel("问题出现的次数")
# plt.ylabel("问题数量")

p = df_train['is_duplicate'].mean()
print("预测所得分数：", log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))  #逻辑回归函数

#基础输出，测试
# df_test = pd.read_csv('C:/Users/user/Documents/pythonRelated/kaggle/quora question pairs/test.csv')
# sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
# sub.to_csv('naive_submission.csv', index = False)
# print(sub[0:5])

#测试样本
df_test = pd.read_csv('C:/Users/user/Documents/pythonRelated/kaggle/quora question pairs/test.csv')
#print(df_test[0:5])
#print("测试样本中问题总对数：{}".format(len(df_test)))

#每个问题下字母数量分析
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
# plt.figure(figsize=(15,10))
# plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('每个问题的字数', fontsize=15)
# plt.legend()
# plt.xlabel('字母数量', fontsize=15)
# plt.ylabel('概率', fontsize=15)

# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
# 	                                        dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))

#每个问题下单词数量分析
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

# plt.figure(figsize=(15, 10))
# plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('每个问题下单词数量', fontsize=15)
# plt.legend()
# plt.xlabel('单词数量', fontsize=15)
# plt.ylabel('概率', fontsize=15)

# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
#                           dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))

#cloud = WordCloud(width = 1440, height = 1080).generate(" ".join(train_qs.astype(str)))
#plt.figure(figsize=(20, 15))
#plt.imshow(cloud)
#plt.axis('off')

#文本分析
#特征分析
#nltk.data.path.append('path_to_nltk_data')

stops =set(stopwords.words("english"))

def word_match_share(row):
	q1words = {}
	q2words = {}
	for word in str(row['question1']).lower().split():
		if word not in stops:
			q1words[word] = 1
	for word in str(row['question2']).lower().split():
		if word not in stops:
			q2words[word] = 1
	if len(q1words) == 0 or len(q2words) == 0:
		return 0
	sharedWordsq1 = [w for w in q1words.keys() if w in q2words]
	sharedWordsq2 = [w for w in q2words.keys() if w in q1words]
	R = (len(sharedWordsq2) + len(sharedWordsq1))/(len(q2words) + len(q1words))
	return R

plt.figure(figsize=(15,5))
train_word_match = df_train.apply(word_match_share, axis = 1, raw = True)
# plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
# plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)

#TF-IDF (term frequency-inverse document frequency)
# def 

# XGBoost算法
# Rebalancing the Data
x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis = 1, raw = True)

y_train = df_train['is_duplicate'].values

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]
print(pos_train)

p = 0.165
scale = ((len(pos_train)/(len(pos_train) + len(neg_train)))/p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -= 1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train)/(len(pos_train) + len(neg_train)))


x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

# 交叉验证
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 4242)

#XGBoost算法
params = {}
params['objective'] = 'binary:logistic'
params['eva_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label = y_train)
d_valid = xgb.DMatrix(x_valid, label = y_valid)

watch_list = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watch_list, early_stopping_rounds = 50, verbose_eval = 10)

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)