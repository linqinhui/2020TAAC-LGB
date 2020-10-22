# 2020TAAC-LGB
赛题介绍-广告受众基础属性估：
这次腾讯的赛题啊，总结来说，就是利用用户的行为序列去反推用户的年龄和性别。
##赛题理解:
1.	数据
在比赛期间，主办方将为参赛者提供一组用户在长度为 90天（3 个月）的时间窗口内的广告点击历史记录作为训练数据集。
每条记录中包含了日期（从 1 到 91）、用户信息（年龄，性别），被点击的广告的信息（creative_id、ad_id、product_id、 category_id、advertiser_id、industury、product_category 等），以及该用户当天点击该广告的次数。测试数据集将会是另一组用户的广告点击历史记录。
![image](https://github.com/linqinhui/2020TAAC-LGB/blob/master/v2-06840c84e302411b49aee9432ecf155f_r.jpg) 

2.	目标
提供给参赛者的测试数据集中不会包含这些用户的年龄和性别信息。本赛题要求参赛者预测测试数据集中出现的用户的年龄和性别，并以约定的格式提交预测结果。
3.	评价指标
用户年龄预测的ACC+用户性别预测的ACC.

#一、机器学习方案：

##特征工程：
阶段一：统计特征
•	用户点击不同广告、产品、类别、素材、广告主的总数。
•	用户每天每条广告点击的平均次数，均值和方差等。
阶段二：词频统计特征
       在第一阶段的特征工程基础上，我们团队考虑词频统计特征，我们将考虑词频统计特征。我们将用户点击的一个广告看作一个word，把一个用户90天内点击的所有广告按时间排序后看作一个点击序列。然后使用NLP领域中的TF-IDF方法进行编码。我们把它用在用户点击的广告上面，可以体现用户点击的广告在他90天内点击的所有广告中的重要程度。

用户一：（[creative_id_1, creative_id_2, creative_id_3,……..]）,（[ad_id_1, ad_id_2, ad_id_3,……..]），（[advertiser_id_1, advertiser_id_2, advertiser_id_3,……..]），………..。

![image](https://github.com/linqinhui/2020TAAC-LGB/blob/master/微信图片_20201021213543.png)

由于每个广告有不同的属性: creative_id、ad_id、product_id、category_id、advertiser_id、 indusury_id .product_category等，并且每个用户点击的广告也有不同。这样每个用户可以得到六种文本输入。
但是TF-idf会有一个缺点，就是由于用户点击的广告类型过多，编码后会出现维度爆炸。
       

阶段三：W2V特征
       针对阶段二中的TF-IDF编码出现维度爆炸现象，我们选择NLP领域的另外一种方法word2vec。做法还是和TF-idf的一样。

w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=20, seed=2020, workers=64, min_count=1, iter=10)
其中input_docs是输入的是各个用户按照时间排序的点击序列。
size=输出的向量维度

#二、深度学习方案：
基于LSTM的多分类文本输入模型网络结构，使用的是pytorch深度学习框架。
整个模型由五部分构成：输入层-LSTM层-self-attention层-池化层-全连接层。
1.	输入层：我选了“creative_id”,”ad_id”,”advertiser_id”作为模型的输入，使用fasttext作为预训练模型。即预先训练各个id的fasttext模型 ,将预训练后的模型嵌入embedding层。
2.	LSTM ：我们是单独将各个id，embedding后，接入lstm+attention+池化层+全连接层，最后在多分类层进行拼接。
3.	Self-attention 
4.	池化层，使用max-pooling来减少模型参数。
5.	全连接层-作用就是分类。softmax

![image](https://github.com/linqinhui/2020TAAC-LGB/blob/master/微信图片_20201021211523.png) 
