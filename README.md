# 2020TAAC-LGB
赛题介绍-广告受众基础属性估：
本届算法大赛的题目来源于一个重要且有趣的问题。众所周知，像用户年龄和性
赛题理解:
1.	数据
在比赛期间，主办方将为参赛者提供一组用户在长度为 91 天（3 个月）的时间窗口内的广告点击历史记录作为训练数据集。
每条记录中包含了日期（从 1 到 91）、用户信息（年龄，性别），被点击的广告的信息（素材 id、广告 id、产品 id、产品类目 id、广告主id、广告主行业 id 等），
以及该用户当天点击该广告的次数。
测试数据集将会是另一组用户的广告点击历史记录。
2.	目标
提供给参赛者的测试数据集中不会包含这些用户的年龄和性别信息。
本赛题要求参赛者预测测试数据集中出现的用户的年龄和性别，并以约定的格式提交预测结果。
3.	评价指标
用户年龄预测的ACC+用户性别预测的ACC.
特征工程
阶段一：统计特征
•	用户点击不同广告、产品、类别、素材、广告主的总数。他可以反映用户的兴趣范围。
•	用户每天每条广告点击的平均次数，均值和方差。
阶段二：
在第一阶段的特征工程基础上，我们团队考虑词频统计特征，我们将考虑词频统计特征。
我们将用户点击的一个广告看作一个word，把一个用户90天内点击的所有广告按时间排序后看作一个点击序列。
然后使用NLP领域中的TF-IDF方法进行编码。我们把它用在点击的广告上面，可以体现用户在他90天内点击的所有广告中的重要程度。

用户一：（[素材_id_1, 素材_id_2, 素材_id_3,……..]）,（[广告_id_1, 广告_id_2, 广告_id_3,……..]），（[广告主_id_1, 广告主_id_2, 广告主_id_3,……..]），………..。

由于每个广告有不同的属性: 素材 id、广告 id、产品 id、产品类目 id、广告主id、广告主行业 id 等，并且每个用户点击的广告也有不同。这样每个用户可以得到八种文本输入。
但是TF-idf会有一个缺点，就是由于用户点击的广告类型过多，编码后会出现维度爆炸。
       
![image](https://img-blog.csdnimg.cn/20201021195751798.jpg)
阶段三：针对阶段二中的TF-IDF编码出现维度爆炸现象，我们选择NLP领域的另外一种方法word2vec。做法还是和TF-idf的一样。
