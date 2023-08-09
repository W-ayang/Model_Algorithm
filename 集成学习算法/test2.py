import pandas as pd
#训练与测试集的分类 
from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier
url = " "
df = pd.read_csv(url, header = None)
# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
# 构建随机森林分类器
forest = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=-1)
# 喂入数据
forest.fit(x_train, y_train)