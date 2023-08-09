# 导入需要的库
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
 
le = LabelEncoder()
label_mapping = {0: 'AD', 1: 'CN', 2: 'EMCI', 3: 'LMCI', 4: 'SMC'}
X = pd.DataFrame(pd.read_excel('D:\code\python\集成学习算法\DataX.xlsx')).values  # 输入特征
y = pd.DataFrame(pd.read_excel('D:\code\python\集成学习算法\DataY.xlsx')).values  # 目标变量
y = le.fit_transform(y)
# 将数据分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)
# 训练XGBoost分类器
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# xgb.plot_tree(model)
# 使用测试数据预测类别
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# 输出混淆矩阵
for i, true_label in enumerate(label_mapping.values()):
    row = ''
    for j, pred_label in enumerate(label_mapping.values()):
        row += f'{cm[i, j]} ({pred_label})\t'
    print(f'{row} | {true_label}')
 
# 输出混淆矩阵
print(classification_report(y_test, y_pred,target_names=['AD', 'CN', 'EMCI', 'LMCI', 'SMC']))  # 输出混淆矩阵
print("Accuracy:")
print(accuracy_score(y_test, y_pred))
 
 
# label_names 是分类变量的取值名称列表
label_names = ['AD', 'CN', 'EMCI', 'LMCI', 'SMC']
cm = confusion_matrix(y_test, y_pred)
 
# 绘制混淆矩阵图
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=label_names, yticklabels=label_names,
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')
 
# 在矩阵图中显示数字标签
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")


fig.tight_layout()
plt.show()
plt.savefig('XGBoost_Conclusion.png', dpi=300)
# 上面的代码首先计算混淆矩阵，然后使用 matplotlib 库中的 imshow 函数将混淆矩阵可视化，最后通过 text 函数在混淆矩阵上添加数字，并使用 show/savefig 函数显示图像。