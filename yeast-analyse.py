import json
import math
import os
import re
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
# 打开数据集文件  这里是数据集在本地的路径
with open("C:/Users/lenovo/Desktop/ml/yeast.data", "r") as f1:
    with open("yeast1.csv", "w") as f:
        for i in f1:
            line = re.split("\s+", i)
            newline = ",".join(line)
            newline = newline.strip(",") + "\n"
            f.write(newline)


import pandas as pd
# 转换文件
data = pd.read_csv("yeast1.csv", sep=",", header=None,
                   names=["sequence name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"])

data_id = data["sequence name"]
data.drop("sequence name", axis=1, inplace=True)

# data.set_index("sequence name", inplace=True)

# 使用不同分类器只需更改关键指令即可
print("许松成20225773")
print("以下为最佳参数的模型准确率")

class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X, y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
# SMOTE算法采样
class grid():
    def __init__(self, model, use_smote=False):
        self.model = model
        self.use_smote = use_smote

    def grid_get(self, X, y, param_grid):
        if self.use_smote:
            # 如果使用SMOTE，创建SMOTE对象
            smote = SMOTE(sampling_strategy='auto', k_neighbors=3)
            # 在训练集上应用SMOTE
            X_resampled, y_resampled = smote.fit_resample(X, y)
            # 使用GridSearchCV在经过SMOTE处理后的数据上进行网格搜索
            grid_search = GridSearchCV(self.model, param_grid, cv=5)
            grid_search.fit(X_resampled, y_resampled)
        else:
            # 如果不使用SMOTE，直接使用原始数据进行网格搜索
            grid_search = GridSearchCV(self.model, param_grid, cv=5)
            grid_search.fit(X, y)

        # 打印最佳参数和分数
        print(grid_search.best_params_, grid_search.best_score_)
        # 打印CV结果
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
Y_train = data["class"]

# class Grid:
#     def __init__(self, model, use_smote=False, sampling_method=None):
#         self.model = model
#         self.use_smote = use_smote
#         self.sampling_method = sampling_method
#
#     def grid_get(self, X, y, param_grid):
#         if self.use_smote:
#             if self.sampling_method == 'nearmiss':
#                 sampler = NearMiss(version=2, n_neighbors=3)
#                 X_resampled, y_resampled = sampler.fit_resample(X, y)
#             else:
#                 # Handle other sampling methods if needed
#                 X_resampled, y_resampled = X, y
#
#             grid_search = GridSearchCV(self.model, param_grid, cv=5)
#             grid_search.fit(X_resampled, y_resampled)
#         else:
#             grid_search = GridSearchCV(self.model, param_grid, cv=5)
#             grid_search.fit(X, y)
#
#         print("Best Parameters:", grid_search.best_params_)
#         print("Best Score:", grid_search.best_score_)
#         print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
#
# # Assuming Y_train and X_train are defined previously
# Y_train = data["class"]
# data.drop("class", axis=1, inplace=True)
# X_train = data
#
# param_grid = {
#     'n_estimators': [100, 300, 500],
#     # Include other hyperparameters as needed
# }
#
# # Use NearMiss for undersampling
# grid = Grid(model=RandomForestClassifier(), use_smote=True, sampling_method='nearmiss')
# grid.grid_get(X_train, Y_train, param_grid)


# data.drop("class", axis=1, inplace=True)
# X_train = data
#
# param_grid = {
#     'learning_rate': [0.05, 0.1],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.8, 0.9, 1.0],
# }
#
# grid(XGBClassifier()).grid_get(X_train, y_train_encoded, param_grid)
