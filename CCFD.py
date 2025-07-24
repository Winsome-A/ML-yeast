import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. 读取信用卡欺诈检测数据集
data = pd.read_csv("C:/Users/lenovo/Desktop/ml/CCFD.csv")

# 2. 对 'Amount' 特征进行标准化（其他特征已 PCA）
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# 3. 特征和标签分离
X = data.drop(columns=['Class', 'Time'])  # 'Time' 特征不需要用于训练
y = data['Class']

# 4. 显示样本分布
print("类别分布：")
print(y.value_counts())
print("许松成20225773")
print("以下为最佳参数的模型准确率")

# 5. 网格搜索类（支持 SMOTE）
class grid():
    def __init__(self, model, use_smote=False):
        self.model = model
        self.use_smote = use_smote

    def grid_get(self, X, y, param_grid):
        if self.use_smote:
            smote = SMOTE(sampling_strategy='auto', k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1')  # 适合不均衡任务
            grid_search.fit(X_resampled, y_resampled)
        else:
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1')
            grid_search.fit(X, y)

        print("最佳参数：", grid_search.best_params_)
        print("最佳F1得分：", grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])

# 6. 示例模型调用（Random Forest）
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]
}

model = RandomForestClassifier(random_state=42)
g = grid(model, use_smote=True)
g.grid_get(X, y, rf_params)

"""
##切换模型
# 支持向量机
svm_params = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
model = SVC()
g = grid(model, use_smote=True)
g.grid_get(X, y, svm_params)

# XGBoost 示例
xgb_params = {'n_estimators': [100], 'max_depth': [4, 6]}
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
g = grid(model, use_smote=True)
g.grid_get(X, y, xgb_params)
"""

