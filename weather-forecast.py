import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 載入資料
data = pd.read_csv("weatherAUS.csv")

# 刪除目標欄位為空的資料筆
data = data.dropna(subset=["RainTomorrow"])

# 處理缺失值，這裡我們將NA值替換為該列的平均值
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
for col in numeric_cols:
    data.loc[:, col] = data.loc[:, col].fillna(data[col].mean())

# 將日期轉換為自1970年1月1日以來的天數
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = (data["Date"] - pd.Timestamp("1970-01-01")).dt.days

# 將類別型欄位轉換為數值
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if col != "RainTomorrow":  # 我們不轉換目標欄位
        le = LabelEncoder()
        data.loc[:, col] = le.fit_transform(data[col].astype(str))

# 定義特徵和標籤
X = data.drop(["RainTomorrow"], axis=1)  # 特徵
y = data["RainTomorrow"].map({"No": 0, "Yes": 1})  # 標籤

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 建立隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測測試集
predictions = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, predictions)
print(f"準確率: {accuracy:.2f}")
