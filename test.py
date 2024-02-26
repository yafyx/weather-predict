import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.tree

dataset = pd.read_csv("weather.csv", sep=";")
dataset.head()

print("Shape data (baris, kolom):" + str(dataset.shape))
print(dataset.info())

dataset.describe()

for col in dataset.columns:
    if dataset.dtypes[col] != "object":
        continue
    print("-" * 40 + col + "-" * 40, end=" - ")
    display(dataset[col].value_counts().head(10))

msno.matrix(dataset)

dataset["wind"].fillna(value=0, inplace=True)
dataset["direction"].fillna(value=0, inplace=True)
dataset["direction2"].fillna(value=0, inplace=True)

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
sns.boxplot(ax=axes[0], data=dataset["temperature"])
axes[0].set_title("Persentase Temperatur")
sns.boxplot(ax=axes[1], data=dataset["humidity"])
axes[1].set_title("Persentase Humidity")
sns.boxplot(ax=axes[2], data=dataset["wind"])
axes[2].set_title("Persentase Wind")
sns.boxplot(ax=axes[3], data=dataset["pressure"])
axes[3].set_title("Persentase Pressure")

plt.tight_layout()
plt.show()

Q1 = dataset[["wind", "temperature", "humidity"]].quantile(0.25)
Q3 = dataset[["wind", "temperature", "humidity"]].quantile(0.75)
IQR = Q3 - Q1
filter = (
    (dataset["wind"] >= Q1["wind"] - 1.5 * IQR["wind"])
    & (dataset["wind"] <= Q3["wind"] + 1.5 * IQR["wind"])
    & (dataset["temperature"] >= Q1["temperature"] - 1.5 * IQR["temperature"])
    & (dataset["temperature"] <= Q3["temperature"] + 1.5 * IQR["temperature"])
    & (dataset["humidity"] >= Q1["humidity"] - 1.5 * IQR["humidity"])
    & (dataset["humidity"] <= Q3["humidity"] + 1.5 * IQR["humidity"])
)
data_filtered = dataset.loc[filter]

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
sns.boxplot(ax=axes[0], data=data_filtered["temperature"])
axes[0].set_title("Persentase Temperatur")
sns.boxplot(ax=axes[1], data=data_filtered["humidity"])
axes[1].set_title("Persentase Humidity")
sns.boxplot(ax=axes[2], data=data_filtered["wind"])
axes[2].set_title("Persentase Wind")
sns.boxplot(ax=axes[3], data=data_filtered["pressure"])
axes[3].set_title("Persentase Pressure")

plt.tight_layout()
plt.show()

dataset.replace(
    [
        "north",
        "east",
        "south",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "north-northeast",
        "north-northwest",
        "south-southeast",
        "south-southwest",
        "east-northeast",
        "west-northwest",
        "east-southeast",
        "west-southwest",
    ],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    inplace=True,
)

for data in dataset["wind"]:
    if pd.isna(data):
        dataset.replace(data, "0", inplace=True)
        break


def getDataset():
    return dataset.drop(columns=["weather", "direction2"]), dataset["weather"]


def decisionTree(data=[], mode="classify"):
    model = sklearn.tree.DecisionTreeClassifier()
    if mode == "classify":
        xTrain, yTrain = getDataset()
        model.fit(xTrain.values, yTrain.values)
        predict = model.predict([data])
        return predict[0].upper()
    elif mode == "accuracy":
        xTrain, xTest, yTrain, yTest = getDatasetSplits()
        model.fit(xTrain, yTrain)
        classify = model.predict(xTest)
        accuracy = sklearn.metrics.accuracy_score(classify, yTest)
        return round(accuracy * 100, 2)


def getDatasetSplits():
    return sklearn.model_selection.train_test_split(
        dataset.drop(columns=["weather"]), dataset["weather"], test_size=0.5
    )


accuracy = decisionTree(mode="accuracy")
print("Accuracy:", accuracy)

# def decisionTree(data=[], mode="classify"):
#     model = sklearn.tree.DecisionTreeClassifier()
#     if mode == "classify":
#         xTrain, yTrain = getDataset()
#         model.fit(xTrain.values, yTrain.values)
#         predict = model.predict([data])
#         return predict[0].upper()
#     elif mode == "accuracy":
#         xTrain, xTest, yTrain, yTest = getDatasetSplits()
#         model.fit(xTrain, yTrain)
#         classify = model.predict(xTest)
#         accuracy = sklearn.metrics.accuracy_score(classify, yTest)

#         return round(accuracy * 100, 2)


# accuracy = decisionTree(mode="accuracy")

# predict = decisionTree(data=[13, 1, 9, 1, 320, 1011, 6], mode="classify")

# print("Cuaca:", predict)

# print("Accuracy:", accuracy)
