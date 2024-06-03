import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain
    
    def prediction(self, xTest):
        distance = {}
        counter = 0  # Start counter from 0

        for i in self.xTrain:
            distance[counter] = ((xTest[0] - i[0]) ** 2 + (xTest[1] - i[1]) ** 2) ** 0.5
            counter += 1

        distance = sorted(distance.items(), key=lambda item: item[1])
        self.classify(distance[:self.k])

    def classify(self, distance):
        label = []
        for i in distance:
            label.append(self.yTrain[i[0]])

        most_common_label = Counter(label).most_common(1)[0][0]
        print("Predicted label:", most_common_label)

data = pd.read_csv("C:\\Users\\nikhil deshmukh\\Desktop\\python\\knn\\Social_Network_Ads.csv")
x = data.iloc[:, 2:4].values
y = data.iloc[:, -1].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

st = StandardScaler()
xTrain = st.fit_transform(xTrain)
xTest = st.transform(xTest)

k = int(np.sqrt(xTrain.shape[0]))

knn = KNN(k)
knn.fit(xTrain, yTrain)


# Your existing code here...

user_input = []
age = int(32)
salary = int(150000)
user_input.append(age)
user_input.append(salary)


knn.prediction(user_input)
