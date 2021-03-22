import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

fish = pd.read_csv('https://bit.ly/fish_csv')
#print(fish.head())
#print(pd.unique(fish['Species']))
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', "Width"]].to_numpy()
#print(fish_input[:5])
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
#print(kn.score(train_scaled, train_target))
#print(kn.score(test_scaled, test_target))
#다중분류

#print(kn.classes_)
#print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=4))

distances, indexes = kn.kneighbors(test_scaled[3:4])
#print(train_target[indexes])

#z = np.arange(-5, 5, 0.1)
#phi = 1 / (1+np.exp(-z))
#plt.plot(z, phi)
#plt.show()

#char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
#print(char_arr[[True, False, True, False, False]])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
#print(lr.predict(train_bream_smelt[:5]))
#print(lr.predict_proba(train_bream_smelt[:5]))
#print(lr.classes_)

#print(lr.coef_, lr.intercept_)

#decisions = lr.decision_function(train_bream_smelt[:5])
#print(decisions)

from scipy.special import expit, softmax
#print(expit(decisions))

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
#print(lr.score(train_scaled, train_target))
#print(lr.score(test_scaled, test_target))
#print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
#print(np.round(proba, decimals=3))

#print(lr.classes_)
#print(lr.coef_.shape, lr.intercept_.shape)
#다중분류에서는 시그모이드가 아닌 소프트맥스 함수를 사용하여 z값을 반환한다.
#소프트맥스 함수는 여러 개의 선형방정식의 출력값을 0~1사이로 압축하고 전체 합이 1이 되도록 만든다.

decision = lr.decision_function(test_scaled[:5])
#print(np.round(decision, decimals=2))

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))



