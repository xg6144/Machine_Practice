#다중회귀
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

df = pd.read_csv('https://bit.ly/perch_csv')
perch_full = df.to_numpy()

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state= 42)

#poly = PolynomialFeatures()
#poly.fit([[2,3]])
#print(poly.transform([[2,3]])) 1은 항상 절편에 곱해지는 수 사이킷런 모델은 자동으로 무시한다.

#poly = PolynomialFeatures(include_bias=False)
#poly.fit([[2,3]])
#print(poly.transform([[2,3]]))

#poly.fit(train_input)
#train_poly = poly.transform(train_input)
#print(train_poly.shape, poly.get_feature_names())

#test_poly = poly.transform(test_input)

#lr = LinearRegression()
#lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target))
#print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
#print(train_poly.shape)
lr = LinearRegression()
lr.fit(train_poly, train_target)
#print(lr.score(train_poly, train_target))

#print(lr.score(test_poly, test_target))
# 특성의 개수를 늘리면 선형 모델은 굉장히 강력해진다. 그러나 이런 모델은 훈련 세트에 대해 과대적합 되므로 테스트세트 점수는 기대하기 어렵다.

#규제

ss = StandardScaler() #표준점수 변환
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#Ridge 계수를 제곱한 값을 기준으로 규제
#Lasso 계수의 절대값을 기준으로 규제

#ridge = Ridge()
#ridge.fit(train_scaled, train_target)

#print(ridge.score(train_scaled, train_target))
#print(ridge.score(test_scaled, test_target))

#train_score = []
#test_score = []

#alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
#for alpha in alpha_list:
    #ridge = Ridge(alpha=alpha)
    #ridge.fit(train_scaled, train_target)
    #train_score.append(ridge.score(train_scaled, train_target))
    #test_score.append(ridge.score(test_scaled, test_target))

#plt.plot(np.log10(alpha_list), train_score)
#plt.plot(np.log10(alpha_list), test_score)
#plt.show()

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
#print(ridge.score(train_scaled, train_target))
#print(ridge.score(test_scaled, test_target))

lasso = Lasso()
lasso.fit(train_scaled, train_target)
#print(lasso.score(train_scaled, train_target))
#print(lasso.score(test_scaled, test_target))

#train_score = []
#test_score = []
#alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
#for alpha in alpha_list:
#    lasso = Lasso(alpha=alpha, max_iter=10000)
#    lasso.fit(train_scaled, train_target)
#    train_score.append(lasso.score(train_scaled, train_target))
#    test_score.append(lasso.score(test_scaled, test_target))

#plt.plot(np.log10(alpha_list), train_score)
#plt.plot(np.log10(alpha_list), test_score)
#plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
#print(lasso.score(train_scaled, train_target))
#print(lasso.score(test_scaled, test_target))

print(np.sum(lasso.coef_ == 0))
