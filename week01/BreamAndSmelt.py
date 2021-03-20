import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
                #도미의 길이

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
                #도미의 무게

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
#plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length,weight)]
fish_target = [1] * 35 + [0] * 14

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

kn.score(fish_data, fish_target)

#print(kn.predict([[30, 600]]))

#print(kn._fit_X) k-최근접 이웃 알고리즘은 모든 x와 y값을 가지고 있다. 그래서 아주 많은 데이터를 상대로 사용하기에는 부적절하다.
#fit()메서드에 전달한 데이터를 모두 저장하고 있다가 새로운 데이터가 등장하면 가장 가까운 데이터를 참고하여 구분한다.
#기본값으로 가장 가까운 데이터 5개를 참고한다.

kn49 = KNeighborsClassifier(n_neighbors=49) #참고데이터를 49개로 한 kn49모델

kn49.fit(fish_data, fish_target)
# print(kn49.score(fish_data, fish_target)) #데이터 49개 중 도미가 35개를 차지하므로 어떤 데이터를 넣어도 도미로 예측한다.
# print(35/49) score와 같은 값이 출력된다.
# 이 데이터에서는 참고데이터를 49개로 하는 것은 부적절하다.