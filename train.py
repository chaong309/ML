## 개인 프로젝트 진행
- ### 주제: DNN 모델과 기존 머신러닝 모델간의 수치 예측 성능 비교
- ### 사용하는 데이터: 치안안전도 예측 프로젝트를 위해 만들어 둔 데이터
    > 17년 상반기부터 19년 하반기까지의 41개 경찰서의 자료로, 타겟은 총 41 * 6 = 246개의 관측치 존재 
    > 이에 맞게 제공받은 데이터들로 타겟의 틀에 맞게 데이터를 가공해서 넣어줌 
    > group by, pivot 등 사용하고 집계함수 이용해 Dataset 생성
    > Google Colab을 이용해 작업을 진행하였습니다.
    
``` python
## 필요모듈 임포트
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns
### 코랩 내에 catboost가 없으므로 install
!pip install catboost
from catboost import CatBoostRegressor
```

### Dataset 확인
``` python
dataset = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/dataset.csv', encoding = 'utf-8') # feature + target 데이터 프레임
total = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/total_data.csv', encoding = 'utf-8') # feature들만 있는 데이터 프레임

print('dataset: feature + target!!')
print('')
print('dataset의 shape은 {}'.format(dataset.shape))
print('')
print('dataset은 {}개의 컬럼이 있고 컬럼명은: {}'.format(dataset.shape[1], dataset.columns.values))
```
![](https://images.velog.io/images/chaong309/post/5dc66fc3-07ba-485c-8480-d50c76e5f568/image.png)

### Feature와 Target간 관계 파악
``` py
## 한글 깨짐 방지
## 처음 실행한다면 아래의 다운로드 코드에서 주석 해제후 실행
## 다 설치하고 나서도 안된다면 런타임 재시작 하고 다운로드 부분은 주석처리 하고 나머지 실행하면 해결
#!apt-get update -qq
#!apt-get install fonts-nanum* -qq

import matplotlib.font_manager as fm

path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf'
font_name = fm.FontProperties(fname=path, size=10).get_name()

plt.rc('font', family=font_name)

fm._rebuild()
```
### Q1~5와 Feature 간의 관계 파악을 위한 pairs plot
``` python
print('feature의 총 컬럼수는 = {}'.format(dataset.shape[1]-8-3))
```
![](https://images.velog.io/images/chaong309/post/036137e6-8ecb-4e90-abee-120a1f97aed9/image.png)

- 총 feature의 수는 75개 이므로 한 번에 타겟과 pairs plot을 그리면 figsize 한계 때문에 변수와 타겟간 관계 파악이 어렵다
- 따라서, 가시성을 위해 feature 10개씩 pairs plot을 그려줌. 맨 마지막은 5개만!

```py
for j in range(5):
    print('Q{} ~ Feature'.format(j+1))
    for i in range(1, 9):
        if i == 8:
            sns.pairplot(dataset.iloc[:, list(range(3 + 10*(i-1), 3 + 10*(i-1) + 5)) + [-(5-j)]])
        else:
            sns.pairplot(dataset.iloc[:, list(range(3 + 10*(i-1), 3 + 10*i)) + [-(5-j)]])
        plt.show()
        print('{}reps'.format(i))
        print('')
    plt.savefig('savefig_default.png')
```

![](https://images.velog.io/images/chaong309/post/dd843f80-9d79-4483-91e8-fb365c4c08a9/savefig_default0-1.png)
![](https://images.velog.io/images/chaong309/post/874b9b79-2085-4b58-9ddf-a818e04eabc9/savefig_default1-1.png)
![](https://images.velog.io/images/chaong309/post/d13bbe3f-166f-48c8-a456-e4fbe66e0a3a/savefig_default2-1.png)
![](https://images.velog.io/images/chaong309/post/b9cf3f36-0b9c-453b-81c8-de3ec554656f/savefig_default3-1.png)![](https://images.velog.io/images/chaong309/post/ba2a7369-31df-4129-8f36-8b8b234450c2/savefig_default4-1.png)

- plot이 8번씩 5개 총 40번이 생성되므로 각 문항별로 첫 10개 컬럼에 대해서만 업로드.
- 타겟의 분포는 대체적으로 좌우 대칭적인 모습으로 보이지만 feature의 분포는 skewed 형태를 띈다. 

### 분석 진행
- #### 우선 모든 변수들을 다 사용
- #### train set 구성
    - 1. 현재의 수치로 다음을 예측해야 함
    - 2. 현재 반기 기준 => 다음 반기를 예측하도록
    - 3. 현재 반기 feature와 다음 반기 target으로 dataset 구성 ex) 2017상반기(feature)로 2017하반기(target) 예측하게 dataset 구성
```py
# feature와 target 분리
# date / police 순으로 정렬해서 반기내 경찰서별 순서가 같도록 정렬
tmp = dataset.sort_values(['date', 'police']).reset_index(drop = True).copy()
feature = tmp.iloc[:, :-8]
target = tmp.iloc[:, [0, 1, 2, -8, -7, -6, -5, -4 ,-3, -2, -1]] # dataset 구성하기 편하게 반기도 같이 넣어줌

## 2019하반기는 제외 -> 추후 2020년 상반기 예측할 때의 feature로 사용
train_feature = feature.loc[feature.date != '2019하반기']
train_target = target.loc[target.date != '2017상반기'].reset_index(drop = True)

### 모델 생성 전 feature들 scaling
train_feature.drop(['date', 'police', 'PNAME'], axis = 1, inplace = True)
train_target.drop(['date', 'police', 'PNAME'], axis = 1, inplace = True)

### 정규화 스케일러 사용
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_feature)
X = scaler.transform(train_feature)

## 추후 모델 성능 비교를 위해 test set 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, train_target, test_size = 0.2, random_state = 2020)
X_train = pd.DataFrame(X_train, columns = train_feature.columns)
X_test = pd.DataFrame(X_test, columns = train_feature.columns)
## 후에 K-fold cv를 위해 인덱스 재정렬
y_train.reset_index(drop = True, inplace = True)
```

### 5-fold CV를 통해 문항별 최적 모델 선정
```py
from sklearn.model_selection import KFold
kf = KFold() # default가 5

## q1~q5 별 최적의 모델을 찾을 것. 이를 이용해서 최종 종합 안전도 점수를 계산
## 모델 생성
## 평가지표는 mae
from sklearn.metrics import mean_absolute_error as mae
lm = LinearRegression()
svm = SVR()
# 학습하는 과정에서 random한 부분이 존재하므로 seed를 주어 고정시켜서 결과 확인
dt = DecisionTreeRegressor(random_state=2020)
rf = RandomForestRegressor(random_state=2020)
ada = AdaBoostRegressor(random_state=2020)
xgb = XGBRegressor(random_state=2020)
cat = CatBoostRegressor(random_state=2020)
question_score = {}
best_model = {}

for i in train_target.columns[3:]:
    ## dataset 분할
    k = 1
    tmp = pd.DataFrame(index = ['lm', 'svm', 'dt', 'rf', 'ada', 'xgb', 'cat'])
    for train, test in kf.split(X_train):
        ## 모델 학습 - 문항별 / fold별. y_train[i]는 Series를 리턴하므로 iloc 대신 인덱싱 사용
        lm.fit(X_train.iloc[train], y_train[i][train])
        svm.fit(X_train.iloc[train], y_train[i][train])
        dt.fit(X_train.iloc[train], y_train[i][train])
        rf.fit(X_train.iloc[train], y_train[i][train])
        ada.fit(X_train.iloc[train], y_train[i][train])
        xgb.fit(X_train.iloc[train], y_train[i][train])
        cat.fit(X_train.iloc[train], y_train[i][train], silent = True) # silent True를 주어 학습되는 과정에서 출력되는 결과물 off
        ## k별 test set에 대한 mae 값을 담아줌
        val_list = [mae(y_train[i][test], lm.predict(X_train.iloc[test])), mae(y_train[i][test], svm.predict(X_train.iloc[test])),
                    mae(y_train[i][test], dt.predict(X_train.iloc[test])), mae(y_train[i][test], rf.predict(X_train.iloc[test])),
                    mae(y_train[i][test], ada.predict(X_train.iloc[test])), mae(y_train[i][test], xgb.predict(X_train.iloc[test])),
                    mae(y_train[i][test], cat.predict(X_train.iloc[test]))]
        tmp['{}'.format(k)] = val_list
        k += 1

    ## 문항별로 score df 넣어줌
    question_score[i] = tmp
    ## 문항별 k-fold cv의 결과 최적의 모델 명을 넣어줌. k번의 결과의 평균을 내 가장 적은 값을 가지는 모델이 최적 모델
    best_model[i] = tmp.apply(np.mean, axis = 1).sort_values().head(1).index[0]
```
![](https://images.velog.io/images/chaong309/post/a037b159-a785-49b7-83c5-90a19f68952e/image.png)
- 5-fold CV를 통해 다양한 regressor들을 비교해본 결과 문항별 best 모델은 위와 같다.
- 각 문항별 최적 모델로 전체 데이터에 대해 학습 후 test set을 이용해 성능을 평가
### ML과 DL의 비교
- 기존 훈련과 검증을 생각하면 전체 Datasets에서 train / test를 분리하고 이 train set에 대해서 정말 train만 시킬 set과 하이퍼 파라미터 튜닝용 valid set을 분리했음
- 따라서 datasets = train(train/valid) + test의 형태로 datasetes split
- GridSearch CV를 이용하여 사용자가 입력한 하이퍼 파라미터로 만들어지는 모든 조합에 대해 학습하고, valid set을 이용해 검증해보며 최적의 파라미터 조합과 이를 이용해 훈련까지 마친 모델을 return 받아 실제 predict에 사용한다.
- 실행시키면 훈련과 파라미터 튜닝까지 해주는 method라 상당히 유용하다. Catboost와 같은 모델들은 자체 매서드로 grid_search가 있지만 best params, best scores만 return 하는 것 같아서 그냥 scikitlearn의 GridSearchCV 이용!
```py
from sklearn.model_selection import GridSearchCV
import time
# Create the parameter grid for each model
param_rf = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

param_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

param_cat = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}        

start = time.time()
# Instantiate the grid search model
## Q1
grid_search1 = GridSearchCV(estimator = rf, param_grid = param_rf, cv = 3, n_jobs = -1, verbose = 2)
grid_search1.fit(X_train, y_train['Q1_절도폭력'])

## Q2
grid_search2 = GridSearchCV(estimator = rf, param_grid = param_rf, cv = 3, n_jobs = -1, verbose = 2)
grid_search2.fit(X_train, y_train['Q2_강도살인'])

## Q3
grid_search3 = GridSearchCV(estimator = rf, param_grid = param_rf, cv = 3, n_jobs = -1, verbose = 2)
grid_search3.fit(X_train, y_train['Q3_교통사고'])

## Q4 -> cat boost의 경우엔 클래스 자체에 메서드로 grid search가 존재. 리턴이 파라미터 조합과 스코어만 주는 것 같아서 sklearn의 그리드서치 사용
## sklearn의 그리드서치는 파라미터 조합, 해당 파라미터로 학습된 모델까지 return 해주므로 후에 다시 최적 조합으로 학습시켜볼 필요가 없이 모델 생성 완료
grid_search4 = GridSearchCV(estimator = cat, param_grid = param_cat, cv = 3, n_jobs = -1, verbose = 2)
grid_search4.fit(X_train, y_train['Q4_법질서준수'], silent = True)

## Q5
grid_search5 = GridSearchCV(estimator = xgb, param_grid = param_xgb, cv = 3, n_jobs = -1, verbose = 2)
grid_search5.fit(X_train, y_train['Q5_전반적'])

end = time.time()
print('셀 실행에 소요된 시간은 {}분 {}초 입니다.'.format((end - start)//60, round((end-start) % 60)))
```
![](https://images.velog.io/images/chaong309/post/b733de41-7f4d-420f-920b-0d2ce5f8b461/image.png)

- 학습하는데 상당한 시간이 소요된다. 왜냐면, 하이퍼 파라미터의 모든 조합으로 모델을 생성하여 학습 후에 test 하는 것이므로 모든 조합의 수만큼 학습을 진행한다. 
- 학습 데이터 양이 많을수록, 가능한 경우의 수가 더 많은수록 소요시간이 증가하므로 트레이드 오프를 생각하고 사용해야 함.
- 코랩에서 일정시간 움직임이 없거나 할당받은 ram을 다 쓰거나 하는 경우 런타임 연결이 종료된다. 코랩은 가상환경에서 작업하는 것이라 연결이 종료되는 순간 작업했던 내용들은 전부 사라지게 된다. 
- ram을 다쓰는 것은 어쩔 수 없는 부분이지만(정기적인 결제를 한다면 더 빠르고, 더 많은 양의 작업공간 제공), 실행시켜놓고 다른 일을 하는 것이 더 생산적이기 때문에 자바 스크립트를 이용해서 일정 시간마다 페이지를 클릭하는 매크로를 생성해준다.

- ctrl+shift+i 혹은 F12키를 눌러 개발자 도구를 열고 콘솔을 클릭하고 맨 아래에 다음과 같은 코드를 입력하고 Enter를 눌러 실행!
```js
function ClickConnect(){
    console.log("코랩 연결 끊김 방지"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60 * 1000)
```
![](https://images.velog.io/images/chaong309/post/4eae1699-9b89-43b9-a56d-f64c97d4ed2d/image.png)

- 60초 마다 클릭 매크로가 실행되며 로그에 '코랩 연결 끊김 방지'라는 메시지를 출력한다. 시간은 본인 지정하고 싶은 대로 지정하길.

### GS CV를 종료하고 생성된 각 문항별 최적 모델을 가지고 test set에 대해 pred
- grid_search의 return중 best_estimator_에 최적 파라미터로 학습된 모델이 생성되어 있다.
- 이를 이용해서 test set에 대해 prediction을 진행
- 사용법은 grid_search.best_estimator_.predict(원하는 데이터)
```py
### ML model training and testing 
### 문항별 모델에 대해 Grid Search CV를 통해 전체 트레인 셋에 대해 학습시킴과 동시에 최적 파라미터까지
### Q1
q1_pred = grid_search1.best_estimator_.predict(X_test) 

### Q2
q2_pred = grid_search2.best_estimator_.predict(X_test)

### Q3
q3_pred = grid_search3.best_estimator_.predict(X_test)

### Q4
q4_pred = grid_search4.best_estimator_.predict(X_test)

### Q5
q5_pred = grid_search5.best_estimator_.predict(X_test)

## 문항별 mae
ml_mae = [mae(y_test['Q1_절도폭력'], q1_pred).round(2), mae(y_test['Q2_강도살인'], q2_pred).round(2), mae(y_test['Q3_교통사고'], q3_pred).round(2),
            mae(y_test['Q4_법질서준수'], q4_pred).round(2), mae(y_test['Q5_전반적'], q5_pred).round(2)]

## 최종 데이터셋
tmp = pd.DataFrame()
tmp['q1'] = q1_pred
tmp['q2'] = q2_pred
tmp['q3'] = q3_pred
tmp['q4'] = q4_pred
tmp['q5'] = q5_pred

### 최종적으로 제출해야 할것은 종합체감안전도.
### 종합체감안전도는 분야별, 범죄 두 항목의 가중 합으로 계산되어지고
### 분야별, 범죄는 q1~q5의 가중 합으로 계산되어진다.
tmp.insert(loc = 0, column = 'crime_safety', value = (tmp.q1+tmp.q2)/2)
tmp.insert(loc = 0, column = 'dept_safety', value = (tmp.crime_safety *0.343) + (tmp.q3*0.305) + (tmp.q4*0.352))
tmp.insert(loc = 0, column = 'tot_f_safety', value = (tmp.q5*0.3) + (tmp.dept_safety*0.7))

tmp.head()
```
![](https://images.velog.io/images/chaong309/post/6b44d6f2-163c-44ec-ac79-1af1e3474a5a/image.png)

### 문항별 최적 모델의 예측 값을 이용해 종합 안전도를 구한 후 MAE 계산
![](https://images.velog.io/images/chaong309/post/7f185897-a662-4a93-b60c-f6c29850f3a9/image.png)

- 모델의 예측치와 실제의 차이가 평균 1.68정도 임을 알 수 있다. 
- Absolute Error를 이용한 것이므로 예측이 over인지 under인지는 알 수 없음.

### DNN을 이용해 예측을 해보고 ML과 비교
#### TF의 from_tensor_slices를 이용해 Datasets을 iterable 하게 준비
``` py
## dataset for train
### X는 164 * 75의 shape이므로 164 * 1 * 75로 shape 변환. y는 164, 이므로 164 * 1 * 1 로 변환
q1_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train, axis = 1), 
                                               np.expand_dims(y_train['Q1_절도폭력'].values.reshape(-1,1), axis = -1)))
q2_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train, axis = 1), 
                                               np.expand_dims(y_train['Q2_강도살인'].values.reshape(-1,1), axis = -1)))
q3_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train, axis = 1), 
                                               np.expand_dims(y_train['Q3_교통사고'].values.reshape(-1,1), axis = -1)))
q4_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train, axis = 1), 
                                               np.expand_dims(y_train['Q4_법질서준수'].values.reshape(-1,1), axis = -1)))
q5_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train, axis = 1), 
                                               np.expand_dims(y_train['Q5_전반적'].values.reshape(-1,1), axis = -1)))

## dataset for test
## test도 위와 같은 shape으로 변경
q1_test = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_test, axis = 1), 
                                               np.expand_dims(y_test['Q1_절도폭력'].values.reshape(-1,1), axis = -1)))
q2_test = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_test, axis = 1), 
                                               np.expand_dims(y_test['Q2_강도살인'].values.reshape(-1,1), axis = -1)))
q3_test = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_test, axis = 1), 
                                               np.expand_dims(y_test['Q3_교통사고'].values.reshape(-1,1), axis = -1)))
q4_test = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_test, axis = 1), 
                                               np.expand_dims(y_test['Q4_법질서준수'].values.reshape(-1,1), axis = -1)))
q5_test = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_test, axis = 1), 
                                               np.expand_dims(y_test['Q5_전반적'].values.reshape(-1,1), axis = -1)))
```

### DNN 모델을 세개 생성하고 최적의 모델로 최종 예측을 진행
- dnn1: 입력 - 은닉(4개) - 출력 구조로 layer를 쌓았고 활성함수는 relu
- dnn2: 입력 - 은닉(3개) - 출력 구조로 layer를 쌓았고 활성함수는 tanh
- dnn3: 입력 - 은닉(3개) - 출력 구조로 layer를 쌓았고 활성함수는 linear

```py
### DL model training and testing
### Q1~Q5별로 dnn1~3 중 최적인 모델로 test set에 대해 mae를 계산
for i in y_train.columns[3:]:
    ## dnn1 => 은닉층 4개. 각각 kernelsize = 75, 50, 25, 10. activation function은 relu
    dnn1 = tf.keras.Sequential([
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(75, activation = 'relu'),
                            tf.keras.layers.Dense(50, activation = 'relu'),
                            tf.keras.layers.Dense(25, activation = 'relu'),
                            tf.keras.layers.Dense(10, activation = 'relu'),
                            tf.keras.layers.Dense(1)
    ])
    dnn1.compile(optimizer = 'adam', loss = 'mse', metrics = 'mae')

    ## dnn2 => 은닉층 3개. 각각 kernelsize = 128, 64, 32. activation function은 tanh
    dnn2 = tf.keras.Sequential([
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128, activation = 'tanh'),
                                tf.keras.layers.Dense(64, activation = 'tanh'),
                                tf.keras.layers.Dense(32, activation = 'tanh'),
                                tf.keras.layers.Dense(1)
    ])
    dnn2.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

    ## dnn3 => 
    dnn3 = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(100, activation='linear'),
                                        tf.keras.layers.Dense(50, activation='linear'),
                                        tf.keras.layers.Dense(25, activation='linear'),
                                        tf.keras.layers.Dense(1)
                                        ])
    dnn3.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

    if i == 'Q1_절도폭력':
        dnn1.fit(q1_train, epochs = 50, verbose = 0)
        dnn2.fit(q1_train, epochs = 50, verbose = 0)
        dnn3.fit(q1_train, epochs = 50, verbose = 0)
        compare = {'dnn1': min(dnn1.history.history['mae']), 'dnn2': min(dnn2.history.history['mae']), 'dnn3': min(dnn3.history.history['mae'])}
        best = sorted(compare, key = lambda x: compare[x])[0]
        print('q1의 최적 모델: {}, train mae: {}'.format(best, compare[best]))
        ## 최적 모델로 q1 test set에 대해 mae 값 get
        if best == 'dnn1':
            q1_mae = dnn1.evaluate(q1_test, verbose = 0)[1]
            q1_pred = dnn1(X_test.values)
        elif best == 'dnn2':
            q1_mae = dnn2.evaluate(q1_test, verbose = 0)[1]
            q1_pred = dnn2(X_test.values)
        else:
            q1_mae = dnn3.evaluate(q1_test, verbose = 0)[1]
            q1_pred = dnn3(X_test.values)

    elif i == 'Q2_강도살인':
        dnn1.fit(q2_train, epochs = 50, verbose = 0)
        dnn2.fit(q2_train, epochs = 50, verbose = 0)
        dnn3.fit(q2_train, epochs = 50, verbose = 0)
        compare = {'dnn1': min(dnn1.history.history['mae']), 'dnn2': min(dnn2.history.history['mae']), 'dnn3': min(dnn3.history.history['mae'])}
        best = sorted(compare, key = lambda x: compare[x])[0]
        print('q2의 최적 모델: {}, train mae: {}'.format(best, compare[best]))
        ## 최적 모델로 q2 test set에 대해 mae 값 get
        if best == 'dnn1':
            q2_mae = dnn1.evaluate(q2_test, verbose = 0)[1]
            q2_pred = dnn1(X_test.values)
        elif best == 'dnn2':
            q2_mae = dnn2.evaluate(q2_test, verbose = 0)[1]
            q2_pred = dnn2(X_test.values)
        else:
            q2_mae = dnn3.evaluate(q2_test, verbose = 0)[1]
            q2_pred = dnn3(X_test.values)

    elif i == 'Q3_교통사고':
        dnn1.fit(q3_train, epochs = 50, verbose = 0)
        dnn2.fit(q3_train, epochs = 50, verbose = 0)
        dnn3.fit(q3_train, epochs = 50, verbose = 0)
        compare = {'dnn1': min(dnn1.history.history['mae']), 'dnn2': min(dnn2.history.history['mae']), 'dnn3': min(dnn3.history.history['mae'])}
        best = sorted(compare, key = lambda x: compare[x])[0]
        print('q3의 최적 모델: {}, train mae: {}'.format(best, compare[best]))
        ## 최적 모델로 q3 test set에 대해 mae 값 get
        if best == 'dnn1':
            q3_mae = dnn1.evaluate(q3_test, verbose = 0)[1]
            q3_pred = dnn1(X_test.values)
        elif best == 'dnn2':
            q3_mae = dnn2.evaluate(q3_test, verbose = 0)[1]
            q3_pred = dnn2(X_test.values)
        else:
            q3_mae = dnn3.evaluate(q3_test, verbose = 0)[1]
            q3_pred = dnn3(X_test.values)
    
    elif i == 'Q4_법질서준수':
        dnn1.fit(q4_train, epochs = 50, verbose = 0)
        dnn2.fit(q4_train, epochs = 50, verbose = 0)
        dnn3.fit(q4_train, epochs = 50, verbose = 0)
        compare = {'dnn1': min(dnn1.history.history['mae']), 'dnn2': min(dnn2.history.history['mae']), 'dnn3': min(dnn3.history.history['mae'])}
        best = sorted(compare, key = lambda x: compare[x])[0]
        print('q4의 최적 모델: {}, train mae: {}'.format(best, compare[best]))
        ## 최적 모델로 q4 test set에 대해 mae 값 get
        if best == 'dnn1':
            q4_mae = dnn1.evaluate(q4_test, verbose = 0)[1]
            q4_pred = dnn1(X_test.values)
        elif best == 'dnn2':
            q4_mae = dnn2.evaluate(q4_test, verbose = 0)[1]
            q4_pred = dnn2(X_test.values)
        else:
            q4_mae = dnn3.evaluate(q4_test, verbose = 0)[1]
            q4_pred = dnn3(X_test.values)

    else:
        dnn1.fit(q5_train, epochs = 50, verbose = 0)
        dnn2.fit(q5_train, epochs = 50, verbose = 0)
        dnn3.fit(q5_train, epochs = 50, verbose = 0)
        compare = {'dnn1': min(dnn1.history.history['mae']), 'dnn2': min(dnn2.history.history['mae']), 'dnn3': min(dnn3.history.history['mae'])}
        best = sorted(compare, key = lambda x: compare[x])[0]
        print('q5의 최적 모델: {}, train mae: {}'.format(best, compare[best]))
         ## 최적 모델로 q5 test set에 대해 mae 값 get
        if best == 'dnn1':
            q5_mae = dnn1.evaluate(q5_test, verbose = 0)[1]
            q5_pred = dnn1(X_test.values)
        elif best == 'dnn2':
            q5_mae = dnn2.evaluate(q5_test, verbose = 0)[1]
            q5_pred = dnn2(X_test.values)
        else:
            q5_mae = dnn3.evaluate(q5_test, verbose = 0)[1]
            q5_pred = dnn3(X_test.values)
dl_mae = [q1_mae, q2_mae, q3_mae, q4_mae, q5_mae]
```
![](https://images.velog.io/images/chaong309/post/0f158ea3-7e30-4894-920b-e64a01ded1f7/image.png)

### 문항별 DL과 ML의 최적 모델 비교
``` py
for i in range(5):
    if ml_mae[i] < dl_mae[i]:
        print('Q{}의 최적 모델은 ML모델 중 {}입니다.\nDL모델과의 차이는 {} 입니다.\n'.format(i+1, best_model[y_train.columns[i+3]], abs(ml_mae[i] - dl_mae[i])))
    else:
        print('Q{}의 최적 모델은 DL모델 중 {}입니다.'.format(i+1, 'dnn2'))
```
![](https://images.velog.io/images/chaong309/post/95148742-09f4-422f-8a85-00cc35a7183e/image.png)

- DNN 모델이 train set에 대한 mae에 비해 test set에 대한 mae가 상당히 큰 수준이다
- 다층을 이룬 모델에 뉴런 개수도 많아 과적합이 발생하기 좋은 상황이엇지만 Dropout이나 batch normalization을 하지 않았기 때문에 과적합이 발생하지 않았을까 싶다.

```py
dnn_tmp = pd.DataFrame({'q1': q1_pred.numpy().reshape(-1,), 'q2': q2_pred.numpy().reshape(-1,), 'q3': q3_pred.numpy().reshape(-1,), 
                        'q4': q4_pred.numpy().reshape(-1,), 'q5': q5_pred.numpy().reshape(-1,)})
dnn_tmp.insert(loc = 0, column = 'crime_safety', value = (dnn_tmp.q1+dnn_tmp.q2)/2)
dnn_tmp.insert(loc = 0, column = 'dept_safety', value = (dnn_tmp.crime_safety *0.343) + (dnn_tmp.q3*0.305) + (dnn_tmp.q4*0.352))
dnn_tmp.insert(loc = 0, column = 'tot_f_safety', value = (dnn_tmp.q5*0.3) + (dnn_tmp.dept_safety*0.7))

dnn_tmp.head()
```
![](https://images.velog.io/images/chaong309/post/f4a1625a-38f9-4b75-b934-54bd03b380a2/image.png)

- 학습과정 중 성능이 가장 좋았던 모델을 이용해 test set에 대해 최종 데이터프레임을 만들었다. 


```py
print('DL의 종합체감안전도 MAE:', mae(y_test['종합체감'], dnn_tmp['tot_f_safety']))
```
![](https://images.velog.io/images/chaong309/post/69ce875e-5945-40f4-9eaa-ba78fd841574/image.png)

- MAE가 5로 ML에 비해 Simple DNN의 성능이 많이 부족해 보인다
- 이는 과적합을 고려하지 않은 학습을 했기 때문에 그렇지 않을까라 생각했고 데이터 자체도 너무 부족했기 때문에 복잡한 신경망 모델보단 조금 단순한 머신러닝 모델들을 사용하는게 효과적일수 있다 생각한다.
- 아직 드롭아웃이나 배치정규화 같은 과적합 방지를 위한 처리를 전혀 하지 않았기 때문에 모델을 새로 생성해서 성능을 비교해봐야 한다.
- 추가적으로 타겟 자체가 시간의 흐름에 따라 관측된 자료이기 때문에 RNN계열의 모델을 사용해볼 필요가 있다.
- 다양한 모델들에 대해 비교해보고 파라미터 튜닝까지 해볼 수 있는 좋은 기회였고 무엇보다 딥러닝 모델을 사용할 수 있게 텐서의 shape을 구성하고 api를 쓰며 데이터 셋을 만들 기회가 되어서 좋았다.
- 아직 발전해야 할 부분이 많음!
