
# Machine Learning Engineer Nanodegree
## Capstone
## Project: Predict failure of heavy duty trucks out of sensors data (Air pressure system failures in Scania trucks )

Hajime Kawata  
February 26st, 2019

## 0. Abstract


## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

#### Background

Reducing the failure rate of trucks in logictics is an important issue for improving profitability and customer satisfaction in logistics business.
Until now, logistics companies rely on regular maintenance to prevent the occurrence of failures.
However, since the utilization rate of trucks exceeds the days what has been assumed to be covered by periodic maintenance, the percentage of post-maintenance caused by failures increasing, which has been a problem for some time.

For this reason, it is strongly required that inspection and maintenance work be performed before regular failure, by appropriately detecting signs of failure during regular maintenance
In order to properly capture signs of failure, it has been considered to use data collected from sensors installed in trucks, bu sensor data is gathered in milliseconds at longest, and collection targets are also diverse.
As a result, several hundred megabytes of data may be generated from one track on a day.
A large data cost has been anticipated.

In this project in order to quickly and flexibly arrange trucks based on failure prediction,
- Reduce the amount of features derived from the target sensor to be collected and suppress the arithmetic cost to be generated
- Perform a failure based on the logic derived from the data pattern leading to the failure on the truck side.

This would make it possible to appropriately send alerts before the actual failure occurrence to the monitoring center and to adjust the arrangement of the trucks.

この領域は必要性が長く叫ばれてはいたものの、
メーカーが情報を出したがらないこと、センサーの選別と、適切に計測するための取り付け位置の選択が難しいこと、
収集した大量のデータの収集手段など、ハードルが多く、
実効性の調査は行われてこなかった。
今回([Kaggle](https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set/home))
にて十分なデータが提供されたことから、そのデータを利用して実効性の検証を行った。

### Problem Statement

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_
、
提供データの一部を用いて、機械学習によりモデルの構築を行う。
提供されている試験データに対し、このモデルを適用して性能評価を行う。

モデルの構築にあたっては、モデルが適切に機能するために、
データの偏りやスケール等を考慮して、構築の際に用いるデータを適切に処理する手段を検討たうえで、
いくつかのモデルの動作を決めるパラメータのチューニングを行い、


Here, using the training data([Kaggle](https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set/home)), we derive the prediction model of the occurrence of the failure (label) from the sensor data (features). This prediction logic is applied to the test data and the ability of the prediction logic is evaluated.
The sensor data is anonymized for confidentiality reasons, and so we have to make the prediction model purely on mathematical and statistical approach.



### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


As benchmark, adopts 60% correct answer rate.
This is based on a hearing from an interview myself conducted that the failure rate prevented by regular maintenance is about 60% by experience.
It is worth considering the devided model in this project, if failure prediction is more than 60%.

In evaluating the prediction logic, it is a challenge to extract failure events as much as possible while maintaining the operation rate.
For that reason, it is necessary to accurately derive the judgment of correctness.
Here, to make the balance of Confusion Matrix, adopt a model with a high result of scoring by **ROC-AUC**.


### Project Design

1. Split the training data set to establish model and evaluate
2. Apply PCA to reduce the dimension of the model.
3. Device predictor with XGBoost classifier, while seeking the best hyperparameter values with Grid Search by ROC-AUC.
4. Evaluate with the model with given test set, with confusion matrix, and accuracy, precision, and f1 scores.
5. Using the given test set to evaluate the model
6. Evaluate the contribution of feature, to discuss the possibility of reducing the sensors to predict failures.


## II. Analysis
_(approx. 2-4 pages)_
## Data Exploration

In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.

このデータ・セットにおいて訓練データとテストデータは所与のものとして指定されているので、これらを用いる。
センサーデータは8bit単位でセンサーから取得される（参照　I2CやSPIなどの電子回路では１byte(=8bit)データとして取得できる）。これについては、実際の物理的な計測値に変換された下記データを用いるものとする。
    train = pd.read_csv('data/aps_failure_training_set_processed_8bit.csv')
    test = pd.read_csv('data/aps_failure_test_set_processed_8bit.csv')

### Datasets and Inputs

Data is provided in :
*Kaggle* : https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set/home

*The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that is utilized in various functions in a truck, such as braking and gear changes.* (*Kaggle*)

| Category | Description |
|----------|-------------|
| positive class | *consists of component failures for a specific component of the APS system.* |
| negative class | *consists of trucks with failures for components not related to the APS.*  | 


Following is the data volume information.
予めテストデータが提供されており、今回はこのテストデータを用いてモデルの評価をおこなった。

| _ | Total | Positive | Negative | features | file |
|-|-|-|-|-|-|
| Train | 60000 | 1000 | 59000 | 171 | data/aps_failure_training_set_processed_8bit.csv |
| Test  | 16000 |  -    | -     | 171 | data/aps_failure_test_set_processed_8bit.csv |

class列で。。。
positive class 0.992188 	
negative class -0.992188


### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>aa_000</th>
      <th>ab_000</th>
      <th>ac_000</th>
      <th>ad_000</th>
      <th>ae_000</th>
      <th>af_000</th>
      <th>ag_000</th>
      <th>ag_001</th>
      <th>ag_002</th>
      <th>...</th>
      <th>ee_002</th>
      <th>ee_003</th>
      <th>ee_004</th>
      <th>ee_005</th>
      <th>ee_006</th>
      <th>ee_007</th>
      <th>ee_008</th>
      <th>ee_009</th>
      <th>ef_000</th>
      <th>eg_000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.992188</td>
      <td>0.117188</td>
      <td>-0.289062</td>
      <td>0.992188</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.03125</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>0.687500</td>
      <td>0.515625</td>
      <td>0.234375</td>
      <td>0.070312</td>
      <td>0.007812</td>
      <td>-0.109375</td>
      <td>-0.140625</td>
      <td>-0.171875</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.992188</td>
      <td>-0.179688</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.03125</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.023438</td>
      <td>-0.062500</td>
      <td>-0.132812</td>
      <td>-0.132812</td>
      <td>-0.187500</td>
      <td>-0.148438</td>
      <td>-0.085938</td>
      <td>-0.140625</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.992188</td>
      <td>-0.125000</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.03125</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.140625</td>
      <td>-0.093750</td>
      <td>-0.015625</td>
      <td>0.015625</td>
      <td>-0.007812</td>
      <td>-0.109375</td>
      <td>-0.093750</td>
      <td>-0.164062</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.992188</td>
      <td>-0.406250</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.007812</td>
      <td>-0.007812</td>
      <td>-0.03125</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.382812</td>
      <td>-0.382812</td>
      <td>-0.375000</td>
      <td>-0.351562</td>
      <td>-0.312500</td>
      <td>-0.195312</td>
      <td>-0.304688</td>
      <td>-0.171875</td>
      <td>0.890625</td>
      <td>0.992188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.992188</td>
      <td>0.007812</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.03125</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>0.156250</td>
      <td>0.031250</td>
      <td>-0.031250</td>
      <td>-0.039062</td>
      <td>-0.046875</td>
      <td>-0.015625</td>
      <td>0.656250</td>
      <td>-0.148438</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 171 columns</p>
</div>



機械学習を行うにあたり、所与のトレーニングデータの特徴を把握する。これによりデータに対し必要な特徴量の前処理について検討する。


各特徴量をボックス図にプロットした。
![png](figs/sphx_glr_boxplot_demo_pyplot_001.webp)
75パーセンタイルのデータが-1から1までの範囲に広がっているもの、0.01のオーダーに収まっているものなど、各特徴量の広がりが異なっている。有効数字等を考慮した場合、これらについては特徴量間で揃えたほうば良い。 MinとMaxは特徴量ごとに異なっている。またデータの偏りがある各パラメータは同列に評価せずに調整してから評価したほうがよい。

偏りのある特徴量がある。これは、一部の値の近辺にデータの値が集中している場合、こうした偏りのあるデータ (skewed data)を扱う場合、分類精度 (Classification Accuracy)は信用できる検証指標ではないため、F値 (F score)を採用して評価する。

外れ値も結構ある

分散が極端に少ないデータ、多いデータがあり、


```python
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
train.boxplot(ax=ax)
plt.show()
```


![png](figs/output_11_0.png)

統計量

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>aa_000</th>
      <th>ab_000</th>
      <th>ac_000</th>
      <th>ad_000</th>
      <th>ae_000</th>
      <th>af_000</th>
      <th>ag_000</th>
      <th>ag_001</th>
      <th>ag_002</th>
      <th>...</th>
      <th>ee_002</th>
      <th>ee_003</th>
      <th>ee_004</th>
      <th>ee_005</th>
      <th>ee_006</th>
      <th>ee_007</th>
      <th>ee_008</th>
      <th>ee_009</th>
      <th>ef_000</th>
      <th>eg_000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>...</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
      <td>60000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.959115</td>
      <td>-0.124611</td>
      <td>-0.071121</td>
      <td>-0.198529</td>
      <td>-0.007737</td>
      <td>-0.033483</td>
      <td>-0.040633</td>
      <td>-0.006584</td>
      <td>-0.026241</td>
      <td>-0.040699</td>
      <td>...</td>
      <td>-0.104808</td>
      <td>-0.098734</td>
      <td>-0.094976</td>
      <td>-0.089227</td>
      <td>-0.103374</td>
      <td>-0.088961</td>
      <td>-0.084540</td>
      <td>-0.067471</td>
      <td>-0.020035</td>
      <td>-0.018417</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.254040</td>
      <td>0.367680</td>
      <td>0.356812</td>
      <td>0.564872</td>
      <td>0.004138</td>
      <td>0.107086</td>
      <td>0.111752</td>
      <td>0.032016</td>
      <td>0.065200</td>
      <td>0.105864</td>
      <td>...</td>
      <td>0.356547</td>
      <td>0.362066</td>
      <td>0.363148</td>
      <td>0.336121</td>
      <td>0.320314</td>
      <td>0.237613</td>
      <td>0.363893</td>
      <td>0.261009</td>
      <td>0.051907</td>
      <td>0.061751</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.992188</td>
      <td>-0.406250</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.031250</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.382812</td>
      <td>-0.382812</td>
      <td>-0.382812</td>
      <td>-0.351562</td>
      <td>-0.312500</td>
      <td>-0.195312</td>
      <td>-0.304688</td>
      <td>-0.171875</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.992188</td>
      <td>-0.398438</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.031250</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.382812</td>
      <td>-0.382812</td>
      <td>-0.375000</td>
      <td>-0.343750</td>
      <td>-0.312500</td>
      <td>-0.195312</td>
      <td>-0.304688</td>
      <td>-0.171875</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.992188</td>
      <td>-0.195312</td>
      <td>-0.289062</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.031250</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.179688</td>
      <td>-0.179688</td>
      <td>-0.195312</td>
      <td>-0.179688</td>
      <td>-0.226562</td>
      <td>-0.171875</td>
      <td>-0.296875</td>
      <td>-0.171875</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-0.992188</td>
      <td>-0.070312</td>
      <td>-0.000000</td>
      <td>-0.468750</td>
      <td>-0.007812</td>
      <td>-0.046875</td>
      <td>-0.054688</td>
      <td>-0.007812</td>
      <td>-0.031250</td>
      <td>-0.054688</td>
      <td>...</td>
      <td>-0.007812</td>
      <td>0.015625</td>
      <td>0.015625</td>
      <td>0.007812</td>
      <td>-0.054688</td>
      <td>-0.101562</td>
      <td>-0.000000</td>
      <td>-0.132812</td>
      <td>-0.023438</td>
      <td>-0.023438</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>...</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
      <td>0.992188</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 171 columns</p>
</div>




```python
train['class'] = train['class'].apply(lambda x: 0 if x<=0 else 1)
test['class'] = test['class'].apply(lambda x: 0 if x<=0 else 1)

X = train.drop('class', axis=1)
y = train['class']

# use the given test set, instead of creating from the training samples
X_test_given = test.drop('class', axis=1)
y_test_given = test['class']
```

The column data range varies by sensors


```python
y_test_given.value_counts()
```




    0    15625
    1      375
    Name: class, dtype: int64

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_



### Solution Statement

For the criteria listed in **Domain Background***,
take the following solution approach in this project

- Reduce sensor data features from 171 by PCA (Primary Components Analysis)
- Construct the model from reduced features with XGBoost classifier, by adjusting hyperparameters with Grid Search CV 



## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_


```python
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
```

75パーセンタイルのデータは中央に集中しているものの、残りのデータについては幅広く広がっていることがわかる。 PCA適用後のデータであり、クラスタ化されていて

外れ値にFitする過学習

外れ値である可能性がある。外れ値を除外してみる。ここでは下記を参照にOne Class SVMにより外れ値の除去を行った。 https://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection


```python
pca = PCA(0.98)
pca.fit(X_scaled)
pca.n_components_
X_reduced_data = pca.transform(X_scaled)
X_reduced_data.shape
```




    (60000, 77)




```python
df_pca_components = pd.DataFrame(pca.components_, columns=list(X.columns))
df_pca_components.plot(figsize=(100, 50), title='contribution of features')
plt.show()
```


![png](figs/output_20_0.png)



```python
plt.xlabel("Number of components")
plt.ylabel("Cumulated Sum of Ration of variance explained")
plt.xticks(range(0,87))
plt.grid(True)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
```


![png](figs/output_21_0.png)



```python
plt.xlabel("Number of components")
plt.ylabel("Ration of variance explained")
plt.xticks(range(0,87))
plt.grid(True)
plt.plot(pca.explained_variance_ratio_)
plt.show()
```


![png](figs/output_22_0.png)



```python
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
pd.DataFrame(X_reduced_data).boxplot(ax=ax)
plt.show()
```


![png](figs/output_23_0.png)



```python
X_test_given_scaled = scaler.transform(X_test_given)
X_reduced_test_data = pca.transform(X_test_given_scaled)
X_reduced_test_data.shape
```
    (16000, 77)

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


### Benchmark Model

As benchmark, adopts 60% correct answer rate.
This is based on a hearing from an interview myself conducted that the failure rate prevented by regular maintenance is about 60% by experience.
It is worth considering the devided model in this project, if failure prediction is more than 60%.


```python
%%time
predictor = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=5,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=500,
       n_jobs=1, nthread=4, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=True,
       subsample=0.5, tree_method='hist', verbose=10)

predictor.fit(X_reduced_data, y)
```

    [02:47:56] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
    CPU times: user 1min 11s, sys: 208 ms, total: 1min 11s
    Wall time: 19 s


### Evaluation Metrics

In evaluating the prediction logic, it is a challenge to extract failure events as much as possible while maintaining the operation rate.
For that reason, it is necessary to accurately derive the judgment of correctness.
Here, to make the balance of Confusion Matrix, adopt a model with a high result of scoring by **ROC-AUC**.


```python
score = predictor.predict_proba(X_reduced_data)
print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y, score[:, 1], average='macro')))
```

    Best ROC-AUC: 1.0000



```python
score = predictor.predict_proba(X_reduced_data)
print('Best ROC-AUC: {:.4f}'.format(roc_auc_score(y, score[:, 1], average='macro')))predict = predictor.predict(pca.transform(X_test_given_scaled))
print("accuracy score : {}".format(accuracy_score( y_test_given, predict)))
print(classification_report( y_true = y_test_given, y_pred = predict ))
confusion_matrix(y_true = y_test_given, y_pred = predict )
```

    accuracy score : 0.989125
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99     15625
               1       0.90      0.61      0.72       375
    
       micro avg       0.99      0.99      0.99     16000
       macro avg       0.94      0.80      0.86     16000
    weighted avg       0.99      0.99      0.99     16000
    





    array([[15599,    26],
           [  148,   227]])




```python
fig, ax = plt.subplots(1, 1, figsize=(7, 25))
plot_importance(predictor, max_num_features = pca.n_components_, ax=ax)
plt.show()
```


![png](figs/output_30_0.png)



```python
plot_tree(predictor, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.show()
```


![png](figs/output_31_0.png)


### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_

## Seek hyperparameters

https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier



params = {
'max_depth':[4,5,6,7],
'learning_rate':[0.1, 0.15, 0.2],
'gamma':[0.0],
'min_child_weight':[1],
'max_delta_step':[5],
'colsample_bytree':[0.8],
'n_estimators':[300, 500, 700],
'subsample':[0.5],
'objective':['binary:logistic'],
'nthread':[4],
'scale_pos_weight':[1],
'seed':[random_seed],
'verbose': [10]}

model = xgb.XGBClassifier(tree_method='hist')
cv = GridSearchCV(model, params, cv=5, n_jobs=4, scoring='roc_auc')

結果以下のモデルを得られた

実行時間（Lenove X230)
   CPU times: user 57.9 s, sys: 577 ms, total: 58.4 s
    Wall time: 16h 22min 29s


    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=0.8, gamma=0.0, learning_rate=0.1,
           max_delta_step=5, max_depth=7, min_child_weight=1, missing=None,
           n_estimators=300, n_jobs=1, nthread=4, objective='binary:logistic',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=42, silent=True, subsample=0.5, tree_method='hist', verbose=10)
 

print(predictor.score(pca.transform(X_test_given_scaled), y_test_given))
print("Best parameters: %s" % cv.best_params_)
print("Best auroc score: %s" % cv.best_score_)

    0.9888125
    Best parameters: {'colsample_bytree': 0.8, 'gamma': 0.0, 'learning_rate': 0.1, 'max_delta_step': 5, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 300, 'nthread': 4, 'objective': 'binary:logistic', 'scale_pos_weight': 1, 'seed': 42, 'subsample': 0.5, 'verbose': 10}
    Best auroc score: 0.9897447457627119


学習結果

    Best ROC-AUC: 1.0000
    accuracy score : 0.9888125

                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99     15625
               1       0.87      0.61      0.72       375
    
       micro avg       0.99      0.99      0.99     16000
       macro avg       0.93      0.80      0.86     16000
    weighted avg       0.99      0.99      0.99     16000
    

Confusion Matrix

    array([[15592,    33],
           [  146,   229]])



特徴量の重要度をプロット

![png](figs/output_38_0.png)


学習結果で得られた決定木

![png](figs/output_39_0.png)


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

