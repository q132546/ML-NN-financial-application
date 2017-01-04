import numpy as np
import pandas as pd
import pandas_talib as pt
import pywt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
url = 'Deu.csv'
name = 'Deutsche Bank'


def data_read(path):
    data = pd.read_csv(path)
    data = data[::-1]
    return data


def pywave(index, wavefunction, lv, m, n):
    """

    :param index: 待處理的時間序列
    :param dataframe: 消噪後的序列放置處
    :param wavefunction: 選用的小波分析函數
    :param lv: 分解層數
    :param m,n: 決定閾值處理時的係數層數
    :return:
    """
    coeff = pywt.wavedec(index, wavefunction, mode='sym', level=lv)
    """
    :param index: 待處理的時間序列
    :param wavefunction: 選用的小波分析函數
    :param mode:symmetric
    :param lv: 分解層數

    :return cA cD: cA=尺度係數 cD=小波係數
    """

    def check(x):
        if x > 0:
            return 1
        else:
            return -1

    # 選取小波係數(m,n)層
    for i in range(m, n + 1):
        cD = coeff[i]

        # 閾值決定準則 sqtwolog ＝ sqrt(2*log(n))
        tr = np.sqrt(2 * np.log(len(cD)))

        for j in range(len(cD)):
            if cD[j] >= tr:
                coeff[i][j] = check(cD[j]) - tr
            else:
                coeff[i][j] = 0

    # 重構
    denoised_index = pywt.waverec(coeff, wavefunction)

    return denoised_index


def label_make(data):
    p = data['Adj Close']
    a, b, c, d, e = 0, 0, 0, 0, 0
    for i in range(len(p) - 20):
        value = p[i + 20] - p[i]
        if value / p[i] <= -0.10:
            p[i] = 1
            a += 1

        elif -0.02 >= value / p[i] > -0.10:
            p[i] = 2
            b += 1

        elif 0.02 >= value / p[i] > -0.02:
            p[i] = 3
            c += 1

        elif 0.10 >= value / p[i] > 0.02:
            p[i] = 4
            d += 1

        elif value / p[i] > 0.10:
            p[i] = 5
            e += 1
    p = p[:-20]
    return p


def label_make_regression(prediciton):
    p = prediciton
    for i, value in enumerate(p):
        if value < 1:
            p[i] = 1

        elif 2 >= value > 1:
            p[i] = 2

        elif 3 >= value > 2:
            p[i] = 3

        elif 4 >= value > 3:
            p[i] = 4

        elif value > 3.5:
            p[i] = 5

    return p


def data_make(url, name):
    data = data_read(url)
    c_p = data['Close']
    close_p = data['Adj Close']
    open_p = data['Open']
    high_p = data['High']
    low_p = data['Low']
    vol = data['Volume']
    date = data['Date']

    c = pywave(close_p, 'db4', 4, 2, 4)[1:]
    cp = pywave(c_p, 'db4', 4, 2, 4)[1:]
    o = pywave(open_p, 'db4', 4, 2, 4)[1:]
    h = pywave(high_p, 'db4', 4, 2, 4)[1:]
    l = pywave(low_p, 'db4', 4, 2, 4)[1:]

    hstack = np.vstack((date, c, cp, o, h, l, vol))
    new_data = pd.DataFrame(hstack.T, columns=['Date', name, 'Close', 'Open', 'High', 'Low', 'Volume'])
    return new_data


data = data_make(url, 'Adj Close')
n = 20

"""
factor computing

"""
"""
p is label(target)

"""

date = data['Date']

# ---------------

vix_data = data_make('Vix1.csv','VIX')
vix = vix_data['VIX']

crude_data = data_make('crude.csv','CRUDE')
crude = crude_data['CRUDE']

ma = pt.MA(data, n, price='Adj Close')
ma = ma['MA_20']

ema = pt.EMA(data, n, price='Adj Close')
ema = ema['EMA_20']

mom = pt.MOM(data, n, price='Adj Close')
mom = mom['Momentum_20']

roc = pt.ROC(data, n, price='Adj Close')
roc = roc['ROC_20']

atr = pt.ATR(data, n)
atr = atr['ATR_20']

bands = pt.BBANDS(data, n, price='Adj Close')
bands = bands.loc[:, ['BollingerB_20', 'Bollinger%b_20']]

stok = pt.STOK(data)
stok = stok['SO%k']

sto = pt.STO(data, n)
sto = sto['SO%d_20']

trix = pt.TRIX(data, n)
trix = trix['Trix_20']

adx = pt.ADX(data, n, n)
adx = adx['ADX_20_20']

macd = pt.MACD(data, 10, n, price='Adj Close')
macd = macd.loc[:, ['MACD_10_20', 'MACDsign_10_20', 'MACDdiff_10_20']]

massi = pt.MassI(data)
massi = massi['Mass Index']

# --------------------


def data_split(x_data, y_data, size):
    """

    資料分割成7:3
    :param x_data:
    :param y_data:
    :size train的比例
    :return:
    """
    n = round(size*len(y_data))
    x_train = x_data[:n]
    y_train = y_data[:n]
    x_test  = x_data[n:]
    y_test  = y_data[n:]

    return x_train, x_test, y_train, y_test

p = label_make(data)
d = pd.concat([date, ma, ema, mom, roc, atr, bands, stok, sto, trix, adx, macd, massi, vix, crude, p], axis=1,
              join='inner')
d = d.dropna()


def normalization(training_data, testing_data):
    """

    正規化min-max(normalization)
    :param training_data:  your data
    :return:
    """

    std = StandardScaler()
    x_train_norm = std.fit_transform(training_data)
    x_test_norm = std.transform(testing_data)

    return x_train_norm, x_test_norm

"""
training and testing data building

"""

y_data = d['Adj Close']
x_data = d.drop(['Date', 'Adj Close'], axis = 1)

x_train, x_test, y_train, y_test = data_split(x_data, y_data, size=0.8)
x_train, x_test = normalization(x_train, x_test)
xtrain = x_train
ytrain = np.asarray(y_train, dtype="|S6")




# ----------------------------------------

# machine learning strategy

# Linear Regression Adaboost

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

n_estimators = 80
lr = 0.7

Ada_linear = AdaBoostRegressor(base_estimator=linear, learning_rate=lr, n_estimators=n_estimators, loss='square')
Ada_linear.fit(xtrain, y_train)
linear_prediction = Ada_linear.predict(x_test)
result = label_make_regression(linear_prediction)


# logistic Regression Adaboost

logistic = linear_model.LogisticRegression()
logistic.fit(x_train, ytrain)

Ada_logistic = AdaBoostClassifier(base_estimator=logistic, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME.R')
Ada_logistic.fit(xtrain, ytrain)
logistic_prediction = Ada_logistic.predict(x_test)
logistic_prediction = np.asarray(logistic_prediction, dtype=np.int16)

# decision tree

tree = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=4)
tree.fit(x_train, ytrain)

Ada_tree = AdaBoostClassifier(base_estimator=tree, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME.R')
Ada_tree.fit(xtrain, ytrain)
tree_prediction = Ada_tree.predict(x_test)
tree_prediction = np.asarray(tree_prediction, dtype=np.int16)


# RandomForest Adaboost

RFC = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2)
RFC.fit(xtrain, ytrain)


Ada_rfc=AdaBoostClassifier(base_estimator=RFC, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME.R')
Ada_rfc.fit(xtrain, ytrain)
rfc_prediction = Ada_rfc.predict(x_test)
rfc_prediction = np.asarray(rfc_prediction, dtype=np.int16)


# linear support vector machine Adaboost

l_svm = SVC(kernel='linear', C=1.0, random_state=0)
l_svm.fit(x_train, ytrain)

Ada_lsvm = AdaBoostClassifier(base_estimator=l_svm, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME')
Ada_lsvm.fit(xtrain, ytrain)
lsvm_prediction = Ada_lsvm.predict(x_test)
lsvm_prediction = np.asarray(lsvm_prediction, dtype=np.int16)


# kernal support vector machine Adaboost

k_svm = SVC(kernel='rbf', gamma=1.0, C=10.0)
k_svm.fit(x_train, ytrain)

Ada_ksvm = AdaBoostClassifier(base_estimator=k_svm, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME')
Ada_ksvm.fit(xtrain, ytrain)
ksvm_prediction = Ada_ksvm.predict(x_test)
ksvm_prediction = np.asarray(ksvm_prediction, dtype=np.int16)


# Navie Bayes

nb = GaussianNB()
nb.fit(x_train, ytrain)

Ada_nb = AdaBoostClassifier(base_estimator=nb, learning_rate=lr, n_estimators=n_estimators, algorithm='SAMME')
Ada_nb.fit(xtrain, ytrain)
nb_prediction = Ada_nb.predict(x_test)
nb_prediction = np.asarray(nb_prediction, dtype=np.int16)


# knn

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, ytrain)

Ada_knn = AdaBoostRegressor(base_estimator=knn, learning_rate=lr, n_estimators=n_estimators, loss='square')
Ada_knn.fit(xtrain, y_train)
knn_prediction = Ada_knn.predict(x_test)
knn_prediction = np.asarray(knn_prediction, dtype=np.int16)


# k - means

k_means = KMeans(n_clusters=5)
k_means.fit(x_train)

Ada_kmeans = AdaBoostRegressor(base_estimator= k_means, learning_rate=lr, n_estimators=n_estimators, loss='square')
Ada_kmeans.fit(xtrain, y_train)
kmeans_prediction = Ada_kmeans.predict(x_test)
kmeans_prediction = np.asarray(kmeans_prediction, dtype=np.int16)


# beck testing

def becktesting():

    signal = np.vstack((result, logistic_prediction, tree_prediction, rfc_prediction, lsvm_prediction, ksvm_prediction,
                        nb_prediction, knn_prediction, kmeans_prediction))
    signal -= 3
    data_orignal = data_read(url)
    price = data_orignal['Close']
    date = data_orignal['Date']
    price_train, price_test, date_train, date_test = data_split(price, date, size=0.8)

    stock = np.zeros(0)
    price_test = price_test[(len(price_test) - len(signal[0])):]
    date_test = np.array(date_test[(len(date_test) - len(signal[0])):])
    price_test = np.array([price_test])
    t_signal = 0

    # holding time
    holding_signal = 0
    holding_number = 0
    holding_period = []
    # trading number
    trading_number = 0
    period_trade = []
    trading_p_l = 0

    for i in range(len(price_test[0])):
        score = 0
        for j in range(len(signal)):
            score += signal[j][i]

        if score >= 1 and t_signal == 0:
            stock = np.append(stock, 0)
            t_signal = 1

        elif score >= 1 and t_signal == 1:
            if i == 0:
                stock = np.append(stock, 0)
            else:
                stock = np.append(stock, (price_test[0][i] - price_test[0][i - 1]))

        elif score <= -3 and t_signal == 0:
            stock = np.append(stock, 0)
            t_signal = 0

        elif score <= -3 and t_signal == 1:
            if i == 0:
                stock = np.append(stock, 0)
            else:
                stock = np.append(stock, (price_test[0][i] - price_test[0][i - 1]))
            t_signal = 0

        elif -3 < score < 1:
            if t_signal == 0:
                stock = np.append(stock, 0)

            if t_signal == 1:
                if i == 0:
                    stock = np.append(stock, 0)
                else:
                    stock = np.append(stock, (price_test[0][i] - price_test[0][i - 1]))

        if t_signal == 1:
            if holding_signal == 0:
                holding_number = 1
                holding_signal = 1
                trading_number += 1
                trading_p_l = (price_test[0][i] - price_test[0][i - 1])
            else:
                holding_number += 1
                holding_signal = 1
                trading_p_l = (price_test[0][i] - price_test[0][i - 1])
        elif t_signal == 0:
            if holding_signal == 0:
                holding_signal = 0
            else:
                holding_period.append(holding_number)
                holding_number = 0
                holding_signal = 0
                period_trade.append(trading_p_l)
                trading_p_l = 0

    holding_time = sum(holding_period)
    average_holding = holding_time / trading_number

    return stock, price_test, date_test, holding_time, average_holding, trading_number, period_trade


def accumulate(data):
    p_l = 0
    p_larray = []
    for i, value in enumerate(data):
        p_l += value
        p_larray.append(p_l)

    p_l_array = np.array(p_larray)
    return p_l_array


def real_accumulate(data):
    pl = 0
    p_larray = [0]
    for i in range(len(data) - 1):
        r_pl = (data[i+1] - data[i])
        pl += r_pl
        p_larray.append(pl)

    p_l_array = np.array(p_larray)
    return p_l_array


def p_l_judge(data):
    p_score = 0
    n_score = 0
    p_n = 0
    l_n = 0
    for i in stock:
        if i > 0:
            p_score += i
            p_n += 1
        elif i < 0:
            n_score += i
            l_n += 1

    return p_score, n_score, p_n, l_n


def win_Ratio(period_trade):
    p_score = 0
    n_score = 0
    for i in period_trade:
        if i > 0:
            p_score += 1
        elif i < 0:
            n_score += 1
    win_ratio = p_score / (p_score + n_score)
    return win_ratio


stock, price_test, date_test, holding_time, average_holding, trading_nuber, period_trade = becktesting()
p_l = accumulate(stock)
price = real_accumulate(price_test[0])
p_score, n_score, p_n, l_n = p_l_judge(stock)
win_ratio = win_Ratio(period_trade)

print('trade number: ', trading_nuber)
print('win ratio: ', win_ratio)
print('average profit: ', p_score * 1000 / trading_nuber)
print('average loss: ', n_score * 1000 / trading_nuber)
print('holding period', average_holding)


# """
print('標的股票 : ', name)
print('滿倉為一張股票1000股，不得放空')
print('最終損益 : ',  p_l[-1]*1000, '元')
print('最大回測 : ', np.min(p_l)*1000, '元')
print('最大收益 : ', np.max(p_l)*1000, '元')

# plot


df = pd.DataFrame({'date': date_test, 'price': price, 'stock': p_l})
df.plot(x='date', title= name + ' accumulate profit  ')
plt.legend(['real return', 'strategy return'])
plt.xlabel('date')
plt.ylabel('profit')

plt.show()

# """







