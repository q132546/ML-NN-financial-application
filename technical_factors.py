import numpy as np
import pandas as pd


#simple average line

def ma(data,N):
    ma = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        MA_mean = sum(data[i:i+N]) / N
        ma.append(MA_mean)

    return ma

def STD(data,N):
    std = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        std_std = np.std(data[i:i+N])
        std.append(std_std)

    return std

#-------------Bollinger Bands----------------

#中轨线 = N日的移动平均线
#上轨线 = 中轨线 + 两倍的标准差
#下轨线 = 中轨线 - 两倍的标准差
#return three matric BOOL,UB,LB

def BOLL(data, m):
    boll = ma(data,m)
    std = STD(data,m)
    ub = []
    lb = []
    len1 = len(data)
    for i in range(len1 - m + 1):
        ub_value = boll[i] + 2*std[i]
        lb_value = boll[i] - 2*std[i]
        ub.append(ub_value)
        lb.append(lb_value)

    return boll,ub,lb

#--------------CCI(Commodity Channel Index)-----------------------

#CCI指標區間的劃分
#1.按市場的通行的標準，CCI指標的運行區間可分為三大類：大於﹢100、小於﹣100和﹢100——﹣100之間。
#2、當CCI＞﹢100時，表明股價已經進入非常態區間——超買區間，股價的異動現象應多加關註。
#3、當CCI＜﹣100時，表明股價已經進入另一個非常態區間——超賣區間，投資者可以逢低吸納股票。
#4、當CCI介於﹢100——﹣100之間時表明股價處於窄幅振蕩整理的區間——常態區間，投資者應以觀望為主。

#
#1當CCI指標從下向上突破﹢100線而進入非常態區間時，表明股價脫離常態而進入異常波動階段
#，中短線應及時買入，如果有比較大的成交量配合，買入信號則更為可靠。

#2當CCI指標從上向下突破﹣100線而進入另一個非常態區間時，表明股價的盤整階段已經結束，
#將進入一個比較長的尋底過程，投資者應以持幣觀望為主。

#3當CCI指標從上向下突破﹢100線而重新進入常態區間時，表明股價的上漲階段可能結束，
#將進入一個比較長時間的盤整階段。投資者應及時逢高賣出股票。

#4當CCI指標從下向上突破﹣100線而重新進入常態區間時，表明股價的探底階段可能結束，
#又將進入一個盤整階段。投資者可以逢低少量買入股票。

#5當CCI指標在﹢100線——﹣100線的常態區間運行時，
#投資者則可以用KDJ、CCI等其他超買超賣指標進行研判。

#----------simple average line---------------

def MA(data,N):  
    ma = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        MA_mean = sum(data[i:i+N]) / N
        ma.append(MA_mean)

    len2 = len(ma)

    return ma

#computing MD = sum(MA - closed price) / N

def MD(closed_price,N):
    md = []
    value = []
    len1 = len(closed_price)
    print('your input length of data = ',len1)
    MA_mean = MA(closed_price,N)
    len2 = len(MA_mean)
    for i in range(len2):           
        value_value = MA_mean[i] - data[i + N -1]
        value.append(value_value)

    len3 = len(value)
    for i in range(len3 - N + 1):
        value_mean = sum(value[i:i+N]) / N
        md.append(value_mean)

    len4 = len(md)
    print('your MD length of data = ',len4)

    return md
    
#computing TP = (high price + low price - closed price) / 3

def TP(high_price,low_price,closed_price):
    h = high_price
    l = low_price
    c = closed_price
    tp = []
    len1 = len(c)
    for i in range(len1):
        TP_value = (h[i] + l[i] - c[i]) / 3
        tp.append(TP_value)

    len2 = len(tp)
    print('your TP length of data = ',len2)
    return tp

#computing CCI = (TP - MA) / MD * 0.015

def CCI(high_price,low_price,closed_price,N):
    h = high_price
    l = low_price
    c = closed_price
    cci = []
    ma = MA(c,N)
    md = MD(c,N)
    tp = TP(h,l,c)
    len1 = len(tp)
    len2 = len(ma) #len1 - N + 1
    len3 = len(md) #len2 - N + 1

    for i in range(len3):
        ma_value = ma[i + N - 1]
        md_value = md[i]
        tp_value = tp[i + 2*N - 2]
        cci_value = (tp_value - ma_value) / tp_value*0.015
        cci.append(cci_value)

    len4 = len(cci)
    print('your CCI length of data = ',len4)

    return cci

#-------------Chande Momentum Oscillator--------------

#1.當CMO大於50時，處於超買狀態；當CMO小於50時處於超賣狀態。

#2.CMO的絕對值日越高，趨勢越強。較低的CMO絕對值（0附近）標示標的證券在水平方向波動。

#3.投資者還可利用CMO衡量趨勢強度的能力來改進趨勢跟蹤機制。
#例如當CMO的絕對值較高時僅根據趨勢跟蹤指標來操作；當COM的絕對值較低時轉而採用交易範圍指標。


def CMO(data,period):  
    len1 = period - 1
    up = []
    dn = []
    cmo = np.zeros(len1)  # the value matrix of cmo
    upsum = np.zeros(len1)  #the cumulative matrix of up
    dnsum = np.zeros(len1)  #the cumulative matrix of dn
    for i in range(len1):
        minus = data[i + 1] - data[i]
        if minus > 0:
            up.append(minus)
            dn.append(0)
        elif minus < 0:
            up.append(0)
            dn.append(minus)
        else:
            up.append(0)
            dn.append(0)
    upsum[0]= up[0]
    uptotal = up[0]
    dnsum[0]= dn[0]
    dntotal = dn[0]
    for i in range(1,len1):
        uptotal = uptotal + up[i]
        upsum[i] = uptotal
        dntotal = dntotal + dn[i]
        dnsum[i] = dntotal

    for i in range(len1):
        u = upsum[i]
        d = dnsum[i]
        cmo[i] = (u - d) / (u + d)

    return cmo

#----------------DMI趨向指標---------------------------

#應用法則

#+DI為上漲方向指標，+DI值愈高，代表漲勢愈明確且強烈；
#-DI為下跌方向指標，-DI值愈高時，代表跌勢愈明確且乏力。

#+DI線由下向上突破 -DI線時，為買進訊號，若ADX線再上揚，則漲勢將更強。
#因股價上漲，+DI線會向上攀升，顯示上升動量的增強，-DI線則會下跌，反映下跌動量的減弱。

#+DI線由上向下跌破 -DI線時，為賣出訊號，若ADX線再走上揚，則跌勢將更凶。

#ADX為趨勢動量指標，在漲勢或跌勢明顯的階段，ADX線會逐漸增加，代表上漲或下跌的力量已經增強。
#因此若ADX經常在低檔徘徊時，表示行情處於漲升乏力的弱勢市場中；
#若ADX經常在高檔徘徊，則代表行情處於作多有利的強勢市場中。

#+DI線與 -DI線經常接近甚至糾纏不清，此時若ADX值亦降至20以下時，代表行情處於盤整的牛皮階段，作多或作空均不易獲利。

#當股價到達高峰或谷底時，，ADX會在其前後達到最高點後反轉，因此，當ADX從上升的走向轉而為下降時，顯示行情即將反轉。
#故在漲勢中，ADX在高檔處由升轉跌，表示漲勢即將結束；反之，在跌勢中，ADX也在高檔處由升轉跌，亦表示跌勢將告結束。


#-------------DM---------------
#用當日最高價減去前一日最高價：+DM＝HIGH-HIGH[1]

def average(X,N):
    len1 = len(x)
    sum_value = sum(X[0:N])
    x_average = [sum_value/N]
    for i in X[N:len1]:
        s = sum_value
        sum_value = (s*(N - 1) + i)
        x_average.append((sum_value/N))

    return x_average


def DM_p(high_price,N):
    h = high_price
    DM_p = []
    len1 = len(high_price)
    for i in range(len1 - 1):
        h_value = h[i+1] -h[i]
        DM_p.append(h_value)

    real_DM_p = []
    
    for i in DM_p:
        if i>0:
            real_DM_p.append(i)

        else:
            real_DM_p.append(0)

    dm_p = average(real_DM_p,N)

    return dm_p

def DM_n(low_price,N):
    l = low_price
    DM_n = []
    len1 = len(low_price)
    for i in range(len1 - 1):
        l_value = l[i] -l[i + 1]
        DM_n.append(l_value)

    real_DM_n = []
    
    for i in DM_n:
        if i>0:
            real_DM_n.append(i)

        else:
            real_DM_n.append(0)

    dm_n = average(real_DM_n,N)

    return dm_n

#-------------TR-----------

def tr(close_price,high_price,low_price,N):
    c = close_price
    h = high_price
    l = low_price
    len1 = len(c)
    TR = []
    for i in range(len1 - 1):
        TR.append(max(h[i+1] - l[i+1],abs(h[i+1] - c[i]),abs(c[i] - l[i+1])))

    TR_value = average(TR,N)

    return TR_value

#------------+DI----------------

def DI_p(close_price,high_price,low_price,N):
    di_p = []
    dm_p = DM_p(high_price,N)
    TR = tr(close_price,high_price,low_price,N)
    for i,j in zip(dm_p,TR):
        di_p.append((i/j) * 100)

    return di_p

#----------- -DI-----------------

def DI_n(close_price,high_price,low_price,N):
    di_n = []
    dm_n = DM_n(low_price,N)
    TR = tr(close_price,high_price,low_price,N)
    for i,j in zip(dm_n,TR):
        di_n.append((i/j) * 100)

    return di_n

#-------------DX-----------------

def DX(close_price,high_price,low_price,N):
    dx = []
    di_p = DI_p(close_price,high_price,low_price,N)
    di_n = DI_n(close_price,high_price,low_price,N)
    for i,j in zip(di_p,di_n):
        dx.append(((i-j)/(i+j))*100)

    return dx

#-------------ADX-----------------

def ADX(close_price,high_price,low_price,N):
    dx = DX(close_price,high_price,low_price,N)
    adx = average(dx,N)

    return adx

#--------------簡易波動指標(Ease of Movement Value)----------------

#簡易波動指標（Ease of Movement Value）又稱EMV指標，它是由RichardW．ArmJr
#根據等量圖和壓縮圖的原理設計而成,目的是將價格與成交量的變化結合成一個波動指標來反映股價或指數的變動狀況。
#由於股價的變化和成交量的變化都可以引發該指標數值的變動,因此,EMV實際上也是一個量價合成指標。

#---------指標的運用---------

#1.當EMV由下往上穿越0軸時，買進。
#2.當EMV由上往下穿越0軸時，賣出。

#EMV simple volatility index
#two factors index for price and volume

def EMV(highprice,lowprice,vol,period,N=15):
    hp = highprice
    lp = lowprice
    len1 = period
    mid = []
    bro = []
    em = []
    for i in range(1,len1):
        midvalue = (hp[i] + lp[i])/2 - (hp[i-1] + lp[i-1])/2
        mid.append(midvalue)
        brovalue = vol[i] / (hp[i] - lp[i])
        bro.append(brovalue)
        emvalue = midvalue / brovalue 
        em.append(emvalue)

#--------------compute simple average index (N-day)-------------
    saemv = []
    emv_num = len1 - N + 1
    for i in range(emv_num):
        emv_sum = sum(em[i:i+N])
        emv_mean = emv_sum / N
        saemv.append(emv_mean)

    return saemv

#-------------RSI相對強弱指數-------------------
#相對強弱指數（RSI）是通過比較一段時期內的平均收盤漲數和平均收盤跌數來分析市場買沽盤的意向和實力，
#從而作出未來市場的走勢。

#RSI＝[上升平均數÷(上升平均數＋下跌平均數)]×100

def upvalue(data,N):
    change_value = []
    len1 = len(data)

    for i in range(N + 1):
        c_value = data[i+1] - data[i]
        change_value.append(c_value)

    up_value = []

    for i in range(N):
        if change_value[i] > 0:
            c_value = change_value[i]
            up_value.append(c_value)

    up_mean_value = sum(up_value) / N

    return up_mean_value

def downvalue(data,N):
    change_value = []
    len1 = len(data)

    for i in range(N + 1):
        c_value = data[i+1] - data[i]
        change_value.append(c_value)

    down_value = []
    len2 = len(change_value)

    for i in range(N):
        if change_value[i] < 0:
            c_value = -(change_value[i])
            down_value.append(c_value)

    down_mean_value = sum(down_value) / N

    return down_mean_value

def RSI(data,N):
    d = downvalue(data,N)
    u = upvalue(data,N)
    rsi_matrix = []
    rsi_value = (u / (u + d)) * 100
    rsi_matrix.append(rsi_value)
    len1 = len(data)

    for i in range(N+1,len1 - 1):
        day_u = u
        day_d = d

        if data[i] - data[i -1] > 0:
            u = (day_u*(N - 1) + (data[i] - data[i - 1])) / N
            d = day_d*(N - 1) / N
            rsi_value = (u / (u + d)) * 100
            rsi_matrix.append(rsi_value)

        else:
            u = day_u*(N - 1) / N
            d = (day_d*(N - 1) - (data[i] - data[i - 1])) / N
            rsi_value = (u / (u + d)) * 100
            rsi_matrix.append(rsi_value)

    return rsi_matrix
        
    
#-------Triple Exponentially Smoothed Moving Average (TRIX)--------

#三重指数平滑移动平均，长线操作时采用本指标的讯号，可以过滤掉一些短期波动的干扰，
#避免交易次数过于频繁，造成部分无利润的买卖，及手续费的损失。
#本指标是一项超长周期的指标

#--------TRIX的运用-------------

#1.当TRIX线一旦从下向上突破TRMA线，形成“金叉”时，预示着股价开始进入强势拉升阶段，投资者应及时买进股票。
#2.当TRIX线向上突破TRMA线后，TRIX线和TRMA线同时向上运动时，预示着股价强势依旧，投资者应坚决持股待涨。
#3.当TRIX线在高位有走平或掉头向下时，可能预示着股价强势特征即将结束，投资者应密切注意股价的走势，
#一旦K线图上的股价出现大跌迹象，投资者应及时卖出股票。
#4.当TRIX线在高位向下突破TRMA线，形成“死叉”时，预示着股价强势上涨行情已经结束，投资者应坚决卖出余下股票，及时离场观望。
#5.当TRIX线向下突破TRMA线后，TRIX线和TRMA线同时向下运动时，预示着股价弱势特征依旧，投资者应坚决持币观望。
#6.当TRIX线在TRMA下方向下运动很长一段时间后，并且股价已经有较大的跌幅时，如果TRIX线在底部有走平或向上勾头迹象时，
#一旦股价在大的成交量的推动下向上攀升时，投资者可以及时少量地中线建仓。
#7.当TRIX线再次向上突破TRMA线时，预示着股价将重拾升势，投资者可及时买入，持股待涨。


#计算N日的指数移动平均线EMA

def EMA(data,N):
    ma = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        MA_mean = sum(data[i:i+N]) / N
        ma.append(MA_mean)

    return ma

def TR(data,N):
    len1 = len(data)
    ma = EMA(data,N)
    for i in range(2):
        ma = EMA(ma,N)

    len2 = len(ma)
    print('the TR length is = ',len2)

    return ma

def TRIX(data,N):
    TR_array = TR(data,N)
    len1 = len(TR_array)
    trix = []
    for i in range(len1 - 1):
        trix_value = ((TR_array[i+1] - TR_array[i]) / TR_array[i])*100
        trix.append(trix_value)

    len2 = len(trix)
    print('the TRIX length is = ',len2)
    
    return trix

def TRMA(data,N,M):
    trix = TRIX(data,N)
    trma = EMA(trix,M)

    len1 = len(trma)
    print('the TRMA length is =',len1)

    return trma

def ix_minus_ma(data,N,M):
    trix = TRIX(data,N)
    trma = TRMA(data,N,M)
    t_m_t = []
    len1 = len(trma)
    for i in range(len1):
        t_m_t_value = trix[i+M-1] - trma[i]
        t_m_t.append(t_m_t_value)

    return t_m_t

#---------------KD index------------

#1.超買超賣區域的判斷──％Ｋ值在８０以上，％Ｄ值在７０以上為超買的一般標準。％Ｋ值輕２０以下，％Ｄ值在３０以下，及時為超賣的一般標準。 

#2.背馳判斷──當股價走勢一峰比一峰高時，隨機指數的曲線一峰比一峰低，
#或股價走勢一底比一底低時，隨機指數曲線一底比一底高，這種現象被稱為背馳，隨機指數與股價走勢產生背馳時，一般為轉勢的訊號，表明中期或短期走勢已到頂或見底，此時應選擇正確的買賣時機。 

#3.％Ｋ線與％Ｄ線交叉突破判斷──當％Ｋ值大於％Ｄ值時，表明當前是一種向上漲升的趨勢，因此％Ｋ線從下向上突破％Ｄ線時，是買進的訊號，
#反之，當％Ｄ值大於％Ｋ值，表明當前的趨勢向下跌落，因而％Ｋ線從上向下跌破％Ｄ線時，是賣出訊號。 
#％Ｋ線與％Ｄ線的交叉突破，在８０以上或２０以下較為準確，ＫＤ線與強弱指數不同之處是，它不僅能夠反映市場的超買或超賣程度，還能通過交叉突破達到歸出買賣訊號的功能，
#但是，當這種交叉突破在５０左右發生，走勢又陷入盤局時，買賣訊號應視為無效。 

#4.Ｋ線形狀判斷──當％Ｋ線傾斜度趨於平緩時，是短期轉勢的警告訊號，這種情況在大型熱門股及指數中准確度較高；而在冷門股或小型股中准確度則較低。 
#5.另外隨機指數還有一些理論上的轉向訊號：Ｋ線和Ｄ線上升或下跌的速度減弱，出現屈曲，通常都表示短期內會轉勢；
#Ｋ線在上升或下跌一段時期後，突然急速穿越Ｄ線，顯示市勢短期內會轉向：Ｋ線跌至零時通常會出現反彈至２０至２５之間，短期內應回落至接近零。
#這時，市勢應開始反彈。如果Ｋ線升至１００，情況則剛好相反

#--------------------------------------------

def max_value(data,N):
    m_v = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        mv = max(data[i:i+N])
        m_v.append(mv)

    return m_v

def min_value(data,N):
    m_v = []
    len1 = len(data)
    for i in range(len1 - N + 1):
        mv = min(data[i:i+N])
        m_v.append(mv)

    return m_v
        

def K(closed_price,high_price,low_price,K):
    c,h,l = closed_price,high_price,low_price
    K_value = []
    len1 = len(c)
    m_hk = max_value(h,K)
    m_lk = min_value(l,K)
    for i in range(len1 - K + 1):
        kv = ((c[i+K-1] - m_lk[i])/(m_hk[i] - m_lk[i])) * 100
        K_value.append(kv)

    return K_value

def D(closed_price,high_price,low_price,K,D):
    c,h,l = closed_price,high_price,low_price
    D_value = []
    m_hk = max_value(h,D)
    m_lk = min_value(l,D)
    len1 = len(m_hk)
    for i in range(len1 - K + D):
        dv = (m_hk[i+K-D] / m_lk[i+K-D])*100
        D_value.append(dv)

    return D_value

def KD_dis(closed_price,high_price,low_price,K_n,D_n):
    c,h,l = closed_price,high_price,low_price
    dis = []
    for i,j in zip(K(c,h,l,K_n),D(c,h,l,K_n,D_n)):
        dis_value = i - j
        dis.append(dis_value)

    return dis
