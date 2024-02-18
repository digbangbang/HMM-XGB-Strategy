# -*- coding: utf-8 -*-
import dolphindb as ddb
import talib as ta
import re
s=ddb.session()
import pandas as pd
import tqdm
#s.connect('10.0.60.55',8509,'admin','123456')
s.connect('192.90.11.102',8502,'admin','dolphin40')
import datetime as dt
#import akshare as ak
class GetData():

    @classmethod
    def _get_alpha_str(self,s):  # 获取字母
        result = ''.join(re.findall(r'[A-Za-z]', s))
        return result

    @classmethod
    def Future_hist_Mcandle(self,product,dateStart,dateEnd,types='main',ktype='1'):
        product=product.upper()
        if types=='index': #主力指数
            sql = f"TL_Get_Future_Index(`{product},{dateStart},{dateEnd},`{ktype})"
        if types=='main':#主力K线
            sql=f"TL_Get_Future_Main([`{product}],{dateStart},{dateEnd},`{ktype})"
        if types=='main_weight':#主力K线加权,向后赋权
            sql=f'TL_Get_Future_Main_w([`{product}],{dateStart},{dateEnd},`{ktype})'
        data = s.run(sql)
        return data

    @classmethod
    def get_multi_future(self,datatype,products,dateStart,dateEnd,ktype=1):
        sql=f"get_multi_future(`{datatype},{products},{dateStart},{dateEnd},{int(ktype)})"
        open, high, low, close,volume, oi ,_=s.run(sql)
        return open, high, low, close,volume, oi

    @classmethod
    def Future_hist_Mtick(self,product,dateStart,dateEnd,type='main'):
        product = product.upper()
        if type=='main':#主力TICK
            sql=f"select  * from loadTable('dfs://G2FutureMainTick2','FutureMainTick') where between(tradingDay,{dateStart}:{dateEnd}),product in [`{product}]; "
        elif type=='index':#主力tick指数
            sql=f"select  * from loadTable('dfs://G2FutureIndex2','FutureIndex') where between(tradingDay,{dateStart}:{dateEnd}),product in [`{product}]; "
        data = s.run(sql)
        return data

    @classmethod
    def Future_hist_indexsymbols(self,product,dateStart,dateEnd,ktype='1'):
        '''
        期货指数对应合约K线
        '''
        sql=f"TL_Get_indexsymbols(`{product},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def Future_hist_tick(self,symbol,dateStart,dateEnd):#期货历史TICK
        p=self._get_alpha_str(symbol)
        sql=f"select * from loadTable('dfs://FUTURE_TAQ','FUTURE_TAQ') where  between (tradingDay,{dateStart}:{dateEnd}),product= `{p},symbol=`{symbol}"
        data = s.run(sql)
        return data

    # @classmethod
    # def Future_realtime_tick(self,symbols): #期货实时数据
    #     sql=f"select last(timestamp(add(timestamp(date),time))) as date,last(product) as product,last(preClose) as preClose,last(last) as last ,last(volume) as volume,last(oi) as oi,last(limitDown) as limitDown," \
    #         f"last(limitUp) as limitUp,last(askPrice1) as askPrice1,last(bidPrice1) as bidPrice1,last(askVolume1) as askVolume1,last(bidVolume1) as bidVolume1 from Future_stream where symbol in {symbols} group by symbol"
    #     return z.run(sql)

    @classmethod
    def Future_hist_candle(self,symbol,dateStart,dateEnd,ktype='1'): #期货历史K线
        sql=f"TL_Get_Future(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def Hist_candle_spread(self,symbol1,c1,symbol2,c2,dateStart,dateEnd,ktype='1'):
        '''
        历史k线价差 期货,股票
        '''
        sql=f"Get_History_Spread({dateStart},{dateEnd},`{symbol1},{c1},`{symbol2},{c2},{int(ktype)})"
        return s.run(sql)

    @classmethod
    def Hist_Tick_spread(self,symbol1,c1,symbol2,c2,dateStart,dateEnd): #历史tick价差，不限期货
        sql=f"Get_History_Spread_Tick({dateStart},{dateEnd},`{symbol1},{c1},`{symbol2},{c2})"
        return s.run(sql)

    @classmethod
    def Get_future_spread_Mtick(self,product1,c1,product2,c2,dateStart,dateEnd): #历史tick价差，主力TICK,按品种分
        sql=f"Get_Future_Spread_MTick(`{product1},{c1},`{product2},{c2},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_sh_trade(self,symbol,dateStart,dateEnd):#上交所分笔成交
        sql=f"stock_l2_trade(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_sz_trade(self,symbol,dateStart,dateEnd):#深交所分笔成交
        sql=f"stock_szl2_trade(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_sz_order(self,symbol,dateStart,dateEnd):#深交所分笔委托
        sql=f"stock_szl2_order(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_sh_order(self,symbol,dateStart,dateEnd):#上交所分笔委托
        sql=f"stock_shl2_order(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_candle(self,symbol,dateStart,dateEnd,ktype='1'): #股票分时数据
        sql=f"stock_candle(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def Stock_l2_Tick(self,symbol,dateStart,dateEnd): #股票L2行情 全部字段
        sql=f"stock_tick(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_Tick(self,symbol,dateStart,dateEnd): #股票L2行情  只有常用字段
        sql=f"stock_l2_tick(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_index_Tick(self,symbol,dateStart,dateEnd):  #证券指数
        sql=f"stock_index_tick(`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Stock_index_candle(self,symbol,dateStart,dateEnd,ktype='1'):#证券指数K线
        sql=f"stock_index_candle(`{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def Finance(self, symbol_list, dateStart, dateEnd, type='balance'):  # 财务数据
        '''
        symbol_list:list  如查询全部，直接赋 []
        dateStart:date string
        dateEnd:date string
        '''
        if type == 'balance':  # 股票.合并资产负债表
            sql = f"Balance_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'Income':  # 股票. 合并利润分配表
            sql = f"Income_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'cashflow':  # 股票.合并现金流量表
            sql = f"CashFlow_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'indicator':  # 股票. 主要财务指标
            sql = f"MajorIndicator_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'shareholder':  # 股票.十大股东
            sql = f"Shareholder_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'rzrq':  # 股票. 融资融券明细
            sql = f"RZRQ_data({symbol_list},{dateStart},{dateEnd})"
        elif type == 'fhsg':  # 分红送股
            sql = f"FHSG_data({symbol_list},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Option_Candle(self,underlying,symbolS,dateStart,dateEnd,ktype='1'):#期权k线
        '''
         symbol:list,取全部合约赋值[]
         underlying:可以为空
         '''
        sql=f"TL_option_candle(`{underlying},{symbolS},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def Option_Tick(self,underlying,symbol,dateStart,dateEnd):#期权tick
        sql=f"TL_option_TICK(`{underlying},`{symbol},{dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def Option_Greeks_Candle(self,underlying,symbol,dateStart,dateEnd,ktype='1'):
        '''
        symbol:list,取全部合约赋值[]
        '''
        sql=f"TL_option_greeks_candle(`{underlying},{symbol},{dateStart},{dateEnd},`{ktype})"
        return s.run(sql)

    @classmethod
    def TradingDays(self,dateStart,dateEnd): #获取交易日,返回列表
        sql=f"Get_tradingday({dateStart},{dateEnd})"
        return s.run(sql)

    @classmethod
    def clearcache(self): #dolphin清除缓存
        s.run("pnodeRun(clearAllCache)")

    @classmethod
    def SH_symbols(self,date):#上海证券市场合约
        sql=f"SH_symbol({date})"
        return s.run(sql)

    @classmethod
    def SZ_symbols(self,date):#深圳证券市场合约
        sql=f"SZ_symbol({date})"
        return s.run(sql)

    @classmethod
    def close(self):
        s.close()

    @classmethod
    def alphabeta(self,symbol, dateStart,dateEnd, index:str='000300',i:int=0): #股票alpha，beta
        '''
        :param symbol: list
        :param dateStart:
        :param dateEnd:
        :param index:
        :param i:
        :return: dataframe
        '''
        from sklearn.linear_model import LinearRegression
        import numpy as np
        i = pow(i + 1, 1 / 252) - 1
        stock_daily=pd.DataFrame(columns=['symbol','date','close','pre_close'])
        for s in symbol:
            sd=(self.Stock_candle(s,dateStart,dateEnd,'D'))[['symbol','date','close']]
            sd['pre_close'] = sd['close'].shift(1)
            sd=sd.dropna(0)
            stock_daily =pd.concat([stock_daily, sd])
        stock_daily.index=stock_daily.symbol
        index_daily = self.Stock_index_candle(index,dateStart,dateEnd,'D')[['symbol','date','close']]
        index_daily['pre_close'] = index_daily['close'].shift(1)
        index_daily=index_daily.dropna(0)
        # 构造指数的收益率
        x = np.array((index_daily['close'] - index_daily['pre_close']) / index_daily['pre_close'] - i)
        x = x.reshape(len(x), 1)
        ab_list = []
        for stock in symbol:
            # 构造股票的收益率
            pre_close_series = stock_daily['pre_close'][stock]
            pre_close_series = pre_close_series.fillna(1)
            close_series = stock_daily['close'][stock]
            close_series = close_series.fillna(1)
            y = np.array((close_series - pre_close_series) / pre_close_series - i)
            y = y.reshape(len(y), 1)
            # 线性回归
            line_reg = LinearRegression()
            # 训练数据集,训练完成后，参数会保存在对象line_reg中
            line_reg.fit(x, y)
            # line_reg.intercept_为截距，就是w0，line_reg.coef_为其他参数，coef的全拼为coefficient
            ab_list.append([line_reg.intercept_[0], line_reg.coef_[0][0]])
        return pd.DataFrame(data=ab_list, index=symbol, columns=['alpha', 'beta'])

    @classmethod
    def Indicators(self,df):
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low','volume':'Volume'}, inplace=True)
        dfindicator = pd.DataFrame({
            'Open': df.Open, 'Close': df.Close, 'High': df.High, 'Low': df.Low,'Volume':df.Volume,
            #'Y': df.Close.shift(-1) / df.Close - 1,
            'Return': df.Close / df.Close.shift(1) - 1,
            'Return1': df.Close.shift(1) / df.Close.shift(2) - 1,
            'Return2': df.Close.shift(2) / df.Close.shift(3) - 1,
            'HC': df.High / df.Close, 'LC': df.Low / df.Close, 'HL': df.High / df.Low, 'OL': df.Open / df.Low,
            ############Overlap Studies Functions
            'DEMA': ta.DEMA(df.Close, timeperiod=30),
            'EMA12': ta.EMA(df.Close, timeperiod=12),
            'EMA26': ta.EMA(df.Close, timeperiod=26),
            'HT_TRENDLINE': ta.HT_TRENDLINE(df.Close),
            'KAMA': ta.KAMA(df.Close, timeperiod=30),
            'MA': ta.MA(df.Close, timeperiod=30, matype=0),
            #'MAVP': ta.MAVP(df.Close, 5, minperiod=2, maxperiod=30, matype=0),
            'MIDPOINT': ta.MIDPOINT(df.Close, timeperiod=14),
            'MIDPRICE': ta.MIDPRICE(df.High, df.Low, timeperiod=14),
            'SAR': ta.SAR(df.High, df.Low, acceleration=0, maximum=0),
            'SAREXT': ta.SAREXT(df.High, df.Low, startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                                accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0,
                                accelerationmaxshort=0),
            'T3': ta.T3(df.Close, timeperiod=5, vfactor=0),
            'SMA': ta.SMA(df.Close, timeperiod=30),
            'TEMA': ta.TEMA(df.Close, timeperiod=30),
            'TRIMA': ta.TRIMA(df.Close, timeperiod=30),
            'WMA': ta.WMA(df.Close, timeperiod=30),
            #######  Momentum Indicator Functions
            'ADX': ta.ADX(df.High, df.Low, df.Close, timeperiod=14),
            'ADXR': ta.ADXR(df.High, df.Low, df.Close, timeperiod=14),
            'APO': ta.APO(df.Close, fastperiod=12, slowperiod=26, matype=0),
            'AROONDOWN':ta.AROON(df.High, df.Low, timeperiod=14)[0],
            'AROONUP':ta.AROON(df.High, df.Low, timeperiod=14)[1],
            'AROONOSC': ta.AROONOSC(df.High, df.Low, timeperiod=14),
            'BOP': ta.BOP(df.Open, df.High, df.Low, df.Close),
            'CCI': ta.CCI(df.High, df.Low, df.Close, timeperiod=14),
            'CMO': ta.CMO(df.Close, timeperiod=14),
            'DX': ta.DX(df.High, df.Low, df.Close, timeperiod=14),
            'MFI': ta.MFI(df.High, df.Low, df.Close, df.Volume, timeperiod=14),
            'MINUS_DI': ta.MINUS_DI(df.High, df.Low, df.Close, timeperiod=14),
            'MINUS_DM': ta.MINUS_DM(df.High, df.Low, timeperiod=14),
            'MOM': ta.MOM(df.Close, timeperiod=10),
            'RSI': ta.RSI(df.Close, timeperiod=14),
            'MACD_macd':ta.MACD(df.Close,12,26,9)[0],
            'MACD_sign': ta.MACD(df.Close, 12, 26, 9)[1],
            'MACD_hist': ta.MACD(df.Close, 12, 26, 9)[2],
            'SLOWK':ta.STOCH(df.High,df.Low,df.Close,fastk_period=9,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)[0],
            'SLOWD': ta.STOCH(df.High, df.Low, df.Close, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3,
                              slowd_matype=0)[1],
            ######Volume Indicator Functions
            'AD': ta.AD(df.High, df.Low, df.Close, df.Volume),
            'ADOSC': ta.ADOSC(df.High, df.Low, df.Close, df.Volume, fastperiod=3, slowperiod=10),
            'OBV': ta.OBV(df.Close, df.Volume),
            ######Volatility Indicator Functions
            'ATR': ta.ATR(df.High, df.Low, df.Close, timeperiod=14),
            'NATR': ta.NATR(df.High, df.Low, df.Close, timeperiod=14),
            'TRANGE': ta.TRANGE(df.High, df.Low, df.Close),
            ####Price Transform Functions
            'AVGPRICE': ta.AVGPRICE(df.Open, df.High, df.Low, df.Close),
            'MEDPRICE': ta.MEDPRICE(df.High, df.Low),
            'TYPPRICE': ta.TYPPRICE(df.High, df.Low, df.Close),
            'WCLPRICE': ta.WCLPRICE(df.High, df.Low, df.Close),
            ######Cycle Indicator Functions
            'HT_DCPERIOD': ta.HT_DCPERIOD(df.Close),
            'HT_DCPHASE': ta.HT_DCPHASE(df.Close),
            'HT_TRENDMODE': ta.HT_TRENDMODE(df.Close),
            #######Statistic Functions
            'BETA': ta.BETA(df.High, df.Low, timeperiod=5),
            'CORREL': ta.CORREL(df.High, df.Low, timeperiod=30),
            'LINEARREG': ta.LINEARREG(df.Close, timeperiod=14),
            'LINEARREG_ANGLE': ta.LINEARREG_ANGLE(df.Close, timeperiod=14),
            'LINEARREG_INTERCEPT': ta.LINEARREG_INTERCEPT(df.Close, timeperiod=14),
            'LINEARREG_SLOPE': ta.LINEARREG_SLOPE(df.Close, timeperiod=14),
            'STDDEV': ta.STDDEV(df.Close, timeperiod=5, nbdev=1),
            'TSF': ta.TSF(df.Close, timeperiod=14),
            'VAR': ta.VAR(df.Close, timeperiod=5, nbdev=1),
            #######Math Transform Functions
            # 'ACOS': ta.ACOS(df.Close), 'ASIN': ta.ASIN(df.Close), 'COSH': ta.COSH(df.Close),
            # 'EXP': ta.EXP(df.Close), 'SINH': ta.SINH(df.Close),
            'CEIL': ta.CEIL(df.Close), 'COS': ta.COS(df.Close), 'ATAN': ta.ATAN(df.Close),
            'FLOOR': ta.FLOOR(df.Close), 'LN': ta.LN(df.Close), 'SIN': ta.SIN(df.Close),
            'LOG10': ta.LOG10(df.Close),
            'SQRT': ta.SQRT(df.Close), 'TAN': ta.TAN(df.Close), 'TANH': ta.TANH(df.Close),
            ###### Math Operator Functions
            'ADD': ta.ADD(df.High, df.Low), 'SUB': ta.SUB(df.High, df.Low),
            'MULT': ta.MULT(df.High, df.Low), 'DIV': ta.DIV(df.High, df.Low),
            'MAX': ta.MAX(df.Close, timeperiod=30), 'MIN': ta.MIN(df.Close, timeperiod=30),
            'MAXINDEX': ta.MAXINDEX(df.Close, timeperiod=30),
            'MININDEX': ta.MININDEX(df.Close, timeperiod=30),
            'SUM': ta.SUM(df.Close, timeperiod=30),
            'PPO':ta.PPO(df.Close),
            'TRIX':ta.TRIX(df.Close),
            'ROCR100':ta.ROCR100(df.Close),
            'ROC':ta.ROC(df.Close),
            'ROCP':ta.ROCP(df.Close),
            'ROCR':ta.ROCR(df.Close),
            'PLUS_DI':ta.PLUS_DI(df.High,df.Low,df.Close),
            'PLUS_DM':ta.PLUS_DM(df.High,df.Low),
            'WILLR':ta.WILLR(df.High,df.Low,df.Close),
            'ULTOSC':ta.ULTOSC(df.High,df.Low,df.Close)

        })
        return dfindicator

    @classmethod
    def Pattern(self,df):
            df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
            dfpattern = pd.DataFrame({
                'Y': df.Close.shift(-1) / df.Close - 1,
                'TwoCrows': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # D
                'ThreeBlackCrows': ta.CDL3BLACKCROWS(df.Open, df.High, df.Low, df.Close),  # D
                'ThreeInsideUD': ta.CDL3INSIDE(df.Open, df.High, df.Low, df.Close),  # U
                'ThreeLineStrike': ta.CDL3LINESTRIKE(df.Open, df.High, df.Low, df.Close),  # D
                'ThreeOutsideUD': ta.CDL3OUTSIDE(df.Open, df.High, df.Low, df.Close),  # U
                'ThreeStarsInTheSouth': ta.CDL3STARSINSOUTH(df.Open, df.High, df.Low, df.Close),  # U
                'ThreeAdvancingWhiteSoldiers': ta.CDL3WHITESOLDIERS(df.Open, df.High, df.Low, df.Close),  # U
                'AdvanceBlock': ta.CDLADVANCEBLOCK(df.Open, df.High, df.Low, df.Close),  # U
                'AbandonedBaby': ta.CDLABANDONEDBABY(df.Open, df.High, df.Low, df.Close, penetration=0),  # R
                'BeltHold': ta.CDLBELTHOLD(df.Open, df.High, df.Low, df.Close),  # U
                'Breakaway': ta.CDLBREAKAWAY(df.Open, df.High, df.Low, df.Close),  # U
                'ClosingMarubozu': ta.CDLCLOSINGMARUBOZU(df.Open, df.High, df.Low, df.Close),  # M
                'ConcealingBabySwallow': ta.CDLCONCEALBABYSWALL(df.Open, df.High, df.Low, df.Close),  # U
                'Counterattack': ta.CDLCOUNTERATTACK(df.Open, df.High, df.Low, df.Close),  #
                'DarkCloudCover': ta.CDLDARKCLOUDCOVER(df.Open, df.High, df.Low, df.Close, penetration=0),  # D
                'Doji': ta.CDLDOJI(df.Open, df.High, df.Low, df.Close),  #
                'DojiStar': ta.CDLDOJISTAR(df.Open, df.High, df.Low, df.Close),  # R
                'DragonflyDoji': ta.CDLDRAGONFLYDOJI(df.Open, df.High, df.Low, df.Close),  # R
                'EngulfingPattern': ta.CDLENGULFING(df.Open, df.High, df.Low, df.Close),  # R
                'EveningDojiStar': ta.CDLEVENINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RD
                'EveningStar': ta.CDLEVENINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
                'UDgapSideWhiteLines': ta.CDLGAPSIDESIDEWHITE(df.Open, df.High, df.Low, df.Close),  # M
                'GravestoneDoji': ta.CDLGRAVESTONEDOJI(df.Open, df.High, df.Low, df.Close),  # RU
                'Hammer': ta.CDLHAMMER(df.Open, df.High, df.Low, df.Close),  # R
                'HangingMan': ta.CDLHANGINGMAN(df.Open, df.High, df.Low, df.Close),  # R
                'HaramiPattern': ta.CDLHARAMI(df.Open, df.High, df.Low, df.Close),  # RU
                'HaramiCrossPattern': ta.CDLHARAMICROSS(df.Open, df.High, df.Low, df.Close),  # R
                'HighWaveCandle': ta.CDLHIGHWAVE(df.Open, df.High, df.Low, df.Close),  # R
                'HikkakePattern': ta.CDLHIKKAKE(df.Open, df.High, df.Low, df.Close),  # R
                'ModifiedHikkakePattern': ta.CDLHIKKAKEMOD(df.Open, df.High, df.Low, df.Close),  # M
                'HomingPigeon': ta.CDLHOMINGPIGEON(df.Open, df.High, df.Low, df.Close),  # R
                'IdenticalThreeCrow': ta.CDLIDENTICAL3CROWS(df.Open, df.High, df.Low, df.Close),  # D
                'InNeckPattern': ta.CDLINNECK(df.Open, df.High, df.Low, df.Close),  # D
                'InvertedHammer': ta.CDLINVERTEDHAMMER(df.Open, df.High, df.Low, df.Close),  # R
                'Kicking': ta.CDLKICKING(df.Open, df.High, df.Low, df.Close),  #
                'KickingByLength': ta.CDLKICKINGBYLENGTH(df.Open, df.High, df.Low, df.Close),  #
                'LadderBottom': ta.CDLLADDERBOTTOM(df.Open, df.High, df.Low, df.Close),  # RU
                'LongLeggedDoji': ta.CDLLONGLEGGEDDOJI(df.Open, df.High, df.Low, df.Close),  #
                'LongLineCandle': ta.CDLLONGLINE(df.Open, df.High, df.Low, df.Close),  #
                'Marubozu': ta.CDLMARUBOZU(df.Open, df.High, df.Low, df.Close),  #
                'MatchingLow': ta.CDLMATCHINGLOW(df.Open, df.High, df.Low, df.Close),  #
                'MatHold': ta.CDLMATHOLD(df.Open, df.High, df.Low, df.Close, penetration=0),  # M
                'MorningDoji': ta.CDLMORNINGDOJISTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
                'MorningStar': ta.CDLMORNINGSTAR(df.Open, df.High, df.Low, df.Close, penetration=0),  # RU
                'OnNeckPattern': ta.CDLONNECK(df.Open, df.High, df.Low, df.Close),  # MD
                'PiercingPattern': ta.CDLPIERCING(df.Open, df.High, df.Low, df.Close),  # RU
                'RickshawMan': ta.CDLRICKSHAWMAN(df.Open, df.High, df.Low, df.Close),  #
                'RFThreeMethods': ta.CDLRISEFALL3METHODS(df.Open, df.High, df.Low, df.Close),  # U
                'SeparatingLines': ta.CDLSEPARATINGLINES(df.Open, df.High, df.Low, df.Close),  # M
                'ShootingStar': ta.CDLSHOOTINGSTAR(df.Open, df.High, df.Low, df.Close),  # D
                'ShortLineCandle': ta.CDLSHORTLINE(df.Open, df.High, df.Low, df.Close),  #
                'SpinningTop': ta.CDLSPINNINGTOP(df.Open, df.High, df.Low, df.Close),  #
                'StalledPattern': ta.CDLSTALLEDPATTERN(df.Open, df.High, df.Low, df.Close),  # EU
                'StickSandwich': ta.CDLSTICKSANDWICH(df.Open, df.High, df.Low, df.Close),  #
                'Takuri': ta.CDLTAKURI(df.Open, df.High, df.Low, df.Close),  #
                'TasukiGap': ta.CDLTASUKIGAP(df.Open, df.High, df.Low, df.Close),  # MU
                'ThrustingPattern': ta.CDLTHRUSTING(df.Open, df.High, df.Low, df.Close),  # M
                'TristarPattern': ta.CDLTRISTAR(df.Open, df.High, df.Low, df.Close),  # R
                'UniqueRiver': ta.CDLUNIQUE3RIVER(df.Open, df.High, df.Low, df.Close),  # R
                'UGapTwoCrows': ta.CDLUPSIDEGAP2CROWS(df.Open, df.High, df.Low, df.Close),  # U
                'UDGapThreeMethods': ta.CDLXSIDEGAP3METHODS(df.Open, df.High, df.Low, df.Close)  # U
            })
            return dfpattern


class TLData():
    def __init__(self,IP='101',is_print=True):
        self.IP=IP
        self.connect()
        self.nid = 0
        self.is_print=is_print

    def connect(self):
        self.s = ddb.session()
        if self.IP=='55':
            self.s.connect('10.0.60.55', 8509, 'admin', '123456')
        elif self.IP=='101':
            self.s.connect('192.90.11.102', 8503, 'admin', 'dolphin40')
        else:
            raise ImportError('error,IP must 55 or 101')

    def yy_tick(self,symbol:str, dateStart:str, dateEnd:str,starttime:str,endtime:str):
        sql=f"yy_tick(`{symbol},{dateStart},{dateEnd},{starttime},{endtime})"
        print(sql)
        return self.s.run(sql)

    def TL_future(self, symbol:str, dateStart:str, dateEnd:str, cycle:str='60')->pd.DataFrame:
        '''
        symbol:str
        dateStart :str
        dateEnd:str
        cycle:str

        说明：
           TL_future函数，可以查询期货主力，期货指数，期货主力加权数据
           期货合约：symbol 例RB2201
           期货合约：symbol 例RB1 按合约顺序排序
           期货指数：symbol 例RB88
           期货主力：symbol 例RB99
           期货主力加权： symbol 例RB77

           cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        例：
        查询螺纹指数的30秒K线
        TL_future('RB88','2021.01.01','2021.05.05','30')
        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        sql = f"TL_Future(`{symbol},{dateStart},{dateEnd},`{cycle})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df = df.ffill()
        if 'symbol' not in df.columns:
            df['symbol'] = "Fut" + str(self.nid) + "_" + df['product']
        else:
            df['symbol'] = "Fut" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_stock(self, symbol:str, dateStart:str, dateEnd:str, cycle:str='60', ftype:str='')->pd.DataFrame:
        '''
        symbol:str
        dateStart :str
        dateEnd:str
        cycle:str
        ftype:str

        说明：
        TL_stock函数，可以查询股票数据，及加权数据，
        ftype=''：不复权，ftype='hfq'：后赋权，ftype='qfq':前复权
        cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        例：
        查询510050的前复权1分钟K线数据
        TL_stock(510050,'2021.01.01','2021.06.06','60',ftype='qfq')
        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        if '.' in str(symbol):
            symbol=str(symbol).split('.')[0]
        sql = f"TL_stock(`{symbol},{dateStart},{dateEnd},`{cycle},`{ftype})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df = df.ffill()
        df['symbol'] = "Stk" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_stock_index(self, symbol:str, dateStart:str, dateEnd:str, cycle:str='60')->pd.DataFrame:
        '''
        symbol:str
        dateStart :str
        dateEnd:str
        cycle:str
        说明：
        TL_stock_index函数，查询股票指数数据
        cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        例：
        查询000016，日级数据
        TL_stock_index('000016','2021.01.01','2021.06.06','D')
        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        if '.' in str(symbol):
            symbol=str(symbol).split('.')[0]
        sql = f"TL_stock_index(`{symbol},{dateStart},{dateEnd},`{cycle})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df['symbol'] = "Sti" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_option(self, undelier:str, symbol:str, dateStart:str, dateEnd:str, cycle:str='60')->pd.DataFrame:
        '''
        undelier:str  标的合约
        symbol:list    期权合约列表
        dateStart :str
        dateEnd:str
        cycle:str

        说明：
        TL_option函数，查询期权行情数据
        cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        undelier可以为空字符
        symbol为空列表，表示查询标的合约所有对应的期权合约
        例1：
        查询510050的所有期权合约
        TL_option('510050',[],'2021.01.01','2021.06.06',cycle='60')
        例2：
        仅查询某两个合约数据
        TL_option('510050',['10003246','10003245'],'2021.01.01','2021.06.06',cycle='60')
        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        sql = f"TL_option(`{undelier},{symbol},{dateStart},{dateEnd},`{cycle})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df['symbol'] = "Opt" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_option_M(self, undelier:str, dateStart:str, dateEnd:str, cycle:str='60', cp:str='c', expm:str='m1', optype:int=0,earlyday:int=0,greeks=False)->pd.DataFrame:
        '''
        undelier:str  标的合约
        dateStart :str
        dateEnd:str
        cycle:str
        cp:str  ; 'c' 或者'p'
        expm：str; 'm1'、'm2' 、'q1'、 'q2'
        optype：int ;   0:平直合约；1：实一档；-1：虚一档
        earlyday：int;提前缓合约
        greeks:bool; ture数据中有greeks
        说明：
        TL_option_M函数  查询期权档位数据
        cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        例1：
        查询510050的call,当月的平值行数数据
        TL_option_M('510050','2021.01.01','2021.06.06','60','c','m1',0)
        例2：
        查询510050的call,次月的实2档行数数据
        TL_option_M('510050','2021.01.01','2021.06.06','60','c','m2',2)
        expm='md',主连，可设置earlyday天进行提前换月；
        expm='mm' 选定合约后，在接下去的交割期内不换档，直至交割，可设置earlyday天进行提前换月；
        计算说明：
        合约根据标的指数当日开盘价来判断平值，实一档，虚一档
        比如510050 2021.09.14开盘价为3.278，那期权执行价和3.278差值绝对值最小的平值合约，除它平值期权外，其余合约根据绝对值大小判断实一档，虚一档

        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        if greeks==True or greeks=='true':
            greeks='true'
        else:
            greeks = 'false'
        sql = f"TL_option_M(`{undelier},{dateStart},{dateEnd},`{cycle},`{cp},`{expm},{optype},{earlyday},{greeks})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df = df.ffill()
        df['symbol'] = "Opt" + str(self.nid) + "_" + df['symbol']
        if 'presymbol_symbol' in df.columns:
            df['presymbol_symbol'] = "Opt" + str(self.nid) + "_" + df['presymbol_symbol']
        self.nid += 1
        return df

    def TL_option_M2(self, undelier:str, dateStart:str, dateEnd:str, cycle:str='60', cp:str='c', expm:str='m1', optype:int=0,earlyday:int=0,greeks=False)->pd.DataFrame:
        '''
        undelier:str  标的合约
        dateStart :str
        dateEnd:str
        cycle:str
        cp:str  ; 'c' 或者'p'
        expm：str; 'm1'、'm2' 、'q1'、 'q2'
        optype：int ;   0:平直合约；1：实一档；-1：虚一档
        earlyday：int;提前缓合约
        greeks:bool; ture数据中有greeks
        说明：
        TL_option_M函数  查询期权档位数据
        cycle='D',cycle='W',合约日级别和周级别数据。cycle=`0 表示查询合约的tick数据，cycle=`60 表示查询合约1分钟k线，以此内推，建议cycle为60 的公约数或公倍数
        例1：
        查询510050的call,当月的平值行数数据
        TL_option_M2('510050','2021.01.01','2021.06.06','60','c','m1',0)
        例2：
        查询510050的call,次月的实2档行数数据
        TL_option_M2('510050','2021.01.01','2021.06.06','60','c','m2',2)
        expm='md',主连，可设置earlyday天进行提前换月；
        expm='mm' 选定合约后，在接下去的交割期内不换档，直至交割，可设置earlyday天进行提前换月；
        计算说明：
        合约根据标的指数当日开盘价来判断平值，实一档，虚一档
        比如510050 2021.09.14开盘价为3.278，那期权执行价和3.278差值绝对值最小的平值合约，除它平值期权外，其余合约根据绝对值大小判断实一档，虚一档
        和TL_option_M1的区别是ATM的虚实度的方法不一样，数据库要选择60.55 ，集群数据库无当前数据
        '''
        if '.' in str(cycle):
            cycle=str(cycle).split('.')[0]
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        if greeks==True or greeks=='true':
            greeks='true'
        else:
            greeks = 'false'
        sql = f"TL_option_M2(`{undelier},{dateStart},{dateEnd},`{cycle},`{cp},`{expm},{optype},{earlyday},{greeks})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df = df.ffill()
        df['symbol'] = "Opt" + str(self.nid) + "_" + df['symbol']
        if 'presymbol_symbol' in df.columns:
            df['presymbol_symbol'] = "Opt" + str(self.nid) + "_" + df['presymbol_symbol']
        self.nid += 1
        return df

    def TL_option_pro(self,undelier:str, dateStart:str, dateEnd:str, cycle:str='60', cp:str='c', atm:int=0,expm:str='m1',next_exp='')->pd.DataFrame:
        '''
        数据来源 ("dfs://SHSOL1_CANDLE").("SHSOL1_CANDLE")
        返回  tradingDay,time,symbol,strikeprice,synf,TD,expterm,cp,askPrice1,bidPrice1,vol,delta,gamma,vega,theta,vix,underlying,underlying_close,atm
        '''
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        cycle = int(cycle) // 60
        if cycle < 1:
            raise ValueError('周期需大于等于1分钟')
        sql = f"TL_option_pro(`{undelier},{dateStart},{dateEnd},`{cp},`{expm},'{atm}',{cycle},`{next_exp})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        #df = df.ffill()
        df['symbol'] = "Opt" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_option_T(self,undelier:str, dateStart:str, dateEnd:str, cycle:str='60', cp:str='c', expm:str='m1',expmy='')->pd.DataFrame:
        '''
        说明  平值  和 虚实度重复
        expmy  输入远月  例expm='m1' 那expmy可以 ’m2‘

        '''
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        if '.' in cycle:
            cycle=cycle.split('.')[0]
        sql = f"TL_option_T(`{undelier},{dateStart},{dateEnd},{cycle},`{cp},`{expm},`{expmy})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        df = df.ffill()
        df['symbol'] = "Opt" + str(self.nid) + "_" + df['symbol']
        self.nid += 1
        return df

    def TL_option_vix(self,undelier:str, dateStart:str, dateEnd:str,cycle:str='60')->pd.DataFrame:
        if '.' in str(undelier):
            undelier=str(undelier).split('.')[0]
        cycle = int(cycle) // 60
        if cycle < 1:
            raise ValueError('周期需大于等于1分钟')
        sql=f"get_vix(`{undelier},{dateStart},{dateEnd},'',{cycle})"
        if self.is_print:
            print(sql)
        df=self.s.run(sql)
        df['uptime']=df['uptime'].dt.time
        df['time']=df.apply(lambda x:dt.datetime.combine(x.tradingDay,x.uptime),axis=1)
        return df

    def TL_get_underlying(self,symbol:str, dateStart:str, dateEnd:str)->pd.DataFrame:
        '''
        根据期权合约 返回表的合约  和这时间段内所有期权合约
        '''
        sql=f"get_underlying({dateStart},{dateEnd},'{symbol}')"
        underlying,symbols = self.s.run(sql)
        return underlying,symbols

    def TL_get_greeks(self ,undelier:str, symbol:str, dateStart:str, dateEnd:str, cycle='60')->pd.DataFrame:
        '''
        获取希腊字母指标损益
        '''
        sql = f"get_greeks(`{undelier},'{symbol}',{dateStart},{dateEnd},{cycle})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        return df

    def TL_get_greeks_incash(self ,undelier:str, symbol:str, dateStart:str, dateEnd:str, cycle='60')->pd.DataFrame:
        '''
        获取希腊字母指标损益
        '''
        sql = f"get_greeks_incash(`{undelier},'{symbol}',{dateStart},{dateEnd},{cycle})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        return df

    def TL_get_factors(self,undelier:str, symbol:str,factorname:str,dateStart:str, dateEnd:str,cp:str='',expm:str='',atm=999,cycle:str='60',types='last')->pd.DataFrame:
        '''
        获取指标
        symbol:str
        dateStart :str
        dateEnd:str
        cycle:str
        atn:int
        factorname:str
        cp:str
        expm:str
        说明对于不需要atm的指标默认999，不需要expm默认‘’，不需要cp默认‘’
        '''
        sql = f"TL_get_factors(`{factorname},`{undelier},'{symbol}',{dateStart},{dateEnd},`{expm},`{cp},{atm},{cycle},`{types})"
        if self.is_print:
            print(sql)
        df = self.s.run(sql)
        return df

    def close(self):
        self.s.close()
    def __del__(self):
        self.s.close()
#DF=GetData.Hist_candle_spread('IF2012',1,'IC2012',1,'2020.10.05','2020.11.25')

def spread(product1,start,end,cycle='D'):  #跨期价差
    import datetime
    sp_dff=pd.DataFrame()
    sql = f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product1}], tradingDay between({start}:{end}) "
    sp_df = s.run(sql)

    for day in tqdm.tqdm(sp_df.tradingDay.drop_duplicates()):
        day_symbol=sp_df[sp_df['tradingDay']==day]['symbol'].values
        symbol1=day_symbol[0]
        symbol2 = day_symbol[1]
        day=datetime.datetime.strftime(day, '%Y.%m.%d')
        if cycle=='0':
            sp = GetData.Hist_Tick_spread(symbol1, 1, symbol2, 1, day, day)
        else:
            sp=GetData.Hist_candle_spread(symbol1,1,symbol2,1,day,day,str(cycle))

        if len(sp_dff)==0:
            sp_dff=sp
        else:
            sp_dff=sp_dff.append(sp,ignore_index=True,sort=False)
    # s.close()
    # GetData.close()
    return sp_dff

def spread1(product1,start,end,cycle='D'):
    import datetime
    sp_dff=pd.DataFrame()
    sql = f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product1}], tradingDay between({start}:{end}) "
    sp_df = s.run(sql)
    for day in tqdm.tqdm(sp_df.tradingDay.drop_duplicates()):
        day_symbol=sp_df[sp_df['tradingDay']==day]['symbol'].values
        symbol1=day_symbol[0]
        symbol2 = day_symbol[1]
        day=datetime.datetime.strftime(day, '%Y.%m.%d')
        df1=GetData.Future_hist_candle(symbol1,day,day,str(cycle))
        df2 = GetData.Future_hist_candle(symbol2, day, day, str(cycle))
        sp=pd.merge(df1,df2,on='date')
        if len(sp_dff)==0:
            sp_dff=sp
        else:
            sp_dff=sp_dff.append(sp,ignore_index=True,sort=False)
    # s.close()
    # GetData.close()
    return sp_dff

def future_spread(product1,c1,product2,c2,start,end):#Tick
    import datetime,os
    sp_dff=pd.DataFrame()
    sql = f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product1}], tradingDay between({start}:{end}) "
    sp_df = s.run(sql)
    sql_=f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product2}], tradingDay between({start}:{end}) "
    sp_df_ = s.run(sql_)
    for day in tqdm.tqdm(sp_df.tradingDay.drop_duplicates()):
        try:
            day_symbol1=sp_df[sp_df['tradingDay']==day]['symbol'].values
            symbol1=day_symbol1[0]
            day_symbol2 = sp_df_[sp_df_['tradingDay'] == day]['symbol'].values
            symbol2 = day_symbol2[0]
            day=datetime.datetime.strftime(day, '%Y.%m.%d')
            sp=GetData.Get_future_spread_Mtick(product1,c1,product2,c2,day,day)
            path='resultss/'
            if not os.path.exists(path):
                os.mkdir(path)
            if len(sp)>0:
                sp.to_csv(path+symbol1+'_'+symbol2+'_'+str(day)+'.csv',encoding='gbk')
        except:pass

def wlc_spread(product1,start,end,ktype='D'):
    import datetime
    sp_dff=pd.DataFrame()
    sql = f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product1}], tradingDay between({start}:{end}) "
    print(sql)
    # sp_df = s.run(sql)
    # left_list=[]
    # right_list=[]
    # for day in sp_df.tradingDay.drop_duplicates():
    #     day_symbol = sp_df[sp_df['tradingDay'] == day]['symbol'].values
    #     symbol1=day_symbol[0]
    #     symbol2 = day_symbol[1]
    #     day=datetime.datetime.strftime(day, '%Y.%m.%d')
    #     left=GetData.Future_hist_candle(symbol1,day,day,'D')
    #     right=GetData.Future_hist_candle(symbol2,day,day,'D')
    #     left_list.append([symbol1,day,left['close'].values])
    #     right_list.append([symbol2, day, right['close'].values])
    # return left_list,right_list

def get_alpha_str(s):  # 获取字母
    result = ''.join(re.findall(r'[A-Za-z]', s))
    return result

def spread2(product1,start,end,cycle='D'):#跨期
    import datetime
    sql = f"select tradingDay,symbol from loadTable('dfs://G2FutureIndexComp2','FutureIndexComp') where product in [`{product1}], tradingDay between({start}:{end}) "
    sp_df = s.run(sql)
    tl=TLData()
    sp_dff=pd.DataFrame()
    for day in tqdm.tqdm(sp_df.tradingDay.drop_duplicates()):
        day_symbol=sp_df[sp_df['tradingDay']==day]['symbol'].values
        symbol1=day_symbol[0]
        symbol2 = day_symbol[1]
        day=datetime.datetime.strftime(day, '%Y.%m.%d')
        d1=tl.TL_future(symbol1,day,day,cycle)
        d2 = tl.TL_future(symbol2, day, day, cycle)
        df=pd.merge(d1,d2,on='time')
        if len(sp_dff)==0:
            sp_dff=df
        else:
            sp_dff=sp_dff.append(df,ignore_index=True,sort=False)
    tl.close()
    return sp_dff

if __name__ == '__main__':

    TL=TLData('55')
    #stki=TL.TL_stock_index('000300','2022.08.24','2022.08.24','0')
    #v=TL.TL_option_S('510050','2022.01.01','2022.02.02',cp='c',cycle='60')
    # #future_spread('CU',5,'BC',5,'2020.06.06','2021.05.26')
    # d=GetData.Stock_l2_Tick('515700', '2021.01.16', '2021.04.30')
    # # sql = f"data=stock_index_tick(`{'000016'},{'2021.08.16'},{'2021.08.25'});select symbol,`{'re'} as exchange,timestamp(add(timestamp(date),time)) as datetime,symbol as name,volume,0 as open_interest,last as last_price,volume as last_volume,0 as limit_up,0 as limit_down,open as open_price,high as high_price,low as low_price,preClose as pre_close," \
    # #       f" last as asd_price_1,last as bid_price_1,volume as ask_volume_1,volume as bid_volume_1 from data"
    # #d=s.run(sql)
    # GetData.close()
    #open, high, low, close,volume, oi=GetData.get_multi_future('fut',['A','AL','RB','CU'],'2022.01.01','2022.04.01',10)
    d=TL.TL_future('M','2023.06.21','2023.06.26',cycle='60')
    #SS = TL.TL_stock_index('000905', '2021.12.08', '2021.12.31', cycle='60')
    #f = TL.TL_option_vix('510300', '2022.12.19', '2022.12.23', cycle='600')
    #df=GetData.Future_hist_Mcandle('IF','2021.01.13', '2022.01.17')
    #df1 = GetData.Stock_index_candle('000905', '2016.01.01', '2017.01.01',ktype='10')
    #op=GetData.Option_Greeks_Candle('510050',[],'2021.01.13', '2021.03.17','10')
    #d1f=GetData.Future_hist_candle('BU2206','2022.03.23', '2022.03.28')
    #index=GetData.Stock_index_candle('000300','2020.01.10', '2022.01.17','10')
    #f = TL.TL_stock('159869', '2023.01.01', '2023.06.21', cycle='600')
    #l=GetData.Stock_Tick('300220','2021.12.28','2021.12.28')
    #stk=GetData.Stock_candle('159919','2021.12.24','2022.01.15')
    #idx = GetData.Future_hist_tick('RB2205', '2021.12.27', '2021.12.27')
    #idx1 = GetData.Future_hist_tick('ZC205', '2021.12.27', '2021.12.27')
    #dd=GetData.Future_hist_Mtick('AG', '2022.01.01', '2022.01.18')
    #f=GetData.Future_hist_Mcandle('AG','2022.01.01', '2022.03.07',types='main',ktype='10')
    #op=TL.TL_option_M('510050','2022.01.01','2022.08.06','30','c','m1',0)
    #OP=TL.TL_option('',['10003528'],'2021.12.15','2021.12.28','600')
    #df2 = TL.TL_option_M('510300', '2022.12.15','2022.12.28')
    #df3 = TL.TL_option_pro('510500', '2022.09.20', '2022.12.31', '600', 'c', 'm1')
    #ddf=TL.TL_stock('510050', '2016.12.20', '2021.12.28', cycle='900')
    YYTICK=TL.yy_tick('IC','2023.06.21','2023.06.21','10:01:02.000','10:02:02.000')
    TL.close()
    #df=spread2('A','2022.01.01', '2022.01.18','30')
    # GetData.clearcache()
    # GetData.close()





