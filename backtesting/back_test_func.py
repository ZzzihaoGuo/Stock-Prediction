import warnings
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings('ignore')
color_names = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
# coding: utf-8
color_list_all ='blue red green black pink purple orange grey yellow deepskyblue lightgreen brown hotpink lavender whitesmoke'.split()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
plt.rcParams['axes.unicode_minus'] = False   #解决中文乱码

def plot_n_series_line(series, s_name=None, save_indi=0, save_name='line.png', color_map=None, figsize=(21,9), split_x=1, plot_constraints=None, y_values_pairs_names=None):
    '''
    【输入】n series线（有index）,名字list for title（可没有）
    【输出】plot
    备注：index时间要做成str格式; 要保障它们index是一样的
    '''
    fig, ax1 = plt.subplots()
    title_name = ''
    color_list_all = 'red green blue orange pink black'.split()
    # series = series[:3]
    # s_name = s_name[:3]
    # color_list_all = color_list_all[:3]

    if (s_name is None) or (len(series) > 15):  # 没有定义名字或序列太多都不显示title 和
        for i in series:
            plt.plot(i)
    else:  # 不是很多条且有名字
        color_list = color_list_all[:len(series)]
        for i, j, c in zip(series, s_name, color_list):
            title_name += j + ','
            if color_map:  # 用 颜色字典
                c = color_map[j]
            if j == 'all':
                ax2 = ax1.twinx()
                ax2.plot(i, label=j, color=c)
            else:
                ax1.plot(i, label=j, color=c)
        title_name = title_name[:-1]
        plt.title(title_name)
        plt.legend(loc=1)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    ax1.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(20))


    if save_indi != 0:
        plt.savefig(save_name)

    if plot_constraints is not None:
        color_list_all = color_list_all[::-1]
        for i, constraint in enumerate(plot_constraints):
            color = color_list_all[i % len(color_list_all)]
            plt.hlines(constraint, xmin=0, xmax=len(series[0]), colors=color, linestyles='dashed')
            if y_values_pairs_names is not None and i < len(y_values_pairs_names):
                # Add labels for each y_values pair
                for y_value in constraint:
                    plt.text(len(series[0]), y_value, f'{y_values_pairs_names[i]}: {y_value}', verticalalignment='bottom', color=color)

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(split_x))
    plt.show()

def get_time_str(x):  # 日期time格式to str 【【给apply用】】
    '''
    日期time格式to str
    '''
    return(str(x).split()[0])

import pandas as pd

import itertools
def get_trading_group_simple(data_trading_act,date_col='date',return_col='return',method='mean'):
    '''
    输入 日期col，所有股票，收益；
    输出 每日【平均】收益（和累计收益），每日机会；日期设为index
    mean sum
    '''
    data_all_consider_group = data_trading_act.groupby([date_col]).agg({return_col:method,date_col:'count'})
    data_all_consider_group.columns = ['return','count']
    data_all_consider_group['cum_return'] = np.cumsum(data_all_consider_group['return'])
    data_all_consider_group.index = [str(i) for i in list(data_all_consider_group.index)]
    return data_all_consider_group
#n 全量（选择股票之后）每日收益（可累加 可平均）
def get_cum_result(data_consider_after_n_max,date_all0,date_col='date',return_col='return',method='sum'):  # 没交易的收益为0
    '''
    输入 选择好股票的 data_consider_after_n_max， 所有股票的日期（没有交易的后面收益会被设为0）
    return_col（收益的col） method（sum：和（之前已经除好了），mean：均值（之前没有除））
    返回每日收益和累计收益
    '''
    data_consider_group = data_consider_after_n_max.groupby([date_col]).agg({return_col:method,date_col:'count'})
    data_consider_group.columns = ['return','count']
    dd = pd.DataFrame({'date':date_all0})
    data_consider_group2 = data_consider_group.reset_index()[[date_col,'return','count']]
    data_consider_group2.columns = ['date','return','count']
    data_consider_group2['date'] = data_consider_group2['date'].astype(int)
    dd = pd.merge(dd,data_consider_group2,on='date',how='left')
    dd['return'] =dd['return'].fillna(0)
    dd['count'] =dd['count'].fillna(0)
    dd['cum_return'] = np.cumsum(dd['return'])
    dd = dd.set_index('date')
    dd.index = [str(i) for i in list(dd.index)]
    return dd
#h2 回测效果相关
# 【return base的效果】
#n 最大回撤
def get_mdd_info(accu_value):
    '''
    日期是index
    输入 累计价值 （dataframe里面的series）
    '''
    accu_value.index = [int(i) for i in accu_value.index]  #得是数字
    accu_value = accu_value.astype(float)
    #添加首日 0
    l_index = [list(accu_value.index)[0]]
    l_index.extend(list(accu_value.index))
    l_list = [0]
    l_list.extend(list(accu_value))
    accu_value = pd.Series(l_list, index=l_index)

    try:
        value_last = accu_value[-1]
    except:
        value_last = list(accu_value)[-1]
    #Dt 【回撤】
    running_max_value = accu_value.cummax()
    D= (running_max_value-accu_value)/(1+running_max_value)   #(running max - 减去现在)/running_max
    D = D.astype(float)  #如果type不是这个 ，idxmax会报错
    #d=D/(D+value)
    #最大回撤
    #MDD=D.max()   #累计值算出来的， 改成率
    #mdd=d.max()
    ## 高值 低值日期输出
    high_date = get_time_str(accu_value[accu_value.index <= D.idxmax()].idxmax())
    low_date = get_time_str(accu_value[accu_value.index == D.idxmax()].idxmax())
    # high_value = str_percent(accu_value[accu_value.index <= D.idxmax()].max(),2)
    # low_value = str_percent(accu_value[accu_value.index == D.idxmax()].max(),2)
    high_value = accu_value[accu_value.index <= D.idxmax()].max()
    low_value = accu_value[accu_value.index == D.idxmax()].max()
    MDD =(high_value-low_value)/(1+high_value)   #最大回撤率
    #重返高点日
    accu_value_temp = accu_value[accu_value.index>accu_value[accu_value.index <= D.idxmax()].idxmax()]
    try:
        back_high_date = list(accu_value_temp[accu_value_temp>=high_value].index)[0]
    except:
        back_high_date ='na'
    return value_last,MDD,high_date,low_date,back_high_date,high_value,low_value
#value_last,MDD,high_date,low_date,back_high_date,high_value,low_value = get_mdd_info(accu_value)
#n 日胜率，日开仓等情况
def get_daily_return_stat(data_consider_group, return_col='return',count_col='count'):
    '''
    股票每日收益汇总，开仓股票数量
    输出 总天数 上涨天数 win rate ，win avg
    '''
    trading_days = len(data_consider_group)
    up_days = len(data_consider_group[data_consider_group[return_col] > 0])
    if trading_days == 0:
        day_win_rate = 0
    else:
        day_win_rate = up_days / trading_days
    win_s = [i for i in data_consider_group[return_col] if i > 0]
    if len(win_s) == 0:
        win_avg = 0
    else:
        win_avg = sum(win_s) / len(win_s)
    lose_s = [i for i in data_consider_group[return_col] if i < 0]
    if len(lose_s) == 0:
        lose_avg = 0
    else:
        lose_avg = sum(lose_s) / len(lose_s)

    if len(data_consider_group) == 0:
        stock_day_num_avg = 0
    else:
        stock_day_num_avg = data_consider_group[count_col].mean() #sum(data_consider_group[count_col]) / len(data_consider_group)
    return trading_days, up_days, day_win_rate, win_avg, lose_avg, stock_day_num_avg
# trading_days,up_days,day_win_rate,win_avg,lose_avg,stock_day_num_avg= get_daily_return_stat(data_consider_group,return_col='return')

#n 股票胜率
def get_stock_return_stat(data_consider_after_n_max, return_col='y'):
    '''
    股票每次收益
    输出 总次数  win rate ，win avg
    '''
    stock_count = len(data_consider_after_n_max)
    s_win_s = [i for i in data_consider_after_n_max[return_col] if i > 0]
    stock_win_rate = len(s_win_s) / stock_count
    stock_avg = sum(data_consider_after_n_max[return_col]) / len(data_consider_after_n_max)
    if len(s_win_s) == 0:
        s_win_avg = 0
    else:
        s_win_avg = sum(s_win_s) / len(s_win_s)
    s_lose_s = [i for i in data_consider_after_n_max[return_col] if i < 0]
    if len(s_lose_s) == 0:
        s_lose_avg = 0
    else:
        s_lose_avg = sum(s_lose_s) / len(s_lose_s)
    return stock_count, stock_win_rate, stock_avg, s_win_avg, s_lose_avg
#stock_count, stock_win_rate, stock_avg, s_win_avg, s_lose_avg = get_stock_return_stat(data_consider_after_n_max, return_col='y')

# stock_count,stock_win_rate,stock_avg,s_win_avg,s_lose_avg = get_stock_return_stat(data_consider_after_n_max,return_col='y')

#n 回测情况汇总
def back_test_day_returnbase(trading_data,return_col='strategy_return',return_cum_method='multi',return_cum_col=None,trade_indi_col='indi',hold_indi = 'return base',hold_indi_col='indi',result='for list',date_start=0,plot_indi=0,split_x=10):
    '''
    return base的效果  日期是index【利用收益率做的验证】
    备注：
    strategy_return_cum不用传了
    【输入】简单版date（int index）	return	cum_return
    dataframe(date(index),indi,hold_indi,strategy_return,strategy_return_cum,{price,trading_cost}),
          ,hold_indi = 'return base',hold_indi_col='indi',result='for list'
    【输出】回测开始日期，结束日期，年化夏普,日均收益,累积收益,value_with_cost,最大回撤，高值日期，低值日期，高值，
          低值，测试天数，持有天数，交易次数，买次数，卖次数，交易成本,胜率,win_avg,lose_avg,win_lost_hist
           plot 日均&累计收益
    '''
    bt_start = get_time_str(trading_data.index[date_start])
    bt_end  = get_time_str(trading_data.index[-1])
    N = 244 #【年化夏普】
    ##实际交易天数
    actual_trading_returns = trading_data[trading_data[return_col]!=0][return_col]
    #annua_sharpe = np.sqrt(N) * actual_trading_returns.mean() / actual_trading_returns.std()

    annua_sharpe = np.sqrt(N) * trading_data[return_col].mean() / trading_data[return_col].std()

    #累计价值
    #accu_value = trading_data[return_cum_col]
    if return_cum_method=='multi':
        accu_value = (1 + trading_data[return_col]).cumprod() - 1
    else:
        accu_value = np.cumsum(trading_data[return_col]) 
    if return_cum_col:
        accu_value = trading_data[return_cum_col]
    trading_data.index = [str(i) for i in trading_data.index]
    if plot_indi==1:
        plot_n_series_line([trading_data[return_cum_col]],['strategy'],split_x=split_x)
   
    value_last, MDD, high_date, low_date, back_high_date, high_value, low_value = get_mdd_info(accu_value)



    #back-test time
    ##测试天数
    back_days= len(trading_data)-date_start
    ##持有天数
    if hold_indi =='return base':
        hold_days= len(actual_trading_returns)
    else:
        hold_days= sum(trading_data[hold_indi_col]>0)

    #胜率
    #winpct=round(sum(actual_trading_returns>0)/sum(actual_trading_returns!=0),3)  #不等于0
    winpct=round(sum(actual_trading_returns>0.00001)/sum(abs(actual_trading_returns)>0.00001),3)  #不等于0
    ##胜平均
    #win_avg = round(actual_trading_returns[actual_trading_returns>0].mean(),4)
    try:
        ##交易天数
        trading_days = sum(trading_data[trade_indi_col]!=0)
        win_avg = round(sum(actual_trading_returns[actual_trading_returns>0.00001])/trading_days,4)
    ##输平均
    #lose_avg = round(actual_trading_returns[actual_trading_returns<0].mean(),4)
        lose_avg = round(sum(actual_trading_returns[actual_trading_returns<-0.00001])/trading_days,4)
    except:
        trading_days = len(trading_data)
        win_avg = round(sum(actual_trading_returns[actual_trading_returns>0.00001])/trading_days,4)
    ##输平均
    #lose_avg = round(actual_trading_returns[actual_trading_returns<0].mean(),4)
        lose_avg = round(sum(actual_trading_returns[actual_trading_returns<-0.00001])/trading_days,4)
    if result=='for list':
        return(bt_start,bt_end,annua_sharpe,value_last,MDD,high_date,low_date,back_high_date,high_value,low_value,
                                back_days,hold_days,trading_days,winpct,win_avg,lose_avg)
    else:
        back_index= 'bt_start,bt_end,annua_sharpe,value_last,MDD,high_date,low_date,back_to_high_date,high_value,low_value,back_days,hold_days,trading_days,winpct,win_avg,lose_avg'.split(',')
        bt_start_list,bt_end_list,annua_sharpe_list,value_last_list,MDD_list,high_date_list,\
        low_date_list,high_value_list,low_value_list,\
        back_days_list,hold_days_list,trading_days_list,winpct_list,win_avg_list,lose_avg_list,back_high_date_list =[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for i,j in zip([bt_start_list,bt_end_list,annua_sharpe_list,value_last_list,
                                MDD_list,high_date_list,low_date_list,back_high_date_list,high_value_list,low_value_list,back_days_list,
                                hold_days_list,trading_days_list,winpct_list,win_avg_list,lose_avg_list],[bt_start,bt_end,annua_sharpe,value_last,MDD,high_date,low_date,back_high_date,high_value,low_value,
                                            back_days,hold_days,trading_days,winpct,win_avg,lose_avg]):
               i.append(j)
        back_result_group = pd.DataFrame([bt_start_list,bt_end_list,annua_sharpe_list,value_last_list,
                        MDD_list,high_date_list,low_date_list,back_high_date_list,high_value_list,low_value_list,back_days_list,
                        hold_days_list,trading_days_list,winpct_list,win_avg_list,lose_avg_list],index=back_index)#.transpose()
        #back_result_group.sort_values(by='annua_sharpe',ascending=False,inplace=True)
        back_result_group=back_result_group.drop(['hold_days','trading_days'],axis=0)
        return(back_result_group)

    
    
def data_process_v0(data):
    '''
    预处理
    '''
    data_all = data.copy()[['date','wind_code','time','price','y10price','Prediction','Probability', 'label']]
    data_all = data_all[data_all['time']<=144200000]
    data_all['profit_rate'] = data_all['y10price']/data_all['price']-1

    
    try:
        data_all['predict_prob'] =data_all['Probability'].apply(lambda x:float(x[2])) #[random.random() for _ in range(len(data_all))]
    except:
        data_all['predict_prob'] =data_all['Probability'].apply(lambda x:float(eval(x)[2])) #[random.random() for _ in range(len(data_all))]

    data_all = data_all.sort_values(by=['date','time','predict_prob'],ascending=[True,True,False]).reset_index(drop=True)
    data_all['time'] = data_all['time'].apply(lambda x:int(str(x)[:-5]))

    # 将 label 列中的字符串转换为整数
    def convert_label(x):
        if isinstance(x, str):
            match = re.search(r'\d+', x)
            return int(match.group()) if match else x
        return x

    data_all['label'] = data_all['label'].apply(convert_label)

    data_label = data_all[data_all['label']==2]
    data_consider = data_all[data_all['Prediction']==2]

    return data_all, data_consider, data_label

class Backtest_v0():
    def __init__(self,data_all,data_consider,data_label,time_col='time',stock_col='wind_code',profit_col='profit_rate',
                 sort_col='predict_prob',file_name='test'):
        self.data_all = data_all
        self.data_consider = data_consider
        self.data_label = data_label
        self.time_col = time_col
        self.stock_col=stock_col
        self.profit_col = profit_col
        self.sort_col = sort_col
        #
        self.time_list = list(set(data_all[time_col]))
        self.time_list.sort()
        self.file_name=file_name
        #
        self._get_data_consider_stat()
        
    def _get_data_consider_stat(self):
        data_up_avg_g = self.data_consider.groupby([self.time_col]).agg({self.stock_col:'count',self.profit_col:'mean'})
        data_up_avg_g[self.stock_col].hist()
        self.count_median = data_up_avg_g[self.stock_col].median()
        print('count median: ',self.count_median)

    def get_trading_data(self, data_consider, count_adj_portion = 1, n_max = None):
        '''
        获取交易的数据
        '''
        if not n_max:
            n_max = int(self.count_median*count_adj_portion)
        data_consider['sort_id'] = data_consider[self.sort_col].apply(lambda x:-x).groupby(data_consider[self.time_col]).rank(method='first')
        data_consider_after_n_max=data_consider[data_consider['sort_id']<=n_max]
        data_consider_after_n_max['y_adj'] = data_consider_after_n_max[self.profit_col]/n_max
        return data_consider_after_n_max
            
    def get_result(self,data_consider_after_n_max,data_label_after_n_max, split_x=10):
        '''
        取前n次 vs 全量平均
        '''
        zt_select_daily = get_cum_result(data_consider_after_n_max,self.time_list,date_col=self.time_col,return_col='y_adj',method='sum')#计算累计收益
        zt_all_daily = get_cum_result(self.data_all,self.time_list,date_col=self.time_col,return_col=self.profit_col,method='mean')#计算累计收益

        label_select_daily = get_cum_result(data_label_after_n_max,self.time_list,date_col=self.time_col,return_col='y_adj',method='sum')

        back_result = back_test_day_returnbase(zt_select_daily,return_col='return',return_cum_method='multi',return_cum_col=None,trade_indi_col='indi',hold_indi = 'return base',hold_indi_col='indi',result='for table',date_start=0,plot_indi=0,split_x=split_x)
        plot_n_series_line([zt_select_daily['cum_return'],zt_all_daily['cum_return'],zt_select_daily['cum_return']-zt_all_daily['cum_return'], label_select_daily['cum_return']],s_name=['select','all','exceed','best'],save_indi=1,save_name=self.file_name,split_x=30, plot_constraints=None, y_values_pairs_names=None)
        return back_result    
