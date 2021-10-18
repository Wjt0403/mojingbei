import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import ipdb
from sklearn.preprocessing import StandardScaler





pd.set_option('display.max_columns', None)

def read_data():
    LogInfo = pd.read_csv('PPD_LogInfo_3_1_Training_Set.csv')
    Master = pd.read_csv('PPD_Training_Master_GBK_3_1_Training_Set.csv',encoding = 'utf-8')
    UserUpdate = pd.read_csv('PPD_Userupdate_Info_3_1_Training_Set.csv')
    return Master,LogInfo,UserUpdate

def delete_useless_columns(Master):#删除缺失率过大的列和行
    col=Master.columns
    drop_list_missing=[]
    for i in col:
        if Master[i].count()/Master.shape[0]<0.2:
            drop_list_missing.append(i)
    drop_list_const=[]
    for i in col:
        if i != 'target':
            if Master[i].value_counts().iloc[0]/Master[i].count()>0.95:
                drop_list_const.append(i)
    drop_list=list(set(drop_list_missing+drop_list_const))
    Master=Master.drop(drop_list,axis=1)
    Master['usermissing']=Master.isnull().sum(axis=1) # 衍生
    Master=Master[~Master['Idx'].isin(Master[Master['usermissing']>80]['Idx'])]
    return Master

def proc_Userinfo(Master):
    Master['UserInfo_2']=Master['UserInfo_2'].replace({np.nan:'不详'})
    Master['UserInfo_4']=Master['UserInfo_4'].replace({np.nan:'不详'})
    #将“市”截取掉
    Master['UserInfo_2']=Master['UserInfo_2'].map(lambda x: x[:-1] if x.find('市')>0 else x)
    Master['UserInfo_4']=Master['UserInfo_4'].map(lambda x: x[:-1] if x.find('市')>0 else x)
    Master['UserInfo_8']=Master['UserInfo_8'].map(lambda x: x[:-1] if x.find('市')>0 else x)
    Master['UserInfo_20']=Master['UserInfo_20'].map(lambda x: x[:-1] if x.find('市')>0 else x)
    
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-1] if x.find('省')>0 else x)
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-1] if x.find('市')>0 else x)
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-5] if x.find('壮族自治区')>0 else x)
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-5] if x.find('回族自治区')>0 else x)
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-6] if x.find('维吾尔自治区')>0 else x)
    Master['UserInfo_19']=Master['UserInfo_19'].map(lambda x: x[:-3] if x.find('自治区')>0 else x)

    #逾期率前五省份UserInfo_7
    a=pd.DataFrame()
    a['total']=Master.groupby('UserInfo_7')['target'].count()
    a['bad']=Master.groupby('UserInfo_7')['target'].sum()
    a['bad_rate']=a['bad']/a['total']
    a=a.sort_values('bad_rate',ascending=False).iloc[0:5,-1]
    #print('逾期率前五省份UserInfo_7')
    #print(a)

    #逾期率前五省份UserInfo_19
    b=pd.DataFrame()
    b['total']=Master.groupby('UserInfo_19')['target'].count()
    b['bad']=Master.groupby('UserInfo_19')['target'].sum()
    b['bad_rate']=b['bad']/b['total']
    b=b.sort_values('bad_rate',ascending=False).iloc[0:5,-1]
    #print('逾期率前五省份UserInfo_19')
    #print(b)

    #逾期率前五省份二值化
    Master['UserInfo_7_shandong']=Master['UserInfo_7'].map(lambda x: 1 if x=='山东' else 0)
    Master['UserInfo_7_tianjin']=Master['UserInfo_7'].map(lambda x: 1 if x=='天津' else 0)
    Master['UserInfo_7_sichaun']=Master['UserInfo_7'].map(lambda x: 1 if x=='四川' else 0)
    Master['UserInfo_7_hunan']=Master['UserInfo_7'].map(lambda x: 1 if x=='湖南' else 0)
    Master['UserInfo_7_hainan']=Master['UserInfo_7'].map(lambda x: 1 if x=='海南' else 0)
    #---------------------------------------------------------------------------------
    Master['UserInfo_19_xizang']=Master['UserInfo_19'].map(lambda x: 1 if x=='天津' else 0)
    Master['UserInfo_19_shandong']=Master['UserInfo_19'].map(lambda x: 1 if x=='山东' else 0)
    Master['UserInfo_19_jilin']=Master['UserInfo_19'].map(lambda x: 1 if x=='吉林' else 0)
    Master['UserInfo_19_heilongjiang']=Master['UserInfo_19'].map(lambda x: 1 if x=='黑龙江' else 0)
    Master['UserInfo_19_liaoning']=Master['UserInfo_19'].map(lambda x: 1 if x=='辽宁' else 0)

    #定义一线二线城市
    top_order_city = ['北京','上海','广州','深圳']
    first_order_city = ['成都','重庆','杭州','武汉','西安','天津','苏州','南京','郑州','长沙','东莞','沈阳','青岛','合肥','佛山']

    #城市分级化
    Master['UserInfo_2_top']=Master['UserInfo_2'].map(lambda x: 1 if x in top_order_city else 0)
    Master['UserInfo_2_first']=Master['UserInfo_2'].map(lambda x: 1 if x in first_order_city else 0)
    Master['UserInfo_2_other']=Master['UserInfo_2'].map(lambda x: 1 if (x not in top_order_city and x not in first_order_city) else 0)
    #---------------------------------------------------------------------------------
    Master['UserInfo_4_top']=Master['UserInfo_4'].map(lambda x: 1 if x in top_order_city else 0)
    Master['UserInfo_4_first']=Master['UserInfo_4'].map(lambda x: 1 if x in first_order_city else 0)
    Master['UserInfo_4_other']=Master['UserInfo_4'].map(lambda x: 1 if (x not in top_order_city and x not in first_order_city) else 0)
    #---------------------------------------------------------------------------------
    Master['UserInfo_8_top']=Master['UserInfo_8'].map(lambda x: 1 if x in top_order_city else 0)
    Master['UserInfo_8_first']=Master['UserInfo_8'].map(lambda x: 1 if x in first_order_city else 0)
    Master['UserInfo_8_other']=Master['UserInfo_8'].map(lambda x: 1 if (x not in top_order_city and x not in first_order_city) else 0)
    #---------------------------------------------------------------------------------
    Master['UserInfo_20_first']=Master['UserInfo_20'].map(lambda x: 1 if x in top_order_city else 0)
    Master['UserInfo_20_first']=Master['UserInfo_20'].map(lambda x: 1 if x in first_order_city else 0)
    Master['UserInfo_20_other']=Master['UserInfo_20'].map(lambda x: 1 if (x not in top_order_city and x not in first_order_city) else 0)

    #对同级别地区变量作衍生，看是否相同
    Master['different_2_4']=np.where((Master['UserInfo_2']==Master['UserInfo_4']),0,1)
    Master['different_2_8']=np.where((Master['UserInfo_2']==Master['UserInfo_8']),0,1)
    Master['different_2_20']=np.where((Master['UserInfo_2']==Master['UserInfo_20']),0,1)
    Master['different_4_8']=np.where((Master['UserInfo_4']==Master['UserInfo_8']),0,1)
    Master['different_4_20']=np.where((Master['UserInfo_4']==Master['UserInfo_20']),0,1)
    Master['different_8_20']=np.where((Master['UserInfo_8']==Master['UserInfo_20']),0,1)
    Master['different_7_19']=np.where((Master['UserInfo_7']==Master['UserInfo_19']),0,1)

    #户籍地与现在的两个地址都不同
    Master['different_20_2_4']=np.where((Master['UserInfo_2']==Master['UserInfo_20']) | (Master['UserInfo_4']==Master['UserInfo_20']),0,1)
    #运营商归属地与所有地址都不同
    Master['different_8_2_4_20']=np.where((Master['UserInfo_8']==Master['UserInfo_2']) | (Master['UserInfo_8']==Master['UserInfo_4']) | (Master['UserInfo_8']==Master['UserInfo_20']),0,1)

    Master = Master.drop('UserInfo_7',1)
    Master = Master.drop('UserInfo_19',1)
    Master = Master.drop('UserInfo_2',1)
    Master = Master.drop('UserInfo_4',1)
    Master = Master.drop('UserInfo_8',1)
    Master = Master.drop('UserInfo_20',1)

    #运营商
    Master['UserInfo_9']=Master['UserInfo_9'].map(lambda x:x.split(' ',1)[0])
    Master['UserInfo_9_yidong']=Master['UserInfo_9'].map(lambda x: 1 if x == '中国移动' else 0)
    Master['UserInfo_9_liantong']=Master['UserInfo_9'].map(lambda x: 1 if x == '中国联通' else 0)
    Master['UserInfo_9_dianxin']=Master['UserInfo_9'].map(lambda x: 1 if x == '中国电信' else 0)
    Master['UserInfo_9_unknown']=Master['UserInfo_9'].map(lambda x: 1 if x == '不详' else 0)
    Master = Master.drop('UserInfo_9',1)

    Master['UserInfo_22_D']=Master['UserInfo_22'].map(lambda x: 1 if x == 'D' else 0)
    Master['UserInfo_22_weihun']=Master['UserInfo_22'].map(lambda x: 1 if x == '未婚' else 0)
    Master['UserInfo_22_yihun']=Master['UserInfo_22'].map(lambda x: 1 if x == '已婚' else 0)
    Master['UserInfo_22_buxiang']=Master['UserInfo_22'].map(lambda x: 1 if x == '不详' else 0)
    Master['UserInfo_22_lihun']=Master['UserInfo_22'].map(lambda x: 1 if x == '离婚' else 0)
    Master['UserInfo_22_zaihun']=Master['UserInfo_22'].map(lambda x: 1 if x == '再婚' else 0)
    Master['UserInfo_22_chuhun']=Master['UserInfo_22'].map(lambda x: 1 if x == '初婚' else 0)
    Master = Master.drop('UserInfo_22',1)

    Master['UserInfo_23_D']=Master['UserInfo_23'].map(lambda x: 1 if x == 'D' else 0)
    Master['UserInfo_23_G']=Master['UserInfo_23'].map(lambda x: 1 if x == 'G' else 0)
    Master['UserInfo_23_AB']=Master['UserInfo_23'].map(lambda x: 1 if x == 'AB' else 0)
    Master['UserInfo_23_O']=Master['UserInfo_23'].map(lambda x: 1 if x == 'O' else 0)
    Master['UserInfo_23_DXBK']=Master['UserInfo_23'].map(lambda x: 1 if x == '大学本科（简称“大学' else 0)
    Master = Master.drop('UserInfo_23',1)

    Master['UserInfo_24']=Master['UserInfo_24'].map(lambda x: 1 if x == 'D' else 0)
    Master = Master.drop('UserInfo_24',1)

    UserInfo_list=[i for i in Master.columns if i.split('_',1)[0]=='UserInfo']
    for i in UserInfo_list:
        Master[i]=Master[i].fillna(Master[i].median())
    return Master

def proc_WeblogInfo(Master):
    WeblogInfo_list=[i for i in Master.columns if i.split('_',1)[0]=='WeblogInfo']
    
    for i in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
        Master[i]=Master[i].fillna(Master[i].mode()[0])
    
    Master['WeblogInfo_19_I']=Master['WeblogInfo_19'].map(lambda x: 1 if x == 'I' else 0)
    Master['WeblogInfo_19_D']=Master['WeblogInfo_19'].map(lambda x: 1 if x == 'D' else 0)
    Master['WeblogInfo_19_F']=Master['WeblogInfo_19'].map(lambda x: 1 if x == 'F' else 0)
    Master = Master.drop('WeblogInfo_19',1)

    Master['WeblogInfo_20_I5']=Master['WeblogInfo_20'].map(lambda x: 1 if x == 'I5' else 0)
    Master['WeblogInfo_20_C21']=Master['WeblogInfo_20'].map(lambda x: 1 if x == 'C21' else 0)
    Master['WeblogInfo_20_I4']=Master['WeblogInfo_20'].map(lambda x: 1 if x == 'I4' else 0)
    Master['WeblogInfo_20_U']=Master['WeblogInfo_20'].map(lambda x: 1 if x == 'U' else 0)
    Master['WeblogInfo_20_I3']=Master['WeblogInfo_20'].map(lambda x: 1 if x == 'I3' else 0)
    Master = Master.drop('WeblogInfo_20',1)

    Master['WeblogInfo_21_A']=Master['WeblogInfo_21'].map(lambda x: 1 if x == 'A' else 0)
    Master['WeblogInfo_21_B']=Master['WeblogInfo_21'].map(lambda x: 1 if x == 'B' else 0)
    Master['WeblogInfo_21_C']=Master['WeblogInfo_21'].map(lambda x: 1 if x == 'C' else 0)
    Master['WeblogInfo_21_D']=Master['WeblogInfo_21'].map(lambda x: 1 if x == 'D' else 0)
    Master = Master.drop('WeblogInfo_21',1)

    for i in WeblogInfo_list:
        if i not in ['WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']:
            Master[i]=Master[i].fillna(Master[i].median())
    return Master

def proc_Education(Master):
    Education_list=[i for i in Master.columns if i.split('_',1)[0]=='Education']

    Master['Education_Info2_E']=Master['Education_Info2'].map(lambda x: 1 if x == 'E' else 0)
    Master['Education_Info2_A']=Master['Education_Info2'].map(lambda x: 1 if x == 'A' else 0)
    Master['Education_Info2_AM']=Master['Education_Info2'].map(lambda x: 1 if x == 'AM' else 0)
    Master['Education_Info2_AQ']=Master['Education_Info2'].map(lambda x: 1 if x == 'AQ' else 0)
    Master['Education_Info2_AN']=Master['Education_Info2'].map(lambda x: 1 if x == 'AN' else 0)
    Master['Education_Info2_U']=Master['Education_Info2'].map(lambda x: 1 if x == 'U' else 0)
    Master['Education_Info2_B']=Master['Education_Info2'].map(lambda x: 1 if x == 'B' else 0)
    Master = Master.drop('Education_Info2',1)

    Master['Education_Info3_E']=Master['Education_Info3'].map(lambda x: 1 if x == 'E' else 0)
    Master['Education_Info3_biye']=Master['Education_Info3'].map(lambda x: 1 if x == '毕业' else 0)
    Master['Education_Info3_jieye']=Master['Education_Info3'].map(lambda x: 1 if x == '结业' else 0)
    Master = Master.drop('Education_Info3',1)

    Master['Education_Info4_E']=Master['Education_Info4'].map(lambda x: 1 if x == 'E' else 0)
    Master['Education_Info4_T']=Master['Education_Info4'].map(lambda x: 1 if x == 'T' else 0)
    Master['Education_Info4_F']=Master['Education_Info4'].map(lambda x: 1 if x == 'F' else 0)
    Master['Education_Info4_AR']=Master['Education_Info4'].map(lambda x: 1 if x == 'AR' else 0)
    Master['Education_Info4_V']=Master['Education_Info4'].map(lambda x: 1 if x == 'V' else 0)
    Master['Education_Info4_AE']=Master['Education_Info4'].map(lambda x: 1 if x == 'AE' else 0)
    Master = Master.drop('Education_Info4',1)
    return Master

def proc_Socialnetwork(Master):
    SocialNetwork_list=[i for i in Master.columns if i.split('_',1)[0]=='SocialNetwork']
    return Master

def proc_Listinginfo(Master):
    Master['ListingInfo']=pd.to_datetime(Master['ListingInfo'])
    Master['Month']=Master['ListingInfo'].map(lambda x:x.month)
    Master['Weekday']=Master['ListingInfo'].map(lambda x:x.weekday())+1
    lbl = preprocessing.LabelEncoder()
    Master['ListingInfo'] = lbl.fit_transform(list(Master['ListingInfo'].values))
    return Master

def proc_Thirdlist(Master):
    third_list=[]
    for i in Master.columns:
        if i.find('Period')>0:
            third_list.append(i)
    data_third=Master[third_list]

    '''
    # 第一种
    Period_list_1=[]
    for j in range(1,7):
        temporary_list=[]
        for i in third_list:
            if i.split('_',3)[2]=='Period{}'.format(j):
            #if i.find('Period'+str(j))>0:
                temporary_list.append(i)
        Period_list_1.append(temporary_list)

    # 第二种
    Period_list_2=[]
    for j in range(1,18):
        temporary_list=[]
        for i in third_list:
            if i.split('_',3)[3]==str(j):
            #if i.find('Period'+str(j))>0:
                temporary_list.append(i)
        Period_list_2.append(temporary_list)
    
    start=time.time()
    data_third_1=pd.DataFrame()
    for i in range(len(Period_list_1)):
        third_max=[]
        third_min=[]
        third_avg=[]
        for j in range(data_third.shape[0]):
            a=data_third[Period_list_1[i]].iloc[j]
            third_max.append(np.max(a))
            third_min.append(np.min(a))
            third_avg.append(np.average(a))
        data_third_1['third_max_1'+str(i)]=third_max
        data_third_1['third_min_1'+str(i)]=third_min
        data_third_1['third_avg_1'+str(i)]=third_avg
    end=time.time()
    print('第一类衍生耗时{}秒'.format(round(end-start),0))

    start=time.time()
    data_third_2=pd.DataFrame()
    for i in range(len(Period_list_2)):
        third_max=[]
        third_min=[]
        third_avg=[]
        for j in range(data_third.shape[0]):
            a=data_third[Period_list_2[i]].iloc[j]
            third_max.append(np.max(a))
            third_min.append(np.min(a))
            third_avg.append(np.average(a))
        data_third_2['third_max_2'+str(i)]=third_max
        data_third_2['third_min_2'+str(i)]=third_min
        data_third_2['third_avg_2'+str(i)]=third_avg
    end=time.time()
    print('第二类衍生耗时{}秒'.format(round(end-start),0))
    '''
    start=time.time()
    data_third_3=pd.DataFrame()
    k=0
    new_col=['new_col_{}'.format(i) for i in range(0,8000)]
    for i in range(len(third_list)-1):
        for j in range(i+1,len(third_list)):
            data_third_3[new_col[k]]=data_third[third_list[i]]/data_third[third_list[j]]
            k=k+1
    end=time.time()
    print('除法暴力衍生耗时{}秒'.format(round(end-start),0))
    '''
    start=time.time()
    data_third_4=pd.DataFrame()
    k=0
    new_col=['new_col_{}'.format(i) for i in range(0,8000)]
    for i in range(len(third_list)-1):
        for j in range(i+1,len(third_list)):
            data_third_4[new_col[k]]=np.log(data_third[third_list[i]]*data_third[third_list[j]])
            k=k+1
    end=time.time()
    print('log乘法暴力衍生耗时{}秒'.format(round(end-start),0))
    '''
    #full_third_party = data_third_1.join(data_third_2)
    #full_third_party = full_third_party.join(data_third_3)
    full_third_party = data_third_3
    full_third_party = remove_infnan(full_third_party)
    return full_third_party


def remove_infnan(df):
    na_list = []
    for i in df.columns:
        if df[i].isna().sum() > 0:
            na_list.append(i)
    inf_list = []
    for i in df.columns:
        if np.isinf(df[i]).sum() > 0:
            inf_list.append(i)
    for i in na_list:
        df[i]=df[i].fillna(df[i].mode()[0])
    for i in inf_list:
        df[i]=df[i].replace(np.inf,0)
    return df

def reprocess_third(Master,full_third_party):
    x=full_third_party
    y=Master['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    xgb_classifier = xgb.XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=1,seed=27)
    xgb_classifier.fit(x_train, y_train)
    feature_name_list=full_third_party.columns
    feature_importance_list=list(xgb_classifier.feature_importances_)
    feature_data=pd.DataFrame({'feature_name':feature_name_list,'feature_importance':feature_importance_list})
    feature_data=feature_data.sort_values(by='feature_importance',ascending=False).reset_index(drop=True)
    #feature_zero=list(feature_data[feature_data['feature_importance']<feature_data['feature_importance'].mean()]['feature_name'])
    feature_useful = list(feature_data.iloc[:200,:]['feature_name'])
    full_third_party=full_third_party[feature_useful]
    return full_third_party


def preprocess_Master(Master):
    Master = delete_useless_columns(Master)
    Master = proc_Userinfo(Master)
    Master = proc_WeblogInfo(Master)
    Master = proc_Education(Master)
    Master = proc_Socialnetwork(Master)
    Master = proc_Listinginfo(Master)
    return Master

def merge_Master(Master,Third):
    Third['Idx'] = Master['Idx']
    Master_full = pd.merge(Master,Third,on='Idx',how='left')
    return Master_full

def preprocess_LogInfo(LogInfo):
    LogCnt=LogInfo.groupby('Idx',as_index=False)['Listinginfo1'].count().rename(columns={'Listinginfo1':'LogCnt'})
    LogInfo['Listinginfo1']=pd.to_datetime(LogInfo['Listinginfo1'])
    LogInfo['LogInfo3']=pd.to_datetime(LogInfo['LogInfo3'])
    LogTimeSpan=LogInfo.groupby('Idx',as_index=False).agg({'Listinginfo1':np.max,'LogInfo3':np.max})
    LogTimeSpan['LogTimespan']=LogTimeSpan['Listinginfo1']-LogTimeSpan['LogInfo3']
    LogTimeSpan['LogTimespan']=LogTimeSpan['LogTimespan'].map(lambda x:x.days)
    LogTimeSpan=LogTimeSpan[['Idx','LogTimespan']]
    LogTimeSpanAverage=LogInfo.sort_values(['Idx','LogInfo3'])
    LogTimeSpanAverage['LogInfo4']=LogTimeSpanAverage.groupby('Idx')['LogInfo3'].apply(lambda x:x.shift(1))
    LogTimeSpanAverage['LogTimeSpanAverage']=LogTimeSpanAverage['LogInfo3']-LogTimeSpanAverage['LogInfo4']
    LogTimeSpanAverage['LogTimeSpanAverage']=LogTimeSpanAverage['LogTimeSpanAverage'].map(lambda x:x.days)
    LogTimeSpanAverage['LogTimeSpanAverage']=LogTimeSpanAverage['LogTimeSpanAverage'].fillna(0)
    LogTimeSpanAverage=LogTimeSpanAverage.groupby('Idx',as_index=False)['LogTimeSpanAverage'].mean()
    MaxLogInfo1Cnt=LogInfo.groupby(['Idx','LogInfo1'],as_index=False)['LogInfo2'].count().rename(columns={'LogInfo1':'MaxLogInfo1','LogInfo2':'cnt'})
    MaxLogInfo1Cnt=MaxLogInfo1Cnt.sort_values('cnt',ascending=False).groupby(['Idx'],as_index=False).head(1)
    MaxLogInfo1Cnt=MaxLogInfo1Cnt.drop(['cnt'],axis=1)
    MaxLogInfo2Cnt=LogInfo.groupby(['Idx','LogInfo2'],as_index=False)['LogInfo1'].count().rename(columns={'LogInfo2':'MaxLogInfo2','LogInfo1':'cnt'})
    MaxLogInfo2Cnt=MaxLogInfo2Cnt.sort_values('cnt',ascending=False).groupby(['Idx'],as_index=False).head(1)
    MaxLogInfo2Cnt=MaxLogInfo2Cnt.drop(['cnt'],axis=1)
    MinLogInfo1Cnt=LogInfo.groupby(['Idx','LogInfo1'],as_index=False)['LogInfo2'].count().rename(columns={'LogInfo1':'MinLogInfo1','LogInfo2':'cnt'})
    MinLogInfo1Cnt=MinLogInfo1Cnt.sort_values('cnt').groupby(['Idx'],as_index=False).head(1)
    MinLogInfo1Cnt=MinLogInfo1Cnt.drop(['cnt'],axis=1)
    MinLogInfo2Cnt=LogInfo.groupby(['Idx','LogInfo2'],as_index=False)['LogInfo1'].count().rename(columns={'LogInfo2':'MinLogInfo2','LogInfo1':'cnt'})
    MinLogInfo2Cnt=MinLogInfo2Cnt.sort_values('cnt').groupby(['Idx'],as_index=False).head(1)
    MinLogInfo2Cnt=MinLogInfo2Cnt.drop(['cnt'],axis=1)
    LogInfo_new=pd.merge(LogCnt,LogTimeSpan,on='Idx',how='left')
    LogInfo_new=pd.merge(LogInfo_new,LogTimeSpanAverage,on='Idx',how='left')
    LogInfo_new=pd.merge(LogInfo_new,MaxLogInfo1Cnt,on='Idx',how='left')
    LogInfo_new=pd.merge(LogInfo_new,MaxLogInfo2Cnt,on='Idx',how='left')
    LogInfo_new=pd.merge(LogInfo_new,MinLogInfo1Cnt,on='Idx',how='left')
    LogInfo_new=pd.merge(LogInfo_new,MinLogInfo2Cnt,on='Idx',how='left')
    return LogInfo_new

def preprocess_UserUpdate(Userupdate):
    Userupdate['UserupdateInfo1']=Userupdate['UserupdateInfo1'].map(lambda x:x.lower())
    UpdateCnt=Userupdate.groupby(['Idx'],as_index=False)['UserupdateInfo1'].count().rename(columns={'UserupdateInfo1':'UpdateCnt'})
    Userupdate['ListingInfo1']=pd.to_datetime(Userupdate['ListingInfo1'])
    Userupdate['UserupdateInfo2']=pd.to_datetime(Userupdate['UserupdateInfo2'])
    UpdateTimeSpanAverage=Userupdate.sort_values(['Idx','UserupdateInfo2'])#默认升序
    UpdateTimeSpanAverage['UserupdateInfo3']=UpdateTimeSpanAverage.groupby('Idx')['UserupdateInfo2'].apply(lambda x:x.shift(1))
    UpdateTimeSpanAverage['UpdateTimeSpanAverage']=UpdateTimeSpanAverage['UserupdateInfo2']-UpdateTimeSpanAverage['UserupdateInfo3']
    UpdateTimeSpanAverage['UpdateTimeSpanAverage']=UpdateTimeSpanAverage['UpdateTimeSpanAverage'].map(lambda x:x.days)
    UpdateTimeSpanAverage['UpdateTimeSpanAverage']=UpdateTimeSpanAverage['UpdateTimeSpanAverage'].fillna(0)
    UpdateTimeSpanAverage=UpdateTimeSpanAverage.groupby('Idx',as_index=False)['UpdateTimeSpanAverage'].mean()
    UpdateTimeSpanAverage=UpdateTimeSpanAverage[['Idx','UpdateTimeSpanAverage']]
    UpdateTimeSpan=Userupdate.groupby('Idx',as_index=False).agg({'ListingInfo1':np.max,'UserupdateInfo2':np.max})
    UpdateTimeSpan['UpdateTimeSpan']=UpdateTimeSpan['ListingInfo1']-UpdateTimeSpan['UserupdateInfo2']
    UpdateTimeSpan['UpdateTimeSpan']=UpdateTimeSpan['UpdateTimeSpan'].map(lambda x:x.days)
    UpdateTimeSpan=UpdateTimeSpan[['Idx','UpdateTimeSpan']]
    MaxUpdateCnt=Userupdate.groupby(['Idx','UserupdateInfo1'],as_index=False)['UserupdateInfo2'].count().rename(columns={'UserupdateInfo1':'MaxUserupdateInfo1','UserupdateInfo2':'cnt'})
    MaxUpdateCnt=MaxUpdateCnt.sort_values('cnt',ascending=False).groupby(['Idx'],as_index=False).head(1)
    MaxUpdateCnt=MaxUpdateCnt.drop(['cnt'],axis=1)
    MinUpdateCnt=Userupdate.groupby(['Idx','UserupdateInfo1'],as_index=False)['UserupdateInfo2'].count().rename(columns={'UserupdateInfo1':'MinUserupdateInfo1','UserupdateInfo2':'cnt'})
    MinUpdateCnt=MinUpdateCnt.sort_values('cnt').groupby(['Idx'],as_index=False).head(1)
    MinUpdateCnt=MinUpdateCnt.drop(['cnt'],axis=1)
    UpdateInfo=pd.merge(UpdateCnt,MaxUpdateCnt,on='Idx',how='left')
    UpdateInfo=pd.merge(UpdateInfo,MinUpdateCnt,on='Idx',how='left')
    UpdateInfo=pd.merge(UpdateInfo,UpdateTimeSpan,on='Idx',how='left')
    UpdateInfo=pd.merge(UpdateInfo,UpdateTimeSpanAverage,on='Idx',how='left')
    UpdateInfo_category_list=[]
    for i in UpdateInfo.columns:
        if UpdateInfo[i].dtype=='object':
            UpdateInfo_category_list.append(i)
    for i in UpdateInfo_category_list:
        lbl = preprocessing.LabelEncoder()
        UpdateInfo[i] = lbl.fit_transform(list(UpdateInfo[i].values))
    return UpdateInfo


def merge_data(Master,LogInfo,UserUpdate):
    full_data=pd.merge(Master,LogInfo,on='Idx',how='left')
    full_data=pd.merge(full_data,UserUpdate,on='Idx',how='left')
    full_data.drop(['Idx','ListingInfo'],axis=1,inplace=True)
    for i in full_data.columns:
        if full_data[i].isna().sum()>0:
            full_data[i]=full_data[i].fillna(0)
    return full_data



if __name__ == "__main__":
    Master = pd.read_csv('./data/Master.csv').iloc[:,1:]
    #LogInfo = pd.read_csv('./data/LogInfo_new.csv')
    #UserUpdate = pd.read_csv('./data/UserUpdate_new.csv')
    #full_data = merge_data(Master,LogInfo,UserUpdate)
    #full_data.to_csv('full_data1.csv',index=False)
    #third_data = proc_Thirdlist(Master)
    #third_data.to_csv('./data/third4.csv',index=False)
    third_data = pd.read_csv('./data/third4.csv')
    full_third = reprocess_third(Master,third_data)
    full_third.to_csv('./data/third_cut.csv',index=False)
    







