#!/usr/bin/python3
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import dateutil
import pymysql
ptime=datetime.datetime.now()
import warnings
import confReader
import logCreate
warnings.filterwarnings("ignore")


def generatesignals(df,sym):
#        dsql="select max(timestamp) from nse.nsedailybhavhist where timestamp>(select max(timestamp) from nse.nsesignals where symbol="+"'"+sym+"'"+");"
#        NSEDBCursor.execute(dsql)
#        date=NSEDBCursor.fetchall()[0][0]
    if df.empty==True:
        return None
    else:
        df=df[df['symbol']==sym].sort_values('timestamp',ascending=True).reset_index(drop=True)
        tstamp=max(df['timestamp'])
        ptime=datetime.datetime.now()
        h=pd.Series(df['high_adj']).reset_index(drop=True)
        l=pd.Series(df['low_adj']).reset_index(drop=True)
        c=pd.Series(df['close_adj']).reset_index(drop=True)
        v=pd.Series(df['tottrdqty_adj']).reset_index(drop=True)
        o=pd.Series(df['open_adj']).reset_index(drop=True)
        t=pd.Series(df['timestamp']).reset_index(drop=True).to_list()
        pc=pd.Series(df['prevclose_adj']).reset_index(drop=True).to_list()
        closeadj=c.iloc[-1]
        prevcloadj=pc[-1]
        #tstamp=t[-1]
        def exponential_moving_average(period=12):
            ema = pd.Series(h.ewm(span=period, min_periods=period).mean(), name='EMA_' + str(period))
            ema=ema.fillna(0.)
            ema=ema.replace([-np.inf,np.inf],0.)
            ema=round(ema,3)
            return list(ema)
        def exp_ma_cat(c,s,param):
            s=pd.Series(s)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            s_grad=pd.Series(np.gradient(s.ewm(span=1, min_periods=1).mean())).tolist()
            if param=='classic':
                ema_l=['strong buy' if (i<1598.674) else 'sell' if (1598.674<=i and i<3198.348) else 'neutral' if (i>=3198.348 and i<4798.022) else 'buy' if (i>4798.022 and i<6397.696) else 'strong sell' for i in s]
            elif param=='divergence':
                ema_l=['strong buy' if i<1598.674 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=6397.696) else 'neutral' for i,j,k in zip(s,s_grad,c_grad)]
            elif param=='slope':
                ema_l=['strong buy' if (i>1 and i>j and k>=l) else 'strong sell' if (i<-1 and i<j and k<=l) else 'sell' if(i<0 and i<=j and k<=l) else 'buy' if (i>0 and i>=j and k>=l) else 'neutral' for i,j,k,l in zip(c_grad,s_grad,c,s)]
            return ema_l
          
        def momentum(n):
            mom = pd.Series(c.diff(n), name='momentum_' + str(n))
            mom=mom.fillna(0.)
            mom=mom.replace([-np.inf,np.inf],0.)
            mom=round(mom,3)
            return list(mom)
        def momen_cat(m,c,param):
            m=pd.Series(m)
            m_grad=pd.Series(np.gradient(m.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            cma=c.rolling(10,min_periods=10).mean()
            cma_grad=pd.Series(np.gradient(cma.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            m=m.tolist()
             
            if param=='divergence':
                m_l=['strong buy' if (k<=-300.0) else 'strong sell' if (k>=300.0) else 'buy' if (i>0 and j<0) else 'sell' if (i<0 and j>0) else 'neutral' for i,j in zip(m_grad,c_grad)]
            elif param=='slope':
                m_l=['strong buy' if (k<=-300.0) else 'strong sell' if (k>=300.0) else 'buy' if (i>j and k>=m) else 'sell' if(i<j and k<=m) else 'neutral' for i,j,k,m in zip(m_grad,cma_grad,m,cma)]
            elif param=='classic':
                m_l=['strong buy' if (k<=-300.0) else 'strong sell' if (k>=300.0) else 'buy' if (i>=-300.0 and i<-100.0) else 'sell' if(i>-100.0 and i<0) else 'buy' if (i>0 and i<=100.0) else 'sell' if(i>100.0 and i<300.0)  else 'neutral' for i in m]   
            return m_l

        def chaikin_oscillator():
            ad = ((2*c-h-l)/(h-l))*v
            chaikin_osc = pd.Series(round(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(),3), name='chaikin_osc')
            chaikin_osc=chaikin_osc.fillna(0.)
            chaikin_osc=chaikin_osc.replace([-np.inf,np.inf],0.)
            chaikin_osc=round(chaikin_osc,3)
            return list(chaikin_osc)
        def chaikosc_cat(co,c,param):
            co=pd.Series(co)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            co_grad=pd.Series(np.gradient(co.ewm(span=1, min_periods=1).mean())).tolist()
            co=co.tolist()
            if param=='classic':
                co_l=['strong buy' if (i<= -1799805.777) else 'buy' if (i<=-599858.578 and i> -1799805.777) else 'sell' if ((i<0 and i>-599858.578) or k<0) else 'buy' if ((i>0 and i< 600088.621) or k>0) else 'sell' if(i<1800035.82 and i>=600088.621) else 'strong sell' if (i>=1800035.82) else 'neutral' for i,k in zip(co,co_grad)]
            elif param=='divergence':
                co_l=['strong buy' if i<= -1799805.777 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=1800035.82) else 'neutral' for i,j,k in zip(co,co_grad,c_grad)]
            return co_l

        def chaikin_volatility(ema_periods=10,change_periods=10):
            ch_vol_hl = pd.Series(h-l)
            ch_vol_ema = ch_vol_hl.ewm(ignore_na=False,min_periods=0,com=ema_periods,adjust=True).mean()
            chaikin_volatility= [0.]*c.shape[0]    
            for i in range(c.shape[0]):
                if i>=change_periods:            
                    prev_value = ch_vol_ema[i-change_periods]
                    if prev_value == 0:
                        #this is to avoid division by zero below
                        prev_value = 0.0001
                    chaikin_volatility[i]=(round((ch_vol_ema[i]- prev_value)/prev_value,3))
            cv=pd.Series(chaikin_volatility,name='chaikin_vol_'+str(ema_periods))
            cv=cv.fillna(0.5)
            cv=cv.replace([-np.inf,np.inf],0.5)
            cv=round(cv,3)
            return list(cv)
          
        def average_true_range(period):
            i = 0
            tr_l = [0]
            for i in range(len(c.index) - 1):
                tr = max(h[i + 1], c[i]) - min(l[i + 1], c[i])
                tr_l.append(tr)
            tr_s = pd.Series(tr_l)
            atr = pd.Series(tr_s.ewm(span=period, min_periods=period).mean(), name='atr_' + str(period))
            atr=atr.fillna(0)
            atr=atr.replace([-np.inf,np.inf],0.)
            atr=round(atr,3)
            return list(atr)

        def bollinger_bands(n):
            tp=pd.Series((h+l+c)/3)
            ma = pd.Series(tp.rolling(n, min_periods=n).mean())
            msd = pd.Series(tp.rolling(n, min_periods=n).std())
            #b1 = round(4 * msd / ma,3)
            b_m=ma
            b_u=ma+2*msd
            b_d=ma-2*msd
            #B1 = pd.Series(b1, name='BollingerB_' + str(n))
            b2 = round((c- ma + 2 * msd) / (4 * msd),3)
            B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
            b_m= b_m.fillna(0.)
            b_m=b_m.replace([-np.inf,np.inf],0.)
            b_u= b_u.fillna(0.)
            b_u=b_u.replace([-np.inf,np.inf],0.)
            b_d= b_d.fillna(0.)
            b_d=b_d.replace([-np.inf,np.inf],0.)
            B2=B2.replace([-np.inf,np.inf],0.)
            B2= B2.fillna(0.)
            b_m=round(b_m,3)
            b_u=round(b_u,3)
            b_d=round(b_d,3)
            B2=round(B2,3)
            return list(b_m),list(b_u),list(b_d),list(B2)
        def bol_ba_per_cat(em,param):
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            if param=='slope':
                b_l=['strong buy' if (i<=-0.5 and sl>0) else 'buy' if (i<=0.2 and i>-0.5) else 'sell' if (i>0.8 and i<=1.4) else 'strong sell' if (sl<0 and i>1.4) else 'neutral' for i,sl in zip(em,c_grad)]
            return b_l
        def bol_ba_cat(b_m,b_u,b_d,c,param):
            b_m=pd.Series(b_m)
            b_u=pd.Series(b_u)
            b_d=pd.Series(b_d)
            b_u_std1=((b_u/2)+(b_m/2)).tolist()
            b_d_std1=((b_d/2)+(b_m/2)).tolist()
            b_m=b_m.tolist()
            b_u=b_u.tolist()
            b_d=b_d.tolist()             
            if param=='classic':
                b_l=['sell' if l<k else 'buy' if l>j else 'strong buy' if (l<=j and l>=m) else 'strong sell' if (l>=j and l<=n) else 'neutral' for i,j,k,l,m,n in zip(b_u,b_m,b_d,c,b_u_std1,b_d_std1)]
            return b_l

        def williams_ad():
            wd=[0.]*c.shape[0]
            i=0
            while (i+1)<=c.index[-1]:
                if i > 0:
                    prev_value =wd[i-1]
                    prev_close = c[i-1]
                    if c[i] > prev_close:
                        ad = c[i] - min(prev_close,l[i])
                    elif c[i] < prev_close:
                        ad = c[i] - max(prev_close, h[i])
                    else:
                        ad = 0.
                    wd[i]= ad+prev_value
                i+=1
            wa = pd.Series(wd, name='William_dist')
            wa=round(wa,3)
            return list(wa)
        def will_ad_cat(w,c,param):
            w=pd.Series(w)
            w_grad=pd.Series(np.gradient(w.ewm(span=1, min_periods=1).mean())).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            w=w.tolist()
             
            if param=='divergence':
                wad_l=['strong buy' if (i>0 and j<0 and k<=-3600.234) else 'sell' if (i<0 and j>0 and k<=-1200.838 and k>-3600.234) else 'neutral' if (k<=1198.558 and k>-1200.838) else 'buy' if (i>0 and j<0 and k<=3597.954 and k>1198.558) else 'strong sell' for i,j,k in zip(w_grad,c_grad,w)]
            return wad_l

        def williams_r_per(n):   
            i=0
            will_list=[np.nan]*c.shape[0]
            while (i+1)<=c.index[-1]:
              if i > n:
                den=max(h[i-n:i]) - min(l[i-n:i])
                if den==0:
                    den=0.0001
                will_list[i]=(-1*(max(h[i-n:i]) - c[i]) / den)
              i+=1
            w = pd.Series(will_list, name='William_r_per_' + str(n))
            w= w.fillna(-0.5)
            w=round(w,3)
            return list(w)
        def william_rper_cat(wir,c,param):
            wir=pd.Series(wir)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            wir_grad=pd.Series(np.gradient(wir.ewm(span=1, min_periods=1).mean())).tolist()
            wir=wir.tolist()
            if param=='classic':
              wir_l=['strong buy' if i<=-0.5003 else 'buy' if (i>-0.5003 and i<-0.5001) else 'sell' if (i>=-0.5001 and i<-0.500) else 'buy' if (i<=-0.4999 and i>-0.500) else 'sell' if (i<=-0.4997 and i>-0.4999) else 'strong sell' if(i>-0.4997) else 'neutral' for i in wir]
            elif param=='divergence':
              wir_l=['strong buy' if i<=-0.5003 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>-0.4997) else 'neutral' for i,j,k in zip(wir,wir_grad,c_grad)]
            return wir_l

        def trix(n):
            exp1 = c.ewm(span=n, min_periods=n).mean()
            exp2 = exp1.ewm(span=n, min_periods=n).mean()
            exp3 = exp2.ewm(span=n, min_periods=n).mean()
            i=0
            roc_list = [np.nan]
            while (i + 1) <= c.index[-1]:
                roc = round((exp3[i + 1] - exp3[i]) / exp3[i],3)
                roc_list.append(roc)
                i+=1
            tr = pd.Series(roc_list, name='Trix_' + str(n))
            tr= tr.fillna(0.)
            tr=tr.replace([-np.inf,np.inf],0.)
            tr=round(tr,3)
            return list(tr)
        def trix_cat(tr,c,param):
            tr=pd.Series(tr)
            tr_grad=pd.Series(np.gradient(tr.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            tr=tr.tolist()
            if param=='slope':
                tr_l=['strong buy' if (i<=-68.531) else 'buy' if (i<=-27.829 and i>-68.531) else 'sell' if ((i<0 and i>-27.829) or (i<0 and j<0)) else 'buy' if ((i>0 and i<12.872) or (i>0 and j>0)) else 'sell' if (i<=53.574 and i>12.872) else 'strong sell' if(i>53.574) else 'neutral' for i,j in zip(tr,c_grad)]
            elif param=='divergence':
                tr_l=['strong buy' if (i<=-68.531) else 'buy' if (j>0 and k<0) else 'sell' if (j<0 and k>0) else 'strong sell' if(i>53.574) else 'neutral' for i,j,k in zip(tr,tr_grad,c_grad)]
            return tr_l

        def ultimate_oscillator():
            i = 0
            truerange = [0]
            buypressure = [0]
            while i < c.index[-1]:
                tr = max(h[i+1], c[i]) - min(l[i+1], c[i])
                truerange.append(tr)
                bp = c[i+1] - min(l[i+1],c[i])
                buypressure.append(bp)
                i+=1
            ultOsc = pd.Series(round((4 * pd.Series(buypressure).rolling(7).sum() / pd.Series(truerange).rolling(7).sum()),3) +\
                        round((2 * pd.Series(buypressure).rolling(14).sum() / pd.Series(truerange).rolling(14).sum()),3) + \
                              round((pd.Series(buypressure).rolling(28).sum() / pd.Series(truerange).rolling(28).sum()),3),\
                            name='Ultimate_Osc')
            ultOsc=ultOsc.fillna(0.)
            ultOsc=ultOsc.replace([-np.inf,np.inf],0.)
            ultOsc=round(ultOsc,3)
            return list(ultOsc)
        def ultosc_cat(uo,c,param):
            uo=pd.Series(uo)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            uo_grad=pd.Series(np.gradient(uo.ewm(span=1, min_periods=1).mean())).tolist()
            uo=uo.tolist()
            if param=='classic':
              uo_l=['strong buy' if (i<=-61.115) else 'sell' if (i<=-41.303 and i>-61.115) else 'neutral' if (i<-21.491 and i>-41.303) else 'buy' if(i>-21.491 and i<=-1.679) else 'strong sell' for i in uo]
            elif param=='divergence':
              uo_l=['strong buy' if (i<=-61.115) else 'strong sell' if i>-1.679 else 'buy' if (j>0 and k<0) else 'sell' if (j<0 and k>0) else 'neutral' for i,j,k in zip(uo,uo_grad,c_grad)]
            return uo_l


        def true_strength_index(r,s):
            mom = pd.Series(c.diff(1))
            absMom = abs(mom)
            ema1 = pd.Series(mom.ewm(span=r, min_periods=r).mean())
            abs_ema1 = pd.Series(absMom.ewm(span=r, min_periods=r).mean())
            ema2 = pd.Series(ema1.ewm(span=s, min_periods=s).mean())
            abs_ema2 = pd.Series(abs_ema1.ewm(span=s, min_periods=s).mean())
            tsi = pd.Series(round(ema2/abs_ema2,3), name='TSI_' + str(r) + '_' + str(s))
            tsi=tsi.fillna(0.)
            tsi=tsi.replace([-np.inf,np.inf],0.)
            tsi=round(tsi,3)
            return list(tsi)
        def tsi_cat(ts,param):
            ts=pd.Series(ts)
            tsma=ts.rolling(1,min_periods=1).mean()
            ts_grad=pd.Series(np.gradient(ts.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            tsma_grad=pd.Series(np.gradient(tsma.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            ts=ts.tolist()
            if param=='classic':
                ts_l=['strong buy' if (i<=-0.602) else 'buy' if (i<=-0.204 and i>-0.602) else 'sell' if (i<0 and i>-0.204) else 'buy' if(i>0 and i<=0.195) else 'sell' if (i<=0.593 and i>0.195) else 'strong sell' if(i>0.593) else 'neutral' for i in ts]
            elif param=='slope':
                ts_l=['strong buy' if (i<=-0.602) else 'buy' if (i<=j and i<-0.204 and k<l) else 'strong sell' if (i>0.593) else 'sell' if (i>0.195 and i>=j and k>l) else 'neutral' for i,j,k,l in zip(ts,tsma,ts_grad,tsma_grad)]
            return ts_l


        def force_index(n):
            force_ind = pd.Series(round(c.diff(n) * v.diff(n),3), name='Force_' + str(n))                
            force_ind=force_ind.fillna(0.)
            force_ind=force_ind.replace([-np.inf,np.inf],0.)
            force_ind=round(force_ind,3)
            return list(force_ind)
        def forceidx_cat(f,c,param):
            f=pd.Series(f)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            f_grad=pd.Series(np.gradient(f.ewm(span=1, min_periods=1).mean())).tolist()
            f=f.tolist()
            if param=='slope':
                fi_l=['strong buy' if (i>0 and j>1 and k>1) else 'buy' if (i>0 and j>0 and k>0) else 'sell' if (i<0 and j<0 and k<0) else 'strong sell' if (i<0 and j<-1 and k<-1) else 'neutral' for i,j,k in zip(f,c_grad,f_grad)]
            return fi_l

        def coppock_curve(n):
            m1 = c.diff(int(n * 11 / 10) - 1)
            n1 = c.shift(int(n * 11 / 10) - 1)
            roc1 = m1 / n1
            m2 = c.diff(int(n * 14 / 10) - 1)
            n2 = c.shift(int(n * 14 / 10) - 1)
            roc2 = m2 / n2
            copp = pd.Series(round((roc1 + roc2).ewm(span=n, min_periods=n).mean(),3), name='Copp_' + str(n))
            copp=copp.fillna(0.5)
            copp=copp.replace([-np.inf,np.inf],0.)
            copp=round(copp,3)
            return list(copp)
        def copcurve_cat(co,param):
            if param=='classic':
                cc_l=['strong buy' if (i<=-189.341) else 'buy' if (i<=-11.319 and i>-189.341) else 'sell' if (i<0 and i>-11.319) else 'buy' if(i>0 and i<=166.702) else 'sell' if (i<=344.724 and i>166.702) else 'strong sell' if(i>344.724) else 'neutral' for i in co]
            return cc_l

        def vortex_indicator(n=14):
            i = 0
            tr = [0]
            while i < c.index[-1]:
                range1 = max(h[i+1],c[i]) - min(l[i+1],c[i])
                tr.append(range1)
                i+=1
            i = 0
            vm_pl=[0]
            vm_mi=[0]
            while i < c.index[-1]:
                vm1=abs(h[i+1]-l[i])
                vm2=abs(l[i+1]-h[i])
                vm_pl.append(vm1)
                vm_mi.append(vm2)
                i+=1
            vi_pl=pd.Series(round(pd.Series(vm_pl).rolling(n).sum()/pd.Series(tr).rolling(n).sum(),3))
            vi_mi=pd.Series(round(pd.Series(vm_mi).rolling(n).sum()/pd.Series(tr).rolling(n).sum(),3))
            vi_pl=vi_pl.fillna(0.5)
            vi_pl=vi_pl.replace([-np.inf,np.inf],1)
            vi_mi=vi_mi.fillna(0.5)
            vi_mi=vi_mi.replace([-np.inf,np.inf],1)
            vi_pl=round(vi_pl,3)
            vi_mi=round(vi_mi,3)
            return list(vi_pl),list(vi_mi)
        def vortind_cat(vip,vim,param):
            if param=='classic':
                vi_l=['strong buy' if ((i>1.7 and j<0.9) or (i>1.1 and j<0)) else 'buy' if (i>=1.1 and j<=0.9) else 'sell' if (i<=0.9 and j>=1.1) else 'strong sell' if((i<0.9 and j>1.7) or (i<0 and j>1.1)) else 'neutral' for i,j in zip(vip,vim)]
            return vi_l

        def know_sure_thing_oscillator(r1,r2,r3,r4,p1,p2,p3,p4):
            m1 = c.diff(r1 - 1)
            n1 = c.shift(r1 - 1)
            roc1 = m1/n1
            m2 = c.diff(r2 - 1)
            n2 = c.shift(r2 - 1)
            roc2 = m2/n2
            m3 = c.diff(r3 - 1)
            n3 = c.shift(r3 - 1)
            roc3 = m3/n3
            m4 = c.diff(r4 - 1)
            n4 = c.shift(r4 - 1)
            roc4 = m4/n4
            kst_osc = pd.Series(round(roc1.rolling(p1).sum()+2*roc2.rolling(p2).sum()+3*roc3.rolling(p3).sum()+4*roc4.rolling(p4).sum(),3),name='know_sure_thing_oscillator')
            kst_osc=kst_osc.fillna(0.)
            kst_osc=kst_osc.replace([-np.inf,np.inf],0.)
            kst_osc=round(kst_osc,3)
            return list(kst_osc)
        def kst_cat(k,c,param):
            k=pd.Series(k)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            k_grad=pd.Series(np.gradient(k.ewm(span=1, min_periods=1).mean())).tolist()
            k=k.tolist()
            if param=='classic':
                k_l=['strong buy' if (i<-359.586) else 'buy' if (-359.586<=i and i<-119.8) else 'sell' if (-119.8<=i and i<0) else 'buy' if (i>0 and i<119.987) else 'sell' if (i>119.987 and i<359.773) else 'strong sell' if(i>=359.773) else 'neutral' for i in k]
            elif param=='divergence':
                k_l=['strong buy' if i<-359.586 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=359.773) else 'neutral' for i,j,k in zip(k,k_grad,c_grad)]  
            return k_l

        def standard_deviation(n):
            std = pd.Series(round(c.rolling(n,min_periods=n).std(),3), name='STD_'+str(n))
            std=std.fillna(0.)
            std=std.replace([-np.inf,np.inf],0.)
            std=round(std,3)
            return list(std)

        def rate_of_change(n):
            m = c.diff(n - 1)
            n1 = c.shift(n - 1)
            roc = pd.Series(round(m/n1,3),name='ROC_'+str(n))
            roc=roc.fillna(0.5)
            roc=roc.replace([-np.inf,np.inf],0.5)
            roc=round(roc,3)
            return list(roc)
        def roc_cat(r,c,param):
            r=pd.Series(r)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            r_grad=pd.Series(np.gradient(r.ewm(span=1, min_periods=1).mean())).tolist()
            r=r.tolist()
            if param=='classic':
                roc_l=['strong buy' if (i<=-1579.8) else 'sell' if (i<=-933.6 and i>-1579.8) else 'neutral' if (i<-287.4 and i>-933.6) else 'buy' if(i>=-287.4 and i<358.8) else 'strong sell' for i in r]
            elif param=='divergence':
                roc_l=['strong buy' if (i<=-1579.8) else 'strong sell' if i>=358.8 else 'buy' if (j>0 and k<0) else 'sell' if (j<0 and k>0) else 'neutral' for i,j,k in zip(r,r_grad,c_grad)]
            return roc_l

        def ppsr():
            piv_po = pd.Series(round((h+l+c)/3,3))
            r1 = pd.Series(round(2*piv_po-l,3))
            s1 = pd.Series(round(2*piv_po-h,3))
            r2 = pd.Series(round(piv_po+h-l,3))
            s2 = pd.Series(round(piv_po-h+l,3))
            r3 = pd.Series(round(h+2*(piv_po-l),3))
            s3 = pd.Series(round(l-2*h-piv_po,3))
            psr = {'piv_po': list(piv_po), 'r1': list(r1), 's1': list(s1), 'r2': list(r2), 's2': list(s2), 'r3': list(r3), 's3': list(s3)}
            return psr
        def ppsr_cat(pp,r1,r2,s1,s2,c,param):            
            if param=='classic':
              ppsr_l=['buy' if (i>j and i<k) else 'strong buy' if(i>k and i<l) else 'sell' if(i>l) else 'sell' if (i<j and i>m) else 'strong sell' if(i<m and i>n) else 'buy' if(i<n) else 'neutral' for i,j,k,l,m,n in zip(c,pp,r1,r2,s1,s2)]
            return ppsr_l


        def stochastic_oscillator_k():
            sto_osc_k = pd.Series(round((c-l)/(h-l),3), name='Sto_Osc%k')
            sto_osc_k=sto_osc_k.fillna(0.5)
            sto_osc_k = sto_osc_k.replace([np.inf, -np.inf],0.5)
            sto_osc_k=round(sto_osc_k,3)
            return list(sto_osc_k)
        def sto_osc_cat(k,d,param):
            k=pd.Series(k)
            d=pd.Series(d)
            sk_grad=pd.Series(np.gradient(k.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            sk=k.tolist()
            sd=d.tolist()
            if param=='slope':
                s_l=['buy' if i>j else 'sell' if i<j  else 'neutral' for i,j in zip(sk,sd)]
            elif param=='classic':
                s_l=['strong buy' if (i<=10 and j>0) else 'sell' if (i>10 and i<20 and j<0) else 'buy' if(i>80 and i<90 and j>0) else 'strong sell' if(i>=90 and j<0) else 'neutral' for i,j in zip(sk,sk_grad)]
            return s_l

        def stochastic_oscillator_d(n):
            sto_osc_k=stochastic_oscillator_k()
            sto_osc_k=pd.Series(sto_osc_k)
            sto_osc_d=pd.Series(round(sto_osc_k.ewm(span=n, min_periods=n).mean(),3), name='Sto_Osc%d_' + str(n))
            sto_osc_d=sto_osc_d.fillna(0.5)
            sto_osc_d = sto_osc_d.replace([np.inf, -np.inf],0.5)
            sto_osc_d=round(sto_osc_d,3)
            return list(sto_osc_d)

        def commodity_chan_ind(n):  
          piv_po=pd.Series((h+l+c)/3)  
          cci= pd.Series(round((piv_po-piv_po.rolling(n).mean())/(0.015*piv_po.rolling(n).std()),3))
          cci = cci.replace([np.inf, -np.inf],0.)
          cci=cci.fillna(0.)
          cci=round(cci,3)    
          return list(cci)
        def cci_cat(cc,c,param):
            cc=pd.Series(cc)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            cc_grad=pd.Series(np.gradient(cc.ewm(span=1, min_periods=1).mean())).tolist()
            if param=='classic':
                cci_l=['strong sell' if i>=150 else 'strong buy' if i<=-150 else 'buy' if (i>=100 and i<150) else 'sell' if (i<=-100 and i>-150) else 'neutral' for i in cc]
            elif param=='divergence':
                cci_l=['strong buy' if i<=-150 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=150) else 'neutral' for i,j,k in zip(cc,cc_grad,c_grad)]
            return cci_l

        def awesome_oscillator(n1,n2):
            cal=pd.Series((h+l)/2)
            awe_osc =pd.Series(round(cal.rolling(n1).mean()-cal.rolling(n2).mean(),3),name='awes_osc_'+str(n1)+'_'+str(n2))
            awe_osc=awe_osc.fillna(0.)
            awe_osc = awe_osc.replace([np.inf, -np.inf],0.)
            awe_osc=round(awe_osc,3)
            return list(awe_osc)
        def aweosc_cat(a,c,param):
            a=pd.Series(a)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            a_grad=pd.Series(np.gradient(a.ewm(span=1, min_periods=1).mean())).tolist()
            a=a.tolist()
            if param=='classic':
                ao_l=['strong buy'if i<=-179.958 else 'buy' if (i>-179.958 and i<=-60.019) else 'sell' if (i>-60.019 and i<0) else 'buy' if(i>0 and i<=59.919) else 'sell' if(i>59.919 and i<179.858) else 'strong sell' if(i>=179.858) else 'neutral' for i in a]
            elif param=='divergence':
                ao_l=['strong buy' if i<-179.958 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=179.858) else 'neutral' for i,j,k in zip(a,a_grad,c_grad)] 
            return ao_l

        def chaik_monflow_cat(cmf,c,param):
            cmf=pd.Series(cmf)
            cmf_grad=pd.Series(np.gradient(cmf.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            cmf=cmf.tolist()
            if param=='slope': 
                cmf_l=['strong buy' if (i<-0.7) else 'sell' if(i>=-0.7 and i<0 and j<0 and k>0) else 'buy' if(i>0 and i<=0.7 and j>0 and k<0) else 'strong sell' if (i>0.7) else 'neutral' for i,j,k in zip(cmf,cmf_grad,c_grad)]
            return cmf_l


        def detrended_price_oscillator(n):
            c_n=c.shift(int((n/2))+1) 
            c_sma=pd.Series(c.rolling(n).mean())
            dpo=pd.Series(round(c_n-c_sma,3), name='etrended_price_oscillator_'+str(n))
            dpo=dpo.fillna(0.)
            dpo = dpo.replace([np.inf, -np.inf],0.)
            dpo=round(dpo,3)
            return list(dpo)
        def dpo_cat(d,c,param):
            d=pd.Series(d)
            d_grad=pd.Series(np.gradient(d.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            d=d.tolist()
             
            if param=='slope':
                dto_l=['strong buy' if (i<=-299.954 and j>0 and k>0) else 'sell' if (i<0 and j<0 and i>-299.954) else 'buy' if (i>0 and j>0 and i<299.883) else 'strong sell' if (i>=299.883 and j<0 and k<0) else 'neutral' for i,j,k in zip(d,d_grad,c_grad)]
            return dto_l

        def directional_movement(n):
            i=0
            dm_plus=[0.]*c.shape[0]
            dm_min=[0.]*c.shape[0]
            while (i+1)<=c.index[-1]:
                if i > 0:
                  prev_high =h[i-1]
                  prev_low= l[i-1]
                  up_mov=h[i]-prev_high
                  down_mov=prev_low-l[i]
                  if up_mov > down_mov and up_mov > 0:
                    dm_plus[i]=up_mov
                  elif down_mov > up_mov and down_mov > 0:
                    dm_min[i]=down_mov
                i+=1
            di_plus=pd.Series((pd.Series(dm_plus)/pd.Series(average_true_range(n))).ewm(span=n, min_periods=n).mean(),name='di_plus_' + str(n))
            di_min=pd.Series((pd.Series(dm_min)/pd.Series(average_true_range(n))).ewm(span=n, min_periods=n).mean(),name='di_min_' + str(n))
            di_plus=pd.Series(di_plus)
            di_min=pd.Series(di_min)
            di=pd.Series(di_plus-di_min)
            adx=pd.Series(round(abs(((di_plus-di_min)/(di_plus+di_min))*100).ewm(span=n, min_periods=n).mean(),5),name='di_min_' + str(n))
            adx=adx.fillna(0.)
            di=round(di,3)
            return list(adx),list(di)
        def dirmove_cat(adx,di,c,param):
            adx=pd.Series(adx)
            adx_grad=pd.Series(np.gradient(adx.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            adx=adx.tolist()
            if param=='classic':
              adx_l=['buy' if (i>25 and j>0) else 'sell' if (i>25 and j<0) else 'strong buy' if (i>50 and j>0) else 'strong sell' if ((i>50 and j<0)) else 'neutral' for i,j in zip(adx,di)]
              return adx_l
            elif param=='slope':
              adx_l=['buy' if (i>25 and k>0) else 'sell' if (i>25 and j<0 and k<0) else 'strong buy' if (i>50 and k>0) else 'strong sell' if (i>50 and j<0 and k<0) else 'buy' if (i<=25 and j>0 and k<0) else 'sell' if (i<=25 and j<0 and k>0) else 'neutral' for i,j,k in zip(adx,adx_grad,c_grad)]
              return adx_l


        def elders_force_index(n):
            i=0
            efi=[0.]*c.shape[0]
            while (i+1)<=c.index[-1]:
              if i > 0:
                  prev_close=c[i-1]
                  efi[i]=int((c[i]-prev_close))*v[i]
              i+=1
            efi=pd.to_numeric(pd.Series(efi),errors='coerce')
            eld_for_idx=pd.Series(round(efi.ewm(span=n, min_periods=n).mean(),3),name='elders_force_index_'+str(n))
            eld_for_idx=eld_for_idx.fillna(0.)
            eld_for_idx=round(eld_for_idx,3)
            return list(eld_for_idx)
        def elder_fi_cat(efi,c,param):
            efi=pd.Series(efi)
            efi_grad=pd.Series(np.gradient(efi.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            efi=efi.tolist()
            if param=='slope':
                efi_l=['strong buy' if (i>0 and j>1) else 'buy' if(i>0 and j>0) else 'sell' if(i<0 and j<0) else 'strong sell' if (i<0 and j<-1) else 'neutral' for i,j in zip(efi_grad,c_grad)]
            elif param=='classic':
                efi_l=['strong buy' if i<=-5999584.459 else 'sell' if (i>-5999584.459 and i<=-1999807.004) else 'buy' if (i>=1999970.452 and i<5999747.907) else 'strong sell' if(i>=5999747.907) else 'neutral' for i in efi]
            return efi_l

        def envelope(n):
            upp_env=pd.Series(round(c.rolling(n).sum()+c.rolling(n).sum()*0.1,3),name='upper_env_'+str(n))
            low_env=pd.Series(round(c.rolling(n).sum()-c.rolling(n).sum()*0.1,3),name='lower_env_'+str(n))
            upp_env=upp_env.fillna(0.)
            low_env=low_env.fillna(0.)
            upp_env=round(upp_env,3)
            low_env=round(low_env,3)
            return list(upp_env),list(low_env)
        def env_cat(up,lo,c,param):           
            if param=='classic':
                env_l=['strong sell' if(k>i and k>16892.652) else 'sell' if(k>i and k<=16892.652) else 'strong buy' if(k<j and k<=13821.261) else 'buy' if(k<j and k>13821.261) else 'neutral' for i,j,k in zip(up,lo,c)]
            return env_l

        def acceleration_bands(n):
            ab_middle_band =pd.Series(round(c.rolling(window=n,center=False).mean(),3))
            aupband = pd.Series(h*(1+4*(h-l)/(h+l)))
            ab_upper_band = pd.Series(round(aupband.rolling(window=n,center=False).mean(),3))
            adownband = pd.Series(l*(1-4*(h-l)/(h+l)))
            ab_lower_band= pd.Series(round(adownband.rolling(window=n,center=False).mean(),3))
            ab_middle_band=ab_middle_band.fillna(0.)
            ab_upper_band=ab_upper_band.fillna(0.)
            ab_lower_band=ab_lower_band.fillna(0.)
            ab_lower_band=round(ab_lower_band,3)
            ab_upper_band=round(ab_upper_band,3)
            ab_middle_band=round(ab_middle_band,3)
            return list(ab_middle_band),list(ab_upper_band),list(ab_lower_band)
        def acc_bands_cat(au,am,al,c,param):
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()             
            if param=='slope':
              ab_l=['strong buy' if (i>j and m>0) else 'buy' if(i>=j and m<0) else 'sell' if(i<=l and m>0) else 'strong sell' if(i<l and m<0) else 'neutral' for i,j,l,m in zip(c,au,al,c_grad)]
            return ab_l

        def parabolic_stop_and_return(step_raising=0.02,step_falling=0.02,max_af_raising=0.2,max_af_falling=0.2):
            raising_sar = [0.]*h.shape[0] 
            falling_sar = [0.]*h.shape[0]
            #extreme point
            ep = h[0]
            #acceleration factor
            af = step_raising
            sar = l[0]
            up = True
            for i in range(1, len(h)):
              if up:
              # Rising SAR
                ep = np.max([ep, h[i]])
                af = np.min([af + step_raising if (ep == h[i]) else af, max_af_raising]) 
                sar = sar + af *(ep - sar)
                raising_sar[i] = round(sar,3) 
              else: 
              # Falling SAR 
                ep = np.min([ep, l[i]])
                af = np.min([af + step_falling if (ep == l[i]) else af, max_af_falling])
                sar = sar + af * (ep - sar)
                falling_sar[i] = round(sar,3)
              # Trend switch
              if up and (sar>l[i] or sar>h[i]):
                up = False
                sar = ep
                af = step_falling
              elif not up and (sar<l[i] or sar<h[i]):
                up = True
                sar = ep
                af = step_raising 
            rise=pd.Series(raising_sar)
            fall=pd.Series(falling_sar)
            rise=round(rise,3)
            fall=round(fall,3)
            return list(rise),list(fall)


        def price_channel(n):
            n_day_high = pd.Series(round(h.rolling(n).max(),3),name='high_price_ch_'+str(n))
            n_day_low = pd.Series(round(l.rolling(n).min(),3),name='low_price_ch_'+str(n))
            center = pd.Series(round((n_day_high + n_day_low)/2.0,3),name='center_price_ch_'+str(n))
            n_day_high=n_day_high.fillna(0.)
            n_day_low=n_day_low.fillna(0.)
            center=center.fillna(0.)
            n_day_high=round(n_day_high,3)
            n_day_low=round(n_day_low,3)
            center=round(center,3)
            return list(n_day_high),list(n_day_low),list(center)
        def pri_chann_cat(ph,pl,pc,c,param):
            ph=pd.Series(ph)
            pl=pd.Series(pl)
            pc=pd.Series(pc)
            ph_grad=pd.Series(np.gradient(ph.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            pl_grad=pd.Series(np.gradient(pl.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            ph=ph.tolist()
            pl=pl.tolist()             
            if param=='slope':
              prichan_l=['strong buy' if(i>=j and l>0 and n>0) else 'strong sell' if(i<=k and m<0 and n<0) else 'buy' if (n>0 and i>k) else 'sell' if(n<0 and i<j) else 'neutral' for i,j,k,l,m,n in zip(c,ph,pl,ph_grad,pl_grad,c_grad)]
              return prichan_l

        def percentage_price_oscillator(n1,n2,n3):
            ppo = pd.Series(round(100.0*(c.ewm(span=n1,min_periods=n1).mean()-c.ewm(span=n2,min_periods=n2).mean())/c.ewm(span=n2,min_periods=n2).mean(),3),name='percentage_price_oscillator')
            ppo_signal = pd.Series(round(ppo.ewm(span=n3,min_periods=n3).mean(),3),name='percentage_price_oscillator_signal')
            ppo_hist = pd.Series(round(ppo - ppo_signal,3),name='percentage_price_oscillator_hist')
            ppo=ppo.fillna(0.5)
            ppo_signal=ppo_signal.fillna(0.5)
            ppo_hist=ppo_hist.fillna(0.5)
            ppo_hist=round(ppo_hist,3)
            ppo_signal=round(ppo_signal,3)
            ppo=round(ppo,3)
            return list(ppo),list(ppo_signal),list(ppo_hist)
        def ppo_cat(p,ps,ph,c,param):
            p=pd.Series(p)
            ph=pd.Series(ph)
            p_grad=pd.Series(np.gradient(p.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            p=p.tolist()             
            if param=='divergence':
              ppo_l=['strong buy' if(i<-3505.676) else 'sell' if(i>=-3505.676 and i<21663.365 and j<0 and k>0) else 'buy' if(i>46832.407 and i<=72001.448 and j>0 and k<0) else 'strong sell' if (i>72001.448) else 'neutral' for i,j,k in zip(p,p_grad,c_grad)]
              return ppo_l
            elif param=='slope':
              ppo_l=['strong buy' if(i<-3505.676) else 'sell' if((i<=j) and k<0) else 'buy' if(i>=j and k>0) else 'strong sell' if(i>72001.448) else 'neutral' for i,j,k in zip(p,ps,p_grad)]
              return ppo_l

        def price_momentum_oscillator(n1,n2,n3):
            pmo = pd.Series(round((10*((100*(c/c.shift(1)))-100.0).ewm(span=n2,min_periods=n2).mean()).ewm(span=n1,min_periods=n1).mean(),3),name='price_momentum_oscillator')
            signal = pd.Series(round(pmo.ewm(span=n3,min_periods=n3).mean(),3),name='price_momentum_oscillator_signal')
            pmo=pmo.fillna(0.)
            signal=signal.fillna(0.)
            pmo=round(pmo,3)
            signal=round(signal,3)
            return list(pmo),list(signal)
        def pmo_cat(p,ps,c,param):
            p=pd.Series(p)
            p_grad=pd.Series(np.gradient(p.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            p=p.tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()             
            if param=='divergence':
              pmo_l=['strong buy' if(i<-5995.173) else 'sell' if(i>=-5995.173 and i<-1999.286 and j<0 and k>0) else 'buy' if(i>1996.6 and i<=5992.487 and j>0 and k<0) else 'strong sell' if (i>5992.487) else 'neutral' for i,j,k in zip(p,p_grad,c_grad)]
              return pmo_l
            elif param=='slope':
              pmo_l=['strong buy' if(i<-5995.173) else 'sell' if((i<=j) and k<0) else 'buy' if(i>=j and k>0) else 'strong sell' if(i>5992.487) else 'neutral' for i,j,k in zip(p,ps,p_grad)]
              return pmo_l
            elif param=='classic':
              pmo_l=['strong buy' if(i<-5995.173) else 'sell' if(i<-1999.286 and i>=-5995.173) else 'buy' if(i>1996.6 and i<=5992.487) else 'strong sell' if(i>5992.487) else 'neutral' for i in p]
              return pmo_l


        def volatility(n):
            volat=pd.Series(round(c.rolling(n).std(),3),name='volatility')
            volat=volat.fillna(0.)
            volat=round(volat,3)
            return list(volat)

        def quadrant_range():
            size =pd.Series(round((h-l)/4.0,3))
            l1 = pd.Series(l,name='quad_range_1')
            l2 = pd.Series(l1+size,name='quad_range_2')
            l3 = pd.Series(l2+size,name='quad_range_3')
            l4 = pd.Series(l3+size,name='quad_range_4')
            l5 = pd.Series(l4+size,name='quad_range_5')
            l1=round(l1,3)
            l2=round(l2,3)
            l3=round(l3,3)
            l4=round(l4,3)
            l5=round(l5,3)
            return list(l1),list(l2),list(l3),list(l4),list(l5)

        def drawdown():
            highest_high=c.expanding().max()
            draw_down = pd.Series(round((c/highest_high)-1.0,3),name='drawdown')
            return list(draw_down)

        def ichimoku():
            period_high = h.rolling(window=9,center=False).max()
            period_low = l.rolling(window=9,center=False).min()
            conversion_line = (period_high + period_low) / 2
            period_high1= h.rolling(window=26,center=False).max()
            period_low1 = l.rolling(window=26,center=False).min()
            base_line = (period_high1 + period_low1) / 2
            lead_span1= ((conversion_line+ base_line) / 2 ).shift(26)
            period_high2 = h.rolling(window=52,center=False).max()
            period_low2 = l.rolling(window=52,center=False).min()
            lead_span2 = ((period_high2 + period_low2) / 2).shift(26)
            lead_span1=lead_span1.fillna(0.)
            lead_span1=round(lead_span1,3)
            lead_span2=lead_span2.fillna(0.)
            lead_span2=round(lead_span2,3)
            return list(lead_span1),list(lead_span2)
        def ichmoku_cat(ia,ib,c,param):
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            if param=='slope':
                ich_l=['strong buy' if (i>1 and (j>k or k>l)) else 'strong sell' if(i<-1 and (j<k or k<l)) else 'buy' if(j>k or k>l) else 'sell' if(j<k or k<l) else 'neutral' for i,j,k,l in zip(c_grad,c,ia,ib)] 
            return ich_l

        def macd(period_fast=26, period_slow=12):
            ema_f = c.ewm(span=period_fast, min_periods=period_slow).mean()
            ema_s = c.ewm(span=period_slow, min_periods=period_slow).mean()
            macd = pd.Series(ema_f - ema_s, name='macd_' + str(period_fast) + '_' + str(period_slow))
            macd_signal = pd.Series(macd.ewm(span=9, min_periods=9).mean(), name='macd_signal_' + str(period_fast) + '_' + str(period_slow))
            macd_diff = pd.Series(macd - macd_signal, name='macd_diff' + str(period_fast) + '_' + str(period_slow))
            macd=macd.fillna(0.5)
            macd=round(macd,3)
            macd_signal=macd_signal.fillna(0.5)
            macd_signal=round(macd_signal,3)
            return list(macd),list(macd_signal)
        def macd_cat(m,ms,c,param):
            m=pd.Series(m)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            m_grad=pd.Series(np.gradient(m.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            m=m.tolist()
            if param=='classic':
                macd_l=['strong buy' if (i<=-59.999 and i<j) else 'sell' if i<j else 'strong sell' if (i>j and i>=59.996) else 'buy' if i>j else 'neutral' for i,j in zip(m,ms)]
            elif param=='slope':
                macd_l=['strong buy' if(i<=-59.999) else 'sell' if((i>-59.999 and i<=-20.001) or (j<0 and k>0)) else 'buy' if((i>19.998 and i<=59.996) or (j>0 and k<0)) else 'strong sell' if (i>59.996) else 'neutral' for i,j,k in zip(m,m_grad,c_grad)]
            return macd_l

        def on_balance_volume(n=10):
            i = 0
            obv = [0.]
            for i in range(len(c.index) - 1):
                if c.iloc[i + 1] - c.iloc[i] > 0:
                    obv.append(v.iloc[i])
                elif c.iloc[i + 1] - c.iloc[i] < 0:
                    obv.append(-v.iloc[i + 1])
                else:
                    obv.append(0)
            obv = pd.Series(obv)
            obv1 = pd.Series(obv.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
            obv1=obv1.fillna(0)
            obv1=round(obv1,3)
            return list(obv1)
        def obv_cat(ob,c,param):
            ob=pd.Series(ob)
            o_grad=pd.Series(np.gradient(ob.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()                           
            if param=='slope':
                obv_l=['strong buy' if(i>0 and j>0) else 'strong sell' if(i<0 and j<0) else 'buy' if (i>0 and j<0) else 'sell' if(i<0 and j>0) else 'neutral' for i,j in zip(o_grad,c_grad)]
            return obv_l
          
        def accumulation_distribution(period=21):
            ad =((2*c-h-l)/(h-l))*v
            ad=pd.to_numeric(pd.Series(ad),errors='coerce')
            acc= ad.ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
            acc= pd.Series(acc, name='acc_dist' + str(period))
            acc=acc.fillna(0.)
            acc = acc.replace([np.inf, -np.inf],0.)
            acc=round(acc,3)
            return list(acc)

        def acc_dist_cat(ad,c,param):
            ad=pd.Series(ad)
            ad_grad=pd.Series(np.gradient(ad.ewm(span=1, min_periods=1).mean())).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            ad=ad.tolist()             
            if param=='divergence':
                ad_l=['strong buy' if (k<=-1800052.523) else 'sell' if (i<0 and j>0) else 'buy' if (i>0 and j<0) else 'strong sell' if(k>=1798933.444) else 'neutral' for i,j,k in zip(ad_grad,c_grad,ad)]
            return ad_l
        def keltner_channel(n):
            kelChM = pd.Series(round(((h + l + c)/3).rolling(n, min_periods=n).mean(),3),\
                              name='KelChM_' + str(n))
            kelChU = pd.Series(round(((4 * h - 2 * l + c)/3).rolling(n, min_periods=n).mean(),3),\
                              name='KelChU_' + str(n))
            kelChD = pd.Series(round(((-2 *h + 4 * l + c)/3).rolling(n, min_periods=n).mean(),3),\
                              name='KelChD_' + str(n))
            kelChM=kelChM.fillna(0.)
            kelChU=kelChU.fillna(0.)
            kelChD=kelChD.fillna(0.)
            kelChM=round(kelChM,3)
            kelChU=round(kelChU,3)
            kelChD=round(kelChD,3)
            return list(kelChM),list(kelChU),list(kelChD)
        def keltcha_cat(k_m,k_u,k_d,c,param):            
            if param=='classic':
                k_l=['strong sell' if m<k else 'strong buy' if m>i else 'buy' if (m<j and m>=k) else 'sell' if (m>j and m<=i) else 'neutral' for i,j,k,m in zip(k_u,k_m,k_d,c)]
            return k_l

        def relative_strength_index(n=14):
            diff = c.diff(1)
            which_dn = diff < 0
            up, dn = diff, diff*0
            up[which_dn], dn[which_dn] = 0, -up[which_dn]
            emaup = up.ewm(span=n, min_periods=n).mean()
            emadn = dn.ewm(span=n, min_periods=n).mean()
            rsi = 100 * emaup / (emaup + emadn)
            rsi=rsi.fillna(50)
            rsi = rsi.replace([np.inf, -np.inf],50)
            rsi=round(rsi,3)
            return list(rsi)
        def rsi_cat(r,c,param):
            r=pd.Series(r)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            r_grad=pd.Series(np.gradient(r.ewm(span=1, min_periods=1).mean())).tolist()
            r=r.tolist()
            if param=='slope':
                r_l=['strong buy' if (i<=20 and j>0) else 'strong sell' if (i>=80 and j<0)  else 'sell' if ((i>20 and i<=40) and j<0) else 'buy' if ((i<80 and i>=40) and j>0) else 'neutral' for i,j in zip(r,c_grad)]
            elif param=='divergence':
                r_l=['strong buy' if i<=20 else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'strong sell' if(i>=80) else 'neutral' for i,j,k in zip(r,r_grad,c_grad)]
            return r_l

        retracements = [23.6,38.2,50.00,61.8,76.4,78.6,85.40]
        extensions = [127.2,138.2,150.00,161.8,176.4,261.8,423.6]
        def chaikin_money_flow(n=21):
            mfv = ((c - l) - (h - c)) / (h - l)
            mfv = mfv.fillna(0.0)
            mfv *= v
            cmf = (mfv.rolling(n, min_periods=0).sum()/ v.rolling(n, min_periods=0).sum())
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
            cmf=round(cmf,3)
            return list(cmf)
        def chaik_monflow_cat(cmf,c,param):
            cmf=pd.Series(cmf)
            cmf_grad=pd.Series(np.gradient(cmf.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            cmf=cmf.tolist()
            if param=='slope':
              cmf_l=['strong buy' if (i<-0.7) else 'sell' if(i>=-0.7 and i<0 and j<0 and k>0) else 'buy' if(i>0 and i<=0.7 and j>0 and k<0) else 'strong sell' if (i>0.7) else 'neutral' for i,j,k in zip(cmf,cmf_grad,c_grad)]
            return cmf_l

        def typical_price():
            typical_price = (h + l + c) / 3
            typical_price=round(typical_price,3)
            return list(typical_price)
        def typ_pri_cat(t,c,param):
            t=pd.Series(t)
            t_grad=pd.Series(np.gradient(t.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            t=t.tolist()             
            if param=='slope':
                typri_l=['buy' if (i==j and k<0 and l>0) else 'sell' if (i==j and k>0 and l<0) else 'strong buy' if (k>0 and l>0) else 'strong sell' if(k<0 and l<0) else 'neutral' for i,j,k,l in zip(t,c,t_grad,c_grad)]
            return typri_l

        def ease_of_movement(period=14):
            num1=pd.Series((((h+l)/2)-((h.shift(-1)+l.shift(-1))/2))).fillna(0)
            num=num1.astype('int')
            den=pd.Series((((v)/100000000)/((h-l)))).fillna(0.)
            eom=num/den
            eom_ma = pd.Series(eom.rolling(period).mean(), name = 'eom_' + str(period))
            eom_ma = eom_ma.replace([np.inf, -np.inf],0.)
            eom_ma=eom_ma.fillna(0.)
            eom=round(eom_ma,3)
            return list(eom_ma)
        def ease_mov_cat(e,c,param):
            e=pd.Series(e)
            e_grad=pd.Series(np.gradient(e.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            sma=pd.Series(c.rolling(14, min_periods=14).mean()).fillna(0.)
            sma_grad=pd.Series(np.gradient(sma.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            e=e.tolist()             
            if param=='classic':
                eom_l=['strong buy' if (i<=-179992377.072) else 'sell' if(i>-179992377.072 and i<=-60028401.943) else 'buy' if(i>=59935573.185 and i<179899548.314) else 'strong sell' if(i>=179899548.314) else 'neutral' for i in e]
            elif param=='slope':
                eom_l=['strong sell' if ((i==j and l>1 and l>m) or k>179899548.314) else 'strong buy' if((i==j and l<-1 and l<m) or k<=-179992377.072) else 'buy' if((i==j and l>0 and l>m) or k>=59935573.185) else 'sell' if((i==j and l<0 and l<m) or k<=-60028401.943) else 'neutral' for i,j,k,l,m in zip(c,sma,e,c_grad,sma_grad)]
            return eom_l

        def mass_index(period=9):
            diff=h-l
            ex1=diff.ewm(span=9,min_periods=9).mean()
            ex2=ex1.ewm(span=9,min_periods=9).mean()
            mass=ex1/ex2
            massindex=pd.Series(mass.rolling(25).sum(),name='massindex')
            massindex=massindex.fillna(0.)
            massindex=round(massindex,3)
            return list(massindex)
          
        def calc_ulcer_index(periods=14):
            period_high_close = c.rolling(periods + 1).apply(lambda x: np.amax(x), raw=True)
            percentage_drawdown = c
            percentage_drawdown = (percentage_drawdown - period_high_close)/period_high_close * 100
            percentage_drawdown = np.clip(percentage_drawdown, a_min=None, a_max=0)
            percentage_drawdown = percentage_drawdown ** 2
            percentage_drawdown = percentage_drawdown.fillna(0)
            period_sum = percentage_drawdown.rolling(periods+1).sum()
            squared_average = round((period_sum / periods), 2)
            ulcer_index = round(squared_average ** 0.5, 2)
            ulcer_index.fillna(1,inplace=True)
            ulcer_index=round(ulcer_index,3)
            return list(ulcer_index)
        def money_flow_index(period=14):
            typical_price=(h+l+c)/3
            typical_price=pd.Series(typical_price)
            pos_neg=typical_price.diff(periods=1)
            pos_neg=pd.Series(pos_neg.fillna(0.)).astype('int')
            pos_money_flow=[]
            pos_money_flow.append((pos_neg.multiply(v)).where(pos_neg>0))
            pos_money_flow=pd.Series(v for v in pos_money_flow)
            pos_money_flow=pos_money_flow.fillna(0)
            neg_money_flow=[]
            neg_money_flow.append((pos_neg.multiply(v)).where(pos_neg<0))
            neg_moey_flow=pd.Series(v for v in neg_money_flow)
            neg_moey_flow=neg_moey_flow.fillna(0)
            pos_money_flow_sum=pos_money_flow[0].rolling(window=period, min_periods=0).sum()
            neg_money_flow_sum=neg_money_flow[0].rolling(window=period,min_periods=0).sum()  
            money_ratio=abs(pos_money_flow_sum/neg_money_flow_sum)
            mfi=100-(100/(1+money_ratio)) 
            mfi=pd.Series(mfi,name='money_flow_index')
            mfi = mfi.replace([np.inf, -np.inf],0.)
            mfi.fillna(1,inplace=True)
            mfi=round(mfi,3)
            return list(mfi)
        def mfi_cat(m,c,param):
            m=pd.Series(m)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).tolist()
            m_grad=pd.Series(np.gradient(m.ewm(span=1, min_periods=1).mean())).tolist()
            m=m.tolist()
            if param=='classic':
                m_l=['strong buy' if i<=10 else 'strong sell' if i>=90 else 'sell' if (i>10 and i<50) else 'buy' if (i<90 and i>50) else 'neutral' for i in m]
            elif param=='divergence':
                m_l=['strong buy' if (k<=10) else 'sell' if (i<0 and j>0) else 'buy' if (i>0 and j<0) else 'strong sell' if(i>=90) else 'neutral' for i,j,k in zip(m_grad,c_grad,m)]
            return m_l

        def aroon(period=25):
            aroonup = []
            aroondown = []
            x = period
            while x< len(h):
                aroon_up = ((h[x-period:x].tolist().index(max(h[x-period:x])))/float(period))*100
                aroon_down = ((l[x-period:x].tolist().index(min(l[x-period:x])))/float(period))*100
                aroonup.append(aroon_up)
                aroondown.append(aroon_down)
                x+=1
            aroonup=pd.Series(aroonup)
            aroondown=pd.Series(aroondown)
            return list(aroonup), list(aroondown)
        def aroon_updown_cat(au,ad,param):
            au=pd.Series(au)
            ad=pd.Series(ad)
            au_grad=pd.Series(np.gradient(au.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            ad_grad=pd.Series(np.gradient(ad.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            au=au.tolist()
            ad=ad.tolist()
            if param=='slope':
                aroon_l=['strong buy' if (i>80 and j<20) else 'strong sell' if(i<20 and j>80) else 'buy' if(i>50 and j<50) else 'sell' if(i<50 and j>50) else 'buy' if(i>=j and k>=0 and l<=0) else 'sell' if(j>=i and k<=0 and l>=0) else 'neutral' for i,j,k,l in zip(au,ad,au_grad,ad_grad)] 
            return aroon_l
          
        def heikin():
            ha_close = pd.Series(round((o + h + l + c)/4,4))
            ha_open = round((o.shift(-1) + c.shift(-1))/2,4)
            elements = np.array([h, l, ha_open, ha_close])
            ha_high = elements.max(0)
            ha_low = elements.min(0)
            h_close=pd.Series(ha_close)
            h_open=pd.Series(ha_open)
            h_low=pd.Series(ha_low)
            h_high=pd.Series(ha_high)
            h_close=h_close.fillna(0.0001)
            h_open=h_open.fillna(0.0001)
            h_low=h_low.fillna(0.0001)
            h_high=h_high.fillna(0.0001)
            h_high=round(h_high,3)
            h_low=round(h_low,3)
            h_open=round(h_open,3)
            h_close=round(h_close,3)
            return list(h_close),list(h_open),list(h_low),list(h_high)
          
        def uptrend():
            diff = h - l
            for r,e in zip(retracements,extensions):
                uptr1=round((h - (diff * r/100)),3)
                uptr2=round((l + (diff * e/100)),3)
                return list(uptr1),list(uptr2)

        def downtrend():
            diff = h - l
            for r,e in zip(retracements,extensions):
                downtr1=round((l + (diff * r/100)),3)
                downtr2=round((h - (diff * e/100)),3)
                return list(downtr1),list(downtr2)
        def trendlines_cat(upl,loh,c,param):
            upl=pd.Series(upl)
            loh=pd.Series(loh)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            upl_grad=pd.Series(np.gradient(upl.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            loh_grad=pd.Series(np.gradient(loh.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            if param=='slope':
                trenlin_l=['strong buy' if (i>0 and k>1 and l>m) else 'strong sell' if(j<0 and k<-1 and l<n) else 'buy' if(i>0 and k>0 and l>=m) else 'sell' if(j<0 and k<0 and l<=n) else 'neutral' for i,j,k,l,m,n in zip(upl_grad,loh_grad,c_grad,c,upl,loh)] 
            return trenlin_l
            
        def aroon_oscillator(period=25):
            aroonup = []
            aroondown = []
            x = period
            while x< len(h):
                aroon_up = ((h[x-period:x].tolist().index(max(h[x-period:x])))/float(period))*100
                aroon_down = ((l[x-period:x].tolist().index(min(l[x-period:x])))/float(period))*100
                aroonup.append(aroon_up)
                aroondown.append(aroon_down)
                x+=1
            aroonup=pd.Series(aroonup)
            aroondown=pd.Series(aroondown)
            oscillator=aroonup-aroondown
            oscillator=round(oscillator,3)
            return list(oscillator)
        def aroon_osc_cat(osc,c,param):
            osc=pd.Series(osc)
            c_grad=pd.Series(np.gradient(c.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            osc_grad=pd.Series(np.gradient(osc.ewm(span=1, min_periods=1).mean())).fillna(0.).tolist()
            osc=osc.tolist()
            if param=='classic':
                aroon_l=['strong buy' if (i>50) else 'strong sell' if(i<-50) else 'buy' if(i>0) else 'sell' if(i<0) else 'neutral' for i in osc] 
            elif param=='divergence':
                aroon_l=['strong buy' if (i>50) else 'strong sell' if(i<-50) else 'buy' if(j>0 and k<0) else 'sell' if(j<0 and k>0) else 'neutral' for i,j,k in zip(osc,osc_grad,c_grad)]
            return aroon_l
          
        def donchian_channel(n):
            i = 0
            dc_list = []
            while i < n:
                dc_list.append(0)
                i += 1
            i = 0
            while i + n - 1 < c.index[-1]:
                dc = round(max(h.loc[i:i + n - 1]) - min(l.loc[i:i + n - 1]),3)
                dc_list.append(dc)
                i += 1
            donchian_chan = pd.Series(dc_list, name='Donchian_' + str(n))
            donchian_chan = donchian_chan.shift(n - 1)
            donchian_chan=donchian_chan.fillna(0.)
            donchian_chan=round(donchian_chan,3)
            return list(donchian_chan)
        def donchan_cat(dc,param):
            if param=='classic':
                dc_l=['strong buy' if (i<320.0) else 'sell' if (320.0<=i and i<640.0) else 'neutral' if (640.0<=i and i<960.0) else 'buy' if (960.0<=i and i<1280.0) else 'strong sell' for i in dc]
            return dc_l
        try:
            money_flow_idx=money_flow_index(14)[-1]
            money_flow_index_decisions=mfi_cat((money_flow_index(14)),c,'divergence')[-1]
            acc_dist=accumulation_distribution(21)[-1]
            accumulation_distribution_decisions=acc_dist_cat(accumulation_distribution(21),c,'divergence')[-1]
            on_bal_vol=on_balance_volume()[-1]
            on_balance_volume_decisions=obv_cat(on_balance_volume(),c,'slope')[-1]
            ease_of_move=ease_of_movement()[-1]
            ease_of_movement_decisions=ease_mov_cat(ease_of_movement(),c,'slope')[-1]
            mass_idx=mass_index()[-1]
            calc_ulcer_idx=calc_ulcer_index()[-1]
            exp_mov_avg=exponential_moving_average()[-1]
            exponential_moving_average_decisions=exp_ma_cat(c,exponential_moving_average(),'slope')[-1]
            momen=momentum(10)[-1]
            momentum_decisions=momen_cat(momentum(10),c,'slope')[-1]        
            chaikin_osc=chaikin_oscillator()[-1]
            chaikin_oscillator_decisions=chaikosc_cat(chaikin_oscillator(),c,'divergence')[-1]        
            chai_vol=chaikin_volatility()[-1]
            avg_true_range=average_true_range(14)[-1]
            bollinger_bands_m=bollinger_bands(20)[0][-1]
            bollinger_bands_u=bollinger_bands(20)[1][-1]
            bollinger_bands_d=bollinger_bands(20)[2][-1]
            bollinger_bands_decisions=bol_ba_cat(bollinger_bands(20)[0],bollinger_bands(20)[1],bollinger_bands(20)[2],c,'classic')[-1]
            bollinger_bands_per=bollinger_bands(20)[3][-1]
            bollinger_bands_per_decisions=bol_ba_per_cat(bollinger_bands(20)[3],'slope')[-1]
            williams_acc_dist=williams_ad()[-1]
            williams_ad_decisions=will_ad_cat(williams_ad(),c,'divergence')[-1]
            williams_r=williams_r_per(14)[-1]
            williams_r_per_decisions=william_rper_cat(williams_r_per(14),c,'divergence')[-1]        
            trx=trix(9)[-1]
            trix_decisions=trix_cat(trix(9),c,'divergence')[-1]
            ultimate_osc=ultimate_oscillator()[-1]
            ultimate_oscillator_decisions=ultosc_cat(ultimate_oscillator(),c,'divergence')[-1]
            true_strength_idx=true_strength_index(25,13)[-1]
            true_strength_index_decisions=tsi_cat(true_strength_index(25,13),'slope')[-1]       
            force_idx=force_index(13)[-1]
            force_index_decisions=forceidx_cat(force_index(13),c,'slope')[-1]
            coppock_cur=coppock_curve(10)[-1]
            coppock_curve_decisions=copcurve_cat(coppock_curve(10),'classic')[-1]       
            chai_mon_flow=chaikin_money_flow(21)[-1]
            chaikin_money_flow_decisions=chaik_monflow_cat(chaikin_money_flow(21),c,'slope')[-1]
            keltner_channel_m=keltner_channel(20)[0][-1]
            keltner_channel_u=keltner_channel(20)[1][-1]
            keltner_channel_d=keltner_channel(20)[2][-1]
            keltner_channel_decisions=keltcha_cat(keltner_channel(20)[0],keltner_channel(20)[1],keltner_channel(20)[2],c,'classic')[-1]
            vortex_indicator_plus=vortex_indicator(25)[0][-1]
            vortex_indicator_minus=vortex_indicator(25)[1][-1]
            vortex_indicator_decisions=vortind_cat(vortex_indicator(25)[0],vortex_indicator(25)[1],'classic')[-1]        
            know_sure_thing_osc=know_sure_thing_oscillator(10, 15, 20, 30, 10, 10, 10, 15)[-1]
            know_sure_thing_oscillator_decisions=kst_cat(know_sure_thing_oscillator(10, 15, 20, 30, 10, 10, 10, 15),c,'divergence')[-1]
            std_dev=standard_deviation(10)[-1]
            rate_of_chan=rate_of_change(10)[-1]
            rate_of_change_decisions=roc_cat(rate_of_change(10),c,'divergence')[-1]
            piv_po=ppsr()['piv_po'][-1]
            r1=ppsr()['r1'][-1]
            r2=ppsr()['r2'][-1]
            r3=ppsr()['r3'][-1]
            s1=ppsr()['s1'][-1]
            s2=ppsr()['s2'][-1]
            s3=ppsr()['s3'][-1]
            piv_po_decisions=ppsr_cat(ppsr()['piv_po'],ppsr()['r1'],ppsr()['r2'],ppsr()['s1'],ppsr()['s2'],c,'classic')[-1]        
            stochastic_osc_k=stochastic_oscillator_k()[-1]
            stochastic_osc_d=stochastic_oscillator_d(14)[-1]
            stochastic_oscillator_decisions=sto_osc_cat(stochastic_oscillator_k(),stochastic_oscillator_d(14),'classic')[-1]        
            commodity_chan_idx=commodity_chan_ind(20)[-1]
            commodity_chan_ind_decisions=cci_cat(commodity_chan_ind(20),c,'divergence')[-1]        
            awesome_osc=awesome_oscillator(5,34)[-1]
            awesome_oscillator_decisions=aweosc_cat(awesome_oscillator(5,34),c,'divergence')[-1]
            detrended_price_osc=detrended_price_oscillator(10)[-1]
            detrended_price_oscillator_decisions=dpo_cat(detrended_price_oscillator(10),c,'slope')[-1]       
            directional_movement_adx=directional_movement(14)[0][-1]
            directional_movement_net_di=directional_movement(14)[1][-1]
            directional_movement_oscillator_decisions=dirmove_cat(directional_movement(14)[0],directional_movement(14)[1],c,'classic')[-1]        
            elders_force_idx=elders_force_index(13)[-1]
            elders_force_index_decisions=elder_fi_cat(elders_force_index(13),c,'slope')[-1]        
            upper_envelope=envelope(20)[0][-1]
            lower_envelope=envelope(20)[1][-1]
            envelope_decisions=env_cat(envelope(20)[0],envelope(20)[1],c,'classic')[-1]       
            acceleration_bands_middle=acceleration_bands(20)[0][-1]
            acceleration_bands_upper=acceleration_bands(20)[1][-1]
            acceleration_bands_lower=acceleration_bands(20)[2][-1]
            acceleration_bands_decisions=acc_bands_cat(acceleration_bands(20)[1],acceleration_bands(20)[0],acceleration_bands(20)[2],c,'slope')[-1]
            parabolic_sar_raise=parabolic_stop_and_return()[0][-1]
            parabolic_sar_fall=parabolic_stop_and_return()[1][-1]
            price_channel_high=price_channel(20)[0][-1]
            price_channel_low=price_channel(20)[1][-1]
            price_channel_center=price_channel(20)[2][-1]
            price_channel_decisions=pri_chann_cat(price_channel(20)[0],price_channel(20)[1],price_channel(20)[2],c,'slope')[-1]        
            percentage_price_osc=percentage_price_oscillator(12,26,9)[0][-1]
            percentage_price_oscillator_signal=percentage_price_oscillator(12,26,9)[1][-1]
            percentage_price_oscillator_hist=percentage_price_oscillator(12,26,9)[2][-1]
            percentage_price_oscillator_decisions=ppo_cat(percentage_price_oscillator(12,26,9)[0],percentage_price_oscillator(12,26,9)[1],percentage_price_oscillator(12,26,9)[2],c,'slope')[-1]                  
            price_momentum_osc=price_momentum_oscillator(20,35,10)[0][-1]
            price_momentum_oscillator_signal=price_momentum_oscillator(20,35,10)[1][-1]
            price_momentum_oscillator_decisions=pmo_cat(price_momentum_oscillator(20,35,10)[0],price_momentum_oscillator(20,35,10)[1],c,'slope')[-1]
            volatility=volatility(20)[-1]
            quadrant_range_1=quadrant_range()[0][-1]
            quadrant_range_2=quadrant_range()[1][-1]
            quadrant_range_3=quadrant_range()[2][-1]
            quadrant_range_4=quadrant_range()[3][-1]
            quadrant_range_5=quadrant_range()[4][-1]
            drawdown=drawdown()[-1]
            ichimoku_lead_span1=ichimoku()[0][-1]
            ichimoku_lead_span2=ichimoku()[1][-1]
            ichimoku_lead_decisions=ichmoku_cat(ichimoku()[0],ichimoku()[1],c,'slope')[-1]
            mov_avg_con_div=macd()[0][-1]
            macd_signal=macd()[1][-1]
            macd_decisions=macd_cat(macd()[0],macd()[1],c,'slope')[-1]
            relative_strength_idx=relative_strength_index()[-1]
            relative_strength_index_decisions=rsi_cat(relative_strength_index(),c,'slope')[-1]
            typical_pri=typical_price()[-1]
            typical_price_decisions=typ_pri_cat(typical_price(),c,'slope')[-1]                
            heikin_close=heikin()[0][-1]
            heikin_open=heikin()[1][-1]
            heikin_low=heikin()[2][-1]
            heikin_high=heikin()[3][-1]        
            uptrend_high=uptrend()[0][-1]
            uptrend_low=uptrend()[1][-1]
            downtrend_low=downtrend()[0][-1]
            downtrend_high=downtrend()[1][-1]
            trendline_decisions=trendlines_cat(uptrend()[1],downtrend()[1],c,'slope')[-1]        
            aroon_osc=aroon_oscillator()[-1]
            aroon_oscillator_decisions=aroon_osc_cat(aroon_oscillator(),c,'divergence')[-1]                  
            aroon_up=aroon()[0][-1]
            aroon_down=aroon()[1][-1]
            aroon_decisions=aroon_updown_cat(aroon()[0],aroon()[1],'slope')[-1]
            donchian_chann=donchian_channel(20)[-1]
            donchian_channel_decisions=donchan_cat(donchian_channel(20),'classic')[-1]
            lst=[str(sym),str('EQ'),str(tstamp),str(closeadj),str(prevcloadj),str(money_flow_idx), str(money_flow_index_decisions), str(acc_dist), str(accumulation_distribution_decisions), str(on_bal_vol), str(on_balance_volume_decisions), str(ease_of_move), str(ease_of_movement_decisions), str(exp_mov_avg), str(exponential_moving_average_decisions), str(momen), str(momentum_decisions), str(chaikin_osc), str(chaikin_oscillator_decisions), str(bollinger_bands_m),str(bollinger_bands_u), str(bollinger_bands_d), str(bollinger_bands_decisions),str(bollinger_bands_per), str(bollinger_bands_per_decisions), str(williams_acc_dist),str(williams_ad_decisions), str(williams_r), str(williams_r_per_decisions), str(trx),str(trix_decisions), str(ultimate_osc), str(ultimate_oscillator_decisions), str(true_strength_idx),str(true_strength_index_decisions), str(force_idx), str(force_index_decisions), str(coppock_cur), str(coppock_curve_decisions), str(chai_mon_flow), str(chaikin_money_flow_decisions),str(keltner_channel_m),str(keltner_channel_u),str(keltner_channel_d),str(keltner_channel_decisions),str(vortex_indicator_plus),str(vortex_indicator_minus),str(vortex_indicator_decisions),str(know_sure_thing_osc),str(know_sure_thing_oscillator_decisions),str(rate_of_chan),str(rate_of_change_decisions),str(piv_po),str(r1),str(r2),str(r3),str(s1),str(s2),str(s3),str(piv_po_decisions),str(stochastic_osc_k),str(stochastic_osc_d),str(stochastic_oscillator_decisions),str(commodity_chan_idx),str(commodity_chan_ind_decisions),str(awesome_osc),str(awesome_oscillator_decisions),str(detrended_price_osc),str(detrended_price_oscillator_decisions),str(directional_movement_adx),str(directional_movement_net_di),str(directional_movement_oscillator_decisions),str(elders_force_idx),str(elders_force_index_decisions),str(upper_envelope),str(lower_envelope),str(envelope_decisions),str(acceleration_bands_middle),str(acceleration_bands_upper),str(acceleration_bands_lower),str(acceleration_bands_decisions),str(price_channel_high),str(price_channel_low),str(price_channel_center),str(price_channel_decisions),str(percentage_price_osc),str(percentage_price_oscillator_signal),str(percentage_price_oscillator_hist),str(percentage_price_oscillator_decisions),str(price_momentum_osc),str(price_momentum_oscillator_signal),str(price_momentum_oscillator_decisions),str(ichimoku_lead_span1),str(ichimoku_lead_span2),str(ichimoku_lead_decisions),str(mov_avg_con_div),str(macd_signal),str(macd_decisions),str(relative_strength_idx),str(relative_strength_index_decisions),str(typical_pri),str(typical_price_decisions),str(uptrend_high),str(uptrend_low),str(downtrend_low),str(downtrend_high),str(trendline_decisions),str(aroon_osc),str(aroon_oscillator_decisions),str(aroon_up),str(aroon_down),str(aroon_decisions),str(donchian_chann),str(donchian_channel_decisions),str(mass_idx),str(calc_ulcer_idx),str(chai_vol),str(avg_true_range),str(std_dev),str(parabolic_sar_raise),str(parabolic_sar_fall),str(volatility),str(quadrant_range_1),str(quadrant_range_2),str(quadrant_range_3),str(quadrant_range_4),str(quadrant_range_5),str(drawdown),str(heikin_close),str(heikin_open),str(heikin_low),str(heikin_high)]
            lst=[i.replace('nan', '0') for i in lst]
            return tuple(lst)
        except:
            pass
#        return((sym,'EQ',tstamp,sma5,ema5,smadecision5,smasignal5,sma20,ema20,smadecision20,smasignal20,sma50,ema50,smadecision50,smasignal50,sma100,ema100,smadecision100,smasignal100,sma200,ema200,smadecision200,smasignal200,sma240,ema240,smadecision240,smasignal240,rsivalue,rsisignal,rsidivergence,goldencrosssignal,macdvalue ,signalvalue,macddecision,macdsignal,stochrsival,stochrsidecision,adxsignal,adxdecision))
 
def db_connet(database,user,password,host,logger):
    try:
        cnx = pymysql.connect(user=user, password=password,host=host,database=database,charset='utf8', autocommit=True)
        cur=cnx.cursor()
        logger.info("******************connected to output table in mysql******************")
        return cur,cnx
    except Exception as e:
        logger.info('error:: some issue for connecting to mysql ...check conf file intputdbprop in etc folder....')
        logger.exception('error :: connection error %s',e)
        raise e

def getSignals(databasename,cur,cnx,logger):
    try:
        logger.info('******************get signal execution started***********************')

        usedb="USE %s "% (databasename)
        cur.execute(usedb)

        logger.info("*******************Database connection established********************")

        fsql="select symbol,series,open_adj,high_adj,low_adj,close_adj,prevclose_adj,last_adj,tottrdqty_adj,tottrdval,timestamp from nsedailybhavhist where timestamp > (SELECT DATE_SUB(date(sysdate()), INTERVAL 1 YEAR)) and series='EQ' order by timestamp;"
        cur.execute(fsql)
        logger.info("****************Required data fetched from database*********************")
        df=pd.DataFrame(list(cur.fetchall()),columns=['symbol', 'series', 'open_adj', 'high_adj', 'low_adj', 'close_adj','prevclose_adj','last_adj', 'tottrdqty_adj', 'tottrdval', 'timestamp'])
        fetchsymbols="select symbol from nsesymbols;"
        cur.execute(fetchsymbols)
        symbols=list((pd.DataFrame(list(cur.fetchall())))[0])
        logger.info("*************************Function for signal mining created******************")
        count=0
        try:
            fsql_signals="select timestamp from nsesignals;"
            #fsql_bhavhist="select timestamp from nsedailybhavhist;"
            cur.execute(fsql_signals)
            signaldate=list((pd.DataFrame(list(cur.fetchall())))[0])
            #cur.execute(fsql_bhavhist)
            bhavdate=list(df['timestamp'])
            if max(signaldate)>=max(bhavdate):
                return None
        except:
            pass
        #lst=[]
        insertSQL="INSERT INTO nsesignals (symbol, series, timestamp, close_adj, prev_close_adj, money_flow_index, money_flow_index_decisions, accumulation_distribution, accumulation_distribution_decisions, on_balance_volume, on_balance_volume_decisions, ease_of_movement, ease_of_movement_decisions, exponential_moving_average, exponential_moving_average_decisions, momentum, momentum_decisions, chaikin_oscillator, chaikin_oscillator_decisions, bollinger_bands_m,bollinger_bands_u, bollinger_bands_d, bollinger_bands_decisions,bollinger_bands_per, bollinger_bands_per_decisions, williams_ad,williams_ad_decisions, williams_r_per, williams_r_per_decisions, trix,trix_decisions, ultimate_oscillator, ultimate_oscillator_decisions, true_strength_index,true_strength_index_decisions, force_index, force_index_decisions, coppock_curve, coppock_curve_decisions, chaikin_money_flow, chaikin_money_flow_decisions,keltner_channel_m,keltner_channel_u,keltner_channel_d,keltner_channel_decisions,vortex_indicator_plus,vortex_indicator_minus,vortex_indicator_decisions,know_sure_thing_oscillator,know_sure_thing_oscillator_decisions,rate_of_change,rate_of_change_decisions,piv_po,r1,r2,r3,s1,s2,s3,piv_po_decisions,stochastic_oscillator_k,stochastic_oscillator_d,stochastic_oscillator_decisions,commodity_chan_ind,commodity_chan_ind_decisions,awesome_oscillator,awesome_oscillator_decisions,detrended_price_oscillator,detrended_price_oscillator_decisions,directional_movement_adx,directional_movement_net_di,directional_movement_oscillator_decisions,elders_force_index,elders_force_index_decisions,upper_envelope,lower_envelope,envelope_decisions,acceleration_bands_middle,acceleration_bands_upper,acceleration_bands_lower,acceleration_bands_decisions,price_channel_high,price_channel_low,price_channel_center,price_channel_decisions,percentage_price_oscillator,percentage_price_oscillator_signal,percentage_price_oscillator_hist,percentage_price_oscillator_decisions,price_momentum_oscillator,price_momentum_oscillator_signal,price_momentum_oscillator_decisions,ichimoku_lead_span1,ichimoku_lead_span2,ichimoku_lead_decisions,macd,macd_signal,macd_decisions,relative_strength_index,relative_strength_index_decisions,typical_price,typical_price_decisions,uptrend_high,uptrend_low,downtrend_low,downtrend_high,trendline_decisions,aroon_oscillator,aroon_oscillator_decisions,aroon_up,aroon_down,aroon_decisions,donchian_channel,donchian_channel_decisions,mass_index,calc_ulcer_index,chaikin_volatility,average_true_range,standard_deviation,parabolic_sar_raise,parabolic_sar_fall,volatility,quadrant_range_1,quadrant_range_2,quadrant_range_3,quadrant_range_4,quadrant_range_5,drawdown,heikin_close,heikin_open,heikin_low,heikin_high) VALUES "
        lstN=[]
        for sym in symbols:
            tp=generatesignals(df[df.symbol==sym].sort_values('timestamp').reset_index(drop=True),sym)
            if tp!=None:
                #lst.append(tp)
                val=str(tp)
                isql=insertSQL+val
                print(insertSQL,'',val)
                try:
                    cur.execute(isql)
                except Exception as e:
                    logger.exception('error ::%s',e)
                    pass
                count=count+1
                if (count!=0)&((count%100)==0):
                    print(count)
            else:
                print('this row is not uploaded')
                lstN.append(tp)

        #insertSQL="INSERT INTO nsesignals (symbol, series, timestamp, money_flow_index, money_flow_index_decisions, accumulation_distribution, accumulation_distribution_decisions, on_balance_volume, on_balance_volume_decisions, ease_of_movement, ease_of_movement_decisions, exponential_moving_average, exponential_moving_average_decisions, momen, momentum_decisions, chaikin_oscillator, chaikin_oscillator_decisions, bollinger_bands_m,bollinger_bands_u, bollinger_bands_d, bollinger_bands_decisions,bollinger_bands_per, bollinger_bands_per_decisions, williams_ad,williams_ad_decisions, williams_r_per, williams_r_per_decisions, trix,trix_decisions, ultimate_oscillator, ultimate_oscillator_decisions, true_strength_index,true_strength_index_decisions, force_index, force_index_decisions, coppock_curve, coppock_curve_decisions, chaikin_money_flow, chaikin_money_flow_decisions,keltner_channel_m,keltner_channel_u,keltner_channel_d,keltner_channel_decisions,vortex_indicator_plus,vortex_indicator_minus,vortex_indicator_decisions,know_sure_thing_oscillator,know_sure_thing_oscillator_decisions,rate_of_change,rate_of_change_decisions,piv_po,r1,r2,r3,s1,s2,s3,piv_po_decisions,stochastic_oscillator_k,stochastic_oscillator_d,stochastic_oscillator_decisions,commodity_chan_ind,commodity_chan_ind_decisions,awesome_oscillator,awesome_oscillator_decisions,detrended_price_oscillator,detrended_price_oscillator_decisions,directional_movement_adx,directional_movement_net_di,directional_movement_oscillator_decisions,elders_force_index,elders_force_index_decisions,upper_envelope,lower_envelope,envelope_decisions,acceleration_bands_middle,acceleration_bands_upper,acceleration_bands_lower,acceleration_bands_decisions,price_channel_high,price_channel_low,price_channel_center,price_channel_decisions,percentage_price_oscillator,percentage_price_oscillator_signal,percentage_price_oscillator_hist,percentage_price_oscillator_decisions,price_momentum_oscillator,price_momentum_oscillator_signal,price_momentum_oscillator_decisions,ichimoku_lead_span1,ichimoku_lead_span2,ichimoku_lead_decisions,macd,macd_signal,macd_decisions,relative_strength_index,relative_strength_index_decisions,typical_price,typical_price_decisions,uptrend_high,uptrend_low,downtrend_low,downtrend_high,trendline_decisions,aroon_oscillator,aroon_oscillator_decisions,aroon_up,aroon_down,aroon_decisions,donchian_channel,donchian_channel_decisions,mass_index,calc_ulcer_index,chaikin_volatility,average_true_range,standard_deviation,parabolic_sar_raise,parabolic_sar_fall,volatility,quadrant_range_1,quadrant_range_2,quadrant_range_3,quadrant_range_4,quadrant_range_5,drawdown,heikin_close,heikin_open,heikin_low,heikin_high) VALUES "

        #val=str(lst)[1:-1]
        #isql=insertSQL+val
        print(str(datetime.datetime.now()-ptime))
        logger.info("| %s | ",str(datetime.datetime.now()-ptime))
    except Exception as e:
        logger.exception('error ::%s',e)
        pass

def main():
    logger =logCreate.create_logger()
    try:
        config = confReader.get_config(logger)
        logger.info("config file data :: %s",config)
    except Exception as e:
        logger.exception('error:: some issue in reading the config...check confReader.py script in bin folder..%s',e)
        raise e
    inputdbprop = config.get("inputdbprop", None)
    database_name=inputdbprop.get('dbname')
    cur,cnx = db_connet(inputdbprop.get('dbname'),inputdbprop.get('dbusername'),inputdbprop.get('dbpassword'),inputdbprop.get('hostname'),logger)
    databasename=inputdbprop.get('dbname')
    getsignls=getSignals(databasename,cur,cnx,logger)

if __name__ == "__main__":
        main()