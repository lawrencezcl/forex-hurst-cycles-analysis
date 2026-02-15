#!/usr/bin/env python3
"""
蔡森技术分析系统 - 多时间框架指数分析
使用 yfinance 获取 SPX, NASDAQ100, NIKKEI225, DAX 数据
"""

import json
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

INDEX_SYMBOLS = {
    'SPX': {'yf': '^GSPC', 'name': 'S&P 500'},
    'ND100': {'yf': '^NDX', 'name': 'Nasdaq 100'},
    'JP225': {'yf': '^N225', 'name': 'Nikkei 225'},
    'DAX': {'yf': '^GDAXI', 'name': 'DAX 40'},
}

def fetch_data(symbol, interval='1d', period='1y'):
    config = INDEX_SYMBOLS.get(symbol, {})
    yf_symbol = config.get('yf', symbol)
    try:
        time.sleep(3)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None

def calc_indicators(df):
    df = df.copy()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Sig'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()
    
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    bb = BollingerBands(close=df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    return df

def analyze(df, tf):
    df = calc_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    result = {'tf': tf, 'price': float(last['Close'])}
    
    # 均线
    if pd.notna(last['MA20']):
        if last['Close'] > last['MA5'] > last['MA10'] > last['MA20']:
            result['ma'] = '多头排列'
            result['ma_score'] = 2
        elif last['Close'] < last['MA5'] < last['MA10'] < last['MA20']:
            result['ma'] = '空头排列'
            result['ma_score'] = -2
        else:
            result['ma'] = '均线纠缠'
            result['ma_score'] = 0
    else:
        result['ma'] = 'N/A'
        result['ma_score'] = 0
    
    # MACD
    if prev['MACD'] <= prev['MACD_Sig'] and last['MACD'] > last['MACD_Sig']:
        result['macd'] = '金叉'
        result['macd_score'] = 2
    elif prev['MACD'] >= prev['MACD_Sig'] and last['MACD'] < last['MACD_Sig']:
        result['macd'] = '死叉'
        result['macd_score'] = -2
    elif last['MACD'] > last['MACD_Sig']:
        result['macd'] = '多头'
        result['macd_score'] = 1
    else:
        result['macd'] = '空头'
        result['macd_score'] = -1
    result['macd_pos'] = '零轴上' if last['MACD'] > 0 else '零轴下'
    
    # RSI
    rsi = last['RSI']
    result['rsi'] = float(rsi)
    if rsi >= 70:
        result['rsi_status'] = '超买'
        result['rsi_score'] = -1
    elif rsi >= 50:
        result['rsi_status'] = '强势'
        result['rsi_score'] = 1
    elif rsi >= 30:
        result['rsi_status'] = '弱势'
        result['rsi_score'] = -1
    else:
        result['rsi_status'] = '超卖'
        result['rsi_score'] = 1
    
    # KDJ
    if prev['K'] <= prev['D'] and last['K'] > last['D']:
        result['kdj'] = '金叉'
        result['kdj_score'] = 2
    elif prev['K'] >= prev['D'] and last['K'] < last['D']:
        result['kdj'] = '死叉'
        result['kdj_score'] = -2
    elif last['K'] > last['D']:
        result['kdj'] = '多头'
        result['kdj_score'] = 1
    else:
        result['kdj'] = '空头'
        result['kdj_score'] = -1
    
    # 趋势
    recent = df.tail(60)
    highs = recent['High'].values
    lows = recent['Low'].values
    hp = [highs[i] for i in range(2,len(highs)-2) if highs[i]>highs[i-1] and highs[i]>highs[i+1]]
    lp = [lows[i] for i in range(2,len(lows)-2) if lows[i]<lows[i-1] and lows[i]<lows[i+1]]
    
    if len(hp)>=2 and len(lp)>=2:
        if hp[-1]>hp[-2] and lp[-1]>lp[-2]:
            result['trend'] = '上升'
            result['trend_score'] = 2
        elif hp[-1]<hp[-2] and lp[-1]<lp[-2]:
            result['trend'] = '下降'
            result['trend_score'] = -2
        else:
            result['trend'] = '震荡'
            result['trend_score'] = 0
    else:
        result['trend'] = '不明'
        result['trend_score'] = 0
    
    # 支撑阻力
    resist = []
    suppt = []
    for i in range(2, len(recent)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            resist.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            suppt.append(lows[i])
    resist = sorted(set(resist), reverse=True)[:3]
    suppt = sorted(set(suppt), reverse=True)[:3]
    
    nr = next((r for r in resist if r > last['Close']), None)
    ns = next((s for s in reversed(suppt) if s < last['Close']), None)
    
    result['resistance'] = [float(x) for x in resist]
    result['support'] = [float(x) for x in suppt]
    result['near_r'] = float(nr) if nr else None
    result['near_s'] = float(ns) if ns else None
    
    # 综合得分
    total = (result['ma_score']*2 + result['macd_score'] + result['rsi_score'] + 
             result['kdj_score'] + result['trend_score'])
    result['total'] = total
    
    if total >= 5:
        result['dir'] = '看多'
        result['act'] = '买入'
    elif total >= 2:
        result['dir'] = '偏多'
        result['act'] = '可买'
    elif total <= -5:
        result['dir'] = '看空'
        result['act'] = '卖出'
    elif total <= -2:
        result['dir'] = '偏空'
        result['act'] = '可卖'
    else:
        result['dir'] = '震荡'
        result['act'] = '观望'
    
    return result

def main():
    results = {}
    
    for sym in INDEX_SYMBOLS:
        cfg = INDEX_SYMBOLS[sym]
        print(f"\n{'='*60}")
        print(f"蔡森技术分析 - {cfg['name']} ({sym})")
        print(f"{'='*60}")
        
        results[sym] = {'name': cfg['name'], 'tfs': {}}
        
        for tf, cfg_tf in [('daily', {'int':'1d','per':'1y'}), ('1h', {'int':'1h','per':'1mo'})]:
            print(f"\n【{tf.upper()}】")
            df = fetch_data(sym, cfg_tf['int'], cfg_tf['per'])
            if df is None or len(df) < 30:
                print(f"  数据不足")
                continue
            
            r = analyze(df, tf)
            results[sym]['tfs'][tf] = r
            
            print(f"  价格: {r['price']:,.2f}")
            print(f"  均线: {r['ma']}")
            print(f"  MACD: {r['macd']} ({r['macd_pos']})")
            print(f"  RSI: {r['rsi']:.1f} ({r['rsi_status']})")
            print(f"  KDJ: {r['kdj']}")
            print(f"  趋势: {r['trend']}")
            if r['near_r']: print(f"  阻力: {r['near_r']:,.2f}")
            if r['near_s']: print(f"  支撑: {r['near_s']:,.2f}")
            print(f"  >>> 得分:{r['total']} | {r['dir']} | {r['act']} <<<")
    
    # 交易计划
    print(f"\n{'='*60}")
    print("下周交易计划")
    print(f"{'='*60}")
    
    for sym, data in results.items():
        daily = data['tfs'].get('daily', {})
        if not daily:
            continue
        name = data['name']
        price = daily['price']
        score = daily['total']
        nr = daily.get('near_r')
        ns = daily.get('near_s')
        
        print(f"\n【{name}】")
        print(f"  当前: {price:,.2f} | 得分: {score}")
        
        if score >= 5:
            print(f"  方向: 做多")
            if ns:
                print(f"  入场: {ns*1.005:,.2f} - {ns*1.02:,.2f}")
                print(f"  止损: {ns*0.98:,.2f}")
            if nr:
                print(f"  目标: {nr*0.99:,.2f} -> {nr*1.02:,.2f}")
            print(f"  仓位: 标准(2-3%)")
        elif score >= 2:
            print(f"  方向: 偏多")
            if ns:
                print(f"  入场: {ns*1.01:,.2f} - {ns*1.03:,.2f}")
                print(f"  止损: {ns*0.97:,.2f}")
            print(f"  仓位: 轻仓(1-2%)")
        elif score <= -5:
            print(f"  方向: 做空")
            if nr:
                print(f"  入场: {nr*0.98:,.2f} - {nr*0.995:,.2f}")
                print(f"  止损: {nr*1.02:,.2f}")
            if ns:
                print(f"  目标: {ns*1.01:,.2f} -> {ns*0.98:,.2f}")
            print(f"  仓位: 标准(2-3%)")
        elif score <= -2:
            print(f"  方向: 偏空")
            if nr:
                print(f"  入场: {nr*0.985:,.2f} - {nr*1.0:,.2f}")
                print(f"  止损: {nr*1.03:,.2f}")
            print(f"  仓位: 轻仓(1-2%)")
        else:
            print(f"  方向: 震荡 - 观望")
            if ns and nr:
                print(f"  区间: {ns:,.2f} - {nr:,.2f}")
            print(f"  仓位: 极轻仓或观望")
    
    # 保存
    with open('/root/ideas/caishen/index_mtf.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
