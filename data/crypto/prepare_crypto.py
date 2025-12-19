import os
import requests
import pandas as pd
import numpy as np
import pickle
import time
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Top 10 High-Liquidity Assets (Mix of Layer 1s and Legacy)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 
    'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT'
]

INTERVAL = '1h'
LIMIT = 1000 
# Total Chunks per coin (120 chunks * 1000 hours = ~13 years, capped by existence)
TOTAL_CHUNKS = 60 
START_TIME = 1502942400000 # Aug 2017 (Earliest Binance Data)

# THE SEPARATOR TOKEN (Crucial for preventing context bleed)
SEPARATOR_TOKEN = "ASSET_SWITCH"

# -----------------------------------------------------------------------------
# 1. FETCH LOGIC
# -----------------------------------------------------------------------------
def fetch_data_for_symbol(symbol):
    print(f"\n>>> DOWNLOADING: {symbol} <<<")
    all_klines = []
    current_start = START_TIME
    
    for i in range(TOTAL_CHUNKS):
        # Using binance.us per your previous snippet (or use api.binance.com)
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={LIMIT}&startTime={current_start}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                # 400 error usually means symbol didn't exist yet, keep trying forward
                # print(f"  ...waiting for listing ({response.status_code})")
                current_start += (LIMIT * 3600000)
                continue
                
            data = response.json()
            if not data or len(data) == 0:
                break # End of data
                
            all_klines.extend(data)
            current_start = data[-1][0] + 3600000
            
            # Print progress every 10 chunks to keep terminal clean
            if i % 10 == 0:
                print(f"  Chunk {i}/{TOTAL_CHUNKS}: {len(all_klines)} candles total")
            
            time.sleep(0.1) # Fast but polite
            
        except Exception as e:
            print(f"  Fetch Error: {e}")
            break
            
    if len(all_klines) == 0:
        print(f"  WARNING: No data found for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=['time','open','high','low','close','vol','x','y','z','a','b','c'])
    df = df[['time','open','high','low','close','vol']].astype(float)
    return df

# -----------------------------------------------------------------------------
# 2. INDICATORS (BBWP + RSI)
# -----------------------------------------------------------------------------
def calculate_bbwp(df, bb_period=20, bbwp_lookback=500):
    if len(df) < bbwp_lookback: return df
    
    bb = BollingerBands(close=df['close'], window=bb_period, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    
    # Handle Zero Division for BBWP
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, 1)
    
    # Rolling Rank
    df['bbwp'] = df['bb_width'].rolling(window=bbwp_lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False
    )
    return df

# -----------------------------------------------------------------------------
# 3. TOKENIZER (The "Sniper" Logic)
# -----------------------------------------------------------------------------
def tokenize(df):
    if df.empty: return []
    
    # Indicators
    df['pct'] = df['close'].pct_change()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['vol_ma'] = df['vol'].rolling(20).mean()
    df = calculate_bbwp(df)
    
    tokens = []
    
    # Skip warmup period for indicators
    for i, row in df.iterrows():
        if i < 550: continue 
        
        # --- DIMENSION 1: PRICE ACTION (BTC Sniper Thresholds) ---
        if row['pct'] > 0.02: p_state = "MOON"
        elif row['pct'] > 0.0075: p_state = "PUMP"
        elif row['pct'] > -0.0075: p_state = "FLAT"
        elif row['pct'] > -0.02: p_state = "DUMP"
        else: p_state = "CRASH"
        
        # --- DIMENSION 2: VOLUME ---
        if row['vol'] > 1.5 * row['vol_ma']: v_state = "HIGHVOL"
        elif row['vol'] < 0.5 * row['vol_ma']: v_state = "LOWVOL"
        else: v_state = "MIDVOL"
        
        # --- DIMENSION 3: MOMENTUM ---
        if row['rsi'] > 70: m_state = "OVERBOUGHT"
        elif row['rsi'] < 30: m_state = "OVERSOLD"
        else: m_state = "NEUTRAL"
        
        # --- DIMENSION 4: BBWP (SQUEEZE) ---
        if pd.isna(row['bbwp']): continue
        
        if row['bbwp'] < 20: bbwp_state = "COMPRESSED" # Tighter threshold for squeeze
        elif row['bbwp'] < 80: bbwp_state = "NORMAL"
        else: bbwp_state = "EXPANDED"
        
        token = f"{p_state}_{v_state}_{m_state}_{bbwp_state}"
        tokens.append(token)
        
    return tokens

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION LOOP
# -----------------------------------------------------------------------------
all_tokens_combined = []

for symbol in SYMBOLS:
    df = fetch_data_for_symbol(symbol)
    if not df.empty:
        symbol_tokens = tokenize(df)
        print(f"  -> Generated {len(symbol_tokens)} tokens")
        
        # Add to main pile
        all_tokens_combined.extend(symbol_tokens)
        
        # INSERT SEPARATOR (Inject 10 times to flush the context window partially)
        # This tells the model "Space is resetting"
        all_tokens_combined.extend([SEPARATOR_TOKEN] * 5)

print("-" * 30)
print(f"TOTAL TOKENS: {len(all_tokens_combined)}")

# -----------------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------------
# Create Vocabulary
vocab = sorted(list(set(all_tokens_combined)))
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}

print(f"Vocab Size: {len(vocab)}")

with open('meta.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos}, f)

# Encode
ids = [stoi[t] for t in all_tokens_combined]
ids = np.array(ids, dtype=np.uint16)

# 90/10 Split
n = len(ids)
train_ids = ids[:int(n*0.9)]
val_ids = ids[int(n*0.9):]

train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

print("Dataset Saved. Ready for SpinNet.")
