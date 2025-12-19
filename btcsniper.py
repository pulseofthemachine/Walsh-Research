"""
SpinNet Sniper (Phase 11: HUD Upgrade)
--------------------------------------
Logic: Matches 'prepare_crypto.py' exactly.
       - Price (MOON/PUMP/FLAT/DUMP/CRASH)
       - Volume (HIGH/LOW/MID)
       - Momentum (OVERBOUGHT/OVERSOLD/NEUTRAL)
       - BBWP (COMPRESSED/NORMAL/EXPANDED)
"""
import os
import sys
import pickle
import torch
import requests
import pandas as pd
import numpy as np
import inspect
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Adjust path to find src.model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from src.model import SpinNetConfig, SpinNet

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
out_dir = 'experiments/out-crypto' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TARGET
symbol = 'BTCUSDT'     
context_window = 1024 
bbwp_lookback = 500    
num_simulations = 100   
forecast_horizon = 24  

# -----------------------------------------------------------------------------
# 1. LOAD MODEL
# -----------------------------------------------------------------------------
print(f"Loading Sniper from {out_dir}...")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')

if not os.path.exists(ckpt_path):
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

# Sanitize Config 
valid_keys = set(inspect.signature(SpinNetConfig).parameters)
sanitized_args = {k: v for k, v in model_args.items() if k in valid_keys}
gptconf = SpinNetConfig(**sanitized_args)
model = SpinNet(gptconf)

# Handle Prefix 
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# CUDA Optimization: 2x faster simulation, 6x faster inference
from src.model.cayley_dickson_cuda import optimize_for_inference
model = optimize_for_inference(model)
print("Applied CUDA optimization for inference")

model.eval()
model.to(device)

# Load Meta
meta_path = os.path.join('data', 'crypto', 'meta.pkl')
if not os.path.exists(meta_path):
    print("ERROR: meta.pkl not found. Run prepare_crypto.py first.")
    sys.exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# -----------------------------------------------------------------------------
# 2. FETCH DATA & CALC INDICATORS
# -----------------------------------------------------------------------------
print(f"Acquiring Target: {symbol}...")
try:
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1h&limit=1000"
    data = requests.get(url, timeout=10).json()
except Exception as e:
    print(f"Connection Failed: {e}")
    sys.exit(1)

if isinstance(data, dict) and 'code' in data:
    print(f"Binance API Error: {data}")
    sys.exit(1)

df = pd.DataFrame(data, columns=['time','open','high','low','close','vol','x','y','z','a','b','c'])
df = df[['time','open','high','low','close','vol']].astype(float)

# --- INDICATORS ---
df['pct'] = df['close'].pct_change()
df['rsi'] = RSIIndicator(df['close']).rsi()
df['vol_ma'] = df['vol'].rolling(20).mean()

# BBWP Calculation
bb = BollingerBands(close=df['close'], window=20, window_dev=2)
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
df['bb_mid'] = bb.bollinger_mavg()
df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, 1)
df['bbwp'] = df['bb_width'].rolling(window=bbwp_lookback).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
    raw=False
)

# Exclude forming candle
df_closed = df.iloc[:-1].copy()

# -----------------------------------------------------------------------------
# 3. TOKENIZATION
# -----------------------------------------------------------------------------
tokens = []
start_idx = len(df_closed) - context_window
if start_idx < 0: start_idx = 0

print(f"Analyzing {len(df_closed) - start_idx} candles...")

def get_token_str(row):
    # Re-usable logic for current state display
    if row['pct'] > 0.02: p_s = "MOON"
    elif row['pct'] > 0.0075: p_s = "PUMP"
    elif row['pct'] > -0.0075: p_s = "FLAT"
    elif row['pct'] > -0.02: p_s = "DUMP"
    else: p_s = "CRASH"
    
    if row['vol'] > 1.5 * row['vol_ma']: v_s = "HIGHVOL"
    elif row['vol'] < 0.5 * row['vol_ma']: v_s = "LOWVOL"
    else: v_s = "MIDVOL"
    
    if row['rsi'] > 70: m_s = "OVERBOUGHT"
    elif row['rsi'] < 30: m_s = "OVERSOLD"
    else: m_s = "NEUTRAL"
    
    if row['bbwp'] < 20: b_s = "COMPRESSED"
    elif row['bbwp'] < 80: b_s = "NORMAL"
    else: b_s = "EXPANDED"
    
    return f"{p_s}_{v_s}_{m_s}_{b_s}"

for i in range(start_idx, len(df_closed)):
    row = df_closed.iloc[i]
    if pd.isna(row['bbwp']): continue 
    
    t_str = get_token_str(row)
    
    if t_str in stoi:
        tokens.append(stoi[t_str])

if len(tokens) == 0:
    print("\n[CRITICAL ERROR] No tokens generated.")
    sys.exit(1)

print(f"Locked on. Input sequence length: {len(tokens)}")
x = (torch.tensor(tokens, dtype=torch.long, device=device)[None, ...])

# -----------------------------------------------------------------------------
# 4. EXECUTE ANALYSIS & HUD
# -----------------------------------------------------------------------------
print("\n>>> SPINNET MARKET FORECAST <<<")
print("=" * 60)

with torch.no_grad():
    # --- A: IMMEDIATE PROBS ---
    logits, _ = model(x)
    logits = logits[:, -1, :] 
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    
    print(f"IMMEDIATE TIMEFRAME (Next 1H):")
    for i in range(5):
        prob = top_probs[0][i].item()
        idx = top_indices[0][i].item()
        print(f"  {itos[idx]:<35} : {prob*100:.2f}%")

    # --- B: MACRO SIMULATION ---
    print(f"\nMACRO TIMEFRAME (Next {forecast_horizon}H Trend):")
    print(f"Running {num_simulations} simulations...")
    x_sim = x.repeat(num_simulations, 1)
    y_sim = model.generate(x_sim, max_new_tokens=forecast_horizon, temperature=0.9, top_k=20)
    futures = y_sim[:, -forecast_horizon:]
    
    bull_score, bear_score = 0, 0
    
    for i in range(num_simulations):
        sim_tokens = futures[i].tolist()
        for idx in sim_tokens:
            t_str = itos[idx]
            if "MOON" in t_str: bull_score += 3.0
            elif "PUMP" in t_str: bull_score += 1.0
            if "CRASH" in t_str: bear_score += 3.0
            elif "DUMP" in t_str: bear_score += 1.0

    total = bull_score + bear_score + 1e-9
    bull_pct = (bull_score / total) * 100
    bear_pct = (bear_score / total) * 100
    
    # --- C: CURRENT STATE HUD ---
    last = df_closed.iloc[-1]
    last_token = get_token_str(last)
    
    # Time formatting
    ts_obj = datetime.fromtimestamp(last['time'] / 1000)
    ts_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
    
    # Regime Logic
    bbwp = last['bbwp']
    if bbwp > 85: regime = "HIGH VOLATILITY / DANGER"
    elif bbwp < 15: regime = "SQUEEZE / LOADING"
    else: regime = "NORMAL / SUSTAINED"
    
    # Trend Logic
    if bull_pct > 65: trend_txt = "STRONG BULLISH"
    elif bull_pct > 50: trend_txt = "WEAK BULLISH"
    elif bear_pct > 65: trend_txt = "STRONG BEARISH"
    elif bear_pct > 50: trend_txt = "WEAK BEARISH"
    else: trend_txt = "NEUTRAL / CHOP"

    print(f"\n>>> CURRENT MARKET STATE (INPUT) <<<")
    print(f"Timestamp      : {ts_str} (Closed)")
    print(f"Close Price    : ${last['close']:.2f}")
    print(f"Recent Volume  : {last['vol'] / last['vol_ma']:.2f}x vs Avg")
    print(f"Geometric Token: {last_token}")
    print("-" * 60)
    print(f"REGIME : {regime}")
    print(f"TREND  : {trend_txt} ({max(bull_pct, bear_pct):.1f}% confidence)")
    print("=" * 60)
