# %% [Setup]
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
from dotenv import load_dotenv
import os
import time as t
import pytz


# -----------------------
# Load API keys
# -----------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://data.alpaca.markets/v2"

SYMBOL = "SPY"
RSI_PERIOD = 14
SMA_PERIOD = 5
EMA_PERIOD = 5
BB_PERIOD = 20
OVERBOUGHT = 70
OVERSOLD = 30
WARMUP_MINUTES = 14

ET = pytz.timezone('US/Eastern')

# -----------------------
# Utility: get current ET reliably
# -----------------------
def now_et():
    return datetime.now(timezone.utc).astimezone(ET)

# -----------------------
# Data storage
# -----------------------
columns = ['timestamp', 'close', 'volume', 'RSI', 'SMA', 'EMA', 'BB_upper', 'BB_lower', 'Signal']
df = pd.DataFrame(columns=columns)
buy_x, buy_y = [], []
sell_x, sell_y = [], []

HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# -----------------------
# Market hours check
# -----------------------
def is_market_open():
    now = now_et()
    return time(9,30) <= now.time() <= time(16,0)

# -----------------------
# Wait until market open
# -----------------------
def wait_for_market_open():
    now = now_et()
    open_time = ET.localize(datetime.combine(now.date(), time(9,30)))
    preopen_time = ET.localize(datetime.combine(now.date(), time(9,0)))

    if now < preopen_time:
        wait_seconds = max(0, (preopen_time - now).total_seconds())
        print(f"⏳ Waiting until 9:00 AM ET (pre-open window starts)...")
        t.sleep(wait_seconds)
        now = now_et()

    if preopen_time <= now < open_time:
        print(f"⏳ Market opens at 9:30 AM ET. Waiting...")
        while now < open_time:
            now = now_et()
            remaining = max(0, (open_time - now).total_seconds())
            mins, secs = divmod(int(remaining), 60)
            print(f"⌛ Countdown: {mins:02d}:{secs:02d}", end="\r", flush=True)
            t.sleep(1)

    print("\n🚀 Market is open. Starting monitoring...")

# -----------------------
# Fetch latest 1-minute bar
# -----------------------
def get_latest_minute_bar(symbol):
    url = f"{BASE_URL}/stocks/{symbol}/bars/latest"
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        bar = resp.json().get("bar")
        if not bar:
            return None
        bar_time_et = datetime.fromisoformat(bar['t']).astimezone(ET)
        return {'timestamp': bar_time_et, 'close': bar['c'], 'volume': bar['v']}
    except Exception as e:
        print(f"❌ Failed to fetch bar: {e}")
        return None

# -----------------------
# Indicators
# -----------------------
#Wilder RSI
def calculate_rsi(closes, period: int = 14):
    """
    Wilder's RSI (matches TradingView, TOS, etc.)
    closes: list or numpy array of closing prices
    period: lookback period (default 14)
    """
    import numpy as np
    closes = np.array(closes, dtype=float)

    if len(closes) < period + 1:
        return None  # Not enough data yet

    deltas = np.diff(closes)
    seed = deltas[:period]

    # Separate gains and losses
    gains = seed[seed >= 0].sum() / period
    losses = -seed[seed < 0].sum() / period

    # Wilder smoothing starts here
    avg_gain = gains
    avg_loss = losses

    for delta in deltas[period:]:
        gain = max(delta, 0)
        loss = -min(delta, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0  # Prevent division by zero

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_orig_rsi(closes, period=RSI_PERIOD):
    if len(closes) < period+1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas>0,deltas,0)
    losses = np.where(deltas<0,-deltas,0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain/avg_loss if avg_loss!=0 else np.inf
    return 100 - (100/(1+rs))

def calculate_sma(closes, period=SMA_PERIOD):
    return np.mean(closes[-period:]) if len(closes)>=period else None

def calculate_ema(closes, period=EMA_PERIOD):
    if len(closes)<period: return None
    weights = np.exp(np.linspace(-1.,0.,period))
    weights/=weights.sum()
    return np.dot(closes[-period:],weights)

def calculate_bollinger(closes, period=BB_PERIOD):
    if len(closes)<period: return None, None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper = sma + 2*std
    lower = sma - 2*std
    return upper, lower

def get_signal(rsi):
    if rsi is None: return "WAIT"
    elif rsi>OVERBOUGHT: return "SELL"
    elif rsi<OVERSOLD: return "BUY"
    else: return "HOLD"

# -----------------------
# Append to CSV
# -----------------------
def log_to_csv(bar, rsi, sma, ema, bb_upper, bb_lower, signal):
    today_str = now_et().strftime("%m%d%Y")
    log_file = os.path.join(HISTORY_FOLDER, f"{today_str}.csv")
    log_df = pd.DataFrame([{
        'timestamp': bar['timestamp'],
        'close': bar['close'],
        'volume': bar['volume'],
        'RSI': np.nan if rsi is None else rsi,
        'SMA': np.nan if sma is None else sma,
        'EMA': np.nan if ema is None else ema,
        'BB_upper': np.nan if bb_upper is None else bb_upper,
        'BB_lower': np.nan if bb_lower is None else bb_lower,
        'Signal': signal
    }])
    if not os.path.isfile(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)

# %% [Load History if Exists]
today_str = now_et().strftime("%m%d%Y")
log_file = os.path.join(HISTORY_FOLDER, f"{today_str}.csv")
warmup_count = 0

if os.path.isfile(log_file):
    df = pd.read_csv(log_file, parse_dates=['timestamp'])
    print(f"📂 Loaded {len(df)} rows from history for {today_str}.")
    warmup_count = len(df)  # Continue from where you left off
    if warmup_count >= RSI_PERIOD:
        print("✅ Enough history for RSI, skipping warm-up.")
    else:
        print(f"⚠️ Only {warmup_count} records, will continue warm-up until {RSI_PERIOD}.")
else:
    df = pd.DataFrame(columns=columns)
    print("📂 No history for today, starting fresh.")

# %% [Main Loop]
now = now_et()
if is_market_open():
    print("🚀 Market is open! Starting monitoring...")
elif now.time() < time(9,30):
    wait_for_market_open()
else:
    print("⛔ Market is closed. Live update stopped.")
    raise SystemExit

print(f"🚀 Starting intraday monitoring for {SYMBOL}...")

while is_market_open():
    bar = get_latest_minute_bar(SYMBOL)
    if bar:
        df = pd.concat([df, pd.DataFrame([bar])], ignore_index=True)
        df = df.tail(max(RSI_PERIOD+1, SMA_PERIOD, EMA_PERIOD, BB_PERIOD))
        closes = df['close'].to_numpy()
        warmup_count += 1

        # --- Warm-up or normal calculation ---
        if warmup_count < WARMUP_MINUTES:
            print(f"[{bar['timestamp']}] Warm-up ({warmup_count}/{WARMUP_MINUTES}) - collecting data")
            signal = "WARMUP"
            rsi = None
            sma = calculate_sma(closes)
            ema = calculate_ema(closes)
            bb_upper, bb_lower = calculate_bollinger(closes)
        else:
            rsi = calculate_rsi(closes)
            sma = calculate_sma(closes)
            ema = calculate_ema(closes)
            bb_upper, bb_lower = calculate_bollinger(closes)
            signal = get_signal(rsi)

            if signal=="BUY":
                buy_x.append(bar['timestamp']); buy_y.append(bar['close'])
            elif signal=="SELL":
                sell_x.append(bar['timestamp']); sell_y.append(bar['close'])

        log_to_csv(bar, rsi, sma, ema, bb_upper, bb_lower, signal)

        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
        print(f"[{bar['timestamp']}] Close: {bar['close']:.2f}, RSI: {rsi_str}, Signal: {signal}")
    else:
        print("No new bar yet.")

    t.sleep(60)

print("⛔ Market closed. Live update stopped.")
