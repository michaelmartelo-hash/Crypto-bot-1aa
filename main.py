# main.py
import os
import asyncio
import datetime
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from telegram import Bot
from fastapi import FastAPI

# ============================
# CONFIG
# ============================
TOKEN = os.getenv("TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

bot = Bot(token=TOKEN)
app = FastAPI()
TZ = ZoneInfo("America/Bogota")

COINBASE_SYMBOL = {"bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP"}
COINGECKO_ID = {"bitcoin": "bitcoin", "ethereum": "ethereum", "ripple": "ripple"}
SEND_ORDERBOOK_FALLBACK_ZERO = (0.0, 0.0, 0.0, 0.0)

# ============================
# UTIL
# ============================
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

def get_coinbase_price(coin_id):
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/ticker"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        if "price" in data:
            return float(data["price"])
    except Exception:
        pass
    # fallback CoinGecko
    try:
        cg_id = COINGECKO_ID.get(coin_id, coin_id)
        r2 = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd", timeout=8).json()
        return float(r2[cg_id]["usd"])
    except Exception:
        return None

def get_coinbase_orderbook(coin_id):
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/book?level=1"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        bid_price = float(data["bids"][0][0])
        bid_qty = float(data["bids"][0][1])
        ask_price = float(data["asks"][0][0])
        ask_qty = float(data["asks"][0][1])
        return bid_price, bid_qty, ask_price, ask_qty
    except Exception:
        return SEND_ORDERBOOK_FALLBACK_ZERO

def get_history_coingecko(coin_id, days=3):
    cg_id = COINGECKO_ID.get(coin_id, coin_id)
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart",
                         params={"vs_currency":"usd", "days":str(days)}, timeout=10).json()
        prices = r.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp","price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"] = df["price"].astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp","price"])

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def get_news_for_symbol(symbol, max_articles=3):
    if NEWS_API_KEY:
        try:
            r = requests.get("https://newsapi.org/v2/everything",
                             params={"q":f"{symbol} OR crypto OR cryptocurrency OR blockchain",
                                     "language":"en", "pageSize":max_articles, "sortBy":"publishedAt",
                                     "apiKey":NEWS_API_KEY}, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            return "📰 *Noticias relevantes:*\n" + "\n".join([f"• {a.get('title')} ({a.get('source',{}).get('name')})\n  {a.get('url')}" for a in articles]) if articles else ""
        except Exception:
            pass
    if GNEWS_API_KEY:
        try:
            r = requests.get("https://gnews.io/api/v4/search",
                             params={"q":symbol,"lang":"en","max":max_articles,"token":GNEWS_API_KEY}, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            return "📰 *Noticias relevantes:*\n" + "\n".join([f"• {a.get('title')} ({a.get('source',{}).get('name')})\n  {a.get('url')}" for a in articles]) if articles else ""
        except Exception:
            pass
    return "📰 No hay noticias relevantes disponibles."

# ============================
# Chart with SMA + RSI
# ============================
def create_chart_image(df, symbol_label):
    try:
        plt.figure(figsize=(8,4))
        plt.plot(df["timestamp"], df["price"], label="Precio", color="blue", linewidth=1.5)
        if "SMA20" in df.columns:
            plt.plot(df["timestamp"], df["SMA20"], label="SMA20 (media 20 períodos)", color="orange", linewidth=1.2)
        if "RSI14" in df.columns:
            # RSI subplot
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(df["timestamp"], df["RSI14"], label="RSI14", color="green", linestyle="--", alpha=0.5)
            ax2.axhline(70, color="red", linestyle=":")  # sobrecompra
            ax2.axhline(30, color="purple", linestyle=":")  # sobreventa
            ax2.set_ylabel("RSI14")
        plt.title(f"{symbol_label} - últimas 72h")
        plt.xlabel("Hora")
        plt.ylabel("USD")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf
    except Exception:
        return None

# ============================
# ANALYSIS
# ============================
async def analyze_coin(coin_id):
    label = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    now = datetime.datetime.now(TZ)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        price = get_coinbase_price(coin_id)
        bid_price, bid_qty, ask_price, ask_qty = get_coinbase_orderbook(coin_id)
        df = get_history_coingecko(coin_id, days=3)

        if not df.empty:
            df["SMA20"] = df["price"].rolling(20).mean()
            df["RSI14"] = calc_rsi(df["price"], 14)
            sma_val = df["SMA20"].iloc[-1]
            rsi_val = df["RSI14"].iloc[-1]

            # Buy/sell levels based on min/max 24h
            min24 = df["price"].min()
            max24 = df["price"].max()
            buy_price = round(min24 * 1.02, 2)
            sell_price = round(max24 * 0.98, 2)

            # Trend interpretation simplified
            trend = "Neutra"
            if price > sma_val:
                trend = "Alcista"
            elif price < sma_val:
                trend = "Bajista"

            rsi_status = "Neutral"
            if rsi_val < 30:
                rsi_status = "Sobreventa"
            elif rsi_val > 70:
                rsi_status = "Sobrecompra"

        else:
            sma_val, rsi_val, buy_price, sell_price, trend, rsi_status = None, None, None, None, "N/D", "N/D"

        chart_buf = create_chart_image(df, label)
        news_txt = get_news_for_symbol(label)

        # Build message
        lines = [
            f"📊 *ANÁLISIS EDUCATIVO — {label}*",
            f"⏱ {timestamp_str} (hora Colombia)",
            "",
            f"💵 *Precio actual:* ${price:,.2f}" if price else "💵 *Precio actual:* N/D",
            f"🟢 *Bid:* ${bid_price:,.2f} (qty: {bid_qty})" if bid_price else "",
            f"🔴 *Ask:* ${ask_price:,.2f} (qty: {ask_qty})" if ask_price else "",
            f"💡 *Sugerencia educativa:* Comprar ~ ${buy_price:,} — Vender ~ ${sell_price:,}" if buy_price and sell_price else "",
            f"📈 Tendencia aproximada: {trend}",
            f"📉 Estado RSI: {rsi_status}",
            "",
            news_txt,
            "",
            "_Este análisis es educativo, no es asesoramiento financiero._"
        ]

        message = "\n".join([l for l in lines if l])

        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")
        if chart_buf:
            bot.send_photo(chat_id=CHAT_ID, photo=chart_buf)
        else:
            bot.send_message(chat_id=CHAT_ID, text="(No chart available)")

        print(f"✅ Enviado análisis de {label} a las {timestamp_str}")
    except Exception as e:
        print("❌ Error analyzing", coin_id, e)

# ============================
# LOOP
# ============================
async def loop_crypto():
    try:
        bot.send_message(chat_id=CHAT_ID, text="🤖 Crypto Bot iniciado (educativo). Enviaré análisis cada hora entre 06:00 y 21:30 hora Colombia.")
    except Exception:
        pass

    while True:
        now = datetime.datetime.now(TZ)
        if 6 <= now.hour < 21 or (now.hour == 21 and now.minute <= 30):
            for coin in ["bitcoin","ethereum","ripple"]:
                await analyze_coin(coin)
        next_run = (now + datetime.timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait_seconds = max((next_run - now).total_seconds(), 60)
        await asyncio.sleep(wait_seconds)

# ============================
# FastAPI
# ============================
@app.get("/")
def home():
    return {"status":"alive"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(loop_crypto())
