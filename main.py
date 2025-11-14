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
import uvicorn

# ============================
# CONFIG - leer de Secrets
# ============================
TOKEN = os.getenv("TOKEN")                # Telegram bot token
CHAT_ID = int(os.getenv("CHAT_ID"))      # tu chat id (int)
NEWS_API_KEY = os.getenv("NEWS_API_KEY") # opcional - NewsAPI.org (mejor); si no está, intentaremos GNews si GNEWS_API_KEY existe
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

bot = Bot(token=TOKEN)
app = FastAPI()

# Zona horaria Colombia
TZ = ZoneInfo("America/Bogota")

# Mapeos entre "ids" usadas en distintas APIs
COINBASE_SYMBOL = {"bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP"}
COINGECKO_ID = {"bitcoin": "bitcoin", "ethereum": "ethereum", "ripple": "ripple"}
SEND_ORDERBOOK_FALLBACK_ZERO = (0.0, 0.0, 0.0, 0.0)

# ============================
# UTIL - manejo seguros de APIs
# ============================
def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

# ============================
# 1) PRECIO: Coinbase -> fallback CoinGecko
# ============================
def get_coinbase_price(coin_id):  # coin_id = "bitcoin"/"ethereum"/"ripple"
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/ticker"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        if isinstance(data, dict) and "price" in data:
            return float(data["price"])
        # fallback
    except Exception as e:
        print("Coinbase price error:", e)

    # Fallback CoinGecko
    try:
        cg_id = COINGECKO_ID.get(coin_id, coin_id)
        cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        r2 = requests.get(cg_url, timeout=8).json()
        return float(r2[cg_id]["usd"])
    except Exception as e:
        print("CoinGecko fallback error:", e)
        return None

def get_coinbase_orderbook(coin_id):
    symbol = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    url = f"https://api.exchange.coinbase.com/products/{symbol}-USD/book?level=1"
    try:
        r = requests.get(url, timeout=8)
        data = safe_json(r)
        if isinstance(data, dict) and "bids" in data and "asks" in data:
            bid_price = float(data["bids"][0][0])
            bid_qty = float(data["bids"][0][1])
            ask_price = float(data["asks"][0][0])
            ask_qty = float(data["asks"][0][1])
            return bid_price, bid_qty, ask_price, ask_qty
    except Exception as e:
        print("Coinbase orderbook error:", e)

    # fallback neutral
    return SEND_ORDERBOOK_FALLBACK_ZERO

# ============================
# 2) HISTORICO (CoinGecko) -> para RSI/SMA + grafica
# ============================
def get_history_coingecko(coin_id, days=3):
    cg_id = COINGECKO_ID.get(coin_id, coin_id)
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        prices = r.get("prices", [])
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"] = df["price"].astype(float)
        return df
    except Exception as e:
        print("Coingecko history error:", e)
        return pd.DataFrame(columns=["timestamp", "price"])

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================
# 3) NOTICIAS (NewsAPI -> GNews fallback)
# ============================
def get_news_for_symbol(symbol, max_articles=3):
    # Primero NewsAPI.org si está configurado
    if NEWS_API_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{symbol} OR crypto OR cryptocurrency OR blockchain",
                "language": "en",
                "pageSize": max_articles,
                "sortBy": "publishedAt",
                "apiKey": NEWS_API_KEY,
            }
            r = requests.get(url, params=params, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            if articles:
                lines = []
                for a in articles:
                    title = a.get("title")
                    src = a.get("source", {}).get("name")
                    link = a.get("url")
                    lines.append(f"• {title} ({src})\n  {link}")
                return "📰 *Noticias relevantes:*\n" + "\n".join(lines)
        except Exception as e:
            print("NewsAPI error:", e)

    # Fallback GNews si existe key
    if GNEWS_API_KEY:
        try:
            url = f"https://gnews.io/api/v4/search"
            params = {"q": symbol, "lang": "en", "max": max_articles, "token": GNEWS_API_KEY}
            r = requests.get(url, params=params, timeout=8).json()
            articles = r.get("articles", [])[:max_articles]
            if articles:
                lines = []
                for a in articles:
                    title = a.get("title")
                    src = a.get("source", {}).get("name")
                    link = a.get("url")
                    lines.append(f"• {title} ({src})\n  {link}")
                return "📰 *Noticias relevantes:*\n" + "\n".join(lines)
        except Exception as e:
            print("GNews error:", e)

    return "📰 No hay noticias relevantes disponibles."

# ============================
# 4) Construcción gráfico PNG en memoria
# ============================
def create_chart_image(df, symbol_label):
    try:
        plt.figure(figsize=(8, 3.6))
        plt.plot(df["timestamp"], df["price"], label="Precio", linewidth=1.4)
        if "SMA20" in df.columns:
            plt.plot(df["timestamp"], df["SMA20"], label="SMA20", linewidth=1.2)
        plt.title(f"{symbol_label} - últimas 72h")
        plt.xlabel("Hora")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(alpha=0.3)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print("Chart creation error:", e)
        return None

# ============================
# 5) ANALISIS POR MONEDA (todo integrado)
# ============================
async def analyze_coin(coin_id):
    # coin_id: 'bitcoin' / 'ethereum' / 'ripple'
    label = COINBASE_SYMBOL.get(coin_id, coin_id).upper()
    now = datetime.datetime.now(TZ)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # price + orderbook
        price = get_coinbase_price(coin_id)
        bid_price, bid_qty, ask_price, ask_qty = get_coinbase_orderbook(coin_id)

        # history + indicators
        df = get_history_coingecko(coin_id, days=3)
        if not df.empty:
            df["SMA20"] = df["price"].rolling(20).mean()
            df["RSI14"] = calc_rsi(df["price"], 14)

            sma_val = df["SMA20"].iloc[-1] if not pd.isna(df["SMA20"].iloc[-1]) else None
            rsi_val = df["RSI14"].iloc[-1] if not pd.isna(df["RSI14"].iloc[-1]) else None
        else:
            sma_val, rsi_val = None, None

        # buy/sell suggestion (simple, educational)
        buy_price = None
        sell_price = None
        try:
            min24 = df["price"].min()
            max24 = df["price"].max()
            buy_price = round(min24 * 1.02, 2) if pd.notna(min24) else None
            sell_price = round(max24 * 0.98, 2) if pd.notna(max24) else None
        except Exception:
            pass

        # create chart
        chart_buf = create_chart_image(df, label) if not df.empty else None

        # news
        news_txt = get_news_for_symbol(label)

        # Interpretation (educational)
        rsi_text = "N/D"
        if rsi_val is not None:
            if rsi_val < 30:
                rsi_text = f"RSI {rsi_val:.2f} → posible sobreventa"
            elif rsi_val > 70:
                rsi_text = f"RSI {rsi_val:.2f} → posible sobrecompra"
            else:
                rsi_text = f"RSI {rsi_val:.2f} → neutro"

        sma_text = "N/D"
        if sma_val is not None and price is not None:
            sma_text = f"Precio {'por encima' if price > sma_val else 'por debajo'} de SMA20 (${sma_val:,.2f})"

        # Build message
        lines = [
            f"📊 *ANÁLISIS EDUCATIVO — {label}*",
            f"⏱ {timestamp_str} (hora Colombia)",
            "",
        ]
        if price is not None:
            lines.append(f"💵 *Precio actual:* ${price:,.2f}")
        else:
            lines.append("💵 *Precio actual:* N/D")

        if bid_price and ask_price and bid_price > 0 and ask_price > 0:
            lines.append(f"🟢 *Bid:* ${bid_price:,.2f} (qty: {bid_qty})")
            lines.append(f"🔴 *Ask:* ${ask_price:,.2f} (qty: {ask_qty})")
        else:
            lines.append("🟢 Bid / Ask: N/D")

        if buy_price and sell_price:
            lines.append(f"💡 *Sugerencia educativa:* Comprar ~ ${buy_price:,} — Vender ~ ${sell_price:,}")
        else:
            lines.append("💡 Sugerencia educativa: N/D")

        lines.append(f"📈 {sma_text}")
        lines.append(f"📉 {rsi_text}")
        lines.append("") 
        lines.append(news_txt)
        lines.append("")
        lines.append("_Este análisis es educativo, no es asesoramiento financiero._")

        message = "\n".join(lines)

        # send text
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

        # send chart if available
        if chart_buf:
            await bot.send_photo(chat_id=CHAT_ID, photo=chart_buf)
        else:
            # small text to indicate no chart
            await bot.send_message(chat_id=CHAT_ID, text="(No chart available)")

        print(f"Sent {label} analysis at {timestamp_str}")
    except Exception as e:
        print("❌ Error analyzing", coin_id, e)

# ============================
# 6) LOOP SCHEDULED - ejecuta entre 6:00 y 21:30 hora Colombia
# ============================
async def loop_crypto():
    # startup message once (catch exceptions silently)
    try:
        await bot.send_message(chat_id=CHAT_ID, text="🤖 Crypto Bot iniciado (educativo). Enviaré análisis cada hora entre 06:00 y 21:30 hora Colombia.")
    except Exception as e:
        print("Startup message error:", e)

    while True:
        now = datetime.datetime.now(TZ)
        # comprobar horario: entre 06:00 y 21:30 inclusive
        if 6 <= now.hour < 21 or (now.hour == 21 and now.minute <= 30):
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Within active window — analyzing")
            for coin in ["bitcoin", "ethereum", "ripple"]:
                await analyze_coin(coin)
        else:
            print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Outside active window — sleeping")
        # Esperar hasta la próxima hora exacta en Colombia
        # Para alinear con la hora, calculamos los segundos hasta el próximo minuto 0
        now = datetime.datetime.now(TZ)
        next_run = (now + datetime.timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        wait_seconds = (next_run - now).total_seconds()
        # Pero nunca esperar menos de 60s
        wait_seconds = max(wait_seconds, 60)
        await asyncio.sleep(wait_seconds)

# ============================
# 7) FastAPI para BetterStack (mantener vivo)
# ============================
@app.get("/")
def home():
    return {"status": "alive"}

@app.on_event("startup")
async def startup_event():
    # start background loop
    asyncio.create_task(loop_crypto())


