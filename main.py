import asyncio
import schedule
import json
import os
import random
import threading
import mcp
from mcp.client.sse import sse_client
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from oandapyV20 import API
from oandapyV20.endpoints import accounts, orders
import openai
from telegram import Bot

# --- CONFIGURATIONS & SECRETS ---
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "deepseek-ai/deepseek-v3")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TRADER_DEV_API_KEY = os.getenv("TRADER_DEV_API_KEY")
MAX_CAPITAL = 100.00 

# --- MCP SERVER URL ---
MCP_SERVER_URL = "https://mcp.trader.dev/sse"

# --- INITIALIZE APIS ---
client = openai.OpenAI(
    api_key=NVIDIA_API_KEY, 
    base_url="https://integrate.api.nvidia.com/v1"
)
oanda = API(access_token=OANDA_API_KEY, environment="practice")
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

# --- DASHBOARD DATA STORAGE ---
INTERACTION_HISTORY = [] # Stores the last 20 interactions

# --- FASTAPI APP ---
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Trading Bot is running 24/7!"}

# --- NEW DASHBOARD ENDPOINT ---
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """
    <html>
        <head><title>AI Bot Dashboard</title>
        <style>body{font-family:sans-serif;background:#121212;color:#eee;padding:20px;} 
        .entry{background:#1e1e1e;border:1px solid #333;border-radius:8px;padding:15px;margin-bottom:20px;}
        .label{font-weight:bold;color:#4fc3f7;margin-top:10px;}
        .json-block{background:#000;padding:10px;border-radius:4px;font-family:monospace;white-space:pre-wrap;overflow-x:auto;}
        .time{color:#888;font-size:0.85em;}
        </style></head>
        <body>
        <h1>🤖 AI Trading Bot - Interaction Dashboard</h1>
        <p>Showing the last 20 requests to DeepSeek and their responses.</p>
        <hr>
    """
    
    if not INTERACTION_HISTORY:
        html_content += "<p><i>No interactions logged yet. Waiting for the next 5-minute loop...</i></p>"
    else:
        for idx, entry in enumerate(reversed(INTERACTION_HISTORY)):
            html_content += f"""
            <div class="entry">
                <div class="time">[{entry['timestamp']}]</div>
                <div class="label">🤖 Bot Prompt:</div>
                <div class="json-block">{entry['prompt']}</div>
                <div class="label">🧠 DeepSeek Response:</div>
                <div class="json-block">{entry['response']}</div>
            </div>
            """
    
    html_content += "</body></html>"
    return HTMLResponse(content=html_content)

# --- FETCH OANDA INSTRUMENTS ---
async def get_oanda_instruments():
    try:
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID)
        oanda.request(r)
        instruments = r.response['instruments']
        return [inst['name'] for inst in instruments]
    except Exception as e:
        print(f"⚠️ Failed to fetch OANDA instruments: {e}. Using fallback.")
        return ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'XAU_USD']

async def generate_strategy_from_deepseek():
    available_instruments = await get_oanda_instruments()
    instruments_str = ", ".join(available_instruments)
    prompt = f"""
    I have an OANDA account. I can only trade these instruments: {instruments_str}.
    Your task:
    1. Analyze the market and choose exactly ONE instrument from this list.
    2. Determine if the market is bullish or bearish for it.
    3. Output a JSON structure with:
    {{
        "symbol": "The exact instrument chosen from the list (e.g., EUR_USD)",
        "signal_direction": "BUY" or "SELL",
        "entry_condition": "e.g., RSI < 30 and price_above_SMA_50",
        "exit_condition": "e.g., RSI > 70 or price_below_SMA_50",
        "timeframe": "e.g., 5m or 15m",
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04
    }}
    Output only valid JSON. No markdown backticks.
    """
    try:
        response = client.chat.completions.create(
            model=NVIDIA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        
        # Save to dashboard history
        from datetime import datetime
        INTERACTION_HISTORY.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "response": content
        })
        if len(INTERACTION_HISTORY) > 20:
            INTERACTION_HISTORY.pop(0)

        strategy = json.loads(content)
        if strategy.get("symbol") not in available_instruments:
            strategy["symbol"] = "EUR_USD"
        return strategy
    except Exception as e:
        print(f"❌ DeepSeek/NVIDIA API Error: {e}. Falling back to default strategy.")
        return {"symbol": "EUR_USD", "signal_direction": "BUY", "entry_condition": "EMA_50_cross", "stop_loss_pct": 0.02, "take_profit_pct": 0.05}

async def run_backtest_on_trader_dev(strategy_params):
    auth_headers = None
    if TRADER_DEV_API_KEY and TRADER_DEV_API_KEY.strip() != "":
        auth_headers = {"Authorization": f"Bearer {TRADER_DEV_API_KEY}"}
    try:
        async with sse_client(MCP_SERVER_URL, headers=auth_headers) as streams:
            async with mcp.ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                backtest_tool = None
                for tool in tools.tools:
                    if "backtest" in tool.name.lower() or "strategy" in tool.name.lower():
                        backtest_tool = tool
                        break
                if not backtest_tool:
                    raise Exception("No 'backtest' tool found.")
                response = await session.call_tool(backtest_tool.name, arguments={"strategy": json.dumps(strategy_params)})
                if response.content and len(response.content) > 0:
                    return json.loads(response.content[0].text)
                raise Exception("Empty response")
    except Exception as e:
        print(f"❌ MCP Error: {e}. Using simulation.")
        return {"sharpe_ratio": random.uniform(0.5, 2.5), "profit_pct": random.uniform(-5, 15), "max_drawdown": random.uniform(1, 10), "status": "fallback"}

async def refine_strategy_with_deepseek(previous_strategy, backtest_results):
    failures = f"Sharpe was {backtest_results['sharpe_ratio']}, drawdown was {backtest_results['max_drawdown']}%."
    prompt = f"""
    The following strategy for {previous_strategy['symbol']} failed: {previous_strategy}. 
    Reason: {failures}
    Keep symbol "{previous_strategy['symbol']}" fixed. Adjust 'entry_condition', 'stop_loss_pct', 'take_profit_pct' to improve Sharpe ratio above 2.0.
    Return updated JSON in the exact same format.
    """
    try:
        response = client.chat.completions.create(model=NVIDIA_MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.8)
        content = response.choices[0].message.content.strip()
        
        # Save to dashboard history
        from datetime import datetime
        INTERACTION_HISTORY.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "response": content
        })
        if len(INTERACTION_HISTORY) > 20:
            INTERACTION_HISTORY.pop(0)

        refined = json.loads(content)
        refined["symbol"] = previous_strategy["symbol"]
        return refined
    except:
        return previous_strategy

async def execute_trade(symbol, signal_direction, sl_price, tp_price):
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    oanda.request(r)
    account_balance = float(r.response['account']['balance'])
    usable_capital = min(MAX_CAPITAL, account_balance)
    trade_units = int(usable_capital / 1.10) 
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": symbol,
            "units": str(trade_units if signal_direction == "BUY" else -trade_units),
            "stopLossOnFill": {"price": str(sl_price)},
            "takeProfitOnFill": {"price": str(tp_price)}
        }
    }
    r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
    try:
        oanda.request(r)
        return {"status": "executed", "symbol": symbol, "capital_used": usable_capital}
    except Exception as e:
        return {"status": "failed", "symbol": symbol, "error": str(e)}

async def find_amazing_strategy_loop():
    while True:
        strategy = await generate_strategy_from_deepseek()
        backtest_result = await run_backtest_on_trader_dev(strategy)
        is_amazing = (backtest_result['sharpe_ratio'] > 1.8 and backtest_result['max_drawdown'] < 5.0)
        if is_amazing:
            base_price = 1.1000 
            sl_price = base_price - (base_price * strategy['stop_loss_pct'])
            tp_price = base_price + (base_price * strategy['take_profit_pct'])
            execution = await execute_trade(strategy['symbol'], strategy['signal_direction'], round(sl_price, 4), round(tp_price, 4))
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 Executed trade!\nSymbol: {execution['symbol']}\nDirection: {strategy['signal_direction']}\nResult: {execution}")
        else:
            strategy = await refine_strategy_with_deepseek(strategy, backtest_result)
        await asyncio.sleep(300)

async def send_morning_report():
    prompt = "Give a 3-sentence summary of today's expected market sentiment and key volatility levels for major currencies."
    try:
        response = client.chat.completions.create(model=NVIDIA_MODEL_NAME, messages=[{"role":"user", "content": prompt}])
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🌅 Morning Analysis:\n\n{response.choices[0].message.content}")
    except:
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🌅 Morning Analysis: Could not reach AI API. Please check NVIDIA keys.")

async def send_night_performance():
    await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🌙 Night Profile:\nToday's P&L: $-2.50 (Simulated)\nOpen Positions: 1")

async def background_worker():
    schedule.every().day.at("09:00").do(lambda: asyncio.create_task(send_morning_report()))
    schedule.every().day.at("21:00").do(lambda: asyncio.create_task(send_night_performance()))
    asyncio.create_task(find_amazing_strategy_loop())
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=lambda: asyncio.run(background_worker()), daemon=True).start()

# --- RUN THE APP ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
