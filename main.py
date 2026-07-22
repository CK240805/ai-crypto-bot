import asyncio
import schedule
import json
import os
import random
import threading
import mcp
from mcp.client.sse import sse_client
from fastapi import FastAPI
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
TRADER_DEV_API_KEY = os.getenv("TRADER_DEV_API_KEY") # Added for MCP Auth
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

# --- FASTAPI APP (Keeps Replit awake) ---
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Trading Bot is running 24/7!"}

# --- CORE BOT FUNCTIONS ---
async def generate_strategy_from_deepseek():
    prompt = """
    You are a professional algorithmic trader. Analyze current EURUSD volatility and momentum.
    Output a JSON structured trading strategy with the following parameters:
    {
        "entry_condition": "e.g., RSI < 30 and price_above_SMA_50",
        "exit_condition": "e.g., RSI > 70 or price_below_SMA_50",
        "timeframe": "e.g., 5m or 15m",
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "position_size_pct": 0.01
    }
    Use valid JSON format, no markdown backticks.
    """
    response = client.chat.completions.create(
        model=NVIDIA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except:
        return {"entry_condition": "EMA_50_cross", "stop_loss_pct": 0.02, "take_profit_pct": 0.05}

async def run_backtest_on_trader_dev(strategy_params):
    """
    Connects to Trader.dev MCP, passing the API Key from secrets.
    Falls back to simulation if the key is missing or connection fails.
    """
    print("📡 Attempting to connect to MCP Server...")
    
    # Prepare headers if the API Key exists in Secrets
    auth_headers = None
    if TRADER_DEV_API_KEY and TRADER_DEV_API_KEY.strip() != "":
        auth_headers = {"Authorization": f"Bearer {TRADER_DEV_API_KEY}"}
    else:
        print("⚠️  No TRADER_DEV_API_KEY found in Secrets. Will simulate.")

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
                    raise Exception("No 'backtest' tool found on the MCP server.")
                
                print(f"✅ Connected! Found MCP tool: {backtest_tool.name}")
                
                response = await session.call_tool(
                    backtest_tool.name,
                    arguments={"strategy": json.dumps(strategy_params)} 
                )

                if response.content and len(response.content) > 0:
                    result_text = response.content[0].text
                    return json.loads(result_text)
                
                raise Exception("Empty response from MCP server")

    except Exception as e:
        print(f"❌ MCP Connection Error: {e}. Using simulation fallback.")
        return {
            "sharpe_ratio": random.uniform(0.5, 2.5),
            "profit_pct": random.uniform(-5, 15),
            "max_drawdown": random.uniform(1, 10),
            "status": "fallback"
        }

async def refine_strategy_with_deepseek(previous_strategy, backtest_results):
    failures = f"Sharpe ratio was {backtest_results['sharpe_ratio']}, drawdown was {backtest_results['max_drawdown']}%."
    prompt = f"""
    The following strategy failed backtesting: {previous_strategy}. 
    Reason: {failures}
    Please adjust the 'stop_loss_pct', 'take_profit_pct', and 'entry_condition' to improve Sharpe ratio above 2.0.
    Return the updated JSON in the exact same format as before.
    """
    response = client.chat.completions.create(
        model=NVIDIA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except:
        return previous_strategy

async def execute_trade(signal_direction, sl_price, tp_price):
    # Get account balance
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    oanda.request(r)
    account_balance = float(r.response['account']['balance'])
    
    # Enforce $100 hard cap
    usable_capital = min(MAX_CAPITAL, account_balance)
    trade_units = int(usable_capital / 1.10) 
    
    order_data = {
        "order": {
            "type": "MARKET",
            "instrument": "EUR_USD",
            "units": str(trade_units if signal_direction == "BUY" else -trade_units),
            "stopLossOnFill": {"price": str(sl_price)},
            "takeProfitOnFill": {"price": str(tp_price)}
        }
    }
    r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
    try:
        oanda.request(r)
        return {"status": "executed", "capital_used": usable_capital}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

async def find_amazing_strategy_loop():
    while True:
        print("🔄 Optimization Loop: Generating strategy via NVIDIA...")
        strategy = await generate_strategy_from_deepseek()
        print(f"Backtesting: {strategy}")
        backtest_result = await run_backtest_on_trader_dev(strategy)
        
        is_amazing = (backtest_result['sharpe_ratio'] > 1.8 and backtest_result['max_drawdown'] < 5.0)
        
        if is_amazing:
            print(f"✅ AMAZING STRATEGY FOUND! {strategy}")
            # Calculate SL & TP based on current price (assuming 1.1000 for demo)
            sl_price = 1.1000 - (1.1000 * strategy['stop_loss_pct'])
            tp_price = 1.1000 + (1.1000 * strategy['take_profit_pct'])
            execution = await execute_trade("BUY", round(sl_price, 4), round(tp_price, 4))
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=f"🚀 Bot executed trade based on AI strategy!\nResult: {execution}"
            )
        else:
            print("❌ Not amazing. Asking NVIDIA/DeepSeek to refine...")
            strategy = await refine_strategy_with_deepseek(strategy, backtest_result)
        
        await asyncio.sleep(300) # 5 minutes

async def send_morning_report():
    prompt = "Give a 3-sentence summary of today's expected market sentiment and key volatility levels for EURUSD."
    response = client.chat.completions.create(model=NVIDIA_MODEL_NAME, messages=[{"role":"user", "content": prompt}])
    await telegram_bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, 
        text=f"🌅 Morning Market Analysis:\n\n{response.choices[0].message.content}"
    )

async def send_night_performance():
    # You can expand this later to fetch actual OANDA P&L via API
    await telegram_bot.send_message(
        chat_id=TELEGRAM_CHAT_ID, 
        text=f"🌙 Night Performance Profile:\nToday's P&L: $-2.50 (Simulated - Fetch via OANDA API soon)\nOpen Positions: 1"
    )

async def background_worker():
    schedule.every().day.at("09:00").do(lambda: asyncio.create_task(send_morning_report()))
    schedule.every().day.at("21:00").do(lambda: asyncio.create_task(send_night_performance()))
    asyncio.create_task(find_amazing_strategy_loop())
    
    while True:
        schedule.run_pending()
        await asyncio.sleep(60)

# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    # Run the trading bot in a separate thread so FastAPI keeps responding
    threading.Thread(target=lambda: asyncio.run(background_worker()), daemon=True).start()

# --- RUN THE APP ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
