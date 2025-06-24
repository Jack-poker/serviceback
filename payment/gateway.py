import hashlib
import datetime
import json
import os
import requests
import asyncio
import sqlite3
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel
import time
from contextlib import contextmanager

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
class Config:
    BASE_URL = "https://www.intouchpay.co.rw/api"
    CALLBACK_URL = "http://localhost:8001/webhook/intouchpay/"
    DB_FILE = "transactions.db"
    REQUEST_TIMEOUT = 60  # As per documentation
    RETRY_ATTEMPTS = 5
    RETRY_BACKOFF_FACTOR = 2
    TRANSACTION_MONITOR_INTERVAL = 10

# Configure logging
logger = logging.getLogger("intouchpay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("intouchpay.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------------------
# Data Models
# ---------------------------------------
class PaymentRequest(BaseModel):
    phone_number: str
    amount: float
    reason: Optional[str] = None

class DepositRequest(BaseModel):
    phone_number: str
    amount: float
    withdraw_charge: Optional[int] = 0
    reason: Optional[str] = None
    sid: Optional[int] = None

class TransactionStatusRequest(BaseModel):
    request_transaction_id: str
    transaction_id: Optional[str] = None

# ---------------------------------------
# Database Management
# ---------------------------------------
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(Config.DB_FILE, timeout=10)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                phone TEXT,
                amount REAL,
                status TEXT,
                timestamp TEXT,
                delivered INTEGER DEFAULT 0,
                intouch_id TEXT,
                type TEXT,
                reason TEXT,
                withdraw_charge INTEGER,
                sid INTEGER
            )
        """)
        conn.commit()

def save_transaction(
    tx_id: str,
    phone: str,
    amount: float,
    status: str,
    timestamp: str,
    intouch_id: Optional[str] = None,
    tx_type: str = "payment",
    reason: Optional[str] = None,
    withdraw_charge: Optional[int] = None,
    sid: Optional[int] = None
):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO transactions 
            (tx_id, phone, amount, status, timestamp, delivered, intouch_id, type, reason, withdraw_charge, sid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tx_id, phone, amount, status, timestamp, 0, intouch_id, tx_type, reason, withdraw_charge, sid))
        conn.commit()

def update_transaction_status(tx_id: str, status: str, timestamp: str, intouch_id: Optional[str] = None):
    with get_db_connection() as conn:
        c = conn.cursor()
        update_query = "UPDATE transactions SET status = ?, timestamp = ?"
        params = [status, timestamp]
        if intouch_id:
            update_query += ", intouch_id = ?"
            params.append(intouch_id)
        update_query += " WHERE tx_id = ?"
        params.append(tx_id)
        c.execute(update_query, params)
        conn.commit()

# ---------------------------------------
# Helpers
# ---------------------------------------
def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

def generate_password(username: str, accountno: str, partnerpassword: str, timestamp: str) -> str:
    string = username + str(accountno) + partnerpassword + timestamp
    return hashlib.sha256(string.encode()).hexdigest()

def generate_transaction_id() -> str:
    file_path = "transaction_id.txt"
    try:
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("44555")
                return "44555"
        with open(file_path, "r+") as f:
            current_id = int(f.read().strip())
            new_id = current_id + 1
            f.seek(0)
            f.write(str(new_id))
            return str(new_id)
    except Exception as e:
        logger.error(f"Failed to generate transaction ID: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate transaction ID")

# ---------------------------------------
# IntouchPay API Client
# ---------------------------------------
class IntouchPayClient:
    def __init__(self, username: str, accountno: str, partnerpassword: str):
        self.username = username
        self.accountno = accountno
        self.partnerpassword = partnerpassword

    def _make_request(self, endpoint: str, data: Dict[str, Any], is_json: bool = False) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"} if is_json else {}
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    f"{Config.BASE_URL}/{endpoint}",
                    json=data if is_json else data,
                    headers=headers,
                    timeout=Config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                return response.json() if response.text else {}
            except requests.exceptions.RequestException as e:
                logger.error(f"Request to {endpoint} failed (attempt {attempt + 1}/{Config.RETRY_ATTEMPTS}): {str(e)}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
                time.sleep(Config.RETRY_BACKOFF_FACTOR ** attempt)
        return {}

    def get_balance(self) -> Dict[str, Any]:
        timestamp = get_timestamp()
        password = generate_password(self.username, self.accountno, self.partnerpassword, timestamp)
        data = {
            "username": self.username,
            "timestamp": timestamp,
            "password": password,
            "accountno": self.accountno
        }
        return self._make_request("getbalance/", data)

    def request_payment(self, request: PaymentRequest) -> Dict[str, Any]:
        timestamp = get_timestamp()
        password = generate_password(self.username, self.accountno, self.partnerpassword, timestamp)
        tx_id = generate_transaction_id()
        phone = f"25{request.phone_number}"

        data = {
            "username": self.username,
            "timestamp": timestamp,
            "amount": request.amount,
            "password": password,
            "mobilephone": phone,
            "requesttransactionid": tx_id,
            "callbackurl": Config.CALLBACK_URL,
            "accountno": self.accountno
        }

        try:
            response = self._make_request("requestpayment/", data)
            status = "Pending" if response.get("success", False) else "Failed"
            intouch_id = response.get("transactionid")
            save_transaction(
                tx_id=tx_id,
                phone=phone,
                amount=request.amount,
                status=status,
                timestamp=timestamp,
                intouch_id=intouch_id,
                tx_type="payment",
                reason=request.reason
            )
            return response
        except Exception as e:
            logger.error(f"Payment request failed: {str(e)}", exc_info=True)
            save_transaction(
                tx_id=tx_id,
                phone=phone,
                amount=request.amount,
                status="Failed",
                timestamp=timestamp,
                tx_type="payment",
                reason=request.reason
            )
            raise HTTPException(status_code=500, detail=f"Payment request failed: {str(e)}")

    def request_deposit(self, request: DepositRequest) -> Dict[str, Any]:
        timestamp = get_timestamp()
        password = generate_password(self.username, self.accountno, self.partnerpassword, timestamp)
        tx_id = generate_transaction_id()
        phone = f"25{request.phone_number}"

        data = {
            "username": self.username,
            "timestamp": timestamp,
            "amount": request.amount,
            "password": password,
            "mobilephone": phone,
            "requesttransactionid": tx_id,
            "accountno": self.accountno,
            "withdrawcharge": request.withdraw_charge,
            "reason": request.reason,
            "sid": request.sid
        }

        try:
            response = self._make_request("requestdeposit/", data)
            status = "Successful" if response.get("success", False) else "Failed"
            intouch_id = response.get("referenceid")
            save_transaction(
                tx_id=tx_id,
                phone=phone,
                amount=request.amount,
                status=status,
                timestamp=timestamp,
                intouch_id=intouch_id,
                tx_type="deposit",
                reason=request.reason,
                withdraw_charge=request.withdraw_charge,
                sid=request.sid
            )
            return response
        except Exception as e:
            logger.error(f"Deposit request failed: {str(e)}", exc_info=True)
            save_transaction(
                tx_id=tx_id,
                phone=phone,
                amount=request.amount,
                status="Failed",
                timestamp=timestamp,
                tx_type="deposit",
                reason=request.reason,
                withdraw_charge=request.withdraw_charge,
                sid=request.sid
            )
            raise HTTPException(status_code=500, detail=f"Deposit request failed: {str(e)}")

    def get_transaction_status(self, request: TransactionStatusRequest) -> Dict[str, Any]:
        timestamp = get_timestamp()
        password = generate_password(self.username, self.accountno, self.partnerpassword, timestamp)
        data = {
            "username": self.username,
            "timestamp": timestamp,
            "requesttransactionid": request.request_transaction_id,
            "password": password
        }
        if request.transaction_id:
            data["transactionid"] = request.transaction_id

        try:
            response = self._make_request("gettransactionstatus/", data, is_json=True)
            if response.get("success", False):
                update_transaction_status(
                    tx_id=request.request_transaction_id,
                    status=response.get("status", "Unknown"),
                    timestamp=timestamp,
                    intouch_id=request.transaction_id
                )
            return response
        except Exception as e:
            logger.error(f"Transaction status check failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Transaction status check failed: {str(e)}")

# ---------------------------------------
# FastAPI App
# ---------------------------------------
def create_app(username: str = "", accountno: str = "", partnerpassword: str = "") -> FastAPI:
    app = FastAPI()
    client = IntouchPayClient(username, accountno, partnerpassword)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # WebSocket clients
    clients = []

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        clients.append(websocket)
        try:
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            if websocket in clients:
                clients.remove(websocket)

    @app.post("/payment")
    async def make_payment(request: PaymentRequest):
        return client.request_payment(request)

    @app.post("/deposit")
    async def make_deposit(request: DepositRequest):
        return client.request_deposit(request)

    @app.post("/transaction-status")
    async def check_transaction_status(request: TransactionStatusRequest):
        return client.get_transaction_status(request)

    @app.get("/balance")
    async def get_balance():
        return client.get_balance()

    @app.post("/webhook")
    async def webhook(data: Dict[str, Any]):
        payload = data 
        if not isinstance(payload, dict):
            payload = data.get("jsonpayload", {})
        try:
            tx_id = payload.get("requesttransactionid")
            status = payload.get("status")
            intouch_id = payload.get("transactionid")
            timestamp = get_timestamp()
            
            if tx_id and status:
                update_transaction_status(tx_id, status, timestamp, intouch_id)
                event = {
                    "event": "payment_status",
                    "tx_id": tx_id,
                    "status": status,
                    "intouch_id": intouch_id,
                    "timestamp": timestamp
                }
                for client in clients:
                    try:
                        await client.send_text(json.dumps(event))
                    except:
                        clients.remove(client)
            return {"message": "success", "success": True, "request_id": tx_id}
        except Exception as e:
            logger.error(f"Webhook processing error: {str(e)}")
            raise HTTPException(status_code=500, detail="Webhook processing error")

    async def transaction_monitor():
        while True:
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT tx_id, phone, amount, status, intouch_id, type FROM transactions WHERE status = 'Pending'")
                    rows = c.fetchall()

                    for tx_id, phone, amount, _, intouch_id, tx_type in rows:
                        try:
                            request = TransactionStatusRequest(
                                request_transaction_id=tx_id,
                                transaction_id=intouch_id
                            )
                            result = client.get_transaction_status(request)

                            # Print emoji status for each transaction
                            if result.get("success", False):
                                status_text = result.get("status", "").lower()
                                emoji = "✅" if status_text == "success" else "❌" if status_text == "failed" else "⏳"
                                print(f"{emoji} [TX_ID: {tx_id}] Status: {status_text.upper()}")

                                if status_text != "pending":
                                    update_transaction_status(
                                        tx_id=tx_id,
                                        status=result["status"],
                                        timestamp=get_timestamp(),
                                        intouch_id=intouch_id
                                    )
                                    payload = json.dumps({
                                        "event": "payment_status",
                                        "tx_id": tx_id,
                                        "phone": phone,
                                        "amount": amount,
                                        "status": result["status"],
                                        "timestamp": get_timestamp(),
                                        "type": tx_type
                                    })
                                    for ws_client in clients[:]:
                                        try:
                                            await ws_client.send_text(payload)
                                        except:
                                            clients.remove(ws_client)
                            else:
                                print(f"⚠️ [TX_ID: {tx_id}] No success field or result error")

                        except Exception as e:
                            logger.error(f"Transaction monitor error for tx_id {tx_id}: {str(e)}")

                    conn.commit()
            except Exception as e:
                logger.error(f"Transaction monitor loop error: {str(e)}")

            await asyncio.sleep(Config.TRANSACTION_MONITOR_INTERVAL)


    @app.on_event("startup")
    async def startup_event():
        init_db()
        asyncio.create_task(transaction_monitor())

    return app

# ---------------------------------------
# CLI
# ---------------------------------------
def start_cli(username: str, accountno: str, partnerpassword: str):
    client = IntouchPayClient(username, accountno, partnerpassword)
    while True:
        print("\n1. Check Balance\n2. Request Payment\n3. Request Deposit\n4. Check Transaction Status\n5. Exit")
        choice = input("Select option (1-5): ")

        try:
            if choice == "1":
                print(json.dumps(client.get_balance(), indent=2))
            elif choice == "2":
                phone = input("Phone number (no +): ")
                amount = float(input("Amount: "))
                reason = input("Reason (optional, press enter to skip): ") or None
                request = PaymentRequest(phone_number=phone, amount=amount, reason=reason)
                print(json.dumps(client.request_payment(request), indent=2))
            elif choice == "3":
                phone = input("Phone number (no +): ")
                amount = float(input("Amount: "))
                withdraw_charge = int(input("Withdraw charge (0 or 1, optional): ") or 0)
                reason = input("Reason (optional, press enter to skip): ") or None
                sid = input("Service ID (optional, press enter to skip): ")
                sid = int(sid) if sid else None
                request = DepositRequest(
                    phone_number=phone,
                    amount=amount,
                    withdraw_charge=withdraw_charge,
                    reason=reason,
                    sid=sid
                )
                print(json.dumps(client.request_deposit(request), indent=2))
            elif choice == "4":
                tx_id = input("Request Transaction ID: ")
                intouch_id = input("Intouch Transaction ID (optional, press enter to skip): ") or None
                request = TransactionStatusRequest(
                    request_transaction_id=tx_id,
                    transaction_id=intouch_id
                )
                print(json.dumps(client.get_transaction_status(request), indent=2))
            elif choice == "5":
                break
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example credentials (should be set via environment variables in production)
    username = os.getenv("INTOUCH_USERNAME", "")
    accountno = os.getenv("INTOUCH_ACCOUNTNO", "")
    partnerpassword = os.getenv("INTOUCH_PARTNERPASSWORD", "")
    
    Thread(target=lambda: uvicorn.run(
        create_app(username, accountno, partnerpassword),
        host="0.0.0.0",
        port=8000
    ), daemon=True).start()
    start_cli(username, accountno, partnerpassword)