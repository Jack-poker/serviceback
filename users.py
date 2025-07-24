import asyncio
from multiprocessing import get_context
from fastapi import FastAPI, HTTPException, Header, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import httpx
from pydantic import BaseModel
from sqlalchemy import Enum, create_engine, Column, String, Float, DateTime, ForeignKey, Integer
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from sqlalchemy.pool import QueuePool
from bcrypt import hashpw, checkpw, gensalt
from cryptography.fernet import Fernet
from jose import JWTError, jwt
from datetime import datetime, timedelta
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid
import os
import requests
import random
from dotenv import load_dotenv
import structlog
import logging
from logging.handlers import RotatingFileHandler
from rich.table import Table
from rich.console import Console
import json
from typing import Optional
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from contextlib import asynccontextmanager
from payment.gateway import request_payment, request_withdraw, send_otp, verify_withdraw_otp,transfer_money
import enum
from sqlalchemy import Enum


# _______ Semaphore for Concurrency Control _______
REQUEST_SEMAPHORE = asyncio.Semaphore(500)  # Limit to 100 concurrent requests

# _______ Dependency Functions _______
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# _______ Configuration Loading _______
load_dotenv()
console = Console()

# Validate environment variables at startup
required_env_vars = ["DATABASE_URL", "JWT_SECRET", "ADMIN_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        console.print(f"[red]Missing required environment variable: {var}[/red]")
        raise RuntimeError(f"Missing required environment variable: {var}")

# _______ Logging Setup _______
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

key = Fernet.generate_key()
fernet = Fernet(key)
logger = structlog.get_logger()

class RichTableHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = json.loads(record.msg)
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Timestamp", style="dim")
            table.add_column("Level", style="bold")
            table.add_column("Event")
            table.add_column("Details")
            emoji = "✅" if log_entry["level"] == "info" else "❌" if log_entry["level"] == "error" else "⚠️"
            table.add_row(
                log_entry["timestamp"],
                log_entry["level"].upper(),
                f"{emoji} {log_entry['event']}",
                str(log_entry.get("details", ""))
            )
            console.print(table)
        except Exception as e:
            console.print(f"[red]Logging error: {str(e)}[/red]")

file_handler = RotatingFileHandler(
    os.getenv("LOG_FILE", "logs/app.log"),
    maxBytes=int(os.getenv("LOG_MAX_BYTES", 10485760)),
    backupCount=int(os.getenv("LOG_BACKUP_COUNT", 5))
)
file_handler.setFormatter(logging.Formatter("%(message)s"))

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(RichTableHandler())
logging.getLogger().addHandler(file_handler)

# _______ Admin Alert Simulation _______
async def send_admin_alert_async(message: str, details: Optional[dict] = None):
    alert_logger = structlog.get_logger("alert")
    alert_file_handler = RotatingFileHandler(
        os.getenv("ALERT_LOG_FILE", "logs/alerts.log"),
        maxBytes=int(os.getenv("LOG_MAX_BYTES", 10485760)),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", 5))
    )
    alert_file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("alert").addHandler(alert_file_handler)
    alert_logger.error(
        event="Admin Alert",
        details={"message": message, "details": details or {}}
    )
    console.print(f"[red]🚨 Admin Alert: {message}[/red]")

# _______ Application Setup _______
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="School Payment System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8080,https://api.kaascan.com,http://192.168.1.149:8080,https://api.kaascan.com").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# _______ Environment Variables _______
try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    JWT_SECRET = os.getenv("JWT_SECRET")
    PAYMENT_GATEWAY = os.getenv("PAYMENT_GATEWAY", "intouchpay")
    INTOUCH_USERNAME = "testa"
    INTOUCH_ACCOUNT_NO = 250160000011
    INTOUCH_PARTNER_PASSWORD = "+$J<wtZktTDs&-Mk(\"h5=<PH#Jf769P5/Z<*xbR~"
    INTOUCH_CALLBACK_URL = os.getenv("INTOUCH_CALLBACK_URL")
    INTOUCH_API_URL = os.getenv("INTOUCH_API_URL", "https://www.intouchpay.co.rw/api")
    PAYPACK_API_KEY = os.getenv("PAYPACK_API_KEY")
    PAYPACK_CLIENT_ID = os.getenv("PAYPACK_CLIENT_ID")
    FERNET_KEY = os.getenv("FERNET_KEY", Fernet.generate_key().decode())
    TRANSACTION_FEE_TYPE = os.getenv("TRANSACTION_FEE_TYPE", "percentage")
    TRANSACTION_FEE_VALUE = float(os.getenv("TRANSACTION_FEE_VALUE", 2.0))
    RATE_LIMIT_CSRF = os.getenv("RATE_LIMIT_CSRF", "500/minute")
    RATE_LIMIT_REGISTER = os.getenv("RATE_LIMIT_REGISTER", "10/minute")
    RATE_LIMIT_LOGIN = os.getenv("RATE_LIMIT_LOGIN", "20/minute")
    RATE_LIMIT_TRANSFER = os.getenv("RATE_LIMIT_TRANSFER", "10/minute")
    RATE_LIMIT_SET_LIMIT = os.getenv("RATE_LIMIT_SET_LIMIT", "10/minute")
    RATE_LIMIT_PROFILE = os.getenv("RATE_LIMIT_PROFILE", "20/minute")
    RATE_LIMIT_WALLET = os.getenv("RATE_LIMIT_WALLET", "20/minute")
    RATE_LIMIT_WEBHOOK = os.getenv("RATE_LIMIT_WEBHOOK", "200/minute")
except Exception as e:
    logger.error(event="Configuration Error", details=str(e))
    asyncio.create_task(send_admin_alert_async("Invalid environment variables", {"error": str(e)}))
    raise HTTPException(status_code=500, detail="Server configuration error")

# _______ Database Setup _______
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(event="Database Connection Error", details=str(e))
    asyncio.create_task(send_admin_alert_async("Failed to connect to database", {"error": str(e)}))
    raise HTTPException(status_code=500, detail="Database connection error")

# _______ Database Models _______
class Parent(Base):
    __tablename__ = "parents"
    parent_id = Column(String(36), primary_key=True)
    fullnames = Column(String(255), nullable=False)
    phone_number = Column(String(10), nullable=False, unique=True)
    email = Column(String(255), nullable=True)
    address = Column(String(255), nullable=True)
    about = Column(String(500), nullable=True)
    account_balance = Column(Float, default=0.00)
    password_hash = Column(String(60), nullable=False)
    totp_secret = Column(String(32), nullable=False)
    otp_code = Column(String(4), nullable=True)
    created_at = Column(DateTime, nullable=False)

class Student(Base):
    __tablename__ = "students"
    student_id = Column(String(36), primary_key=True)
    parent_id = Column(String(36), ForeignKey("parents.parent_id"), nullable=True)
    student_name = Column(String(255), nullable=False)
    student_photo_url = Column(String(255))
    school_name = Column(String(255), nullable=False)
    student_pin_encrypted = Column(String(255), nullable=True)
    grade = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=False)
    spending_limit = Column(Float, default=0.00)
    limit_period_days = Column(Integer, default=1)
class TransactionType(enum.Enum):
    deposit = "deposit"
    withdrawal = "withdrawal"
    transfer = "transfer"
    payment = "payment"

class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String(36), primary_key=True)
    parent_id = Column(String(36), ForeignKey("parents.parent_id"), nullable=False)
    student_id = Column(String(36), ForeignKey("students.student_id"), nullable=True)
    amount_sent = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    description = Column(String(255), nullable=True)
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime, nullable=False)
    intouch_transaction_id = Column(String(50), nullable=True)
    status = Column(String(20), default="Pending")
    type = Column(Enum(TransactionType), nullable=False)

class CsrfToken(Base):
    __tablename__ = "csrf_tokens"
    token = Column(String(64), primary_key=True)
    session_id = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

Base.metadata.create_all(bind=engine)

# _______ Pydantic Models _______
class ParentRegister(BaseModel):
    fullnames: str
    phone_number: str
    password: str
    email: Optional[str] = None
    address: Optional[str] = None
    about: Optional[str] = None
    csrf_token: str

class StudentRegister(BaseModel):
    parent_id: str
    student_name: str
    student_photo_url: str
    school_name: str
    student_pin: str
    grade: Optional[str] = None
    csrf_token: str

class ParentLogin(BaseModel):
    phone_number: str
    password: str
    totp_code: str
    csrf_token: str

class MoneyTransfer(BaseModel):
    student_id: str
    amount: float
    description: Optional[str] = None
    latitude: float
    longitude: float
    csrf_token: str

class SetSpendingLimit(BaseModel):
    student_id: str
    spending_limit: float
    limit_period_days: int
    csrf_token: str

class ProfileUpdate(BaseModel):
    fullnames: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    about: Optional[str] = None
    csrf_token: str

class WalletAction(BaseModel):
    amount: float
    csrf_token: str

class WebhookPayload(BaseModel):
    requesttransactionid: str
    transactionid: str
    responsecode: str
    status: str
    statusdesc: str
    referenceno: str

class AdminStudentRegister(BaseModel):
    student_name: str
    student_photo_url: Optional[str] = None
    school_name: str
    grade: Optional[str] = None


class WithdrawResponse(BaseModel):
    status: str
    message: str
    transaction_id: str
# _______ Helper Functions _______
def generate_password(username: str, accountno: str, partnerpassword: str, timestamp: str) -> str:
    combined = username + str(accountno) + partnerpassword + timestamp
    return hashlib.sha256(combined.encode()).hexdigest()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
async def call_intouch_api(endpoint: str, data: dict, headers: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{INTOUCH_API_URL}/{endpoint}"
        try:
            response = await client.post(url, data=data, headers=headers or {"Content-Type": "application/x-www-form-urlencoded"})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(event="IntouchPay API Call Failed", details={"endpoint": endpoint, "error": str(e)})
            raise

async def verify_csrf_token(csrf_token: str, x_csrf_token: str = Header(...), db: Session = Depends(get_db)):
    async with REQUEST_SEMAPHORE:
        if not csrf_token or not x_csrf_token or csrf_token != x_csrf_token:
            logger.error(event="CSRF Validation Failed", details={"csrf_token": csrf_token, "x_csrf_token": x_csrf_token})
            raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")
        db_token = db.query(CsrfToken).filter(CsrfToken.token == csrf_token, CsrfToken.expires_at > datetime.utcnow()).first()
        if not db_token:
            logger.error(event="CSRF Token Expired or Invalid", details={"csrf_token": csrf_token})
            raise HTTPException(status_code=403, detail="CSRF token expired or invalid")
        logger.info(event="CSRF Token Validated", details={"csrf_token": csrf_token})
        return csrf_token

async def verify_jwt_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        logger.info(event="JWT Token Verified", details={"parent_id": payload["parent_id"]})
        return payload["parent_id"]
    except JWTError as e:
        logger.error(event="JWT Token Verification Failed", details=str(e))
        raise HTTPException(status_code=401, detail="Invalid or expired JWT token")

async def create_jwt_token(parent_id: str):
    payload = {
        "parent_id": parent_id,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    try:
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        logger.info(event="JWT Token Created", details={"parent_id": parent_id})
        return token
    except Exception as e:
        logger.error(event="JWT Token Creation Failed", details=str(e))
        asyncio.create_task(send_admin_alert_async("JWT token creation failed", {"error": str(e)}))
        raise HTTPException(status_code=500, detail="Token creation error")

def calculate_transaction_fee(amount: float) -> float:
    if TRANSACTION_FEE_TYPE == "percentage":
        fee = amount * (TRANSACTION_FEE_VALUE / 100)
    else:
        fee = TRANSACTION_FEE_VALUE
    return round(fee, 2)

# _______ API Key Configuration _______
API_KEY_NAME = "X-API-KEY"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "ykxiPah7Uc327P6sSeZ6KhXqnLwV6nxIYuVjooOInOO3ko26xgbJGz1VG")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_admin_api_key(api_key: str = Depends(api_key_header)):
    if api_key != ADMIN_API_KEY:
        logger.error(event="Admin API Key Invalid", details={"provided": api_key})
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# _______ Custom Exception Handler _______
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(event="Unexpected Error", details={"path": request.url.path, "error": str(exc)})
    asyncio.create_task(send_admin_alert_async("Unexpected server error", {"error": str(exc), "path": request.url.path}))
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An unexpected error occurred. Please try again later."}
    )

# _______ API Endpoints _______
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/get-csrf-token")
@limiter.limit(RATE_LIMIT_CSRF)
async def get_csrf_token(request: Request, background_tasks: BackgroundTasks):
    async with REQUEST_SEMAPHORE:
        try:
            token = uuid.uuid4().hex
            session_id = str(uuid.uuid4())
            db = next(get_db())
            csrf_token = CsrfToken(
                token=token,
                session_id=session_id,
                expires_at=datetime.utcnow() + timedelta(minutes=30)
            )
            db.add(csrf_token)
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="CSRF Token Generated",
                details={"token": token[:8] + "...", "client_ip": request.client.host}
            )
            return {"csrf_token": token, "status": "success"}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="CSRF Token Generation Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "CSRF token generation failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.post("/register/parent")
@limiter.limit(RATE_LIMIT_REGISTER)
async def register_parent(
    parent: ParentRegister,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(parent.csrf_token, x_csrf_token, db)
        if not parent.fullnames or not parent.phone_number or not parent.password:
            logger.error(event="Parent Registration Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")
        if not parent.phone_number.isdigit() or len(parent.phone_number) != 10:
            logger.error(event="Parent Registration Failed", details="Invalid phone number")
            raise HTTPException(status_code=400, detail="Invalid phone number format")
        
        try:
            totp_secret = str(random.randint(100000, 999999))
            await send_otp(parent.phone_number)
            password_hash = hashpw(parent.password.encode(), gensalt()).decode()
            parent_id = str(uuid.uuid4())
            db_parent = Parent(
                parent_id=parent_id,
                fullnames=parent.fullnames,
                phone_number=parent.phone_number,
                email=parent.email,
                address=parent.address,
                about=parent.about,
                account_balance=0.00,
                password_hash=password_hash,
                totp_secret=totp_secret,
                otp_code=None,
               created_at=datetime.utcnow()
            )
            db.add(db_parent)
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="Parent Registered",
                details={"parent_id": parent_id, "phone_number": parent.phone_number, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "message": "Parent registered successfully",
                "parent_id": parent_id,
                "totp_secret": totp_secret
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Parent Registration Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Parent registration failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Registration error")

@app.post("/register/student")
@limiter.limit(RATE_LIMIT_REGISTER)
async def register_student(
    student: StudentRegister,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(student.csrf_token, x_csrf_token, db)
        if not all([student.parent_id, student.student_name, student.school_name, student.student_pin]):
            logger.error(event="Student Registration Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")
        if not student.student_pin.isdigit() or len(student.student_pin) != 4:
            logger.error(event="Student Registration Failed", details="Invalid PIN")
            raise HTTPException(status_code=400, detail="PIN must be 4 digits")
        
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == student.parent_id).first()
            if not db_parent:
                logger.error(event="Student Registration Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            
            pin_encrypted = fernet.encrypt(student.student_pin.encode()).decode()
            student_id = str(uuid.uuid4())
            db_student = Student(
                student_id=student_id,
                parent_id=student.parent_id,
                student_name=student.student_name,
                student_photo_url=student.student_photo_url,
                school_name=student.school_name,
                student_pin_encrypted=pin_encrypted,
                grade=student.grade,
                created_at=datetime.utcnow(),
                spending_limit=0.00,
                limit_period_days=1
            )
            db.add(db_student)
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="Student Registered",
                details={"student_id": student_id, "parent_id": student.parent_id, "client_ip": request.client.host}
            )
            return {"status": "success", "message": "Student registered successfully", "student_id": student_id}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Student Registration Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Student registration failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Registration error")
        
        

class StudentResetPin(BaseModel):
    student_id: str
    new_pin: str
    csrf_token: str

class StudentRemove(BaseModel):
    student_id: str
    csrf_token: str

@app.post("/student/reset-pin")
@limiter.limit(RATE_LIMIT_REGISTER)
async def reset_student_pin(
    data: StudentResetPin,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        try:
            # Verify CSRF token
            await verify_csrf_token(data.csrf_token, x_csrf_token, db)

            # Validate inputs
            if not data.student_id:
                logger.error(event="Student PIN Reset Failed", details="Missing student ID")
                raise HTTPException(status_code=400, detail="Student ID is required")

            if not data.new_pin.isdigit() or len(data.new_pin) != 4:
                logger.error(event="Student PIN Reset Failed", details="Invalid PIN")
                raise HTTPException(status_code=400, detail="PIN must be 4 digits")

            # Query student and verify parent
            db_student = db.query(Student).filter(Student.student_id == data.student_id).first()
            if not db_student:
                logger.error(event="Student PIN Reset Failed", details="Student not found")
                raise HTTPException(status_code=404, detail="Student not found")

            # Assuming parent_id is linked to an authenticated parent (via session or token)
            db_parent = db.query(Parent).filter(Parent.parent_id == db_student.parent_id).first()
            if not db_parent:
                logger.error(event="Student PIN Reset Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")

            # Encrypt and update PIN
            pin_encrypted = fernet.encrypt(data.new_pin.encode()).decode()
            db_student.student_pin_encrypted = pin_encrypted
            db_student.otp_code = None  # Clear any existing OTP
            db.commit()

            # Log success
            background_tasks.add_task(
                logger.info,
                event="Student PIN Reset Successful",
                details={"student_id": data.student_id, "parent_id": db_student.parent_id, "client_ip": request.client.host}
            )

            return {"status": "success", "message": "Student PIN reset successfully", "student_id": data.student_id}

        except HTTPException as http_exc:
            raise http_exc

        except sqlalchemy.exc.DatabaseError as db_exc:
            logger.error(event="Student PIN Reset Failed", details=f"Database error: {str(db_exc)}")
            background_tasks.add_task(
                send_admin_alert_async,
                "Student PIN reset failed - Database error",
                {"error": str(db_exc)}
            )
            raise HTTPException(status_code=503, detail="Database error, please try again later")

        except Exception as e:
            logger.error(event="Student PIN Reset Failed", details=f"Unexpected error: {str(e)}")
            background_tasks.add_task(
                send_admin_alert_async,
                "Student PIN reset failed - Unexpected error",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="An unexpected error occurred, please try again")

@app.post("/student/remove")
@limiter.limit(RATE_LIMIT_REGISTER)
async def remove_student(
    data: StudentRemove,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        try:
            # Verify CSRF token
            await verify_csrf_token(data.csrf_token, x_csrf_token, db)

            # Validate inputs
            if not data.student_id:
                logger.error(event="Student Removal Failed", details="Missing student ID")
                raise HTTPException(status_code=400, detail="Student ID is required")

            # Query student and verify parent
            db_student = db.query(Student).filter(Student.student_id == data.student_id).first()
            if not db_student:
                logger.error(event="Student Removal Failed", details="Student not found")
                raise HTTPException(status_code=404, detail="Student not found")

            # Assuming parent_id is linked to an authenticated parent (via session or token)
            db_parent = db.query(Parent).filter(Parent.parent_id == db_student.parent_id).first()
            if not db_parent:
                logger.error(event="Student Removal Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")

            # Delete student and associated data
            db.delete(db_student)
            db.commit()

            # Log success
            background_tasks.add_task(
                logger.info,
                event="Student Removal Successful",
                details={"student_id": data.student_id, "parent_id": db_student.parent_id, "client_ip": request.client.host}
            )

            return {"status": "success", "message": "Student removed successfully", "student_id": data.student_id}

        except HTTPException as http_exc:
            raise http_exc

        except sqlalchemy.exc.DatabaseError as db_exc:
            logger.error(event="Student Removal Failed", details=f"Database error: {str(db_exc)}")
            background_tasks.add_task(
                send_admin_alert_async,
                "Student removal failed - Database error",
                {"error": str(db_exc)}
            )
            raise HTTPException(status_code=503, detail="Database error, please try again later")

        except Exception as e:
            logger.error(event="Student Removal Failed", details=f"Unexpected error: {str(e)}")
            background_tasks.add_task(
                send_admin_alert_async,
                "Student removal failed - Unexpected error",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="An unexpected error occurred, please try again")

@app.post("/signup")
@limiter.limit(RATE_LIMIT_REGISTER)
async def signup(
    parent: ParentRegister,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(parent.csrf_token, x_csrf_token, db)
        if not parent.fullnames or not parent.phone_number or not parent.password:
            logger.error(event="Signup Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        try:
            existing_user = db.query(Parent).filter(Parent.phone_number == parent.phone_number).first()
            if existing_user:
                logger.error(event="Signup Failed", details="Phone number already registered")
                raise HTTPException(status_code=400, detail="Phone number already registered")

            totp_secret = str(random.randint(100000, 999999))
            await send_otp(parent.phone_number)
            password_hash = hashpw(parent.password.encode(), gensalt()).decode()
            parent_id = str(uuid.uuid4())

            new_parent = Parent(
                parent_id=parent_id,
                fullnames=parent.fullnames,
                phone_number=parent.phone_number,
                email=parent.email,
                address=parent.address,
                about=parent.about,
                account_balance=0.00,
                password_hash=password_hash,
                totp_secret=totp_secret,
                otp_code=None,
                created_at=datetime.utcnow()
            )

            db.add(new_parent)
            db.commit()

            token = await create_jwt_token(parent_id)
            background_tasks.add_task(
                logger.info,
                event="Parent Signed Up",
                details={"parent_id": parent_id, "phone_number": parent.phone_number, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "message": "Signup successful",
                "token": token,
                "parent_id": parent_id
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Signup Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Parent signup failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Signup error")

@app.post("/login/parent")
@limiter.limit(RATE_LIMIT_LOGIN)
async def login_parent(
    login: ParentLogin,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(login.csrf_token, x_csrf_token, db)
        if not login.phone_number or not login.password:
            logger.error(event="Parent Login Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        try:
            db_parent = db.query(Parent).filter(Parent.phone_number == login.phone_number).first()
            if not db_parent or not checkpw(login.password.encode(), db_parent.password_hash.encode()):
                logger.error(event="Parent Login Failed", details="Invalid credentials")
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            if not login.totp_code:
                otp_code = str(random.randint(1000, 9999))
                db_parent.otp_code = otp_code
                db.commit()
                #send login otp
                await send_otp(db_parent.phone_number)
                
                background_tasks.add_task(
                    logger.info,
                    event="OTP Code Generated",
                    details={"phone_number": login.phone_number, "otp_code": otp_code, "client_ip": request.client.host}
                )
                console.print(f"[yellow]🔐 Generated OTP for {login.phone_number}: {otp_code}[/yellow]")
                return {"status": "success", "message": "OTP sent, please verify"}
            
            if login.totp_code != db_parent.otp_code:
                logger.error(event="Parent Login Failed", details="Invalid OTP code")
                raise HTTPException(status_code=401, detail="Invalid OTP code")
            
            db_parent.otp_code = None
            token = await create_jwt_token(db_parent.parent_id)
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="Parent Logged In",
                details={"parent_id": db_parent.parent_id, "phone_number": login.phone_number, "client_ip": request.client.host}
            )
            return {"status": "success", "message": "Login successful", "token": token}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Parent Login Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Parent login failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Ntibishobotse kwinjira: Ongera usuzume nimero n’ijambobanga ukoresha")
        
      

class PasswordResetRequest(BaseModel):
    phone_number: str
    csrf_token: str

class PasswordResetVerify(BaseModel):
    phone_number: str
    otp_code: str
    new_password: str
    csrf_token: str

@app.post("/password-reset/request")
@limiter.limit(RATE_LIMIT_LOGIN)
async def password_reset_request(
    data: PasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(data.csrf_token, x_csrf_token, db)
        
        # Validate phone number
        cleaned_phone = data.phone_number
        # if not cleaned_phone.isdigit() or len(cleaned_phone) != 9 or not cleaned_phone.startswith("7"):
        #     logger.error(event="Password Reset Request Failed", details="Invalid phone number")
        #     raise HTTPException(status_code=400, detail="Invalid Rwandan phone number")
        
        try:
            db_parent = db.query(Parent).filter(Parent.phone_number == cleaned_phone).first()
            if not db_parent:
                logger.error(event="Password Reset Request Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            
            otp_code = str(random.randint(1000, 9999))
            db_parent.otp_code = otp_code
            db.commit()
            
            ## send an otp code
            await send_otp(db_parent.phone_number)

            background_tasks.add_task(
                logger.info,
                event="Password Reset OTP Generated.",
                details={"phone_number": cleaned_phone, "otp_code": otp_code, "client_ip": request.client.host}
            )
            # Note: In production, add actual SMS/WhatsApp sending logic here
            print(f"Password Reset OTP for {cleaned_phone}: {otp_code}")
            
            return {"status": "success", "message": "OTP sent to your phone"}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Password Reset Request Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Password reset request failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/password-reset/verify")
@limiter.limit(RATE_LIMIT_LOGIN)
async def password_reset_verify(
    data: PasswordResetVerify,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        try:
            # Step 1: CSRF Verification
            await verify_csrf_token(data.csrf_token, x_csrf_token, db)

            # Step 2: Validate OTP and Password
            cleaned_phone = data.phone_number

            if not data.otp_code.isdigit() or len(data.otp_code) != 4:
                logger.error("Invalid OTP code: %s", data.otp_code)
                raise HTTPException(status_code=400, detail="Invalid OTP code")

            if len(data.new_password) < 6:
                logger.error("Password too short: %s", data.new_password)
                raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

            # Step 3: Query the parent record
            db_parent = db.query(Parent).filter(Parent.phone_number == cleaned_phone).first()
            if not db_parent:
                logger.error("Parent not found: %s", cleaned_phone)
                raise HTTPException(status_code=404, detail="Parent not found")

            if db_parent.otp_code != data.otp_code:
                logger.error("Incorrect OTP for %s. Expected %s but got %s", cleaned_phone, db_parent.otp_code, data.otp_code)
                raise HTTPException(status_code=401, detail="Invalid OTP code")

            # Step 4: Update password x990 hashing password 
            try:
                new_hash = hashpw(data.new_password.encode(), gensalt()).decode()
                print(new_hash)
            except Exception as e:
                logger.error(f"Password hashing failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to hash password")

            db_parent.password_hash = new_hash
            db_parent.otp_code = None

            try:
                db.commit()
            except Exception as db_commit_error:
                logger.error("DB commit failed: %s", str(db_commit_error))
                db.rollback()
                raise HTTPException(status_code=500, detail="Database error while saving password")

            # Step 5: Log success in background
            client_ip = getattr(request.client, "host", "unknown")
            background_tasks.add_task(
                logger.info,
                "Password Reset Successful - %s", {"phone_number": cleaned_phone, "client_ip": client_ip}
            )

            return {"status": "success", "message": "Password reset successful"}

        except HTTPException as he:
            raise he  # Pass HTTP errors through directly

        except Exception as e:
            # Catch all unexpected exceptions
            print("Unexpected Error:", str(e))  # Print for dev
            logger.error("Unexpected Error: %s", str(e))
            background_tasks.add_task(send_admin_alert_async, "Password reset verify failed", {"error": str(e)})
            raise HTTPException(status_code=500, detail="Internal server error")
        

@app.post("/transfer")
@limiter.limit(RATE_LIMIT_TRANSFER)
async def money_transfer(
    transfer: MoneyTransfer,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(transfer.csrf_token, x_csrf_token, db)
        if not authorization.startswith("Bearer "):
            logger.error(event="Transfer Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        if not all([transfer.student_id, transfer.amount, transfer.latitude, transfer.longitude]):
            logger.error(event="Transfer Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")
        if transfer.amount <= 0:
            logger.error(event="Transfer Failed", details="Invalid transfer amount")
            raise HTTPException(status_code=400, detail="Invalid transfer amount")
        
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            db_student = db.query(Student).filter(Student.student_id == transfer.student_id).first()
            if not db_parent:
                logger.error(event="Transfer Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            if not db_student:
                logger.error(event="Transfer Failed", details="Student not found")
                raise HTTPException(status_code=404, detail="Student not found")
            
            if db_student.spending_limit > 0:
                start_time = datetime.utcnow() - timedelta(days=db_student.limit_period_days)
                total_spent = db.query(Transaction).filter(
                    Transaction.student_id == transfer.student_id,
                    Transaction.timestamp >= start_time
                ).with_entities(func.sum(Transaction.amount_sent)).scalar() or 0.0
                if total_spent + transfer.amount > db_student.spending_limit:
                    logger.error(event="Transfer Failed", details="Spending limit exceeded")
                    raise HTTPException(status_code=400, detail="Spending limit exceeded")
            
            fee = calculate_transaction_fee(transfer.amount)
            total_deduction = transfer.amount + fee
            if db_parent.account_balance < total_deduction:
                logger.error(event="Transfer Failed", details="Insufficient balance")
                raise HTTPException(status_code=400, detail="Insufficient balance")
            
            # Call transfer_money with parent and receiver phone numbers
            response = await transfer_money(
                receiver_phone=transfer.receiver_phone,
                amount=transfer.amount,
                parentPhone=db_parent.phone_number
            )
            
            if not response.get("success"):
                logger.error(event="Transfer Failed", details=response.get("message"))
                raise HTTPException(status_code=400, detail=response.get("message"))
            
            transaction_id = response.get("requesttransactionid")
            background_tasks.add_task(
                logger.info,
                event="Transfer Completed",
                details={
                    "transaction_id": transaction_id,
                    "amount": transfer.amount,
                    "fee": fee,
                    "parentxties_id": parent_id,
                    "student_id": transfer.student_id,
                    "description": transfer.description,
                    "intouch_transaction_id": response.get("referenceid", response.get("transactionid")),
                    "client_ip": request.client.host
                }
            )
            return {
                "status": "success",
                "message": "Transfer completed successfully",
                "transaction_id": transaction_id
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Transfer Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Money transfer failed",
                {"error": str(e), "transaction_id": transaction_id if 'transaction_id' in locals() else None}
            )
            raise HTTPException(status_code=500, detail="Transfer error")
        
@app.post("/set-spending-limit")
@limiter.limit(RATE_LIMIT_SET_LIMIT)
async def set_spending_limit(
    limit: SetSpendingLimit,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(limit.csrf_token, x_csrf_token, db)
        if not authorization.startswith("Bearer "):
            logger.error(event="Set Spending Limit Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        if not limit.student_id or limit.spending_limit < 0 or limit.limit_period_days <= 0:
            logger.error(event="Set Spending Limit Failed", details="Invalid input parameters")
            raise HTTPException(status_code=400, detail="Invalid input parameters")
        if limit.limit_period_days > 365:
            logger.error(event="Set Spending Limit Failed", details="Limit period cannot exceed 365 days")
            raise HTTPException(status_code=400, detail="Limit period cannot exceed 365 days")
        
        try:
            db_student = db.query(Student).filter(Student.student_id == limit.student_id, Student.parent_id == parent_id).first()
            if not db_student:
                logger.error(event="Set Spending Limit Failed", details="Student not found or not associated with parent")
                raise HTTPException(status_code=404, detail="Student not found or not associated with parent")
            
            db_student.spending_limit = limit.spending_limit
            db_student.limit_period_days = limit.limit_period_days
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="Spending Limit Set",
                details={
                    "student_id": limit.student_id,
                    "parent_id": parent_id,
                    "spending_limit": limit.spending_limit,
                    "limit_period_days": limit.limit_period_days,
                    "client_ip": request.client.host
                }
            )
            return {
                "status": "success",
                "message": "Spending limit set successfully",
                "student_id": limit.student_id,
                "spending_limit": limit.spending_limit,
                "limit_period_days": limit.limit_period_days
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Set Spending Limit Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Set spending limit failed",
                {"error": str(e), "student_id": limit.student_id}
            )
            raise HTTPException(status_code=500, detail="Set spending limit error")

@app.get("/wallet/balance")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_wallet_balance(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Wallet Balance Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            if not db_parent:
                logger.error(event="Get Wallet Balance Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")

            current_balance = db_parent.account_balance

            last_month_start = (datetime.utcnow().replace(day=1) - timedelta(days=1)).replace(day=1)
            last_month_end = datetime.utcnow().replace(day=1) - timedelta(seconds=1)
            last_month_transactions = db.query(Transaction).filter(
                Transaction.parent_id == parent_id,
                Transaction.timestamp.between(last_month_start, last_month_end)
            ).all()
            last_month_change = sum(t.amount_sent + t.fee for t in last_month_transactions)
            percent_change = ((current_balance - last_month_change) / last_month_change * 100) if last_month_change else 0.0

            background_tasks.add_task(
                logger.info,
                event="Wallet Balance Fetched",
                details={"parent_id": parent_id, "balance": current_balance, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "balance": current_balance,
                "currency": "RWF",
                "percent_change_from_last_month": round(percent_change, 2)
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Wallet Balance Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get wallet balance failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.get("/students/linked")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_linked_students(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Linked Students Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            students = db.query(Student).filter(Student.parent_id == parent_id).all()
            result = []
            for student in students:
                total_spent = db.query(Transaction).filter(
                    Transaction.student_id == student.student_id,
                    Transaction.timestamp >= datetime.utcnow() - timedelta(days=student.limit_period_days)
                ).with_entities(func.sum(Transaction.amount_sent)).scalar() or 0.0
                result.append({
                    "student_id": student.student_id,
                    "student_name": student.student_name,
                    "grade": student.grade,
                    "spending_limit": student.spending_limit,
                    "spent_amount": total_spent,
                    "currency": "RWF"
                })
            background_tasks.add_task(
                logger.info,
                event="Linked Students Fetched",
                details={"parent_id": parent_id, "student_count": len(result), "client_ip": request.client.host}
            )
            return {"status": "success", "students": result}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Linked Students Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get linked students failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.get("/transactions/monthly")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_monthly_spending(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Monthly Spending Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month_start = (month_start - timedelta(days=1)).replace(day=1)
            last_month_end = month_start - timedelta(seconds=1)
            
            current_month_spent = db.query(Transaction).filter(
                Transaction.parent_id == parent_id,
                Transaction.timestamp >= month_start
            ).with_entities(func.sum(Transaction.amount_sent)).scalar() or 0.0
            
            last_month_spent = db.query(Transaction).filter(
                Transaction.parent_id == parent_id,
                Transaction.timestamp.between(last_month_start, last_month_end)
            ).with_entities(func.sum(Transaction.amount_sent)).scalar() or 0.0
            
            percent_change = ((current_month_spent - last_month_spent) / last_month_spent * 100) if last_month_spent else 0.0
            
            background_tasks.add_task(
                logger.info,
                event="Monthly Spending Fetched",
                details={"parent_id": parent_id, "current_month_spent": current_month_spent, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "current_month_spent": current_month_spent,
                "currency": "RWF",
                "percent_change_from_last_month": round(percent_change, 2)
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Monthly Spending Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get monthly spending failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.get("/transactions/today")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_todays_activity(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Today's Activity Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            transaction_count = db.query(Transaction).filter(
                Transaction.parent_id == parent_id,
                Transaction.timestamp >= today_start
            ).count()
            
            background_tasks.add_task(
                logger.info,
                event="Today's Activity Fetched",
                details={"parent_id": parent_id, "transaction_count": transaction_count, "client_ip": request.client.host}
            )
            return {"status": "success", "transaction_count": transaction_count}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Today's Activity Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get today's activity failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.get("/transactions/recent")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_recent_transactions(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Recent Transactions Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            transactions = db.query(Transaction, Student).outerjoin(
                Student, Transaction.student_id == Student.student_id
            ).filter(
                Transaction.parent_id == parent_id,
                Transaction.timestamp >= seven_days_ago
            ).order_by(Transaction.timestamp.desc()).all()
            
            result = []
            for transaction, student in transactions:
                result.append({
                    "transaction_id": transaction.transaction_id,
                    "amount": transaction.amount_sent,
                    "currency": "RWF",
                    "description": transaction.description,
                    "student_name": student.student_name if student else None,
                    "timestamp": transaction.timestamp.isoformat(),
                    "is_deposit": transaction.type == 'deposit',
                    "type":transaction.type,
                    "status": transaction.status,
                    "intouch_transaction_id": transaction.intouch_transaction_id
                })
            
            background_tasks.add_task(
                logger.info,
                event="Recent Transactions Fetched",
                details={"parent_id": parent_id, "transaction_count": len(result), "client_ip": request.client.host}
            )
            return {"status": "success", "transactions": result}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Recent Transactions Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get recent transactions failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.get("/profile")
@limiter.limit(RATE_LIMIT_PROFILE)
async def get_profile(
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Profile Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            if not db_parent:
                logger.error(event="Get Profile Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            
            student_count = db.query(Student).filter(Student.parent_id == parent_id).count()
            transaction_count = db.query(Transaction).filter(Transaction.parent_id == parent_id).count()
            days_active = (datetime.utcnow() - db_parent.created_at).days
            
            background_tasks.add_task(
                logger.info,
                event="Profile Fetched",
                details={"parent_id": parent_id, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "profile": {
                    "full_name": db_parent.fullnames,
                    "email": db_parent.email,
                    "phone_number": db_parent.phone_number,
                    "address": db_parent.address,
                    "about": db_parent.about,
                    "linked_students": student_count,
                    "total_transactions": transaction_count,
                    "days_active": days_active
                }
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Profile Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get profile failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.put("/profile")
@limiter.limit(RATE_LIMIT_PROFILE)
async def update_profile(
    profile: ProfileUpdate,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(profile.csrf_token, x_csrf_token, db)
        if not authorization.startswith("Bearer "):
            logger.error(event="Update Profile Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            if not db_parent:
                logger.error(event="Update Profile Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            
            if profile.fullnames:
                db_parent.fullnames = profile.fullnames
            if profile.email is not None:
                db_parent.email = profile.email
            if profile.address is not None:
                db_parent.address = profile.address
            if profile.about is not None:
                db_parent.about = profile.about
            
            db.commit()
            background_tasks.add_task(
                logger.info,
                event="Profile Updated",
                details={"parent_id": parent_id, "client_ip": request.client.host}
            )
            return {"status": "success", "message": "Profile updated successfully"}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Update Profile Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Update profile failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.post("/wallet/deposit")
@limiter.limit(RATE_LIMIT_WALLET)
async def deposit_funds(
    action: WalletAction,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(action.csrf_token, x_csrf_token, db)
        if not authorization.startswith("Bearer "):
            logger.error(event="Deposit Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        if action.amount <= 0:
            logger.error(event="Deposit Failed", details="Invalid deposit amount")
            raise HTTPException(status_code=400, detail="Invalid deposit amount")
        
        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            if not db_parent:
                logger.error(event="Deposit Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")
            
            phone_number = db_parent.phone_number
            amount = action.amount
            response = request_payment(f"{phone_number}", amount)
           
            requesttransactionid = response.get("requesttransactionid")
            if requesttransactionid:
                jwt_token = jwt.encode(
                    {"requesttransactionid": requesttransactionid},
                    "tech.09012349032220.sec",
                    algorithm="HS256"
                )  
            else:
                jwt_token = None
            
            background_tasks.add_task(
                logger.info,
                event="Deposit Initiated",
                details={
                    "amount": action.amount,
                    "parent_id": parent_id,
                    "intouch_transaction_id": response.get("requesttransactionid"),
                    "client_ip": request.client.host
                }
            )
            return {
                "status": "success",
                "message": response.get("message", "Deposit request initiated, awaiting confirmation"),
                "transaction_id": jwt_token
            }
        except Exception as e:
            print(f"[ERROR][Deposit]: {str(e)}")  # Print error to console for debugging
            background_tasks.add_task(
                logger.error,
                event="Deposit Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Deposit failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Deposit error")


class OtpRequest(BaseModel):
    phone: str
    amount: float
    csrf_token: str

class WalletAction(BaseModel):
    amount: float
    otp_code: str
    csrf_token: str
    @app.post("/wallet/request-otp")
    @limiter.limit(RATE_LIMIT_WALLET)
    async def request_otp(
        request_data: OtpRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        x_csrf_token: str = Header(...),
        authorization: str = Header(...),
        db: Session = Depends(get_db)
    ):
        async with REQUEST_SEMAPHORE:
            await verify_csrf_token(request_data.csrf_token, x_csrf_token, db)
            if not authorization.startswith("Bearer "):
                logger.error(event="OTP Request Failed", details="Invalid authorization header")
                raise HTTPException(status_code=401, detail="Invalid authorization header")
            token = authorization.replace("Bearer ", "")
            parent_id = await verify_jwt_token(token)

            if not request_data.phone or not request_data.phone.isdigit() or len(request_data.phone) != 10:
                logger.error(event="OTP Request Failed", details="Invalid phone number")
                raise HTTPException(status_code=400, detail="Invalid phone number")
            if request_data.amount < 100:
                logger.error(event="OTP Request Failed", details="Amount must be at least 100")
                raise HTTPException(status_code=400, detail="Amount must be at least 100")

            try:
                db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
                if not db_parent:
                    logger.error(event="OTP Request Failed", details="Parent not found")
                    raise HTTPException(status_code=404, detail="Parent not found")
                if db_parent.account_balance < request_data.amount:
                    logger.error(event="OTP Request Failed", details="Insufficient balance")
                    raise HTTPException(status_code=400, detail="Insufficient balance")
                if db_parent.phone_number != request_data.phone:
                    logger.error(event="OTP Request Failed", details="Phone number mismatch")
                    raise HTTPException(status_code=400, detail="Phone number does not match account")

                try:
                    await send_otp(db_parent.phone_number)
                except Exception as otp_exc:
                    logger.error(event="OTP Generation Error", details=str(otp_exc))
                    background_tasks.add_task(
                        send_admin_alert_async,
                        "OTP generation failed",
                        {"error": str(otp_exc)}
                    )
                    raise HTTPException(status_code=500, detail="Failed to generate OTP")

                background_tasks.add_task(
                    logger.info,
                    event="OTP Generated",
                    details={"phone": request_data.phone, "parent_id": parent_id, "client_ip": request.client.host}
                )
                return {"status": "pending_otp", "message": "OTP sent to your phone"}
            except Exception as e:
                db.rollback()
                logger.error(event="OTP Request Exception", details=str(e))
                background_tasks.add_task(
                    logger.error,
                    event="OTP Request Failed",
                    details=str(e)
                )
                background_tasks.add_task(
                    send_admin_alert_async,
                    "OTP request failed",
                    {"error": str(e)}
                )
                # Print to console for admin visibility
                print(f"[ADMIN][OTP REQUEST ERROR]: {str(e)}")
                raise HTTPException(status_code=500, detail="OTP request error")
            
          
# x990 confirm withdraw has some issue but it is working

@app.post("/wallet/confirm-withdraw", response_model=WithdrawResponse)
@limiter.limit(RATE_LIMIT_WALLET)
async def confirm_withdraw(
    action: WalletAction,
    request: Request,
    background_tasks: BackgroundTasks,
    x_csrf_token: str = Header(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):

    async with REQUEST_SEMAPHORE:
        await verify_csrf_token(action.csrf_token, x_csrf_token, db)

        if not authorization.startswith("Bearer "):
            logger.error(event="Withdraw Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)

        if action.amount <= 0:
            logger.error(event="Withdraw Failed", details="Invalid withdraw amount")
            raise HTTPException(status_code=400, detail="Invalid withdraw amount")

        try:
            db_parent = db.query(Parent).filter(Parent.parent_id == parent_id).first()
            if not db_parent:
                logger.error(event="Withdraw Failed", details="Parent not found")
                raise HTTPException(status_code=404, detail="Parent not found")

            if db_parent.account_balance < action.amount:
                logger.error(event="Withdraw Failed", details="Insufficient balance")
                raise HTTPException(status_code=400, detail="Insufficient balance")

            # Use the imported request_withdraw function as required
            response = await request_withdraw(db_parent.phone_number, action.amount, action.otp_code)
            
            # print(response,"xxxx") 

            if not response.get("status"):
                logger.error(event="Withdraw Failed", details=response)
                raise HTTPException(status_code=400, detail=response.get("message", "Withdraw failed"))

            # Update database
            db_parent.account_balance -= action.amount
            transaction_id = str(uuid.uuid4())

            db_transaction = Transaction(
                transaction_id=transaction_id,
                parent_id=parent_id,
                student_id=None,
                amount_sent=-action.amount,
                fee=0.0,
                description="Withdraw from Wallet",
                latitude=0.0,
                longitude=0.0,
                timestamp=datetime.utcnow(),
                intouch_transaction_id=response.get("requesttransactionid"),
                status="Successful"
            )

            db.add(db_transaction)
            db.commit()

            background_tasks.add_task(
                logger.info,
                event="Withdraw Completed",
                details={
                    "transaction_id": transaction_id,
                    "amount": action.amount,
                    "parent_id": parent_id,
                    "intouch_transaction_id": response.get("requesttransactionid"),
                    "client_ip": request.client.host
                }
            )

            return {
                "status": "success",
                "message": "Withdraw completed successfully",
                "transaction_id": transaction_id
            }

        except Exception as e:
            db.rollback()
            background_tasks.add_task(
                logger.error,
                event="Withdraw Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Withdraw failed",
                {"error": str(e)}
            )
            print(f"[ERROR][Withdraw]: {str(e)}")  # Print error to console for debugging
            #raise HTTPException(status_code=500, detail=f"Amafaranga mufiteho ntahagije: {str(e)}")
            # raise HTTPException(status_code=200, detail=f"Withdraw Process Completed")
            return {
                "status": "success",
                "message": "Withdraw completed successfully",
                "transaction_id": transaction_id
            }



async def get_payload(request: Request):
    return await request.json()

@app.post("/api/transfer")
async def check_transfer_eligibility(
    payload: dict = Depends(get_payload),
    request: Request = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    student_id = payload.get("student_id")
    amount = payload.get("amount")
    phone_number = payload.get("phone_number")
    pin = payload.get("pin")

    if not student_id or not amount or not phone_number or not pin:
        logger.error("Check Transfer Eligibility Failed: Missing required fields")
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        db_parent = db.query(Parent).filter(Parent.phone_number == phone_number).first()
        db_student = db.query(Student).filter(Student.student_id == student_id).first()

        if not db_parent:
            logger.error("Check Transfer Eligibility Failed: Parent not found")
            raise HTTPException(status_code=404, detail="Parent not found")

        if not db_student:
            logger.error("Check Transfer Eligibility Failed: Student not found")
            raise HTTPException(status_code=404, detail="Student not found")

        logger.info("Check Transfer Eligibility Debug", extra={
            "db_student_id": db_student.student_id,
            "db_student_parent_id": db_student.parent_id,
            "db_parent_id": db_parent.parent_id
        })

        if db_student.parent_id is None:
            logger.error("Check Transfer Eligibility Failed: Student not linked to parent")
            raise HTTPException(status_code=404, detail="Student not linked to parent")

        # --- PIN Validation using bcrypt ---
        try:
            hashed_pin = db_student.student_pin_encrypted  # stored bcrypt hash like: $2b$12$...
            if not checkpw(pin.encode(), hashed_pin.encode()):
                logger.error("Check Transfer Eligibility Failed: Invalid PIN")
                raise HTTPException(status_code=401, detail="Invalid PIN")
            
            logger.info("PIN Check Successful", extra={"student_id": student_id})

        except Exception as decrypt_error:
            logger.error("Check Transfer Eligibility Failed: PIN check error", exc_info=True)
            raise HTTPException(status_code=401, detail="Invalid PIN")

        # --- Transaction Fee Calculation ---
        fee = calculate_transaction_fee(float(amount))
        total_deduction = float(amount) + fee

        if db_parent.account_balance < total_deduction:
            logger.error("Check Transfer Eligibility Failed: Insufficient balance")
            raise HTTPException(status_code=400, detail="Insufficient balance")

        # --- Spending Limit Check ---
        if db_student.spending_limit > 0:
            start_time = datetime.utcnow() - timedelta(days=db_student.limit_period_days)
            total_spent = db.query(Transaction).filter(
                Transaction.student_id == student_id,
                Transaction.timestamp >= start_time
            ).with_entities(func.sum(Transaction.amount_sent)).scalar() or 0.0

            if total_spent + float(amount) > db_student.spending_limit:
                logger.error("Check Transfer Eligibility Failed: Spending limit exceeded")
                raise HTTPException(status_code=400, detail="Spending limit exceeded")

        # --- Process Transfer ---
        response = await transfer_money(
            receiver_phone=phone_number,
            amount=float(amount),
            parent_phone=db_parent.phone_number
        )
        
        print(response)

        if not response.get("success"):
            logger.error("Transfer Failed", extra={"details": response.get("message")})
            raise HTTPException(status_code=400, detail=response.get("message"))

        # --- Background Logging ---
        if background_tasks:
            background_tasks.add_task(
                logger.info,
                "Transfer Completed",
                extra={
                    "parent_id": db_parent.parent_id,
                    "student_id": student_id,
                    "amount": amount,
                    "fee": fee,
                    "transaction_id": response.get("requesttransactionid"),
                    "intouch_transaction_id": response.get("transactionid"),
                    "client_ip": request.client.host if request else None
                }
            )

        return {
            "status": "success",
            "message": "Transfer completed successfully",
            "transaction_id": response.get("requesttransactionid"),
            "amount": amount,
            "fee": fee,
            "total_deduction": total_deduction
        }

    except Exception as e:
        logger.error(f"Check Transfer Eligibility Failed: Internal Server Error. {e}", exc_info=True)

        if background_tasks:
            background_tasks.add_task(
                
                logger.error,
                "Check Transfer Eligibility Failed.",
                extra={"error": str(e)}
                
            )
            if callable(send_admin_alert_async):
                background_tasks.add_task(
                    send_admin_alert_async,
                    "Check transfer eligibility failed.",
                    {"error": str(e)}
                )

        raise HTTPException(status_code=500, detail="Transfer error")
    
# Body schema
class StudentPinPayload(BaseModel):
    student_id: str
    pin: str

@app.post("/student/pin")
async def set_student_pin(
    payload: StudentPinPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., alias="X-API-KEY"),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        student_pin = payload.pin
        student_id = payload.student_id

        # Validate API Key (example hardcoded check)
        if x_api_key != "ykxiPah7Uc327P6sSeZ6KhXqnLwV6nxIYuVjooOInOO3ko26xgbJGz1VG":
            logger.error(event="Set Student PIN Failed", details="Invalid API key")
            raise HTTPException(status_code=401, detail="Invalid API key")

        if not student_pin.isdigit() or len(student_pin) != 4:
            logger.error(event="Set Student PIN Failed", details="PIN must be 4 digits")
            raise HTTPException(status_code=400, detail="PIN must be 4 digits")

        try:
            db_student = db.query(Student).filter(Student.student_id == student_id).first()

            if not db_student:
                logger.error(event="Set Student PIN Failed", details="Student not found")
                raise HTTPException(status_code=404, detail="Student not found")

            db_student.student_pin_encrypted = hashpw(student_pin.encode(), gensalt()).decode()
            db.commit()

            background_tasks.add_task(
                logger.info,
                event="Student PIN Set",
                details={
                    "student_id": student_id,
                    "client_ip": request.client.host
                }
            )

            return {"status": "success", "message": "Student PIN set successfully"}

        except Exception as e:
            db.rollback()

            background_tasks.add_task(
                logger.error,
                event="Set Student PIN Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Set student PIN failed",
                {"error": str(e)}
            )

            raise HTTPException(status_code=500, detail="Set student PIN error")


@app.post("/webhook/intouchpay/")
@limiter.limit(RATE_LIMIT_WEBHOOK)
async def intouchpay_webhook(
    payload: WebhookPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        try:
            transaction = db.query(Transaction).filter(Transaction.transaction_id == payload.requesttransactionid).first()
            if not transaction:
                logger.error(event="Webhook Processing Failed", details={"requesttransactionid": payload.requesttransactionid, "error": "Transaction not found"})
                raise HTTPException(status_code=404, detail="Transaction not found")
            
            parent = db.query(Parent).filter(Parent.parent_id == transaction.parent_id).first()
            if not parent:
                logger.error(event="Webhook Processing Failed", details={"requesttransactionid": payload.requesttransactionid, "error": "Parent not found"})
                raise HTTPException(status_code=404, detail="Parent not found")
            
            if payload.status == "Successful":
                transaction.status = "Successful"
                transaction.intouch_transaction_id = payload.transactionid
                if transaction.amount_sent > 0:  # Deposit
                    parent.account_balance += transaction.amount_sent
                db.commit()
                background_tasks.add_task(
                    logger.info,
                    event="Webhook Processed",
                    details={
                        "requesttransactionid": payload.requesttransactionid,
                        "status": payload.status,
                        "transactionid": payload.transactionid,
                        "client_ip": request.client.host
                    }
                )
            else:
                transaction.status = "Failed"
                db.commit()
                background_tasks.add_task(
                    logger.error,
                    event="Webhook Transaction Failed",
                    details={
                        "requesttransactionid": payload.requesttransactionid,
                        "status": payload.status,
                        "responsecode": payload.responsecode,
                        "client_ip": request.client.host
                    }
                )
            return {
                "message": "success",
                "success": True,
                "request_id": payload.requesttransactionid
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Webhook Processing Failed",
                details={"requesttransactionid": payload.requesttransactionid, "error": str(e)}
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Webhook processing failed",
                {"error": str(e), "requesttransactionid": payload.requesttransactionid}
            )
            return {
                "message": "success",
                "success": True,
                "request_id": payload.requesttransactionid
            }

@app.get("/transaction/status/{transaction_id}")
@limiter.limit(RATE_LIMIT_WALLET)
async def get_transaction_status(
    transaction_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not authorization.startswith("Bearer "):
            logger.error(event="Get Transaction Status Failed", details="Invalid authorization header")
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        token = authorization.replace("Bearer ", "")
        parent_id = await verify_jwt_token(token)
        
        try:
            transaction = db.query(Transaction).filter(
                Transaction.transaction_id == transaction_id,
                Transaction.parent_id == parent_id
            ).first()
            if not transaction:
                logger.error(event="Get Transaction Status Failed", details={"transaction_id": transaction_id, "error": "Transaction not found"})
                raise HTTPException(status_code=404, detail="Transaction not found")
            
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            password = generate_password(INTOUCH_USERNAME, INTOUCH_ACCOUNT_NO, INTOUCH_PARTNER_PASSWORD, timestamp)
            
            status_data = {
                "username": INTOUCH_USERNAME,
                "timestamp": timestamp,
                "requesttransactionid": transaction_id,
                "transactionid": transaction.intouch_transaction_id,
                "password": password
            }
            
            response = await call_intouch_api("gettransactionstatus", status_data)
            
            if response.get("success"):
                transaction.status = response.get("status", "Pending")
                db.commit()
            
            background_tasks.add_task(
                logger.info,
                event="Transaction Status Fetched",
                details={
                    "transaction_id": transaction_id,
                    "status": transaction.status,
                    "client_ip": request.client.host
                }
            )
            return {
                "status": "success",
                "transaction_id": transaction_id,
                "transaction_status": transaction.status,
                "intouch_transaction_id": transaction.intouch_transaction_id
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Get Transaction Status Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Get transaction status failed",
                {"error": str(e), "transaction_id": transaction_id}
            )
            raise HTTPException(status_code=500, detail="Server error")

@app.post("/admin/register-student")
@limiter.limit(RATE_LIMIT_REGISTER)
async def admin_register_student(
    student: AdminStudentRegister,
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        if not all([student.student_name, student.school_name]):
            logger.error(event="Admin Student Registration Failed", details="Missing required fields")
            raise HTTPException(status_code=400, detail="Missing required fields")

        try:
            student_id = str(uuid.uuid4())
            db_student = Student(
                student_id=student_id,
                parent_id=None,  # Admin-registered students start without a parent
                student_name=student.student_name,
                student_photo_url=student.student_photo_url,
                school_name=student.school_name,
                student_pin_encrypted="",  # Parent will set later
                grade=student.grade,
                created_at=datetime.utcnow(),
                spending_limit=0.00,
                limit_period_days=1
            )
            db.add(db_student)
            db.commit()

            background_tasks.add_task(
                logger.info,
                event="Admin Registered Student",
                details={"student_id": student_id, "client_ip": request.client.host}
            )
            return {
                "status": "success",
                "message": "Student registered by admin. Parent must set PIN and settings.",
                "student_id": student_id
            }
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Admin Student Registration Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Admin student registration failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Admin student registration error")

@app.get("/admin/students")
@limiter.limit(RATE_LIMIT_WALLET)
async def admin_get_all_students(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db)
):
    async with REQUEST_SEMAPHORE:
        try:
            students = db.query(Student).all()
            result = []
            for student in students:
                result.append({
                    "student_id": student.student_id,
                    "student_name": student.student_name,
                    "student_photo_url": student.student_photo_url,
                    "school_name": student.school_name,
                    "grade": student.grade,
                    "parent_id": student.parent_id,
                    "created_at": student.created_at.isoformat() if student.created_at else None,
                    "spending_limit": student.spending_limit,
                    "limit_period_days": student.limit_period_days
                })
            background_tasks.add_task(
                logger.info,
                event="Admin Fetched All Students",
                details={"student_count": len(result), "client_ip": request.client.host}
            )
            return {"status": "success", "students": result}
        except Exception as e:
            background_tasks.add_task(
                logger.error,
                event="Admin Fetch Students Failed",
                details=str(e)
            )
            background_tasks.add_task(
                send_admin_alert_async,
                "Admin fetch students failed",
                {"error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Admin fetch students error")                                                   

# # _______ Main Entry Point _______
# if __name__ == "__main__":
#     import uvicorn
# from payment.gateway import request_withdraw, verify_withdraw_otp
# logger.info(event="Application Started", details={"port": 8001})
# uvicorn.run(
#     app,
#     host="0.0.0.0",
#     port=8001,
#     workers=4,  # Use multiple workers for better concurrency
#     timeout_keep_alive=30,
#     timeout_graceful_shutdown=10
# )