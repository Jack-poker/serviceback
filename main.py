import sys
import os
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional, List
from datetime import datetime, timedelta
import uuid
from passlib.context import CryptContext
from jose import jwt, JWTError
import secrets
from db.connection import db_Query
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

# Semaphore for controlling concurrent database operations
DB_SEMAPHORE = asyncio.Semaphore(10)

# Add trusted websites
origins = [
    "http://localhost:8080",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Security configurations
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
CSRF_TOKEN_EXPIRE_MINUTES = 30

# Password encryption
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")
bearer_scheme = HTTPBearer()

# Database operation wrapper with reconnection logic
async def execute_db_query(query: str, params: list = None, fetch: str = "none"):
    async with DB_SEMAPHORE:
        for attempt in range(3):  # Retry logic
            try:
                # Ensure db_Query.execute is called and result is processed
                db_Query.execute(query, params or [])
                
                if fetch == "one":
                    result = db_Query.fetchone()
                    return result if result is not None else None
                elif fetch == "all":
                    result = db_Query.fetchall()
                    return result if result is not None else []
                else:
                    return db_Query.rowcount
            except Exception as e:
                logger.warning(f"Database query attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    logger.error(f"Database query failed after retries: {str(e)}")
                    raise HTTPException(status_code=500, detail="Database operation failed")
                await asyncio.sleep(1)  # Wait before retry

# Pydantic models (unchanged)
class Admin(BaseModel):
    status: Optional[str] = None
    admin_id: str
    fullnames: str
    email: EmailStr
    password_hash: str
    created_at: str
    last_activity: str

class AdminLogin(BaseModel):
    status: Optional[str] = None
    email: EmailStr
    password: str

class Token(BaseModel):
    status: Optional[str] = None
    access_token: str
    token_type: str

class CSRFToken(BaseModel):
    status: Optional[str] = None
    csrf_token: str

class Product(BaseModel):
    status: Optional[str] = None
    product_id: str
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class ProductCreate(BaseModel):
    status: Optional[str] = None
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class Ad(BaseModel):
    status: Optional[str] = None
    ad_id: str
    title: str
    subtitle: str
    cta_text: str
    cta_link: HttpUrl
    image_url: HttpUrl
    status: str
    start_date: str
    end_date: str
    impressions: int
    clicks: int

class AdCreate(BaseModel):
    status: Optional[str] = None
    title: str
    subtitle: str
    cta_text: str
    cta_link: HttpUrl
    image_url: HttpUrl
    start_date: str
    end_date: str
    status: str = "active"

class AdStatusUpdate(BaseModel):
    status: str

class Transaction(BaseModel):
    status: Optional[str] = None
    transaction_id: str
    parent_id: str
    student_id: Optional[str] = None
    amount_sent: float
    fee: float
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: str
    parent: Optional[dict] = None
    student: Optional[dict] = None

class Parent(BaseModel):
    status: Optional[str] = None
    parent_id: str
    fullnames: str
    phone_number: str
    email: Optional[EmailStr] = None
    account_balance: float
    created_at: str

class Settings(BaseModel):
    status: Optional[str] = None
    settings_id: str
    platform_name: str
    platform_email: EmailStr
    support_email: EmailStr
    max_transaction_amount: float
    min_transaction_amount: float
    transaction_fee_percentage: float
    require_two_factor_auth: bool
    password_min_length: int
    session_timeout: int
    email_notifications: bool
    sms_notifications: bool
    push_notifications: bool
    primary_color: str
    secondary_color: str

class SettingsUpdate(BaseModel):
    status: Optional[str] = None
    platform_name: Optional[str] = None
    platform_email: Optional[EmailStr] = None
    support_email: Optional[EmailStr] = None
    max_transaction_amount: Optional[float] = None
    min_transaction_amount: Optional[float] = None
    transaction_fee_percentage: Optional[float] = None
    require_two_factor_auth: Optional[bool] = None
    password_min_length: Optional[int] = None
    session_timeout: Optional[int] = None
    email_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None

# Helper functions
async def hash_password(password: str) -> str:
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Password hashing failed")

async def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Password verification failed")

async def create_access_token(data: dict) -> str:
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "sub": data.get("admin_id")})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Token creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Token creation failed")

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        admin_id: str = payload.get("sub")
        if admin_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        query = "UPDATE admins SET last_activity = %s WHERE admin_id = %s"
        await execute_db_query(query, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), admin_id])
        return admin_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")
    except Exception as e:
        logger.error(f"Admin authentication failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

async def verify_csrf_token(csrf_token: str, session_id: str):
    try:
        query = "SELECT token FROM csrf_tokens WHERE token = %s AND session_id = %s AND expires_at > NOW()"
        result = await execute_db_query(query, [csrf_token, session_id], fetch="one")
        if not result:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired CSRF token")
    except Exception as e:
        logger.error(f"CSRF verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="CSRF verification failed")

async def get_csrf_and_session(request: Request):
    csrf_token = request.headers.get("x-csrf-token")
    session_id = request.cookies.get("session_id")
    if not csrf_token or not session_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing CSRF token or session ID")
    return csrf_token, session_id

# CSRF token generation
@app.get("/admin/get-csrf-token", response_model=CSRFToken)
async def get_csrf_token(response: Response):
    try:
        csrf_token = secrets.token_hex(16)
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(minutes=CSRF_TOKEN_EXPIRE_MINUTES)

        query = "INSERT INTO csrf_tokens (token, session_id, created_at, expires_at) VALUES (%s, %s, %s, %s)"
        await execute_db_query(query, [csrf_token, session_id, created_at, expires_at])

        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=True,
            samesite="strict"
        )
        return {"status": "success", "csrf_token": csrf_token}
    except Exception as e:
        logger.error(f"CSRF token generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="CSRF token generation failed")

# Admin initialization
async def initDb_admin():
    try:
        admin = Admin(
            status="",
            admin_id=str(uuid.uuid4()),
            fullnames="Tuyishimire Emmanuel",
            email="tuyishimireemmanuel24@gmail.com",
            password_hash=await hash_password("admin123"),
            created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            last_activity=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        admin_data = admin.dict()
        query = "SELECT * FROM admins WHERE email = %s"
        result = await execute_db_query(query, [admin_data["email"]], fetch="one")

        if result:
            logger.info("Admin already exists")
            return "Admin already exists"

        query = """
            INSERT INTO admins (admin_id, fullnames, email, password_hash, created_at, last_activity)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        await execute_db_query(query, [
            admin_data["admin_id"],
            admin_data["fullnames"],
            admin_data["email"],
            admin_data["password_hash"],
            admin_data["created_at"],
            admin_data["last_activity"]
        ])
    except Exception as e:
        logger.error(f"Admin initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Admin initialization failed")

# Admin login
@app.post("/admin/login", response_model=Token)
async def admin_login(login: AdminLogin):
    try:
        query = "SELECT email, password_hash, admin_id FROM admins WHERE email = %s"
        db_result = await execute_db_query(query, [login.email], fetch="one")

        if not db_result:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        email, hash_password, admin_id = db_result
        if not await verify_password(login.password, hash_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        token = await create_access_token({"admin_id": admin_id})
        return {"status": "success", "access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Admin login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

# Admin profile
@app.get("/admin/profile")
async def admin_profile(admin_id: str = Depends(get_current_admin)):
    try:
        query = "SELECT admin_id, fullnames, email, created_at, last_activity FROM admins WHERE admin_id = %s"
        result = await execute_db_query(query, [admin_id], fetch="one")

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

        return {
            "status": "success",
            "admin_id": result[0],
            "fullnames": result[1],
            "email": result[2],
            "created_at": result[3],
            "last_activity": result[4]
        }
    except Exception as e:
        logger.error(f"Profile fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Profile fetch failed")

# Admin dashboard stats
@app.get("/admin/stats")
async def get_stats(admin_id: str = Depends(get_current_admin)):
    try:
        stats = {
            "total_users": 0,
            "total_transactions": 0,
            "total_revenue": 0.0,
            "active_ads": 0
        }

        # Total users
        result = await execute_db_query("SELECT COUNT(*) FROM parents", fetch="one")
        stats["total_users"] = result[0] if result else 0

        # Total transactions
        result = await execute_db_query("SELECT COUNT(*) FROM transactions", fetch="one")
        stats["total_transactions"] = result[0] if result else 0

        # Total revenue
        result = await execute_db_query("SELECT SUM(fee) FROM transactions", fetch="one")
        stats["total_revenue"] = float(result[0]) if result and result[0] is not None else 0.0

        # Active ads
        result = await execute_db_query("SELECT COUNT(*) FROM ads WHERE status = 'active'", fetch="one")
        stats["active_ads"] = result[0] if result else 0

        return stats
    except Exception as e:
        logger.error(f"Stats fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Stats fetch failed")

# Product management
@app.get("/admin/products", response_model=List[Product])
async def get_products(admin_id: str = Depends(get_current_admin)):
    try:
        query = "SELECT product_id, name, price, affiliate_link, description, image_url FROM products"
        results = await execute_db_query(query, fetch="all")

        return [Product(
            product_id=r[0],
            name=r[1],
            price=float(r[2]),
            affiliate_link=r[3],
            description=r[4],
            image_url=r[5]
        ) for r in results]
    except Exception as e:
        logger.error(f"Product fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product fetch failed")

@app.get("/admin/products/shop", response_model=List[Product])
async def get_shop_products():
    try:
        query = "SELECT product_id, name, price, affiliate_link, description, image_url FROM products"
        results = await execute_db_query(query, fetch="all")

        return [Product(
            product_id=r[0],
            name=r[1],
            price=float(r[2]),
            affiliate_link=r[3],
            description=r[4],
            image_url=r[5]
        ) for r in results]
    except Exception as e:
        logger.error(f"Shop products fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Shop products fetch failed")

@app.post("/admin/products", response_model=Product)
async def create_product(product: ProductCreate, admin_id: str = Depends(get_current_admin)):
    try:
        product_id = str(uuid.uuid4())
        query = """
            INSERT INTO products (product_id, name, price, affiliate_link, description, image_url)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        await execute_db_query(query, [
            product_id,
            product.name,
            product.price,
            str(product.affiliate_link),
            product.description,
            str(product.image_url)
        ])

        return Product(**product.dict(), product_id=product_id)
    except Exception as e:
        logger.error(f"Product creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product creation failed")

@app.put("/admin/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product: ProductCreate, admin_id: str = Depends(get_current_admin)):
    try:
        query = """
            UPDATE products SET name = %s, price = %s, affiliate_link = %s, description = %s, image_url = %s
            WHERE product_id = %s
        """
        rowcount = await execute_db_query(query, [
            product.name,
            product.price,
            str(product.affiliate_link),
            product.description,
            str(product.image_url),
            product_id
        ])

        if rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

        return Product(**product.dict(), product_id=product_id)
    except Exception as e:
        logger.error(f"Product update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product update failed")

@app.delete("/admin/products/{product_id}")
async def delete_product(product_id: str, admin_id: str = Depends(get_current_admin)):
    try:
        query = "DELETE FROM products WHERE product_id = %s"
        rowcount = await execute_db_query(query, [product_id])

        if rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

        return {"status": "success", "message": "Product deleted successfully"}
    except Exception as e:
        logger.error(f"Product deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product deletion failed")

# Ad management
@app.get("/admin/ads", response_model=List[Ad])
async def get_ads(admin_id: str = Depends(get_current_admin)):
    try:
        query = "SELECT ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks FROM ads"
        results = await execute_db_query(query, fetch="all")

        return [Ad(
            ad_id=r[0],
            title=r[1],
            subtitle=r[2],
            cta_text=r[3],
            cta_link=r[4],
            image_url=r[5],
            status=r[6],
            start_date=r[7].strftime('%Y-%m-%d') if r[7] else None,
            end_date=r[8].strftime('%Y-%m-%d') if r[8] else None,
            impressions=r[9],
            clicks=r[10]
        ) for r in results]
    except Exception as e:
        logger.error(f"Ads fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ads fetch failed")

@app.get("/admin/ads/active", response_model=List[Ad])
async def get_active_ads():
    try:
        query = "SELECT ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks FROM ads WHERE status = 'active'"
        results = await execute_db_query(query, fetch="all")

        return [Ad(
            ad_id=r[0],
            title=r[1],
            subtitle=r[2],
            cta_text=r[3],
            cta_link=r[4],
            image_url=r[5],
            status=r[6],
            start_date=r[7].strftime('%Y-%m-%d') if r[7] else None,
            end_date=r[8].strftime('%Y-%m-%d') if r[8] else None,
            impressions=r[9],
            clicks=r[10]
        ) for r in results]
    except Exception as e:
        logger.error(f"Active ads fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Active ads fetch failed")

@app.post("/admin/ads", response_model=Ad)
async def create_ad(ad: AdCreate, admin_id: str = Depends(get_current_admin)):
    try:
        ad_id = str(uuid.uuid4())
        query = """
            INSERT INTO ads (ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0)
        """
        await execute_db_query(query, [
            ad_id,
            ad.title,
            ad.subtitle,
            ad.cta_text,
            str(ad.cta_link),
            str(ad.image_url),
            ad.status,
            ad.start_date,
            ad.end_date
        ])

        return Ad(**ad.dict(), ad_id=ad_id, impressions=0, clicks=0)
    except Exception as e:
        logger.error(f"Ad creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ad creation failed")

@app.put("/admin/ads/{ad_id}", response_model=Ad)
async def update_ad(ad_id: str, ad: AdCreate, admin_id: str = Depends(get_current_admin)):
    try:
        query = """
            UPDATE ads SET title = %s, subtitle = %s, cta_text = %s, cta_link = %s, image_url = %s, status = %s, start_date = %s, end_date = %s
            WHERE ad_id = %s
        """
        rowcount = await execute_db_query(query, [
            ad.title,
            ad.subtitle,
            ad.cta_text,
            str(ad.cta_link),
            str(ad.image_url),
            ad.status,
            ad.start_date,
            ad.end_date,
            ad_id
        ])

        if rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")

        query = "SELECT * FROM ads WHERE ad_id = %s"
        result = await execute_db_query(query, [ad_id], fetch="one")

        return Ad(
            ad_id=result[0],
            title=result[1],
            subtitle=result[2],
            cta_text=result[3],
            cta_link=str(result[4]),
            image_url=str(result[5]),
            status=result[6],
            start_date=result[7].strftime('%Y-%m-%d') if result[7] else None,
            end_date=result[8].strftime('%Y-%m-%d') if result[8] else None,
            impressions=result[9],
            clicks=result[10]
        )
    except Exception as e:
        logger.error(f"Ad update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ad update failed")

@app.put("/admin/ads/{ad_id}/status")
async def update_ad_status(ad_id: str, status_update: AdStatusUpdate, admin_id: str = Depends(get_current_admin)):
    try:
        query = "UPDATE ads SET status = %s WHERE ad_id = %s"
        rowcount = await execute_db_query(query, [status_update.status, ad_id])

        if rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")

        return {"status": "success", "message": "Ad status updated successfully"}
    except Exception as e:
        logger.error(f"Ad status update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ad status update failed")

@app.delete("/admin/ads/{ad_id}")
async def delete_ad(ad_id: str, admin_id: str = Depends(get_current_admin)):
    try:
        query = "DELETE FROM ads WHERE ad_id = %s"
        rowcount = await execute_db_query(query, [ad_id])

        if rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")

        return {"status": "success", "message": "Ad deleted successfully"}
    except Exception as e:
        logger.error(f"Ad deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ad deletion failed")

# Transaction management
@app.get("/admin/transactions", response_model=List[Transaction])
async def get_transactions(admin_id: str = Depends(get_current_admin)):
    try:
        query = """
            SELECT t.transaction_id, t.parent_id, t.student_id, t.amount_sent, t.fee, t.description, t.latitude, t.longitude, t.timestamp,
                   p.fullnames AS parent_name, s.student_name
            FROM transactions t
            LEFT JOIN parents p ON t.parent_id = p.parent_id
            LEFT JOIN students s ON t.student_id = s.student_id
        """
        results = await execute_db_query(query, fetch="all")

        return [Transaction(
            transaction_id=r[0],
            parent_id=r[1],
            student_id=r[2],
            amount_sent=float(r[3]),
            fee=float(r[4]),
            description=r[5],
            latitude=r[6],
            longitude=r[7],
            timestamp=r[8],
            parent={"fullnames": r[9]} if r[9] else None,
            student={"student_name": r[10]} if r[10] else None
        ) for r in results]
    except Exception as e:
        logger.error(f"Transactions fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Transactions fetch failed")

@app.get("/admin/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str, admin_id: str = Depends(get_current_admin)):
    try:
        query = """
            SELECT t.transaction_id, t.parent_id, t.student_id, t.amount_sent, t.fee, t.description, t.latitude, t.longitude, t.timestamp,
                   p.fullnames AS parent_name, s.student_name
            FROM transactions t
            LEFT JOIN parents p ON t.parent_id = p.parent_id
            LEFT JOIN students s ON t.student_id = s.student_id
            WHERE t.transaction_id = %s
        """
        result = await execute_db_query(query, [transaction_id], fetch="one")

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")

        return Transaction(
            transaction_id=result[0],
            parent_id=result[1],
            student_id=result[2],
            amount_sent=float(result[3]),
            fee=float(result[4]),
            description=result[5],
            latitude=result[6],
            longitude=result[7],
            timestamp=result[8],
            parent={"fullnames": result[9]} if result[9] else None,
            student={"student_name": result[10]} if result[10] else None
        )
    except Exception as e:
        logger.error(f"Transaction fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Transaction fetch failed")

# User management
@app.get("/admin/parents", response_model=List[Parent])
async def get_parents(admin_id: str = Depends(get_current_admin)):
    try:
        query = "SELECT parent_id, fullnames, phone_number, email, account_balance, created_at FROM parents"
        results = await execute_db_query(query, fetch="all")

        return [Parent(
            parent_id=r[0],
            fullnames=r[1],
            phone_number=r[2],
            email=r[3],
            account_balance=float(r[4]),
            created_at=r[5].strftime('%Y-%m-%d %H:%M:%S') if r[5] else None
        ) for r in results]
    except Exception as e:
        logger.error(f"Parents fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Parents fetch failed")

# Settings management
@app.get("/admin/settings", response_model=Settings)
async def get_settings(admin_id: str = Depends(get_current_admin)):
    try:
        query = """
            SELECT settings_id, platform_name, platform_email, support_email, max_transaction_amount, min_transaction_amount,
                   transaction_fee_percentage, require_two_factor_auth, password_min_length, session_timeout,
                   email_notifications, sms_notifications, push_notifications, primary_color, secondary_color
            FROM settings
            LIMIT 1
        """
        result = await execute_db_query(query, fetch="one")

        if not result:
            settings_id = str(uuid.uuid4())
            default_settings = {
                "settings_id": settings_id,
                "platform_name": "StudentPay",
                "platform_email": "admin@studentpay.com",
                "support_email": "support@studentpay.com",
                "max_transaction_amount": 1000.0,
                "min_transaction_amount": 5.0,
                "transaction_fee_percentage": 2.5,
                "require_two_factor_auth": True,
                "password_min_length": 8,
                "session_timeout": 30,
                "email_notifications": True,
                "sms_notifications": False,
                "push_notifications": True,
                "primary_color": "#059669",
                "secondary_color": "#10b981"
            }
            query = """
                INSERT INTO settings (settings_id, platform_name, platform_email, support_email, max_transaction_amount,
                                     min_transaction_amount, transaction_fee_percentage, require_two_factor_auth,
                                     password_min_length, session_timeout, email_notifications, sms_notifications,
                                     push_notifications, primary_color, secondary_color)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            await execute_db_query(query, list(default_settings.values()))

            return Settings(**default_settings)

        return Settings(
            settings_id=result[0],
            platform_name=result[1],
            platform_email=result[2],
            support_email=result[3],
            max_transaction_amount=float(result[4]),
            min_transaction_amount=float(result[5]),
            transaction_fee_percentage=float(result[6]),
            require_two_factor_auth=bool(result[7]),
            password_min_length=result[8],
            session_timeout=result[9],
            email_notifications=bool(result[10]),
            sms_notifications=bool(result[11]),
            push_notifications=bool(result[12]),
            primary_color=result[13],
            secondary_color=result[14]
        )
    except Exception as e:
        logger.error(f"Settings fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Settings fetch failed")

@app.put("/admin/settings", response_model=Settings)
async def update_settings(settings: SettingsUpdate, admin_id: str = Depends(get_current_admin)):
    try:
        query = "SELECT settings_id FROM settings LIMIT 1"
        result = await execute_db_query(query, fetch="one")
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Settings not found")
        settings_id = result[0]

        update_fields = []
        update_values = []
        for field, value in settings.dict(exclude_unset=True, exclude={'status'}).items():
            update_fields.append(f"{field} = %s")
            update_values.append(value)

        if not update_fields:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

        query = f"UPDATE settings SET {', '.join(update_fields)} WHERE settings_id = %s"
        update_values.append(settings_id)
        await execute_db_query(query, update_values)

        return await get_settings(admin_id=admin_id)
    except Exception as e:
        logger.error(f"Settings update failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Settings update failed")

# Initialize admin on startup
@app.on_event("startup")
async def startup_event():
    for attempt in range(3):
        try:
            await initDb_admin()
            break
        except Exception as e:
            logger.warning(f"Startup attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:
                logger.error("Failed to initialize admin after retries")
                raise
            await asyncio.sleep(1)