# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request  # Add Request import
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl, validator
from typing import Optional, List
from db.connection import db_Query
from datetime import datetime, timedelta
import uuid
from passlib.context import CryptContext
from jose import jwt, JWTError
import secrets

# FastAPI app initialization
app = FastAPI()



# Add trusted websites
origins = [
    "*",
    "http://localhost:8080",
     "http://localhost:8001" # Your frontend# Your frontend
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"]   # Allow all headers
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

# Pydantic models (unchanged)
class Admin(BaseModel):
    status:Optional[str] = None
    admin_id: str
    fullnames: str
    email: EmailStr
    password_hash: str
    created_at: str
    last_activity: str

class AdminLogin(BaseModel):
    status:Optional[str] = None
    email: EmailStr
    password: str

class Token(BaseModel):
    status:Optional[str] = None
    access_token: str
    token_type: str

class CSRFToken(BaseModel):
    status:Optional[str] = None
    csrf_token: str

class Product(BaseModel):
    status:Optional[str] = None
    product_id: str
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class ProductCreate(BaseModel):
    status:Optional[str] = None
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class Ad(BaseModel):
    status:Optional[str] = None
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
    status:Optional[str] = None
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
    status:Optional[str] = None
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
    status:Optional[str] = None
    parent_id: str
    fullnames: str
    phone_number: str
    email: Optional[EmailStr] = None
    account_balance: float
    created_at: str

class Settings(BaseModel):
    status:Optional[str] = None
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
    status:Optional[str] = None
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
    return pwd_context.hash(password)

async def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)

async def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "sub": data.get("admin_id")})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        admin_id: str = payload.get("sub")
        if admin_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        query = "UPDATE admins SET last_activity = %s WHERE admin_id = %s"
        db_Query.execute(query, [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), admin_id])
        
        
        return admin_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed")

async def verify_csrf_token(csrf_token: str, session_id: str):
    query = "SELECT token FROM csrf_tokens WHERE token = %s AND session_id = %s AND expires_at > NOW()"
    db_Query.execute(query, [csrf_token, session_id])
    if not db_Query.fetchone():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired CSRF token")

# Dependency to get CSRF token and session ID
async def get_csrf_and_session(request: Request):
    csrf_token = request.headers.get("x-csrf-token")
    session_id = request.cookies.get("session_id")
    if not csrf_token or not session_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing CSRF token or session ID")
    return csrf_token, session_id

# CSRF token generation
@app.get("/admin/get-csrf-token", response_model=CSRFToken)
async def get_csrf_token(response: Response):
    csrf_token = secrets.token_hex(16)
    session_id = str(uuid.uuid4())
    created_at = datetime.now()
    expires_at = created_at + timedelta(minutes=CSRF_TOKEN_EXPIRE_MINUTES)
    
    query = "INSERT INTO csrf_tokens (token, session_id, created_at, expires_at) VALUES (%s, %s, %s, %s)"
    db_Query.execute(query, [csrf_token, session_id, created_at, expires_at])
    
    
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True, samesite="strict")
    return {"status":"success","csrf_token": csrf_token}

# Admin initialization
async def initDb_admin():
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
    db_Query.execute(query, [admin_data["email"]])
    
    if db_Query.fetchone():
        print("Admin already exists")
        return "Admin already exists"
    
    query = "INSERT INTO admins (admin_id, fullnames, email, password_hash, created_at, last_activity) VALUES (%s, %s, %s, %s, %s, %s)"
    db_Query.execute(query, [
        admin_data["admin_id"],
        admin_data["fullnames"],
        admin_data["email"],
        admin_data["password_hash"],
        admin_data["created_at"],
        admin_data["last_activity"]
    ])
    

# Admin login
@app.post("/admin/login", response_model=Token)
async def admin_login(login: AdminLogin):
    query = "SELECT email, password_hash, admin_id FROM admins WHERE email = %s"
    db_Query.execute(query, [login.email])
    db_result = db_Query.fetchone()
    
    if not db_result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    email, hash_password, admin_id = db_result
    if not await verify_password(login.password, hash_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    token = await create_access_token({"admin_id": admin_id})
    return {"status":"success","access_token": token, "token_type": "bearer"}

# Admin profile
@app.get("/admin/profile")
async def admin_profile(admin_id: str = Depends(get_current_admin)):
    query = "SELECT admin_id, fullnames, email, created_at, last_activity FROM admins WHERE admin_id = %s"
    db_Query.execute(query, [admin_id])
    result = db_Query.fetchone()
    
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")
    
    return {"status":"success",
        "admin_id": result[0],
        "fullnames": result[1],
        "email": result[2],
        "created_at": result[3],
        "last_activity": result[4]
    }

# Admin dashboard stats (fixed endpoint)
@app.get("/admin/stats")
async def get_stats(admin_id: str = Depends(get_current_admin)):
   # csrf_token, session_id = csrf_data
   # await verify_csrf_token(csrf_token, session_id)
    
    stats = {
        "totalUsers": 0,
        "totalTransactions": 0,
        "totalRevenue": 0,
        "activeAds": 0
    }
    
    # Total users
    query = "SELECT COUNT(*) FROM parents"
    db_Query.execute(query)
    stats["totalUsers"] = db_Query.fetchone()[0]
    
    # Total transactions
    query = "SELECT COUNT(*) FROM transactions"
    db_Query.execute(query)
    stats["totalTransactions"] = db_Query.fetchone()[0]
    
    # Total revenue (sum of fees)
    query = "SELECT SUM(fee) FROM transactions"
    db_Query.execute(query)
    revenue = db_Query.fetchone()[0]
    stats["totalRevenue"] = float(revenue) if revenue else 0.0
    
    # Active ads
    query = "SELECT COUNT(*) FROM ads WHERE status = 'active'"
    db_Query.execute(query)
    stats["activeAds"] = db_Query.fetchone()[0]
    
    return stats

# Product management (update all endpoints with CSRF dependency)
@app.get("/admin/products", response_model=List[Product])
async def get_products(admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "SELECT product_id, name, price, affiliate_link, description, image_url FROM products"
    db_Query.execute(query)
    results = db_Query.fetchall()
    
    return [Product(
        product_id=r[0],
        name=r[1],
        price=r[2],
        affiliate_link=r[3],
        description=r[4],
        image_url=r[5]
    ) for r in results]

@app.get("/admin/products/shop", response_model=List[Product])
async def get_shop_products(csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "SELECT product_id, name, price, affiliate_link, description, image_url FROM products"
    db_Query.execute(query)
    results = db_Query.fetchall()
    
    return [Product(
        product_id=r[0],
        name=r[1],
        price=r[2],
        affiliate_link=r[3],
        description=r[4],
        image_url=r[5]
    ) for r in results]

@app.post("/admin/products", response_model=Product)
async def create_product(product: ProductCreate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    product_id = str(uuid.uuid4())
    query = """
        INSERT INTO products (product_id, name, price, affiliate_link, description, image_url)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    db_Query.execute(query, [product_id, product.name, product.price, str(product.affiliate_link), str(product.description), str(product.image_url)])
    
    
    return Product(**product.dict(), product_id=product_id)

@app.put("/admin/products/{product_id}", response_model=Product)
async def update_product(product_id: str, product: ProductCreate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id) 
    
    query = """
        UPDATE products SET name = %s, price = %s, affiliate_link = %s, description = %s, image_url = %s
        WHERE product_id = %s
    """
    db_Query.execute(query, [product.name, product.price, str(product.affiliate_link), product.description, str(product.image_url), product_id])
    if db_Query.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    
    
    return Product(**product.dict(), product_id=product_id)

@app.delete("/admin/products/{product_id}")
async def delete_product(product_id: str, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "DELETE FROM products WHERE product_id = %s"
    db_Query.execute(query, [product_id])
    if db_Query.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    
    
    return {"status":"success","message": "Product deleted successfully"}

# Ad management
@app.get("/admin/ads", response_model=List[Ad])
async def get_ads(admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "SELECT ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks FROM ads"
    db_Query.execute(query)
    results = db_Query.fetchall()
    
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

@app.get("/admin/ads/active", response_model=List[Ad])
async def get_active_ads():
    query = "SELECT ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks FROM ads WHERE status = 'active'"
    db_Query.execute(query)
    results = db_Query.fetchall()
    
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

@app.post("/admin/ads", response_model=Ad)
async def create_ad(ad: AdCreate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    ad_id = str(uuid.uuid4())
    query = """
        INSERT INTO ads (ad_id, title, subtitle, cta_text, cta_link, image_url, status, start_date, end_date, impressions, clicks)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0)
    """
    db_Query.execute(query, [
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

@app.put("/admin/ads/{ad_id}", response_model=Ad)
async def update_ad(ad_id: str, ad: AdCreate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = """
        UPDATE ads SET title = %s, subtitle = %s, cta_text = %s, cta_link = %s, image_url = %s, status = %s, start_date = %s, end_date = %s
        WHERE ad_id = %s
    """
    db_Query.execute(query, [ad.title, ad.subtitle, ad.cta_text,str(ad.cta_link), str(ad.image_url), ad.status, ad.start_date, ad.end_date, ad_id])
    if db_Query.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")
    
    
    query = "SELECT * FROM ads WHERE ad_id = %s"
    db_Query.execute(query, [ad_id])
    result = db_Query.fetchone()
    
    return Ad(
        ad_id=result[0],
        title=result[1],
        subtitle=result[2],
        cta_text=result[3],
        cta_link=str(result[4]),
        image_url=str(result[5]),
        status=result[6],
        start_date=result[7],
        end_date=result[8],
        impressions=result[9],
        clicks=result[10]
    )

@app.put("/admin/ads/{ad_id}/status")
async def update_ad_status(ad_id: str, status_update: AdStatusUpdate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "UPDATE ads SET status = %s WHERE ad_id = %s"
    db_Query.execute(query, [status_update.status, ad_id])
    if db_Query.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")
    
    
    return {"status":"success","message": "Ad status updated successfully"}

@app.delete("/admin/ads/{ad_id}")
async def delete_ad(ad_id: str, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "DELETE FROM ads WHERE ad_id = %s"
    db_Query.execute(query, [ad_id])
    if db_Query.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ad not found")
    
    
    return {"status":"success","message": "Ad deleted successfully"}

# Transaction management
@app.get("/admin/transactions", response_model=List[Transaction])
async def get_transactions(admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = """
        SELECT t.transaction_id, t.parent_id, t.student_id, t.amount_sent, t.fee, t.description, t.latitude, t.longitude, t.timestamp,
               p.fullnames AS parent_name, s.student_name
        FROM transactions t
        LEFT JOIN parents p ON t.parent_id = p.parent_id
        LEFT JOIN students s ON t.student_id = s.student_id
    """
    db_Query.execute(query)
    results = db_Query.fetchall()
    
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

@app.get("/admin/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = """
        SELECT t.transaction_id, t.parent_id, t.student_id, t.amount_sent, t.fee, t.description, t.latitude, t.longitude, t.timestamp,
               p.fullnames AS parent_name, s.student_name
        FROM transactions t
        LEFT JOIN parents p ON t.parent_id = p.parent_id
        LEFT JOIN students s ON t.student_id = s.student_id
        WHERE t.transaction_id = %s
    """
    db_Query.execute(query, [transaction_id])
    result = db_Query.fetchone()
    
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

# User management
@app.get("/admin/parents", response_model=List[Parent])
async def get_parents(admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "SELECT parent_id, fullnames, phone_number, email, account_balance, created_at FROM parents"
    db_Query.execute(query)
    results = db_Query.fetchall()
    
    return [Parent(
        parent_id=r[0],
        fullnames=r[1],
        phone_number=r[2],
        email=r[3],
        account_balance=float(r[4]),
        created_at=r[5].strftime('%Y-%m-%d %H:%M:%S') if r[5] else None
    ) for r in results]

# Settings management
@app.get("/admin/settings", response_model=Settings)
async def get_settings(admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = """
        SELECT settings_id, platform_name, platform_email, support_email, max_transaction_amount, min_transaction_amount,
               transaction_fee_percentage, require_two_factor_auth, password_min_length, session_timeout,
               email_notifications, sms_notifications, push_notifications, primary_color, secondary_color
        FROM settings
        LIMIT 1
    """
    db_Query.execute(query)
    result = db_Query.fetchone()
    
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
        db_Query.execute(query, list(default_settings.values()))
        
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

@app.put("/admin/settings", response_model=Settings)
async def update_settings(settings: SettingsUpdate, admin_id: str = Depends(get_current_admin), csrf_data: tuple = Depends(get_csrf_and_session)):
    csrf_token, session_id = csrf_data
    await verify_csrf_token(csrf_token, session_id)
    
    query = "SELECT settings_id FROM settings LIMIT 1"
    db_Query.execute(query)
    result = db_Query.fetchone()
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
    db_Query.execute(query, update_values)
    
    
    return await get_settings(admin_id=admin_id, csrf_data=csrf_data)

# Initialize admin on startup
@app.on_event("startup")
async def startup_event():
    await initDb_admin()
    
    
