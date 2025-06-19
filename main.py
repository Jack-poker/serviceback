import json
from datetime import datetime, timedelta
from typing import Optional, List
import uuid
import secrets
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from pydantic import BaseModel, EmailStr, HttpUrl
from jose import jwt, JWTError
from passlib.context import CryptContext

# Pydantic models (status field removed)
class Admin(BaseModel):
    admin_id: str
    fullnames: str
    email: EmailStr
    password_hash: str
    created_at: str
    last_activity: str

class AdminLogin(BaseModel):
    email: EmailStr
    password: str
    csrf_token: str

class Token(BaseModel):
    access_token: str
    token_type: str

class CSRFToken(BaseModel):
    csrf_token: str

class Product(BaseModel):
    product_id: str
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class ProductCreate(BaseModel):
    name: str
    price: float
    affiliate_link: HttpUrl
    description: Optional[str] = None
    image_url: Optional[HttpUrl] = None

class Ad(BaseModel):
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
    parent_id: str
    fullnames: str
    phone_number: str
    email: Optional[EmailStr] = None
    account_balance: float
    created_at: str

class Settings(BaseModel):
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

# Security configurations
SECRET_KEY =  secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
CSRF_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize Appwrite client
def init_appwrite(req):
    client = Client()
    client.set_endpoint("https://app.kaascan.com/v1")
    client.set_project("6843680c002d4c2afcc6")
    client.set_key("standard_0db4d53bae8f2281e896df8158d14cb67b7d6aded0db6b242e6e457b3794c5d9fbab2cd18391126302abc60be76750d2b84672f3bb823317b1974d1a8afc03a2cef2f642b13f8a4bd1016c03706ae1d6dee0908ab0e6a9c2f243f8ffb0510635d4eb8edeb865bba31b26c889195c039f027453a22b0e03a949511dd9ae9cf7f7")
    return Databases(client)

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

async def get_current_admin(token: str, db: Databases):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        admin_id: str = payload.get("sub")
        if not admin_id:
            return {"status": "error", "detail": "Invalid token"}, 401
        # Update last_activity
        db.update_document(
            collection_id="admins",
            document_id=admin_id,
            data={"last_activity": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        )
        return admin_id, 200
    except JWTError:
        return {"status": "error", "detail": "Token verification failed"}, 401

async def verify_csrf_token(csrf_token: str, session_id: str, db: Databases):
    try:
        documents = db.list_documents(
            collection_id="csrf_tokens",
            queries=[Query.equal("token", csrf_token), Query.equal("session_id", session_id)]
        )["documents"]
        if not documents or datetime.fromisoformat(documents[0]["expires_at"]) <= datetime.now():
            return {"status": "error", "detail": "Invalid or expired CSRF token"}, 403
        return None, 200
    except Exception as e:
        return {"status": "error", "detail": f"CSRF verification failed: {str(e)}"}, 403

# Initialize database collections
async def init_collections(db: Databases):
    collections = [
        {
            "id": "admins",
            "attributes": [
                {"key": "fullnames", "type": "string", "required": True},
                {"key": "email", "type": "string", "required": True},
                {"key": "password_hash", "type": "string", "required": True},
                {"key": "created_at", "type": "string", "required": True},
                {"key": "last_activity", "type": "string", "required": True}
            ]
        },
        {
            "id": "csrf_tokens",
            "attributes": [
                {"key": "token", "type": "string", "required": True},
                {"key": "session_id", "type": "string", "required": True},
                {"key": "created_at", "type": "string", "required": True},
                {"key": "expires_at", "type": "string", "required": True}
            ]
        },
        {
            "id": "products",
            "attributes": [
                {"key": "name", "type": "string", "required": True},
                {"key": "price", "type": "double", "required": True},
                {"key": "affiliate_link", "type": "string", "required": True},
                {"key": "description", "type": "string", "required": False},
                {"key": "image_url", "type": "string", "required": False}
            ]
        },
        {
            "id": "ads",
            "attributes": [
                {"key": "title", "type": "string", "required": True},
                {"key": "subtitle", "type": "string", "required": True},
                {"key": "cta_text", "type": "string", "required": True},
                {"key": "cta_link", "type": "string", "required": True},
                {"key": "image_url", "type": "string", "required": True},
                {"key": "status", "type": "string", "required": True},
                {"key": "start_date", "type": "string", "required": True},
                {"key": "end_date", "type": "string", "required": True},
                {"key": "impressions", "type": "integer", "required": True},
                {"key": "clicks", "type": "integer", "required": True}
            ]
        },
        {
            "id": "transactions",
            "attributes": [
                {"key": "parent_id", "type": "string", "required": True},
                {"key": "student_id", "type": "string", "required": False},
                {"key": "amount_sent", "type": "double", "required": True},
                {"key": "fee", "type": "double", "required": True},
                {"key": "description", "type": "string", "required": False},
                {"key": "latitude", "type": "double", "required": False},
                {"key": "longitude", "type": "double", "required": False},
                {"key": "timestamp", "type": "string", "required": True}
            ]
        },
        {
            "id": "parents",
            "attributes": [
                {"key": "fullnames", "type": "string", "required": True},
                {"key": "phone_number", "type": "string", "required": True},
                {"key": "email", "type": "string", "required": False},
                {"key": "account_balance", "type": "double", "required": True},
                {"key": "created_at", "type": "string", "required": True}
            ]
        },
        {
            "id": "settings",
            "attributes": [
                {"key": "platform_name", "type": "string", "required": True},
                {"key": "platform_email", "type": "string", "required": True},
                {"key": "support_email", "type": "string", "required": True},
                {"key": "max_transaction_amount", "type": "double", "required": True},
                {"key": "min_transaction_amount", "type": "double", "required": True},
                {"key": "transaction_fee_percentage", "type": "double", "required": True},
                {"key": "require_two_factor_auth", "type": "boolean", "required": True},
                {"key": "password_min_length", "type": "integer", "required": True},
                {"key": "session_timeout", "type": "integer", "required": True},
                {"key": "email_notifications", "type": "boolean", "required": True},
                {"key": "sms_notifications", "type": "boolean", "required": True},
                {"key": "push_notifications", "type": "boolean", "required": True},
                {"key": "primary_color", "type": "string", "required": True},
                {"key": "secondary_color", "type": "string", "required": True}
            ]
        }
    ]
    for collection in collections:
        try:
            db.create_collection(
                collection_id=collection["id"],
                name=collection["id"],
                permission="document",
                read=["role:all"],
                write=["role:admin"]
            )
            for attr in collection["attributes"]:
                db.create_attribute(
                    collection_id=collection["id"],
                    key=attr["key"],
                    type=attr["type"],
                    required=attr["required"],
                    size=255 if attr["type"] == "string" else None
                )
        except:
            pass  # Collection or attribute may already exist

async def init_admin(db: Databases):
    admin = Admin(
        admin_id=str(uuid.uuid4()),
        fullnames="Tuyishimire Emmanuel",
        email="tuyishimireemmanuel24@gmail.com",
        password_hash=await hash_password("admin123"),
        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        last_activity=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    try:
        documents = db.list_documents(collection_id="admins", queries=[Query.equal("email", admin.email)])["documents"]
        if documents:
            print("Admin already exists")
            return
        db.create_document(
            collection_id="admins",
            document_id=admin.admin_id,
            data=admin.dict()
        )
    except Exception as e:
        print(f"Failed to initialize admin: {str(e)}")

# Main Appwrite function
def main(req, res):
    db = init_appwrite(req)
    
    # Initialize collections and admin on first execution
    if req.env.get("APPWRITE_FUNCTION_TRIGGER") == "http":
        init_collections(db)
        init_admin(db)

    # Parse request payload
    try:
        payload = json.loads(req.payload) if req.payload else {}
    except json.JSONDecodeError:
        return res.json({"status": "error", "detail": "Invalid JSON payload"}, status=400)

    # Extract headers
    auth_header = req.headers.get("authorization", "")
    csrf_token = req.headers.get("x-csrf-token", "")
    session_id = req.headers.get("cookie", "").split("session_id=")[-1] if "session_id=" in req.headers.get("cookie", "") else ""

    # Route handling
    path = req.path
    method = req.method

    # CSRF Token Generation
    if path == "/admin/get-csrf-token" and method == "GET":
        try:
            csrf_token = secrets.token_hex(16)
            session_id = str(uuid.uuid4())
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            expires_at = (datetime.now() + timedelta(minutes=CSRF_TOKEN_EXPIRE_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            db.create_document(
                collection_id="csrf_tokens",
                document_id=str(uuid.uuid4()),
                data={"token": csrf_token, "session_id": session_id, "created_at": created_at, "expires_at": expires_at}
            )
            response = res.json({"status": "success", "csrf_token": csrf_token}, status=200)
            response.set_cookie(key="session_id", value=session_id, httponly=True, secure=True, samesite="strict")
            return response
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to generate CSRF token: {str(e)}"}, status=500)

    # Admin Login
    if path == "/admin/login" and method == "POST":
        try:
            login_data = AdminLogin(**payload)
            if not session_id or not csrf_token:
                return res.json({"status": "error", "detail": "Missing CSRF token or session ID"}, status=403)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="admins", queries=[Query.equal("email", login_data.email)])["documents"]
            if not documents:
                return res.json({"status": "error", "detail": "Invalid credentials"}, status=401)
            admin = documents[0]
            if not verify_password(login_data.password, admin["password_hash"]):
                return res.json({"status": "error", "detail": "Invalid credentials"}, status=401)
            token = create_access_token({"admin_id": admin["$id"]})
            return res.json({"status": "success", "access_token": token, "token_type": "bearer"}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Login failed: {str(e)}"}, status=400)

    # Admin Profile
    if path == "/admin/profile" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            document = db.get_document(collection_id="admins", document_id=admin_id)
            if not document:
                return res.json({"status": "error", "detail": "Admin not found"}, status=404)
            return res.json({
                "status": "success",
                "admin_id": document["$id"],
                "fullnames": document["fullnames"],
                "email": document["email"],
                "created_at": document["created_at"],
                "last_activity": document["last_activity"]
            }, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch profile: {str(e)}"}, status=500)

    # Admin Stats
    if path == "/admin/stats" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            stats = {
                "totalUsers": db.list_documents(collection_id="parents")["total"],
                "totalTransactions": db.list_documents(collection_id="transactions")["total"],
                "totalRevenue": sum(
                    doc["fee"] for doc in db.list_documents(collection_id="transactions")["documents"]
                ),
                "activeAds": db.list_documents(collection_id="ads", queries=[Query.equal("status", "active")])["total"]
            }
            return res.json({"status": "success", **stats}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch stats: {str(e)}"}, status=500)

    # Product Management
    if path == "/admin/products" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="products")["documents"]
            products = [
                Product(
                    product_id=doc["$id"],
                    name=doc["name"],
                    price=doc["price"],
                    affiliate_link=doc["affiliate_link"],
                    description=doc["description"],
                    image_url=doc["image_url"]
                ).dict() for doc in documents
            ]
            return res.json({"status": "success", "products": products}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch products: {str(e)}"}, status=500)

    if path == "/admin/products/shop" and method == "GET":
        try:
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="products")["documents"]
            products = [
                Product(
                    product_id=doc["$id"],
                    name=doc["name"],
                    price=doc["price"],
                    affiliate_link=doc["affiliate_link"],
                    description=doc["description"],
                    image_url=doc["image_url"]
                ).dict() for doc in documents
            ]
            return res.json({"status": "success", "products": products}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch shop products: {str(e)}"}, status=500)

    if path.startswith("/admin/products") and method == "POST":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            product_data = ProductCreate(**payload)
            product_id = str(uuid.uuid4())
            db.create_document(
                collection_id="products",
                document_id=product_id,
                data=product_data.dict()
            )
            return res.json({"status": "success", **Product(**product_data.dict(), product_id=product_id).dict()}, status=201)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to create product: {str(e)}"}, status=400)

    if path.startswith("/admin/products/") and method == "PUT":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            product_id = path.split("/")[-1]
            product_data = ProductCreate(**payload)
            document = db.update_document(
                collection_id="products",
                document_id=product_id,
                data=product_data.dict()
            )
            if not document:
                return res.json({"status": "error", "detail": "Product not found"}, status=404)
            return res.json({"status": "success", **Product(**product_data.dict(), product_id=product_id).dict()}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to update product: {str(e)}"}, status=400)

    if path.startswith("/admin/products/") and method == "DELETE":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            product_id = path.split("/")[-1]
            db.delete_document(collection_id="products", document_id=product_id)
            return res.json({"status": "success", "message": "Product deleted successfully"}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to delete product: {str(e)}"}, status=400)

    # Ad Management
    if path == "/admin/ads" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="ads")["documents"]
            ads = [
                Ad(
                    ad_id=doc["$id"],
                    title=doc["title"],
                    subtitle=doc["subtitle"],
                    cta_text=doc["cta_text"],
                    cta_link=doc["cta_link"],
                    image_url=doc["image_url"],
                    status=doc["status"],
                    start_date=doc["start_date"],
                    end_date=doc["end_date"],
                    impressions=doc["impressions"],
                    clicks=doc["clicks"]
                ).dict() for doc in documents
            ]
            return res.json({"status": "success", "ads": ads}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch ads: {str(e)}"}, status=500)

    if path == "/admin/ads/active" and method == "GET":
        try:
            documents = db.list_documents(collection_id="ads", queries=[Query.equal("status", "active")])["documents"]
            ads = [
                Ad(
                    ad_id=doc["$id"],
                    title=doc["title"],
                    subtitle=doc["subtitle"],
                    cta_text=doc["cta_text"],
                    cta_link=doc["cta_link"],
                    image_url=doc["image_url"],
                    status=doc["status"],
                    start_date=doc["start_date"],
                    end_date=doc["end_date"],
                    impressions=doc["impressions"],
                    clicks=doc["clicks"]
                ).dict() for doc in documents
            ]
            return res.json({"status": "success", "ads": ads}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch active ads: {str(e)}"}, status=500)

    if path == "/admin/ads" and method == "POST":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            ad_data = AdCreate(**payload)
            ad_id = str(uuid.uuid4())
            db.create_document(
                collection_id="ads",
                document_id=ad_id,
                data={**ad_data.dict(), "impressions": 0, "clicks": 0}
            )
            return res.json({"status": "success", **Ad(**ad_data.dict(), ad_id=ad_id, impressions=0, clicks=0).dict()}, status=201)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to create ad: {str(e)}"}, status=400)

    if path.startswith("/admin/ads/") and path.endswith("/status") and method == "PUT":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            ad_id = path.split("/")[-2]
            status_data = AdStatusUpdate(**payload)
            document = db.update_document(
                collection_id="ads",
                document_id=ad_id,
                data={"status": status_data.status}
            )
            if not document:
                return res.json({"status": "error", "detail": "Ad not found"}, status=404)
            return res.json({"status": "success", "message": "Ad status updated successfully"}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to update ad status: {str(e)}"}, status=400)

    if path.startswith("/admin/ads/") and method == "PUT":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            ad_id = path.split("/")[-1]
            ad_data = AdCreate(**payload)
            document = db.update_document(
                collection_id="ads",
                document_id=ad_id,
                data=ad_data.dict()
            )
            if not document:
                return res.json({"status": "error", "detail": "Ad not found"}, status=404)
            return res.json({"status": "success", **Ad(**ad_data.dict(), ad_id=ad_id, impressions=document["impressions"], clicks=document["clicks"]).dict()}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to update ad: {str(e)}"}, status=400)

    if path.startswith("/admin/ads/") and method == "DELETE":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            ad_id = path.split("/")[-1]
            db.delete_document(collection_id="ads", document_id=ad_id)
            return res.json({"status": "success", "message": "Ad deleted successfully"}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to delete ad: {str(e)}"}, status=400)

    # Transaction Management
    if path == "/admin/transactions" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            transactions = db.list_documents(collection_id="transactions")["documents"]
            results = []
            for t in transactions:
                parent = db.get_document(collection_id="parents", document_id=t["parent_id"]) if t["parent_id"] else None
                student = db.get_document(collection_id="students", document_id=t["student_id"]) if t["student_id"] else None
                results.append(Transaction(
                    transaction_id=t["$id"],
                    parent_id=t["parent_id"],
                    student_id=t["student_id"],
                    amount_sent=float(t["amount_sent"]),
                    fee=float(t["fee"]),
                    description=t["description"],
                    latitude=t["latitude"],
                    longitude=t["longitude"],
                    timestamp=t["timestamp"],
                    parent={"fullnames": parent["fullnames"]} if parent else None,
                    student={"student_name": student["student_name"]} if student else None
                ).dict())
            return res.json({"status": "success", "transactions": results}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch transactions: {str(e)}"}, status=500)

    if path.startswith("/admin/transactions/") and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            transaction_id = path.split("/")[-1]
            t = db.get_document(collection_id="transactions", document_id=transaction_id)
            if not t:
                return res.json({"status": "error", "detail": "Transaction not found"}, status=404)
            parent = db.get_document(collection_id="parents", document_id=t["parent_id"]) if t["parent_id"] else None
            student = db.get_document(collection_id="students", document_id=t["student_id"]) if t["student_id"] else None
            transaction = Transaction(
                transaction_id=t["$id"],
                parent_id=t["parent_id"],
                student_id=t["student_id"],
                amount_sent=float(t["amount_sent"]),
                fee=float(t["fee"]),
                description=t["description"],
                latitude=t["latitude"],
                longitude=t["longitude"],
                timestamp=t["timestamp"],
                parent={"fullnames": parent["fullnames"]} if parent else None,
                student={"student_name": student["student_name"]} if student else None
            ).dict()
            return res.json({"status": "success", **transaction}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch transaction: {str(e)}"}, status=400)

    # User Management
    if path == "/admin/parents" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="parents")["documents"]
            parents = [
                Parent(
                    parent_id=doc["$id"],
                    fullnames=doc["fullnames"],
                    phone_number=doc["phone_number"],
                    email=doc["email"],
                    account_balance=float(doc["account_balance"]),
                    created_at=doc["created_at"]
                ).dict() for doc in documents
            ]
            return res.json({"status": "success", "parents": parents}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch parents: {str(e)}"}, status=500)

    # Settings Management
    if path == "/admin/settings" and method == "GET":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            documents = db.list_documents(collection_id="settings")["documents"]
            if not documents:
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
                db.create_document(collection_id="settings", document_id=settings_id, data=default_settings)
                return res.json({"status": "success", **Settings(**default_settings).dict()}, status=200)
            settings = Settings(**{k: v for k, v in documents[0].items() if k != "$id"}, settings_id=documents[0]["$id"]).dict()
            return res.json({"status": "success", **settings}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to fetch settings: {str(e)}"}, status=500)

    if path == "/admin/settings" and method == "PUT":
        try:
            token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
            admin_id, status_code = get_current_admin(token, db)
            if isinstance(admin_id, dict):
                return res.json(admin_id, status=status_code)
            error, status_code = verify_csrf_token(csrf_token, session_id, db)
            if error:
                return res.json(error, status=status_code)
            settings_data = SettingsUpdate(**payload)
            documents = db.list_documents(collection_id="settings")["documents"]
            if not documents:
                return res.json({"status": "error", "detail": "Settings not found"}, status=404)
            settings_id = documents[0]["$id"]
            update_data = {k: v for k, v in settings_data.dict(exclude_unset=True).items()}
            if not update_data:
                return res.json({"status": "error", "detail": "No fields to update"}, status=400)
            db.update_document(collection_id="settings", document_id=settings_id, data=update_data)
            updated_doc = db.get_document(collection_id="settings", document_id=settings_id)
            return res.json({"status": "success", **Settings(**{k: v for k, v in updated_doc.items() if k != "$id"}, settings_id=settings_id).dict()}, status=200)
        except Exception as e:
            return res.json({"status": "error", "detail": f"Failed to update settings: {str(e)}"}, status=400)

    return res.json({"status": "error", "detail": "Endpoint not found"}, status=404)
