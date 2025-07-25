import json
import random
from bcrypt import checkpw
import mysql.connector
import requests
import hashlib
from datetime import datetime, timedelta
# from db.connection import  db_Query
import threading
from decimal import Decimal



latitude = 123.456  # Replace with real value or geolocation source
longitude = 78.901


db_connection = mysql.connector.connect(
    host="167.86.94.189",         # or your MySQL server IP
    user="root",     # MySQL username
    password="emmy@0790467621", # MySQL password
    database="kaascan_db",  # Name of your 
    autocommit=True  # ✅ AUTO COMMIT ON
    
    
)
db_Query = db_connection.cursor()


# Configuration
username = "testa"
accountno = 250160000011
partnerpassword = "+$J<wtZktTDs&-Mk(\"h5=<PH#Jf769P5/Z<*xbR~"
transaction_fee = 0  # Adjust based on your fee structure



# production
# username = "kacafix.tech"
# accountno = 250240008814
# partnerpassword = "@sXb$9|%!V;^D~o},){CDd_]'-YOsNMwO~eSjRA/"

# Generate timestamp and password
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
concatenated_string = username + str(accountno) + partnerpassword + timestamp
password = hashlib.sha256(concatenated_string.encode()).hexdigest()
data = {
    'username': username,
    'timestamp': timestamp,
    'password': password
}

# Lock for transaction ID generation
txid_lock = threading.Lock()

async def verify_withdraw_otp(phone, otp_code):
    """Verify OTP code for a given phone number."""
    db_Query.execute("SELECT otp_code FROM parents WHERE phone_number = %s", (phone,))
    fetched_otpcode = db_Query.fetchone()
    if fetched_otpcode and fetched_otpcode[0] == otp_code:
        return True
    return False

async def send_otp(phone):
    """Send OTP via SMS to the specified phone number."""
    sms_username = "admin"
    sms_password = "kaascan.tech@5749840"
    
    
    db_Query.execute("SELECT phone_number FROM parents WHERE phone_number = %s", (phone,))
    phonenumber = db_Query.fetchone()
    
    if phonenumber and len(phonenumber[0]) == 10:
        otp_code = str(random.randint(1000, 9999))
        otp_message = f"Your verification code is {otp_code}"
        print(otp_message)
        
        # Store OTP in the database
        db_Query.execute("UPDATE parents SET otp_code = %s WHERE phone_number = %s", (otp_code, phone))
        
        sms_data = {
            'recipients': phonenumber[0],
            'message': otp_message,
            'otp_code': otp_code
        }
        try:
            response = requests.post(
                'https://automation.kaascan.com/webhook/sms',
                data=sms_data,
                auth=(sms_username, sms_password)
            )
            print(response.text)
            return response.text
        except Exception as e:
            print(f"🔴 Error sending OTP: {e}")
            return None
    return None

def accountBalance():
    """Check the system account balance."""
    balance_data = {
        'username': username,
        'timestamp': timestamp,
        'password': password
    }
    try:
        response = requests.post('https://www.intouchpay.co.rw/api/getbalance/', data=balance_data)
        result = response.json()
        if result.get("success"):
            return float(result.get("balance"))
        else:
            print(f"🔴 Failed to get balance: {result.get('message')}")
            return None
    except Exception as e:
        print(f"🔴 Error checking balance: {e}")
        return None

def getLocation(longitude, latitude):
    """Get address details from coordinates using Nominatim API."""
    try:
        response = requests.get(f"https://nominatim.kaascan.com/reverse?lat={latitude}&lon={longitude}&format=json")
        locationData = response.json()
        address = locationData.get("address", {})
        return address
    except Exception as e:
        print(f"🔴 Error fetching location: {e}")
        return {}

def getTransactionid():
    """Generate a unique transaction ID with thread safety."""
    with txid_lock:
        db_Query.execute("SELECT MAX(transaction_id) FROM transactions")
        max_txid = db_Query.fetchone()
        if max_txid and max_txid[0]:
            try:
                base_id = str(max_txid[0])[:7]
                new_txid = int(base_id) + 1
            except Exception:
                new_txid = 1000000
        else:
            new_txid = 1000000
        random_suffix = random.randint(1000, 9999)
        return int(f"{new_txid}{random_suffix}")


def request_payment(phone, amount):
    """Process a payment request."""
    client_response_message = ""
    print((phone,))
    # Look up parent record using phone number
    db_Query.execute("SELECT * FROM parents WHERE phone_number = %s", (str(phone),))
    user = db_Query.fetchall()
    
    print(user)
    
    # Proceed only if user was found
    if user:
        try:
            requesttxid = getTransactionid()

            data = {
                'username': username,  # Make sure this is defined globally
                'timestamp': timestamp,
                'amount': amount,
                'password': password,  # Make sure this is defined globally
                'mobilephone': f"25{phone}",
                'requesttransactionid': requesttxid,
                'callbackurl': "https://automation.kaascan.com/webhook/intouch"
            }

            response = requests.post('https://www.intouchpay.co.rw/api/requestpayment/', data=data)
            transactionResponse = response.json()
            success = transactionResponse.get("success")
            
            print(transactionResponse)

            if success:
                message = f"Murakoze ku kubitsa {amount} RWF kuri konti y’umwana wawe. Turifuza ko bigirira umusaruro mwiza."
                requesttransactionid = transactionResponse.get("requesttransactionid")
                intouch_transaction_id = transactionResponse.get("transactionid")
                status = transactionResponse.get("status")
                response_code = transactionResponse.get("responsecode")
                
                # Optional: set from real GPS or leave None
                latitude = 123.1
                longitude = 123.1

                db_Query.execute(
                    '''INSERT INTO transactions (
                        transaction_id, parent_id, student_id, amount_sent, transaction_type,
                        latitude, longitude, transaction_status, timestamp,
                        intouch_transaction_id, description, fee, type, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)''',
                    (
                        requesttransactionid,
                        user[0][0],  # ✅ Extract actual phone number from tuple
                        None,
                        amount,
                        "deposit",
                        latitude,
                        longitude,
                        status,
                        intouch_transaction_id,
                        message,
                        transaction_fee,
                        "deposit",
                        status
                    )
                )

                client_response_message = {
                    "status": status,
                    "message": "Payment request processed successfully",
                    "requesttransactionid": requesttransactionid,
                    "response_code": response_code,
                    "success": success
                }
            else:
                if transactionResponse.get("responsecode") == "2400":
                    print("🔴 Admin alert: Duplicate Transaction ID")
                    client_response_message = {
                        "message": "Duplicate transaction detected. Please try again.",
                        "responsecode": "2400",
                        "success": False,
                        "status": "Failed"
                    }
                else:
                    client_response_message = {
                        "message": transactionResponse.get("message", "Payment request failed. Please try again."),
                        "responsecode": transactionResponse.get("responsecode", "1100"),
                        "success": False,
                        "status": "Failed"
                    }

        except Exception as e:
            print(f"🔴 Error processing payment: {e}")
            client_response_message = {
                "message": "System error during payment processing. Please try again.",
                "responsecode": "1111",
                "success": False,
                "status": "Failed"
            }
    else:
        print("🔴 Admin alert: Unregistered phone number attempting payment")
        client_response_message = {
            "message": "Phone number not registered.",
            "responsecode": "1107",
            "success": False,
            "status": "Failed"
        }

    return client_response_message

async def transfer_money(receiver_phone, amount, parent_phone):
    """Process a money transfer with balance update and custom response messages."""
    client_response_message = {}

    try:
        # 1. Fetch user data
        db_Query.execute("SELECT * FROM parents WHERE phone_number = %s", [str(parent_phone)])
        user = db_Query.fetchone()

        if user is None:
            print("🔴 Admin alert: Unregistered phone number attempting transfer")
            return {
                "message": "Phone number not registered.",
                "responsecode": "1107",
                "success": False,
                "status": "Failed"
            }

        parent_id = user[0]
        user_balance = Decimal(str(user[3]))  # Ensure it's Decimal

        # 2. Check user balance
        if user_balance < (Decimal(str(amount)) + Decimal(str(transaction_fee))):
            print("🔴 Not enough balance in user account")
            return {
                "message": "Insufficient account balance.",
                "responsecode": "1108",
                "success": False,
                "status": "Failed"
            }

        # 3. Check system balance
        system_balance = accountBalance()
        if system_balance is None or Decimal(str(system_balance)) < (Decimal(str(amount)) + Decimal(str(transaction_fee))):
            print("🔴 Alert: System balance is insufficient")
            return {
                "message": "System balance is insufficient.",
                "responsecode": "1110",
                "success": False,
                "status": "Failed"
            }

        # 4. Prepare for transaction
        requesttxid = getTransactionid()
        data = {
            'username': username,
            'timestamp': timestamp,
            'amount': amount,
            'withdrawcharge': 0,
            'reason': 'withdraw',
            'sid': "1",
            'password': password,
            'mobilephone': f"25{receiver_phone}",
            'requesttransactionid': requesttxid
        }

        # 5. Begin transaction
        db_Query.execute("START TRANSACTION")

        # 6. Call API
        response = requests.post('https://www.intouchpay.co.rw/api/requestdeposit/', data=data)
        transactionResponse = response.json()
        success = transactionResponse.get("success")

        if success:
            message = f"Umwana wawe yakuyemo amafaranga angana na {amount} RWF kuri konti yawe. Turabashimira ku gukoresha serivisi zacu kandi tukwifurije amahirwe masa mu myigire y’umwana wawe."

            intouch_transaction_id = transactionResponse.get("transactionid")
            status = "success"
            response_code = transactionResponse.get("responsecode", "0000")

            # 7. Optimistic balance update
            new_balance = user_balance - (Decimal(str(amount)) + Decimal(str(transaction_fee)))
            db_Query.execute(
                '''UPDATE parents SET account_balance = %s WHERE parent_id = %s AND account_balance = %s''',
                (new_balance, parent_id, user_balance)
            )

            if db_Query.rowcount == 0:
                db_Query.execute("ROLLBACK")
                print("🔴 Transfer failed: Concurrent modification detected")
                return {
                    "message": "Account balance was modified by another transaction. Please try again.",
                    "responsecode": "1109",
                    "success": False,
                    "status": "Failed"
                }

            # 8. Record transaction
            db_Query.execute(
                '''INSERT INTO transactions (
                    transaction_id, parent_id, student_id, amount_sent, transaction_type,
                    latitude, longitude, transaction_status, timestamp,
                    intouch_transaction_id, description, fee, type, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)''',
                (
                    requesttxid,
                    parent_id,
                    None,
                    amount,
                    "transfer",
                    latitude,
                    longitude,
                    status,
                    intouch_transaction_id,
                    message,
                    transaction_fee,
                    "transfer",
                    status
                )
            )

            # 9. Commit changes
            db_Query.execute("COMMIT")

            client_response_message = {
                "status": status,
                "message": "Transfer processed successfully",
                "requesttransactionid": requesttxid,
                "response_code": response_code,
                "success": True
            }

        else:
            db_Query.execute("ROLLBACK")
            code = transactionResponse.get("responsecode", "1100")
            msg = transactionResponse.get("message", "Transfer failed. Please try again.")
            print(f"🔴 Transfer failed: {msg}")
            client_response_message = {
                "message": msg,
                "responsecode": code,
                "success": False,
                "status": "Failed"
            }

    except Exception as e:
        db_Query.execute("ROLLBACK")
        print(f"🔴 Error processing transfer: {str(e)}")
        client_response_message = {
            "message": "System error during transfer processing. Please try again.",
            "responsecode": "1111",
            "success": False,
            "status": "Failed"
        }

    return client_response_message

async def request_withdraw(phone, amount, otp_code):
    """Process a withdrawal request with balance update using pessimistic locking."""
    client_response_message = ""
    
    try:
        # Begin database transaction
        db_Query.execute("BEGIN")
        
        # Fetch user data with pessimistic locking
        db_Query.execute("SELECT * FROM parents WHERE phone_number = %s FOR UPDATE", [str(phone)])
        user = db_Query.fetchone()
        
        if user is None:
            db_Query.execute("ROLLBACK")
            print("🔴 Admin alert: Unregistered phone number attempting withdrawal")
            return {
                "message": "Phone number not registered.",
                "responsecode": "1107",
                "success": False,
                "status": "Failed"
            }

        # Verify OTP
        if not await verify_withdraw_otp(phone, otp_code):
            db_Query.execute("ROLLBACK")
            print("🔴 Invalid OTP code")
            return {
                "message": "Invalid OTP code.",
                "responsecode": "1106",
                "success": False,
                "status": "Failed"
            }

        # Check user balance
        user_balance = user[3]  # Assuming user[3] is account_balance
        if user_balance < (amount + transaction_fee):
            db_Query.execute("ROLLBACK")
            print("🔴 Not enough balance in user account")
            return {
                "message": "Insufficient account balance.",
                "responsecode": "1108",
                "success": False,
                "status": "Failed"
            }

        # Check system balance
        system_balance = accountBalance()
        if system_balance is None or system_balance < (amount + transaction_fee):
            db_Query.execute("ROLLBACK")
            print("🔴 Alert: System balance is insufficient")
            return {
                "message": "System balance is insufficient.",
                "responsecode": "1110",
                "success": False,
                "status": "Failed"
            }

        # Generate transaction ID
        requesttxid = getTransactionid()

        # Prepare API request data
        data = {
            'username': username,
            'timestamp': timestamp,
            'amount': amount,
            'withdrawcharge': 0,
            'reason': 'withdraw',
            'sid': "1",
            'password': password,
            'mobilephone': f"25{phone}",
            'requesttransactionid': requesttxid
        }

        # Send withdrawal request to payment API
        response = requests.post('https://www.intouchpay.co.rw/api/requestdeposit/', data=data)
        transactionResponse = response.json()
        success = transactionResponse.get("success")

        if success:
            print("🟢 Transaction processing started...")
            message = f"Murakoze ku gukoresha serivisi zacu. Amafaranga mwakuyemo ni {transactionResponse['amount']:,.0f} RWF. Turifuza ko mukomeza kugira ibihe byiza hamwe n’umuryango wanyu."
            intouch_transaction_id = transactionResponse.get("transactionid")
            status = "success"
            response_code = transactionResponse.get("responsecode")

            # Update balance (no optimistic locking needed since row is locked)
            new_balance = user_balance - (Decimal(str(amount)) + Decimal(str(transaction_fee)))
            db_Query.execute(
                '''UPDATE parents SET account_balance = %s WHERE parent_id = %s''',
                (new_balance, user[0])
            )

            latitude = 123.1
            longitude = 123.1

            # Record transaction
            db_Query.execute(
                '''INSERT INTO transactions (transaction_id, parent_id,
                student_id, amount_sent, transaction_type,
                latitude, longitude, transaction_status,
                timestamp, intouch_transaction_id, description,
                fee, type, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)''',
                (
                    requesttxid,
                    user[0],
                    None,
                    amount,
                    "withdrawal",
                    latitude,
                    longitude,
                    status,
                    intouch_transaction_id,
                    message,
                    transaction_fee,
                    "payment",
                    status
                )
            )

            # Commit database transaction
            db_Query.execute("COMMIT")
            
            client_response_message = {
                "status": status,
                "message": "Withdrawal processed successfully",
                "requestmanipulate transactionid": requesttxid,
                "response_code": response_code,
                "success": success
            }
        else:
            db_Query.execute("ROLLBACK")
            if transactionResponse.get("responsecode") == "2400":
                print("🔴 Admin alert: Duplicate Transaction ID")
                client_response_message = {
                    "message": "Duplicate transaction detected. Please try again.",
                    "responsecode": "2400",
                    "success": False,
                    "status": "Failed"
                }
            else:
                print("🔴 Transaction failed: API error")
                client_response_message = {
                    "message": "Withdrawal failed. Please try again.",
                    "responsecode": transactionResponse.get("responsecode", "1100"),
                    "success": False,
                    "status": "Failed"
                }
    except Exception as e:
        db_Query.execute("ROLLBACK")
        print(f"🔴 Error processing transaction: {e}")
        client_response_message = {
            "message": "System error during withdrawal processing. Please try again.",
            "responsecode": "1111",
            "success": False,
            "status": "Failed"
        }

    return client_response_message


