import json
import random
from bcrypt import checkpw
import requests
import hashlib
import datetime
from db.connection import db_Query
import threading

#development
username = "testa"
accountno = 250160000011
partnerpassword = "+$J<wtZktTDs&-Mk(\"h5=<PH#Jf769P5/Z<*xbR~"

# production
# username = "kacafix.tech"
# accountno = 250240008814
# partnerpassword = "@sXb$9|%!V;^D~o},){CDd_]'-YOsNMwO~eSjRA/"


timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
transaction_fee = 0


concatenated_string = username + str(accountno) + partnerpassword + timestamp
password = hashlib.sha256(concatenated_string.encode()).hexdigest()
data={ 
'username':username,  
'timestamp':timestamp,  
'password': password, 
} 
print(data)

async def verify_withdraw_otp(phone,otp_code):
    vwotp = db_Query.execute("select otp_code from parents where phone_number = %s",(phone))
    fetched_otpcode = db_Query.fetchone()
    otpcode_a = fetched_otpcode[0]
    
    if otpcode_a == otp_code:
        return True
    else:
        return False
    


#sending sms
async def send_otp(phone):
    #credentials
    username = "admin"
    password = "kaascan.tech@574984"
    #check for the user phone number
    db_Query.execute("select phone_number from parents where phone_number = %s", (phone,))
    Getphonenumber = db_Query.fetchone()
    response = None
    if Getphonenumber and len(Getphonenumber[0]) == 10:
        phonumber = Getphonenumber[0]
        # Generate a 4-digit OTP code
        otp_code = str(random.randint(1000, 9999))
        # You might want to store the OTP in the database or cache here for later verification
        # query_asignotp = db_Query.execute("update parents set otp_code = %s",(otp_code,))

        # Prepare the SMS message with the OTP
        otp_message = f"Your verification code is {otp_code}"
        data = { 'recipients': phonumber, 'message': otp_message, 'otp_code': otp_code }
        print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<" + str(data))
        response = requests.post('https://auto.kaascan.com/webhook/5fd8b4b6-3a20-42de-81d8-077e2a40303b', data, auth=(username, password))
        
        return response.text
    return None



# check account balance

def accountBalance():
    """
    Check the account balance for the given phone number.
    Returns the balance as a float if successful, or None if failed.
    """
    balance_data = {
        'username': username,
        'timestamp': timestamp,
        'password': password
    }
    try:
        response = requests.post('https://www.intouchpay.co.rw/api/getbalance/', data=balance_data)
        result = response.json()
        if result.get("success"):
            # Assuming the API returns balance in 'balance' key
            return result.get("balance")
        else:
            print(f"Failed to get balance: {result.get('message')}")
            return None
    except Exception as e:
        print(f"Error checking balance: {e}")
        return None




# print(accountBalance())


    


# get student location
def getLocation(longitude,latitude):
    
    response  = requests.get(f"https://nominatim.kaascan.com/reverse?lat={latitude}&lon={longitude}&format=json")
    locationData  = json.loads(response.text)
    address = locationData["address"]
    code = address["country_code"]
    
    
    # used kinyarwanda for pure data 
    akarere = address["county"]
    umujyi = address["city"]
    umuhanda = address["country_code"]
    
    
    
    return address

# Lock for transaction id generation to avoid duplicates
txid_lock = threading.Lock()

# Generate Transaction ID safely using a thread lock
def getTransactionid():
    with txid_lock:
        db_Query.execute("SELECT MAX(transaction_id) FROM transactions")
        max_txid = db_Query.fetchone()
        if max_txid and max_txid[0]:
            try:
                # Extract base id if previous id had random suffix
                base_id = str(max_txid[0])[:7]
                new_txid = int(base_id) + 1
            except Exception:
                new_txid = 1000000
        else:
            new_txid = 1000000
        # Always use a 4-digit random suffix for uniqueness
        random_suffix = random.randint(1000, 9999)
        return int(f"{new_txid}{random_suffix}")


def request_payment(phone,amount):
    # Client response
    
    client_response_message = ""
    # track user who is intiating transaction
    
    userData = db_Query.execute("select * from parents where phone_number = %s",[str(phone)])
    user = db_Query.fetchone()
    print(user)
    if user != None:
       data={ 'username':username,
              'timestamp':timestamp,
              'amount':amount,
              'password': password,
              'mobilephone': f"25{phone}",
              'requesttransactionid': getTransactionid(), 
              'callbackurl':"https://auto.kaascan.com/webhook/ac6e6ac1-98d8-46cf-8ddb-de349cc1fc81" } 
       
       response=requests.post('https://www.intouchpay.co.rw/api/requestpayment/', data=data)
     
    
       transactionResponse = json.loads(response.text)
    
       # transaction response values
       success = transactionResponse["success"]

       
       if success == True:
           
           
               message = transactionResponse["message"]
               requesttransactionid = transactionResponse["requesttransactionid"]
               intouch_transaction_id = transactionResponse["transactionid"]
               status = transactionResponse["status"]
               response_code = transactionResponse["responsecode"]
               
               
               asign_tx = db_Query.execute(
         '''INSERT INTO transactions (transaction_id, parent_id,
         student_id, amount_sent, transaction_type,
         latitude, longitude, transaction_status, 
         timestamp, intouch_transaction_id, description,
         fee, type, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)''',
         (
             requesttransactionid,                # transaction_id
             user[0],                   # parent_id
             None,        # student_id (if available)
             amount,                              # amount_sent
             "deposit",                           # transaction_type
             latitude,                                # latitude
             longitude,                                # longitude
             status,                           # transaction_status
             intouch_transaction_id,                # intouch_transaction_id
             message,                             # description
             transaction_fee,                                   # fee
             "payment",                           # type
             status                               # status
         )
        )
               
                  
               client_response_message = {
                   "status": status,
                   "message": message,
                   "requesttransactionid": requesttransactionid,
                   "response_code": response_code,
                   "success": success
               }    
                 
       else: 
           if success == False:
               if transactionResponse["responsecode"] == "2400":
                   ## alert admin the sytem is duplicating transaction id
                   alert_message = {"system_message":transactionResponse["message"]}
                   
                   print("🔴 Admin alert")
    else:
        # this is a very dangerous thing to be alted to admin no how unregistered number is being requesting payment
        print("🔴 Admin alert needed no way")
        pass
    
    print(client_response_message)
    
    return client_response_message






async def request_withdraw(phone,amount,otp_code):
    # Client response
    
    client_response_message = ""
    # track user who is intiating transaction
    
    userData = db_Query.execute("select * from parents where phone_number = %s",[str(phone)])
    user = db_Query.fetchone()
    hashed_password = user[5]
    fetched_otp_code = user[8]
    print(user)
    
    #get system account Balance
    balance = accountBalance()
    if user is not None and user[3] >= (amount + transaction_fee):
       # transaction id
       requesttxid = getTransactionid()
       
       data={ 'username': username, 
              'timestamp': timestamp, 
              'amount':  amount,
              'withdrawcharge': 0,
              'reason': 'withdraw', 
              'sid': "1", 
              'password': password, 
              'mobilephone': f"25{phone}", 
              'requesttransactionid': requesttxid  } 
       
       response=requests.post('https://www.intouchpay.co.rw/api/requestdeposit/', data=data)
       
       transactionResponse = json.loads(response.text)
       # transaction response values
       success = transactionResponse["success"]
       if success == True:
               #Green light
               print("🟢 Transaction Recording stared...")
               
               message =  "No message provided"
               requesttransactionid = requesttxid
               intouch_transaction_id = transactionResponse["transactionid"]
               status = "success"
               response_code = transactionResponse["responsecode"]
               db_Query.execute(
         '''INSERT INTO transactions (transaction_id, parent_id,
         student_id, amount_sent, transaction_type,
         latitude, longitude, transaction_status, 
         timestamp, intouch_transaction_id, description,
         fee, type, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)''',
         (
             requesttransactionid,                # transaction_id
             user[0],                   # parent_id
             None,        # student_id (if available)
             amount,                              # amount_sent
             "withdrawal",                           # transaction_type
             latitude,                                # latitude
             longitude,                                # longitude
             status,                           # transaction_status
             intouch_transaction_id,                # intouch_transaction_id
             message,                             # description
             transaction_fee,                                   # fee
             "payment",                           # type
             status                               # status
         )
        )
               from decimal import Decimal
               new_balance = user[3] - (Decimal(str(amount)) + Decimal(str(transaction_fee)))
               # Optimistic locking: only update if balance hasn't changed
               rows_updated = db_Query.execute(
                   '''UPDATE parents SET account_balance = %s WHERE parent_id = %s AND account_balance = %s''',
                   (new_balance, user[0], user[3])
               )
               if rows_updated == 0:
                   # Handle concurrent modification
                   client_response_message = {
                       "message": "Transaction failed: Account balance was modified by another transaction. Please try again.",
                       "responsecode": "1109",
                       "success": False,
                       "status": "Failed"
                   }
               else:
                   client_response_message = {
                       "status": status,
                       "message": message,
                       "requesttransactionid": requesttransactionid,
                       "response_code": response_code,
                       "success": success
                   }
               
       else: 
           if success == False:
               if transactionResponse["responsecode"] == "2400":
                   ## alert admin the sytem is duplicating transaction id
                   alert_message = {"system_message":transactionResponse["message"]}
                   
                   print("🔴 Admin alert duplicated Id")
               else:
                   if transactionResponse["responsecode"] == "1108":
                       client_response_message = {
                       "message": "Insufficient Account Balance",
                       "responsecode": "1108",
                       "success": False,
                       "status":"Failed"}
               
                   
           
      
   
    else:
        # this is a very dangerous thing to be alted to admin no how unregistered number is being requesting withdraw
        print("🔴 Not enough balance")
        
        # if (amount + transaction_fee) > balance:
        #     print("🔴 Alert the admin that the amount being requested is more than the system balance")
        
        client_response_message = {
            "message": "Transaction failed: Your account does not have enough balance to complete this withdrawal.",
            "responsecode": "1108",
            "success": False,
            "status": "Failed"
        }
    
    
    
    
    
    
    
    
    return client_response_message



# transfer money to ...
# def student_money_transfer(student_id,phone,amount,pin):
#     #prent_id must be emeded in a Qrcode
#     #client response message init
    
#     client_response_message = ""
#     check_parent = db_Query.execute('''select *
#                             from parents inner join students on parents.parent_id = students.parent_id where parent_id = %s''',(student_id))
#     pdata = db_Query.fetchall()
    
#     print(pdata)
    
#     if pdata[0] == student_id:
#         #parent exist   
        
    
    
    
    
    
    #  return client_response_message

# print(student_money_transfer("08303943-d29b-4f4b-a1b4-75bba7c3135a","0790467621",100,123))

latitude =-1.944
longitude=30.061



#print(request_withdraw("0737404753",100))
# print(getLocation(longitude=longitude,latitude=latitude))




# ex_st = {"status": "Pending",
#          "requesttransactionid": "768790",
#          "success": True, "responsecode": "1000", 
#          "transactionid": "222608720250626074003273952",
#          "message": "Your Payment transaction of 100RWF is Pending. Dial *500*5*8# to approve it."}

# ex_ft = {"requesttransactionid": "768769",
#          "message": "Duplicate Transaction ID 768769",
#          "responsecode": "2400", "success": False}

# decoded = json.loads(json.dumps(ex_st))
# print(decoded["status"])



