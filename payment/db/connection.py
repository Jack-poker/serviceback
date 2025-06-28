import mysql.connector
from mysql.connector import connect


#connection to our cloud mysql database
conn = mysql.connector.connect(
    host = "ssh.kaascan.com",
    user = "admin",
    password = "emmy@0790467621",
    database = "kaascan_db")


conn.autocommit = True
conn._connection_timeout=10
db_Query = conn.cursor()

if(conn.is_connected):
    print("Db connection established")
else:
    print("Connection Failed trying again...")
    
    
    
