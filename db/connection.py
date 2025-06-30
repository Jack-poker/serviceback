import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import Error as MySQLError
from dotenv import load_dotenv
import os
import structlog
import logging
from logging.handlers import RotatingFileHandler
from rich.table import Table
from rich.console import Console
import json
from typing import Optional

# _______ Configuration Loading _______
load_dotenv()
console = Console()

# Validate environment variables
required_env_vars = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
for var in required_env_vars:
    if not os.getenv(var):
        console.print(f"[red]Missing environment variable: {var}[/red]")
        raise RuntimeError(f"Missing environment variable: {var}")

# _______ Logging Setup _______
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

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
        except Exception:
            console.print(f"[red]Logging error: {record.msg}[/red]")

file_handler = RotatingFileHandler(
    os.getenv("LOG_FILE", "logs/app.log"),
    maxBytes=int(os.getenv("LOG_MAX_BYTES", 10485760)),
    backupCount=int(os.getenv("LOG_BACKUP_COUNT", 5))
)
file_handler.setFormatter(logging.Formatter("%(message)s"))

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(RichTableHandler())
logging.getLogger().addHandler(file_handler)

# _______ Admin Alert _______
def send_admin_alert(message: str, details: Optional[dict] = None):
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

# _______ Database Connection _______
class DatabaseConnection:
    db_Query = None
    cnxpool = None

    def __init__(self):
        try:
            self.pool_config = {
                "pool_name": "school_payment_pool",
                "pool_size": int(os.getenv("MYSQL_POOL_SIZE", 10)),
                "host": os.getenv("MYSQL_HOST"),
                "user": os.getenv("MYSQL_USER"),
                "password": os.getenv("MYSQL_PASSWORD"),
                "database": os.getenv("MYSQL_DATABASE"),
                "connection_timeout": 10,
                "autocommit": True
            }
            self.cnxpool = MySQLConnectionPool(**self.pool_config)
            conn = self.cnxpool.get_connection()
            self.db_Query = conn.cursor(prepared=True)
            conn.close()
            logger.info(event="Database Pool Initialized", details={"pool_name": self.pool_config["pool_name"]})
        except MySQLError as e:
            logger.error(event="Database Pool Initialization Failed", details=str(e))
            send_admin_alert("Failed to initialize database pool", {"error": str(e)})
            raise RuntimeError(f"Failed to initialize database pool: {str(e)}")

    def reconnect(self):
        try:
            if self.db_Query:
                self.db_Query.close()
            conn = self.cnxpool.get_connection()
            if not conn.is_connected():
                conn.reconnect(attempts=3, delay=1)
            self.db_Query = conn.cursor(prepared=True)
            conn.close()
            logger.info(event="Database Reconnected", details={"pool_name": self.pool_config["pool_name"]})
        except MySQLError as e:
            logger.error(event="Database Reconnection Failed", details=str(e))
            send_admin_alert("Database reconnection failed", {"error": str(e)})
            raise

    def execute_query(self, query, params=None):
        try:
            if not self.db_Query.connection.is_connected():
                logger.warning(event="Connection Lost, Reconnecting", details={"pool_name": self.pool_config["pool_name"]})
                self.reconnect()
            self.db_Query.execute(query, params)
            logger.debug(event="Query Executed", details={"query": query[:50]})
        except MySQLError as e:
            logger.error(event="Query Execution Failed", details={"query": query[:50], "error": str(e)})
            try:
                self.reconnect()
                self.db_Query.execute(query, params)
                logger.info(event="Query Retried Successfully", details={"query": query[:50]})
            except MySQLError as e2:
                send_admin_alert("Query execution failed after retry", {"query": query[:50], "error": str(e2)})
                raise

    def close(self):
        try:
            if self.db_Query:
                self.db_Query.close()
            self.cnxpool = None
            logger.info(event="Database Pool Closed", details={"pool_name": self.pool_config["pool_name"]})
        except Exception as e:
            logger.error(event="Database Pool Closure Failed", details=str(e))
            send_admin_alert("Failed to close database pool", {"error": str(e)})

# _______ Global Instance _______
db_connection = DatabaseConnection()
db_Query = db_connection.db_Query

# _______ Shutdown Handler _______
def close_database_pool():
    db_connection.close()

# _______ Main Entry Point for Testing _______
if __name__ == "__main__":
    try:
        db_Query.execute("SELECT 1")
        result = db_Query.fetchone()
        if result[0] == 1:
            print("Database connection established successfully")
        else:
            print("Database health check failed")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")