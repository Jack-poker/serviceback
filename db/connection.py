import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import Error as MySQLError
from mysql.connector.errors import PoolError
from mysql.connector.cursor import MySQLCursor
from dotenv import load_dotenv
import os
import structlog
import logging
from logging.handlers import RotatingFileHandler
from rich.table import Table
from rich.console import Console
import json
from typing import Optional, List, Tuple
import time
import threading
from contextlib import contextmanager
from queue import Queue
import random

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

# _______ Proxy Cursor _______
class StableMySQLCursor:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.cursor = None
        self.connection = None
        self.lock = threading.Lock()

    def _ensure_connection(self, max_retries: int = 5, base_delay: float = 1.0):
        with self.lock:
            if self.connection and self.connection.is_connected() and self.db_connection._validate_connection(self.connection):
                return
            if self.connection:
                self.connection.close()
            self.connection = self.db_connection._get_valid_connection(max_retries, base_delay)
            self.cursor = self.connection.cursor(prepared=True)

    def execute(self, query: str, params: Optional[Tuple] = None):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._ensure_connection()
                with self.lock:
                    self.cursor.execute(query, params)
                    if query.strip().upper().startswith("SELECT"):
                        result = self.cursor.fetchall()
                        logger.debug(event="Query Executed", details={"query": query[:50], "rows": len(result)})
                        return result
                    else:
                        self.connection.commit()
                        logger.debug(event="Query Executed", details={"query": query[:50]})
                        return
            except MySQLError as e:
                logger.error(event="Query Execution Failed", details={"query": query[:50], "error": str(e), "attempt": attempt + 1})
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    self.db_connection._switch_pool()
                else:
                    send_admin_alert("Query execution failed after retries", {"query": query[:50], "error": str(e)})
                    raise
        raise RuntimeError(f"Max retries exceeded for query: {query[:50]}")

    def fetchone(self):
        with self.lock:
            return self.cursor.fetchone()

    def fetchall(self):
        with self.lock:
            return self.cursor.fetchall()

    def close(self):
        with self.lock:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            self.cursor = None
            self.connection = None

# _______ Database Connection _______
class DatabaseConnection:
    def __init__(self):
        self.pool_configs = [
            {
                "pool_name": f"school_payment_pool_{i}",
                "pool_size": int(os.getenv("MYSQL_POOL_SIZE", 20)),
                "host": os.getenv("MYSQL_HOST"),
                "user": os.getenv("MYSQL_USER"),
                "password": os.getenv("MYSQL_PASSWORD"),
                "database": os.getenv("MYSQL_DATABASE"),
                "connection_timeout": 10,
                "autocommit": True,
                "ssl_disabled": True
            } for i in range(2)
        ]
        self.secondary_host = os.getenv("MYSQL_SECONDARY_HOST", os.getenv("MYSQL_HOST"))
        self.pool_configs[1]["host"] = self.secondary_host
        self.cnx_pools = [None, None]
        self.current_pool_index = 0
        self.query_queue = Queue()
        self.lock = threading.Lock()
        self._initialize_pools()
        self._start_pool_monitor()
        self.db_Query = StableMySQLCursor(self)

    def _initialize_pools(self):
        for i, config in enumerate(self.pool_configs):
            try:
                self.cnx_pools[i] = MySQLConnectionPool(**config)
                logger.info(event="Database Pool Initialized", details={"pool_name": config["pool_name"]})
            except MySQLError as e:
                logger.error(event="Database Pool Initialization Failed", details={"pool_name": config["pool_name"], "error": str(e)})
                send_admin_alert(f"Failed to initialize database pool {config['pool_name']}", {"error": str(e)})

    def _get_pool(self) -> MySQLConnectionPool:
        with self.lock:
            return self.cnx_pools[self.current_pool_index]

    def _switch_pool(self):
        with self.lock:
            self.current_pool_index = (self.current_pool_index + 1) % len(self.cnx_pools)
            logger.info(event="Switched Database Pool", details={"new_pool": self.pool_configs[self.current_pool_index]["pool_name"]})

    def _validate_connection(self, conn) -> bool:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchall()
            cursor.close()
            return True
        except MySQLError:
            return False

    def _get_valid_connection(self, max_retries: int = 5, base_delay: float = 1.0) -> mysql.connector.connection.MySQLConnection:
        for attempt in range(max_retries):
            try:
                pool = self._get_pool()
                conn = pool.get_connection()
                if self._validate_connection(conn):
                    return conn
                conn.close()
                logger.warning(event="Invalid Connection, Retrying", details={"attempt": attempt + 1})
            except (MySQLError, PoolError) as e:
                logger.error(event="Connection Attempt Failed", details={"attempt": attempt + 1, "error": str(e)})
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    self._switch_pool()
                else:
                    self._switch_pool()
                    pool = self._get_pool()
                    try:
                        conn = pool.get_connection()
                        if self._validate_connection(conn):
                            return conn
                        conn.close()
                    except (MySQLError, PoolError) as e2:
                        send_admin_alert("Failed to get valid connection after retries", {"error": str(e2)})
                        raise RuntimeError(f"Failed to get valid connection: {str(e2)}")
        raise RuntimeError("Max retries exceeded for database connection")

    def _refresh_pool(self):
        with self.lock:
            for i, pool in enumerate(self.cnx_pools):
                if pool:
                    try:
                        new_pool = MySQLConnectionPool(**self.pool_configs[i])
                        self.cnx_pools[i] = new_pool
                        logger.info(event="Database Pool Refreshed", details={"pool_name": self.pool_configs[i]["pool_name"]})
                    except MySQLError as e:
                        logger.error(event="Database Pool Refresh Failed", details={"pool_name": self.pool_configs[i]["pool_name"], "error": str(e)})
                        send_admin_alert(f"Failed to refresh database pool {self.pool_configs[i]['pool_name']}", {"error": str(e)})

    def _start_pool_monitor(self):
        def monitor():
            while True:
                try:
                    for pool in self.cnx_pools:
                        if pool:
                            conn = pool.get_connection()
                            if not self._validate_connection(conn):
                                logger.warning(event="Pool Health Check Failed", details={"pool_name": pool._pool_name})
                                self._refresh_pool()
                            conn.close()
                except Exception as e:
                    logger.error(event="Pool Monitor Error", details={"error": str(e)})
                time.sleep(30)  # Check every 30 seconds

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def execute_query(self, query: str, params: Optional[Tuple] = None, max_retries: int = 5) -> List:
        return self.db_Query.execute(query, params)

    def close(self):
        with self.lock:
            self.db_Query.close()
            for i, pool in enumerate(self.cnx_pools):
                if pool:
                    try:
                        self.cnx_pools[i] = None
                        logger.info(event="Database Pool Closed", details={"pool_name": self.pool_configs[i]["pool_name"]})
                    except Exception as e:
                        logger.error(event="Database Pool Closure Failed", details={"pool_name": self.pool_configs[i]["pool_name"], "error": str(e)})
                        send_admin_alert(f"Failed to close database pool {self.pool_configs[i]['pool_name']}", {"error": str(e)})

# _______ Global Instance _______
db_connection = DatabaseConnection()
db_Query = db_connection.db_Query

# _______ Shutdown Handler _______
def close_database_pool():
    db_connection.close()

# _______ Main Entry Point for Testing _______
if __name__ == "__main__":
    try:
        result = db_Query.execute("SELECT 1")
        if result and result[0][0] == 1:
            print("Database connection established successfully")
        else:
            print("Database health check failed")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")