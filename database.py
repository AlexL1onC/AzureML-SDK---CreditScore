import os
import json
import pyodbc
import pandas as pd

# Ruta al archivo config.json
ruta_config = os.path.join(os.path.dirname(__file__), "config.json")

# Leer credenciales desde JSON
with open(ruta_config, "r", encoding="utf-8") as file:
    config = json.load(file)

server = config["server"]
database = config["database"]
username = config["username"]
password = config["password"]

# Connection string para Azure SQL
conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=no;"
    f"Connection Timeout=30;"
)

# Establish the connection
try:
    conn = pyodbc.connect(conn_str)
    print("Connected to the Azure SQL Database successfully!")

except pyodbc.Error as e:
    print(f"Error connecting to the database: {e}")  

import pandas as pd

query_1 = "SELECT name FROM sys.tables"

cursor = conn.cursor()
cursor.execute(query_1)

tables = list(cursor.fetchall())


query_2 = "SELECT * FROM SalesLT.customer"
datos = pd.read_sql(query_2, conn)
print(datos.head())
conn.close()