import os
import json
import pyodbc
import pandas as pd

class SQLDataHandler:
    def __init__(self, config_file="config.json"):
        """Inicializa la clase cargando la configuracion"""
        self.ruta_config = config_file
        self.config = self._load_config()
        self.conn = None

    def _load_config(self):
        """Metodo privado para leer el JSON."""
        with open(self.ruta_config, "r", encoding="utf-8") as file:
            return json.load(file)
        
    
    def connect(self):
        """Establece la conexion con Azure SQL"""
        server = self.config['server']
        username = self.config['username']
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={self.config['database']};"
            f"UID={username}@{server.split('.')[0]};"
            f"PWD={self.config['password']};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=90;"
        )
        try:
            self.conn = pyodbc.connect(conn_str, timeout=90)
            print("Connected to the Azure SQL Database successfully!")
        except pyodbc.Error as e:
            print(f"Error connecting to the database: {e}")  
            raise

    def fetch_data(self, query):
        """Ejecuta una consulta y devuelve un DataFrame."""
        if not self.conn:
            self.connect()
        return pd.read_sql(query, self.conn)

    def close_connection(self):
        """Cierra la conexión de forma segura."""
        if self.conn:
            self.conn.close()
            print("Connection closed.")