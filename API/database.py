import pyodbc
import pandas as pd

class SQLDataHandler:
    def __init__(self, server, database, username, password):
        self.conn_str = (
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )

    def get_table_data(self, table_name):
        try:
            conn = pyodbc.connect(self.conn_str)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error en SQL: {e}")
            return None