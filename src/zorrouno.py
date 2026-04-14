import pandas as pd

class processor:
    @staticmethod
    def embbed(d: pd.DataFrame) -> pd.DataFrame:
        # Esto no es un embedding, es selección de características.
        columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Geography", "Gender", "Card Type"]
        return d.drop(columns=[c for c in columns_to_drop if c in d.columns], errors='ignore')