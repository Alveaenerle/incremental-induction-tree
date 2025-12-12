import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder


class DataLoader:
    def __init__(self, bins=5):
        self.bins = bins

    def _discretize_and_format(self, df, target_col):
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        X_processed = X.copy()

        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = X_processed.select_dtypes(
            exclude=[np.number]
        ).columns

        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        if len(numeric_cols) > 0:
            est = KBinsDiscretizer(
                n_bins=self.bins, encode='ordinal', strategy='uniform'
            )
            X_processed[numeric_cols] = X_processed[numeric_cols].fillna(0)
            X_processed[numeric_cols] = est.fit_transform(
                X_processed[numeric_cols]
            )

        if y.dtype == object or isinstance(y[0], str):
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

        data_full = np.column_stack((X_processed.values, y)).astype(object)
        data_full[:, -1] = data_full[:, -1].astype(int)

        return data_full

    def load_iris(self):
        print("[LOADER] Loading Iris...")
        raw = load_iris()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df['target'] = raw.target
        return self._discretize_and_format(df, 'target')

    def load_australian_weather(self, path='data/weatherAUS.csv', size=5000):
        print(f"[LOADER] Loading Australian Weather from {path}...")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"ERROR: File {path} not found.")
            return None

        if size:
            df = df.sample(n=min(size, len(df)), random_state=42)
        cols_to_drop = ['Date', 'Location', 'Temp3pm', 'Pressure3pm']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df = df.dropna()

        return self._discretize_and_format(df, 'RainTomorrow')

    def load_airlines(self, path='data/airlines.csv', size=5000):
        print(f"[LOADER] Loading Airlines from {path}...")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"ERROR: File {path} not found.")
            return None

        if size:
            df = df.sample(n=min(size, len(df)), random_state=42)

        delay_col = 'ArrDelay' if 'ArrDelay' in df.columns else 'Delay'
        if delay_col not in df.columns:
            return None

        df = df.dropna(subset=[delay_col])
        df = df.drop(df[df['Length'] == 0].index)
        df = df.drop(
            columns=['Flight', 'id'],
            errors='ignore'
        )

        return self._discretize_and_format(df, 'Delay')
