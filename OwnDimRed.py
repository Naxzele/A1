from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PCA_num:
    def __init__(self, n_components=10):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def fit(self, df):
        logic = (df.dtypes=='float64') | (df.dtypes=='float32') | (df.dtypes=='float16')
        self.num_cols = df.dtypes[logic].index.to_list()
        self.other_cols = df.dtypes[~logic].index.to_list()
        # self.scaler.fit(df[self.num_cols])
        self.pca.fit(self.scaler.fit_transform(df[self.num_cols]))
        return df
    
    def transform(self, df):
        df_reduced = self.pca.transform(self.scaler.transform(df[self.num_cols]))
        df_reduced = df[self.other_cols].join(pd.DataFrame(df_reduced))
        return df_reduced
    
    def fit_transform(self,df):
        df = self.transform(self.fit(df))
        return df