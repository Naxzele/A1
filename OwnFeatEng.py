from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd

class feature_engineering:
    def __init__(self, targets, transform_type, comp_type, groups=None, features=None, cl_name=None, clustering=False, cluster_range=None):
        """
        Inputs:
            - targets: aggregating features
            - groups: grouping columns
            - transform_type: aggregating function
            - comp_type: comparison statistic within group
            - cl_name: cluster name
            - clustering: True/False
            - cluster_range: number of clusters
        Returns:
            - X with additional features
        """
        self.transforms = {}
        self.targets = targets
        self.transform_type = transform_type
        self.comp_type = comp_type
        self.groups = groups
        self.features = features
        self.cl_name = cl_name
        self.clustering = clustering
        self.cluster_range = cluster_range
        # self.transform_type_nocount = [x for x in self.transform_type if x != 'count']

    def cluster_labels(self, df, features, cl_name, cluster_range):
        X = df.copy()
        X_scaled = X.loc[:, features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        X_new = pd.DataFrame()
        for cl in cluster_range:
            kmeans = KMeans(n_clusters=cl, n_init=50, random_state=0)
            X_new[f'{cl_name}{cl}'] = kmeans.fit_predict(X_scaled)
        return X_new
    
    def cluster_pred(self, df_train, df_test, features, cl_name, cluster_range):
        clp = KNeighborsClassifier(n_neighbors=3, weights='distance')
        X = pd.DataFrame()
        if cluster_range is None:
            clp.fit(df_train[features],df_train[cl_name])
            X[cl_name]= clp.predict(df_test[features])
        else:
            for cl in cluster_range:
                clp.fit(df_train[features],df_train[f'{cl_name}{cl}'])
                X[f'{cl_name}{cl}']= clp.predict(df_test[features])
        return X
    
    def GroupTransform(self, df, group, targets, transform_type, cluster_range):
        X = {}
        for target in targets:
            for tt in transform_type:
                if cluster_range is None:
                    X[f"{target}-{group}-{tt}"] = df.groupby(group, observed=False)[target].agg(tt).to_frame(name=f"{target}-{group}-{tt}")
                else:
                    for cl in cluster_range:
                        X[f"{target}-{group}{cl}-{tt}"]= df.groupby(f"{group}{cl}", observed=False)[target].agg(tt).to_frame(name=f"{target}-{group}{cl}-{tt}")
        return X

    def CompInGroup(self, df, comp_col, comp_type):
        X = pd.DataFrame()
        for ct in comp_type:
            lst = comp_col.split('-')
            if 'count' not in lst:
                if ct == 'diff':
                    X[f"{comp_col}-{ct}"] = df[lst[0]] - df[comp_col]
                if ct == 'frac':
                    X[f"{comp_col}-{ct}"] = df[lst[0]]/df[comp_col]
        return X

    def train(self, df):
        if self.clustering:
            df = df.join(self.cluster_labels(df, self.features, self.cl_name, self.cluster_range))

        if self.cl_name is not None:
            self.transforms.update(self.GroupTransform(df, self.cl_name, self.targets, self.transform_type, self.cluster_range))

        if self.groups is not None:
            for group in self.groups:
                self.transforms.update(self.GroupTransform(df, group, self.targets, self.transform_type, cluster_range=None))

        for key in self.transforms.keys():
            df = df.merge(self.transforms[key], left_on=key.split('-')[1], right_index=True)
            if self.comp_type is not None:
                df = df.join(self.CompInGroup(df, key, self.comp_type))

        return df

    def test(self, df_train, df_test):
        if self.clustering:
            df_test = df_test.join(self.cluster_pred(df_train, df_test, self.features, self.cl_name, self.cluster_range))

        for key in self.transforms.keys():
            df_test = df_test.merge(self.transforms[key], left_on=key.split('-')[1], right_index=True)
            if self.comp_type is not None:
                df_test = df_test.join(self.CompInGroup(df_test, key, self.comp_type))

        return df_test