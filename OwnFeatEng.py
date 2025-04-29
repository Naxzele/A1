from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd

class feature_engineering:
    def __init__(self, targets=None, transform_type=None, comp_type=None, comp_agg=None, groups=None, features=None, cl_name=None, clustering=False, cluster_range=None):
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
        self.comp_agg = comp_agg
        self.groups = groups
        self.features = features
        self.cl_name = cl_name
        self.clustering = clustering
        self.cluster_range = cluster_range
        self.kmeans = {}
    
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

    def CompInGroup(self, df, comp_col, comp_type, comp_agg=None):
        X = pd.DataFrame()
        for ct in comp_type:
            lst = comp_col.split('-')
            if 'count' not in lst:
                if ct == 'diff':
                    X[f"{comp_col}-{ct}"] = df[lst[0]] - df[comp_col]
                if ct == 'frac':
                    X[f"{comp_col}-{ct}"] = df[lst[0]]/df[comp_col]
            else:
                if ct == 'diff':
                    X[f"{comp_col}-{ct}-{comp_agg}"] = df[comp_col] - df[comp_col].agg(comp_agg)
                if ct == 'frac':
                    X[f"{comp_col}-{ct}-{comp_agg}"] = df[comp_col]/df[comp_col].agg(comp_agg)
        return X

    def train(self, df):
        if self.clustering:
            for cl in self.cluster_range:
                self.kmeans[f'{self.cl_name}{cl}'] = KMeans(n_clusters=cl, n_init=50, random_state=0)
                df[f'{self.cl_name}{cl}'] = self.kmeans[f'{self.cl_name}{cl}'].fit_predict(df[self.features])

        if self.targets is not None:
            if self.cl_name is not None:
                self.transforms.update(self.GroupTransform(df, self.cl_name, self.targets, self.transform_type, self.cluster_range))

            if self.groups is not None:
                for group in self.groups:
                    self.transforms.update(self.GroupTransform(df, group, self.targets, self.transform_type, cluster_range=None))

            for key in self.transforms.keys():
                df = df.merge(self.transforms[key], left_on=key.split('-')[1], right_index=True)
                if self.comp_type is not None:
                    df = df.join(self.CompInGroup(df, key, self.comp_type, self.comp_agg))

        return df

    def test(self, df):
        if self.clustering:
            for cl in self.cluster_range:
                df[f'{self.cl_name}{cl}'] = self.kmeans[f'{self.cl_name}{cl}'].predict(df[self.features])

        if self.targets is not None:
            for key in self.transforms.keys():
                df = df.merge(self.transforms[key], left_on=key.split('-')[1], right_index=True)
                if self.comp_type is not None:
                    df = df.join(self.CompInGroup(df, key, self.comp_type, self.comp_agg))

        return df