from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict
from shapely.strtree import STRtree
from tqdm import tqdm
import numpy as np
import geopandas as gpd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

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
    
class feature_selection:
    def __init__(self, feature_importance=None, threshold=0.9):
        self.fi = feature_importance
        self.threshold = threshold

    def select_low_correlation_features(self, df, threshold=0.9):
        corr_matrix = df.corr().abs()
        clusters = defaultdict(list)
        dropped_features = set()
        
        # Create correlation clusters
        for i, feature in enumerate(corr_matrix.columns):
            if feature not in dropped_features:
                # Find all features highly correlated with current feature
                correlated = corr_matrix.index[corr_matrix[feature] > threshold].tolist()
                clusters[feature].extend([f for f in correlated if f != feature])
                
                # Mark all correlated features (except self) for dropping
                dropped_features.update([f for f in correlated if f != feature])
        
        # Get final selected features (cluster representatives)
        selected_features = [f for f in corr_matrix.columns if f not in dropped_features]
        
        # Create cluster report
        cluster_report = pd.DataFrame([(k, v) for k, v in clusters.items() if v],
                                    columns=['Representative', 'Dropped Features'])
        
        return selected_features, cluster_report

    def select_features_with_importance(self, df, feature_importance, threshold=0.9):
        corr_matrix = df.corr().abs()
        clusters = []
        selected_features = []
        
        remaining_features = set(corr_matrix.columns)
        
        while remaining_features:
            # Get most important remaining feature
            best_feature = max(remaining_features, key=lambda x: feature_importance.get(x, 0))
            selected_features.append(best_feature)
            
            # Find all correlated features
            correlated = set(corr_matrix.index[corr_matrix[best_feature] > threshold].tolist())
            clusters.append((best_feature, list(correlated - {best_feature})))
            
            # Remove all correlated features from remaining set
            remaining_features -= correlated
        
        cluster_report = pd.DataFrame(clusters, columns=['Kept Feature', 'Dropped Features'])
        return selected_features, cluster_report
    
    def select(self, df):
        if self.fi is None:
            self.selected_features_, self.cluster_report_ = self.select_low_correlation_features(df, self.threshold)
        else:
            self.selected_features_, self.cluster_report_ = self.select_features_with_importance(df, self.fi, self.threshold)
        return self.selected_features_, self.cluster_report_

def add_geo_features(houses_gdf, features_gdf, tag, feature_type, radii=[500, 1000, 1500, 2000], max_rad=10000, area_calc=False):
    features = features_gdf[
        (features_gdf[tag] == feature_type) & 
        (~features_gdf.geometry.is_empty)
    ].copy()
    if features.empty:
        return houses_gdf

    utm_crs = houses_gdf.estimate_utm_crs()
    houses_utm = houses_gdf.to_crs(utm_crs)
    features_utm = features.to_crs(utm_crs)

    if area_calc:
        poly_mask = features_utm.geometry.type.isin(['Polygon', 'MultiPolygon'])
        features_utm['area'] = 0.0
        features_utm.loc[poly_mask, 'area'] = features_utm.loc[poly_mask].geometry.area
    else:
        features_utm['area'] = 0.0  # placeholder

    houses_geom = houses_utm.geometry
    features_geom = features_utm.geometry
    tree = STRtree(features_geom)

    # Get all intersecting pairs
    buffers = houses_geom.buffer(max_rad)
    matches_idx = tree.query(buffers, predicate='intersects')
    house_ix, feat_ix = matches_idx

    house_pts = houses_geom.iloc[house_ix].reset_index(drop=True)
    feat_geom = features_geom.iloc[feat_ix].reset_index(drop=True)

    # Use boundaries for polygons
    feat_geom_use = feat_geom.where(
        feat_geom.geom_type == 'Point',
        feat_geom.boundary
    )

    # Compute distances
    distances = house_pts.distance(feat_geom_use)
    areas = features_utm.iloc[feat_ix].area.values

    result_df = pd.DataFrame({
        'house_idx': houses_geom.index[house_ix],
        'distance': distances,
        'area': areas
    })
    new_df = pd.DataFrame(index=houses_gdf.index)
    # Initialize result columns
    for col in [f'closest_{tag}*{feature_type}'] + \
               [f'median_{tag}*{feature_type}_{r}' for r in radii] + \
               [f'mean_{tag}*{feature_type}_{r}' for r in radii] + \
               [f'count_{tag}*{feature_type}_{r}' for r in radii] + \
               ([f'total_area_{tag}*{feature_type}_{r}' for r in radii] if area_calc else []):
        new_df[col] = 0.0 if ('count' in col or 'total_area' in col) else (max_rad + 5000.0)

    # Closest distance
    closest_dist = result_df.groupby('house_idx')['distance'].min()
    new_df.loc[closest_dist.index, f'closest_{tag}*{feature_type}'] = closest_dist.values

    # Radius-based aggregations
    for radius in radii:
        mask = result_df['distance'] <= radius
        group = result_df[mask].groupby('house_idx')

        new_df.loc[group.size().index, f'count_{tag}*{feature_type}_{radius}'] = group.size().values
        new_df.loc[group['distance'].median().index, f'median_{tag}*{feature_type}_{radius}'] = group['distance'].median().values
        new_df.loc[group['distance'].mean().index, f'mean_{tag}*{feature_type}_{radius}'] = group['distance'].mean().values

        if area_calc:
            new_df.loc[group['area'].sum().index, f'total_area_{tag}*{feature_type}_{radius}'] = group['area'].sum().values

    return new_df

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

def SpecialTransform(df):
    X = pd.DataFrame()
    X['area_bed'] = df.area/(df.bedrooms+1)
    X['area_foto'] = df.area/df.foto_amount
    X.loc[df['foto_amount'] == 0, 'area_foto'] = 0
    X['ev_area'] = df.energy_value/df.area
    X['miss_tot'] = df.energy_value_miss*1 + df.area_miss*1 + df.advertiser_miss*1 + df.lat_miss*1 + df.subtype_miss*1
    return X