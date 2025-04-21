class PriorityGroupMedianImputer:
    """
    Custom median imputer with prioritized group-wise imputation.

    Parameters:
    -----------
    target_cols : list or str
        Column(s) to impute missing values in.
    priority_groups : dict
        Dictionary of priority levels to group columns. 
        Example: {1: ['B','C'], 2: ['B']} 
        Higher priority (lower number) will be attempted first.
    """

    def __init__(self, target_cols, priority_groups=None):
        self.priority_groups = priority_groups
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.medians_ = {}  # Stores learned medians for transform()
        self.priority_order_ = None  # Will store sorted priority levels

    def fit(self, df):
        """
        Calculate medians for imputation (grouped by priority groups and global).

        Parameters:
        -----------
        df : pandas.DataFrame
            Input data to compute medians from.
        """
        
        if self.priority_groups is not None:
            self.priority_order_ = sorted(self.priority_groups.keys())
            
            # Calculate medians for each priority level
            for priority in self.priority_order_:
                group_cols = self.priority_groups[priority]
                for col in self.target_cols:
                    key = (priority, tuple(group_cols), col)
                    self.medians_[key] = df.groupby(group_cols)[col].median()
        
        # Calculate global medians as fallback
        for col in self.target_cols:
            self.medians_[('global', None, col)] = df[col].median()
            
        return self

    def transform(self, df):
        """
        Impute missing values using precomputed medians in priority order.

        Parameters:
        -----------
        df : pandas.DataFrame
            Data to impute missing values in.

        Returns:
        --------
        pandas.DataFrame
            Data with missing values imputed.
        """
        df = df.copy()
        
        for col in self.target_cols:
            # Create mask of missing values to track what needs imputation
            missing_mask = df[col].isna()
            
            if not missing_mask.any():
                continue  # No missing values for this column
                
            if self.priority_groups is not None:
                # Try each priority group in order
                for priority in self.priority_order_:
                    group_cols = self.priority_groups[priority]
                    key = (priority, tuple(group_cols), col)
                    
                    # Only impute rows that are still missing
                    current_missing = df.loc[missing_mask, col].isna()
                    
                    if current_missing.any():
                        # Perform the group imputation for current priority
                        df.loc[missing_mask, col] = (
                            df.loc[missing_mask]
                            .groupby(group_cols)[col]
                            .transform(lambda x: x.fillna(self.medians_[key][x.name]))
                        )
                        # Update mask for next priority level
                        missing_mask = df[col].isna()
            
            # Final fallback to global median for any remaining NAs
            if missing_mask.any():
                df.loc[missing_mask, col] = df.loc[missing_mask, col].fillna(
                    self.medians_[('global', None, col)]
                )
                
        return df

    def fit_transform(self, df):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)