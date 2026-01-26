"""
Data Quality Analyzer - Core Analysis Engine
Detects missing values, duplicates, outliers, and generates recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class DataQualityAnalyzer:
    """
    Analyzes dataset for quality issues and generates verified recommendations
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = {
            'missing_values': {},
            'duplicates': {},
            'outliers': {},
            'inconsistencies': {}
        }
    
    def run_full_analysis(self) -> Dict:
        """Run all quality checks"""
        print("🔍 Running data quality analysis...")
        
        self.analyze_missing_values()
        self.detect_duplicates()
        self.detect_outliers()
        self.detect_inconsistencies()
        
        return self.issues
    
    def analyze_missing_values(self):
        """Detect and analyze missing values"""
        print("  → Checking for missing values...")
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            
            if missing_count > 0:
                missing_pct = (missing_count / len(self.df)) * 100
                missing_rows = self.df[self.df[col].isna()].index.tolist()
                
                # Generate recommendations
                recommendations = self._get_imputation_recommendations(col, missing_count, missing_pct)
                
                self.issues['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2),
                    'rows_sample': missing_rows[:10],  # First 10
                    'total_affected_rows': len(missing_rows),
                    'recommendations': recommendations
                }
        
        print(f"     Found missing values in {len(self.issues['missing_values'])} columns")
    
    def _get_imputation_recommendations(self, col: str, count: int, pct: float) -> List[Dict]:
        """Generate verified recommendations for handling missing values"""
        recommendations = []
        col_data = self.df[col].dropna()
        
        if len(col_data) == 0:
            return [{
                'method': 'Column is completely empty',
                'priority': '❌ CRITICAL',
                'reason': 'All values are missing',
                'action': 'Consider removing this column'
            }]
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = float(col_data.mean())
            median_val = float(col_data.median())
            std_val = float(col_data.std())
            skewness = float(col_data.skew())
            
            # Recommendation 1: Median or Mean based on skewness
            if abs(skewness) > 0.5:
                recommendations.append({
                    'method': 'Median Imputation',
                    'priority': '⭐ RECOMMENDED',
                    'value': round(median_val, 2),
                    'reason': f'Data is skewed (skewness={skewness:.2f}). Median is more robust to outliers.',
                    'code': f"df['{col}'].fillna({median_val:.2f}, inplace=True)",
                    'impact': f'Preserves all {count} rows. Robust to outliers.'
                })
            else:
                recommendations.append({
                    'method': 'Mean Imputation',
                    'priority': '⭐ RECOMMENDED',
                    'value': round(mean_val, 2),
                    'reason': f'Data is approximately symmetric (skewness={skewness:.2f}). Mean preserves distribution.',
                    'code': f"df['{col}'].fillna({mean_val:.2f}, inplace=True)",
                    'impact': f'Preserves all {count} rows. Best for symmetric data.'
                })
            
            # Recommendation 2: KNN Imputation
            recommendations.append({
                'method': 'KNN Imputation',
                'priority': '⚡ ADVANCED',
                'reason': 'Predicts missing values based on similar rows using k-nearest neighbors.',
                'code': f"from sklearn.impute import KNNImputer\nimputer = KNNImputer(n_neighbors=5)\ndf_imputed = imputer.fit_transform(df)",
                'impact': 'More accurate but computationally expensive. Use if this feature is critical.'
            })
            
            # Recommendation 3: Remove rows
            if pct < 5:
                recommendations.append({
                    'method': 'Remove Rows',
                    'priority': '✅ ACCEPTABLE',
                    'reason': f'Only {pct:.1f}% missing - minimal data loss.',
                    'code': f"df.dropna(subset=['{col}'], inplace=True)",
                    'impact': f'Dataset size: {len(self.df)} → {len(self.df) - count} rows'
                })
            else:
                recommendations.append({
                    'method': 'Remove Rows',
                    'priority': '❌ NOT RECOMMENDED',
                    'reason': f'Would lose {pct:.1f}% of data - too much!',
                    'code': f"df.dropna(subset=['{col}'], inplace=True)",
                    'impact': f'Would remove {count} patients. Significant data loss.'
                })
        
        else:  # Categorical columns
            if len(col_data.mode()) > 0:
                mode_val = col_data.mode()[0]
                recommendations.append({
                    'method': 'Mode Imputation',
                    'priority': '⭐ RECOMMENDED',
                    'value': str(mode_val),
                    'reason': f'Most frequent value: "{mode_val}"',
                    'code': f"df['{col}'].fillna('{mode_val}', inplace=True)",
                    'impact': f'Fills {count} missing values with most common category.'
                })
        
        return recommendations
    
    def detect_duplicates(self):
        """Find duplicate records"""
        print("  → Checking for duplicates...")
        
        duplicates = self.df[self.df.duplicated(keep=False)]
        
        if len(duplicates) > 0:
            # Group duplicates
            duplicate_groups = []
            seen = set()
            
            for idx in duplicates.index:
                if idx not in seen:
                    row = self.df.loc[idx]
                    matching = self.df[self.df.eq(row).all(axis=1)].index.tolist()
                    if len(matching) > 1:
                        duplicate_groups.append(matching)
                        seen.update(matching)
            
            self.issues['duplicates'] = {
                'count': len(duplicates),
                'groups': duplicate_groups[:5],  # Show first 5 groups
                'recommendation': {
                    'method': 'Remove Duplicates',
                    'priority': '⭐ RECOMMENDED',
                    'reason': 'Duplicates inflate sample size and can cause data leakage in train/test splits.',
                    'code': "df.drop_duplicates(keep='first', inplace=True)",
                    'impact': f'Will remove {len(duplicates) - len(duplicate_groups)} duplicate rows'
                }
            }
            
            print(f"     Found {len(duplicates)} duplicate records in {len(duplicate_groups)} groups")
        else:
            print("     No duplicates found")
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        print("  → Checking for outliers...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) < 4:  # Need at least 4 values for IQR
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_count += len(outliers)
                
                self.issues['outliers'][col] = {
                    'count': len(outliers),
                    'rows_sample': outliers.index.tolist()[:10],
                    'values_sample': outliers[col].tolist()[:10],
                    'bounds': {
                        'lower': round(float(lower_bound), 2),
                        'upper': round(float(upper_bound), 2)
                    },
                    'recommendations': self._get_outlier_recommendations(col, len(outliers))
                }
        
        print(f"     Found outliers in {len(self.issues['outliers'])} columns")
    
    def _get_outlier_recommendations(self, col: str, count: int) -> List[Dict]:
        """Generate recommendations for handling outliers"""
        recommendations = []
        
        # Domain-specific logic
        if any(term in col.lower() for term in ['cholesterol', 'glucose', 'pressure', 'bmi']):
            recommendations.append({
                'method': 'Keep as is',
                'priority': '⭐ RECOMMENDED',
                'reason': f'High {col} values are medically relevant and indicate disease risk. These are real patients, not errors.',
                'code': '# No action needed - keep outliers'
            })
        elif 'age' in col.lower() or 'heart_rate' in col.lower():
            recommendations.append({
                'method': 'Manual Review Required',
                'priority': '⚠️ CRITICAL',
                'reason': 'Could be data entry errors (age=0) or valid edge cases (athletes with low heart rate).',
                'code': '# Review original records before deciding'
            })
        else:
            recommendations.append({
                'method': 'Cap at Percentiles',
                'priority': 'OPTION',
                'reason': 'Reduce extreme outlier influence by capping at 95th percentile.',
                'code': f"upper_limit = df['{col}'].quantile(0.95)\ndf['{col}'] = df['{col}'].clip(upper=upper_limit)"
            })
        
        return recommendations
    
    def detect_inconsistencies(self):
        """Detect inconsistent encoding in categorical columns"""
        print("  → Checking for inconsistencies...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_values = self.df[col].dropna().unique()
            
            # Check for mixed case (Male, male, MALE)
            if len(unique_values) > 1:
                lower_values = [str(v).lower() for v in unique_values]
                if len(lower_values) != len(set(lower_values)):
                    self.issues['inconsistencies'][col] = {
                        'type': 'Mixed case encoding',
                        'values': list(unique_values)[:10],
                        'recommendation': {
                            'method': 'Standardize encoding',
                            'priority': '⭐ RECOMMENDED',
                            'code': f"df['{col}'] = df['{col}'].str.lower().str.strip()",
                            'reason': 'Convert all to lowercase and remove whitespace for consistency.'
                        }
                    }
        
        if self.issues['inconsistencies']:
            print(f"     Found inconsistencies in {len(self.issues['inconsistencies'])} columns")
        else:
            print("     No inconsistencies found")
    
    def get_summary(self) -> Dict:
        """Get analysis summary"""
        return {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'missing_value_columns': len(self.issues['missing_values']),
            'duplicate_records': self.issues['duplicates'].get('count', 0),
            'outlier_columns': len(self.issues['outliers']),
            'inconsistent_columns': len(self.issues['inconsistencies'])
        }


# Test if run directly
if __name__ == "__main__":
    print("Testing Data Quality Analyzer...")
    print("=" * 60)
    
    # Load one of our synthetic datasets
    df = pd.read_csv('../data/synthetic/dataset1_heart_disease_quality.csv')
    
    print(f"\n📊 Analyzing: dataset1_heart_disease_quality.csv")
    print(f"Records: {len(df)}, Features: {len(df.columns)}")
    
    # Run analysis
    analyzer = DataQualityAnalyzer(df)
    issues = analyzer.run_full_analysis()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    summary = analyzer.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ Data Quality Analyzer test complete!")