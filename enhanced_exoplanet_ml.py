#!/usr/bin/env python3
"""
Enhanced NASA Space Apps Challenge 2025 - Advanced Exoplanet Detection ML Pipeline
Incorporating methodologies from top-performing research papers to achieve superior results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import lightgbm as lgb
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available, will skip XGBoost models")
    XGBOOST_AVAILABLE = False
import warnings
import joblib
import os
import time
from collections import Counter

warnings.filterwarnings('ignore')

class EnhancedExoplanetMLPipeline:
    """
    Enhanced ML pipeline incorporating advanced methodologies from research papers
    for superior exoplanet detection performance
    """
    
    def __init__(self, data_dir="./"):
        self.data_dir = data_dir
        self.datasets = {}
        self.processed_data = None
        self.models = {}
        self.best_model = None
        self.ensemble_model = None
        self.scaler = RobustScaler()  # Use RobustScaler as suggested in papers
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.results = {}
        
    def load_datasets(self):
        """Load all three NASA exoplanet datasets with enhanced processing"""
        print("Loading NASA exoplanet datasets...")
        
        # Kepler cumulative dataset
        kepler_file = os.path.join(self.data_dir, "cumulative_2025.09.21_20.01.51.csv")
        if os.path.exists(kepler_file):
            with open(kepler_file, 'r') as f:
                lines = f.readlines()
                header_line = next(i for i, line in enumerate(lines) if line.startswith('kepid'))
            
            self.datasets['kepler'] = pd.read_csv(kepler_file, skiprows=header_line)
            print(f"Loaded Kepler dataset: {self.datasets['kepler'].shape}")
        
        # K2 planets and candidates dataset
        k2_file = os.path.join(self.data_dir, "k2pandc_2025.09.21_20.02.10.csv")
        if os.path.exists(k2_file):
            with open(k2_file, 'r') as f:
                lines = f.readlines()
                header_line = next(i for i, line in enumerate(lines) if line.startswith('pl_name'))
            
            self.datasets['k2'] = pd.read_csv(k2_file, skiprows=header_line)
            print(f"Loaded K2 dataset: {self.datasets['k2'].shape}")
        
        # TESS Objects of Interest dataset
        toi_file = os.path.join(self.data_dir, "TOI_2025.09.21_20.02.02.csv")
        if os.path.exists(toi_file):
            with open(toi_file, 'r') as f:
                lines = f.readlines()
                header_line = next(i for i, line in enumerate(lines) if line.startswith('toi'))
            
            self.datasets['toi'] = pd.read_csv(toi_file, skiprows=header_line)
            print(f"Loaded TESS TOI dataset: {self.datasets['toi'].shape}")
    
    def enhanced_feature_engineering(self):
        """Advanced feature engineering based on research methodologies"""
        if self.processed_data is None:
            return
            
        print("Applying enhanced feature engineering...")
        
        # Original features
        features = self.processed_data.copy()
        
        # 1. Astronomical Physics-Based Features
        # Signal-to-Noise Ratio approximation
        features['snr_estimate'] = features['depth'] / (features['depth'].std() + 1e-8)
        
        # Transit Impact Parameter (b)
        features['impact_param'] = features.get('impact', 
            np.sqrt(1 - (features['duration'] / (features['period'] * features['stellar_radius']))**2))
        
        # Semi-major axis (Kepler's 3rd law)
        features['semi_major_axis'] = ((features['period'] / 365.25) ** (2/3)) * (features['stellar_radius'] ** (1/3))
        
        # Planet density estimate
        features['planet_density'] = features['planet_radius'] / (features['semi_major_axis'] ** 3)
        
        # Stellar flux at planet distance
        features['stellar_flux'] = features['stellar_luminosity'] / (features['semi_major_axis'] ** 2)
        
        # 2. Transit-specific Features
        # Depth-to-duration ratio
        features['depth_duration_ratio'] = features['depth'] / (features['duration'] + 1e-8)
        
        # Normalized transit depth
        features['normalized_depth'] = features['depth'] / (features['stellar_radius'] ** 2)
        
        # Transit frequency
        features['transit_frequency'] = 1 / features['period']
        
        # 3. Statistical Features
        # Ratios and logarithmic transforms
        features['log_period'] = np.log10(features['period'] + 1e-8)
        features['log_planet_radius'] = np.log10(features['planet_radius'] + 1e-8)
        features['log_stellar_temp'] = np.log10(features['stellar_temp'] + 1e-8)
        
        # Interaction features
        features['period_radius_interaction'] = features['period'] * features['planet_radius']
        features['temp_ratio'] = features['equilibrium_temp'] / features['stellar_temp']
        
        # 4. Binned categorical features (as suggested in research)
        # Period bins
        features['period_bin'] = pd.cut(features['period'], bins=5, labels=['ultra_short', 'short', 'medium', 'long', 'ultra_long'])
        features['period_bin'] = features['period_bin'].cat.codes
        
        # Radius bins
        features['radius_bin'] = pd.cut(features['planet_radius'], bins=4, labels=['sub_earth', 'earth_like', 'super_earth', 'mini_neptune'])
        features['radius_bin'] = features['radius_bin'].cat.codes
        
        # 5. Advanced derived features from papers
        # Equilibrium temperature classification
        features['temp_class'] = np.where(features['equilibrium_temp'] < 200, 0,  # Cold
                                 np.where(features['equilibrium_temp'] < 400, 1,  # Cool
                                 np.where(features['equilibrium_temp'] < 1000, 2, 3)))  # Warm, Hot
        
        # Transit timing variations proxy
        features['ttv_proxy'] = features['period'] * features['planet_radius'] / features['stellar_radius']
        
        # Habitable zone indicator
        features['habitable_zone'] = np.where(
            (features['equilibrium_temp'] >= 200) & (features['equilibrium_temp'] <= 300), 1, 0
        )
        
        # Fill missing values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            features[col] = features[col].fillna(features[col].median())
        
        self.processed_data = features
        print(f"Enhanced features created. New shape: {features.shape}")
    
    def advanced_preprocessing(self):
        """Advanced preprocessing pipeline based on research best practices"""
        print("Applying advanced preprocessing...")
        
        # 1. Remove columns with too many missing values (>70%)
        missing_threshold = 0.7
        missing_ratios = self.processed_data.isnull().sum() / len(self.processed_data)
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index
        if len(cols_to_drop) > 0:
            self.processed_data = self.processed_data.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns with >70% missing values")
        
        # 2. Remove constant and quasi-constant features
        constant_features = []
        for col in self.processed_data.select_dtypes(include=[np.number]).columns:
            if col != 'target_class':
                if self.processed_data[col].nunique() <= 1:
                    constant_features.append(col)
        
        if constant_features:
            self.processed_data = self.processed_data.drop(columns=constant_features)
            print(f"Removed {len(constant_features)} constant features")
        
        # 3. Outlier treatment using IQR method
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target_class']
        
        for col in numeric_cols:
            Q1 = self.processed_data[col].quantile(0.25)
            Q3 = self.processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            self.processed_data[col] = np.clip(self.processed_data[col], lower_bound, upper_bound)
    
    def implement_advanced_sampling(self, X, y):
        """Implement advanced sampling strategies from research papers"""
        print("Applying advanced sampling strategies...")
        
        print("Original class distribution:")
        print(Counter(y))
        
        # Use SMOTETomek for combined over and under sampling
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        print("After SMOTETomek sampling:")
        print(Counter(y_resampled))
        
        return X_resampled, y_resampled
    
    def create_ensemble_models(self):
        """Create ensemble models based on research findings"""
        print("Creating advanced ensemble models...")
        
        # Base models with optimized hyperparameters from research
        base_models = {}
        
        # AdaBoost with optimized hyperparameters (from paper: n_estimators=974, learning_rate=0.1)
        base_models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=974,
            learning_rate=0.1,
            random_state=42
        )
        
        # Random Forest with optimized hyperparameters (from paper: n_estimators=1600, criterion='entropy')
        base_models['RandomForest'] = RandomForestClassifier(
            n_estimators=1600,
            criterion='entropy',
            max_depth=None,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees (Extremely Randomized Trees)
        base_models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=200,
            criterion='entropy',
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM (as used in research papers)
        base_models['LightGBM'] = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                eval_metric='mlogloss'
            )
        
        # Gradient Boosting
        base_models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=1600,
            learning_rate=0.1,
            random_state=42
        )
        
        return base_models
    
    def train_with_cross_validation(self, X, y):
        """Train models with 10-fold cross validation as suggested in papers"""
        print("Training models with 10-fold cross validation...")
        
        # Create stratified k-fold cross validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Apply advanced sampling to training data only
        X_train_resampled, y_train_resampled = self.implement_advanced_sampling(X_train, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create models
        models = self.create_ensemble_models()
        
        # Train and evaluate each model
        model_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, 
                                       cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Train on full training set
            model.fit(X_train_scaled, y_train_resampled)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for multiclass
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                auc = 0.0
            
            training_time = time.time() - start_time
            
            model_results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'training_time': training_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Training Time: {training_time:.2f}s")
        
        # Create ensemble using top performers
        top_models = sorted(model_results.items(), 
                           key=lambda x: x[1]['f1_score'], reverse=True)[:3]
        
        print(f"\nCreating ensemble from top 3 models: {[name for name, _ in top_models]}")
        
        # Voting ensemble
        voting_models = [(name, results['model']) for name, results in top_models]
        ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train_resampled)
        
        # Ensemble predictions
        ensemble_pred = ensemble.predict(X_test_scaled)
        ensemble_proba = ensemble.predict_proba(X_test_scaled)
        
        # Ensemble metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        
        try:
            ensemble_auc = roc_auc_score(y_test, ensemble_proba, multi_class='ovr', average='weighted')
        except:
            ensemble_auc = 0.0
        
        model_results['Ensemble'] = {
            'model': ensemble,
            'accuracy': ensemble_accuracy,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1_score': ensemble_f1,
            'auc': ensemble_auc,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        print(f"\nEnsemble Results:")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        print(f"Precision: {ensemble_precision:.4f}")
        print(f"Recall: {ensemble_recall:.4f}")
        print(f"F1 Score: {ensemble_f1:.4f}")
        print(f"AUC: {ensemble_auc:.4f}")
        
        # Select best model
        best_model_name = max(model_results.keys(), 
                             key=lambda k: model_results[k]['f1_score'])
        self.best_model = model_results[best_model_name]['model']
        self.models = model_results
        
        # Store test data
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        print(f"\nBest model: {best_model_name} with F1 Score: {model_results[best_model_name]['f1_score']:.4f}")
        
        return model_results
    
    def comprehensive_evaluation(self):
        """Comprehensive evaluation with multiple metrics as in research papers"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        results_df = []
        
        for name, results in self.models.items():
            row = {
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1 Score': f"{results['f1_score']:.4f}",
                'AUC': f"{results['auc']:.4f}"
            }
            
            if 'cv_mean' in results:
                row['CV Mean'] = f"{results['cv_mean']:.4f}"
                row['CV Std'] = f"{results['cv_std']:.4f}"
                row['Time (s)'] = f"{results['training_time']:.2f}"
            
            results_df.append(row)
        
        results_df = pd.DataFrame(results_df)
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False))
        
        # Detailed classification reports
        target_names = self.label_encoder.classes_
        
        for name, results in self.models.items():
            print(f"\n{name} - Detailed Classification Report:")
            print("-" * 60)
            print(classification_report(self.y_test, results['predictions'], 
                                       target_names=target_names, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, results['predictions'])
            print(f"\nConfusion Matrix:")
            print(cm)
    
    def combine_and_harmonize_data(self):
        """Enhanced data combination with better feature harmonization"""
        print("Preprocessing and combining datasets with enhanced methods...")
        
        # Process individual datasets (using existing methods)
        kepler_processed = self.preprocess_kepler_data()
        toi_processed = self.preprocess_toi_data()
        
        # Combine datasets
        combined_data = []
        
        if kepler_processed is not None:
            # Enhanced Kepler harmonization
            kepler_harmonized = pd.DataFrame()
            kepler_harmonized['period'] = kepler_processed.get('koi_period', np.nan)
            kepler_harmonized['impact'] = kepler_processed.get('koi_impact', np.nan)
            kepler_harmonized['duration'] = kepler_processed.get('koi_duration', np.nan)
            kepler_harmonized['depth'] = kepler_processed.get('koi_depth', np.nan)
            kepler_harmonized['planet_radius'] = kepler_processed.get('koi_prad', np.nan)
            kepler_harmonized['equilibrium_temp'] = kepler_processed.get('koi_teq', np.nan)
            kepler_harmonized['insolation'] = kepler_processed.get('koi_insol', np.nan)
            kepler_harmonized['snr'] = kepler_processed.get('koi_model_snr', np.nan)
            kepler_harmonized['stellar_temp'] = kepler_processed.get('koi_steff', np.nan)
            kepler_harmonized['stellar_logg'] = kepler_processed.get('koi_slogg', np.nan)
            kepler_harmonized['stellar_radius'] = kepler_processed.get('koi_srad', np.nan)
            kepler_harmonized['ra'] = kepler_processed.get('ra', np.nan)
            kepler_harmonized['dec'] = kepler_processed.get('dec', np.nan)
            kepler_harmonized['magnitude'] = kepler_processed.get('koi_kepmag', np.nan)
            kepler_harmonized['target_class'] = kepler_processed['koi_pdisposition']
            kepler_harmonized['dataset_source'] = 'kepler'
            
            # Calculate stellar luminosity for Kepler
            kepler_harmonized['stellar_luminosity'] = (kepler_harmonized['stellar_radius'] ** 2) * \
                                                     ((kepler_harmonized['stellar_temp'] / 5778) ** 4)
            
            combined_data.append(kepler_harmonized)
        
        if toi_processed is not None:
            # Enhanced TOI harmonization
            toi_harmonized = pd.DataFrame()
            toi_harmonized['period'] = toi_processed.get('pl_orbper', np.nan)
            toi_harmonized['impact'] = np.nan  # Not available in TOI
            toi_harmonized['duration'] = toi_processed.get('pl_trandurh', np.nan)
            toi_harmonized['depth'] = toi_processed.get('pl_trandep', np.nan)
            toi_harmonized['planet_radius'] = toi_processed.get('pl_rade', np.nan)
            toi_harmonized['equilibrium_temp'] = toi_processed.get('pl_eqt', np.nan)
            toi_harmonized['insolation'] = toi_processed.get('pl_insol', np.nan)
            toi_harmonized['snr'] = np.nan  # Not available in TOI
            toi_harmonized['stellar_temp'] = toi_processed.get('st_teff', np.nan)
            toi_harmonized['stellar_logg'] = toi_processed.get('st_logg', np.nan)
            toi_harmonized['stellar_radius'] = toi_processed.get('st_rad', np.nan)
            toi_harmonized['ra'] = toi_processed.get('ra', np.nan)
            toi_harmonized['dec'] = toi_processed.get('dec', np.nan)
            toi_harmonized['magnitude'] = toi_processed.get('st_tmag', np.nan)
            toi_harmonized['target_class'] = toi_processed['tfopwg_disp']
            toi_harmonized['dataset_source'] = 'toi'
            
            # Calculate stellar luminosity for TOI
            toi_harmonized['stellar_luminosity'] = (toi_harmonized['stellar_radius'] ** 2) * \
                                                  ((toi_harmonized['stellar_temp'] / 5778) ** 4)
            
            combined_data.append(toi_harmonized)
        
        if combined_data:
            self.processed_data = pd.concat(combined_data, ignore_index=True)
            
            # Handle missing values with improved strategy
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'target_class':
                    # Use dataset-specific medians for better imputation
                    for dataset in self.processed_data['dataset_source'].unique():
                        mask = self.processed_data['dataset_source'] == dataset
                        dataset_median = self.processed_data.loc[mask, col].median()
                        self.processed_data.loc[mask, col] = self.processed_data.loc[mask, col].fillna(dataset_median)
                    
                    # Global median for any remaining NaNs
                    self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].median())
            
            print(f"Enhanced combined dataset shape: {self.processed_data.shape}")
            print("Class distribution:")
            print(self.processed_data['target_class'].value_counts())
            
            return True
        
        return False
    
    def preprocess_kepler_data(self):
        """Enhanced Kepler data preprocessing"""
        if 'kepler' not in self.datasets:
            return None
            
        df = self.datasets['kepler'].copy()
        
        # Enhanced feature selection based on research
        feature_cols = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
            'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg',
            'koi_srad', 'ra', 'dec', 'koi_kepmag', 'koi_time0bk',
            'koi_dor', 'koi_ldm_coeff4', 'koi_ldm_coeff3'  # Additional features
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Target variable
        target_col = 'koi_pdisposition'
        
        features_df = df[available_features + [target_col]].copy()
        features_df = features_df.dropna(subset=[target_col])
        
        # Enhanced class mapping
        class_mapping = {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE', 
            'FALSE POSITIVE': 'FALSE_POSITIVE'
        }
        features_df[target_col] = features_df[target_col].map(class_mapping)
        features_df = features_df.dropna(subset=[target_col])
        
        return features_df
    
    def preprocess_toi_data(self):
        """Enhanced TOI data preprocessing"""
        if 'toi' not in self.datasets:
            return None
            
        df = self.datasets['toi'].copy()
        
        # Enhanced feature selection
        feature_cols = [
            'pl_orbper', 'pl_trandurh', 'pl_trandep', 'pl_rade', 'pl_insol',
            'pl_eqt', 'st_tmag', 'st_dist', 'st_teff', 'st_logg', 'st_rad',
            'ra', 'dec', 'pl_tranmid'  # Additional features
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        target_col = 'tfopwg_disp'
        
        features_df = df[available_features + [target_col]].copy()
        features_df = features_df.dropna(subset=[target_col])
        
        # Enhanced class mapping
        class_mapping = {
            'CP': 'CONFIRMED',
            'KP': 'CONFIRMED',
            'PC': 'CANDIDATE',
            'FP': 'FALSE_POSITIVE'
        }
        features_df[target_col] = features_df[target_col].map(class_mapping)
        features_df = features_df.dropna(subset=[target_col])
        
        return features_df
    
    def prepare_ml_data(self):
        """Enhanced ML data preparation"""
        if self.processed_data is None:
            return None, None
            
        # Select features for ML (excluding metadata)
        exclude_cols = ['target_class', 'dataset_source']
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        X = self.processed_data[feature_cols]
        y = self.processed_data['target_class']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded
    
    def run_enhanced_pipeline(self):
        """Run the complete enhanced ML pipeline"""
        print("Starting Enhanced Exoplanet ML Pipeline...")
        print("="*60)
        
        # Load and preprocess data
        self.load_datasets()
        if not self.combine_and_harmonize_data():
            print("Error: Could not process datasets")
            return
        
        # Enhanced feature engineering
        self.enhanced_feature_engineering()
        
        # Advanced preprocessing
        self.advanced_preprocessing()
        
        # Prepare ML data
        X, y = self.prepare_ml_data()
        if X is None or y is None:
            print("Error: Could not prepare ML data")
            return
        
        # Train models with enhanced methodology
        model_results = self.train_with_cross_validation(X, y)
        
        # Comprehensive evaluation
        self.comprehensive_evaluation()
        
        # Save models
        joblib.dump(self.best_model, 'enhanced_best_model.pkl')
        joblib.dump(self.scaler, 'enhanced_scaler.pkl')
        joblib.dump(self.label_encoder, 'enhanced_label_encoder.pkl')
        
        print("\n" + "="*60)
        print("Enhanced Pipeline completed successfully!")
        print(f"Best model saved as: enhanced_best_model.pkl")
        print(f"Feature scaler saved as: enhanced_scaler.pkl")
        print(f"Label encoder saved as: enhanced_label_encoder.pkl")
        
        return model_results

if __name__ == "__main__":
    # Initialize and run the enhanced pipeline
    pipeline = EnhancedExoplanetMLPipeline("/Users/bharath/Desktop/Nasa space app 1st/")
    pipeline.run_enhanced_pipeline()
