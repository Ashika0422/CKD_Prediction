"""
Chronic Kidney Disease Classification using k-Nearest Neighbors
CSCI 31022 - Machine Learning and Pattern Recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class CKDClassifier:
    def __init__(self, data_path='data/chronic_kidney_disease.arff'):
        """Initialize the CKD classifier"""
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.label_encoders = {}
        self.scaler = None
        self.pca = None
        self.best_model = None
        self.feature_names = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.preprocessor = None
        
    def _manual_arff_parse(self, filepath):
        """Manual ARFF parser as fallback"""
        attributes, data_rows, data_section = [], [], False
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'): continue
                if line.lower() == '@data': 
                    data_section = True
                    continue
                if line.lower().startswith('@attribute'):
                    parts = line.split()
                    if len(parts) > 1:
                        attributes.append(parts[1].strip("'\""))
                elif data_section:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(attributes): 
                        data_rows.append(values)
        return pd.DataFrame(data_rows, columns=attributes).replace('?', np.nan)
    
    def load_data(self):
        """Load the chronic kidney disease dataset"""
        print("="*80 + "\n1. LOADING DATA\n" + "="*80)
        
        # Handle different file locations
        if not os.path.exists(self.data_path):
            # Try in data folder
            data_folder_path = os.path.join('data', os.path.basename(self.data_path))
            if os.path.exists(data_folder_path):
                self.data_path = data_folder_path
            elif os.path.exists(os.path.basename(self.data_path)):
                self.data_path = os.path.basename(self.data_path)
            else:
                raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        try:
            data, meta = arff.loadarff(self.data_path)
            self.df = pd.DataFrame(data)
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    self.df[col] = self.df[col].str.decode('utf-8')
        except:
            print("Using manual ARFF parser...")
            self.df = self._manual_arff_parse(self.data_path)
        
        print(f"✓ Loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*80 + "\n2. EXPLORATORY DATA ANALYSIS\n" + "="*80)
        
        print("\n📊 Dataset Overview:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Total Missing: {self.df.isnull().sum().sum()}")
        
        print("\n📊 Class Distribution:")
        print(self.df['class'].value_counts())
        
        # Identify numeric vs categorical columns
        temp_df = self.df.copy().replace('?', np.nan)
        numeric_cols = []
        categorical_cols = []
        
        for col in self.df.columns:
            if col == 'class': continue
            try:
                pd.to_numeric(temp_df[col].dropna())
                numeric_cols.append(col)
            except:
                categorical_cols.append(col)
        
        print(f"\n✓ Numeric features: {len(numeric_cols)}")
        print(f"✓ Categorical features: {len(categorical_cols)}")
        
        return numeric_cols, categorical_cols
    
    def visualize_data(self, numeric_cols, categorical_cols):
        """Create visualizations for EDA"""
        print("\n" + "="*80 + "\n3. DATA VISUALIZATION\n" + "="*80)
        
        # Class distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts = self.df['class'].value_counts()
        bars = ax.bar(class_counts.index.astype(str), class_counts.values, 
                      color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
        ax.set_title('Class Distribution', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300)
        print("✓ Saved: class_distribution.png")
        plt.close()
        
        # Missing values visualization - Enhanced
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top: Heatmap with better colors
        sns.heatmap(self.df.isnull(), cbar=True, cmap='RdYlGn_r', 
                   ax=axes[0], yticklabels=False, cbar_kws={'label': 'Missing (red) / Present (green)'})
        axes[0].set_title('Missing Values Pattern Across Dataset', 
                         fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Features', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Samples', fontweight='bold', fontsize=12)
        
        # Bottom: Bar chart showing count and percentage
        missing_data = pd.DataFrame({
            'Feature': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100)
        }).sort_values('Missing_Count', ascending=False)
        
        # Only show features with missing values
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        if len(missing_data) > 0:
            bars = axes[1].barh(missing_data['Feature'], missing_data['Missing_Percentage'], 
                               color='#e74c3c', edgecolor='black', linewidth=1.2)
            axes[1].set_xlabel('Missing Values (%)', fontweight='bold', fontsize=12)
            axes[1].set_ylabel('Features', fontweight='bold', fontsize=12)
            axes[1].set_title('Missing Values by Feature', fontsize=16, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, row) in enumerate(zip(bars, missing_data.itertuples())):
                axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{row.Missing_Percentage:.1f}% ({int(row.Missing_Count)})',
                           va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('missing_values.png', dpi=300)
        print("✓ Saved: missing_values.png")
        plt.close()
        
        # Correlation heatmap for numeric features
        df_num = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(14, 12))
            corr = df_num.corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300)
            print("✓ Saved: correlation_heatmap.png")
            plt.close()
    
    def preprocess_data(self, numeric_cols, categorical_cols):
        """Clean and prepare the data (no imputation or encoding yet)"""
        print("\n" + "="*80 + "\n4. DATA PREPROCESSING\n" + "="*80)
        
        df_clean = self.df.replace('?', np.nan).copy()
        
        # Convert numeric columns to proper numeric type
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Separate features and target
        X = df_clean.drop('class', axis=1)
        y = df_clean['class']
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        self.label_encoders['class'] = le_target
        
        # Store column names for later
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        
        print(f"✓ Data cleaned. Shape: {X.shape}")
        print(f"✓ Numeric features: {len(numeric_cols)}")
        print(f"✓ Categorical features: {len(categorical_cols)}")
        print(f"✓ Missing values: {X.isnull().sum().sum()}")
        print("   (Will be imputed after train/test split to prevent leakage)")
        
        return X, y_encoded
    
    def split_and_scale(self, X, y, test_size=0.2):
        """Split data and create preprocessing pipeline (fit only on training data)"""
        print("\n" + "="*80 + "\n5. TRAIN-TEST SPLIT & PREPROCESSING PIPELINE\n" + "="*80)
        
        # CRITICAL: Split BEFORE any preprocessing to prevent data leakage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set:  {self.X_test.shape[0]} samples")
        
        # Create preprocessing pipeline
        print("\n⚙️  Creating preprocessing pipeline...")
        
        # Numeric features: impute with KNN, then scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features: impute with most frequent, then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        # Fit preprocessor on training data ONLY
        print("✓ Fitting preprocessor on training data only...")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        
        # Get feature names after transformation
        num_features = self.numeric_cols
        try:
            cat_features = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_cols)
        except:
            cat_features = []
        self.feature_names = list(num_features) + list(cat_features)
        
        print(f"✓ Preprocessing complete")
        print(f"   - Original features: {len(self.numeric_cols) + len(self.categorical_cols)}")
        print(f"   - After one-hot encoding: {self.X_train.shape[1]}")
        print(f"   - Missing values in train: 0 (imputed)")
        print(f"   - Missing values in test: 0 (imputed)")
    
    def feature_importance(self):
        """Analyze feature importance using mutual information"""
        print("\n" + "="*80 + "\n6. FEATURE IMPORTANCE ANALYSIS\n" + "="*80)
        
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=RANDOM_STATE)
        
        # Use simplified feature names (original numeric + categorical indicators)
        feature_labels = self.numeric_cols + self.categorical_cols
        if len(mi_scores) > len(feature_labels):
            # More features due to one-hot encoding, use first N for display
            mi_aggregated = []
            idx = 0
            for i, col in enumerate(self.numeric_cols):
                mi_aggregated.append((col, mi_scores[idx]))
                idx += 1
            for col in self.categorical_cols:
                # Sum MI scores for all one-hot columns from this categorical feature
                n_categories = len([f for f in self.feature_names if f.startswith(col)])
                if n_categories == 0:
                    n_categories = 1
                scores_sum = sum(mi_scores[idx:idx+n_categories])
                mi_aggregated.append((col, scores_sum))
                idx += n_categories
            mi_series = pd.Series(dict(mi_aggregated)).sort_values(ascending=False)
        else:
            mi_series = pd.Series(mi_scores, index=feature_labels[:len(mi_scores)]).sort_values(ascending=False)
        
        print("\n🎯 Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(mi_series.head(10).items(), 1):
            print(f"   {i:2d}. {feature:15s} : {score:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        mi_series.sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Feature Importance (Mutual Information)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Mutual Information Score', fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("\n✓ Saved: feature_importance.png")
        plt.close()
    
    def evaluate_k_parameter(self, X_train, k_range=range(1, 31)):
        """
        KEY REQUIREMENT: Evaluate effect of k on classification accuracy
        Uses cross-validation on TRAINING data only (no test set peeking)
        """
        print("\n" + "="*80 + "\n7. K-PARAMETER ANALYSIS (Assignment Requirement)\n" + "="*80)
        
        train_scores = []
        cv_scores = []
        
        print("\n⚙️  Testing k values from 1 to 30 using 5-fold CV...")
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            # Training accuracy
            knn.fit(X_train, self.y_train)
            train_scores.append(accuracy_score(self.y_train, knn.predict(X_train)))
            # Cross-validation accuracy on training set
            cv_score = cross_val_score(knn, X_train, self.y_train, cv=5, scoring='accuracy')
            cv_scores.append(cv_score.mean())
        
        best_k = k_range[np.argmax(cv_scores)]
        best_cv_acc = max(cv_scores)
        
        print(f"\n🏆 BEST k VALUE (based on CV): {best_k}")
        print(f"   CV Accuracy: {best_cv_acc*100:.2f}%")
        print(f"   Train Accuracy: {train_scores[best_k-1]*100:.2f}%")
        print(f"\n✓ No test set was used in k selection (preventing leakage)")
        
        # Detailed plot
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(k_range, [s*100 for s in train_scores], 
               label='Training Accuracy', marker='o', linewidth=2, markersize=6)
        ax.plot(k_range, [s*100 for s in cv_scores], 
               label='Cross-Validation Accuracy', marker='s', linewidth=2, markersize=6)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
                  label=f'Best k = {best_k}')
        ax.set_xlabel('k (Number of Neighbors)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Effect of k Parameter on Classification Accuracy (5-Fold CV)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('k_value_analysis.png', dpi=300)
        print("\n✓ Saved: k_value_analysis.png")
        plt.close()
        
        return best_k, best_cv_acc
    
    def compare_dimensionality_reduction(self):
        """Compare performance with and without PCA using CV on training data only"""
        print("\n" + "="*80 + "\n8. DIMENSIONALITY REDUCTION (PCA)\n" + "="*80)
        
        results = {}
        
        # 1. Baseline (No PCA)
        print("\n📊 Evaluating WITHOUT PCA...")
        best_k_no_pca, cv_acc_no_pca = self.evaluate_k_parameter(
            self.X_train, range(1, 21)
        )
        results['no_pca'] = {
            'cv_accuracy': cv_acc_no_pca,
            'best_k': best_k_no_pca,
            'n_features': self.X_train.shape[1]
        }
        
        # 2. PCA with 95% variance (fit on training data only)
        print("\n📊 Evaluating WITH PCA (95% variance)...")
        self.pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train_pca = self.pca.fit_transform(self.X_train)
        
        best_k_pca, cv_acc_pca = self.evaluate_k_parameter(
            X_train_pca, range(1, 21)
        )
        results['pca_95'] = {
            'cv_accuracy': cv_acc_pca,
            'best_k': best_k_pca,
            'n_features': self.pca.n_components_,
            'variance_explained': np.sum(self.pca.explained_variance_ratio_)
        }
        
        # Summary
        print("\n" + "="*60)
        print("DIMENSIONALITY REDUCTION COMPARISON (CV on training set):")
        print("="*60)
        print(f"{'Method':<20} {'Features':<12} {'Best k':<10} {'CV Accuracy':<12}")
        print("-"*60)
        print(f"{'Original (No PCA)':<20} {results['no_pca']['n_features']:<12} "
              f"{results['no_pca']['best_k']:<10} {results['no_pca']['cv_accuracy']*100:<11.2f}%")
        print(f"{'PCA (95% var)':<20} {results['pca_95']['n_features']:<12} "
              f"{results['pca_95']['best_k']:<10} {results['pca_95']['cv_accuracy']*100:<11.2f}%")
        print("="*60)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Comparison bar chart
        methods = ['No PCA', 'PCA (95%)']
        accuracies = [results['no_pca']['cv_accuracy']*100, results['pca_95']['cv_accuracy']*100]
        
        bars = axes[0].bar(methods, accuracies, color=['#3498db', '#e67e22'], 
                          edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('CV Accuracy (%)', fontweight='bold')
        axes[0].set_title('CV Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylim([min(accuracies)-5, 100])
        for bar, acc in zip(bars, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.2f}%', ha='center', fontweight='bold')
        
        # PCA variance explained
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum)+1), cumsum*100, marker='o', linewidth=2)
        axes[1].axhline(y=95, color='red', linestyle='--', label='95% threshold')
        axes[1].set_xlabel('Number of Components', fontweight='bold')
        axes[1].set_ylabel('Cumulative Variance Explained (%)', fontweight='bold')
        axes[1].set_title('PCA Variance Analysis', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_comparison.png', dpi=300)
        print("\n✓ Saved: pca_comparison.png")
        plt.close()
        
        # Return the better performing configuration based on CV
        if cv_acc_pca >= cv_acc_no_pca:
            print(f"\n✓ PCA selected based on CV performance")
            print(f"   Will transform test set at final evaluation")
            return 'pca', best_k_pca
        else:
            print(f"\n✓ Original features selected based on CV performance")
            return 'no_pca', best_k_no_pca
    
    def final_model_evaluation(self, use_pca, best_k):
        """Train and evaluate the final optimized model on held-out test set"""
        print("\n" + "="*80 + "\n9. FINAL MODEL TRAINING & EVALUATION\n" + "="*80)
        print("\n🔒 Using held-out test set for final evaluation (first time)")
        
        # Apply PCA to test set if needed
        if use_pca == 'pca':
            print("   Applying PCA transformation to test set...")
            X_train_final = self.pca.transform(self.X_train)
            X_test_final = self.pca.transform(self.X_test)
        else:
            X_train_final = self.X_train
            X_test_final = self.X_test
        
        # Fine-tune hyperparameters using GridSearch on training data
        print("\n⚙️  Final hyperparameter tuning on training set...")
        param_grid = {
            'n_neighbors': [max(1, best_k-1), best_k, min(best_k+1, len(self.y_train))],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_final, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"\n🏆 BEST MODEL PARAMETERS:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        print(f"\n   Best CV score: {grid_search.best_score_*100:.2f}%")
        
        # Predictions on test set
        y_pred = self.best_model.predict(X_test_final)
        y_pred_proba = self.best_model.predict_proba(X_test_final)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n📊 FINAL MODEL PERFORMANCE:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        
        # Classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoders['class'].classes_
        ))
        
        # Visualizations
        self._plot_final_evaluation(y_pred, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _plot_final_evaluation(self, y_pred, y_pred_proba):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        target_names = self.label_encoders['class'].classes_
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'label': 'Count'})
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Label', fontweight='bold')
        
        # ROC Curve (for binary classification)
        if len(target_names) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                           label='Random Classifier')
            axes[0, 1].set_xlabel('False Positive Rate', fontweight='bold')
            axes[0, 1].set_ylabel('True Positive Rate', fontweight='bold')
            axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
            axes[0, 1].legend(loc='lower right')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Performance Metrics Bar Chart
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(self.y_test, y_pred, average='weighted')
        }
        bars = axes[1, 0].bar(metrics.keys(), [v*100 for v in metrics.values()],
                             color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                             edgecolor='black', linewidth=1.5)
        axes[1, 0].set_ylabel('Score (%)', fontweight='bold')
        axes[1, 0].set_title('Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 105])
        for bar, (name, value) in zip(bars, metrics.items()):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value*100:.1f}%', ha='center', fontweight='bold')
        
        # Class Distribution: True vs Predicted
        x = np.arange(len(target_names))
        width = 0.35
        true_counts = pd.Series(self.y_test).value_counts().sort_index().values
        pred_counts = pd.Series(y_pred).value_counts().sort_index().values
        
        axes[1, 1].bar(x - width/2, true_counts, width, label='True',
                      color='steelblue', edgecolor='black')
        axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted',
                      color='coral', edgecolor='black')
        axes[1, 1].set_xlabel('Class', fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontweight='bold')
        axes[1, 1].set_title('True vs Predicted Distribution', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(target_names)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300)
        print("\n✓ Saved: model_evaluation.png")
        plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" "*15 + "CHRONIC KIDNEY DISEASE CLASSIFICATION")
    print(" "*20 + "k-Nearest Neighbors Classifier")
    print("="*80 + "\n")
    
    # Initialize classifier
    ckd = CKDClassifier(data_path='chronic_kidney_disease_full.arff')
    
    # Pipeline - ensuring no data leakage
    ckd.load_data()
    numeric_cols, categorical_cols = ckd.exploratory_data_analysis()
    ckd.visualize_data(numeric_cols, categorical_cols)
    X, y = ckd.preprocess_data(numeric_cols, categorical_cols)
    
    # CRITICAL: Split first, then fit preprocessing on training data only
    ckd.split_and_scale(X, y, test_size=0.2)
    
    ckd.feature_importance()
    
    # Compare with/without PCA using CV (no test set peeking)
    use_pca, best_k = ckd.compare_dimensionality_reduction()
    
    # Final evaluation on test set (first and only time)
    results = ckd.final_model_evaluation(use_pca, best_k)
    
    # Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    files = [
        'class_distribution.png',
        'missing_values.png',
        'correlation_heatmap.png',
        'feature_importance.png',
        'k_value_analysis.png',
        'pca_comparison.png',
        'model_evaluation.png'
    ]
    for f in files:
        print(f"   ✓ {f}")
    
    print(f"\n🎯 FINAL ACCURACY: {results['accuracy']*100:.2f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()