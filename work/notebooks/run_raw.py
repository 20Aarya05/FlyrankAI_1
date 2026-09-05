import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

sys.stdout.reconfigure(line_buffering=True)
print("Started")

# Load data
df = pd.read_csv('../../data/raw/content_refresh_anonymized.csv')
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)

# Baseline score definition (from W04)
df['stale'] = (df['days_since_last_update'] >= 180).astype(int)
df['visible'] = (df['impressions_90d'] >= 500).astype(int)
df['baseline_score'] = df['stale'] * df['visible'] * df['impressions_90d']

# Features (exclude leakage like trend_pct, trend_direction, is_declining_label)
num_features = ['search_volume', 'cpc', 'word_count', 'char_count', 'impressions_90d',
                'clicks_90d', 'pageviews_90d', 'sessions_90d', 'users_90d',
                'engaged_sessions_90d', 'ai_sessions_90d', 'scroll_events_90d',
                'days_with_impressions', 'days_with_sessions', 'content_age_days',
                'days_since_last_update', 'ctr', 'avg_position', 'engagement_rate', 
                'scroll_rate', 'ai_traffic_pct']
cat_features = ['competition_level', 'content_type', 'main_intent', 'provider_used', 'model_used']

X = df[num_features + cat_features]
y = df['is_declining_label']
groups = df['client_id']

# Split: 80% train, 20% test, grouped by client_id
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
baseline_train, baseline_test = df['baseline_score'].iloc[train_idx], df['baseline_score'].iloc[test_idx]
client_test = df['client_id'].iloc[test_idx]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Base rate (Train): {y_train.mean():.3f} | Base rate (Test): {y_test.mean():.3f}")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Models
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Random Forest (Depth 10)': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100, n_jobs=1))
    ])
}

def eval_scores(y_true, y_pred_proba):
    return {
        'ROC AUC': roc_auc_score(y_true, y_pred_proba),
        'Avg Precision': average_precision_score(y_true, y_pred_proba)
    }

results = {}
results['Baseline (W04)'] = eval_scores(y_test, baseline_test)

for name, model in models.items():
    print(f"Fitting {name}...")
    model.fit(X_train, y_train)
    print(f"Predicting {name}...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results[name] = eval_scores(y_test, y_pred_proba)

print("@@@SPLIT@@@")
results_df = pd.DataFrame(results).T
results_df.index.name = 'Method'
print("### Model vs Baseline Comparison")
print(results_df.round(3))

print("@@@SPLIT@@@")
rf_model = models['Random Forest (Depth 10)']
rf_classifier = rf_model.named_steps['classifier']
cat_encoder = rf_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
all_feature_names = num_features + list(cat_feature_names)

importances = pd.Series(rf_classifier.feature_importances_, index=all_feature_names)
print("\\n### Top 10 Features (Random Forest)")
print(importances.sort_values(ascending=False).head(10).round(4))

y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
df_test = df.iloc[test_idx].copy()
df_test['rf_proba'] = y_pred_proba_rf

print("\\n### Hardest False Positives (Predicted high risk, but label is 0)")
fp = df_test[(df_test['is_declining_label'] == 0)].sort_values('rf_proba', ascending=False).head(3)
for idx, row in fp.iterrows():
    print(f"Content {row['content_id'][:10]} | Proba: {row['rf_proba']:.3f} | Age: {row['content_age_days']} | Imp_90d: {row['impressions_90d']}")

print("\\n### Hardest False Negatives (Predicted low risk, but label is 1)")
fn = df_test[(df_test['is_declining_label'] == 1)].sort_values('rf_proba', ascending=True).head(3)
for idx, row in fn.iterrows():
    print(f"Content {row['content_id'][:10]} | Proba: {row['rf_proba']:.3f} | Age: {row['content_age_days']} | Imp_90d: {row['impressions_90d']}")
