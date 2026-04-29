# =============================================================================
# TASK 1: DATA PREPROCESSING
# Dating App Match Predictor
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# STEP 1: Load the Dataset
# WHY: First we need to understand what we're working with before touching it.
# -----------------------------------------------------------------------------
df = pd.read_csv('dating_dataset.csv')

print("=" * 60)
print("STEP 1: INITIAL DATA EXPLORATION")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nData types:\n{df.dtypes}")

# -----------------------------------------------------------------------------
# STEP 2: Check Data Quality
# WHY: We need to know if there are missing values, duplicates, or anomalies
#      before doing anything else. Dirty data = bad model.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA QUALITY CHECK")
print("=" * 60)

print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nNumerical summary:\n{df.describe()}")

# No nulls found - this is a clean dataset, but we still validate ranges
print(f"\napp_usage_time_min range: {df['app_usage_time_min'].min()} - {df['app_usage_time_min'].max()}")
print(f"swipe_right_ratio range: {df['swipe_right_ratio'].min()} - {df['swipe_right_ratio'].max()}")
print(f"profile_pics_count range: {df['profile_pics_count'].min()} - {df['profile_pics_count'].max()}")

# -----------------------------------------------------------------------------
# STEP 3: Explore the Target Variable
# WHY: Understanding class distribution tells us if we have a balanced or
#      imbalanced classification problem - this affects model choice and metrics.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TARGET VARIABLE ANALYSIS")
print("=" * 60)

print(f"\nmatch_outcome distribution:\n{df['match_outcome'].value_counts()}")
print(f"\nClass balance (%):\n{(df['match_outcome'].value_counts(normalize=True)*100).round(2)}")

# The 10 outcomes can be grouped into POSITIVE (good match) and NEGATIVE outcomes
# WHY: For a dating app, we care about predicting "good" outcomes vs "bad" ones.
# This transforms a 10-class problem into a cleaner binary classification.
POSITIVE_OUTCOMES = ['Mutual Match', 'Date Happened', 'Relationship Formed', 'Instant Match']
NEGATIVE_OUTCOMES = ['No Action', 'Chat Ignored', 'Ghosted', 'One-sided Like', 'Blocked', 'Catfished']

df['is_good_match'] = df['match_outcome'].apply(
    lambda x: 1 if x in POSITIVE_OUTCOMES else 0
)

print(f"\nBinary target distribution:")
print(df['is_good_match'].value_counts())
print(f"Good match rate: {df['is_good_match'].mean()*100:.1f}%")

# -----------------------------------------------------------------------------
# STEP 4: Feature Engineering on Interest Tags
# WHY: The interest_tags column contains comma-separated strings like
#      "Fitness, Politics, Traveling". Raw text can't go into ML models.
#      We extract the NUMBER of interests as a useful numeric feature,
#      and we also identify the top interests as binary features.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING - INTEREST TAGS")
print("=" * 60)

# Count number of interests per user
df['interest_count'] = df['interest_tags'].apply(lambda x: len(x.split(',')))
print(f"Interest count stats:\n{df['interest_count'].describe()}")

# Find all unique interests across all users
all_interests = []
for tags in df['interest_tags']:
    all_interests.extend([t.strip() for t in tags.split(',')])

from collections import Counter
interest_freq = Counter(all_interests)
top_interests = [interest for interest, _ in interest_freq.most_common(15)]
print(f"\nTop 15 interests: {top_interests}")

# Create binary columns for top 15 interests (1 = user has this interest)
# WHY: These become concrete features the model can use to detect interest patterns
for interest in top_interests:
    col_name = f"has_{interest.lower().replace(' ', '_')}"
    df[col_name] = df['interest_tags'].apply(lambda x: 1 if interest in x else 0)

print(f"\nNew interest feature columns added: {len(top_interests)}")

# -----------------------------------------------------------------------------
# STEP 5: Feature Engineering - Behavioral Signals
# WHY: Raw numbers sometimes need to be combined into more meaningful signals.
#      A user's "engagement score" is more informative than any single number.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: FEATURE ENGINEERING - BEHAVIORAL SIGNALS")
print("=" * 60)

# Engagement score: composite of likes, messages, and mutual matches
# WHY: High engagement signals an active user who is more likely to match
df['engagement_score'] = (
    df['likes_received'] * 0.3 +
    df['mutual_matches'] * 0.4 +
    df['message_sent_count'] * 0.3
)

# Profile completeness score
# WHY: Users with more photos and longer bios tend to get more matches
df['profile_completeness'] = (
    (df['profile_pics_count'] / df['profile_pics_count'].max()) * 0.5 +
    (df['bio_length'] / df['bio_length'].max()) * 0.5
)

# Activity score: combines usage time and swipe ratio
# WHY: Active users with high swipe ratios are more likely to generate matches
df['activity_score'] = df['app_usage_time_min'] * df['swipe_right_ratio']

print("New engineered features: engagement_score, profile_completeness, activity_score")
print(f"\nEngagement score stats:\n{df['engagement_score'].describe()}")

# -----------------------------------------------------------------------------
# STEP 6: Encode Categorical Variables
# WHY: ML models work with numbers, not strings. We need to convert
#      categorical columns like gender, education_level etc. to numbers.
#      We use Label Encoding for ordinal features and keep track of encoders
#      so we can decode predictions later.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

categorical_cols = [
    'gender', 'sexual_orientation', 'location_type',
    'income_bracket', 'education_level',
    'app_usage_time_label', 'swipe_right_label', 'swipe_time_of_day'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {le.classes_.tolist()}")

# -----------------------------------------------------------------------------
# STEP 7: Select Final Feature Set
# WHY: We drop raw columns we've already processed, label columns that are
#      redundant with their numeric counterparts, and the original target.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: SELECTING FINAL FEATURES")
print("=" * 60)

# Columns to drop - raw categoricals replaced by encoded versions,
# label columns are redundant with their ratio counterparts,
# original multi-class target replaced by binary target
drop_cols = [
    'gender', 'sexual_orientation', 'location_type', 'income_bracket',
    'education_level', 'interest_tags', 'app_usage_time_label',
    'swipe_right_label', 'swipe_time_of_day', 'match_outcome'
]

df_model = df.drop(columns=drop_cols)
print(f"Final feature set shape: {df_model.shape}")
print(f"Features: {[c for c in df_model.columns if c != 'is_good_match']}")

# -----------------------------------------------------------------------------
# STEP 8: Scale Numerical Features
# WHY: Tree-based models (like Random Forest) don't require scaling,
#      but it's good practice and needed if we ever try SVM/logistic regression.
#      We scale AFTER splitting to avoid data leakage.
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: FEATURE SCALING (saved for model training pipeline)")
print("=" * 60)
print("StandardScaler will be applied inside the model training script")
print("to avoid data leakage between train and test sets.")

# -----------------------------------------------------------------------------
# STEP 9: Save Clean Dataset
# -----------------------------------------------------------------------------
df_model.to_csv('dating_dataset_clean.csv', index=False)
print(f"\n Clean dataset saved: dating_dataset_clean.csv")
print(f"   Shape: {df_model.shape}")
print(f"   Good match rate: {df_model['is_good_match'].mean()*100:.1f}%")

# Also save the label encoders and top interests for use in the app
import joblib
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(top_interests, 'top_interests.pkl')
print(" Label encoders saved: label_encoders.pkl")
print(" Top interests saved: top_interests.pkl")
