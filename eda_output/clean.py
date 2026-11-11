# Automated data cleaning produced by EDAgent
df_clean = df.copy()
# Fill numerical nulls with median and categoricals with mode
num_cols = df_clean.select_dtypes(include=['number']).columns
for c in num_cols:
    if df_clean[c].isna().sum() > 0:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
cat_cols = df_clean.select_dtypes(exclude=['number']).columns
for c in cat_cols:
    if df_clean[c].isna().sum() > 0:
        mode_val = df_clean[c].mode()[0] if not df_clean[c].mode().empty else 'Unknown'
        df_clean[c] = df_clean[c].fillna(mode_val)