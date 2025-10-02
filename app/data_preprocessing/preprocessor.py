from sklearn.preprocessing import OneHotEncoder
import pandas as pd



def encoding(df):
  encoder = OneHotEncoder(sparse_output=False, drop=None)  # sparse=False gives a dense array

  # Fit and transform 'Target'
  encoded = encoder.fit_transform(df[['Target']])  # make sure it's 2D

  # Create DataFrame for encoded columns
  encoded_df = pd.DataFrame(
      encoded,
      columns=encoder.get_feature_names_out(['Target']),
      index=df.index
  )

  # Combine with original df (drop 'Target' if needed)
  df_encoded = pd.concat([df.drop('Target', axis=1), encoded_df], axis=1)

  return df_encoded
