from hopsworks_connection import connect_to_hopsworks, get_or_create_feature_group



# =================================== Temporary file to check data added in Hopsworks or not ===================================
project = connect_to_hopsworks()
fg = get_or_create_feature_group(project)
df = fg.read()

print(f"Total rows: {len(df)}")
print(df.sort_values("timestamp").tail(5))