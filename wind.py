import pandas as pd
import numpy as np

# Read the wind data
df = pd.read_csv('Wind data_texas.csv')

# Get the single windmill power in kW (column name might have spaces/pipes)
power_col = 'System power generated | (kW)'

# Calculate windfarm power: single windmill kW * 400 windmills / 1000 to convert to MW
df['windfarm power (MW)'] = (df[power_col] * 400) / 1000

# Add 10% variability (random noise between -10% and +10%)
np.random.seed(42)  # For reproducibility
variability = np.random.uniform(-0.10, 0.10, len(df))  # ±10%
df['windfarm power (MW)'] = df['windfarm power (MW)'] * (1 + variability)

# Round to 2 decimal places for readability
df['windfarm power (MW)'] = df['windfarm power (MW)'].round(2)

# Save to new file (or overwrite if you want)
df.to_csv('Wind data_texas.csv', index=False)

print(f"✅ Added 'windfarm power (MW)' column")
print(f"   Sample values:")
print(df[['Time stamp', power_col, 'windfarm power (MW)']].head(10))
print(f"\n   Total rows: {len(df)}")
print(f"   Windfarm power range: {df['windfarm power (MW)'].min():.2f} - {df['windfarm power (MW)'].max():.2f} MW")