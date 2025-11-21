import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Load dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Examine class distribution
print("Class distribution:", Counter(df['Diabetes_binary']))
print("Dataset shape:", df.shape)
print("Features:", df.columns.tolist())