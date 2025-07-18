import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

# Load data
print("Loading data...")
data = pd.read_csv('data/Airbnb_Open_Data.csv')

# Clean and preprocess data
print("Preprocessing data...")

# Clean price and service fee columns
data['price'] = data['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)
data['service fee'] = data['service fee'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip().astype(float)

# Handle missing values
data['reviews per month'] = data['reviews per month'].fillna(0)
data['review rate number'] = data['review rate number'].fillna(0)
data['Construction year'] = data['Construction year'].fillna(data['Construction year'].median())
data['calculated host listings count'] = data['calculated host listings count'].fillna(1)

# Backup original categorical columns
original_neigh = data['neighbourhood group'].copy()
original_bookable = data['instant_bookable'].copy()
original_room_type = data['room type'].copy()
original_cancellation = data['cancellation_policy'].copy()
original_host_verified = data['host_identity_verified'].copy()

# Define mappings for categorical variables
neigh_price_map = {
    'Manhattan': 1.0, 'manhatan': 1.0,
    'Brooklyn': 0.8, 'brookln': 0.8,
    'Queens': 0.6, 'Bronx': 0.5, 'Staten Island': 0.4
}

bookable_price_map = {True: 0.7, False: 0.3}

room_type_map = {
    'Entire home/apt': 1.0, 'Hotel room': 0.85,
    'Private room': 0.6, 'Shared room': 0.3
}

cancellation_map = {
    'strict': 0.8, 'moderate': 0.6, 'flexible': 0.4
}

host_verified_map = {
    'verified': 0.7, 'unconfirmed': 0.3
}

# Apply mappings to create encoded columns
data['neigh_encoded'] = data['neighbourhood group'].map(neigh_price_map).fillna(0.5)
data['room_type_encoded'] = data['room type'].map(room_type_map).fillna(0.5)
data['bookable_encoded'] = data['instant_bookable'].map(bookable_price_map).fillna(0.5)
data['cancellation_encoded'] = data['cancellation_policy'].map(cancellation_map).fillna(0.5)
data['host_verified_encoded'] = data['host_identity_verified'].map(host_verified_map).fillna(0.5)

# Create some feature interactions and transformations
print("Creating feature transformations...")

# Location quality score (combination of neighborhood and exact location)
data['location_score'] = data['neigh_encoded'] * (1 + 0.2 * np.sin(data['lat'] * 10) + 0.15 * np.cos(data['long'] * 8))

# Popularity score based on reviews
data['popularity_score'] = np.log1p(data['number of reviews']) * (0.8 + 0.2 * data['review rate number'] / 5)

# Premium factor (higher-end properties)
data['premium_factor'] = data['room_type_encoded'] * data['cancellation_encoded'] * (1 + 0.3 * data['host_verified_encoded'])

# Seasonal demand (based on availability)
data['demand_factor'] = 1 - np.tanh(data['availability 365'] / 180)

# Experience factor (host listings and construction year)
data['experience_factor'] = np.log1p(data['calculated host listings count']) * (1 + (data['Construction year'] - 2000) / 100)

# Minimum stay impact
data['stay_impact'] = np.tanh(data['minimum nights'] / 10)

# Service quality (based on reviews and service fee)
data['service_quality'] = (data['review rate number'] / 5) * np.sqrt(data['service fee'] / 100)

# Generate synthetic price with meaningful relationships
print("Generating synthetic price relationships...")

# Base price components with different weights to create varied correlations
data['price'] = (
    # Location factors (strong correlation ~0.5)
    2000 * data['location_score'] + 
    800 * data['neigh_encoded'] +
    
    # Room type and property factors (medium-strong correlation ~0.4-0.5)
    1500 * data['room_type_encoded'] +
    700 * data['room_type_encoded'] ** 2 +
    
    # Review and popularity factors (medium correlation ~0.3-0.4)
    300 * data['popularity_score'] +
    200 * data['review rate number'] +
    
    # Service and quality factors (medium correlation ~0.3)
    400 * data['service_quality'] +
    250 * np.log1p(data['service fee']) +
    
    # Booking flexibility factors (weak-medium correlation ~0.2-0.3)
    300 * data['bookable_encoded'] +
    200 * data['cancellation_encoded'] +
    
    # Stay duration impact (weak correlation ~0.15-0.25)
    150 * data['stay_impact'] +
    100 * np.log1p(data['minimum nights']) +
    
    # Host factors (weak correlation ~0.1-0.2)
    120 * data['host_verified_encoded'] +
    80 * np.log1p(data['calculated host listings count']) +
    
    # Availability and demand (weak correlation ~0.1-0.2)
    200 * data['demand_factor'] +
    
    # Nonlinear transformations and interactions
    500 * data['premium_factor'] * data['location_score'] +
    300 * data['popularity_score'] * data['service_quality'] +
    250 * data['location_score'] * np.log1p(data['service fee']) +
    180 * data['room_type_encoded'] * data['demand_factor'] +
    150 * np.sin(data['lat'] * 5) * np.cos(data['long'] * 5) * data['neigh_encoded'] +
    120 * data['experience_factor'] * data['premium_factor'] +
    
    # Add some noise for realism
    np.random.normal(0, 100, len(data))
)

# Ensure price is non-negative and clip outliers
data['price'] = data['price'].clip(lower=50)

# Rescale the price to a realistic range
min_price = 50
max_price = 1500

# Compute the current min, max of the 'price' column
current_min = data['price'].min()
current_max = data['price'].max()

# Rescale to the desired range
scaled_price = (data['price'] - current_min) / (current_max - current_min)  # Normalize to [0, 1]
data['price'] = scaled_price * (max_price - min_price) + min_price  # Scale to desired range

# Add some outliers for realism (about 2% of data)
outlier_indices = np.random.choice(len(data), size=int(len(data) * 0.02), replace=False)
data.loc[outlier_indices, 'price'] = data.loc[outlier_indices, 'price'] * np.random.uniform(1.5, 3, size=len(outlier_indices))

# Check correlations with price
print("\nCorrelations with price:")
numeric_cols = data.select_dtypes(include=[np.number]).columns
correlations = data[numeric_cols].corr()['price'].sort_values(ascending=False)
print(correlations)

# Restore original categorical columns
data['neighbourhood group'] = original_neigh
data['instant_bookable'] = original_bookable
data['room type'] = original_room_type
data['cancellation_policy'] = original_cancellation
data['host_identity_verified'] = original_host_verified

# Drop intermediate columns used for calculation
columns_to_drop = [
    'neigh_encoded', 'room_type_encoded', 'bookable_encoded', 
    'cancellation_encoded', 'host_verified_encoded',
    'location_score', 'popularity_score', 'premium_factor',
    'demand_factor', 'experience_factor', 'stay_impact',
    'service_quality'
]
data.drop(columns_to_drop, axis=1, inplace=True)

print("Generating synthetic data...")
# Generate synthetic data
metadata = Metadata.detect_from_dataframe(data)
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=280000)

print("Saving synthetic data...")
synthetic_data.to_csv('data/synthetic_airbnb_data.csv', index=False)
print("Done!")