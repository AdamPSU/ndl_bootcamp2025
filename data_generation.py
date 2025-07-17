import pandas as pd 
import numpy as np 
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

# Load data
data = pd.read_csv('data/Airbnb_Open_Data.csv')

# Backup original categorical columns
original_neigh = data['neighbourhood group'].copy()
original_bookable = data['instant_bookable'].copy()
original_room_type = data['room type'].copy()

# Step 1: Define mappings for categorical variables
neigh_price_map = {
    'Manhattan': 1.0, 'manhatan': 1.0,
    'Brooklyn': 0.8, 'brookln': 0.8,
    'Queens': 0.6, 'Bronx': 0.5, 'Staten Island': 0.4
}
bookable_price_map = {True: 0.6, False: 0.4}
room_type_map = {
    'Entire home/apt': 1.0, 'Hotel room': 0.9,
    'Private room': 0.6, 'Shared room': 0.3
}

# Step 2: Apply mappings to temporary columns
data['neigh_score'] = data['neighbourhood group'].map(neigh_price_map).fillna(0.5)
data['bookable_score'] = data['instant_bookable'].map(bookable_price_map).fillna(0.5)
data['room_type_score'] = data['room type'].map(room_type_map).fillna(0.5)
data['service_fee_float'] = data['service fee'].str.replace('$', '', regex=False).astype(float)

import numpy as np
import pandas as pd

# 1. Mappings for score encoding
neigh_price_map = {
    'Manhattan': 1.0, 'manhatan': 1.0,
    'Brooklyn': 0.8, 'brookln': 0.8,
    'Queens': 0.6, 'Bronx': 0.5, 'Staten Island': 0.4
}
room_type_map = {
    'Entire home/apt': 1.0, 'Hotel room': 0.9,
    'Private room': 0.6, 'Shared room': 0.3
}
bookable_price_map = {True: 0.6, False: 0.4}

# 2. Encode from original categorical columns
data['neigh_encoded'] = data['neighbourhood group'].map(neigh_price_map).fillna(0.5)
data['room_type_encoded'] = data['room type'].map(room_type_map).fillna(0.5)
data['bookable_encoded'] = data['instant_bookable'].map(bookable_price_map).fillna(0.5)

# 3. Ensure service_fee is numeric
data['service_fee_float'] = (
    data['service fee']
    .str.replace('$', '', regex=False)
    .astype(float)
    .fillna(0)
)

data['price'] = (
    10 * data['number of reviews'] +  # Increased weight for 'number of reviews'
    20 * data['lat'] +                 # Simple linear relationship with lat
    20 * data['long'] +                # Simple linear relationship with long
    150 * data['room_type_encoded'] +  # Room type encoded effect
    25 * data['room_type_encoded'] ** 2 +  # Polynomial effect of room type encoding
    30 * np.sqrt(data['number of reviews']) +  # Square root transformation for reviews
    25 * data['neigh_encoded'] * data['service_fee_float'] +  # Interaction term
    3700 * data['bookable_encoded'] * np.log1p(data['service_fee_float']) +  # Interaction term
    100 * np.log1p(data['reviews per month']) +  # Log transformation on reviews per month
    100 * data['reviews per month'] ** 2 +  # Polynomial transformation
    70 * np.exp(data['service_fee_float'] / 100) +  # Exponential transformation on service fees
    3000 * np.log1p(data['minimum nights']) +  # Log transformation for minimum nights
    35 * np.log1p(np.sqrt(data['number of reviews'])) +  # Log-sqrt transformation
    30 * data['number of reviews'] * data['room_type_encoded'] +  # Interaction between reviews and room type
    20 * data['lat'] * data['room_type_encoded'] +  # Interaction between lat and room type
    50 * data['long'] * np.log1p(data['reviews per month']) +  # Interaction between long and reviews
    -2 * data['availability 365'] * np.log1p(data['service_fee_float']) +  # Interaction between availability and service fees
    0.5 * data['Construction year'] * np.sqrt(data['service_fee_float']) +  # Interaction between construction year and service fees
    10 * data['lat'] ** 2 +  # Square of latitude (nonlinear)
    50 * data['Construction year'] +
    5 * np.sqrt(data['number of reviews'] + 1) +  # Sqrt transformation with offset for reviews
    70 * np.exp(np.sqrt(data['reviews per month'] + 1)) +  # Exponential on sqrt transformation
    170 * data['bookable_encoded'] * data['neigh_encoded'] +  # Interaction between bookable and neighborhood encoding
    30 * np.log1p(data['number of reviews'] + 1) * data['room_type_encoded'] +  # Interaction between reviews and room type
    800 * data['review rate number'] ** 2 + 
    np.random.normal(0, 20, len(data))  # Mild noise added
)

data['price'] = data['price'].clip(lower=0)  # Ensure price is non-negative

# Rescale the 'price' column
min_price = 150
max_price = 73000

# Compute the current min, max, and mean of the 'price' column
current_min = data['price'].min()
current_max = data['price'].max()
current_mean = data['price'].mean()

# Step 1: Rescale to the desired range (150 to 27000)
scaled_price = (data['price'] - current_min) / (current_max - current_min)  # Normalize to [0, 1]
rescaled_price = scaled_price * (max_price - min_price) + min_price  # Scale to desired range

# Step 2: Adjust to ensure the mean is approximately 250
# Compute the current mean of the rescaled prices
adjusted_price = rescaled_price - rescaled_price.mean() + 250  # Adjust mean to 250

# Set the adjusted price back into the dataframe
data['price'] = adjusted_price.clip(lower=min_price, upper=max_price)  # Ensure the final range is [150, 27000]

# Step 6: Restore original categorical columns
data['neighbourhood group'] = original_neigh
data['instant_bookable'] = original_bookable
data['room type'] = original_room_type

# Step 7: Drop intermediate numeric columns used for calculation
data.drop([
    'neigh_score', 
    'bookable_score', 
    'room_type_score', 
    # 'service_fee_float', 
], axis=1, inplace=True)

metadata = Metadata.detect_from_dataframe(data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=280_000)

synthetic_data.to_csv('data/synthetic_airbnb_data.csv', index=False)

