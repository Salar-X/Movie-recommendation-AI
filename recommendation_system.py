import requests
import time
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import Model
from keras.layers import Input, Dense
from concurrent.futures import ThreadPoolExecutor
import json

# TMDB API configuration
API_KEY = '5e0672d607b3b41c0cf51a6583bf8440'  # Replace with your TMDB API key
BASE_URL = 'https://api.themoviedb.org/3'

# Create data directory if it doesn’t exist
if not os.path.exists('data'):
    os.makedirs('data')

# Fetch popular movies from TMDB
def get_popular_movies(page: int = 1) -> list:
    url = f'{BASE_URL}/movie/popular?api_key={API_KEY}&page={page}'
    response = requests.get(url)
    return response.json()['results']

# Fetch detailed movie info with rate limiting
def get_movie_details(movie_id: int) -> dict:
    url = f'{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&append_to_response=keywords'
    response = requests.get(url)
    time.sleep(0.25)  # Respect TMDB’s 40 requests/10s limit
    return response.json()

# Step 1: Fetch movies concurrently
print("🎥 Fetching movie data from TMDB...")
movies = []
num_pages = 10  # Fetch 200 movies (20 per page)
for page in range(1, num_pages + 1):
    movies.extend(get_popular_movies(page))
movie_ids = [movie['id'] for movie in movies]

# Step 2: Fetch details with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    movie_data = list(executor.map(get_movie_details, movie_ids))

# Step 3: Save raw data
with open('data/tmdb_movies.json', 'w') as f:
    json.dump(movie_data, f)
print("💾 Movie data saved to 'data/tmdb_movies.json'.")

# Step 4: Process into DataFrame
df = pd.DataFrame(movie_data)
df = df[['id', 'title', 'genres', 'keywords']]
df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in x])
df['keywords'] = df['keywords'].apply(lambda x: [k['name'] for k in x['keywords']])

# Step 5: Extract unique genres
all_genres = set().union(*df['genres'])
all_genres = list(all_genres)

# Step 6: Get top 50 keywords
all_keywords = sum(df['keywords'], [])
top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(50)]

# Step 7: Multi-hot encode features
for genre in all_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)
for keyword in top_keywords:
    df[keyword] = df['keywords'].apply(lambda x: 1 if keyword in x else 0)

# Step 8: Create feature matrix
feature_columns = all_genres + top_keywords
X = df[feature_columns].values
print(f"🧮 Feature matrix shape: {X.shape}")

# Step 9: Define and train autoencoder
input_dim = X.shape[1]
encoding_dim = 50
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
print("🤖 Training autoencoder...")
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, verbose=1)

# Step 10: Extract embeddings
encoder = Model(input_layer, encoded)
movie_embeddings = encoder.predict(X)
print(f"🔍 Movie embeddings shape: {movie_embeddings.shape}")

# Step 11: Save embeddings
np.save('data/movie_embeddings.npy', movie_embeddings)
print("💾 Embeddings saved to 'data/movie_embeddings.npy'.")

# Step 12: Test recommendations
title_to_index = {title: idx for idx, title in enumerate(df['title'])}

def recommend_movies(title: str, n: int = 5) -> list:
    if title not in title_to_index:
        return [f"Movie '{title}' not found."]
    idx = title_to_index[title]
    embedding = movie_embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(embedding, movie_embeddings)[0]
    similar_indices = similarities.argsort()[-n-1:-1][::-1]
    return df['title'].iloc[similar_indices].tolist()

# Example output
example_title = df['title'].iloc[0]
recs = recommend_movies(example_title)
print(f"\n🎬 Recommendations for '{example_title}':")
for i, movie in enumerate(recs, 1):
    print(f"{i}. {movie}")