from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load precomputed data
with open('data/tmdb_movies.json', 'r') as f:
    movie_data = json.load(f)
df = pd.DataFrame(movie_data)[['title']]
movie_embeddings = np.load('data/movie_embeddings.npy')
title_to_index = {title: idx for idx, title in enumerate(df['title'])}

def recommend_movies(title: str, n: int = 5) -> list:
    """Generate movie recommendations based on cosine similarity."""
    if title not in title_to_index:
        return ["Movie not found."]
    idx = title_to_index[title]
    embedding = movie_embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(embedding, movie_embeddings)[0]
    similar_indices = similarities.argsort()[-n-1:-1][::-1]
    return df['title'].iloc[similar_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle web requests and render the recommendation page."""
    if request.method == 'POST':
        title = request.form['title']
        recommendations = recommend_movies(title)
        return render_template('index.html', recommendations=recommendations, input_title=title)
    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)