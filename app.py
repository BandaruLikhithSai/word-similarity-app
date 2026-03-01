# app.py - Main backend for Word Similarity Finder
# Uses Flask for the web server and gensim for word embeddings

from flask import Flask, render_template, request, jsonify
import gensim.downloader as api
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# Load a pre-trained Word2Vec model (Google News vectors)
# 'glove-wiki-gigaword-50' is smaller and faster to download
# Switch to 'word2vec-google-news-300' for better accuracy
# ---------------------------------------------------------
print("Loading pre-trained word embedding model... (this may take a moment)")
model = api.load("glove-wiki-gigaword-50")
print("Model loaded successfully!")


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/find_similar", methods=["POST"])
def find_similar():
    """
    Receive a word from the frontend,
    find the top 5 most similar words using the pre-trained model,
    and return them as JSON.
    """
    data = request.get_json()
    word = data.get("word", "").strip().lower()

    # Check if the word exists in the model's vocabulary
    if word not in model:
        return jsonify({"error": f'"{word}" was not found in the vocabulary.'})

    # Use the model's built-in most_similar method (uses cosine similarity internally)
    similar_words = model.most_similar(word, topn=5)

    # Format results: list of { word, score }
    results = [
        {"word": w, "score": round(float(score), 4)}
        for w, score in similar_words
    ]

    return jsonify({"word": word, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
