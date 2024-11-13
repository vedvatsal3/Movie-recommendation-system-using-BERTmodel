import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import torch
from transformers import BertTokenizer, BertModel

# Load smaller datasets
movies = pd.read_csv('dataset/movie.csv').sample(n=2000, random_state=42)  # Randomly sample 1000 movies for testing
ratings = pd.read_csv('dataset/rating.csv').query("userId <= 100")  # Filter user ratings for fewer users


# Step 1: BERT-based Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def bert_embedding(description):
    """Generate BERT embeddings for movie descriptions."""
    inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Generate embeddings for movie genres
movies['embedding'] = movies['genres'].apply(lambda x: bert_embedding(x) if isinstance(x, str) else np.zeros(768))

def recommend_bert(user_profile, candidate_movies):
    recommendations = []
    for _, movie in candidate_movies.iterrows():
        embedding = movie['embedding']
        similarity = cosine_similarity(user_profile.reshape(1, -1), embedding.reshape(1, -1)).flatten()[0]
        recommendations.append((movie['title'], similarity))
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

# Step 2: Collaborative Filtering Model
def collaborative_filtering(user_ratings, candidate_movies, ratings):
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    movie_similarity = cosine_similarity(user_item_matrix.T)
    
    # Align movie IDs with their actual positions in the similarity matrix
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}
    
    recommendations = []
    user_idx = user_ratings['userId'].iloc[0]
    user_ratings_series = user_item_matrix.loc[user_idx]
    
    for movie_id in candidate_movies['movieId']:
        if movie_id in movie_id_to_index:
            # Get similarity scores for the movie
            sim_scores = movie_similarity[movie_id_to_index[movie_id]]
            # Calculate weighted score
            weighted_score = np.dot(user_ratings_series, sim_scores) / np.sum(sim_scores)
            title = candidate_movies[candidate_movies['movieId'] == movie_id]['title'].values[0]
            recommendations.append((title, weighted_score))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]


# Step 3: Matrix Factorization Model (SVD)
def matrix_factorization(user_ratings, candidate_movies, ratings):
    # User-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=50)
    svd.fit(user_item_matrix)

    # Project users and items to latent space
    user_factors = svd.transform(user_item_matrix)
    item_factors = svd.components_.T
    recommendations = []

    # Map movie IDs to SVD matrix indices
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}
    
    # Get the user factors for the target user
    user_idx = user_ratings['userId'].iloc[0]
    user_factor = user_factors[user_idx]
    for _, movie in candidate_movies.iterrows():
        movie_id = movie['movieId']
        if movie_id in movie_id_to_index:  # Ensure the movie ID is in item_factors
            item_index = movie_id_to_index[movie_id]
            item_factor = item_factors[item_index]
            similarity = cosine_similarity(user_factor.reshape(1, -1), item_factor.reshape(1, -1)).flatten()[0]
            recommendations.append((movie['title'], similarity))
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

# Evaluation
def evaluate_model(user_ratings, recommendations, threshold=0.7):
    actual_labels = [1 if rating >= 4 else 0 for rating in user_ratings['rating']]
    predicted_labels = [1 if sim >= threshold else 0 for _, sim in recommendations]
    min_length = min(len(actual_labels), len(predicted_labels))
    actual_labels, predicted_labels = actual_labels[:min_length], predicted_labels[:min_length]
    cm = confusion_matrix(actual_labels, predicted_labels)
    report = classification_report(actual_labels, predicted_labels, target_names=['Not Recommended', 'Recommended'])
    return cm, report

# Running the pipeline for a sample user
user_id = 1
user_ratings = ratings[ratings['userId'] == user_id]
candidate_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]

# Generate user profile for BERT
user_profile_embedding = np.mean(np.array([embedding for embedding in movies.loc[movies['movieId'].isin(user_ratings['movieId']), 'embedding']]), axis=0)

# BERT Recommendations
bert_recommendations = recommend_bert(user_profile_embedding, candidate_movies)
cm_bert, report_bert = evaluate_model(user_ratings, bert_recommendations)

# Collaborative Filtering Recommendations
collab_recommendations = collaborative_filtering(user_ratings, candidate_movies, ratings)
cm_collab, report_collab = evaluate_model(user_ratings, collab_recommendations)

# Matrix Factorization Recommendations
mf_recommendations = matrix_factorization(user_ratings, candidate_movies, ratings)
cm_mf, report_mf = evaluate_model(user_ratings, mf_recommendations)

# Display results
print("BERT Confusion Matrix:\n", cm_bert)
print("BERT Classification Report:\n", report_bert)

print("Collaborative Filtering Confusion Matrix:\n", cm_collab)
print("Collaborative Filtering Classification Report:\n", report_collab)

print("Matrix Factorization Confusion Matrix:\n", cm_mf)
print("Matrix Factorization Classification Report:\n", report_mf)