import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load smaller datasets
movies = pd.read_csv('dataset/movie.csv').sample(n=2000, random_state=42)  # Randomly sample 1000 movies for testing
ratings = pd.read_csv('dataset/rating.csv').query("userId <= 100")  # Filter user ratings for fewer users

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_movie_description(description):
    if pd.isna(description):
        return None
    try:
        inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
        return inputs
    except Exception as e:
        print(f"Error tokenizing: {description}, error: {str(e)}")
        return None

movies['tokenized_description'] = movies['genres'].apply(tokenize_movie_description)

# Profile Generation
def generate_user_profile(user_movie_history):
    embeddings = []
    for _, movie in user_movie_history.iterrows():
        tokenized_description = movie.get('tokenized_description')
        if tokenized_description is not None:
            try:
                input_ids = tokenized_description['input_ids'].to(torch.device('cpu'))
                attention_mask = tokenized_description['attention_mask'].to(torch.device('cpu'))
                with torch.no_grad():
                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                    movie_embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(movie_embedding)
            except Exception as e:
                print(f"Error generating embedding for {movie['title']}: {e}")
        else:
            print(f"No tokenized description for {movie['title']}")
    
    if embeddings:
        return torch.mean(torch.stack(embeddings), dim=0)
    else:
        print("No embeddings found.")
        return None

# Recommendation Model
def recommend_movies(user_profile, candidate_movies):
    recommendations = []
    for _, movie in candidate_movies.iterrows():
        tokenized_description = movie.get('tokenized_description')
        if tokenized_description is not None:
            try:
                input_ids = tokenized_description['input_ids'].to(torch.device('cpu'))
                attention_mask = tokenized_description['attention_mask'].to(torch.device('cpu'))
                with torch.no_grad():
                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                    movie_embedding = outputs.last_hidden_state.mean(dim=1)
                similarity = torch.cosine_similarity(user_profile, movie_embedding, dim=1)
                recommendations.append((movie['title'], similarity.item()))
            except Exception as e:
                print(f"Error generating embedding for {movie['title']}: {e}")
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

# Preference-based Feedback Simulation
def feedback_loop(user_profile, candidate_movies, user_movie_history):
    # Calculate genre-based user preferences
    preferred_genres = user_movie_history['genres'].str.split('|').explode().value_counts(normalize=True)
    print(f"User preferred genres: {preferred_genres.to_dict()}")

    adjusted_recommendations = []
    for movie_title, similarity in candidate_movies:
        movie_genres = movies[movies['title'] == movie_title]['genres'].values[0]
        
        # Calculate a preference score based on shared genres with user's watched movies
        genre_match_score = sum(preferred_genres.get(genre, 0) for genre in movie_genres.split('|'))
        adjusted_similarity = similarity * (1 + genre_match_score)  # Boost based on genre match
        
        print(f"Adjusting similarity for '{movie_title}' with genre match score {genre_match_score:.2f} -> {adjusted_similarity:.4f}")
        adjusted_recommendations.append((movie_title, adjusted_similarity))
    
    # Sort by adjusted similarity in descending order
    return sorted(adjusted_recommendations, key=lambda x: x[1], reverse=True)

# Main pipeline with inferred feedback
def run_pipeline(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_ratings = user_ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    user_movie_history = movies[movies['movieId'].isin(user_ratings['movieId'])]
    candidate_movies = movies[~movies['movieId'].isin(user_movie_history['movieId'])][['title', 'tokenized_description']]
    
    print(f"\n--- Generating Profile for User {user_id} ---")
    user_profile = generate_user_profile(user_movie_history)
    if user_profile is None:
        return
    
    print(f"\n--- Generating Recommendations for User {user_id} ---")
    recommendations = recommend_movies(user_profile, candidate_movies)
    print("Initial Recommendations:", recommendations)

    print("\n--- Adjusting Recommendations Based on Inferred Feedback ---")
    adjusted_recommendations = feedback_loop(user_profile, recommendations, user_movie_history)
    print("Adjusted Recommendations:", adjusted_recommendations)

# Run the system for a sample user
run_pipeline(user_id=1)