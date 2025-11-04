from model1 import convert_single_text_to_emotion # (model1에 이런 함수가 있다고 가정)
from model2 import MLP, relu, deriv_relu
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # (영화 DB 로드를 위해)
import numpy as np

# --- 1. (핵심) 사전 학습된 Model 2 로드 ---
# (학습을 건너뛰고, 저장된 가중치만 불러옴)
print("사전 학습된 모델 로딩 중...")
Model2_recommender = MLP(
    batch_size=32, input_dim=4, hidden1_dim=64, hidden2_dim=32, output_dim=20,
    activation_fn=relu, activation_deriv=deriv_relu
)
Model2_recommender.load_weights("my_trained_model.pkl")

# --- 2. 영화 DB 로드 ---
# model1의 결과로 나온 csv 파일 참조 ex) movie_db_for_cosine.csv
movie_db = pd.read_csv("movie_db_for_cosine.csv")
db_vectors = np.array(movie_db['genre_vector'].apply(eval).tolist())


# --- 3. 사용자 일기 입력 및 추천 ---
def get_recommendation(diary_text):
    print(f"\n입력된 일기: {diary_text[:20]}...")

    # 3-1. Model 1 (일기 1개만 변환) -> (빠름)
    diary_emotion_vector = convert_single_text_to_emotion(diary_text) # (1, 4) 배열
    
    # 3-2. Model 2 (예측) -> (빠름)
    ideal_genre_vector = Model2_recommender.predict(diary_emotion_vector)
    
    # 3-3. 코사인 유사도 (추천) -> (빠름)
    similarities = cosine_similarity(ideal_genre_vector, db_vectors)
    movie_db['similarity'] = similarities[0]
    
    recommendations = movie_db.sort_values(by='similarity', ascending=False)
    
    print("--- 추천 결과 ---")
    print(recommendations[['movie_name', 'similarity']].head(3).to_string())


if __name__ == "__main__":
    get_recommendation("Today was a sad and lonely day...")