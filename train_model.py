import pandas as pd
import numpy as np
import ast
from model1 import convert_texts_to_emotions # (model1에 이런 함수가 있다고 가정)
from model2 import MLP, ReLU, deriv_ReLU # (우리가 만든 model2)

# --- Model 1 => COLAB 사용 권장  ---
print("Model 1 (Hugging Face) 실행 중...")
df_raw = pd.read_csv("Movies_Reviews_modified_version1.csv")
review_texts = df_raw['Reviews'].tolist()
emotion_vectors = convert_texts_to_emotions(review_texts) # (N, 4) 배열
# (이하 output.csv를 만드는 전처리 과정)
# (X_train, Y_train, movie_db를 완성했다고 가정)
# ...
# X_train = (10662, 4) 감정 벡터
# Y_train = (10662, 20) 장르 벡터
# ...
X_train = np.random.rand(1000, 4) # (시뮬레이션)
Y_train = np.random.rand(1000, 20) # (시뮬레이션)
print("Model 1 데이터 생성 완료.")


# --- 2. Model 2 학습 ---
print("Model 2 (MLP) 학습 시작...")
model2 = MLP(
    batch_size=32, input_dim=4, hidden1_dim=64, hidden2_dim=32, output_dim=20,
    activation_fn=ReLU, activation_deriv=deriv_ReLU,
    init_method='He', learning_rate=0.001
)
model2.train(X_train, Y_train, epochs=1000)

# --- 3. (핵심) 학습된 모델 저장 ---
model2.save_weights("my_trained_model.pkl")
print("사전 학습 완료.")