import json
import sys
import io
import csv
from transformers import pipeline
API_TOKEN = ''

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-emotion",
    top_k=None,  # Get all emotions
    truncation=True,
    max_length=512
)

with open('input.json', 'r', encoding='utf-8') as f:
    movie_data = json.load(f)

texts = [movie['Reviews'] for movie in movie_data]
results = classifier(texts)

output_data = []
for movie, result in zip(movie_data, results):
    output_data.append({
        'emotions': json.dumps(result), 
        'genres': movie['genres'],
        'name': movie['movie_name']
    })

with open('output.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['emotions', 'genres', 'name']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_data)

print("CSV file 'output.csv' has been created successfully.")
