# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('words')

import requests
import gzip
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from collections import defaultdict
from io import BytesIO
import json
from tqdm import tqdm

api_url = 'https://data.commoncrawl.org/'
wet_filename = 'wet.paths.gz'

lemmatizer = WordNetLemmatizer()
token_counts = defaultdict(int)
english_words = set(words.words())

# Open the .gz file and extract paths
with gzip.open(wet_filename, 'rt') as f:
    paths = [line.strip() for line in f]

try:
    with open('token_counts.json', 'r') as f:
        token_counts = defaultdict(int, json.load(f))
except FileNotFoundError:
    token_counts = defaultdict(int)

for i, path in enumerate(tqdm(paths)):
    response = requests.get(api_url + path)  # download WET file

    with gzip.open(BytesIO(response.content), 'rt', encoding='utf-8') as f:
        text = [
            line.strip() 
            for line in f 
            if not (':' in line and not line.split(':')[0] in english_words)
        ]
        raw_text = ' '.join(text).lower().strip()

        tokens = word_tokenize(raw_text)
        for token in tokens:
            if not token in english_words:
                continue
            lemmatized_token = lemmatizer.lemmatize(token, pos='v')
            token_counts[lemmatized_token] += 1
    
    if i == 23:
        break

# save token counts to json file
with open('token_counts.json', 'w') as f:
    json.dump(token_counts, f)