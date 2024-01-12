import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

text = open('read.txt', encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Using word_tokenize because it's faster than split()
tokenized_words = word_tokenize(cleaned_text, "english")

# Removing Stop Words
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Lemmatization - From plural to single + Base form of a word (example better-> good)
lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

# NLP Emotion Algorithm
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)

# Calculate emotion percentages
total_words = len(lemma_words)
emotion_counts = Counter(emotion_list)
emotion_percentages = {emotion: (count / total_words * 100) for emotion, count in emotion_counts.items()}

print("Pourcentages des émotions :")
for emotion, percentage in emotion_percentages.items():
    print(f"{emotion}: {percentage:.2f}%")

# Sentiment Analysis using VADER
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Sentiment Négatif")
    elif score['neg'] < score['pos']:
        print("Sentiment Positif")
    else:
        print("Sentiment Neutre")

sentiment_analyse(cleaned_text)

# Plotting the emotions on the graph
fig, ax1 = plt.subplots()
ax1.bar(emotion_percentages.keys(), emotion_percentages.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
