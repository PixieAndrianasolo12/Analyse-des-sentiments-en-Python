from flask import Flask, request, render_template
import string
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def start_page():
    return render_template('/home.html')

@app.route('/about')
def about():
    return render_template('/about.html')

@app.route('/welcome')
def page_de_bienvenue():
    return render_template('/bienvenue.html')

@app.route('/analyze_text')
def formulaire_pour_texte():
    return render_template('form_text.html')

@app.route('/analyze_sentence')
def formulaire_pour_phrase():
    return render_template('form_phrase.html')

@app.route('/process_text', methods=['POST'])
def traiter_texte():
    text = request.form['comment']

    # converting to lowercase
    lower_case = text.lower()

    # Removing punctuations
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # splitting text into words
    tokenized_words = cleaned_text.split()

    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these",
                  "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                  "do",
                  "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                  "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                  "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                  "again",
                  "further", "then", "once", "here", "there", "when", "where", "why", " how", "all", "any", "both",
                  "each",
                  "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                  "than",
                  "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    # Removing stop words from the tokenized words list
    final_words = []
    for word in tokenized_words:
        if word not in stop_words:
            final_words.append(word)

    # NLP Emotion Algorithm
    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in final_words:
                emotion_list.append(emotion)

        # Calculate emotion percentages
    total_words = len(final_words)
    emotion_counts = Counter(emotion_list)
    emotion_percentages = {emotion: (count / total_words * 100) for emotion, count in emotion_counts.items()}

    # Pass the emotion percentages to the template


    # Plotting the emotions on the graph
    fig, ax1 = plt.subplots()
    ax1.bar(emotion_percentages.keys(), emotion_percentages.values())
    fig.autofmt_xdate()
    plt.savefig('static/graph.png')

    return render_template('form_text.html', emotion_percentages=emotion_percentages)


@app.route('/prozess_sentence', methods=['POST'])
def traiter_formulaire():
    texte_saisi = request.form['comment']  # Récupérer le texte du champ de texte
    # Convertir en minuscules et supprimer la ponctuation
    cleaned_text = texte_saisi.lower().translate(str.maketrans('', '', string.punctuation))
    # Imprimer texte_nettoye dans le terminal
    print("Texte nettoyé :", cleaned_text)

    # Définir les listes de mots vides pour l'anglais et le français
    stop_words_english = set(stopwords.words('english'))
    stop_words_french = set(stopwords.words('french'))

    # Utiliser la tokenisation pour séparer les mots
    tokenized_words = word_tokenize(cleaned_text)

    # Supprimer les mots vides en anglais et en français
    final_words = []
    for word in tokenized_words:
        if word not in stop_words_english and word not in stop_words_french:
            final_words.append(word)

    def lemmatize_words(word_list):
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(word) for word in word_list]
        return lemma_words

    # Lemmatization - From plural to single + Base form of a word (example better -> good)
    lemma_words = lemmatize_words(final_words)

    emotion_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)

    print(emotion_list)
    w = Counter(emotion_list)
    print(w)

    def sentiment_analyse(sentiment_text):
        score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
        if score['neg'] > score['pos']:
            print("Negative Sentiment")
            return "Negative Sentiment 😞"
        elif score['neg'] < score['pos']:
            print("Positive Sentiment")
            return "Positive Sentiment 😃"
        elif score['neg'] == score['pos']:
            print("Neutral Sentiment")
            return "Neutral Sentiment 😐"

    sentiment_result = sentiment_analyse(cleaned_text)

    # Créez le graphique et sauvegardez-le en tant qu'image
    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    plt.savefig('static/koto.png')  # Assurez-vous que le chemin est correct

    # Retournez la page HTML avec le résultat de l'analyse de sentiment
    return render_template('form_phrase.html', texte_saisi=texte_saisi, sentiment_result=sentiment_result)


if __name__ == '__main__':
    app.run()
