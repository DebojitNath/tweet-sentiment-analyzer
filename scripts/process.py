import re
from nltk.corpus import stopwords
import pandas as pd

df=pd.read_csv("data/tweets.csv",encoding='latin-1', header = None)
df.columns = ['sentiment', 'tweet_id', 'date', 'query', 'user', 'text']
df = df[['text', 'sentiment']]
df['sentiment'] = df['sentiment'].replace(4,1)

stop_words = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does','did',
    'doing','a','an','the','and','but','if','or','because','as','until','while',
    'of','at','by','for','with','about','against','between','into','through','during',
    'before','after','above','below','to','from','up','down','in','out','on','off',
    'over','under','again','further','then','once','here','there','when','where',
    'why','how','all','any','both','each','few','more','most','other','some','such',
    'no','nor','not','only','own','same','so','than','too','very','s','t','can',
    'will','just','don','should','now'
])

#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))

contractions = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would"
}


def expand_contractions(text):
    for word, expanded in contractions.items():
        text = re.sub(r"\b" + word + r"\b", expanded, text)
    return text


def clean_tweet(text):
    text = text.lower()
    text = expand_contractions(text) 
    
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r'\@\w+', '', text)                    
    text = re.sub(r'#(\w+)', r'\1', text)                
    text = re.sub(r'[^a-z\s]', '', text)                 
    
   
    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(word)

    return ' '.join(words)



def process_dataframe(df):
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    return df


