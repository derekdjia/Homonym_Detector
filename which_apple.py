"""
Objectives:
1. Think about how to featurize text
2. Make a dumb count vector
3. Make a text processing pipeline
4. Split documents into words or "tokens"
5. Drop "filler" or "stop" words
6. Drop punctuation, capitalization
7. Make a word frequency vector
8. Make a word frequency / document frequency (TF-IDF) vector
9. Compare documents with cosine similarity
"""

## NB: Capitalization is important in distinguishing between Apple company and fruit
## NB: Numbers are more likely for the company, and potentially the year is important in distinguishing

from collections import Counter, defaultdict
import nltk, re, string
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

punct = set(string.punctuation)

company_corpus = []
company_word_dict = {}


def remove_symbols(word, symbol_set):
    return ''.join(char for char in word if char not in symbol_set)

with open('apple-computers.txt', 'r') as company:
    for line in company:
        line = (line.rstrip('\r\n\t'))
        line.replace('\t', ' ')
        line = re.sub("([\(\[]).*?([\)\]])", "", line)
        line = remove_symbols(line, punct)
        if line: company_corpus.append(line)
            
tokenized_company = [doc.split() for doc in company_corpus]
tokenized_company_lowered = [[word.lower() if word.lower()!= 'apple' else word for word in doc] for doc in tokenized_company]
cleaned_company = [[remove_symbols(word, punct) for word in doc] 
                for doc in tokenized_company_lowered]


# stop_words = set(nltk.corpus.stopwords.words('english'))

stop_words = {'does', 'those', 'be', 'nor', 'if', 'a', 'an', 'below', 'did', 'are', 'about', "it's", "doesn't", "you're", 'of', 'yourselves', 'between', 'but', 'don', 'her', 'themselves', 'and', 'against', 'because', 'm', 'off', 'out', 'when', 'had', 'mustn', 'again', 'or', "should've", 'how', 'more', 'as', 'while', 'up', 'these', 'any', 'can', "you'd", 'under', 'couldn', 'we', 'whom', 'doesn', 'during', 'itself', 'for', 'myself', 'they', 'both', 'should', 'what', "needn't", 'he', 'have', 'where', 'your', 'him', 'to', 'aren', "weren't", 'hers', 'me', 'why', 'all', "didn't", 'down', 'their', 'mightn', 'i', "shouldn't", 've', 'by', 'only', 'do', 'being', "hadn't", "hasn't", 'too', 'here', 'ourselves', 'the', 'than', 'which', 'isn', 't', 'didn', 'wasn', 'having', 'has', "she's", 'on', 'now', 'haven', 'there', 'once', 'through', "won't", 'his', 'my', 'own', 'were', 'after', 'y', "isn't", 'each', "wouldn't", 'most', 'she', 'just', 'is', 'yours', 'some', 'o', 'ma', 'will', 'who', 'll', 'herself', 'above', 'further', "shan't", 'shouldn', "you'll", 'needn', 'that', 'am', 'until', 'this', 's', 'wouldn', 'same', "you've", 'yourself', "wasn't", 'its', 'you', 'no', 'ain', 'ours', "haven't", 'such', 'doing', "mightn't", "aren't", 'other', 'been', 'then', 'at', 'hasn', 'so', 'over', "that'll", 'shan', 'won', 'into', 'from', 'hadn', 'it', 'with', 'theirs', 'not', 'was', 'himself', 'them', "don't", 'weren', 're', 'd', 'before', 'few', 'our', "couldn't", 'very', "mustn't", 'in'}


docs_no_stops_company = [[word for word in doc if word not in stop_words] 
                 for doc in cleaned_company]

stemmer = SnowballStemmer('english')
docs_stemmed_company = [[stemmer.stem(word) for word in doc] for doc in docs_no_stops_company]

flat_company = [item for sublist in docs_stemmed_company for item in sublist]

company_dict = Counter(flat_company)

print(pd.DataFrame.from_dict(company_dict))
