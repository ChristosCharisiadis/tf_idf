import itertools
from math import log
import operator
from nltk.stem.porter import *
from nltk.corpus import stopwords

def wordList(document):
    return re.sub(r'[.!,;?:]', ' ', document).split()

# Documents
document1 = 'A star schema model can be depicted as a simple star: a central table contains fact data and multiple tables radiate out from it, connected by the primary and foreign keys of the database. In a star schema implementation, Warehouse Builder stores the dimension data in a single table or view for all the dimension levels. For example, if you implement the Product dimension using a star schema, Warehouse Builder uses a single table to implement all the levels in the dimension, as shown in the screenshot. The attributes in all the levels are mapped to different columns in a single table called PRODUCT.'
document2 = 'The snowflake schema represents a dimensional model which is also composed of a central fact table and a set of constituent dimension tables which are further normalized into sub-dimension tables. In a snowflake schema implementation, Warehouse Builder uses more than one table or view to store the dimension data. Separate database tables or views store data pertaining to each level in the dimension. The screenshot displays the snowflake implementation of the Product dimension. Each level in the dimension is mapped to a different table.'
documentList = [document1, document2]

stemmer = PorterStemmer()
plurals = ['playing', 'plays', 'Played', 'plaything', 'Player']
singles = [stemmer.stem(plural) for plural in plurals]

# Transform documents into string lists, with removed stopwords and stemmed words
processedDocuments = []
for document in documentList:
    processedDocument = []
    for word in wordList(document):
        if word.lower() not in stopwords.words('english'):
            processedDocument.append(stemmer.stem(word))
    processedDocuments.append(processedDocument)

# Create a list of all terms, from the processed documents
terms = set(list(itertools.chain.from_iterable(processedDocuments)))

# Calculate Term Frequencies
tf = {}
i = 0
for document in processedDocuments:
    tfDict = {}
    for word in terms:
        tfDict[word] = document.count(word)
    tf[i] = tfDict
    i += 1

# Calculate Inverse Document Frequencies
idf = {}
N = len(processedDocuments)
for word in terms:
    n = 0
    for document in processedDocuments:
        if word in document:
            n += 1
    idf[word] = log(N / n, 10)  # Use a base 10 logarithm

# Calculate TF - TDF scores
tf_idf = {}
for word in terms:
    i = 0
    tf_idfDocs = {}
    for document in processedDocuments:
        tf_idfDocs[i] = tf[i][word] * idf [word]
        i += 1
    tf_idf[word] = tf_idfDocs

# Print the scores
print(tf_idf)
