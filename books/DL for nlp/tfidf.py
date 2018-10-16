# other demo see memos/datamining/
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "Ramiess sings classic songs",
    "he listens to old pop ",
    "and rock music", 
    " and also listens to classical songs",
]

cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())

import math
# tfidf
# suppose in one paragraph contains 100 words, a word appear 5 times
tf = 5 / 100
# suppose we have 1,000,000 paragraph, and number of paragraph that
# contains the word is 100, then
idf = math.log10(1000000/100)
tfidf = tf * idf 
print(tfidf)