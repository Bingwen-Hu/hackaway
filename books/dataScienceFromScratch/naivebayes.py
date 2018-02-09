"""
naive bayes is a scalable method usually used in text data classification

bayes formula: 
P(B|A) = P(A|B)P(A) / [P(A|B)P(B) + P(A|~B)P(~B)]

what's naive means? It assumes that all the event is independent.
say: P(X1=x1, X2=x2,..., Xn=xn | S) = P(X1=x1|S)P(X2=x2|S)...P(Xn=xn|S)
this make it possible to compute the P(A|B) as P(Ai|B) mul for i = 1 to n
Note that to avoid underfitting for the small float number, we compute the log.
how to compute P(Ai|B)?
it says: for all the B, what's the fraction of A happen?
concreted: let B means the email is spam, ~B donate a normal email, Ai means a 
word wi appear in the email. P(Ai|B) means if an email is spam, the possibility 
of the word wi appear?
easy. we have labeled data. Just compute the fraction of the word appear in spam
email to estimated P(Ai|B)
However, what if the word wi NEVER appear in the spam emails? It will be 
assigned a ZERO probability. It means if the word wi do not appear then the 
email is not a spam. Obviously, this causes problem.
So, we smooth it as:
P(Ai|B) = (k + number of spams containing wi) | (2k + number of spams)
P(A|~B) is similar. P(A) and P(B) is always easily estimated. In this example, 
it means the probability of whether an email is spam or not, we can simply give
both 0.5
"""

import re
import numpy as np
from collections import defaultdict

def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9]", message)
    return set(all_words)

def count_words(training_set):
    """trainiing set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda:[0, 0])
    for message, is_spam in trainiing_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    "turn the word_counts into a list of triplets w, p(w|span), p(w|~spam)"
    return [(w,
             (spam+k) / (total_spams + 2*k),
             (non_spam+k) / (total_non_spams + 2*k))
            for w, (spam, non_spam) in counts.iteritems()]

# Note that here we just set P(B) = P(~B) = 0.5 to simpify 
def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_non_spam in word_probs:
        # if word appears in the message
        # add the natural log probability of seeing it
        if word in message_words:
            log_prob_if_spam += np.log(prob_if_spam)
            log_prob_if_not_spam += np.log(prob_if_non_spam)
        # if word doesn't appear in the message
        # add the log probability of NOT seeing it
        else:
            log_prob_if_spam += np.log(1 - prob_if_spam)
            log_prob_if_not_spam += np.log(1 - prob_if_non_spam)
    prob_if_spam = np.exp(log_prob_if_spam)
    prob_if_not_spam = np.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, trainiing_set):
        num_spams = len([is_spam
                         for message, is_spam in trainiing_set
                         if is_spam])
        num_non_spams = len(trainiing_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(trainiing_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)
    def classify(self, message):
        return spam_probability(self.word_probs, message)
    


"""
why is the function spam_probability we want to compute the probability when 
the word in our training_set doesn't appear in the input message?
My answer is: this is a reasonable way to make good use of our data
what if all the word in our training set do not appear in the message? Yes, we 
can compute the probability if the word is NOT appear! It matters!
"""
