from sklearn.feature_extraction.text import CountVectorizer # importing the countvectorizer

def get_top_n_words(corpus, n=None): #method to check top scoring words for issue extraction
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


data = [line.replace("\n", "") for line in open('D://ZZ Ashray(698306)//remedytickets//dataset.csv')] #reading the file
common_words = get_top_n_words(data, 20) #selecting the number of common words required
for word, freq in common_words:
    print(word, freq) #printing the word and it's frequency