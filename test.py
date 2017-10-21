from sklearn.feature_extraction.text import CountVectorizer

text = ['on april first i need a ticket from tacoma to san jose departing before 7 am']
text[0] = text[0].split()
a = [x for x in text[0] if ('on' not in x)]
a = ' '.join(a)
print(a)
