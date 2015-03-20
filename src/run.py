from nltk import word_tokenize
from nltk.corpus import reuters 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text));
	words = [word for word in words if word not in cachedStopWords]
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
	return filtered_tokens

# Return the representer, without transforming
def tf_idf(docs):	
	tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90, max_features=1000, use_idf=True, sublinear_tf=True);
	tfidf.fit(docs);
	return tfidf;

def feature_values(doc, representer):
	doc_representation = representer.transform([doc])
	features = representer.get_feature_names()
	return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

def collection_stats():
	# List of documents
	documents = reuters.fileids()
	print(str(len(documents)) + " documents");
	
	train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
	print(str(len(train_docs)) + " total train documents");
	
	test_docs = list(filter(lambda doc: doc.startswith("test"), documents));	
	print(str(len(test_docs)) + " total test documents");

	# List of categories 
	categories = reuters.categories();
	print(str(len(categories)) + " categories");

	# Documents in a category
	category_docs = reuters.fileids("acq");

	# Words for a document
	document_id = category_docs[0]
	document_words = reuters.words(category_docs[0]);
	print(document_words);	

	# Raw document
	print(reuters.raw(document_id));

def main():
	train_docs = []
	test_docs = []

	for doc_id in reuters.fileids():
		if doc_id.startswith("train"):		
			train_docs.append(reuters.raw(doc_id))
		else:
			test_docs.append(reuters.raw(doc_id))

	print(len(train_docs))
	print(len(test_docs))
		
	representer = tf_idf(train_docs);

	for doc in test_docs:
		print(feature_values(doc, representer))

	

#	classifier = SVC(C=1, kernel='poly', degree=1);
  	#classifier = KNeighborsClassifier(n_neighbors=30, p=2, weights='distance');
	#classifier = RandomForestClassifier(n_estimators=10);
	#classifier = MultinomialNB()

#	classifier.fit(representer.transform(train_docs).todense(), train_judgements);
	#classifier.fit(representer.transform(train_docs), train_judgements);

#	print("Model built");
#	decisions = classifier.predict(representer.transform(test_docs).todense());
	#decisions = list(map(lambda doc: classifier.predict(representer.transform(doc), test_docs)));

#	print(len(decisions))
#	print(len(test_judgements))
#	print("MicroF1 " + str(f1_score(test_judgements, decisions, average='micro')));
	#print("MacroF1 " + str(f1_score(test_judgements, decisions, average='macro')));
	
#	same = 0
#	for count, decision in enumerate(decisions):
#		if (decision == test_judgements[count]):
#			same +=1;
		#print(decision +" "+test_judgements[count]);
		#print 
#	print(str(same));

#	for item,doc in enumerate(test_docs):
#		print(show_feature_values(doc, tfidf) + "===" + test_judgements[item] + "===" + decisions[item])

if __name__=='__main__':
    main();
