from nltk import word_tokenize
from nltk.corpus import reuters 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.metrics import f1_score

def tokenize(text): 
	words = word_tokenize(text);
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('[a-zA-Z]+');
	filtered_tokens = list(filter(lambda token: p.search(token), tokens));
	return filtered_tokens

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

# Return the representer, without transforming
def tf_idf(docs):	
	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', min_df=3, max_features=3000, 
							norm='l2', use_idf=True, sublinear_tf=True);
	tfidf.fit(docs);
	return tfidf;

def prepare_data():
	train_docs = []
	train_judgements = []

	test_docs = []
	test_judgements = []

	for category in reuters.categories():
		for doc_id in reuters.fileids(category):
			doc = reuters.raw(doc_id);
			if doc_id.startswith("train"):
				train_docs.append(doc);
				train_judgements.append(category);
			else:
				test_docs.append(doc);
				test_judgements.append(category);

	return {'train_docs': train_docs,
	 		'test_docs': test_docs,
	 		'train_judgements': train_judgements,
	 		'test_judgements': test_judgements};

def main():
	data = prepare_data()
	train_docs = data['train_docs']
	test_docs = data['test_docs']
	train_judgements = data['train_judgements']
	test_judgements = data['test_judgements']

	representer = tf_idf(train_docs);
	print(str(len(representer.get_feature_names())) + " Features");	
	
	classifier = KNeighborsClassifier(n_neighbors=45);

	#classifier.fit(representer.transform(train_docs).todense(), train_judgements);
	classifier.fit(representer.transform(train_docs), train_judgements);

	print("Model built");
	#decisions = list(map(lambda doc: classifier.predict(representer.transform([doc]).todense()), test_docs));
	decisions = list(map(lambda doc: classifier.predict(representer.transform(doc), test_docs)));

	print("MicroF1 " + str(f1_score(test_judgements, decisions, average='micro')));
	print("MacroF1 " + str(f1_score(test_judgements, decisions, average='macro')));

if __name__=='__main__':
    main();