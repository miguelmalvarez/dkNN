from nltk import word_tokenize
from nltk.corpus import reuters 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re

def tokenize(text): 
	words = word_tokenize(text);
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
	p = re.compile('\w');
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

def main():
	category_docs = reuters.fileids("acq");
	raw_docs = list(map(lambda doc: reuters.raw(doc).lower(), category_docs));

	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english');
	tfs = tfidf.fit_transform(raw_docs);

#	text = 'all great and precious things are lonely.';
#	response = tfidf.transform([text]);
#	print(response);
#	feature_names = tfidf.get_feature_names();
#	for col in response.nonzero()[1]: 
#		print(feature_names[col], ' - ', response[0, col]);

if __name__=='__main__':
    main();