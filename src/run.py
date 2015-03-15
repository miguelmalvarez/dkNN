import nltk.classify.util 
from nltk.corpus import reuters 
from nltk.classify import NaiveBayesClassifier 

def main():
	print("Script to run the experiments");

	print(reuters.categories());
	print(reuters.categories());

	train = set()
	for category in reuters.categories():
		print(category + " == " + str(len(reuters.fileids(category))))
		for fileid in reuters.fileids(category):
			if (fileid.startswith("train")):
				train.add(fileid);

	print(len(train));					

if __name__=='__main__':
    main()