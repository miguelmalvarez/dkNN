# dkNN

This repository contains the code and experiments to test the quality, efficiency of an implementation to kNN that optimises the number of neighbours per class instead of globaly, therefore being theoretically similar to having multiple binary kNN classifiers, one per class. This solution has the same complexity of the multi-label kNN, though.

TC experiments have shown that SVM and other advance classifiers outperform the traditional kNN. However, we believe that this comparison have been done based on and unfair biased against kNN. While some methods such as SVM are constructed by implicitly combining multiple binary classifiers (e.g., one-vs-rest), kNN models are usually globally trained. The main reason for this is usually efficiency as multiple kNN models would require substancially more computational time to produce the predicted classes. This paper proposes a variation of kNN that computes and optimises the number of neighbours per class, while maintaining a complexity and efficiency almost identical to the global optimisation. The two main research questions that we are aiming to answer are:

RQ1: Is it a binary kNN optimised per class better than a global kNN and SVM?

RQ2: Is it possible to produce a similar output to multiple binary kNN classifiers with the efficiency of a globally tuned one?

In order to answer this questions, we will evaluate the quality and effeciency for each one of the candidate models based on several text classification collections. 