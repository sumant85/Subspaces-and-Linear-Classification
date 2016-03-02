* Problem statement is given in HW1.pdf.
* The individual code files are pca.py, lda.py and `naive_bayes_classifier.py`.
* The answers for description of methods used, description of results and further analysis of results have been summarized at the end of the report.
* I've also implemented the classifiers for multi-class scenario for possible extra credit.
* The PDF report has all the code sections hidden for better readability. However, I couldn't find a way to resize the plots while converting to PDF so the report is unfortunately clunky in some places.
* I've also included the iPython notebook so that you can go over the code.
* I've used an external library to compute the confusion matrix [link](https://github.com/scls19fr/pandas_confusion)
* For each dataset, I've considered two cases - with and without whitening(0 mean, unit stddev) and reported accuracies for both cases.
* The report contains a final accuracy table summarizing the results.
* For MNIST dataset, all classification is done after first performing PCA. Otherwise 700+ features makes the classifiers too slow and also error-prone.
* The data is assumed to be under data/ directory (relative to the jupyter server)
