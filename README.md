# Advanced_NLP_course_HW1
This assignment is about writing two (or more) linear classifiers and testing them out on multiple kinds of classification tasks. While the key models should not be difficult to implement, expect to spend some time setting up the general framework (which you will largely re-use in HW2) and considerable time investigating different classes of features that will be applicable for each of the tasks.



The repository is structured in the following way: 

`datasets` directory contains all the datasets and also the custom splits of the datasets we used for training and evaulating the performance of the model. 


`results` directory contains all the results including the predictions on the validation set and also the reports we generated based on the performance of the models.

To simply train models and evaulate them on datasets in the `datasets/custom` directory, you can use `train_eval.sh` bash script. It has a three stage pipeline. The first stage is training, secondly predicting the labels for the validation set, and finally evaluating the performance of the model given the predictions and true labels. 

The arguments passed to `classify.py` are as follows:

    -m: modelfile: the name/path of the model to load after training using train.py
    -i: inputfile: the name/path of the test file that has to be read one text per line
    -o: outputfile: the name/path of the output file to be written



The arguments passed to `train.py` are as follows:

    -m: type of model to be trained: naivebayes, perceptron
    -i: path of the input file where training file is in the form <text>TAB<label>
    --dev: path of the input file where evaluation file is in the form <text>
    --devlabels: path of the input file where evaluation true labels file is in the form <label>
    --epochs: Number of epochs for the training stage
    --ngram: cap of the ngram getting used for the bag of words featurization 
    --features: Feature used for training
    --wandb: Wandb name when logging
    --decrypt: whether to decrypt the content of the dataset or not


Also note that the input to `-m` argument can be one of the methods: `naivebayes`, `perceptron`, or `logistic_regression`; and the input to the `--features` argument can be `bow`, `tfidf`, and `word2vec` for perceptron and logistic regression model and `bow` as well as `wbow` (weighted naive bayes) for naive bayes model.


In order to use the `word2vec` features, you must first download the pretrained model through this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g), and extract it in the root directory of the project.