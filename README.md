## Consumer Finance Complaints

### Description:

- The purpose of this project is to **classify Kaggle Consumer Finance Complaints into 11 classes**, namely Debt collection, Consumer Loan, Mortgage, Credit card, Credit reporting, Student loan, Bank account or service, Payday loan, Money transfers, Other financial service, Prepaid card.
- The model was built with **Convolutional Neural Network (CNN)** and **Word Embeddings** on **Tensorflow**.
- The consumer complaints were cleaned and fed into the textCNN; each word in the complaints is embedded and each sentence is padded with 0 if its length is less than the length of the maximum sentence. Within the convolution layer, a convolution sliding window filter is applied and move vertically to detect n-gram word features. The result is then pooled, after which it can feed into another convolutional layer or directly into the fully connected dense layer with dropout enabled and the resulting score runs through softmax function to map to probability distribution for each of the possible 11 output nodes in the output layer.

### Data: [Kaggle Consumer Finance Complaints](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

### Train:

- Command: python3 train.py training_data.file parameters.json
- Example: `python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json`

A directory will be created during training, and the trained model will be saved in this directory.

### Predict:

Provide the model directory (created when running `train.py`) and new data to `predict.py`.

- Command: python3 predict.py ./trained_model_directory/ new_data.file
- Example: `python3 predict.py ./trained_model_1479757124/ ./data/small_samples.json`

### Conclusion:

Text CNN is an excellent model for text classification-based problems. However, it still suffers from the limitation such that it cannot keep track of the entire context of a given word in a sentence. To that end, we can implement bidirectional LSTM to keep track of the context around the word in a given sentence in order to generate more accurate prediction. This is validated when we evaluate the accuracy, precision and recall between the two different models, bidirectional LSTM noticeably have better performance in terms of better accuracy and similar precision and recall when compared to textCNN.

For future investigation, the current state-of-the-art model that outperforms all others is the attention-based model, which would highlight and pay attention to more important words and also take into account of the word context in text-based classification, could also be implemented in order to achieve the best predictive performance.

### Reference:

- [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
