# tfbind8-project

### Dataset
We are using tf bind 8 as a dataset, however the implementation didn't work flawless, so we had to do some changes in the source code to access the data. The process of obtaining the data is omitted here, we have extracted the data to the `data` folder.

### Scorer
We trained a scorer to approximate the black box function f, which are the y-values associated with the x-data. For that a RF model and an MLP were trained as machine learning models. We've also tested multiple deep learning approaches using MLP, CNN, LSTM, CNN-LSTM, LSTM-CNN and Transformers architectures. The best hyperparameter settings have been tested using Random Search. Results are visualized.

### Generator
MSC-Thesis
