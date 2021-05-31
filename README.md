# adverse-drug-reaction-project

The project aims to classify a particular contextual content and determine whether it is related to `Adverse Drug Reaction` or not. After successfully identifying its category, we proceed to pick out and label such words belonging to our predefined labels, such as `DRUG` and `DOSAGE`.

Specifically, in the first task, I tried serveral approaches which are machine learning approach and deep learning approach to perform binary text classification. The input for this model will be a chunk of text and the expected outcome should be a binary indicator 0 or 1, in which 1 indicates **having adverse drug reaction**, while 0 is **the opposite**. Additionally, the sequences which are classified 1 from the previous model will be hence transferred to the information extraction model, which is a Named Entity Recognition deep learning model.

## Text classification model

In this project, to handle the imbalance of text classification dataset, I applied back translation technique to oversample the data in the minor class. To be more specific, I did translate the original corpus from English to three common languages that are French, Japanese, and Spanish, and then back translating them to English. 

  **Experiment details**:

  ● `One-layer LSTM`: An RNN model with only one LSTM layer and two activation functions: ReLU and sigmoid.

  ● `One-layer LSTM with dropout`: For addressing overfitting issue, a regularization technique - dropout was
  applied to the one-LSTM-layer model. I have chosen the dropout rate of 0.5.

  ● `Two-layer LSTM with dropout`: The related work has proved that RNN model performance can improve thanks to the number of LSTM layers, in other words the depth. Hence, I decided to experiment with two LSTM layers with the same dropout rate which is 0.5.

  ● `Three-layer LSTM with dropout`: Inspired by the high performance of Google Translate tool using seven layers of LSTM [10], I stacked the model with another LSTM layer as well as the dropout layer.

  ● `Bidirectional LSTM with dropout`: I replaced one LSTM layer with one Bidirectional LSTM (BLSTM) layer. Although BLSTM has double memory cells compared to the LSTM layer, the text data is processed in both directions, which helps the model capture the patterns that may be fail recognized in a unidirectional LSTM.

  **Experiment result**:
  
| Approach        | Type of Network           | Model            | F1 score|
| ----------------|:-------------------------:| :---------------|---------|
| ML Approach     |                           | MultinomialNB    |   0.857 |
|                 | | LightGBM | 0.828|
|                 | | Logistic Regression | 0.867|
| DL Approach      | RNN | One LSTM layer (baseline) |0.771|
|       | RNN |One LSTM layer with dropout |0.862|
|     | RNN | Two LSTM layers with dropout |0.866|
|    | RNN | Three LSTM layers with dropout |0.854|
|    | RNN | Bidirectional LSTM with dropout |0.864|
|  |`CNN`  | `CNN with dropout` |`0.870`|
|  |CNN  | Deep CNN |0.840|

Although CNN is not widely used in NLP tasks, our CNN with dropout model has been considered as the best one with F1 score of 0.87. The result suggests a great promise of adapting a CNN model in NLP field as well as balancing dataset using back-translation.



## Name Entity Recognition model

The baseline approach was to construct a simple classifier by inheriting scikit-learn base classes **BaseEstimator** and **TransformerMixin**, to implement such functions like `get_params`, `get_params` and `fit_transform`. The result was overall acceptable, but this is just a

To address the issue, I decided to convert words to simple feature vector then apply R**andomForestClassifier** model to recognize entities of the words and make prediction accordingly. Hence, I applied a more sophisticated feature extraction function based upon the,

Further model performance improvement was accomplished by employing **Conditional Random Field (CRF)** model via sklearn-crfsuite and ELI5. The idea was to implement sequence labelling technique to predict the sequences that use the contextual information to add information which will be used by the model to make a correct prediction.

**Experiment Detail**:

   ● `Bi-LSTM baseline with dropout`: by using Bi-LSTM instead of the traditional one, the model is able to preserve context information in both past and future, comparing to LSTM which only capable to intake past information. We also added one dropout layer having rate of 0.1. In the Bi-LSTM layer, dropout layer is a regularization technique to avoid overfitting.

   ● `Bi-LSTM + one more LSTM layer with dropout`: to help the model capturing more information, we added up 1 more LSTM layer.
     
 **Experiment Result**:
 | Approach        | Type of Network           | Model            | Macro F1 score|
| ----------------|:-------------------------:| :---------------|---------|
| ML Approach     |                           | BaseEstimator + TransformerMixin classes   |   0.59 |
|                 | | Random Forest | 0.25|
|                 | | CRF | 0.664|
|                 | | `CRF_tune` | `0.677`|
| DL Approach      | RNN | Bi-LSTM with dropout |0.30|
|       | RNN |Bi-LSTM + LSTM with dropout |0.31|

In terms of Machine Learning approach, the `fine-tuned CRF model` achieve the highest result among other counterparts. This model surpassed our expectation as it foreshadowed Bi-LSTM model, which is well-known for its performance in the NER task. Despite its mediocre performance, **Bi-LSTM** is still one of the most sufficient techniques of the NER task if we can acquire bigger dataset or properly handle imbalance issue with suitable techniques.

## Future work

- Doing experiments with BiLSTM-CRF on NER task.
- Trying transformer and see what the result will be.
- Training on larger-scale dataset, not only Drug and Dosage, but also Symptoms and Diseases.
- Use ORC technique to read input from the Medical Health Record.
