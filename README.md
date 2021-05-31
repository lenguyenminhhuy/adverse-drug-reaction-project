# adverse-drug-reaction-project

The project aims to classify a particular contextual content and determine whether it is related to Adverse Drug Reaction or not. After successfullyidentifyingitscategory, weproceedtopickoutandlabelsuchwordsbelonging to our predefined labels, such as DRUG and DOSAGE.

Specifically, in the first task, we will construct a deep learning model to perform binary classification. The input for this model will be a chunk of text and the expected outcome should be a binary indicator 0 or 1, in which 1 indicates having adverse drug reaction, while 0 is the opposite. Additionally, the sequences which are classified 1 from the previous model will be hence transferred to the information extraction model, which is a Named Entity Recognition deep learning model.
