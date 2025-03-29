# This file contains the code for generating and augmenting SFT data for the model.
# the cleaned data is loaded line by line
# each line is used to create sft data
# examples not from the training set are immediately transformed into the correct format and saved

# for training examples, the process is slightly more difficult
# instead of just transforming it with the original task data, we do that, but also 300 additional tasks of the same format
# from re_arc and thus 300x the amount of training data for those examples

# this data is then saved in an appropriate format to be used for training
