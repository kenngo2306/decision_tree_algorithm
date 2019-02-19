# The program implement the id3 decision tree learning algorithm 
# and post pruning algorithm to resolve overfitting issue
# The program accepts 6 inputs parameters as follow:
# L and K values: 		positive integers for the pruning algorithm
# Training set path: 	path to training data set to be learned by the id3 algorithm
# Validation set path: 	path to data set to be validated in pruning algorithm
# Test set path: 		path to test set to calculate the accuracy of id3 algorithm
# to_print: 			yes/no value to print the output to the console
# In addition to console output,
# all outputs will be stored in outputs.txt file in the same folder as Assignment1.py file
# Assumptions:
# 1. All data sets have been preprocessed without missing value
# 2. Each attributes (include all Xs and target attribute Y) accepts only 0 or 1 as value

# Compilation steps:
# 1. Have python 3 install
# 2. Copy main program "Assignment1.py" and
#    3 data sets in the same folder if possible
# 3. open command prompt in windows
# 4. type the input in following format to run the program
# python .\Assignment1.py <L_value> <K_value> <training_set_path> <validation_set_path> <test_set_path> <yes/no to print>
# The following are examples of commands:
  
python .\Assignment1.py 23 5 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 42 10 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 162 15 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 2 25 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 53 12 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 54 32 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 242 30 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 12 51 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 11 3 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 87 23 .\data_sets1\training_set.csv .\data_sets1\validation_set.csv .\data_sets1\test_set.csv no
python .\Assignment1.py 2 31 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 11 54 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 32 4 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 211 20 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 155 6 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 323 10 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 13 43 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 72 72 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 32 32 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no
python .\Assignment1.py 17 32 .\data_sets2\training_set.csv .\data_sets2\validation_set.csv .\data_sets2\test_set.csv no