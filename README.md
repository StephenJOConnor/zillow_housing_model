# zillow_housing_model
My submission for an internal FI Consulting data science competition.

The FI_kaggle.pynb notebook contains data wrangling and the creation of datapipelines using TPOT. Once TPOT determines
which ML technique to use, the full pipeline (model) code is written and saved the Intermediate Results folder. 

Model_runs.pynb runs the chosen pipeline (using gradient boosting) using the test data and saves the results in a CSV file. 
