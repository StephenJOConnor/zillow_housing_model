import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=55)

# Average CV score on the training set was:-0.1637243646248613
exported_pipeline = GradientBoostingRegressor(alpha=0.8, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.7000000000000001, min_samples_leaf=4, min_samples_split=3, n_estimators=100, subsample=0.8)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
