import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('insurance.csv', sep=',')

# Use a simple label encoder. 
def label_encoder(df):
    cat_cols = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object'] ]
    labelencoder = LabelEncoder()
    for cat in cat_cols:    
        df[cat] = labelencoder.fit_transform(df[cat])        
    return df

tpot_data = label_encoder(tpot_data)
features = tpot_data.drop('charges', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['charges'], random_state=None)



# Average CV score on the training set was: 0.8617748561646719
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=18)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=7, min_samples_split=16, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
