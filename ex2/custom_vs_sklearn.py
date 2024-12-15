import pandas as pd
from sklearn.metrics import mean_squared_error, \
    root_mean_squared_error as RMSE, \
    median_absolute_error as MAE, \
    r2_score as R2
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
from sklearn.model_selection import train_test_split, cross_validate
import time 
import json
import numpy as np

# from sklearn.metrics import root_mean_squared_error

from random_forest.random_forest import RandomForestRegressor

data = pd.read_csv("data/bike_sharing/day.csv")

# split
X = data.iloc[:, 2:-3].values # exclude instant, casual, and registered
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

def compare_regressors(regressor1, regressor2, _X_train, _X_test, _Y_train, _Y_test):
    scores = (
        get_scores_for_regressor(regressor1, _X_train, _X_test, _Y_train, _Y_test), 
        get_scores_for_regressor(regressor2, _X_train, _X_test, _Y_train, _Y_test), 
    )

    regressor1_Y_pred = regressor1.predict(_X_test)
    regressor2_Y_pred = regressor2.predict(_X_test)

    indices = np.random.choice(len(regressor1_Y_pred), 10, replace=False)

    regressor1_results = {
        "Scores": scores[0],
        "True Y": [a[0] for a in _Y_test[indices].tolist()],
        "Predicted Y": regressor1_Y_pred[indices].tolist()
    }

    regressor2_results = {
        "Scores": scores[1],
        "True Y": [a[0] for a in _Y_test[indices].tolist()],
        "Predicted Y": regressor2_Y_pred[indices].tolist()
    }

    return {
        "Regressor1": regressor1_results,
        "Regressor2": regressor2_results,
    }


def get_scores_for_regressor(regressor, _X_train, _X_test, _Y_train, _Y_test) -> dict:
    start = time.time()
    regressor.fit(_X_train, _Y_train)
    Y_pred = regressor.predict(_X_test)
    runtime = time.time() - start

    return {
        "RMSE": RMSE(_Y_test, Y_pred),
        "MAE": MAE(_Y_test, Y_pred),
        "R2": R2(_Y_test, Y_pred),
        "Runtime (s)": runtime
    }

comparison = [
    (RandomForestRegressor(n_trees=100, tree_max_depth=10, tree_min_nodes=2), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=50, tree_max_depth=5, tree_min_nodes=2), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=50, tree_max_depth=20, tree_min_nodes=6), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=200, tree_max_depth=3, tree_min_nodes=10), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=200, tree_max_depth=30, tree_min_nodes=3), RandomForestRegressor_sklearn()),
]

results = [
    compare_regressors(regressor_our, regressor_sklearn, X_train, X_test, Y_train, Y_test)
    for regressor_our, regressor_sklearn in comparison
]

print(json.dumps(results, indent=" "))

# [
#     ({'RMSE': (684.8947655219964), 'MAE': (301.76625960735146), 'R2': 0.8865311549454048}, {'RMSE': (708.3289862855006), 'MAE': (271.0100000000002), 'R2': 0.8786334605046121}), 
#     ({'RMSE': (754.586947753635), 'MAE': (369.0582229738693), 'R2': 0.862263983632545}, {'RMSE': (696.7949988568896), 'MAE': (263.6099999999997), 'R2': 0.8825537944335904}), 
#     ({'RMSE': (691.4529236452819), 'MAE': (332.3602424242408), 'R2': 0.8843477264086306}, {'RMSE': (692.2920662199015), 'MAE': (293.5600000000004), 'R2': 0.8840668464465931}), 
#     ({'RMSE': (841.3583101486646), 'MAE': (477.18516995970003), 'R2': 0.8287656389172814}, {'RMSE': (691.1105198275646), 'MAE': (274.0), 'R2': 0.8844622388326904}), 
#     ({'RMSE': (695.6363379234549), 'MAE': (288.72032738095186), 'R2': 0.8829440589741128}, {'RMSE': (693.1909584392346), 'MAE': (291.5799999999999), 'R2': 0.8837655890139349})
# ]


# [
#   [
#     {
#       "RMSE": 671.2321515881395,    
#       "MAE": 336.732983921439,
#       "R2": 0.8729231664624746,  
#       "Runtime (s)": 67.05198264122009 
#     },
#     {
#       "RMSE": 710.7930924975121,    
#       "MAE": 331.7800000000002,  
#       "R2": 0.8575024883700255,  
#       "Runtime (s)": 0.21100211143493652
#     }
#   ],
#   [
#     {
#       "RMSE": 729.0116247550486,    
#       "MAE": 401.6543775081873,  
#       "R2": 0.8501040874647781,  
#       "Runtime (s)": 21.0895094871521  
#     },
#     {
#       "RMSE": 705.4236856255229,    
#       "MAE": 332.82000000000016,    
#       "R2": 0.8596472396352889,  
#       "Runtime (s)": 0.2180013656616211    
#     }
#   ],
#   [
#     {
#       "RMSE": 690.7039298999591,    
#       "MAE": 333.70866666666643,    
#       "R2": 0.8654434828924206,  
#       "Runtime (s)": 32.66623568534851 
#     },
#     {
#       "RMSE": 706.2415402640642,    
#       "MAE": 308.0300000000002,  
#       "R2": 0.8593216064023091,  
#       "Runtime (s)": 0.21100091934204102
#     }
#   ],
#   [
#     {
#       "RMSE": 832.9994810698657,    
#       "MAE": 475.90355335674076,    
#       "R2": 0.8042912041248612,  
#       "Runtime (s)": 57.5836660861969  
#     },
#     {
#       "RMSE": 709.1069821232325,    
#       "MAE": 330.9499999999998,  
#       "R2": 0.8581777384832986,  
#       "Runtime (s)": 0.20699858665466309
#     }
#   ],
#   [
#     {
#       "RMSE": 685.790474747151,  
#       "MAE": 327.5251388888887,  
#       "R2": 0.8673510610016048,  
#       "Runtime (s)": 143.97328972816467    
#     },
#     {
#       "RMSE": 731.0493531382197,    
#       "MAE": 326.9899999999998,  
#       "R2": 0.8492649402220631,  
#       "Runtime (s)": 0.20799994468688965
#     }
#   ]
# ]