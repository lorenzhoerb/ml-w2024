import pandas as pd
from sklearn.metrics import mean_squared_error, \
    root_mean_squared_error as RMSE, \
    median_absolute_error as MAE, \
    r2_score as R2
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
from sklearn.neighbors import KNeighborsRegressor
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

# Custom RandomForestRegressor, vs sklearn RandomForestRegressor
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
print('Custom RandomForestRegressor, vs sklearn RandomForestRegressor')
print('---------------------------------------------------------')
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

# Custom RandomForestRegressor, vs KNNRegressor

comparison_knn = [
    (RandomForestRegressor(n_trees=100, tree_max_depth=10, tree_min_nodes=2), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=50, tree_max_depth=5, tree_min_nodes=2), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=50, tree_max_depth=20, tree_min_nodes=6), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=200, tree_max_depth=3, tree_min_nodes=10), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=200, tree_max_depth=30, tree_min_nodes=3), KNeighborsRegressor(n_neighbors=5, weights='distance')),
]

results_knn = [
    compare_regressors(regressor_our, knn_regressor, X_train, X_test, Y_train, Y_test)
    for regressor_our, knn_regressor in comparison_knn
]

print('Custom RandomForestRegressor, vs KNNRegressor')
print('---------------------------------------------------------')
print(json.dumps(results_knn, indent=" "))


#[
#    {
#        "Regressor1": {
#            "Scores": {
#                "RMSE": 614.0251533309942,
#                "MAE": 317.2229496336995,
#                "R2": 0.8980653138792102,
#                "Runtime (s)": 30.5993390083313
#            },
#            "True Y": [
#                6118,
#                1842,
#                7347,
#                1693,
#                985,
#                4322,
#                7498,
#                3974,
#                4844,
#                3163
#            ],
#            "Predicted Y": [
#                6761.102701402576,
#                3093.5719999999997,
#                7340.013583649236,
#                2066.3497463768117,
#                1804.8635873015874,
#                4222.416276515152,
#                7294.427987854944,
#                3518.8413333333338,
#                4811.089025542381,
#                2817.738147058824
#            ]
#        },
#        "Regressor2": {
#            "Scores": {
#                "RMSE": 881.8088110130857,
#                "MAE": 481.40055424956427,
#                "R2": 0.7897681396310778,
#                "Runtime (s)": 0.0007002353668212891
#            },
#            "True Y": [
#                6118,
#                1842,
#                7347,
#                1693,
#                985,
#                4322,
#                7498,
#                3974,
#                4844,
#                3163
#            ],
#            "Predicted Y": [
#                [
#                    6914.531806496652
#                ],
#                [
#                    3632.984337235407
#                ],
#                [
#                    7545.57222233457
#                ],
#                [
#                    3058.153284062668
#                ],
#                [
#                    1063.5042694728318
#                ],
#                [
#                    3874.367455884686
#                ],
#                [
#                    7556.297399702122
#                ],
#                [
#                    3756.5702531094644
#                ],
#                [
#                    5465.824645303929
#                ],
#                [
#                    3268.0632625650996
#                ]
#            ]
#        }
#    },
#    {
#        "Regressor1": {
#            "Scores": {
#                "RMSE": 679.3216793986118,
#                "MAE": 433.0106974642804,
#                "R2": 0.8752327466140326,
#                "Runtime (s)": 9.89257287979126
#            },
#            "True Y": [
#                3310,
#                4661,
#                5169,
#                1589,
#                7393,
#                7965,
#                4744,
#                7693,
#                7466,
#                1011
#            ],
#            "Predicted Y": [
#                3031.0832706279234,
#                4972.957155644158,
#                6409.046168213964,
#                1948.7774448582918,
#                6750.55176184988,
#                7259.581972450529,
#                4800.909950590171,
#                7258.276221252777,
#                7000.8921713441305,
#                2393.8474209747906
#            ]
#        },
#        "Regressor2": {
#            "Scores": {
#                "RMSE": 881.8088110130857,
#                "MAE": 481.40055424956427,
#                "R2": 0.7897681396310778,
#                "Runtime (s)": 0.0005702972412109375
#            },
#            "True Y": [
#                3310,
#                4661,
#                5169,
#                1589,
#                7393,
#                7965,
#                4744,
#                7693,
#                7466,
#                1011
#            ],
#            "Predicted Y": [
#                [
#                    4407.555193557375
#                ],
#                [
#                    4512.136254276648
#                ],
#                [
#                    6052.3555392036105
#                ],
#                [
#                    1953.6315459729562
#                ],
#                [
#                    7014.870743652412
#                ],
#                [
#                    7571.240501971186
#                ],
#                [
#                    4743.467614907951
#                ],
#                [
#                    7231.18335637965
#                ],
#                [
#                    6909.277157558489
#                ],
#                [
#                    2331.513035688526
#                ]
#            ]
#        }
#    },
#    {
#        "Regressor1": {
#            "Scores": {
#                "RMSE": 593.1951379043259,
#                "MAE": 297.39500000000044,
#                "R2": 0.9048640124039261,
#                "Runtime (s)": 15.21829605102539
#            },
#            "True Y": [
#                4302,
#                1248,
#                5445,
#                4744,
#                7175,
#                4318,
#                1530,
#                4844,
#                3940,
#                4629
#            ],
#            "Predicted Y": [
#                4288.688666666666,
#                1795.0083333333337,
#                5081.153,
#                4771.270999999999,
#                6508.9153333333325,
#                4013.1313333333337,
#                1646.993333333333,
#                4982.209,
#                3461.773333333333,
#                4560.969999999999
#            ]
#        },
#        "Regressor2": {
#            "Scores": {
#                "RMSE": 881.8088110130857,
#                "MAE": 481.40055424956427,
#                "R2": 0.7897681396310778,
#                "Runtime (s)": 0.0005660057067871094
#            },
#            "True Y": [
#                4302,
#                1248,
#                5445,
#                4744,
#                7175,
#                4318,
#                1530,
#                4844,
#                3940,
#                4629
#            ],
#            "Predicted Y": [
#                [
#                    4552.435669122282
#                ],
#                [
#                    1046.9595916300875
#                ],
#                [
#                    5615.837592665272
#                ],
#                [
#                    4743.467614907951
#                ],
#                [
#                    6360.798390019166
#                ],
#                [
#                    2819.4497928903857
#                ],
#                [
#                    1684.904197256015
#                ],
#                [
#                    5465.824645303929
#                ],
#                [
#                    4197.718147241954
#                ],
#                [
#                    4391.896122098849
#                ]
#            ]
#        }
#    },
#    {
#        "Regressor1": {
#            "Scores": {
#                "RMSE": 772.4542106811931,
#                "MAE": 468.1278461835054,
#                "R2": 0.8386774213805838,
#                "Runtime (s)": 27.422057151794434
#            },
#            "True Y": [
#                3663,
#                3614,
#                7175,
#                2192,
#                5892,
#                3368,
#                4590,
#                3974,
#                3784,
#                4333
#            ],
#            "Predicted Y": [
#                3435.9957746927453,
#                3500.7230222557673,
#                6381.494245434726,
#                2168.9834719958826,
#                5183.934770533916,
#                3278.5602177963865,
#                4804.386147852496,
#                3536.264782851121,
#                4547.1684646035155,
#                4827.615892995502
#            ]
#        },
#        "Regressor2": {
#            "Scores": {
#                "RMSE": 881.8088110130857,
#                "MAE": 481.40055424956427,
#                "R2": 0.7897681396310778,
#                "Runtime (s)": 0.0005791187286376953
#            },
#            "True Y": [
#                3663,
#                3614,
#                7175,
#                2192,
#                5892,
#                3368,
#                4590,
#                3974,
#                3784,
#                4333
#            ],
#            "Predicted Y": [
#                [
#                    3833.4749211164894
#                ],
#                [
#                    3416.766401146176
#                ],
#                [
#                    6360.798390019166
#                ],
#                [
#                    1873.0204136543687
#                ],
#                [
#                    3237.532913488232
#                ],
#                [
#                    2674.4458747531876
#                ],
#                [
#                    4441.357551127666
#                ],
#                [
#                    3756.5702531094644
#                ],
#                [
#                    4380.073287221573
#                ],
#                [
#                    4742.667078201807
#                ]
#            ]
#        }
#    },
#    {
#        "Regressor1": {
#            "Scores": {
#                "RMSE": 602.7412212375691,
#                "MAE": 313.5715277777772,
#                "R2": 0.901777393893126,
#                "Runtime (s)": 65.35612201690674
#            },
#            "True Y": [
#                1096,
#                6118,
#                3777,
#                2660,
#                7466,
#                2475,
#                3249,
#                4792,
#                6565,
#                1589
#            ],
#            "Predicted Y": [
#                4130.373690476191,
#                6810.7780625000005,
#                3894.7185463659143,
#                2666.8074166666665,
#                7386.75,
#                2974.661416666667,
#                2692.206011904762,
#                5062.139749999999,
#                6748.059305555556,
#                2083.182027777778
#            ]
#        },
#        "Regressor2": {
#            "Scores": {
#                "RMSE": 881.8088110130857,
#                "MAE": 481.40055424956427,
#                "R2": 0.7897681396310778,
#                "Runtime (s)": 0.0005719661712646484
#            },
#            "True Y": [
#                1096,
#                6118,
#                3777,
#                2660,
#                7466,
#                2475,
#                3249,
#                4792,
#                6565,
#                1589
#            ],
#            "Predicted Y": [
#                [
#                    4962.643900618456
#                ],
#                [
#                    6914.531806496652
#                ],
#                [
#                    4261.584607486999
#                ],
#                [
#                    1882.8489951073284
#                ],
#                [
#                    6909.277157558489
#                ],
#                [
#                    2188.227369727128
#                ],
#                [
#                    4041.9071138274535
#                ],
#                [
#                    4633.215095840355
#                ],
#                [
#                    5514.285150167734
#                ],
#                [
#                    1953.6315459729562
#                ]
#            ]
#        }
#    }
#]
