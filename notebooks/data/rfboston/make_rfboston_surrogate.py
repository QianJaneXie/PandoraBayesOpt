import os
import sys

import numpy as np
import pickle

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestRegressor


np.random.seed(0)


def random_search(random_iter, x, y):
    # Default Values
    n_estimators = 10
    min_samples_split = 2
    best_score = -np.inf

    if random_iter > 0:
        sys.stdout.write("Do a random search %d times" % random_iter)
        param_dist = {"n_estimators": range(10, 110, 10),
                        "min_samples_split": range(2, 20, 2)}
        param_list = [{"n_estimators": n_estimators,
                        "min_samples_split": min_samples_split}]
        param_list.extend(list(ParameterSampler(param_dist,
                                                n_iter=random_iter-1,
                                                random_state=0)))
        for idx, d in enumerate(param_list):
            print(d)
            rf = RandomForestRegressor(n_estimators=int(d["n_estimators"]),
                                        criterion='mse',
                                        max_depth=None,
                                        min_samples_split=int(d["min_samples_split"]),
                                        min_samples_leaf=1,
                                        max_leaf_nodes=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=1,
                                        random_state=165,
                                        verbose=0, 
                                        )

            train_x, test_x, train_y, test_y = \
                train_test_split(x, y, test_size=0.5,
                                    random_state=0)
            rf.fit(train_x, train_y)
            sc = rf.score(test_x, test_y)
            print(sc)
            # Tiny output
            m = "."
            if idx % 10 == 0:
                m = "#"
            if sc > best_score:
                m = "<"
                best_score = sc
                n_estimators = int(d["n_estimators"])
                min_samples_split = int(d["min_samples_split"])
            sys.stdout.write(m)
            sys.stdout.flush()
        sys.stdout.write("Using n_estimators: %d, min_samples_split: %d\n" % (n_estimators, min_samples_split))
    return n_estimators, min_samples_split


def train(x, y, random_iter=100, **kwargs):

    print("Shape of training data: ", x.shape)
    print("First training sample: ", x[0])

    # Do a random search
    n_estimators, min_samples_split = random_search(random_iter=random_iter, x=x, y=y)

    rf = RandomForestRegressor(n_estimators=n_estimators,
                                criterion='mse',
                                max_depth=None,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=1,
                                max_leaf_nodes=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=1,
                                random_state=0,
                                verbose=0,
                                )
    rf.fit(x, y)
    return rf


script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

X = np.loadtxt(script_dir + "/rfboston_on_grid_X.txt")
objective_X = np.loadtxt(script_dir + "/rfboston_on_grid_objective.txt")
cost_X = np.loadtxt(script_dir + "/rfboston_on_grid_cost.txt")
cost_X *= 1000

objective_rf_surrogate = train(x=X, y=objective_X)
pickle.dump(objective_rf_surrogate, open("rfboston_objective_rf_surrogate.pkl", "wb"))

cost_rf_surrogate = train(x=X, y=cost_X)
pickle.dump(cost_rf_surrogate, open("rfboston_cost_rf_surrogate.pkl", "wb"))
