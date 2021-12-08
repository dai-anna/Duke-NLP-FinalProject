#%%


#%%
import optuna
import numpy as np

#%%
def one_training_run(params: dict):
    # create model with params=params
    # train model
    # load one with best val acc
    # return val acc as a raw number
    pass



def objective(trial):
    embedding_dim = 2 ** trial.suggest_int("embedding_dim", 4, 6)
    hidden_size = 2 ** trial.suggest_int("hidden_size", 4, 8)
    hidden_dense_dim = 2 ** trial.suggest_int("hidden_dense_dim", 4, 8)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 0.000000001, 0.5, log=True)

    print(
        embedding_dim,
        hidden_size,
        hidden_dense_dim,
        dropout_rate,
        l2_reg,
    )
    return np.random.rand()


study = optuna.create_study(
    "sqlite:///tf_hyperparameter_study.db", direction="maximize"
)
study.optimize(objective, n_trials=10)
print("-" * 80)
print(f"Found best params {study.best_params}")


# %%
from optuna.visualization import plot_parallel_coordinate
plot_parallel_coordinate(study)

# %%
