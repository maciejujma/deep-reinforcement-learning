cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "learning_rate": 1e-2
}

q_learning_hyperparameters = {
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 100,
    "max_t": 99,
    "gamma": 0.95,
    "learning_rate": 0.7,
    "decay_rate": 0.005,
    "max_epsilon": 1.0,
    "min_epsilon": 0.05
}

a2c_stable_baselines_hyperparameters = {
    "policy": "MlpPolicy",
    "gae_lambda": 0.9,
    "gamma": 0.99,
    "learning_rate": 0.00096,
    "max_grad_norm": 0.5,
    "n_steps": 8,
    "vf_coef": 0.4,
    "ent_coef": 0.0,
    "policy_kwargs": {
        "log_std_init": -2,
        "ortho_init": False},
    "normalize_advantage": False,
    "use_rms_prop": True,
    "use_sde": True,
    "verbose": 1
}
