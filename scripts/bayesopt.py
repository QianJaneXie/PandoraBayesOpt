import wandb

def run_bayesopt_experiment(config):
    print(config)
    acquisition_function = config['acquisition_function']
    print(acquisition_function)
    return acquisition_function

wandb.init()

acquisition_function = run_bayesopt_experiment(wandb.config)

wandb.log({"acquisition_function": acquisition_function})
wandb.finish()
