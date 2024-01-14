import wandb

def run_bayesopt_experiment(config):
    print(config)
    regret = config['dim']
    time = config['length_scale']
    return (regret,time)

wandb.init()

(regret,time) = run_bayesopt_experiment(wandb.config)

wandb.log({"regret": regret})
wandb.log({"time": time})
wandb.finish()
