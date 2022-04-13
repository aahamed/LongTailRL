"""
PPO Experiment
"""
import argparse
import gym
import os
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

def main( args ):

    # set up exp_dir
    exp_path = os.path.join(args.exp_dir, args.exp_id)
    os.makedirs( exp_path, exist_ok=True )
    # set up logger
    new_logger = configure(exp_path, ["stdout", "log"])
    new_logger.log(f"Starting Experiment: {args.exp_id}")
    new_logger.log(f"args: {args}")

    # Parallel environments
    env = make_vec_env(args.env, n_envs=4)

    # instantiate agent
    model = PPO(args.policy, env, verbose=1)
    model.set_logger(new_logger)
    # train the agent
    model.learn(total_timesteps=int(args.n_steps))
    # save the agent
    model_path = os.path.join(exp_path, "model")
    model.save(model_path)

    # load the trained agent
    model = PPO.load(model_path, env=env)

    # import pdb; pdb.set_trace()

    # Evaluate the agent
    eps_returns, _ = evaluate_policy(model, model.get_env(),
            n_eval_episodes=100, return_episode_rewards=True)

    eps_returns = np.array( eps_returns )
    print(f"mean return: {eps_returns.mean():.2f} " +
        f"std return: {eps_returns.std():.2f}")
    save_dict = {
        "eps-returns" : eps_returns,
    }
    save_path = os.path.join(exp_path, "save.pkl")
    pickle.dump( save_dict, open(save_path, "wb") )

    new_logger.log("Experiment Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Experiment')
    parser.add_argument("--exp-dir", type=str, default="exp_dir/ppo/cartpole",
            help="experiment directory")
    parser.add_argument("--exp-id", type=str, required=True,
            help="experiment id")
    parser.add_argument("--n-steps", type=float, default=1e6,
            help="total timesteps for training")
    parser.add_argument("--env", type=str, default="CartPole-v1",
            help="environment id")
    parser.add_argument("--policy", type=str, default="MlpPolicy",
            help="policy architecture")
    args = parser.parse_args()
    main( args )
