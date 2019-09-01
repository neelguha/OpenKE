# Code for performing grid search over hyperparameters in OpenKE. 
# All results are stored under result/
#       $dataset_name/
#           $grid_name/
#               params_to_indices.json: # maps each parameter combination to a directory index 
#               params_1/               # all results for a particular combination of parameters 
#                   checkpoint/         # checkpoint directory for model 
#                   results/            # results directory for model 

import argparse, os, json, yaml
from utils import *
import random
import warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings

import config 
from models import *

# parser 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default='Transe', help='hyperparameter grid to search over')
parser.add_argument('-d', '--data_dir', help='dataset to learn over')
parser.add_argument('-r', '--result_dir', help='file path to dataset folder')
parser.add_argument('-c', '--cpu', action="store_true", help='train on CPU')
parser.add_argument('-s', '--samples', default=1, help='Number of random configurations to try.')
args = parser.parse_args()

def load_grid():
    ''' Loads hyperparameter grid

        Returns:
            hyper_grid: dictionary of parameters and value ranges, loaded from args.method
    '''
    hpath = os.path.join('job_params', '%s.yaml' % args.method)
    hyper_grid = yaml.load(open(args.method, 'r'))
    return hyper_grid

def generate_hyperparams(grid):
    ''' Samples a set of hyperparamters at random. 

        Args:
            grid: dictionary, where keys correspond to parameter names and values correspond to parameter ranges
    '''
    sampled_params = {}
    for key, value_range in grid.items():
        if isinstance(value_range, list): 
            # pick a random value
            sampled_params[key] = random.choice(value_range)
        else:
            sampled_params[key] = value_range
    
    return sampled_params

def get_model(model_str):
    if model_str == 'TransE': 
        return TransE
    else:
        raise Exception("unknown model.")

def learn_model(hp, data_dir, out_dir):
    '''
    Args: 
        hp: a hyperparameter setting with which to train a model 

    Returns: 
        valid_score: the validation score (hits@10 for triple completion on the validation dataset)
    '''
    con = config.Config()

    if not args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
    con.set_use_gpu(not args.cpu)
    

    # set intput, checkpoint, and results directory 
    if not data_dir[-1] == '/':
        data_dir = data_dir + "/"
    print("Loading data from %s" % data_dir)
    con.set_in_path(data_dir)
    
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    create_directory(checkpoint_dir)
    con.set_checkpoint_dir(checkpoint_dir)

    results_dir = os.path.join(out_dir, 'results')
    create_directory(results_dir)
    con.set_result_dir(results_dir)

    # optimization / model parameters 
    con.set_train_times(hp['max_epochs'])
    con.set_nbatches(hp['nbatches'])	
    con.set_alpha(hp['lr'])
    con.set_bern(hp['neg_strategy'])
    con.set_dimension(hp['dimension'])
    con.set_margin(hp['margin'])
    con.set_ent_neg_rate(hp['ent_neg_rate'])
    con.set_rel_neg_rate(hp['ent_rel_rate'])
    con.set_opt_method(hp['opt_method'])
    con.set_save_steps(hp['save_steps'])
    con.set_valid_steps(hp['valid_steps'])
    con.set_early_stopping_patience(hp['early_stopping_patience'])
    con.set_work_threads(8)
    
    con.init()
    con.set_train_model(get_model(hp['model']))
    model, valid_score = con.train()
    return valid_score


def main():
    
    grid_name = args.method 
    data_dir = args.data_dir
    rand_samples = int(args.samples)
    assert(not None in [grid_name, data_dir, rand_samples])

    print(args)
    out_dir = args.result_dir
    create_directory(out_dir)

    print("-"*20)
    print("Performing random search over grid %s and dataset %s for %d iterations" % (grid_name, data_dir, rand_samples))
    print("-"*20)
    print()
    grid = load_grid()

    scores = {} # dictionary of scores from parameter index to validation score. 
    score_fpath = os.path.join(out_dir, 'scores.json')
    for rand_iter in range(rand_samples):
        hyperparams = generate_hyperparams(grid)
        print("="*20)
        print("Random Search Sample %d" % rand_iter)
        print("HP:", hyperparams)
        param_dir = os.path.join(out_dir, 'params_%d' % rand_iter)
        create_directory(param_dir)

        # save parameters to file 
        param_file = os.path.join(param_dir, 'params.json')
        save_json(hyperparams, param_file)

        # learn model with configuration 
        valid_score = learn_model(hyperparams, data_dir, param_dir)
        print("SAMPLE VALIDATION SCORE: %f" % valid_score)
        print("Saving results...")
        print()
        scores[rand_iter] = valid_score 
        save_json(scores, score_fpath)    
         


if __name__ == '__main__':
    main()
