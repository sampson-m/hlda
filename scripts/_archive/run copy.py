
from _functions import run_gibbs_pipeline

# okay so we've done the filtering now what do we want to do
# we should probably run 14 topic fits with fasttopics, lda, and hlda
# okay so we should beef up the splatter framework to have more activity
# topics
# and also figure out some way to compare the gene means and diff expr
# also would like to have genes be diff expressed lower as opposed to upwards

if __name__ == "__main__":

    topic_hierarchy = {'A': [0,1],
##                       'B': [0,1,3],
##                       'C': [0,4]
                       }
    K = 2
    num_loops = 20000
    burn_in = 5000
    hyperparams = {'alpha_beta': 1, 'alpha_c': 1}

    input_file='../data/hsim_A_var/simulated_counts.csv'
    sample_dir = "../samples/hsim_A_var/"
    est_dir = '../estimates/hsim_A_var/'
    
    run_gibbs_pipeline(topic_hierarchy, K, num_loops, burn_in, hyperparams, sample_dir, est_dir, input_file)
    
    
























