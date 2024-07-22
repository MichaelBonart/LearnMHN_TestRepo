# author(s): Stefan Vocht
#
# this script learns the MHNs as shown in the demo Jupyter notebook, but on the full dataset
#

from mhn.optimizers import cMHNOptimizer, oMHNOptimizer

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)


def main():
    c_opt = cMHNOptimizer()
    c_opt.load_data_from_csv("LUAD_n12.csv")
    c_opt.set_penalty(c_opt.Penalty.L1)

    o_opt = oMHNOptimizer()
    o_opt.load_data_from_csv("LUAD_n12.csv")
    o_opt.set_penalty(o_opt.Penalty.SYM_SPARSE)

    data_properties = c_opt.get_data_properties()

    n_cv_steps = 5
    n_cv_folds = 3
    lambda_min = 0.1 / data_properties["samples"]
    lambda_max = 100 / data_properties["samples"]
    print("Minimum lambda:", lambda_min)
    print("Maximum lambda:", lambda_max)

    print("cross-validate cMHN:")
    cMHN_lambda = c_opt.lambda_from_cv(
        lambda_min=lambda_min, lambda_max=lambda_max, steps=n_cv_steps, nfolds=n_cv_folds, show_progressbar=True)
    print("oMHN lambda:", cMHN_lambda)

    print("cross-validate oMHN:")
    oMHN_lambda = o_opt.lambda_from_cv(
        lambda_min=lambda_min, lambda_max=lambda_max, steps=n_cv_steps, nfolds=n_cv_folds, show_progressbar=True)
    print("oMHN lambda:", oMHN_lambda)

    print("Train cMHN...")
    c_opt.train(lam=cMHN_lambda)
    print("Save cMHN")
    c_opt.result.save(filename="demo_script_cMHN.csv")

    print("Train oMHN...")
    o_opt.train(lam=oMHN_lambda)
    print("Save oMHN")
    o_opt.result.save(filename="demo_script_oMHN.csv")

    print("Plot cMHN")
    c_opt.result.plot()
    plt.savefig("cMHN_plot.png")

    plt.cla()

    print("Plot oMHN")
    o_opt.result.plot()
    plt.savefig("oMHN_plot.png")


if __name__ == '__main__':
    main()
