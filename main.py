import os
from modules.adaptivity import AdaptivityExperiment
from modules.cost_cifar10 import CPCost
from modules.finite_sample import FiniteSampleExperiment
from modules.model_comparison import UncertaintyExperiment
from modules.presentation import presentation_experiments
from modules.distribution_free import DistributionFreeExperiment

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# presentation_experiments()

distribution_free = DistributionFreeExperiment(alpha=0.1)
distribution_free.run()

# adaptivity = AdaptivityExperiment(alpha=0.1)
# adaptivity.run()

# cp_cost = CPCost(alpha=0.1)
# cp_cost.run()

# finite_sample = FiniteSampleExperiment(alpha=0.1, n_cal=50, n_trials=100)
# finite_sample.run()

# exp = UncertaintyExperiment(alpha=0.1) # 90% confidence
# exp.run_all()