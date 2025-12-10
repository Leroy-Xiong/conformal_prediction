import os
from modules.finite_sample import FiniteSampleExperiment
from modules.model_comparison import ModelComparison
from modules.model_free import ModelFreeExperiment
from modules.presentation import presentation_experiments
from modules.distribution_free import DistributionFreeExperiment

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# presentation_experiments()

# finite_sample = FiniteSampleExperiment(alpha=0.1, n_cal=50, n_trials=100)
# finite_sample.run()

# distribution_free = DistributionFreeExperiment(alpha=0.1)
# distribution_free.run()

# model_free = ModelFreeExperiment(alpha=0.1)
# model_free.run()

model_comparison = ModelComparison(alpha=0.1)
model_comparison.run_all()