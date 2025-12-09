import os
from modules.finite_sample import FiniteSampleExperiment
from modules.model_comparison import UncertaintyExperiment
from modules.model_free import ModelFreeExperiment
from modules.presentation import presentation_experiments
from modules.distribution_free import DistributionFreeExperiment

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# presentation_experiments()

# finite_sample = FiniteSampleExperiment(alpha=0.1, n_cal=50, n_trials=100)
# finite_sample.run()

# distribution_free = DistributionFreeExperiment(alpha=0.1)
# distribution_free.run()

exp = ModelFreeExperiment(alpha=0.1)
exp.load_data()
exp.prepare_data()
exp.run_comparison()
exp.plot_results()

# exp = UncertaintyExperiment(alpha=0.1) # 90% confidence
# exp.run_all()