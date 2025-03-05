from config import Config
from arc_rstar.llms import PolicyModel, ProcessPreferenceModel


class Solver:
    def __init__(self, config: Config):
        self.policy = PolicyModel(config)
        self.preference = ProcessPreferenceModel(config)

    def generator_next(self):
        pass

    def select_next(self):
        pass

    def generate_preprocess(self):
        pass

    def generate_postprocess(self):
        pass

    def value_preprocess(self):
        pass

    def value_postprocess(self):
        pass

    def save_intermediate_metrics(self):
        pass

    def save_intermediate_rollouts(self):
        pass

    def output(self):
        pass

    def solve(self, agent):
        pass

