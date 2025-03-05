from dataclasses import field


class Config:
    def __init__(self):
        self.verbose = False
        # paths
        self.policy_model_dir = None
        self.process_preference_model_dir = None
        self.prompt_path = None


        self.temperature = 0.0


        self.seed = 0





