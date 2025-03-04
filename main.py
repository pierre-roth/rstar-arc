from arc_rstar.agents import BeamSearch
from arc_rstar.solver import Solver
from arc_rstar.config import BaseConfig

if __name__ == '__main__':
    config = BaseConfig()
    solver = Solver(config.mode)
    agent = BeamSearch(config)
    solver.solve(agent)
    print("Done!")



