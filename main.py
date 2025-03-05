from arc_rstar.agents import BeamSearch
from arc_rstar.solver import Solver
from cli import CLI
from config import Config

if __name__ == '__main__':
    args = CLI.parse_args()
    config = CLI.create_config(args)

    solver = Solver(config)
    agent = BeamSearch(config)
    solver.solve(agent)

    print("Done!")



