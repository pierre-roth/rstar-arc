import logging

from pebble import ProcessPool
from tqdm import tqdm
# Removed static import of RequestOutput to avoid heavy dependency at module import.

# noinspection PyUnresolvedReferences
from rstar_deepthink.agents import Agent, temperature_lerp, temperature_beta_cdf
from rstar_deepthink.config import Config
from rstar_deepthink.llms import PolicyModel, RewardModel
from rstar_deepthink.node import Node

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, config: Config):
        self.config: Config = config

        self.policy = PolicyModel(config)
        self.reward = RewardModel(config)

        self.policy.init()  # Initialize the policy model
        self.reward.init()  # Initialize the reward model

    @staticmethod
    def processor(agent: Agent, output) -> Agent:
        agent.generate_next_step(output)
        return agent

    @staticmethod
    def selector(agent: Agent, score: list[float]) -> Agent:
        agent.select_next_step(score)
        return agent

    @staticmethod
    def generate_preprocess(agents):
        prompts = []
        prompts_span = [0]
        valid_agents = []
        invalid_agents = []
        expanded_agents = []

        for agent in agents:
            if agent.should_generate_next():
                if agent.has_expanded():
                    expanded_agents.append(agent)
                else:
                    agent_prompts = agent.create_prompts()
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)
            else:
                invalid_agents.append(agent)

        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents

    def generate_postprocess(self, outputs: list[list], valid_agents: list[Agent]) -> list[Agent]:
        post_agents = []

        num_workers = min(len(valid_agents), max(1, self.config.cpus - 1))

        with ProcessPool(max_workers=num_workers) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs)
            iterator = future.result()

        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                logger.critical(f"Exception while generating postprocess (shouldn't happen): {error}")
                post_agents.append(None)
            progress_bar.update(1)
        progress_bar.close()

        # update agents
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents

    @staticmethod
    def value_preprocess(agents: list[Agent]) -> (list[str], list[int]):
        prompts = []
        prompts_span = [0]
        for agent in agents:
            agent_prompts = agent.create_prompts(is_value_only=True)
            prompts.extend(agent_prompts)
            prompts_span.append(prompts_span[-1] + len(agent_prompts))
        return prompts, prompts_span

    def value_postprocess(self, scores, valid_agents) -> list[Agent]:
        for agent, score in zip(valid_agents, scores):
            if agent is not None:
                self.selector(agent, score)
        return valid_agents

    @staticmethod
    def output_nodes(agents: list[Agent]) -> list[list[Node]]:
        return [agent.get_nodes() for agent in agents]

    def solve(self, agents: list[Agent]):
        temperature = self.config.policy_temperature

        for rollout in range(self.config.num_rollouts):
            # Update temperature every time all examples have been used
            if self.config.variable_temperature and rollout % len(self.config.example_names) == 0:
                temperature = temperature_lerp(rollout, self.config.num_rollouts, self.config.min_policy_temperature,
                                               self.config.max_policy_temperature)
                # temperature = temperature_beta_cdf(rollout, self.config.num_simulations, self.config.min_policy_temperature, self.config.max_policy_temperature)

            # Initialize the initial search starting point of agents, and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)
                agent.update(rollout, temperature)

            logger.info(
                f"----------------- Current Rollout: {rollout} ----------------- ({temperature}, {agents[0].example_name})")

            for step in range(self.config.max_depth):
                logger.info(f"----------------- Current Step: {step} -----------------")

                prompts, prompts_span, valid_agents, invalid_agents, expanded_agents = self.generate_preprocess(agents)

                if not valid_agents + expanded_agents:
                    break

                outputs = self.policy.generate(prompts, temperature)

                logger.debug(f"Number of outputs: {len(outputs)}")

                reconstructed_outputs = [outputs[bos_idx: eos_idx] for bos_idx, eos_idx in
                                         zip(prompts_span, prompts_span[1:])]

                # process output and run python code
                valid_agents = self.generate_postprocess(reconstructed_outputs, valid_agents)

                # step evaluation
                prompts, prompts_span = self.value_preprocess(valid_agents)

                scores = self.reward.score(prompts)
                reconstructed_scores = [scores[bos_idx: eos_idx] for bos_idx, eos_idx in
                                        zip(prompts_span, prompts_span[1:])]

                # selection
                valid_agents = self.value_postprocess(reconstructed_scores, valid_agents)
                # for expanded agents, just do selection step
                expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents)

                # keep all agents
                agents = valid_agents + invalid_agents + expanded_agents

        return self.output_nodes(agents)
