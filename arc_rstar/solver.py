import logging
import os
from typing import Any
import json

from vllm.outputs import RequestOutput

from pebble import ProcessPool

from arc_rstar.agents import BS, MCTS
from arc_rstar.llms import PolicyModel, RewardModel
from config import Config, TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

Agent = BS | MCTS


class Solver:
    def __init__(self, config: Config):
        self.config: Config = config

        self.policy = PolicyModel(config)
        self.reward = RewardModel(config)

        self.policy.init()  # Initialize the policy model
        self.reward.init()  # Initialize the reward model

    @staticmethod
    def processor(agent: Agent, output: list[RequestOutput]) -> Agent:
        agent.generate_next_step(output)
        return agent

    @staticmethod
    def selector(agent: Agent, score: list[float]) -> Agent:
        agent.select_next_step(score)
        return agent

    @staticmethod
    def generate_preprocess(agents):
        prompts = []
        rewards = []
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
                    rewards.extend(agent.get_rewards())
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards

    def generate_postprocess(self, outputs: list[list[RequestOutput]], valid_agents: list[Agent]) -> list[Agent]:
        post_agents = []

        with ProcessPool(max_workers=min(len(valid_agents), self.config.cpus - 1)) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()

        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                logger.error(f"Exception while generating postprocess: {error}")
                post_agents.append(None)

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

    def output(self, agents: list[Agent]):
        return [agent.get_nodes() for agent in agents]

    def solve(self, agents: list[Agent]):

        for rollout in range(self.config.num_simulations):
            # Initialize the initial search starting point of agents, and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)
                agent.rollout_idx = rollout

            for step in range(self.config.max_depth):
                logger.debug(f"----------------- Current Rollout: {rollout} -----------------")
                logger.debug(f"----------------- Current Step: {step} -----------------")

                # TODO: handle valid_rewards
                prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards = self.generate_preprocess(
                    agents)

                if not valid_agents + expanded_agents:
                    break

                outputs = self.policy.generate(prompts)

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

        return self.output(agents)
