import json
import numpy as np
from util.logger import Logger
from learning.ppo_agent import PPOAgent
from learning.amp_agent import AMPAgent
from learning.custom_amp_agent import CustomAMPAgent
from learning.trpo_agent_original import TRPOAgent
# from learning.trpo_agent import TRPOAgent

AGENT_TYPE_KEY = "AgentType"

def build_agent(world, id, file):
    agent = None
    with open(file) as data_file:    
        json_data = json.load(data_file)
        
        assert AGENT_TYPE_KEY in json_data
        agent_type = json_data[AGENT_TYPE_KEY]
        
        if (agent_type == PPOAgent.NAME):
            agent = PPOAgent(world, id, json_data)
        elif (agent_type == AMPAgent.NAME):
            agent = AMPAgent(world, id, json_data)
        elif (agent_type == CustomAMPAgent.NAME):
            agent = CustomAMPAgent(world, id, json_data)
        elif (agent_type == TRPOAgent.NAME):
            agent = TRPOAgent(world, id, json_data)
        else:
            assert False, 'Unsupported agent type: ' + agent_type

        Logger.print("="*10)
        Logger.print(type(agent))
        Logger.print("="*10)

    return agent