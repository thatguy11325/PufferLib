import pufferlib.models
from pufferlib.pytorch import LSTM

class Recurrent(LSTM):
    def __init__(input_size=512, hidden_size=512, num_layers=1):
        super().__init__(input_size, hidden_size, num_layers)


class Policy(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=4, flat_size=64*7*7):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
        )
