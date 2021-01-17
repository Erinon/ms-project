class Agent:
    def __init__(self, ob_size, actions, device):
        self.ob_size = ob_size
        self.actions = actions
        self.device = device

    def move(self, state, epsilon=0):
        return random.randrange(self.actions)

    def preprocess_input(self, x):
        return x

    @staticmethod
    def load_pretrained(model_dir):
        return NotImplemented

    def save(save_dir):
        return NotImplemented
