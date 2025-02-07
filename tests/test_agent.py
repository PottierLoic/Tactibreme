import unittest
import os
import torch
from ai.agent import Agent
from ai.network import Network
from color import Color

class TestAgent(unittest.TestCase):
    def setUp(self):
        """Setup test environment by initializing an agent."""
        self.network = Network()
        self.agent = Agent(
            color=Color.BLUE,
            network=self.network,
            epsilon=0.1,
            gamma=0.99,
            learning_rate=1e-3,
            buffer_size=10000,
        )
        self.checkpoint_path = "test_agent.pth"

    def tearDown(self):
        """Cleanup by removing generated checkpoint file."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def test_save_checkpoint(self):
        """Test if the agent successfully saves a checkpoint."""
        self.agent.save_checkpoint(self.checkpoint_path)
        self.assertTrue(os.path.exists(self.checkpoint_path), "Checkpoint file was not created.")

    def test_load_checkpoint(self):
        """Test if the agent successfully loads a saved checkpoint."""
        self.agent.save_checkpoint(self.checkpoint_path)
        new_agent = Agent(
            color=Color.RED,
            network=Network(),
            epsilon=0.5,
            gamma=0.8,
            learning_rate=5e-3,
            buffer_size=5000,
        )
        self.assertNotEqual(new_agent.epsilon, self.agent.epsilon)
        self.assertNotEqual(new_agent.gamma, self.agent.gamma)
        self.assertNotEqual(new_agent.learning_rate, self.agent.learning_rate)
        self.assertNotEqual(new_agent.buffer_size, self.agent.buffer_size)
        new_agent.load_checkpoint(self.checkpoint_path)
        self.assertEqual(new_agent.epsilon, self.agent.epsilon)
        self.assertEqual(new_agent.gamma, self.agent.gamma)
        self.assertEqual(new_agent.learning_rate, self.agent.learning_rate)
        self.assertEqual(new_agent.buffer_size, self.agent.buffer_size)

    def test_load_non_existent_checkpoint(self):
        """Test loading from a non-existent checkpoint file."""
        with self.assertRaises(FileNotFoundError):
            self.agent.load_checkpoint("non_existent_file.pth")

    def test_load_corrupted_checkpoint(self):
        """Test loading from a corrupted checkpoint file."""
        with open(self.checkpoint_path, "w") as f:
            f.write("corrupted data")
        with self.assertRaises(torch.serialization.pickle.UnpicklingError):
            self.agent.load_checkpoint(self.checkpoint_path)

if __name__ == "__main__":
    unittest.main()
