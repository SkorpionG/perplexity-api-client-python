import os
import unittest
from dotenv import load_dotenv
from perplexity_api_client import Perplexity

TEST_MODEL = 'llama-3.1-sonar-small-128k-online'
TEST_ROLE = 'You are a helpful assistant.'


class TestPerplexityLive(unittest.TestCase):
    """實際 API 整合測試"""

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.api_key = os.getenv('PPLX_API_KEY')
        if not cls.api_key:
            raise unittest.SkipTest('未設定 API 金鑰')

    def setUp(self):
        self.client = Perplexity(
            api_key=self.api_key,
            model=TEST_MODEL,
            system_role=TEST_ROLE
        )

    def tearDown(self):
        self.client.close()

    def test_initialization(self):
        """測試初始化"""
        self.assertEqual(self.client.auth_token, self.api_key)
        self.assertEqual(self.client.model, TEST_MODEL)
        self.assertEqual(self.client.system_role, TEST_ROLE)
        self.assertEqual(len(self.client.chat_history), 1)
        default_config = Perplexity.default_config
        default_config.pop('max_tokens')
        self.assertDictEqual(self.client.config, default_config)

    def test_live_ask(self):
        """測試實際 API 呼叫"""
        response = self.client.ask("What is Python?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertEqual(len(self.client.chat_history), 1)

    def test_append_ask(self):
        """Test append user request and llm response using ask method"""
        response = self.client.ask("What is Python?", append_history=True)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertEqual(len(self.client.chat_history), 3)

    def test_live_chat(self):
        """測試實際聊天功能"""
        response = self.client.chat("What is Python?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
