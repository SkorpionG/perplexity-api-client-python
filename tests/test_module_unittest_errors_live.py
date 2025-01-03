import os
import unittest
from dotenv import load_dotenv
from perplexity_api_client import Perplexity
from perplexity_api_client.exceptions import (
    PerplexityConfigError,
    PerplexityAPIError,
    PerplexityAuthError
)

TEST_MODEL = 'llama-3.1-sonar-small-128k-online'
TEST_ROLE = 'You are a helpful assistant.'


class TestPerplexityErrorsLive(unittest.TestCase):
    """測試 Perplexity API 實際錯誤情境"""

    @classmethod
    def setUpClass(cls):
        """設置測試環境變數"""
        load_dotenv()
        cls.valid_api_key = os.getenv('PPLX_API_KEY')
        if not cls.valid_api_key:
            raise unittest.SkipTest('未設定 API 金鑰')

    def test_invalid_api_key(self):
        """測試無效的 API 金鑰"""
        client = Perplexity(
            api_key="invalid_key_12345",
            model=TEST_MODEL,
            system_role=TEST_ROLE
        )
        with self.assertRaises(PerplexityAuthError):
            client.ask("Hello")

    def test_invalid_model(self):
        """測試無效的模型名稱"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model="invalid-model",
            system_role=TEST_ROLE
        )
        with self.assertRaises(PerplexityConfigError):
            client.ask("Hello")

    def test_invalid_temperature(self):
        """測試無效的溫度設定"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model=TEST_MODEL,
            system_role=TEST_ROLE
        )
        with self.assertRaises(PerplexityConfigError):
            client.set_config(temperature=2.0)
            client.chat("Hello")

    def test_empty_message(self):
        """測試空訊息"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model="sonar-small-chat",
            system_role="You are a helpful assistant."
        )
        with self.assertRaises(PerplexityConfigError):
            client.ask("")

    def test_invalid_top_p(self):
        """測試無效的 top_p 值"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model="sonar-small-chat",
            system_role="You are a helpful assistant."
        )
        with self.assertRaises(PerplexityConfigError):
            client.set_config(top_p=1.5)
            client.chat("Hello")

    def test_invalid_config_error(self):
        """測試無效設定錯誤"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model="sonar-small-chat",
            system_role="You are a helpful assistant."
        )
        with self.assertRaises(PerplexityConfigError):
            client.set_config(invalid_key="value")

    def test_invalid_response_type_error(self):
        """測試無效回應類型錯誤"""
        client = Perplexity(
            api_key=self.valid_api_key,
            model="sonar-small-chat",
            system_role="You are a helpful assistant."
        )
        with self.assertRaises(ValueError):
            client.ask("test", response_type="invalid_type")


if __name__ == '__main__':
    unittest.main(verbosity=2)
