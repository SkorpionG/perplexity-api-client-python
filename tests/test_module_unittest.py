import os
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import requests
from perplexity_api_client import Perplexity
from perplexity_api_client.types import ResponseFormatType
from perplexity_api_client.exceptions import PerplexityAuthError, PerplexityConfigError, PerplexityAPIError

load_dotenv()


class TestPerplexity(unittest.TestCase):
    """測試 Perplexity API 客戶端"""
    @classmethod
    def setUpClass(cls):
        """在所有測試開始前執行一次"""
        # 設定測試用的環境變數
        os.environ.setdefault('PERPLEXITY_API_KEY', 'test-api-key')
        os.environ.setdefault('PERPLEXITY_MODEL', 'test-model')
        os.environ.setdefault('PERPLEXITY_SYSTEM_ROLE', 'test-role')

    def setUp(self):
        """設置測試環境"""
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.model = os.getenv('PERPLEXITY_MODEL')
        self.system_role = os.getenv('PERPLEXITY_SYSTEM_ROLE')
        self.client = Perplexity(
            api_key=self.api_key,
            model=self.model,
            system_role=self.system_role
        )

    def tearDown(self):
        """清理測試環境"""
        self.client.close()

    def test_initialization(self):
        """測試初始化"""
        self.assertEqual(self.client.auth_token, self.api_key)
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(self.client.system_role, self.system_role)
        self.assertEqual(len(self.client.chat_history), 1)  # 應該只有 system role

    def test_invalid_initialization(self):
        """測試無效的初始化參數"""
        with self.assertRaises(PerplexityAuthError):
            Perplexity(api_key="", model=self.model,
                       system_role=self.system_role)

        with self.assertRaises(PerplexityConfigError):
            Perplexity(api_key=self.api_key, model="",
                       system_role=self.system_role)
            
    def test_config_validation(self):
        """測試設定值驗證"""
        # 測試有效設定
        valid_config = {"temperature": 0.5, "top_p": 0.8}
        self.client.set_config(**valid_config)
        self.assertEqual(self.client.config["temperature"], 0.5)

        # 測試無效設定
        invalid_config = {"invalid_key": "value"}
        with self.assertRaises(PerplexityConfigError):
            self.client.set_config(**invalid_config)

    def test_config_reset(self):
        """測試重設設定"""
        self.client.set_config(temperature=0.8)
        self.client.reset_config()
        self.assertEqual(self.client.config["temperature"], 0.2)  # 預設值

    @patch('requests.Session.request')
    def test_ask_method(self, mock_request):
        """測試 ask 方法"""
        # 模擬成功回應
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = '{"choices":[{"message":{"content":"test response"}}]}'
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_request.return_value = mock_response

        # 測試不同回應格式
        text_response = self.client.ask("test", response_type="text")
        self.assertTrue(isinstance(text_response, str))
        self.assertEqual(text_response, mock_response.text)

        json_response = self.client.ask("test", response_type="json")
        self.assertTrue(isinstance(json_response, dict))
        self.assertEqual(json_response, mock_response.json())

        llm_response = self.client.ask("test", response_type="llm_response")
        self.assertTrue(isinstance(llm_response, str))
        self.assertEqual(llm_response, "test response")

        # 測試原始回應
        raw_response = self.client.ask("test", response_type="raw")
        self.assertEqual(raw_response, mock_response)

    @patch('requests.Session.request')
    def test_chat_method(self, mock_request):
        """測試 chat 方法"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test chat response"}}]
        }
        mock_request.return_value = mock_response

        response = self.client.chat("test message")
        self.assertEqual(response, "test chat response")
        self.assertEqual(len(self.client.chat_history),
                         3)  # system + user + assistant
        
    @patch('requests.Session.request')
    def test_api_error_handling(self, mock_request):
        """測試 API 錯誤處理"""
        # 模擬 API 錯誤
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "test error"}
        mock_request.side_effect = requests.exceptions.RequestException(
            response=mock_response)

        with self.assertRaises(PerplexityConfigError):
            self.client.ask("test message")

    def test_config_management(self):
        """測試設定管理"""
        test_config = {"temperature": 0.5}
        self.client.set_config(**test_config)
        self.assertEqual(self.client.config["temperature"], 0.5)

        with self.assertRaises(PerplexityConfigError):
            self.client.set_config(invalid_key="value")

    def test_response_type_validation(self):
        """測試回應類型驗證"""
        # 測試有效的回應類型
        for response_type in ["raw", "text", "json", "llm_response"]:
            ResponseFormatType.validate_response_type(response_type)

        # 測試無效的回應類型
        with self.assertRaises(ValueError):
            ResponseFormatType.validate_response_type("invalid_type")


if __name__ == '__main__':
    unittest.main(verbosity=2)
