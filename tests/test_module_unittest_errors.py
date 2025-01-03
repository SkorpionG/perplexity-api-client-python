import unittest
from unittest.mock import patch, MagicMock
import requests
from perplexity_api_client import Perplexity
from perplexity_api_client.exceptions import (
    PerplexityConfigError,
    PerplexityAPIError,
    PerplexityAuthError
)


class TestPerplexityErrors(unittest.TestCase):
    """測試 Perplexity API 錯誤處理"""

    def setUp(self):
        """設置測試環境"""
        self.api_key = "test-api-key"
        self.model = "test-model"
        self.system_role = "test-role"
        self.client = Perplexity(
            api_key=self.api_key,
            model=self.model,
            system_role=self.system_role
        )

    def tearDown(self):
        """清理測試環境"""
        self.client.close()

    def test_missing_required_params(self):
        """測試缺少必要參數"""
        # 測試沒有 API 金鑰
        with self.assertRaises(PerplexityAuthError):
            Perplexity(
                api_key="",
                model=self.model,
                system_role=self.system_role
            )

        # 測試沒有模型名稱
        with self.assertRaises(PerplexityConfigError):
            Perplexity(
                api_key=self.api_key,
                model="",
                system_role=self.system_role
            )

        # 測試沒有系統角色
        with self.assertRaises(PerplexityConfigError):
            Perplexity(
                api_key=self.api_key,
                model=self.model,
                system_role=""
            )

    @patch('requests.Session.request')
    def test_unauthorized_error(self, mock_request):
        """測試未授權錯誤"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "unauthorized"}
        mock_request.side_effect = requests.exceptions.RequestException(
            response=mock_response)

        with self.assertRaises(PerplexityAuthError):
            self.client.ask("test")

    @patch('requests.Session.request')
    def test_invalid_model_error(self, mock_request):
        """測試無效模型錯誤"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "invalid model"}
        mock_request.side_effect = requests.exceptions.RequestException(
            response=mock_response)

        with self.assertRaises(PerplexityConfigError):
            self.client.ask("test")

    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """測試請求限制錯誤"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "rate limit exceeded"}
        mock_request.side_effect = requests.exceptions.RequestException(
            response=mock_response)

        with self.assertRaises(PerplexityAPIError) as context:
            self.client.ask("test")
        self.assertEqual(context.exception.status_code, 429)

    @patch('requests.Session.request')
    def test_server_error(self, mock_request):
        """測試伺服器錯誤"""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "internal server error"}
        mock_request.side_effect = requests.exceptions.RequestException(
            response=mock_response)

        with self.assertRaises(PerplexityAPIError) as context:
            self.client.ask("test")
        self.assertEqual(context.exception.status_code, 500)

    @patch('requests.Session.request')
    def test_timeout_error(self, mock_request):
        """測試請求超時錯誤"""
        mock_request.side_effect = requests.exceptions.Timeout()

        with self.assertRaises(PerplexityAPIError):
            self.client.ask("test")

    @patch('requests.Session.request')
    def test_connection_error(self, mock_request):
        """測試連線錯誤"""
        mock_request.side_effect = requests.exceptions.ConnectionError()

        with self.assertRaises(PerplexityAPIError):
            self.client.ask("test")

    @patch('requests.Session.request')
    def test_invalid_json_response(self, mock_request):
        """測試無效 JSON 回應"""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = "Invalid JSON"
        mock_response.json.side_effect = ValueError()
        mock_request.return_value = mock_response

        response = self.client.ask("test", response_type="json")
        self.assertIsNone(response)


if __name__ == '__main__':
    unittest.main(verbosity=2)
