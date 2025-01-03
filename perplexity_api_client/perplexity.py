"""
A Perplexity API client wrapper module for Python.
"""

from typing import Dict, List, Optional, Union
import requests
from .exceptions import PerplexityAPIError, PerplexityAuthError, PerplexityConfigError
from .constants import PPLX_API_ENDPOINT
from .types import ResponseFormatType


class Perplexity:

    default_config: Dict[str, Union[float, bool, List, str, int]] = {
        "max_tokens": None,
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    def __init__(self, api_key: str, model: str, system_role: str, config: Optional[Dict[str, Union[float, bool, List, str, int]]] = None):
        self.auth_token: str = api_key
        self.model: str = model
        self.system_role: str = system_role
        self._validate_required_params()
        self.chat_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_role,
            },
        ]

        self.__config = {}
        if config is not None:
            self._validate_and_set_config(config)
        self.__session = requests.Session()
        self.__session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    @property
    def config(self) -> dict:
        """
        Retrieves the current configuration.

        This property combines the default configuration with any custom settings
        provided by the user, returning a dictionary of configuration values
        where none of the values are None.

        Returns:
            Dict[str, Union[float, bool, List, str, int]]: A dictionary of the
            current configuration settings with non-None values.
        """
        config = {**self.__class__.default_config, **self.__config}
        return {key: value for key, value in config.items() if value is not None}

    @config.setter
    def config(self, value) -> None:
        self.set_config(**value)

    def set_config(self, **kwargs) -> None:
        """
        Sets the configuration parameters.

        This method updates the configuration settings by validating the provided
        keyword arguments and applying them. Only parameters that differ from the
        default configuration are updated.

        Parameters:
            **kwargs: Arbitrary keyword arguments representing configuration settings.
        """
        self._validate_and_set_config(kwargs)

    def reset_config(self) -> None:
        """
        Resets the configuration to its default values.

        This method does not modify the auth token, model name, or system role.
        """
        self.__config = {}

    def ask(self, message: str, model: str = None, system_role: str = None, append_history: bool = False, response_type: str = "llm_response", **config) -> Union[requests.Response, str, dict, None]:
        """
        Asks the Perplexity AI a question.

        This method sends a single message to the Perplexity API and returns the response.
        The message is formatted as a single user message, with the system role prepended.
        The response is formatted according to the response_type parameter.

        Parameters:
            message (str): The message to send to the AI.
            model (str): Optional model name to use for the request. Defaults to the instance's model.
            system_role (str): Optional system role to use for the request. Defaults to the instance's system role.
            append_history (bool): Whether to append the request and response to the chat history. Defaults to False.
            response_type (str): The type of response to return. Defaults to "llm_response". Valid options are: "raw", "text", "json", and "llm_response".
            **config: Additional configuration parameters to pass to the Perplexity API.

        Returns:
            Union[requests.Response, str, dict, None]: The response from the Perplexity API, formatted according to the response_type parameter.

        Raises:
            PerplexityAPIError: If the request fails or the response is invalid.
        """
        model = model or self.model
        system_role = system_role or self.system_role
        self._validate_required_params(None, model, system_role)
        append_history = append_history or False
        config = self._get_validated_config(config)

        ResponseFormatType.validate_response_type(response_type)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_role
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            **config
        }

        try:
            response = self.__session.request(
                "POST", PPLX_API_ENDPOINT, json=payload, timeout=60)

            if response.ok:
                formatted_response = self._format_response(
                    response)
                if append_history and formatted_response["llm_response"]:
                    self.chat_history.append({
                        "role": "user",
                        "content": message
                    })
                    self.chat_history.append({
                        "role": "assistant",
                        "content": formatted_response["llm_response"]
                    })

                return formatted_response[response_type]
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._raise_error_message(e)

    def chat(self, message: str, response_type: str = "llm_response") -> Union[requests.Response, str, dict, None]:
        """
        Sends a message to the AI and appends the response to the chat history.

        Parameters:
            message (str): The message to send to the AI.
            response_type (str): The type of response to return. Defaults to "llm_response". Valid options are: "raw", "text", "json", and "llm_response".

        Returns:
            Union[requests.Response, str, dict, None]: The response from the Perplexity API, formatted according to the response_type parameter.

        Raises:
            PerplexityAPIError: If the request fails or the response is invalid.
        """
        ResponseFormatType.validate_response_type(response_type)
        self._validate_required_params()

        self.chat_history.append({
            "role": "user",
            "content": message
        })
        payload = {
            "model": self.model,
            "messages": self.chat_history,
            **self.__config
        }

        try:
            response = self.__session.request(
                "POST", PPLX_API_ENDPOINT, json=payload, timeout=60)

            if response.ok:
                formatted_response = self._format_response(
                    response)
                if formatted_response["llm_response"]:
                    self.chat_history.append({
                        "role": "assistant",
                        "content": formatted_response["llm_response"]
                    })
                return formatted_response[response_type]
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._raise_error_message(e)

    def close(self) -> None:
        """
        Closes the underlying session.

        This method closes the underlying session which was created when the class was instantiated.
        It is recommended to call this method when you are finished using the class to free up resources.
        """
        self.__session.close()

    @classmethod
    def is_config_valid(cls, config: Dict[str, Union[float, bool, List, str, int]]) -> bool:
        """
        Checks if the provided configuration is valid.

        Parameters:
            config (Dict[str, Union[float, bool, List, str, int]]): The configuration to validate.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        try:
            cls.validate_config(config)
            return True
        except PerplexityConfigError:
            return False

    @classmethod
    def validate_config(cls, config: Dict[str, Union[float, bool, List, str, int]]) -> None:
        """
        Validates the provided configuration dictionary against the default configuration.

        Parameters:
            config (Dict[str, Union[float, bool, List, str, int]]): The configuration dictionary to validate.

        Raises:
            PerplexityConfigError: If a configuration key is invalid or if the type of a configuration value does not match the expected type.
        """
        if not isinstance(config, dict):
            raise TypeError("The configuration must be a dictionary")

        for key, value in config.items():
            if key not in cls.default_config:
                raise PerplexityConfigError(
                    f"Invalid configuration key: {key}")
            if cls.default_config[key] is not None and not isinstance(value, type(cls.default_config[key])):
                raise PerplexityConfigError(
                    f"Invalid configuration value for key: {key}"
                )

    def _validate_and_set_config(self, config: Dict[str, Union[float, bool, List, str, int]]) -> None:
        self.__class__.validate_config(config)
        for key, value in config.items():
            if value != self.default_config.get(key):
                self.__config[key] = value

    def _get_validated_config(self, config: Dict[str, Union[float, bool, List, str, int]]) -> Dict[str, Union[float, bool, List, str, int]]:
        self.__class__.validate_config(config)
        return {key: value for key, value in config.items() if value != self.default_config.get(key)}

    def _raise_error_message(self, e) -> None:
        error_msg = f"Request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            error_msg += f"\nStatus code: {e.response.status_code}"
            try:
                error_msg += f"\nResponse: {e.response.json()}"
            except ValueError:
                error_msg += f"\nResponse: {e.response.text}"

            if status_code == 400:
                raise PerplexityConfigError(error_msg, status_code)

            if status_code == 401:
                raise PerplexityAuthError(error_msg, status_code)

        raise PerplexityAPIError(error_msg, getattr(
            e.response, 'status_code', None)) from e

    def _format_response(self, response: requests.Response) -> dict:
        formatted = {}

        formatted["raw"] = response
        formatted["text"] = response.text

        try:
            json_data = response.json()
            formatted.update({
                "json": json_data,
                "llm_response": json_data["choices"][0]["message"]["content"],
            })
        except (ValueError, KeyError):
            formatted["json"] = None
            formatted["llm_response"] = None

        return formatted

    def _validate_required_params(self, api_key: str = None, model: str = None, system_role: str = None) -> None:
        api_key = api_key or self.auth_token
        model = model or self.model
        system_role = system_role or self.system_role
        if not api_key:
            raise PerplexityAuthError("API key is required")
        if not model:
            raise PerplexityConfigError("Model name is required")
        if not system_role:
            raise PerplexityConfigError("System role is required")
