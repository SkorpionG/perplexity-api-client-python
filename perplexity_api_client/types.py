from typing import Union
from enum import Enum


class ResponseFormatType(Enum):
    RAW = "raw"
    TEXT = "text"
    JSON = "json"
    LLM_RESPONSE = "llm_response"

    @classmethod
    def validate_response_type(cls, response_type: Union[str, "ResponseFormatType"]) -> None:
        """
        Validates a response_type against the enum values.

        Parameters:
            response_type: the value to validate. Must be a str or ResponseType enum value.

        Raises:
            TypeError: if the argument is not a str or ResponseType enum value.
            ValueError: if the argument is not a valid enum value.

        """
        if not isinstance(response_type, (str, cls)):
            raise TypeError(
                f"Invalid argument type: {type(response_type)}. "
                f"Expected str or {cls.__name__}"
            )
        try:
            cls(response_type)
        except ValueError as exc:
            raise ValueError(
                f"Invalid response_type: {response_type}. "
                f"Valid options are: {[t.value for t in cls]}"
            ) from exc
