class PerplexityError(Exception):
    """Base exception class for Perplexity API errors"""
    pass


class PerplexityAPIError(PerplexityError):
    """Raised when API returns an error response"""

    def __init__(self, message: str, status_code: int = None, response: dict = None) -> None:
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class PerplexityAuthError(PerplexityError):
    """Raised when authentication fails"""
    pass


class PerplexityConfigError(PerplexityError):
    """Raised when configuration is invalid"""
    pass
