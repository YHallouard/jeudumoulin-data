from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from requests import PreparedRequest  # type: ignore[import-untyped]


class CloudflareAccessSettings(BaseSettings):
    cf_access_client_id: str | None = None
    cf_access_client_secret: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


cloudflare_access_settings = CloudflareAccessSettings()


class CloudflareAccessAuth:
    def __init__(self, settings: CloudflareAccessSettings | None = None):
        self._settings = settings if settings is not None else cloudflare_access_settings

    def __call__(self, request: PreparedRequest) -> PreparedRequest:  # type: ignore[no-any-unimported]
        client_id = self._settings.cf_access_client_id
        client_secret = self._settings.cf_access_client_secret
        if not client_id or not client_secret:
            return request
        request.headers["CF-Access-Client-Id"] = client_id
        request.headers["CF-Access-Client-Secret"] = client_secret
        return request


class CloudflareAccessAuthProvider(RequestAuthProvider):
    def get_name(self) -> str:
        return "cloudflare_access"

    def get_auth(self) -> CloudflareAccessAuth:
        return CloudflareAccessAuth()
