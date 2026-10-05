"""Brevo transactional email sender — ported from Learnify's BrevoEmailSender."""
import asyncio
import re
from dataclasses import dataclass, field

import httpx

BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass
class Recipient:
    email: str
    name: str | None = None

    def to_dict(self) -> dict:
        return {"email": self.email, **({"name": self.name} if self.name else {})}


@dataclass
class SendParams:
    to: list[Recipient]
    sender: Recipient
    subject: str
    html_content: str | None = None
    text_content: str | None = None


@dataclass
class SendResult:
    success: bool
    message_id: str | None = None
    error: str | None = None
    status: int | None = None
    retry_count: int = 0


@dataclass
class BrevoEmailSender:
    api_key: str
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 15.0
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False)  # for tests

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Brevo API key is required")

    def _validate(self, params: SendParams) -> None:
        if not params.to:
            raise ValueError("At least one email recipient is required")
        if not EMAIL_PATTERN.match(params.sender.email):
            raise ValueError("Invalid sender email address")
        if any(not EMAIL_PATTERN.match(r.email) for r in params.to):
            raise ValueError("Invalid recipient email address")
        if not params.subject.strip():
            raise ValueError("Email subject is required")
        if not params.html_content and not params.text_content:
            raise ValueError("Email content is required")

    async def send_email(self, params: SendParams) -> SendResult:
        self._validate(params)
        body = {
            "sender":  params.sender.to_dict(),
            "to":      [r.to_dict() for r in params.to],
            "subject": params.subject,
            **({"htmlContent": params.html_content} if params.html_content else {}),
            **({"textContent": params.text_content} if params.text_content else {}),
        }
        headers = {"accept": "application/json", "api-key": self.api_key, "content-type": "application/json"}

        retry_count = 0
        async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
            while True:
                try:
                    response = await client.post(BREVO_ENDPOINT, json=body, headers=headers)
                    data = response.json() if response.content else {}
                    if response.is_success:
                        return SendResult(success=True, message_id=data.get("messageId"), retry_count=retry_count)
                    error = ": ".join(filter(None, [data.get("code"), data.get("message")])) or f"HTTP {response.status_code}"
                    # Retry only server-side failures; 4xx means the request itself is wrong
                    if response.status_code < 500 or retry_count >= self.max_retries:
                        return SendResult(success=False, error=error, status=response.status_code, retry_count=retry_count)
                except (httpx.HTTPError, ValueError) as e:
                    if retry_count >= self.max_retries:
                        return SendResult(success=False, error=str(e) or type(e).__name__, retry_count=retry_count)
                retry_count += 1
                await asyncio.sleep(self.retry_delay_seconds * retry_count)
