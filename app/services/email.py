"""Transactional email — Brevo delivery with Interquis templates (structure adapted from Learnify)."""
import html as html_lib
import logging
import re
from datetime import datetime

from app.config import get_settings
from app.core.pricing import FREE_INTERVIEW_MINUTES
from app.services.brevo_sender import BrevoEmailSender, Recipient, SendParams

logger = logging.getLogger(__name__)
settings = get_settings()

_sender: BrevoEmailSender | None = None

# Every email takes an optional app_name so another app on this backend (e.g.
# Tranquis) can send under its own name. None means the default brand, with the
# sender name from BREVO_FROM_NAME — exactly today's emails.
DEFAULT_APP_NAME = "Interquis"


def _brand(app_name: str | None) -> str:
    return app_name or DEFAULT_APP_NAME


def _wordmark(app_name: str | None) -> str:
    if _brand(app_name) == DEFAULT_APP_NAME:
        return 'Inter<span style="color:#6366f1">quis</span>'
    return escape_html(_brand(app_name))


def email_configured() -> bool:
    return bool(settings.brevo_api_key and settings.brevo_from_email)


def escape_html(text: str) -> str:
    return html_lib.escape(text, quote=True)


def plain_text(html: str) -> str:
    text = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def base_template(content: str, preheader: str = "", app_name: str | None = None) -> str:
    brand = escape_html(_brand(app_name))
    app = escape_html(settings.frontend_url)
    year = datetime.now().year
    hidden = (
        f'<div style="display:none;font-size:1px;color:#f3f5f9;line-height:1px;max-height:0;max-width:0;'
        f'opacity:0;overflow:hidden;mso-hide:all">{escape_html(preheader)}&nbsp;&zwnj;&nbsp;&zwnj;</div>'
        if preheader else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="color-scheme" content="light" />
  <title>{brand}</title>
  <style>
    body, table, td, a {{ -webkit-text-size-adjust:100%; -ms-text-size-adjust:100%; }}
    table {{ border-collapse:collapse !important; }}
    a {{ text-decoration:none; }}
    @media only screen and (max-width:620px) {{
      .email-shell {{ padding:20px 10px !important; }}
      .email-body {{ padding:28px 22px 30px !important; }}
      .email-header, .email-footer {{ padding:20px 22px !important; }}
      .email-button, .email-button a {{ width:100% !important; box-sizing:border-box !important; text-align:center !important; }}
    }}
  </style>
</head>
<body style="margin:0;padding:0;background:#f3f5f9;font-family:Arial,'Helvetica Neue',sans-serif;color:#172033">
  {hidden}
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#f3f5f9">
    <tr><td align="center">
      <table role="presentation" class="email-shell" width="100%" cellpadding="0" cellspacing="0" border="0" style="padding:44px 20px">
        <tr><td align="center">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" bgcolor="#ffffff" style="background:#ffffff;border-radius:24px;border:1px solid #e2e8f0;overflow:hidden;max-width:600px;width:100%">
            <tr><td height="5" bgcolor="#6366f1" style="height:5px;line-height:5px;font-size:0">&nbsp;</td></tr>
            <tr>
              <td class="email-header" style="padding:25px 36px 23px;border-bottom:1px solid #eef2f7">
                <div style="font-size:20px;line-height:24px;font-weight:800;letter-spacing:-0.02em;color:#111827">{_wordmark(app_name)}</div>
                <div style="font-size:11px;line-height:16px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8">Practice interviews. Land the job.</div>
              </td>
            </tr>
            <tr><td class="email-body" style="padding:38px 36px 40px">{content}</td></tr>
            <tr>
              <td class="email-footer" style="padding:22px 36px 24px;border-top:1px solid #eef2f7;background:#f8fafc">
                <p style="margin:0;font-size:11px;line-height:17px;color:#94a3b8;text-align:center">
                  &copy; {year} {brand}&nbsp;&nbsp;·&nbsp;&nbsp;<a href="{app}/privacy" style="color:#64748b">Privacy</a>
                  &nbsp;&nbsp;·&nbsp;&nbsp;<a href="{app}/terms" style="color:#64748b">Terms</a>
                  <br />This transactional email was sent to your registered address.
                </p>
              </td>
            </tr>
          </table>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def btn(text: str, url: str) -> str:
    return (
        '<table role="presentation" class="email-button" cellpadding="0" cellspacing="0" border="0" style="margin:24px 0 4px">'
        '<tr><td align="center" bgcolor="#4f46e5" style="border-radius:12px">'
        f'<a href="{escape_html(url)}" style="display:inline-block;border:1px solid #4f46e5;border-radius:12px;color:#ffffff;'
        f'padding:13px 24px;font-size:14px;line-height:20px;font-weight:700;text-decoration:none">{escape_html(text)}</a>'
        "</td></tr></table>"
    )


def h1(text: str) -> str:
    return f'<h1 style="margin:0 0 12px;font-size:27px;line-height:1.2;font-weight:800;letter-spacing:-0.025em;color:#111827">{text}</h1>'


def p(text: str) -> str:
    return f'<p style="margin:0 0 17px;font-size:15px;line-height:1.68;color:#475569">{text}</p>'


async def send_email(to: str, subject: str, html: str, app_name: str | None = None) -> bool:
    """Send one email. Never raises — a failed email must not break the request that triggered it."""
    global _sender
    if not email_configured():
        logger.warning("Email not configured (BREVO_API_KEY / BREVO_FROM_EMAIL) — not sending %r to %s", subject, to)
        return False
    try:
        _sender = _sender or BrevoEmailSender(settings.brevo_api_key)
        result = await _sender.send_email(SendParams(
            to=[Recipient(to)],
            sender=Recipient(settings.brevo_from_email, app_name or settings.brevo_from_name),
            subject=subject,
            html_content=html,
            text_content=plain_text(html),
        ))
    except Exception:
        logger.exception("Email send failed: %r to %s", subject, to)
        return False
    if not result.success:
        logger.error("Brevo rejected email %r to %s: %s (after %d retries)", subject, to, result.error, result.retry_count)
        return False
    logger.info("Email sent: %r to %s (message %s)", subject, to, result.message_id)
    return True


def verification_url(token: str) -> str:
    return f"{settings.frontend_url}/verify-email?token={token}"


async def send_verification_email(to: str, name: str, token: str, app_name: str | None = None) -> bool:
    url = verification_url(token)
    if not email_configured():
        # Local development without Brevo: log the link so the flow can still be completed
        logger.warning("Email not configured — verification link for %s: %s", to, url)
        return False
    first = escape_html(name.split()[0] if name.strip() else "there")
    brand = _brand(app_name)
    article = "an" if brand[0].lower() in "aeiou" else "a"
    html = base_template(
        h1("Confirm your email")
        + p(f"Hi {first}, thanks for joining {escape_html(brand)}.")
        + p(f"Confirm your email address to unlock your <strong>free {FREE_INTERVIEW_MINUTES}-minute interview</strong> with Alex, our AI interviewer.")
        + btn("Confirm email address", url)
        + p(f'<span style="font-size:13px;color:#94a3b8">This link expires in {settings.email_verify_ttl_hours} hours. '
            f"If you didn't create {article} {escape_html(brand)} account, you can ignore this email.</span>"),
        preheader="Confirm your email to unlock your free interview",
        app_name=app_name,
    )
    return await send_email(to, f"Confirm your email for {brand}", html, app_name=app_name)
