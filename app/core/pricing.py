# Fixed price per session length, in pence — the only durations on sale
PRICES = {
    15: 499,   # £4.99
    30: 899,   # £8.99
    45: 1399,  # £13.99
    60: 1999,  # £19.99
}
FREE_INTERVIEW_MINUTES = 10
INVALID_DURATION = "Invalid duration. Choose 15, 30, 45 or 60 minutes."


def price_display(pence: int) -> str:
    return f"£{pence / 100:.2f}"
