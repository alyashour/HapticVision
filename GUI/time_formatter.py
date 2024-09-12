from time import strftime, localtime

def format_duration(seconds) -> str:
    """Convert seconds into a human-readable format, using appropriate units."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes, sec = divmod(seconds, 60)
        return f"{int(minutes)} minutes"
    elif seconds < 86400:
        hours, remainder = divmod(seconds, 3600)
        minutes, sec = divmod(remainder, 60)
        return f"{int(hours)} hours, {int(minutes)} minutes"
    else:
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, sec = divmod(remainder, 60)
        return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(sec)} seconds"

def clock() -> str:
    return strftime("%I:%M:%S", localtime())