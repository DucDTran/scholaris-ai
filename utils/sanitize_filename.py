import re

def sanitize_filename(filename):
    """Removes special characters to create a valid directory name."""
    s = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    return s[:100]  # Truncate to a reasonable length