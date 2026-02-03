def chunk_text(text: str, chunk_size: int = 60000):
    text = text or ""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        if i + chunk_size < len(text):
            cut = chunk.rfind("\n")
            if cut > int(chunk_size * 0.7):
                chunk = chunk[:cut]
                i += cut + 1
            else:
                i += chunk_size
        else:
            i += chunk_size
        chunks.append(chunk)
    return [c for c in chunks if c.strip()]

