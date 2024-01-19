"""Interface for what I need from the OpenAI API."""

from openai import OpenAI

_client = OpenAI()

def embed_string(string: str, model='text-embedding-ada-002') -> list[float]:
    """Get OpenAI embedding of string."""
    text = string.replace('\n', ' ')
    return _client.embeddings.create(input=text, model=model).data[0].embedding
