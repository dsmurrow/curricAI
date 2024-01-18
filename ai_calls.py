from openai import OpenAI

_client = OpenAI()

def embed_string(string, model='text-embedding-ada-002'):
    text = string.replace('\n', ' ')
    return _client.embeddings.create(input=text, model=model).data[0].embedding