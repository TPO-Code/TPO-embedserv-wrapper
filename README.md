# embedserv-wrapper

A wrapper for TPO-EmbedServ.

## Installation
Ensure you have embedserv installed on your machine visit [The EmbedServ repository](https://github.com/TPO-Code/TPO-embedserv) for installation instructions

add TODO: ADD REQUIREMENTS TEXT
```bash
uv pip install -r requirements.txt
```

## Usage

```python
client = EmbedServ(host="http://127.0.0.1", port=11536)

try:
    # Check if the server is online
    if client.check_server_status():
        print("Server is running.")

        # Generate embeddings
        model_name = "all-MiniLM-L6-v2"
        texts = ["Hello world!", "This is a test."]
        
        response = client.create_embeddings(model_name, texts)
        
        embeddings = response.get('data', [])
        for text, embedding in zip(texts, embeddings):
            print(f"Text: {text}")
            # Print the first few dimensions of the embedding
            print(f"Embedding (first 5 dims): {embedding[:5]}")

except EmbedServError as e:
    print(f"An error occurred: {e}")
```