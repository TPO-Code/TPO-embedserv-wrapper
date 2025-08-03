
# EmbedServ Wrapper 🐍

<p align="center">
  <strong>A simple and intuitive Python wrapper for the <a href="https://github.com/TPO-Code/TPO-embedserv">TPO-EmbedServ</a> high-performance embedding server.</strong>
</p>

---

## ✨ About The Project

[TPO-EmbedServ](https://github.com/TPO-Code/TPO-embedserv) is a powerful, self-hosted server designed for fast and efficient text embedding generation. This wrapper provides a clean, Pythonic interface to the EmbedServ API, making it easy to integrate embedding generation into your Python applications without dealing with raw HTTP requests.

### Key Features
*   🚀 Simple, object-oriented interface.
*   🌐 Check server health and status.
*   ✍️ Generate text embeddings for single or multiple inputs.
*   🔌 Handles all communication and request formatting.
*   ⚠️ Raises a custom `EmbedServError` for clear error handling.

##  Prerequisites

Before using this wrapper, you must have an instance of `TPO-EmbedServ` installed and running on your machine or network.

*   **For installation instructions, please visit the official [EmbedServ repository](https://github.com/TPO-Code/TPO-embedserv).**

## 📦 Installation

You can install the wrapper directly from the GitHub repository using `pip` (or your preferred package manager like `uv`).

### Option 1: Direct Install

Run the following command in your terminal:

```bash
pip install git+https://github.com/TPO-Code/TPO-embedserv-wrapper.git
```
*(For `uv` users: `uv pip install git+https://github.com/TPO-Code/TPO-embedserv-wrapper.git`)*

### Option 2: Using `requirements.txt`

1.  Add the following line to your `requirements.txt` file:

    ```text
    # requirements.txt
    embedserv_wrapper @ git+https://github.com/TPO-Code/TPO-embedserv-wrapper.git
    ```

2.  Install the dependencies from your `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```
    *(For `uv` users: `uv pip install -r requirements.txt`)*

## 🚀 Usage

Here is a basic example of how to use the `embedserv-wrapper` to connect to the server and generate embeddings.

### Example Code

```python
from embedserv_wrapper import EmbedServ, EmbedServError

# Initialize the client, pointing to your EmbedServ instance
# Default host is "http://127.0.0.1" and port is 11536
client = EmbedServ(host="http://127.0.0.1", port=11536)

try:
    # 1. Check if the server is online
    if client.check_server_status():
        print("✅ Server is running.")

        # 2. Define the model and texts for embedding
        model_name = "all-MiniLM-L6-v2"
        texts_to_embed = ["Hello world!", "This is a test."]
        
        print(f"\nGenerating embeddings for {len(texts_to_embed)} texts using model '{model_name}'...")
        
        # 3. Generate embeddings
        response = client.create_embeddings(model_name, texts_to_embed)
        
        # 4. Process the results
        embeddings = response.get('data', [])
        for text, embedding in zip(texts_to_embed, embeddings):
            print(f"\nText: {text}")
            # Print the first 5 dimensions of the embedding for brevity
            print(f"Embedding (first 5 dims): {embedding[:5]}")

    else:
        print("❌ Server is not responding.")

except EmbedServError as e:
    print(f"An error occurred: {e}")

```

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.
