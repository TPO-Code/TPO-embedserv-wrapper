## Guide to Using the EmbedServ Python Client

This guide provides a comprehensive overview of the `EmbedServ` Python client, designed for interaction with the EmbedServ local inference server. It covers all functionalities, from server status checks and model management to embedding generation and vector database operations.

### 1. Introduction and Core Concepts

The `EmbedServ` class is a Python client that communicates with an EmbedServ server. EmbedServ is a self-hosted API server created to run sentence-transformer models for generating embeddings. Its primary design principle is to provide a simple, private, and efficient endpoint for AI applications without requiring the application to manage the model lifecycle directly.

#### Key Server Concepts:

*   **Authentication:** The server is intended for use on a local or private network and does **not** require an API key or any other form of authentication.
*   **Server Management:** The `EmbedServ` client does **not** manage the server process itself. The server must be started, configured, and models must be downloaded using a separate Command-Line Interface (CLI).
    *   `embedserv serve`: Starts the server process.
    *   `embedserv pull <model-name>`: Downloads a model from Hugging Face.
    *   `embedserv config set <key> <value>`: Configures server settings, like the model keep-alive time.
*   **Error Handling:** All client-side errors, whether from failed connections or API error responses (e.g., 4xx, 5xx status codes), are raised as a custom `EmbedServError` exception. This exception will contain a descriptive message from the server when available.

### 2. Initialization

To begin interacting with the server, you must first instantiate the `EmbedServ` client.

```python
client = EmbedServ(host="http://127.0.0.1", port=11536, timeout=60)
```

#### `__init__(self, host, port, timeout)`

*   **`host`** (str): The base URL of the EmbedServ server. Defaults to `"http://127.0.0.1"`.
*   **`port`** (int): The port on which the EmbedServ server is listening. Defaults to `11536`.
*   **`timeout`** (int): The default time in seconds to wait for a server response before raising a timeout error. Defaults to `60`.

### 3. Server Status and Health

These methods allow you to monitor the state and availability of the EmbedServ server.

#### `check_server_status()` -> `bool`

Performs a basic check to see if the server is running and reachable at its root URL. This is a simple connectivity test.

*   **Returns:** `True` if the server responds successfully; `False` otherwise (e.g., due to connection errors).

#### `health_check()` -> `bool`

Pings the dedicated `/health` endpoint of the server. This is the recommended method for automated health checks in environments like Docker or Kubernetes.

*   **Returns:** `True` if the server returns a 200 OK status; `False` for any other status or connection error.

#### `get_server_status()` -> `Dict[str, Any]`

Retrieves a detailed, real-time status report from the server.

*   **Returns:** A dictionary containing live information about the server's state.
    *   **Example Response:**
        ```json
        {
            "status": "running",
            "current_model": "all-MiniLM-L6-v2",
            "current_device": "cuda:0",
            "last_used_at": "2023-10-27T10:05:30Z",
            "keep_alive_seconds": 300.0,
            "pending_queue_jobs": 0
        }
        ```

### 4. Model Management

These methods are used to view, download, and manage the models available on the server.

#### `list_remote_models()` -> `List[str]`

Fetches a list of all models that have been downloaded and are available on the server's disk.

*   **Returns:** A list of model name strings (e.g., `['all-MiniLM-L6-v2', 'bge-large-en-v1.5']`).

#### `pull_model(model_name)` -> `Dict[str, Any]`

Sends a request to the server to download a specified model from Hugging Face. This is an **asynchronous** operation; the server accepts the request and performs the download in the background.

*   **`model_name`** (str): The name of the model on Hugging Face (e.g., `'sentence-transformers/all-MiniLM-L6-v2'`).
*   **Returns:** A confirmation dictionary indicating the request was accepted.

#### `delete_remote_model(model_name)` -> `Dict[str, Any]`

Deletes a model from the server's local storage. This action is permanent.

*   **`model_name`** (str): The name of the model to delete.
*   **Returns:** A confirmation dictionary.
*   **Raises:** `EmbedServError` if the specified model is not found on the server.

#### `unload_model()` -> `Dict[str, Any]`

Instructs the server to immediately unload the currently active model from memory (RAM/VRAM). This frees up computational resources but does not delete the model from the disk. The model can be reloaded on the next request.

*   **Returns:** A confirmation dictionary.

### 5. Generating Embeddings

This is the core function for converting text into vector embeddings.

#### `create_embeddings(model_name, input_texts, device=None, keep_alive=None)` -> `Dict[str, Any]`

*   **`model_name`** (str): The model to use for generating the embeddings.
*   **`input_texts`** (Union[str, List[str]]): A single string or a list of strings to be embedded.
*   **`device`** (Optional[str]): The device for the model to run on (e.g., `'cpu'`, `'cuda'`, `'cuda:1'`). If `None`, the server will auto-detect the optimal device.
*   **`keep_alive`** (Optional[int]): A specific duration in seconds to keep this model loaded in memory after the request completes. This value overrides the server's default setting for this specific call.
*   **Returns:** A dictionary structured similarly to the OpenAI API response, containing the embeddings and usage metadata.
    *   **Example Response:**
        ```json
        {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, ...], "index": 0}
            ],
            "model": "all-MiniLM-L6-v2",
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }
        ```

### 6. Vector Database: Collection Management

EmbedServ includes a built-in vector database. These methods manage the collections within that database.

#### `list_collections()` -> `List[str]`

Retrieves the names of all existing collections on the server.

*   **Returns:** A list of collection name strings.

#### `create_collection(collection_name, model_name)` -> `None`

Creates a new, empty collection. A critical aspect of this operation is that the collection is permanently associated with the specified `model_name`. All subsequent operations (adding, querying) on this collection **must** use the same model.

*   **`collection_name`** (str): The unique name for the new collection.
*   **`model_name`** (str): The embedding model to associate with this collection.
*   **Raises:** `EmbedServError` if a collection with the same name already exists.

#### `delete_collection(collection_name)` -> `None`

Permanently deletes an entire collection and all the documents within it.

*   **`collection_name`** (str): The name of the collection to delete.
*   **Raises:** `EmbedServError` if the collection does not exist.

#### `clear_collection(collection_name)` -> `None`

Removes all documents from a collection but does not delete the collection itself. The collection's name and its associated model remain.

*   **`collection_name`** (str): The name of the collection to clear.
*   **Raises:** `EmbedServError` if the collection does not exist.

#### `count_documents(collection_name)` -> `int`

Counts the number of documents currently stored in a specific collection.

*   **`collection_name`** (str): The name of the collection.
*   **Returns:** An integer representing the total count of documents.
*   **Raises:** `EmbedServError` if the collection does not exist.

### 7. Vector Database: Document Operations

These methods are for adding, retrieving, updating, and deleting documents within a collection.

#### `add_to_collection(collection_name, items, model_name, metadatas=None, ids=None, device=None)` -> `None`

Adds one or more documents to a collection. The server will automatically generate the embeddings for the documents using the specified model.

*   **`collection_name`** (str): The target collection's name.
*   **`items`** (Union[str, List[str]]): A single document string or a list of document strings.
*   **`model_name`** (str): The model for embedding. **Must match the model the collection was created with.**
*   **`metadatas`** (Optional[List[Dict]]): A list of metadata dictionaries, one for each document. The server requires that metadata dictionaries not be empty. If this parameter is omitted, the client automatically provides a placeholder `{'source': 'python-client'}` for each item to satisfy this rule.
*   **`ids`** (Optional[List[str]]): A list of unique string IDs for each document. If `None`, the client will generate a UUID for each item.
*   **`device`** (Optional[str]): The device to use for embedding generation (e.g., `'cpu'`, `'cuda'`).
*   **Raises:** `ValueError` if the lengths of `items`, `ids`, and `metadatas` (if provided) are not equal.

#### `add_batch(collection_name, model_name, ids, documents, metadatas, embeddings)` -> `None`

Adds a batch of documents with **pre-computed embeddings**. This is highly efficient for migrating data from another system, as it bypasses the server's embedding generation step.

*   **`collection_name`** (str): The target collection's name.
*   **`model_name`** (str): The name of the model associated with the provided embeddings. This is used for server-side validation.
*   **`ids`** (List[str]): A list of unique string IDs.
*   **`documents`** (List[str]): A list of document texts.
*   **`metadatas`** (List[Dict]): A list of metadata dictionaries.
*   **`embeddings`** (List[List[float]]): A list of the pre-computed embedding vectors.
*   **Raises:** `ValueError` if the lengths of all provided lists are not identical.

#### `query(collection_name, query_texts, model_name, n_results=5, where=None, device=None)` -> `Dict[str, Any]`

Performs a semantic search on a collection to find documents similar to the query text(s).

*   **`collection_name`** (str): The name of the collection to query.
*   **`query_texts`** (Union[str, List[str]]): A single query string or a list of query strings.
*   **`model_name`** (str): The model to use for embedding the query texts. **Must match the collection's model.**
*   **`n_results`** (int): The number of similar documents to return for each query. Defaults to `5`.
*   **`where`** (Optional[Dict]): A dictionary for filtering results based on metadata, following ChromaDB's syntax (e.g., `{"source": "news"}` or `{"pages": {"$gte": 10}}`).
*   **`device`** (Optional[str]): The device to use for embedding the query texts.
*   **Returns:** A dictionary containing the lists of results. Note that each value is a list of lists to support multiple queries.

#### `get_by_ids(collection_name, ids, include=["metadatas", "documents"])` -> `Dict[str, Any]`

Retrieves specific documents from a collection by their unique IDs.

*   **`collection_name`** (str): The name of the collection.
*   **`ids`** (List[str]): A list of document IDs to retrieve.
*   **`include`** (List[str]): Specifies which fields to return. Valid options are: `"documents"`, `"metadatas"`, and `"embeddings"`. Defaults to `["metadatas", "documents"]`.
*   **Returns:** A dictionary containing the requested data.

#### `update_in_collection(collection_name, ids, model_name, documents=None, metadatas=None, device=None)` -> `None`

Updates the document text and/or metadata for existing items in a collection. If `documents` are provided, their embeddings will be re-calculated by the server.

*   **`collection_name`** (str): The name of the collection.
*   **`ids`** (List[str]): The list of IDs for the documents to be updated.
*   **`model_name`** (str): The model to use if re-calculating embeddings. **Must match the collection's model.**
*   **`documents`** (Optional[List[str]]): A list of new document texts, in the same order as `ids`.
*   **`metadatas`** (Optional[List[Dict]]): A list of new metadata dictionaries, in the same order as `ids`.
*   **`device`** (Optional[str]): The device to use if re-calculating embeddings.
*   **Raises:** `ValueError` if both `documents` and `metadatas` are `None`.

#### `delete_from_collection(collection_name, ids)` -> `None`

Deletes specific documents from a collection based on their IDs.

*   **`collection_name`** (str): The name of the collection.
*   **`ids`** (List[str]): A list of document IDs to delete.
*   **Raises:** `EmbedServError` if the collection does not exist.

### 8. Utility Functions

#### `calculate_similarity(embeddings_a, embeddings_b)` -> `List[List[float]]`

Offloads the computation of cosine similarity between two sets of embeddings to the server. This is useful for avoiding a local dependency on libraries like NumPy or SciPy in the client application.

*   **`embeddings_a`** (List[List[float]]): The first list of embedding vectors.
*   **`embeddings_b`** (List[List[float]]): The second list of embedding vectors.
*   **Returns:** A 2D list containing the cosine similarity scores.
### 9. Working example

```python

import uuid
import time
from embedserv_wrapper import EmbedServ, EmbedServError

# --- 1. CONFIGURATION ---
# Define the model and collection names we will use throughout the project.
# The model should be small, fast, and good for general semantic search. [3, 5]
MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = 'fact-checker-db'

# --- 2. INITIALIZE CLIENT ---
# Connect to the local EmbedServ server.
print("--> Initializing EmbedServ client...")
client = EmbedServ(host="http://127.0.0.1", port=11536)


def setup_collection():
    """
    Ensures the vector database collection is ready for use.
    - Checks if the collection exists.
    - If not, it creates the collection.
    - If it exists, it clears any old data to ensure a fresh start.
    - Adds a set of known facts to the collection.
    """
    print(f"\n--> Setting up collection: '{COLLECTION_NAME}'")
    try:
        existing_collections = client.list_collections()
        if COLLECTION_NAME in existing_collections:
            print(f"Collection '{COLLECTION_NAME}' already exists. Clearing it...")
            client.clear_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
            client.create_collection(collection_name=COLLECTION_NAME, model_name=MODEL_NAME)
            print("Collection created successfully.")

        # --- 3. ADDING DATA ---
        print("\n--> Adding known facts to the collection...")
        known_facts = [
            "The Eiffel Tower is located in Paris, France.",
            "The Great Wall of China is the longest wall in the world.",
            "Mount Everest is the Earth's highest mountain above sea level.",
            "The capital of Japan is Tokyo.",
            "Water boils at 100 degrees Celsius at sea level."
        ]
        # Generate unique IDs for our facts
        fact_ids = [str(uuid.uuid4()) for _ in known_facts]

        client.add_to_collection(
            collection_name=COLLECTION_NAME,
            items=known_facts,
            ids=fact_ids,
            model_name=MODEL_NAME
        )
        # Give the server a moment to process the additions
        time.sleep(1)
        doc_count = client.count_documents(COLLECTION_NAME)
        print(f"Successfully added {doc_count} facts.")

    except EmbedServError as e:
        print(f"An error occurred during setup: {e}")
        exit()


def find_closest_fact(query: str):
    """
    Queries the collection to find the most similar fact to the user's query.
    """
    print(f"\n--> Querying for a fact similar to: '{query}'")
    if not query:
        print("Query is empty. Nothing to do.")
        return

    try:
        # --- 4. QUERYING THE COLLECTION ---
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_texts=[query],
            n_results=1,  # We only want the single best match
            model_name=MODEL_NAME
        )

        # The result structure is a dictionary with lists of lists.
        # We need to extract the first document from the first result list.
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        if documents:
            closest_doc = documents[0]
            similarity_score = 1 - distances[0]  # Convert distance to similarity
            print("\n--- Fact Check Result ---")
            print(f"Your Statement: '{query}'")
            print(f"Closest Fact Found: '{closest_doc}'")
            print(f"Similarity Score: {similarity_score:.2f}")
            print("-------------------------")
        else:
            print("No similar facts found in the database.")

    except EmbedServError as e:
        print(f"An error occurred during query: {e}")


def cleanup():
    """
    Deletes the collection from the server to clean up resources.
    """
    print(f"\n--> Cleaning up by deleting collection: '{COLLECTION_NAME}'")
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleanup successful.")
    except EmbedServError as e:
        print(f"An error occurred during cleanup: {e}")


def main():
    """
    Main function to run the fact-checker project.
    """
    # First, check if the server is even running.
    if not client.check_server_status():
        print("EmbedServ server is not running. Please start it first.")
        return

    print("EmbedServ server is running.")

    # Setup the collection and add initial data
    setup_collection()

    # Run a test query
    user_statement = "Where is the Eiffel Tower?"
    find_closest_fact(user_statement)

    # Clean up the created collection
    cleanup()


if __name__ == "__main__":
    main()
```