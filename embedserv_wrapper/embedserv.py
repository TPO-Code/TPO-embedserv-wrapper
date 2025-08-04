import uuid
from typing import Union, List, Dict, Optional, Any

import requests


class EmbedServError(Exception):
    """Custom exception for EmbedServ client errors."""
    pass


class EmbedServ:
    """
    A Python client for the EmbedServ local inference server.

    EmbedServ is a self-hosted API server designed to run sentence-transformer
    models on consumer hardware. It provides a simple, private endpoint for
    generating embeddings and managing vector database collections without
    needing to handle the model lifecycle in your application code.

    This client is the primary way to interact with the server from a Python
    application.

    **Important Server Context:**
    - **No API Key Needed:** The server is intended for local/private network
      use and does not require authentication.
    - **Server and Model Management:** The server process and model management
      are handled via a separate Command-Line Interface (CLI). You should use
      the CLI for initial setup and configuration. For example:
        - `embedserv serve`: To start the server.
        - `embedserv pull all-MiniLM-L6-v2`: To download models from Hugging Face.
        - `embedserv config set keep_alive 300`: To set the default time (in
          seconds) a model stays loaded in memory after a request.

    This client handles all communication with the server's REST API and will
    raise an `EmbedServError` for any API-level or connection-level failures.

    Example:
        # In your terminal, start the server and pull a model:
        # > embedserv serve &
        # > embedserv pull all-MiniLM-L6-v2

        # In your Python code:
        from embedserv.client import EmbedServ, EmbedServError

        client = EmbedServ(host="http://127.0.0.1", port=11536)

        try:
            if client.check_server_status():
                print("Server is running.")
                embeddings = client.create_embeddings(
                    model_name='all-MiniLM-L6-v2',
                    input_texts=["Hello world!"]
                )
                print("Generated embeddings successfully.")
            else:
                print("EmbedServ server is not reachable.")
        except EmbedServError as e:
            print(f"An API error occurred: {e}")

    """

    def __init__(self, host: str = "http://127.0.0.1", port: int = 11536, timeout: int = 60):
        """
        Initializes the EmbedServ client.

        Args:
            host: The server's host URL.
            port: The server's port.
            timeout: Default request timeout in seconds.
        """
        self.base_url = f"{host}:{port}"
        self.api_v1_url = f"{self.base_url}/api/v1"
        self._session = requests.Session()
        self.timeout = timeout

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Internal method to handle all API requests.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            endpoint: API endpoint path (e.g., 'models', 'db/my_collection/add').
            **kwargs: Additional arguments for the requests library (json, params, etc.).

        Returns:
            The JSON response from the server as a dictionary.

        Raises:
            EmbedServError: If the API returns an error or a connection fails.
        """
        url = f"{self.api_v1_url}/{endpoint}"
        try:
            response = self._session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
            # Handle 204 No Content, which has an empty body
            return response.json() if response.status_code != 204 and response.content else {}
        except requests.exceptions.HTTPError as e:
            # Extract server-provided error detail if possible
            error_detail = e.response.json().get('detail', e.response.text)
            raise EmbedServError(f"API Error on {method} {url} ({e.response.status_code}): {error_detail}") from e
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            raise EmbedServError(f"Connection Error on {method} {url}: {e}") from e

    def check_server_status(self) -> bool:
        """
        Checks if the EmbedServ server is running and reachable at its root URL.

        Returns:
            True if the server is running, False otherwise.
        """
        try:
            response = self._session.get(self.base_url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def health_check(self) -> bool:
        """
        Pings the server's dedicated health check endpoint.

        This is the recommended method for automated health checks (e.g., in Docker or Kubernetes).

        Returns:
            True if the server returns a healthy status (200 OK), False otherwise.
        """
        try:
            # Note: The /health endpoint is not under /api/v1
            response = self._session.get(f"{self.base_url}/health", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """
        Retrieves a detailed, live status report from the server.

        Returns:
            A dictionary containing information about the server's state,
            such as the currently loaded model, device, and pending jobs.
            Example:
            {
                "status": "running",
                "current_model": "all-MiniLM-L6-v2",
                "current_device": "cuda:0",
                "last_used_at": "2023-10-27T10:05:30Z",
                "keep_alive_seconds": 300.0,
                "pending_queue_jobs": 0
            }
        """
        return self._request("GET", "status")



    def list_remote_models(self) -> List[str]:
        """
        Lists all models that have been downloaded and are available on the server.

        Returns:
            A list of model name strings (e.g., ['all-MiniLM-L6-v2', 'bge-large-en-v1.5']).
        """
        response = self._request("GET", "models")
        return response.get('data', [])

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Requests the server to download a model from Hugging Face in the background.
        This operation is asynchronous on the server.

        Args:
            model_name: The name of the model to pull from Hugging Face (e.g., 'all-MiniLM-L6-v2').

        Returns:
            A dictionary confirming the request was accepted.
            Example: {'status': 'accepted', 'message': 'Pull request for model ... accepted...'}
        """
        payload = {"model": model_name}
        return self._request("POST", "pull", json=payload)

    def delete_remote_model(self, model_name: str) -> Dict[str, Any]:
        """
        Requests the server to delete a locally stored model from its disk.

        Args:
            model_name: The Hugging Face name of the model to delete.

        Returns:
            A dictionary confirming the deletion.
            Example: {'status': 'success', 'message': 'Model ... deleted.'}

        Raises:
            EmbedServError: If the model is not found on the server (404).
        """
        return self._request("DELETE", f"models/{model_name}")

    def unload_model(self) -> Dict[str, Any]:
        """
        Requests the server to immediately unload the currently active model from memory (VRAM/RAM).
        This frees up resources but does not delete the model from disk.

        Returns:
            A dictionary confirming the model was unloaded.
        """
        return self._request("POST", "unload")

    def create_embeddings(
        self,
        model_name: str,
        input_texts: Union[str, List[str]],
        device: Optional[str] = None,
        keep_alive: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates embeddings for the given input texts.

        Args:
            model_name: The model to use for generating embeddings.
            input_texts: A single string or a list of strings to embed.
            device: The device to run the model on (e.g., 'cpu', 'cuda', 'cuda:1').
                    If None, the server will auto-detect the best available device.
            keep_alive: Time in seconds to keep this model loaded in memory after the request.
                        Overrides the server's default keep-alive duration.

        Returns:
            A dictionary containing the embedding data, structured according to the OpenAI API format.
            Example:
            {
                'object': 'list',
                'data': [
                    {'object': 'embedding', 'embedding': [0.1, ...], 'index': 0},
                    ...
                ],
                'model': 'all-MiniLM-L6-v2',
                'usage': {'prompt_tokens': 0, 'total_tokens': 0}
            }
        """
        payload = {"model": model_name, "input": input_texts}
        if device:
            payload["device"] = device
        if keep_alive is not None:
            payload["options"] = {"keep_alive": keep_alive}
        return self._request("POST", "embeddings", json=payload)

    # --- Collection Management ---

    def list_collections(self) -> List[str]:
        """
        Lists all vector database collections on the server.

        Returns:
            A list of collection name strings.
        """
        response = self._request("GET", "db")
        collections_info = response.get('collections', [])
        # Extract just the name from each dictionary
        return [c['name'] for c in collections_info]

    def create_collection(self, collection_name: str, model_name: str) -> None:
        """
        Creates a new, empty collection on the server and permanently
        associates it with a specific embedding model.

        This is a critical safeguard: once a collection is created with a model,
        all subsequent add, query, or update operations for that collection
        must use the same model.

        Args:
            collection_name: The name for the new collection.
            model_name: The name of the model to associate with this collection
                        (e.g., 'sentence-transformers/all-MiniLM-L6-v2').

        Raises:
            EmbedServError: If a collection with the same name already exists (409).
        """
        # The payload now includes both the name and the model
        payload = {"name": collection_name, "model": model_name}
        self._request("POST", "db", json=payload)


    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and all its data from the server.

        Args:
            collection_name: The name of the collection to delete.

        Raises:
            EmbedServError: If the collection does not exist (404).
        """
        self._request("DELETE", f"db/{collection_name}")

    def clear_collection(self, collection_name: str) -> None:
        """
        Deletes all documents from a collection but leaves the collection intact.

        Args:
            collection_name: The name of the collection to clear.

        Raises:
            EmbedServError: If the collection does not exist (404).
        """
        self._request("POST", f"db/{collection_name}/clear")


    def count_documents(self, collection_name: str) -> int:
        """
        Counts the number of documents in a specific collection.

        Args:
            collection_name: The name of the collection.

        Returns:
            The integer count of documents in the collection.

        Raises:
            EmbedServError: If the collection does not exist (404).
        """
        response = self._request("GET", f"db/{collection_name}/count")
        return response.get('count', 0)

    # --- Document Operations ---

    def add_to_collection(
        self,
        collection_name: str,
        items: Union[str, List[str]],
        model_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Adds documents to a collection. The server will generate embeddings.

        Args:
            collection_name: The name of the target collection.
            items: A single document string or a list of document strings.
            model_name: The model to use for embedding the documents. Must match the
                        collection's associated model.
            metadatas: An optional list of non-empty metadata dictionaries.
                       **Crucially, the server requires all metadata to be non-empty.**
                       If this argument is omitted, the client will automatically
                       provide a placeholder dictionary (`{'source': 'python-client'}`)
                       for each item to satisfy this server requirement.
            ids: An optional list of unique string IDs. If not provided, UUIDs are generated.
            device: The device to run the embedding model on (e.g., 'cpu', 'cuda').

        Raises:
            ValueError: If the number of items, ids, and metadatas do not match.
            EmbedServError: If the server returns an error.
        """
        if isinstance(items, str):
            items = [items]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in items]

            # --- FIX APPLIED HERE ---
            # The original code generated [{}], which the server rejects.
            # This now generates a valid, non-empty placeholder if no metadata is provided.
        if metadatas is None:
            metadatas = [{"source": "python-client"}] * len(items)

        if not (len(items) == len(ids) == len(metadatas)):
            raise ValueError("The number of items, ids, and metadatas must be the same.")

        payload = {
            "documents": items,
            "metadatas": metadatas,
            "ids": ids,
            "model": model_name
        }
        if device:
            payload["device"] = device
        self._request("POST", f"db/{collection_name}/add", json=payload)

    def add_batch(
            self,
            collection_name: str,
            model_name: str,  # <-- Add this required parameter
            ids: List[str],
            documents: List[str],
            metadatas: List[Dict[str, Any]],
            embeddings: List[List[float]],
    ) -> None:
        """
        Adds a batch of documents with pre-computed embeddings to a collection.

        This method is ideal for migrating an existing database, as it bypasses
        on-the-fly embedding generation by the server.

        Args:
            collection_name: The name of the target collection.
            model_name: The name of the model associated with the provided embeddings.
                    This is required for server-side validation.
            ids: A list of unique string IDs for each document.
            documents: A list of document strings.
            metadatas: A list of metadata dictionaries.
            embeddings: A list of embedding vectors.

        Raises:
            ValueError: If the lengths of ids, documents, metadatas, and embeddings do not match.
            EmbedServError: If the collection does not exist (404).
        """
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("The number of items in ids, documents, metadatas, and embeddings must be the same.")

        payload = {
            "model": model_name,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }
        self._request("POST", f"db/{collection_name}/batch-add", json=payload)

    def query(
        self,
        collection_name: str,
        query_texts: Union[str, List[str]],
        model_name: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Queries a collection for documents semantically similar to the query texts.

        Args:
            collection_name: The name of the collection to query.
            query_texts: A single query string or a list of query strings.
            model_name: The model to use for embedding the query texts.
            n_results: The number of results to return for each query.
            where: A dictionary for metadata filtering, following ChromaDB's syntax.
                   Example: {"source": "news"} or {"pages": {"$gte": 10}}.
            device: The device to run the embedding model on (e.g., 'cpu', 'cuda').

        Returns:
            A dictionary containing the query results, structured like ChromaDB's response.
            Example for a single query:
            {
                'ids': [['id3', 'id9']],
                'distances': [[0.12, 0.23]],
                'metadatas': [[{'source': 'web'}, {'source': 'pdf'}]],
                'documents': [['doc text 3', 'doc text 9']],
                'embeddings': None
            }
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        payload = {
            "query_texts": query_texts,
            "n_results": n_results,
            "model": model_name
        }
        if where:
            payload["where"] = where
        if device:
            payload["device"] = device
        response = self._request("POST", f"db/{collection_name}/query", json=payload)
        return response.get('results', {})

    def get_by_ids(
        self,
        collection_name: str,
        ids: List[str],
        include: List[str] = ["metadatas", "documents"]
    ) -> Dict[str, Any]:
        """
        Retrieves documents from a collection by their unique IDs.

        Args:
            collection_name: The name of the collection.
            ids: A list of document IDs to retrieve.
            include: A list of fields to include in the response. Follows ChromaDB's
                     syntax. Valid options are: "documents", "metadatas", "embeddings".

        Returns:
            A dictionary containing the requested data.
            Example:
            {
                'ids': ['id1', 'id2'],
                'documents': ['doc text 1', 'doc text 2'],
                'metadatas': [{'source': 'a'}, {'source': 'b'}],
                'embeddings': None
            }

        Raises:
            EmbedServError: If the collection does not exist (404).
        """
        payload = {"ids": ids, "include": include}
        return self._request("POST", f"db/{collection_name}/get", json=payload)

    def update_in_collection(
        self,
        collection_name: str,
        ids: List[str],
        model_name: str,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Updates documents and/or metadatas for given IDs in a collection.
        If 'documents' are provided, their embeddings will be re-calculated.

        Args:
            collection_name: The name of the collection.
            ids: The list of IDs of the documents to update.
            model_name: The model to use if embeddings need to be re-calculated.
            documents: An optional list of new document texts. Must match the order of 'ids'.
            metadatas: An optional list of new metadata dictionaries. Must match the order of 'ids'.
            device: The device to run the model on if re-calculating embeddings.

        Raises:
            ValueError: If neither 'documents' nor 'metadatas' is provided.
            EmbedServError: If the collection does not exist (404).
        """
        if documents is None and metadatas is None:
            raise ValueError("You must provide either 'documents' or 'metadatas' to update.")
        payload = {"ids": ids, "model": model_name}
        if documents:
            payload["documents"] = documents
        if metadatas:
            payload["metadatas"] = metadatas
        if device:
            payload["device"] = device
        self._request("POST", f"db/{collection_name}/update", json=payload)

    def delete_from_collection(self, collection_name: str, ids: List[str]) -> None:
        """
        Deletes documents from a collection by their IDs.

        Args:
            collection_name: The name of the collection.
            ids: A list of IDs of the documents to delete.

        Raises:
            EmbedServError: If the collection does not exist (404).
        """
        payload = {"ids": ids}
        self._request("POST", f"db/{collection_name}/delete", json=payload)

    def calculate_similarity(
            self,
            embeddings_a: List[List[float]],
            embeddings_b: List[List[float]]
    ) -> List[List[float]]:
        """
        Calculates cosine similarity between two sets of embeddings using the server.

        This offloads the calculation to the server, removing the need for a local
        NumPy or sentence-transformers dependency in the client application.

        Args:
            embeddings_a: The first list of embedding vectors.
            embeddings_b: The second list of embedding vectors.

        Returns:
            A 2D list containing the cosine similarity scores between
            each embedding in `embeddings_a` and `embeddings_b`.

        Raises:
            EmbedServError: If the server encounters an error during calculation.
        """
        payload = {
            "embeddings_a": embeddings_a,
            "embeddings_b": embeddings_b
        }
        response = self._request("POST", "similarity", json=payload)
        return response.get('similarity_scores', [])