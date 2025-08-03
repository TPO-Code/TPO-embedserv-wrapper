import uuid
from typing import Union, List, Dict, Optional, Any

import requests


class EmbedServError(Exception):
    """Custom exception for EmbedServ client errors."""
    pass


class EmbedServ:
    """
    A client for interacting with a running EmbedServ server.

    This client communicates with the server's API, raising an EmbedServError
    for any API or connection-level failures.
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
        Checks if the EmbedServ server is running and reachable.

        Returns:
            True if the server is running, False otherwise.
        """
        try:
            response = self._session.get(self.base_url, timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def list_remote_models(self) -> List[str]:
        """Lists all models available on the server."""
        response = self._request("GET", "models")
        return response.get('data', [])

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Requests the server to pull a model from Hugging Face in the background."""
        payload = {"model": model_name}
        return self._request("POST", "pull", json=payload)

    def delete_remote_model(self, model_name: str) -> Dict[str, Any]:
        """Requests the server to delete a locally stored model."""
        return self._request("DELETE", f"models/{model_name}")

    def unload_model(self) -> Dict[str, Any]:
        """Requests the server to immediately unload the currently active model."""
        return self._request("POST", "unload")

    def create_embeddings(
        self,
        model_name: str,
        input_texts: Union[str, List[str]],
        device: Optional[str] = None,
        keep_alive: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generates embeddings for the given input texts."""
        payload = {"model": model_name, "input": input_texts}
        if device:
            payload["device"] = device
        if keep_alive is not None:
            payload["options"] = {"keep_alive": keep_alive}
        return self._request("POST", "embeddings", json=payload)

    # --- Collection Management ---

    def list_collections(self) -> List[str]:
        """Lists all vector database collections on the server."""
        response = self._request("GET", "db")
        return response.get('collections', [])

    def create_collection(self, collection_name: str) -> None:
        """Creates a new, empty collection on the server."""
        self._request("POST", "db", json={"name": collection_name})

    def delete_collection(self, collection_name: str) -> None:
        """Deletes a collection and all its data from the server."""
        self._request("DELETE", f"db/{collection_name}")

    def count_documents(self, collection_name: str) -> int:
        """Counts the number of documents in a specific collection."""
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
        """Adds documents to a collection."""
        if isinstance(items, str):
            items = [items]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in items]
        if metadatas is None:
            metadatas = [{} for _ in items]  # Default to empty metadata dict
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

    def query(
        self,
        collection_name: str,
        query_texts: Union[str, List[str]],
        model_name: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queries a collection for similar documents."""
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
        """Retrieves documents from a collection by their IDs."""
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
        """Updates documents and/or metadatas for given IDs in a collection."""
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
        """Deletes documents from a collection by their IDs."""
        payload = {"ids": ids}
        self._request("POST", f"db/{collection_name}/delete", json=payload)