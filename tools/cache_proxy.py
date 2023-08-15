"""Cache server for Bazel."""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from azure.storage.blob import BlobServiceClient

CONTAINER_NAME = "cache"
blob_service_client = BlobServiceClient.from_connection_string(
    os.environ["BAZEL_CACHE_CONNECTION_STRING"]
)


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward proxy for azure blob storage."""

    def do_GET(self):
        """Retrieve cached artifact from azure blob storage."""
        blob_name = self.path.lstrip("/")
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=blob_name
        )

        try:
            blob_data = blob_client.download_blob()
            self.send_response(200)
            self.send_header("Content-Type", blob_data.content_settings.content_type)
            self.end_headers()
            self.wfile.write(blob_data.readall())
        except Exception as e:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_PUT(self):
        """Put artifacts to azure blob storage."""
        blob_name = self.path.lstrip("/")
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=blob_name
        )
        content_length = int(self.headers["Content-Length"])
        data = self.rfile.read(content_length)

        try:
            blob_client.upload_blob(data, overwrite=True)
            self.send_response(200)
            self.end_headers()
        except Exception as e:
            print("Error uploading blob {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())


def _run_server():
    server_address = ("0.0.0.0", 8080)
    httpd = HTTPServer(server_address, ProxyHandler)
    print("Starting proxy server on port 8080...")
    httpd.serve_forever()


if __name__ == "__main__":
    _run_server()
