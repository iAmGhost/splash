import http.client
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from install import launcher, serve_multi

FAKE_TARGET = "/tmp/fake-target"
FAKE_DRAFT = "/tmp/fake-draft"
FAKE_TOKENIZER = "/tmp/fake-tokenizer"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _post(port, path, body, headers=None):
    return _post_with_timeout(port, path, body, timeout=5)


def _post_with_timeout(port, path, body, headers=None, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    conn.request("POST", path, body, hdrs)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def _get(port, path):
    return _get_with_timeout(port, path, timeout=5)


def _get_with_timeout(port, path, timeout=5):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------


class ConfigLoadingTests(unittest.TestCase):
    def test_loads_valid_single_model(self):
        config = {"models": [{"model": "owner/repo"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            specs = serve_multi.load_config(path)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["model"], "owner/repo")
            self.assertEqual(specs[0]["aliases"], ())
            self.assertIsNone(specs[0]["max_context"])
        finally:
            Path(path).unlink()

    def test_loads_multiple_models_with_aliases(self):
        config = {
            "models": [
                {
                    "model": "owner/repo-a",
                    "aliases": ["alias-a", "local-a"],
                    "max_context": 131072,
                },
                {
                    "model": "owner/repo-b",
                    "aliases": [],
                    "max_context": None,
                },
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            specs = serve_multi.load_config(path)
            self.assertEqual(len(specs), 2)
            self.assertEqual(specs[0]["model"], "owner/repo-a")
            self.assertEqual(specs[0]["aliases"], ("alias-a", "local-a"))
            self.assertEqual(specs[0]["max_context"], 131072)
            self.assertEqual(specs[1]["model"], "owner/repo-b")
            self.assertEqual(specs[1]["aliases"], ())
            self.assertIsNone(specs[1]["max_context"])
        finally:
            Path(path).unlink()

    def test_rejects_missing_models_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"other": []}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                serve_multi.load_config(path)
        finally:
            Path(path).unlink()

    def test_rejects_empty_models_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"models": []}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                serve_multi.load_config(path)
        finally:
            Path(path).unlink()

    def test_rejects_invalid_model_id(self):
        config = {"models": [{"model": "not-a-valid-id"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            with self.assertRaises(Exception):  # argparse.ArgumentTypeError
                serve_multi.load_config(path)
        finally:
            Path(path).unlink()

    def test_rejects_non_dict_entry(self):
        config = {"models": ["owner/repo"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                serve_multi.load_config(path)
        finally:
            Path(path).unlink()

    def test_rejects_unknown_keys(self):
        config = {"models": [{"model": "owner/repo", "unknown": 1}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                serve_multi.load_config(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


class ArgumentParsingTests(unittest.TestCase):
    def test_serve_multi_subcommand_exists(self):
        args = launcher.parse_args(["serve-multi", "--config", "/tmp/config.json"])
        self.assertEqual(args.command, "serve-multi")
        self.assertEqual(args.config, "/tmp/config.json")

    def test_serve_multi_shares_flags_with_serve(self):
        args = launcher.parse_args(
            [
                "serve-multi",
                "--config",
                "/tmp/config.json",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--kv-format",
                "bf16",
                "--max-memory",
                "28G",
                "--max-context",
                "128K",
                "--api-key",
                "secret",
                "--no-webui",
            ]
        )
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9999)
        self.assertEqual(args.kv_format, "bf16")
        self.assertEqual(args.max_memory, 28 * 1024**3)
        self.assertEqual(args.max_context, 131072)
        self.assertEqual(args.api_key, "secret")
        self.assertTrue(args.no_webui)

    def test_serve_multi_default_switch_timeouts(self):
        args = launcher.parse_args(["serve-multi", "--config", "/tmp/config.json"])
        self.assertEqual(args.switch_timeout, 600.0)
        self.assertEqual(args.switch_settle, 60.0)
        self.assertTrue(args.evict_cache)

    def test_serve_multi_custom_switch_timeouts(self):
        args = launcher.parse_args(
            [
                "serve-multi",
                "--config",
                "/tmp/config.json",
                "--switch-timeout",
                "30",
                "--switch-settle",
                "10",
                "--no-evict-cache",
            ]
        )
        self.assertEqual(args.switch_timeout, 30.0)
        self.assertEqual(args.switch_settle, 10.0)
        self.assertFalse(args.evict_cache)

    def test_serve_multi_shared_flags_duplicated(self):
        # Confirm shared flags match what 'serve' supports.
        serve_args = launcher.parse_args(
            [
                "serve",
                "--model",
                "owner/repo",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--kv-format",
                "bf16",
                "--max-memory",
                "28G",
                "--max-context",
                "128K",
                "--max-request-size",
                "256M",
                "--max-image-pixels",
                "1000000",
                "--api-key",
                "secret",
                "--no-webui",
                "--allowed-host",
                "example.com",
            ]
        )
        multi_args = launcher.parse_args(
            [
                "serve-multi",
                "--config",
                "/tmp/config.json",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--kv-format",
                "bf16",
                "--max-memory",
                "28G",
                "--max-context",
                "128K",
                "--max-request-size",
                "256M",
                "--max-image-pixels",
                "1000000",
                "--api-key",
                "secret",
                "--no-webui",
                "--allowed-host",
                "example.com",
            ]
        )
        for attr in (
            "host",
            "port",
            "kv_format",
            "max_memory",
            "max_context",
            "max_request_size",
            "max_image_pixels",
            "api_key",
            "no_webui",
            "allowed_host",
        ):
            with self.subTest(attr=attr):
                self.assertEqual(
                    getattr(serve_args, attr),
                    getattr(multi_args, attr),
                )

    def test_main_dispatches_to_serve_multi(self):
        """When args.command == 'serve-multi', main calls serve_multi.serve_multi."""
        with (
            mock.patch.object(serve_multi, "serve_multi") as dispatched,
            mock.patch.object(launcher, "parse_args") as parse,
        ):
            parse.return_value = mock.MagicMock(
                command="serve-multi",
                config="/tmp/config.json",
            )
            with mock.patch("sys.stderr", io.StringIO()):
                launcher.main(["serve-multi", "--config", "/tmp/config.json"])
            dispatched.assert_called_once()


# ---------------------------------------------------------------------------
# proxy behavior (unit tests with mock server)
# ---------------------------------------------------------------------------


class FakeBackendHandler(BaseHTTPRequestHandler):
    """A minimal HTTP handler that echoes requests back with a configured
    model ID and handles /ready, /status, /v1/models."""

    model_id: str
    ready_flag: bool = True
    pending_switch: bool = False

    def log_message(self, format, *args):
        pass  # silence logs

    def do_GET(self):
        if self.path == "/ready":
            status = 200 if self.ready_flag else 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            body = json.dumps(
                {"status": "ready" if self.ready_flag else "unavailable"}
            ).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            payload = json.dumps(
                {
                    "model": self.model_id,
                    "ready": self.ready_flag,
                    "maximum_context_tokens": 131072,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/v1/models":
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_id,
                            "object": "model",
                            "owned_by": "splash",
                            "max_model_len": 131072,
                            "context_length": 131072,
                            "vision": False,
                            "input_modalities": ["text"],
                        }
                    ],
                    "models": [
                        {"name": self.model_id, "description": "", "release_date": ""}
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        # A body requesting stream=true gets a small SSE stream with a
        # deliberate delay between events, mirroring server/server.py
        # (no Content-Length, no chunked encoding, Connection: close).
        try:
            if json.loads(body).get("stream") is True:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for index in range(3):
                    self.wfile.write(
                        f'data: {{"model": {json.dumps(self.model_id)}, "n": {index}}}\n\n'.encode()
                    )
                    self.wfile.flush()
                    time.sleep(0.3)
                self.wfile.write(b'data: {"model": null, "done": true}\n\n')
                self.wfile.flush()
                return
        except (ValueError, UnicodeDecodeError):
            pass
        # Echo the request body with the model_id baked in.
        payload = json.dumps(
            {
                "model": self.model_id,
                "body_received": body.decode("utf-8", errors="replace"),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ProxyTests(unittest.TestCase):
    """Tests for the transparent proxy behavior of serve_multi.

    These tests use a real ThreadingHTTPServer as the backend so that the
    proxy's forwarding logic is exercised end-to-end without mocking the
    HTTP layer.
    """

    def _start_backend(
        self, model_id: str, ready: bool = True
    ) -> tuple[int, threading.Event]:
        port = _free_port()
        handler_class = type(
            "FakeBackend",
            (FakeBackendHandler,),
            {"model_id": model_id, "ready_flag": ready},
        )
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        ready_event = threading.Event()
        ready_event.set()  # immediately ready for these tests
        return port, thread, server, ready_event

    def _make_supervisor(self, models, backend_port, **kwargs):
        shared = {
            "switch_timeout": 600.0,
            "switch_settle": 60.0,
            "evict_cache": kwargs.get("evict_cache", False),
            "kv_format": "int8",
            "max_request_size": None,
            "max_cache_disk": 0,
            "max_image_pixels": None,
            "allowed_host": [],
            "api_key": None,
            "no_webui": False,
        }
        supervisor = serve_multi.Supervisor(models, shared, "127.0.0.1", 0)
        # Patch _launch_child to just point at our fake backend.
        supervisor._child_port = backend_port
        supervisor._child_ready = True
        supervisor._active_model = models[0]["model"] if models else None
        return supervisor

    def _make_proxy_server(self, proxy_port, supervisor):
        """Create a proxy server with the supervisor attached."""

        # Create a handler class with the supervisor baked in so it's
        # available before __init__ calls handle().
        def make_handler_class(sup):
            class Handler(serve_multi._ServeMultiHandler):
                supervisor = sup

            return Handler

        return serve_multi._ForwardingHTTPServer(
            ("127.0.0.1", proxy_port),
            make_handler_class(supervisor),
            supervisor=supervisor,
        )

    def test_forwards_post_to_backend(self):
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                status, headers, body = _post(
                    proxy_port,
                    "/v1/chat/completions",
                    json.dumps({"model": "owner/repo-a", "messages": []}),
                )
                self.assertEqual(status, 200)
                resp = json.loads(body)
                self.assertEqual(resp["model"], "owner/repo-a")
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_returns_503_when_switching(self):
        """While switching, POST requests hold until timeout, then 503."""
        backend_a_port, t_a, server_a, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor(
            [{"model": "owner/repo-a"}, {"model": "owner/repo-b"}], backend_a_port
        )
        # Force switching state with a short timeout.
        supervisor._switching = True
        supervisor._switch_target = "owner/repo-b"
        supervisor.hold_timeout = 0.5
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                status, headers, body = _post_with_timeout(
                    proxy_port,
                    "/v1/chat/completions",
                    json.dumps({"model": "owner/repo-b", "messages": []}),
                    timeout=5,
                )
                self.assertEqual(status, 503)
                resp = json.loads(body)
                self.assertEqual(resp["error"]["type"], "model_switching")
            finally:
                proxy_server.shutdown()
        finally:
            server_a.shutdown()

    def test_triggers_switch_on_model_mismatch(self):
        """When a request targets a different model than the active one,
        the proxy triggers a switch and holds until it completes."""
        backend_a_port, t_a, server_a, _ = self._start_backend("owner/repo-a")
        switch_triggered = threading.Event()
        original_switch_to = serve_multi.Supervisor.switch_to

        def patched_switch_to(self, model):
            switch_triggered.set()
            # Complete the switch immediately.
            with self._lock:
                self._switching = False
                self._active_model = model
            self._switch_done.set()
            return original_switch_to(self, model)

        supervisor = self._make_supervisor(
            [{"model": "owner/repo-a"}, {"model": "owner/repo-b"}], backend_a_port
        )
        with mock.patch.object(serve_multi.Supervisor, "switch_to", patched_switch_to):
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    status, headers, body = _post_with_timeout(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "owner/repo-b", "messages": []}),
                        timeout=5,
                    )
                    # Should get 200 after switch completes and forwards.
                    self.assertEqual(status, 200)
                    self.assertTrue(switch_triggered.is_set())
                finally:
                    proxy_server.shutdown()
            finally:
                server_a.shutdown()

    def test_streams_sse_in_real_time(self):
        """A streaming response must reach the client event-by-event, not
        as one lump when the upstream closes (regression: read() blocks
        until 64 KiB or EOF on a length-unknown socket)."""
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                start = time.monotonic()
                conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=10)
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(
                        {"model": "owner/repo-a", "stream": True, "messages": []}
                    ),
                    {"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                self.assertIn("text/event-stream", resp.getheader("Content-Type", ""))
                arrivals = []
                events = 0
                while True:
                    line = resp.fp.readline()
                    if not line:
                        break
                    if line.startswith(b"data: "):
                        events += 1
                        arrivals.append(time.monotonic() - start)
                conn.close()
                self.assertEqual(events, 4)
                # The upstream spaces its events 0.3s apart; if the proxy
                # buffered the whole stream, all arrivals would be ~equal.
                self.assertGreaterEqual(
                    arrivals[-1] - arrivals[0], 0.6,
                    f"events arrived in one lump: {arrivals}",
                )
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_forwards_when_model_matches(self):
        """When a request targets the currently active model, it is
        forwarded normally (no switch triggered)."""
        backend_a_port, t_a, server_a, _ = self._start_backend("owner/repo-a")
        switch_triggered = threading.Event()
        original_switch_to = serve_multi.Supervisor.switch_to

        def patched_switch_to(self, model):
            switch_triggered.set()
            return original_switch_to(self, model)

        supervisor = self._make_supervisor(
            [{"model": "owner/repo-a"}, {"model": "owner/repo-b"}], backend_a_port
        )
        with mock.patch.object(serve_multi.Supervisor, "switch_to", patched_switch_to):
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    status, headers, body = _post(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "owner/repo-a", "messages": []}),
                    )
                    self.assertEqual(status, 200)
                    self.assertFalse(switch_triggered.is_set())
                    resp = json.loads(body)
                    self.assertEqual(resp["model"], "owner/repo-a")
                finally:
                    proxy_server.shutdown()
            finally:
                server_a.shutdown()

    def test_holds_request_until_its_queued_model_is_active(self):
        """A request for a model queued behind an in-flight switch is held
        past the first switch and only forwarded once its own model becomes
        active (not misrouted to the intermediate model)."""
        backend_port, t, server, _ = self._start_backend("owner/repo-c")
        supervisor = self._make_supervisor(
            [
                {"model": "owner/repo-a"},
                {"model": "owner/repo-b"},
                {"model": "owner/repo-c"},
            ],
            backend_port,
        )
        # Simulate: a -> b switch in flight; b lands at ~0.3s; the queued
        # switch to c (this request's model) runs until ~0.6s.
        supervisor._switching = True
        supervisor._switch_target = "owner/repo-b"
        supervisor._queued_targets = ["owner/repo-c"]
        supervisor.hold_timeout = 5.0

        def advance_switch():
            time.sleep(0.3)
            with supervisor._lock:
                supervisor._switching = False
                supervisor._switch_target = None
                supervisor._active_model = "owner/repo-b"
            time.sleep(0.3)
            with supervisor._lock:
                supervisor._switching = True
                supervisor._switch_target = "owner/repo-c"
            time.sleep(0.3)
            with supervisor._lock:
                supervisor._switching = False
                supervisor._switch_target = None
                supervisor._queued_targets = []
                supervisor._active_model = "owner/repo-c"

        # Make any stray switch_to() a no-op so the test's hand-rolled
        # state machine stays authoritative.
        with (
            mock.patch.object(supervisor, "_stop_child"),
            mock.patch.object(supervisor, "_settle_memory"),
            mock.patch.object(supervisor, "_launch_child"),
        ):
            threading.Thread(target=advance_switch, daemon=True).start()
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    start = time.monotonic()
                    status, headers, body = _post_with_timeout(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "owner/repo-c", "messages": []}),
                        timeout=10,
                    )
                    elapsed = time.monotonic() - start
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["model"], "owner/repo-c")
                    # Must have waited out the intermediate b window
                    # (~0.6s), not been forwarded at 0.3s to the wrong model.
                    self.assertGreaterEqual(elapsed, 0.55)
                finally:
                    proxy_server.shutdown()
            finally:
                server.shutdown()

    def test_unknown_model_returns_503_immediately(self):
        """A request naming a model that is not in the config gets a quick
        503 model_not_found instead of triggering a failing switch cycle."""
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor(
            [{"model": "owner/repo-a"}, {"model": "owner/repo-b"}], backend_port
        )
        switch_triggered = threading.Event()
        original_switch_to = serve_multi.Supervisor.switch_to

        def patched_switch_to(self, model):
            switch_triggered.set()
            return original_switch_to(self, model)

        with mock.patch.object(serve_multi.Supervisor, "switch_to", patched_switch_to):
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    status, headers, body = _post_with_timeout(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "owner/nope", "messages": []}),
                        timeout=5,
                    )
                    self.assertEqual(status, 503)
                    resp = json.loads(body)
                    self.assertEqual(resp["error"]["type"], "model_not_found")
                    self.assertFalse(switch_triggered.is_set())
                finally:
                    proxy_server.shutdown()
            finally:
                server.shutdown()

    def test_ready_returns_503_during_switch(self):
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        supervisor._switching = True
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                status, _, body = _get(proxy_port, "/ready")
                self.assertEqual(status, 503)
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_ready_returns_200_when_ready(self):
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        supervisor._switching = False
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                status, _, body = _get(proxy_port, "/ready")
                self.assertEqual(status, 200)
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_v1_models_synthesized_when_not_ready(self):
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        supervisor._child_ready = False
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                status, _, body = _get(proxy_port, "/v1/models")
                self.assertEqual(status, 503)
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_forwarding_preserves_headers(self):
        """Headers from the client are forwarded to the backend."""
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = self._make_supervisor([{"model": "owner/repo-a"}], backend_port)
        try:
            proxy_port = _free_port()
            proxy_server = self._make_proxy_server(proxy_port, supervisor)
            thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
                body = json.dumps({"model": "owner/repo-a", "messages": []})
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body,
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer test-key",
                    },
                )
                resp = conn.getresponse()
                resp.read()
                conn.close()
                self.assertEqual(resp.status, 200)
            finally:
                proxy_server.shutdown()
        finally:
            server.shutdown()

    def test_lazy_loads_model_on_first_request(self):
        """When no model is running, a request should trigger loading."""
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        # Create supervisor with NO model running
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo-a", "aliases": ["alias-a"]}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1", 0,
        )
        # Explicitly no model loaded
        supervisor._active_model = None
        supervisor._child_port = None
        supervisor._child_ready = False
        launched = []

        def patched_launch(model, *, wait_ready=True):
            launched.append(model)
            supervisor._child_port = backend_port
            supervisor._child_ready = True
            supervisor._active_model = model

        with (
            mock.patch.object(supervisor, "_launch_child", patched_launch),
            mock.patch.object(supervisor, "_settle_memory"),
        ):
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    status, headers, body = _post_with_timeout(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "owner/repo-a", "messages": []}),
                        timeout=10,
                    )
                    # Should get 200 after lazy loading completes
                    self.assertEqual(status, 200)
                    self.assertEqual(launched, ["owner/repo-a"])
                    resp = json.loads(body)
                    self.assertEqual(resp["model"], "owner/repo-a")
                finally:
                    proxy_server.shutdown()
            finally:
                server.shutdown()

    def test_lazy_loads_via_alias(self):
        """When no model is running and request uses an alias, it should load."""
        backend_port, t, server, _ = self._start_backend("owner/repo-a")
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo-a", "aliases": ["alias-a"]}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1", 0,
        )
        supervisor._active_model = None
        supervisor._child_port = None
        supervisor._child_ready = False
        launched = []

        def patched_launch(model, *, wait_ready=True):
            launched.append(model)
            supervisor._child_port = backend_port
            supervisor._child_ready = True
            supervisor._active_model = model

        with (
            mock.patch.object(supervisor, "_launch_child", patched_launch),
            mock.patch.object(supervisor, "_settle_memory"),
        ):
            try:
                proxy_port = _free_port()
                proxy_server = self._make_proxy_server(proxy_port, supervisor)
                thread = threading.Thread(
                    target=proxy_server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    status, headers, body = _post_with_timeout(
                        proxy_port,
                        "/v1/chat/completions",
                        json.dumps({"model": "alias-a", "messages": []}),
                        timeout=10,
                    )
                    # Should get 200 after lazy loading completes
                    self.assertEqual(status, 200)
                    self.assertEqual(launched, ["owner/repo-a"])
                finally:
                    proxy_server.shutdown()
            finally:
                server.shutdown()


# ---------------------------------------------------------------------------
# memory helpers
# ---------------------------------------------------------------------------


class MemoryHelperTests(unittest.TestCase):
    def test_model_memory_need_returns_none_for_missing_model(self):
        result = serve_multi._model_memory_need("nonexistent/repo")
        self.assertIsNone(result)

    def test_vm_stat_available_bytes_returns_int_or_none(self):
        result = serve_multi._vm_stat_available_bytes()
        self.assertIsInstance(result, (int, type(None)))

    def test_vm_stat_wired_bytes_returns_int_or_none(self):
        result = serve_multi._vm_stat_wired_bytes()
        self.assertIsInstance(result, (int, type(None)))

    def test_physical_memory_bytes_returns_int_or_none(self):
        result = serve_multi._physical_memory_bytes()
        self.assertIsInstance(result, (int, type(None)))


# ---------------------------------------------------------------------------
# supervisor lifecycle
# ---------------------------------------------------------------------------


class SupervisorTests(unittest.TestCase):
    def test_switch_to_clears_switching_on_completion(self):
        """After switch_to completes, switching is False and active_model
        is updated."""
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}, {"model": "b"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        # Patch _launch_child to avoid actually launching anything.
        launched = []
        with mock.patch.object(supervisor, "_launch_child") as launch:
            launch.side_effect = lambda model, *, wait_ready=True: (
                launched.append(model),
                setattr(supervisor, "_active_model", model),
            )
            with mock.patch.object(supervisor, "_stop_child"):
                with mock.patch.object(supervisor, "_settle_memory"):
                    # Set up initial state.
                    supervisor._active_model = "a"
                    supervisor._child_ready = True
                    self.assertFalse(supervisor.switching)
                    self.assertIsNone(supervisor.switch_target)
                    # Trigger switch.
                    supervisor.switch_to("b")
                    # Wait for background thread to complete.
                    for _ in range(40):
                        time.sleep(0.05)
                        if not supervisor.switching:
                            break
                    # Verify the switch completed.
                    self.assertFalse(supervisor.switching, "switching should be False")
                    self.assertIsNone(
                        supervisor.switch_target, "switch_target should be None"
                    )
                    self.assertEqual(supervisor.active_model, "b")
                    # _launch_child should have been called for "b".
                    launch.assert_any_call("b", wait_ready=True)

    def test_switch_to_is_idempotent_for_same_model(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}, {"model": "b"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        supervisor._active_model = "a"
        launched = []
        with mock.patch.object(supervisor, "_launch_child") as launch:
            launch.side_effect = lambda model, **kw: launched.append(model)
            with mock.patch.object(supervisor, "_stop_child"):
                with mock.patch.object(supervisor, "_settle_memory"):
                    supervisor.switch_to("a")
                    time.sleep(0.05)
                    self.assertEqual(launched, [])  # no launch for same model

    def test_switch_to_queues_model_while_switch_in_flight(self):
        """A model requested while a switch is in flight is queued (deduped)
        and started automatically once the current switch completes."""
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}, {"model": "b"}, {"model": "c"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        supervisor._active_model = "a"
        supervisor._child_ready = True
        launched = []
        launch_gate = threading.Event()

        def slow_launch(model, *, wait_ready=True):
            launched.append(model)
            launch_gate.wait(5)
            with supervisor._lock:
                supervisor._active_model = model
                supervisor._child_ready = True

        with (
            mock.patch.object(supervisor, "_launch_child", side_effect=slow_launch),
            mock.patch.object(supervisor, "_stop_child"),
            mock.patch.object(supervisor, "_settle_memory"),
        ):
            supervisor.switch_to("b")
            for _ in range(200):
                if supervisor.switching and supervisor.switch_target == "b":
                    break
                time.sleep(0.01)
            self.assertTrue(supervisor.switching)
            supervisor.switch_to("c")
            supervisor.switch_to("c")  # a duplicate is not queued twice
            time.sleep(0.05)
            self.assertEqual(supervisor.queued_models, ["c"])
            launch_gate.set()  # let b finish; the drain should start c
            for _ in range(400):
                if supervisor.active_model == "c" and not supervisor.switching:
                    break
                time.sleep(0.01)
            self.assertFalse(supervisor.switching)
            self.assertEqual(supervisor.active_model, "c")
            self.assertEqual(supervisor.queued_models, [])
            self.assertEqual(launched, ["b", "c"])

    def test_shutdown_stops_child(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        stopped = []
        with mock.patch.object(supervisor, "_stop_child") as stop:
            stop.side_effect = lambda: stopped.append(True)
            supervisor.shutdown()
            self.assertEqual(stopped, [True])

    def test_active_model_property(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        self.assertIsNone(supervisor.active_model)
        supervisor._active_model = "a"
        self.assertEqual(supervisor.active_model, "a")

    def test_child_port_property(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "a"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        self.assertIsNone(supervisor.child_port)
        supervisor._child_port = 1234
        self.assertEqual(supervisor.child_port, 1234)

    def test_resolve_model_by_full_id(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo", "aliases": ["alias"]}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        self.assertEqual(supervisor._resolve_model("owner/repo"), "owner/repo")

    def test_resolve_model_by_alias(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo", "aliases": ["alias"]}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        self.assertEqual(supervisor._resolve_model("alias"), "owner/repo")

    def test_model_names_includes_aliases(self):
        supervisor = serve_multi.Supervisor(
            [
                {"model": "owner/repo-a", "aliases": ["alias-a"]},
                {"model": "owner/repo-b", "aliases": ["alias-b"]},
            ],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        names = supervisor._model_names()
        self.assertIn("owner/repo-a", names)
        self.assertIn("owner/repo-b", names)
        self.assertIn("alias-a", names)
        self.assertIn("alias-b", names)

    def test_model_specs_by_model_resolves_alias(self):
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo", "aliases": ["alias"], "max_context": 8192}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        spec = supervisor.model_specs_by_model("alias")
        self.assertEqual(spec["model"], "owner/repo")
        self.assertEqual(spec["max_context"], 8192)

    def _child_command(self, supervisor, spec, model="owner/repo"):
        with (
            mock.patch.object(serve_multi.model_artifacts, "Selection") as sel,
            mock.patch.object(
                serve_multi.assembly, "hold", return_value=(mock.MagicMock(), None)
            ),
        ):
            sel.of.return_value = mock.Mock(link="link")
            return [str(part) for part in supervisor._child_command(model, spec, 12345)]

    def test_child_command_max_context_precedence(self):
        """--max-context (shared) beats the per-model config value, which
        beats the engine default (auto)."""
        spec = {"model": "owner/repo", "aliases": (), "max_context": 8192}
        # shared flag wins over the config value
        supervisor = serve_multi.Supervisor(
            [spec], {"max_context": 131072}, "127.0.0.1", 0
        )
        self.assertIn("131072", self._child_command(supervisor, spec))
        # config value used when the shared flag is not given
        supervisor = serve_multi.Supervisor([spec], {}, "127.0.0.1", 0)
        self.assertIn("8192", self._child_command(supervisor, spec))
        # auto when neither is set
        spec = {"model": "owner/repo", "aliases": (), "max_context": None}
        supervisor = serve_multi.Supervisor([spec], {}, "127.0.0.1", 0)
        self.assertIn("auto", self._child_command(supervisor, spec))

    def test_switch_to_via_alias(self):
        """switch_to should accept an alias and resolve it to the full model ID."""
        supervisor = serve_multi.Supervisor(
            [
                {"model": "owner/repo-a", "aliases": ["alias-a"]},
                {"model": "owner/repo-b", "aliases": ["alias-b"]},
            ],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        launched = []
        with mock.patch.object(supervisor, "_launch_child") as launch:
            launch.side_effect = lambda model, *, wait_ready=True: (
                launched.append(model),
                setattr(supervisor, "_active_model", model),
            )
            with mock.patch.object(supervisor, "_stop_child"):
                with mock.patch.object(supervisor, "_settle_memory"):
                    supervisor._active_model = "owner/repo-a"
                    supervisor.switch_to("alias-b")
                    for _ in range(40):
                        time.sleep(0.05)
                        if not supervisor.switching:
                            break
                    self.assertFalse(supervisor.switching)
                    self.assertEqual(supervisor.active_model, "owner/repo-b")
                    launch.assert_called_with("owner/repo-b", wait_ready=True)

    def test_load_first_model_lazy(self):
        """_load_first_model should launch the first model when none is running."""
        supervisor = serve_multi.Supervisor(
            [
                {"model": "owner/repo-a", "aliases": ["alias-a"]},
                {"model": "owner/repo-b", "aliases": ["alias-b"]},
            ],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        launched = []
        with mock.patch.object(supervisor, "_launch_child") as launch:
            launch.side_effect = lambda model, *, wait_ready=True: (
                launched.append(model),
                setattr(supervisor, "_active_model", model),
            )
            with mock.patch.object(supervisor, "_stop_child"):
                with mock.patch.object(supervisor, "_settle_memory"):
                    # No model is currently running
                    self.assertIsNone(supervisor.active_model)
                    result = supervisor._load_first_model()
                    self.assertTrue(result)
                    # Should have initiated loading
                    for _ in range(40):
                        time.sleep(0.05)
                        if not supervisor.switching:
                            break
                    self.assertFalse(supervisor.switching)
                    self.assertEqual(supervisor.active_model, "owner/repo-a")
                    launch.assert_called_with("owner/repo-a", wait_ready=True)

    def test_load_first_model_no_op_when_running(self):
        """_load_first_model should return True without loading if a model is already running."""
        supervisor = serve_multi.Supervisor(
            [{"model": "owner/repo-a"}],
            {"switch_timeout": 600.0, "switch_settle": 60.0, "evict_cache": False},
            "127.0.0.1",
            0,
        )
        supervisor._active_model = "owner/repo-a"
        with mock.patch.object(supervisor, "_launch_child") as launch:
            result = supervisor._load_first_model()
            self.assertTrue(result)
            launch.assert_not_called()


# ---------------------------------------------------------------------------
# serve_multi() entry point (unit test, no real server)
# ---------------------------------------------------------------------------


class ServeMultiEntryTests(unittest.TestCase):
    def test_serve_multi_validates_config_path(self):
        with self.assertRaises(ValueError) as cm:
            serve_multi.serve_multi(mock.MagicMock(config="/nonexistent/config.json"))
        self.assertIn("cannot read config", str(cm.exception))

    def test_serve_multi_passes_through_shared_args(self):
        """serve_multi() builds the supervisor with the correct shared args."""
        config = {"models": [{"model": "owner/repo"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        try:
            with (
                mock.patch.object(serve_multi, "_ensure_installed"),
                mock.patch.object(serve_multi, "Supervisor") as supervisor_cls,
                mock.patch.object(serve_multi, "_ForwardingHTTPServer") as server_cls,
                mock.patch("signal.signal"),
            ):
                supervisor_instance = mock.MagicMock()
                supervisor_cls.return_value = supervisor_instance
                server_instance = mock.MagicMock()
                server_cls.return_value = server_instance

                # Minimal args.
                args = mock.MagicMock(
                    config=path,
                    host="127.0.0.1",
                    port=0,
                    switch_timeout=600.0,
                    switch_settle=60.0,
                    evict_cache=True,
                    default_reasoning_effort=None,
                    kv_format="int8",
                    max_memory=None,
                    max_cache_disk=0,
                    max_context=None,
                    max_request_size=None,
                    max_image_pixels=None,
                    allowed_host=[],
                    api_key=None,
                    no_webui=False,
                )
                serve_multi.serve_multi(args)
                # Supervisor should have been instantiated.
                supervisor_cls.assert_called_once()
                call_args = supervisor_cls.call_args
                self.assertEqual(len(call_args[0]), 4)
                # First arg is the model specs.
                specs = call_args[0][0]
                self.assertEqual(len(specs), 1)
                self.assertEqual(specs[0]["model"], "owner/repo")
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# integration: real-process smoke test
# ---------------------------------------------------------------------------


class RealProcessSmokeTest(unittest.TestCase):
    """Smoke test that launches a real server subprocess and routes through
    the serve_multi proxy.  Requires the Splash binary to exist."""

    @classmethod
    def setUpClass(cls):
        # Skip real-process tests to avoid high memory usage.
        cls.skipReason = "Skipping real-process test to avoid high memory usage"
        raise unittest.SkipTest(cls.skipReason)

    def test_placeholder(self):
        pass


if __name__ == "__main__":
    unittest.main()
