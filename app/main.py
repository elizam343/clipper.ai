#!/usr/bin/env python3
"""Clipper.app: starts the local server and shows the page in its own window.

    python3 app/main.py            # the app window
    python3 app/main.py --browser  # the same page in your browser instead
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    threading.Thread(target=server.serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):   # wait for the server to answer
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    if "--browser" in sys.argv:
        import webbrowser
        webbrowser.open(url)
        print(f"Clipper is running at {url} (Ctrl+C to quit)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    import webview
    webview.create_window("Clipper", url, width=1180, height=860, min_size=(720, 560))
    webview.start()   # returns when the window closes; the server thread ends with it


if __name__ == "__main__":
    main()
