"""
Serve attribution graph files using the circuit-tracer built-in frontend.

Usage:
    python serve_graphs.py --graph_file_dir ./graph_files/gemma_gsm8k --port 8046

Then open in browser (with port forwarding if on a remote server):
    http://localhost:8046/index.html
"""

import argparse
import time
from circuit_tracer.frontend.local_server import serve

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_file_dir", default="./graph_files/gemma_gsm8k")
    p.add_argument("--port", type=int, default=8046)
    return p.parse_args()

args = parse_args()
server = serve(data_dir=args.graph_file_dir, port=args.port)
print(f"\nGraph viewer running at: http://localhost:{args.port}/index.html")
print("If on a remote server, set up SSH port forwarding:")
print(f"  ssh -L {args.port}:localhost:{args.port} <your-server>")
print("\nPress Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
    print("Server stopped.")
