"""Local viewer for the OakInk2 result analysis: play test clips next to gold / predicted timelines
and browse all figures.

    python view_oakink2_analysis.py                    # serves models/runs, opens the browser
    python view_oakink2_analysis.py --runs ~/Downloads/runs --port 8800

It finds every folder under --runs that contains per_video_metrics.csv (the output of
analyze_oakink2_predictions.py, e.g. runs/left or runs/analysis/right). Interactive timelines
need that folder's viewer.json; videos need its clips/. Only the Python standard library is used.
"""
import argparse
import json
import mimetypes
import os
import re
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE / "oakink2_viewer.html"
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def build_index(root):
    analyses = []
    for csv_path in sorted(root.rglob("per_video_metrics.csv")):
        d = csv_path.parent
        rel = d.relative_to(root).as_posix()
        analyses.append({
            "name": rel,
            "viewer": (d / "viewer.json").exists(),
            "figures": sorted(p.relative_to(root).as_posix() for p in d.glob("*") if p.suffix in IMAGE_EXT),
            "timelines": sorted(p.relative_to(root).as_posix() for p in (d / "timelines").glob("*.png")),
        })
    skip = {Path(a["name"]) for a in analyses}
    comparison = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.suffix in IMAGE_EXT
                        and not any(s == p.relative_to(root).parent or s in p.relative_to(root).parents for s in skip))
    return {"analyses": analyses, "figures": comparison}


class Handler(SimpleHTTPRequestHandler):
    root = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/index":
            return self._send_bytes(json.dumps(build_index(self.root)).encode(), "application/json")
        if path.startswith("/files/"):
            target = (self.root / path[len("/files/"):]).resolve()
            if self.root not in target.parents or not target.is_file():
                return self.send_error(404)
            return self._send_file(target)
        self.send_error(404)

    def _send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, target):
        # Byte ranges let the <video> element seek (Safari will not play mp4 without them).
        size = target.stat().st_size
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m:
            start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(target, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(HERE / "runs"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()
    root = Path(os.path.expanduser(args.runs)).resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist")
    Handler.root = root
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"ports {args.port}-{args.port + 19} are all in use; pass --port")
    url = f"http://127.0.0.1:{port}/"
    index = build_index(root)
    print(f"serving {root}: {len(index['analyses'])} analysis folders "
          f"({', '.join(a['name'] for a in index['analyses']) or 'none'}), {len(index['figures'])} other figures",
          flush=True)
    print(f"open {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
