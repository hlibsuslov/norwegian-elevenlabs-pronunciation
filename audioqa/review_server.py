"""Serve reviewed clips and optionally run an explicitly budgeted local regen action."""
import json
import secrets
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import soundfile as sf

from scripts.common import read_json
from scripts.regenerate_defects import regenerate_event
from .repair import make_api


def serve(args,report):
    root=args.report.resolve().parents[1]
    review=root/"review"
    token=secrets.token_urlsafe(32)
    permitted={row["id"]:row for row in report["events"] if row.get("repair_context")}
    metadata=read_json(args.generation_metadata) if args.generation_metadata else {}
    api=make_api(args) if args.max_api_budget or args.max_regenerations else None
    lock=Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*values,**kwargs):
            super().__init__(*values,directory=str(review),**kwargs)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path in {"/","/index.html"}:
                html=(review/"index.html").read_text(encoding="utf-8")
                if api:
                    html=html.replace("</head>",f"<script>window.audioqaToken={json.dumps(token)}</script></head>")
                content=html.encode()
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.send_header("Content-Length",str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif path.startswith("/clips/") and ".." not in path:
                super().do_GET()
            else:
                self.send_error(404)

        def reply(self,code,result):
            encoded=json.dumps(result).encode()
            self.send_response(code)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            if self.path!="/api/regenerate" or not api or self.headers.get("X-AudioQA-Token")!=token:
                self.reply(403,{"error":"No active regeneration authorization."})
                return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=4096:
                    raise ValueError("Invalid request length.")
                requested=json.loads(self.rfile.read(length))
                event=permitted.get(requested.get("id"))
                if not event or requested.get("status")!="REGENERATE":
                    raise ValueError("Select an authorized sentence event.")
                with lock:
                    media=root/"cache/decoded.wav"
                    candidate,details=regenerate_event(api,media,report["sample_rate"],event,metadata,"VIDEO_LOCKED")
                    url=f"clips/{event['id']}/candidate.wav"
                    sf.write(review/url,candidate["values"],report["sample_rate"],subtype="FLOAT")
                self.reply(200,{"message":"Candidate is ready for human review.","audio":url,"cached":details["cached"]})
            except (ValueError,RuntimeError,OSError) as exc:
                self.reply(400,{"error":str(exc)})

    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/",flush=True)
        server.serve_forever()
