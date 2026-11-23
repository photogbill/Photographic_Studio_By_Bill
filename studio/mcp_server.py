"""The Studio as an MCP server: an AI model connects and uses the Studio's tools directly (docs/CLI.md, "MCP").

    python -m studio.mcp_server          (studio-mcp.bat on Windows; "Connect Studio to Claude.bat" sets it up)

The Model Context Protocol over stdio: one JSON-RPC message per line on standard input and output, nothing else on
standard output (logs and progress go to standard error). No extra package is needed: the protocol is small enough to
speak directly, and every tool is one of the commands of cli_tools.py, so the two can never drift apart:

  * the tools are built from the command line's own parsers: each command becomes a tool ``studio_<command>``, each
    option a typed property with the option's own help text. A new command or option appears here by itself;
  * a call is turned back into that command line and run in this process (the Studio stays loaded, so a second
    call does not pay the start-up again), and answers with the command's own JSON result;
  * ``studio_view`` shows a picture to the model (a small JPEG in the answer), and any command run with ``preview``
    returns its preview picture too, so the model can look at what it made.

Both protocol eras are served (specification 2026-07-28, "dual-era"): a request that carries its protocol version in
``_meta`` is answered statelessly under 2026-07-28; a client that opens with ``initialize`` gets the handshake of
2025-11-25 and earlier.

What a connected model can do: read pictures anywhere this user can, and write results under the Studio's naming
and never-overwrite rules (a picture is never replaced, nor any file the Studio did not write). It cannot delete,
move or run anything else. No Qt here (rule 2.2).
"""
from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import sys
import traceback

from . import __version__, log as slog

log = slog.get("mcp")

MODERN = ("2026-07-28",)
LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER = "io.modelcontextprotocol/serverInfo"
SERVER_INFO = {"name": "photographic-studio", "title": "Photographic Studio By Bill", "version": __version__}
SKIP_COMMANDS = ("batch", "queue")          # batch: a model calls the tools one after another; queue: the
                                            # studio_queue_* tools below take tool calls, not command lines
SKIP_DESTS = ("help", "json", "result")
PREVIEW_MAX_BYTES = 1_500_000
INSTRUCTIONS = (
    "Photographic Studio By Bill: a free raw developer and photo editor running on this PC. Call studio_schema "
    "first (optionally with a setting or group name, e.g. 'light'): it lists every setting, its default and what it "
    "does. Pictures are given as file paths (raw, TIFF, PNG, JPEG) or folders. Settings go in 'settings' (an object "
    "holding only what differs from the defaults, e.g. {\"exposure_ev\": 0.3, \"light\": {\"shadows\": 25}}) or in "
    "'set' ([\"light.shadows=25\"]). Results are written as NAME_Studio.ext into 'out' (a folder) and never replace "
    "a picture: read the path written from the answer. Use 'preview' (a long edge in pixels, e.g. 1024) to get a "
    "quick picture back and look at it before a full-size run; studio_view shows any picture. Large raws and stacks "
    "take from seconds to minutes. Anything that may take longer than a minute (stack, stacks, foliage, model, intake, "
    "dualpixel on full-size raws, blender) goes through the queue instead of a direct call, which a client may cut "
    "off: studio_queue_add takes the same tool name and arguments as a direct call, checks them, answers at once, and "
    "the job runs to its end in a worker of its own, one job after another; studio_queue_status shows progress and, "
    "when a job is finished, its result.")


# ====================================================================================================== tools
def _commands() -> dict:
    """{command: its argparse parser}, from the command line itself."""
    from . import cli, cli_tools
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    return {name: sub.choices[name] for name in cli_tools.COMMANDS if name not in SKIP_COMMANDS}


def _prop(action) -> dict:
    """One command-line option as a JSON Schema property."""
    help_ = (action.help or "").replace("%(default)s", str(action.default))
    if isinstance(action, argparse._StoreTrueAction):
        return {"type": "boolean", "description": help_}
    base = "integer" if action.type is int else "number" if action.type is float else "string"
    one: dict = {"type": base}
    if action.choices:
        one["enum"] = list(action.choices)
    if isinstance(action, argparse._AppendAction) or action.nargs in ("+", "*"):
        return {"type": "array", "items": one, "description": help_}
    if action.nargs == "?" and action.option_strings:           # --scratches [AMOUNT]: true = on at its default
        d = dict(one, description=(help_ + f" (true = on, at {action.const})").strip())
        d["type"] = [base, "boolean"]
        return d
    if action.default not in (None, False) and action.option_strings:
        help_ = (help_ + f" (default {action.default})").strip()
    return dict(one, description=help_)


def tool_for(name: str, parser) -> dict:
    from . import cli_tools
    props, required = {}, []
    for a in parser._actions:
        if a.dest in SKIP_DESTS or isinstance(a, argparse._HelpAction):
            continue
        props[a.dest] = _prop(a)
        if (not a.option_strings and a.nargs in (None, "+")) or getattr(a, "required", False):
            required.append(a.dest)
    if "recipe" in props:
        props["settings"] = {"type": "object", "description": "Settings as an object holding only what differs from "
                             "the defaults, e.g. {\"exposure_ev\": 0.3, \"light\": {\"shadows\": 25}}. Names are "
                             "checked; studio_schema lists them."}
    tool = {"name": "studio_" + name.replace("-", "_"), "title": "Studio: " + name,
            "description": cli_tools.HELP[name][0].upper() + cli_tools.HELP[name][1:] + ".",
            "inputSchema": {"type": "object", "properties": props, "required": required,
                            "additionalProperties": False}}
    if name in ("schema", "looks", "find"):
        tool["annotations"] = {"readOnlyHint": True}
    return tool


VIEW_TOOL = {
    "name": "studio_view", "title": "Studio: view a picture",
    "description": "Look at a picture: returns it as a small JPEG (a raw is developed with the default settings, or "
                   "with 'settings'). Writes no file.",
    "inputSchema": {"type": "object", "additionalProperties": False, "required": ["path"], "properties": {
        "path": {"type": "string", "description": "the picture (raw, TIFF, PNG, JPEG)"},
        "long_edge": {"type": "integer", "description": "long edge of the returned picture in pixels (default 1024, "
                      "at most 2048)"},
        "settings": {"type": "object", "description": "optional settings to view it with (as for studio_edit)"},
        "use_sidecar": {"type": "boolean", "description": "view it with the photo's own saved edit"}}},
    "annotations": {"readOnlyHint": True},
}

def _queue_tool(name, title, desc, props, required=(), read_only=False) -> dict:
    t = {"name": name, "title": title, "description": desc,
         "inputSchema": {"type": "object", "additionalProperties": False, "required": list(required),
                         "properties": props}}
    if read_only:
        t["annotations"] = {"readOnlyHint": True}
    return t


_JOB_ITEM = {"type": "object", "additionalProperties": False, "required": ["tool", "arguments"], "properties": {
    "tool": {"type": "string", "description": "the tool to run, e.g. studio_stack"},
    "arguments": {"type": "object", "description": "exactly the arguments a direct call of that tool takes"},
    "title": {"type": "string", "description": "a short name for the job, shown in the status"}}}

QUEUE_TOOLS = [
    _queue_tool("studio_queue_add", "Studio: add jobs to the queue",
                "Add one or more jobs to the queue and answer at once. Each job is a tool name and the same arguments "
                "a direct call takes (checked now, so a wrong argument is said straight away). The jobs run in order, "
                "each in a fresh Studio process, in a worker that keeps going until the queue is empty, however long "
                "that takes. 'hold' keeps them waiting until studio_queue_start.",
                {"jobs": {"type": "array", "items": _JOB_ITEM, "description": "the jobs, in the order to run them"},
                 "tool": {"type": "string", "description": "one job: the tool (instead of 'jobs')"},
                 "arguments": {"type": "object", "description": "one job: its arguments"},
                 "title": {"type": "string", "description": "one job: a short name"},
                 "hold": {"type": "boolean", "description": "add them held: nothing runs until studio_queue_start"}}),
    _queue_tool("studio_queue_start", "Studio: start held jobs",
                "Release held jobs (all, or the ids given) and make sure the worker is running.",
                {"ids": {"type": "array", "items": {"type": "string"}, "description": "job ids (default every held job)"}}),
    _queue_tool("studio_queue_status", "Studio: the queue's status",
                "Without 'id': the counts, whether the worker is alive, and the latest jobs (a running job shows its "
                "last progress lines). With 'id': that job in full, with its progress lines and, once finished, the "
                "command's own JSON result (the files it wrote are under result.outputs).",
                {"id": {"type": "string", "description": "one job's id"},
                 "lines": {"type": "integer", "description": "progress lines to show for one job (default 8)"}},
                read_only=True),
    _queue_tool("studio_queue_cancel", "Studio: cancel jobs",
                "Cancel one job by id, or 'all': waiting jobs are dropped, a running job is stopped within seconds. "
                "Files a job already wrote stay.",
                {"id": {"type": "string", "description": "a job id, or all"}}, ["id"]),
]
QUEUE_NAMES = tuple(t["name"] for t in QUEUE_TOOLS)


def queue_call(name: str, args: dict) -> dict:
    """The studio_queue_* tools (jobqueue.py)."""
    from . import cli, jobqueue
    tools()
    if name == "studio_queue_add":
        items = args.get("jobs")
        if items is None:
            if not args.get("tool"):
                raise BadArguments("give 'jobs' (a list of {tool, arguments}) or one 'tool' with its 'arguments'")
            items = [{"tool": args["tool"], "arguments": args.get("arguments") or {}, "title": args.get("title", "")}]
        elif args.get("tool"):
            raise BadArguments("give 'jobs' or one 'tool', not both")
        if not isinstance(items, list) or not items:
            raise BadArguments("'jobs' is a non-empty list of {tool, arguments}")
        argvs = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise BadArguments(f"job {i + 1} is not an object")
            tool = it.get("tool")
            if tool not in _PARSERS:
                near = difflib.get_close_matches(str(tool), list(_PARSERS), 3, 0.5)
                raise BadArguments(f"job {i + 1}: no tool {tool!r} to queue." + (f" Did you mean: {', '.join(near)}?"
                                   if near else ""))
            cmd, parser = _PARSERS[tool]
            argv = to_argv(cmd, parser, it.get("arguments") or {})
            try:
                cli.build_parser().parse_args(argv)
            except SystemExit:
                raise BadArguments(f"job {i + 1}: those arguments do not fit {tool}: {' '.join(argv[1:])}") from None
            argvs.append((argv, str(it.get("title") or "")))
        hold = bool(args.get("hold"))
        added = [jobqueue.add(argv, title, hold) for argv, title in argvs]
        started = False if hold else jobqueue.ensure_worker()
        return {"ok": True, "added": [jobqueue.summary(j) for j in added], "held": hold, "worker_started": started,
                "worker": jobqueue.worker_info()}
    if name == "studio_queue_start":
        return dict(jobqueue.start(args.get("ids")), ok=True)
    if name == "studio_queue_status":
        return dict(jobqueue.status(args.get("id"), int(args.get("lines") or 8)), ok=True)
    if name == "studio_queue_cancel":
        return {"ok": True, "cancelled": jobqueue.cancel(str(args.get("id") or ""))}
    raise BadArguments(f"no queue tool {name!r}")


_TOOLS: list | None = None
_PARSERS: dict = {}


def tools() -> list:
    """The tool list, in a fixed order (the same every time, so a client can cache it)."""
    global _TOOLS
    if _TOOLS is None:
        cmds = _commands()
        _PARSERS.update({"studio_" + n.replace("-", "_"): (n, p) for n, p in cmds.items()})
        _TOOLS = [tool_for(n, p) for n, p in cmds.items()] + [VIEW_TOOL] + QUEUE_TOOLS
    return _TOOLS


class BadArguments(Exception):
    """The call's arguments do not fit the tool: said to the model so it can correct them."""


def to_argv(name: str, parser, args: dict) -> list:
    """A tool call's arguments -> the command line they stand for."""
    actions = {a.dest: a for a in parser._actions if a.dest not in SKIP_DESTS}
    args = dict(args or {})
    settings = args.pop("settings", None)
    unknown = [k for k in args if k not in actions]
    if unknown:
        k = unknown[0]
        near = difflib.get_close_matches(k, list(actions) + ["settings"], 3, 0.5)
        raise BadArguments(f"{name} has no argument {k!r}." + (f" Did you mean: {', '.join(near)}?" if near else "")
                           + f" Its arguments: {', '.join(sorted(actions))}")
    argv, tail = [name], []
    if settings is not None:
        if not isinstance(settings, dict) or "recipe" not in actions:
            raise BadArguments("'settings' is an object of settings, for the commands that take a recipe")
        tail.append("--recipe=" + json.dumps(settings))
    for a in parser._actions:                              # positionals first, in the command's own order
        if a.dest in SKIP_DESTS or a.dest not in args:
            continue
        v = args[a.dest]
        if not a.option_strings:
            argv += [str(x) for x in (v if isinstance(v, list) else [v])]
            continue
        opt = a.option_strings[-1]
        if isinstance(a, argparse._StoreTrueAction):
            if v is True:
                tail.append(opt)
            elif v not in (False, None):
                raise BadArguments(f"{a.dest} is true or false")
        elif isinstance(a, argparse._AppendAction):
            for x in (v if isinstance(v, list) else [v]):
                tail.append(f"{opt}={x if isinstance(x, str) else json.dumps(x)}")
        elif a.nargs == "?" and isinstance(v, bool):
            if v:
                tail.append(opt)
        elif v is not None:
            tail.append(f"{opt}={v}")
    return argv + tail


def _jpeg_content(data: bytes) -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/jpeg"}


def view(args: dict) -> list:
    """studio_view: the picture as a small JPEG, made in memory."""
    import numpy as np
    from PIL import Image as PI
    from . import cli_tools, engine
    from .engine.io.export import ExportSpec, to_output
    path = str((args or {}).get("path") or "")
    if not os.path.isfile(path):
        raise BadArguments(f"no file {path!r}")
    n = int(args.get("long_edge") or 1024)
    n = max(64, min(2048, n))
    img = engine.open(path)
    try:
        edit = engine.load_edit(path) if args.get("use_sidecar") else engine.Edit()
        if args.get("settings"):
            edit = cli_tools.apply_partial(edit, args["settings"])
        ow, oh = engine.output_size(img, edit)
        s = min(1.0, n / float(max(ow, oh)))
        arr = engine.render(img, edit, scale=s) if s < 1.0 else engine.render(img, edit)
        h, w = arr.shape[:2]
        if max(h, w) > n:
            from .engine.pipeline.pipeline import resize
            k = n / float(max(h, w))
            arr = resize(np.ascontiguousarray(arr), max(1, round(w * k)), max(1, round(h * k)))
        rgb = to_output(np.ascontiguousarray(arr, np.float32), ExportSpec(format="jpeg", space="srgb", bits=8))
    finally:
        img.drop_cache()
    buf = io.BytesIO()
    PI.fromarray(rgb).save(buf, "JPEG", quality=88)
    info = {"path": path, "full_size": [int(ow), int(oh)], "shown": [int(rgb.shape[1]), int(rgb.shape[0])]}
    return [{"type": "text", "text": json.dumps(info)}, _jpeg_content(buf.getvalue())]


def call_tool(name: str, args: dict, progress=None) -> dict:
    """Run one tool. Always a tool result: what went wrong is said in it (isError), so the model can correct."""
    from . import cli, cli_tools
    tools()
    try:
        if name == "studio_view":
            return {"content": view(args or {}), "isError": False}
        if name in QUEUE_NAMES:
            try:
                res = queue_call(name, args or {})
            except ValueError as e:                        # no such job, an empty command line
                raise BadArguments(str(e)) from None
            return {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}
        cmd, parser = _PARSERS[name]
        argv = to_argv(cmd, parser, args or {})
        log.info("mcp call: %s", " ".join(argv)[:500])
        try:
            a = cli.build_parser().parse_args(argv)
        except SystemExit:
            raise BadArguments(f"those arguments do not fit {name}: {' '.join(argv[1:])}") from None
        cli_tools.PROGRESS = progress
        try:
            run = cli_tools.execute(a, quiet=True)
        finally:
            cli_tools.PROGRESS = None
        res = cli_tools.plain(run.res, True)
        content = [{"type": "text", "text": json.dumps(res)}]
        if getattr(a, "preview", None):                    # let the model see what it made
            for o in res.get("outputs", [])[:4]:
                p = o.get("output") or ""
                if p.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(p) and \
                        os.path.getsize(p) <= PREVIEW_MAX_BYTES:
                    with open(p, "rb") as fh:
                        content.append(_jpeg_content(fh.read()))
        return {"content": content, "structuredContent": res, "isError": not res.get("ok", False)}
    except (BadArguments, cli_tools.Usage) as e:
        return {"content": [{"type": "text", "text": str(e)}], "isError": True}
    except Exception as e:                                 # noqa: BLE001  (a fault of ours: reported, and said)
        slog.log_exception(f"mcp tool {name} failed", log)
        rep = None if slog.expected_problem(e) else slog.save_error_report(f"MCP tool {name} failed")
        text = f"{name} failed: {e}" + (f" (error report: {rep})" if rep else "")
        return {"content": [{"type": "text", "text": text}], "isError": True}


# ====================================================================================================== protocol
class Server:
    """The message loop. ``handle`` takes one decoded message and returns the reply (or None for a notification);
    ``send`` writes a message the server starts itself (progress)."""

    def __init__(self, send=None):
        self.send = send or (lambda msg: None)

    @staticmethod
    def _error(mid, code: int, message: str, data=None) -> dict:
        err = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": mid, "error": err}

    def _capabilities(self) -> dict:
        return {"tools": {}}

    def handle(self, msg) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return self._error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid request")
        method, mid = msg["method"], msg.get("id")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        if mid is None:                                    # a notification: never answered
            return None
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        modern = META_VERSION in meta

        def result(body: dict) -> dict:
            if modern:                                     # 2026-07-28: every result says its type and who answers
                body = dict(body, resultType="complete")
                body["_meta"] = {META_SERVER: SERVER_INFO}
            return {"jsonrpc": "2.0", "id": mid, "result": body}

        if modern:
            if meta[META_VERSION] not in MODERN:
                return self._error(mid, -32022, "Unsupported protocol version",
                                   {"supported": list(MODERN), "requested": meta[META_VERSION]})
            if not isinstance(meta.get(META_CAPS), dict):
                return self._error(mid, -32602, f"Invalid params: _meta['{META_CAPS}'] is required")
        try:
            if method == "server/discover":
                if not modern:
                    return self._error(mid, -32602, f"Invalid params: _meta['{META_VERSION}'] is required "
                                       f"(supported: {', '.join(MODERN)})")
                return result({"supportedVersions": list(MODERN), "capabilities": self._capabilities(),
                               "instructions": INSTRUCTIONS})
            if method == "initialize":                     # the handshake of 2025-11-25 and earlier
                want = params.get("protocolVersion")
                return result({"protocolVersion": want if want in LEGACY else LEGACY[0],
                               "capabilities": self._capabilities(), "serverInfo": SERVER_INFO,
                               "instructions": INSTRUCTIONS})
            if method == "ping":
                return result({})
            if method == "tools/list":
                return result({"tools": tools()})
            if method == "tools/call":
                name = params.get("name")
                tools()
                if name not in _PARSERS and name != "studio_view" and name not in QUEUE_NAMES:
                    return self._error(mid, -32602, f"Unknown tool: {name}")
                args = params.get("arguments")
                if args is not None and not isinstance(args, dict):
                    return self._error(mid, -32602, "Invalid params: arguments must be an object")
                token = meta.get("progressToken")
                step = [0]

                def progress(text: str):
                    if token is None:
                        return
                    step[0] += 1
                    self.send({"jsonrpc": "2.0", "method": "notifications/progress",
                               "params": {"progressToken": token, "progress": step[0], "message": text.strip()}})
                return result(call_tool(name, args or {}, progress))
            return self._error(mid, -32601, f"Method not found: {method}")
        except Exception as e:                             # noqa: BLE001  (never let a fault end the server)
            slog.log_exception(f"mcp {method} failed", log)
            return self._error(mid, -32603, f"Internal error: {e}", {"trace": traceback.format_exc()[-800:]})


def serve(stdin=None, stdout=None) -> int:
    """Read messages until standard input closes. Standard output carries protocol messages only: everything the
    Studio prints is sent to standard error for as long as the server runs."""
    from . import cli_tools
    out = stdout or sys.stdout.buffer
    inp = stdin or sys.stdin.buffer
    sys.stdout = sys.stderr                                # a stray print must never reach the protocol stream
    slog.setup()
    try:                                                   # a client starts us in its own folder (Claude on Windows:
        from . import paths                                # C:\Windows\system32): a relative path must land beside
        os.chdir(paths.get("OUTPUTS"))                     # the app, never on C: (rule 4; found 2026-11-22)
    except OSError as e:
        log.warning("mcp: could not move to the outputs folder (%s)", e)
    cli_tools.console_to_stderr()

    def send(msg: dict):
        out.write((json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        out.flush()

    server = Server(send)
    log.info("mcp server %s started", __version__)
    while True:
        line = inp.readline()
        if not line:                                       # the client closed our input: the way to shut down
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            send(Server._error(None, -32700, "Parse error"))
            continue
        msgs = msg if isinstance(msg, list) else [msg]     # a batch (older revisions): answered one by one
        for m in msgs:
            reply = server.handle(m)
            if reply is not None:
                send(reply)
    log.info("mcp server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(serve())
