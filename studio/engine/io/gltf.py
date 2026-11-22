"""glTF 2.0 binary (.glb), OBJ + MTL and PLY, written by the Studio's own code (BUILD_PLAN_DP3D F.1, F.2).

glTF is what Blender and Godot 4 both import best: meshes (positions, normals, texture coordinates, tangents when a
normal map is used), PBR metallic-roughness materials (base colour in sRGB, normal, occlusion-roughness-metallic,
alpha mode), PNG textures inside the file, LODs as separate meshes named NAME_LOD1, NAME_LOD2, metres, Y up, and an
"extras" block saying how the Studio made it (measured, estimated, filled). A small reader is here too: the tests
read every file back, and anyone can check a file without another program. No Qt here.

Tangents are written (computed per vertex from the texture coordinates, Lengyel's method) whenever a material has
a normal map: the glTF specification lets an importer compute them itself, but writing them means Blender and Godot
cannot disagree about it.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field

import numpy as np

GLB_MAGIC = 0x46546C67           # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT, UINT, USHORT, UBYTE = 5126, 5125, 5123, 5121
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass
class Material:
    name: str = "studio"
    base_colour: np.ndarray | None = None      # uint8 (h, w, 3|4) sRGB (alpha in the 4th channel when present)
    normal: np.ndarray | None = None           # uint8 (h, w, 3), OpenGL (green up)
    orm: np.ndarray | None = None              # uint8 (h, w, 3): occlusion, roughness, metallic
    base_factor: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.9
    alpha_mode: str = "OPAQUE"                 # OPAQUE | MASK | BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    extras: dict = field(default_factory=dict)


@dataclass
class Mesh:
    name: str
    positions: np.ndarray                      # (n, 3) float32, metres, Y up
    indices: np.ndarray                        # (m, 3) int
    normals: np.ndarray | None = None          # (n, 3)
    uvs: np.ndarray | None = None              # (n, 2), glTF convention (v down from the top of the image)
    colours: np.ndarray | None = None          # (n, 4) float 0..1 (COLOR_0: foliage sway from the stem)
    material: int | None = None
    extras: dict = field(default_factory=dict)


def vertex_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, np.float64)
    t = np.asarray(tri, np.int64)
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    vn = np.zeros_like(p)
    for k in range(3):
        np.add.at(vn, t[:, k], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(ln > 1e-20, vn / np.maximum(ln, 1e-20), np.array([0.0, 1.0, 0.0]))
    return vn.astype(np.float32)


def tangents(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (n, 4) from the texture coordinates (Lengyel 2001), w = handedness."""
    p = np.asarray(pos, np.float64)
    u = np.asarray(uv, np.float64)
    t = np.asarray(tri, np.int64)
    e1, e2 = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]
    d1, d2 = u[t[:, 1]] - u[t[:, 0]], u[t[:, 2]] - u[t[:, 0]]
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) > 1e-20, 1.0 / np.where(np.abs(r) > 1e-20, r, 1.0), 0.0)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    tan1 = np.zeros_like(p)
    tan2 = np.zeros_like(p)
    for k in range(3):
        np.add.at(tan1, t[:, k], sdir)
        np.add.at(tan2, t[:, k], tdir)
    n = np.asarray(nrm, np.float64)
    tg = tan1 - n * np.sum(n * tan1, axis=1, keepdims=True)
    ln = np.linalg.norm(tg, axis=1, keepdims=True)
    fallback = np.cross(n, np.array([0.0, 0.0, 1.0]))
    fb_ok = np.linalg.norm(fallback, axis=1, keepdims=True) > 1e-6
    fallback = np.where(fb_ok, fallback, np.cross(n, np.array([1.0, 0.0, 0.0])))
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-20)
    tg = np.where(ln > 1e-12, tg / np.maximum(ln, 1e-20), fallback)
    w = np.where(np.sum(np.cross(n, tg) * tan2, axis=1) < 0.0, -1.0, 1.0)
    return np.concatenate([tg, w[:, None]], 1).astype(np.float32)


def _png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a * 255.0) if a.dtype.kind == "f" else a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


class _Builder:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {"asset": {"version": "2.0"}, "buffers": [], "bufferViews": [], "accessors": [], "meshes": [],
                    "nodes": [], "scenes": [{"nodes": []}], "scene": 0}

    def view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.doc["bufferViews"].append(v)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, arr: np.ndarray, ctype: int, kind: str, target: int | None, minmax: bool = False,
                 normalized: bool = False) -> int:
        a = np.ascontiguousarray(arr)
        bv = self.view(a.tobytes(), target)
        acc = {"bufferView": bv, "componentType": ctype, "count": int(a.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in a.min(axis=0)]
            acc["max"] = [float(x) for x in a.max(axis=0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def image(self, arr: np.ndarray, name: str) -> int:
        bv = self.view(_png_bytes(arr))
        self.doc.setdefault("images", []).append({"bufferView": bv, "mimeType": "image/png", "name": name})
        if "samplers" not in self.doc:
            self.doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        self.doc.setdefault("textures", []).append({"source": len(self.doc["images"]) - 1, "sampler": 0})
        return len(self.doc["textures"]) - 1


def write_glb(path: str, meshes: list, materials: list, extras: dict | None = None, generator: str = "") -> str:
    """Write a .glb: one node per mesh (LODs are meshes named NAME_LOD1 ...), materials with embedded PNGs."""
    from ... import __version__
    b = _Builder()
    b.doc["asset"]["generator"] = generator or f"Photographic Studio By Bill {__version__}"
    if extras:
        b.doc["asset"]["extras"] = extras
    mats = []
    normal_maps = set()
    for i, m in enumerate(materials):
        d = {"name": m.name, "pbrMetallicRoughness": {"baseColorFactor": list(m.base_factor),
                                                     "metallicFactor": float(m.metallic),
                                                     "roughnessFactor": float(m.roughness)},
             "doubleSided": bool(m.double_sided), "alphaMode": m.alpha_mode}
        if m.alpha_mode == "MASK":
            d["alphaCutoff"] = float(m.alpha_cutoff)
        if m.base_colour is not None:
            d["pbrMetallicRoughness"]["baseColorTexture"] = {"index": b.image(m.base_colour, f"{m.name}_colour")}
        if m.normal is not None:
            d["normalTexture"] = {"index": b.image(m.normal, f"{m.name}_normal")}
            normal_maps.add(i)
        if m.orm is not None:
            t = b.image(m.orm, f"{m.name}_orm")
            d["occlusionTexture"] = {"index": t}
            d["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": t}
            d["pbrMetallicRoughness"]["metallicFactor"] = 1.0       # the texture's B channel holds metallic (0)
            d["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
        if m.extras:
            d["extras"] = m.extras
        mats.append(d)
    if mats:
        b.doc["materials"] = mats
    for mesh in meshes:
        pos = np.asarray(mesh.positions, np.float32)
        tri = np.asarray(mesh.indices).reshape(-1, 3)
        nrm = mesh.normals if mesh.normals is not None else vertex_normals(pos, tri)
        attrs = {"POSITION": b.accessor(pos, FLOAT, "VEC3", ARRAY_BUFFER, True),
                 "NORMAL": b.accessor(np.asarray(nrm, np.float32), FLOAT, "VEC3", ARRAY_BUFFER)}
        if mesh.uvs is not None:
            attrs["TEXCOORD_0"] = b.accessor(np.asarray(mesh.uvs, np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
            if mesh.material is not None and mesh.material in normal_maps:
                attrs["TANGENT"] = b.accessor(tangents(pos, nrm, mesh.uvs, tri), FLOAT, "VEC4", ARRAY_BUFFER)
        if mesh.colours is not None:
            attrs["COLOR_0"] = b.accessor(np.asarray(mesh.colours, np.float32), FLOAT, "VEC4", ARRAY_BUFFER)
        big = pos.shape[0] > 65535
        idx = tri.astype(np.uint32 if big else np.uint16).ravel()
        prim = {"attributes": attrs, "indices": b.accessor(idx, UINT if big else USHORT, "SCALAR",
                                                         ELEMENT_ARRAY_BUFFER), "mode": 4}
        if mesh.material is not None:
            prim["material"] = int(mesh.material)
        md = {"name": mesh.name, "primitives": [prim]}
        if mesh.extras:
            md["extras"] = mesh.extras
        b.doc["meshes"].append(md)
        b.doc["nodes"].append({"mesh": len(b.doc["meshes"]) - 1, "name": mesh.name})
        if not mesh.name.split("_")[-1].startswith("LOD") or mesh.name.endswith("_LOD0"):
            b.doc["scenes"][0]["nodes"].append(len(b.doc["nodes"]) - 1)
        else:                                        # LODs: in the file, not in the scene (Godot makes its own)
            b.doc["scenes"][0]["nodes"].append(len(b.doc["nodes"]) - 1)
    while len(b.bin) % 4:
        b.bin.append(0)
    b.doc["buffers"].append({"byteLength": len(b.bin)})
    js = json.dumps(b.doc, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(b.bin)
    tmp = path + ".partial"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(js), CHUNK_JSON))
        fh.write(js)
        fh.write(struct.pack("<II", len(b.bin), CHUNK_BIN))
        fh.write(bytes(b.bin))
    import os
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------ reading back
def read_glb(path: str) -> tuple[dict, bytes]:
    """(the JSON document, the binary chunk). Raises ValueError on anything that is not a valid glb."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 20:
        raise ValueError("too short for a glb")
    magic, ver, total = struct.unpack("<III", data[:12])
    if magic != GLB_MAGIC or ver != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary file")
    n, typ = struct.unpack("<II", data[12:20])
    if typ != CHUNK_JSON or n % 4:
        raise ValueError("the first chunk is not JSON")
    doc = json.loads(data[20:20 + n].decode("utf-8"))
    off = 20 + n
    binc = b""
    if off < len(data):
        m, typ2 = struct.unpack("<II", data[off:off + 8])
        if typ2 != CHUNK_BIN:
            raise ValueError("the second chunk is not BIN")
        binc = data[off + 8:off + 8 + m]
    return doc, binc


_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_DT = {FLOAT: np.float32, UINT: np.uint32, USHORT: np.uint16, UBYTE: np.uint8}


def accessor_array(doc: dict, binc: bytes, i: int) -> np.ndarray:
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    dt = _DT[a["componentType"]]
    k = _NCOMP[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = np.frombuffer(binc, dtype=dt, count=a["count"] * k, offset=off)
    return arr.reshape(a["count"], k) if k > 1 else arr


def validate(doc: dict, binc: bytes) -> list:
    """The glTF 2.0 schema's required fields and the references between them. [] when valid, else problems."""
    out = []
    if doc.get("asset", {}).get("version") != "2.0":
        out.append("asset.version must be 2.0")
    for k in ("buffers", "bufferViews", "accessors"):
        if k in doc and not isinstance(doc[k], list):
            out.append(f"{k} must be a list")
    if doc.get("buffers") and doc["buffers"][0].get("byteLength") != len(binc):
        out.append("buffer byteLength does not match the BIN chunk")
    for i, v in enumerate(doc.get("bufferViews", [])):
        if "buffer" not in v or "byteLength" not in v:
            out.append(f"bufferView {i} lacks buffer / byteLength")
        elif v.get("byteOffset", 0) + v["byteLength"] > len(binc):
            out.append(f"bufferView {i} runs past the buffer")
    for i, a in enumerate(doc.get("accessors", [])):
        for k in ("componentType", "count", "type"):
            if k not in a:
                out.append(f"accessor {i} lacks {k}")
    for mi, m in enumerate(doc.get("meshes", [])):
        for pi, p in enumerate(m.get("primitives", [])):
            if "POSITION" not in p.get("attributes", {}):
                out.append(f"mesh {mi} primitive {pi} has no POSITION")
            else:
                a = doc["accessors"][p["attributes"]["POSITION"]]
                if "min" not in a or "max" not in a:
                    out.append(f"mesh {mi}: POSITION needs min and max")
            if "material" in p and p["material"] >= len(doc.get("materials", [])):
                out.append(f"mesh {mi} primitive {pi}: no such material")
    for t in doc.get("textures", []):
        if t.get("source", 0) >= len(doc.get("images", [])):
            out.append("a texture names an image that is not there")
    for n in doc.get("nodes", []):
        if "mesh" in n and n["mesh"] >= len(doc.get("meshes", [])):
            out.append("a node names a mesh that is not there")
    return out


# ------------------------------------------------------------------------------------------ OBJ, PLY (F.2)
def write_obj(path: str, mesh: Mesh, texture_names: dict | None = None) -> list:
    """OBJ + MTL beside it (any tool reads them). ``texture_names``: {"colour": file, "normal": file} already
    written beside it. OBJ's texture v runs up: flipped from glTF's. Returns the paths written."""
    import os
    base = os.path.splitext(path)[0]
    mtl = base + ".mtl"
    name = os.path.basename(base)
    pos = np.asarray(mesh.positions, np.float64)
    nrm = mesh.normals if mesh.normals is not None else vertex_normals(pos, mesh.indices)
    lines = [f"# Photographic Studio By Bill\nmtllib {os.path.basename(mtl)}\no {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in pos]
    if mesh.uvs is not None:
        lines += [f"vt {u:.6f} {1.0 - v:.6f}" for u, v in np.asarray(mesh.uvs, np.float64)]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in np.asarray(nrm, np.float64)]
    lines.append("usemtl studio")
    t = np.asarray(mesh.indices).reshape(-1, 3) + 1
    if mesh.uvs is not None:
        lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in t]
    else:
        lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in t]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    tn = texture_names or {}
    m = ["newmtl studio", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1"]
    if tn.get("colour"):
        m.append(f"map_Kd {tn['colour']}")
    if tn.get("normal"):
        m.append(f"norm {tn['normal']}")
        m.append(f"map_Bump -bm 1 {tn['normal']}")
    with open(mtl, "w", encoding="utf-8") as fh:
        fh.write("# Photographic Studio By Bill\n" + "\n".join(m) + "\n")
    return [path, mtl]


def read_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                v.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                f.append([int(x.split("/")[0]) - 1 for x in p[1:4]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


HOW_CODES = {"measured_both": 2, "measured_one": 1, "filled": 0, "stereo": 3}


def write_ply(path: str, points: np.ndarray, colours: np.ndarray | None = None, how: np.ndarray | None = None) -> str:
    """A binary PLY point cloud: x y z (metres or model units), red green blue, and "how" (uchar: 2 measured two
    ways, 1 one way, 0 filled, 3 multi-view stereo)."""
    p = np.asarray(points, np.float32)
    n = p.shape[0]
    c = np.zeros((n, 3), np.uint8) if colours is None else np.clip(np.asarray(colours), 0, 255).astype(np.uint8)
    h = np.zeros(n, np.uint8) if how is None else np.asarray(how, np.uint8)
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment Photographic Studio By Bill\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar how\nend_header\n")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
                             ("h", "u1")])
    rec["x"], rec["y"], rec["z"] = p[:, 0], p[:, 1], p[:, 2]
    rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
    rec["h"] = h
    with open(path, "wb") as fh:
        fh.write(hdr.encode("ascii"))
        fh.write(rec.tobytes())
    return path


def read_ply(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        data = fh.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode("ascii")
    n = int([ln for ln in head.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    rec = np.frombuffer(data[end:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                           ("b", "u1"), ("h", "u1")], count=n)
    return rec
