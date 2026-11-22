"""Build a Blender scene from the Studio's files (BUILD_PLAN_DP3D F.3). Runs INSIDE Blender (Python standard library
and bpy only), never inside the Studio:

    blender --background --python build_material.py -- --surface NAME_surface_Studio.json [--out FOLDER]
    blender --background --python build_material.py -- --model NAME_model_Studio.json [--out FOLDER]
    blender --background --python build_material.py -- --glb NAME_Studio.glb [--out FOLDER]

A surface: a plane the texture's real size (or 1 m wide when the size is not known), subdivided and displaced by the
measured height (the scale from the JSON), with a Principled BSDF material: the colour (or the albedo when there is
one), the normal map in OpenGL mode, the ambient occlusion multiplied in when it was made, and the fine relief as a
second bump at strength 0 (estimated: off until the artist turns it up). A model or a glb: imported, with the
cameras the Studio's solver placed (a check anyone can see). The scene is saved as NAME_Studio.blend beside the
files (or in --out); nothing else is touched. The Studio's "Open in Blender" and the RPG Forge's Blender runner
(forge/blenderrun.py, which passes --out) both run it. GPL-3.0, like the Studio.
"""
import json
import os
import sys


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"surface": None, "model": None, "glb": None, "out": None, "no_save": False}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k in ("--surface", "--model", "--glb", "--out") and i + 1 < len(argv):
            out[k[2:]] = argv[i + 1]
            i += 2
            continue
        if k == "--no-save":
            out["no_save"] = True
        i += 1
    return out


def _image(path):
    import bpy
    img = bpy.data.images.load(path, check_existing=True)
    return img


def _tex(nodes, path, colour_space, x, y):
    n = nodes.new("ShaderNodeTexImage")
    n.image = _image(path)
    try:
        n.image.colorspace_settings.name = "sRGB" if colour_space == "srgb" else "Non-Color"
    except TypeError:                                # an unusual colour-management config
        pass
    n.location = (x, y)
    return n


def surface_scene(json_path, out_dir=None, save=True):
    """The displaced, textured plane of a surface JSON. Returns a dict of what was built."""
    import bpy
    with open(json_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    folder = os.path.dirname(os.path.abspath(json_path))
    files = doc.get("files", {})
    W, H = doc["size"]["width"], doc["size"]["height"]
    lat = doc.get("lateral", {}) or {}
    mmpx = lat.get("mm_per_pixel")
    width_m = (W * mmpx / 1000.0) if mmpx else 1.0
    height_m = width_m * H / float(W)
    hgt = doc.get("height", {}) or {}
    rng_mm = (hgt.get("range") or {}).get("mm")
    disp_m = (rng_mm / 1000.0) if rng_mm else 0.05 * width_m    # relative: 5% of the width, said in the scene
    base = os.path.basename(json_path).replace("_surface_Studio.json", "")
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    ob = bpy.context.active_object
    ob.name = base + "_surface"
    ob.scale = (width_m, height_m, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.new(base + "_material")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    col_key = "albedo" if "albedo" in files else "colour"
    built = {"object": ob.name, "width_m": width_m, "height_m": height_m, "displacement_m": disp_m,
             "maps": [], "colour_from": col_key if col_key in files else None}
    if col_key in files:
        tc = _tex(nodes, os.path.join(folder, files[col_key]), "srgb", -700, 300)
        if "ao" in files:
            ta = _tex(nodes, os.path.join(folder, files["ao"]), "linear", -700, 0)
            mix = nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 1.0
            links.new(tc.outputs["Color"], mix.inputs[6])
            links.new(ta.outputs["Color"], mix.inputs[7])
            links.new(mix.outputs[2], bsdf.inputs["Base Color"])
            built["maps"].append("ao")
        else:
            links.new(tc.outputs["Color"], bsdf.inputs["Base Color"])
        built["maps"].append(col_key)
    last_normal = None
    if "normal" in files:
        tn = _tex(nodes, os.path.join(folder, files["normal"]), "linear", -700, -300)
        nm = nodes.new("ShaderNodeNormalMap")
        nm.space = "TANGENT"
        nm.location = (-350, -300)
        links.new(tn.outputs["Color"], nm.inputs["Color"])
        if (doc.get("normal") or {}).get("convention", "").startswith("DirectX"):
            nm["studio_note"] = "DirectX normal map: flip green before use"
        last_normal = nm
        built["maps"].append("normal")
    if "detail" in files:                            # estimated fine relief: a second bump, strength 0 (off)
        td = _tex(nodes, os.path.join(folder, files["detail"]), "linear", -700, -600)
        bump = nodes.new("ShaderNodeBump")
        bump.location = (-200, -500)
        bump.inputs["Strength"].default_value = 0.0
        links.new(td.outputs["Color"], bump.inputs["Height"])
        if last_normal is not None:
            links.new(last_normal.outputs["Normal"], bump.inputs["Normal"])
        last_normal = bump
        built["maps"].append("detail (strength 0)")
    if last_normal is not None:
        links.new(last_normal.outputs[0], bsdf.inputs["Normal"])
    ob.data.materials.append(mat)
    if "height" in files:                            # real geometry: subdivide and displace by the measured height
        img = _image(os.path.join(folder, files["height"]))
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        tex = bpy.data.textures.new(base + "_height", type="IMAGE")
        tex.image = img
        tex.extension = "EXTEND"
        sub = ob.modifiers.new("Studio subdivision", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = sub.render_levels = 7
        dm = ob.modifiers.new("Studio height", "DISPLACE")
        dm.texture = tex
        dm.texture_coords = "UV"
        dm.strength = disp_m
        dm.mid_level = 0.0
        built["maps"].append("height (displacement)")
    ob["studio_surface"] = json.dumps({"how": doc.get("method"), "height_units": hgt.get("units"),
                                       "near_end": hgt.get("near_end"), "coverage": doc.get("coverage")})
    if save:
        built["blend"] = _save(json_path, base, out_dir)
    return built


def model_scene(path, out_dir=None, save=True):
    """Import a model's glb (from its JSON, or a glb given directly) and place the solved cameras."""
    import bpy
    glb, doc = path, {}
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        glb = os.path.join(os.path.dirname(os.path.abspath(path)), (doc.get("files") or {}).get("glb", ""))
    bpy.ops.import_scene.gltf(filepath=glb)
    imported = [o.name for o in bpy.context.selected_objects]
    cams = 0
    for c in doc.get("cameras", []) or []:
        if not c.get("placed"):
            continue
        cd = bpy.data.cameras.new(c.get("name", "view"))
        if c.get("focal_mm"):
            cd.lens = float(c["focal_mm"])
        co = bpy.data.objects.new(c.get("name", "view"), cd)
        bpy.context.scene.collection.objects.link(co)
        # the solver's camera: x right, y down, z forward (OpenCV); Blender's: x right, y up, looking down -z
        R = c["rotation_world_from_camera"]
        t = c["position"]
        import mathutils
        Rm = mathutils.Matrix(R)
        flip = mathutils.Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        M = (Rm @ flip).to_4x4()
        M.translation = mathutils.Vector(t)
        # glTF Y up -> Blender Z up, as the importer does
        conv = mathutils.Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
        co.matrix_world = conv @ M
        cams += 1
    base = os.path.splitext(os.path.basename(glb))[0].replace("_Studio", "")
    built = {"imported": imported, "cameras": cams}
    if save:
        built["blend"] = _save(glb, base, out_dir)
    return built


def _save(src, base, out_dir):
    import bpy
    d = out_dir or os.path.dirname(os.path.abspath(src))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, base + "_Studio.blend")
    n = 2
    while os.path.exists(p) and not _ours(p):
        p = os.path.join(d, f"{base}_Studio-{n}.blend")
        n += 1
    bpy.ops.wm.save_as_mainfile(filepath=p, check_existing=False)
    return p


def _ours(p):
    """A .blend this script wrote (it carries a mark in the scene): safe to replace."""
    return os.path.basename(p).find("_Studio") >= 0 and os.path.isfile(p + ".studio")


def main():
    import bpy
    a = _args()
    for o in list(bpy.data.objects):                 # a clean scene (a background Blender starts from the default)
        bpy.data.objects.remove(o, do_unlink=True)
    if a["surface"]:
        r = surface_scene(a["surface"], a["out"], not a["no_save"])
    elif a["model"] or a["glb"]:
        r = model_scene(a["model"] or a["glb"], a["out"], not a["no_save"])
    else:
        print("build_material.py: give --surface JSON, --model JSON or --glb FILE after --")
        return 2
    if r.get("blend"):
        with open(r["blend"] + ".studio", "w", encoding="utf-8") as fh:
            json.dump({"written_by": "Photographic Studio By Bill", "built": r}, fh)
    print("STUDIO_RESULT " + json.dumps(r))
    return 0


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("build_material.py runs inside Blender: blender --background --python build_material.py -- ...")
        sys.exit(2)
    sys.exit(main())
