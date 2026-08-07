"""
recog.synth3d.scene - orchestration. Requires bpy.

Draws a parameter set, builds the scene from the other modules, and hands back
the metadata needed to annotate it. This is the only module that knows the
whole pipeline; everything below it is independently testable.

Two invariants are enforced here rather than downstream, because downstream
cannot detect either failure:

  1. Every non-zero pass_index in the scene has an `id_meta` entry.
     `annotate.boxes_from_mask` silently skips an id it has no metadata for and
     does NOT record it in `dropped`, so a visible object with no entry would
     train as background with no audit trail at all.

  2. Every merge group is class-homogeneous.
     `annotate.merge_group_boxes` adopts `members[0]["class"]` for the whole
     merged box without validating it, so a mixed-class group would silently
     mislabel the result.

Both are construction invariants of assets.py's Items, not data variance, so
violating either is a code bug and is raised rather than warned about.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import bpy

from . import assets as A
from . import bay as B
from . import layout as L
from . import materials as M
from . import world as W
from .catalog import role_of
from .config import Config, VARIANTS, Variant


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials,
                 bpy.data.lights, bpy.data.cameras, bpy.data.images,
                 bpy.data.worlds, bpy.data.node_groups, bpy.data.collections):
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass


def pick_variant(rng: random.Random) -> Variant:
    total = sum(v.weight for v in VARIANTS)
    r = rng.uniform(0, total)
    acc = 0.0
    for v in VARIANTS:
        acc += v.weight
        if r <= acc:
            return v
    return VARIANTS[-1]


def sample_params(rng: random.Random, cfg, overrides: dict = None) -> dict:
    """
    Draw one scene's parameters from `cfg.param_space`.

    Every draw happens whether or not an override will replace it, so a scene
    forced to a particular layout mode still consumes the rng identically to
    the unforced one - that is what keeps seeds comparable across runs.
    """
    ps = cfg.param_space
    modes = ps["layout_mode"]
    total = sum(modes.values())
    r, acc, chosen = rng.uniform(0, total), 0.0, list(modes)[-1]
    for name, weight in modes.items():
        acc += weight
        if r <= acc:
            chosen = name
            break
    p = {
        "n_assemblies": rng.randint(*ps["n_assemblies"]),
        "backdrop": rng.choice(ps["backdrop"]),
        "lighting": rng.choice(ps["lighting"]),
        "layout_mode": chosen,
    }
    # Per-scene exposure. Drawn here rather than read off `render.exposure`
    # because a single fixed value tone-maps every scene identically, so the
    # dataset had no light-LEVEL variation at all - only the backdrop moved
    # the frame mean. It is a scene parameter like any other, so it travels in
    # `params` and lands in the sidecar; `generate3d` hands it to
    # `render.render_beauty`, which falls back to `render.exposure` when it is
    # absent. Absent is the supported case: a config without
    # `param_space.exposure` renders exactly as it did before.
    if "exposure" in ps:
        lo, hi = ps["exposure"]
        p["exposure"] = rng.uniform(lo, hi)
    # Per-scene camera framing multiplier - see world.setup_camera. Larger
    # zoom = wider view = smaller parts. Absent is the supported case and
    # reproduces the old fixed framing exactly.
    if "zoom" in ps:
        lo, hi = ps["zoom"]
        p["zoom"] = rng.uniform(lo, hi)
    # Whether THIS scene may place parts touching or occluding each other, at
    # up to `layout.max_overlap_iou`. Per-scene rather than global so the set
    # keeps a majority of unambiguous, fully-visible examples alongside the
    # crowded ones the real trays show.
    if "overlap_prob" in ps:
        p["allow_overlap"] = rng.random() < float(ps["overlap_prob"])
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def scene_generator(n: int, seed: int, cfg, overrides: dict = None):
    """Yields (index, params, rng). Same seed + index => identical scene."""
    for i in range(n):
        rng = random.Random((seed * 1_000_003) ^ (i + 1))
        yield i, sample_params(rng, cfg, overrides), rng


def _check_group_homogeneous(item: A.Item, gid: str):
    """See invariant 2 in the module docstring."""
    classes = {item.labels.get(o.name) for o in item.objects}
    if len(classes) > 1:
        raise ValueError(
            f"merge group {gid} ({item.asset}/{item.variant}) mixes classes "
            f"{sorted(str(c) for c in classes)}; annotate.merge_group_boxes "
            f"would adopt one of them for the whole merged box and silently "
            f"mislabel it")


def _check_id_meta_covers_scene(id_meta: Dict[int, dict]):
    """See invariant 1 in the module docstring."""
    orphans = sorted(
        (o.pass_index, o.name) for o in bpy.data.objects
        if o.pass_index and o.pass_index not in id_meta)
    if orphans:
        raise ValueError(
            f"objects carry a non-zero pass_index with no id_meta entry: "
            f"{orphans}. annotate.boxes_from_mask would drop them from the "
            f"mask without recording them, so they would train as background.")


def build(params: dict, rng: random.Random, library: A.AssetLibrary,
          cfg: Config) -> Tuple[Dict[int, dict], Dict[int, str], dict]:
    """
    Construct a complete scene.

    Returns:
        id_meta   pass_index -> {"class", "asset", "variant", "role"}
        groups    pass_index -> group key, for merging an assembly's sub-part
                  boxes into a single box
        meta      everything drawn, for the sidecar JSON
    """
    reset_scene()
    library._templates.clear()          # templates lived in the wiped scene

    names = library.names()
    meta = {"params": dict(params), "items": [], "materials": []}

    # ---- instantiate assemblies ------------------------------------------ #
    items: List[A.Item] = []
    for _ in range(params["n_assemblies"]):
        asset = rng.choice(names)
        variant = pick_variant(rng)
        items.extend(library.instantiate(asset, variant, rng))
    if not items:
        return {}, {}, meta

    # ---- materials, drawn per instance ----------------------------------- #
    # Drawn JOINTLY with the backdrop, not independently: a preset that renders
    # within materials.MIN_LUMA_DELTA of this backdrop is dropped from the draw,
    # because a correctly labelled object that cannot be told from what is
    # behind it teaches the detector to hallucinate. The backdrop plane itself
    # is built much further down (it has to be, so a jig scene can sink it below
    # the plate), but which backdrop it will be is already known here.
    backdrop_luma = (cfg.backdrops.get(params["backdrop"]) or {}).get("luma_ref")
    meta["backdrop_luma_ref"] = backdrop_luma
    for item in items:
        for obj in item.objects:
            role = role_of(obj.name)
            mat, drawn = M.for_role(role, rng, cfg, avoid_luma=backdrop_luma)
            M.apply_to_object(obj, mat)
            meta["materials"].append(dict(object=obj.name, role=role, **{
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in drawn.items()}))

    # ---- placement -------------------------------------------------------- #
    pockets: List[L.Pocket] = []
    if params.get("layout_mode") == "jig":
        placements, pockets = L.plan_jig([it.footprint for it in items],
                                         cfg.layout, rng)
    else:
        # `allow_overlap` is drawn per scene by sample_params; a config with no
        # `param_space.overlap_prob` never sets it, and then the configured
        # threshold simply applies to every scene. Either way the default
        # `max_overlap_iou = 0` keeps the old exact non-overlap.
        max_ov = (cfg.layout.max_overlap_iou
                  if params.get("allow_overlap", True) else 0.0)
        # Heights are what stop an overlap being an interpenetration - see
        # layout.plan. Measured here rather than cached on the Item because the
        # parts are still at their template pose and lay_flat has already run,
        # so the bbox is exactly the resting height.
        heights = [(lambda b: b[1].z - b[0].z)(A.group_bbox(it.objects))
                   for it in items]
        placements = L.plan([it.footprint for it in items], cfg.layout, rng,
                            heights=heights, max_overlap_iou=max_ov)

    kept: List[A.Item] = []
    for item, plc in zip(items, placements):
        if plc is None:                              # did not fit; discard
            for o in item.objects:
                bpy.data.objects.remove(o, do_unlink=True)
            continue
        A.place_item(item, plc, rng)
        item.rot_deg = plc.rot_deg
        item.placed_xy = (plc.x, plc.y)
        kept.append(item)
        meta["items"].append({"asset": item.asset, "variant": item.variant,
                              "objects": [o.name for o in item.objects],
                              "merge": item.merge,
                              "placement": plc.as_dict()})
    items = kept
    if not items:
        return {}, {}, meta

    # ---- jig plate and electronics module ---------------------------------- #
    # Both are built here, before pass indices are assigned, so the
    # "for o in bpy.data.objects: o.pass_index = 0" reset below covers them
    # even if a future edit forgets to set one explicitly. The jig plate
    # stays at 0 (unlabelled furniture: it must occlude correctly but never
    # produce an annotation). The PCB does NOT stay at 0 - the pass-index
    # loop below gives item.module_object a real id, because unlike the jig
    # plate it is a class the segmenter is meant to learn, not backdrop.
    #
    # `pockets` is non-empty only in jig mode, and only for items that actually
    # got packed - it is parallel to the non-None placements, which is exactly
    # the set of items that survived the loop above.
    if pockets:
        _, jig_meta = W.build_jig(pockets, rng)
        meta["jig"] = jig_meta

    for item in items:
        if item.variant == "open_case" and any(
                role_of(o.name) == "case" for o in item.objects):
            lo, hi = A.group_bbox(item.objects)
            entry = library.catalog_entry(item.asset)
            module_placement = None
            if entry and entry.get("module_bay_mm") \
                    and entry.get("case_interior_mm"):
                module_placement = B.module_world_placement(
                    item.footprint,
                    tuple(entry["module_bay_mm"]),
                    tuple(entry["case_interior_mm"][:4]),
                    item.rot_deg, item.placed_xy,
                )
            board, pcb_meta = W.build_pcb(
                (lo.x, lo.y, hi.x, hi.y), hi.z, rng,
                module_placement=module_placement, rot_deg=item.rot_deg)
            item.module_object = board
            meta.setdefault("pcbs", []).append(pcb_meta)

            # The placement_area proxy: the complementary strip of the same
            # interior, built off the same catalog data the module used
            # just above. Guarded the same way module_placement is - only
            # when the real bay side is known - because the centred
            # module_placement=None fallback has no true bay side for this
            # to be the complement OF.
            if entry and entry.get("module_bay_mm") \
                    and entry.get("case_interior_mm"):
                placement_placement = B.placement_world_placement(
                    item.footprint,
                    tuple(entry["module_bay_mm"]),
                    tuple(entry["case_interior_mm"][:4]),
                    item.rot_deg, item.placed_xy,
                )
                proxy, proxy_meta = W.build_bay_proxy(
                    placement_placement, hi.z, rng, rot_deg=item.rot_deg)
                item.bay_object = proxy
                meta.setdefault("bays", []).append(proxy_meta)

                # Obstructions sitting on that same bay: real cartridges
                # (IMG_4426) have thermal adhesive, foam pads, tape crosses
                # and printed labels in the bay, none of which the CAD
                # models. Sampled in the LOCAL frame `placement_placement`
                # was built from - not in world space - and only carried
                # into world coordinates by `obstruction_world_poses`, the
                # same rotate-the-centre-point trick `placement_world_
                # placement` uses, so a glue blob on a rotated cartridge
                # rotates with it instead of staying at a fixed world (x, y)
                # while the bay turns out from under it.
                placement_rect = B.placement_rect_local(
                    item.footprint,
                    tuple(entry["module_bay_mm"]),
                    tuple(entry["case_interior_mm"][:4]),
                )
                local_poses = B.sample_obstructions(
                    placement_rect, cfg.obstruction, rng)
                world_poses = B.obstruction_world_poses(
                    local_poses, item.rot_deg, item.placed_xy)
                built = W.build_obstructions(world_poses, hi.z, rng)
                item.obstruction_objects = [o for o, _ in built]
                meta.setdefault("obstructions", []).extend(
                    m for _, m in built)

                # Cells seated in the SAME bay, at the packer's own pitch.
                # The deployed robot fills a cartridge one cell at a time, so
                # its camera sees partly-filled bays for most of every run -
                # a generator that only ever produced empty or sealed cases
                # never showed the segmenter the case it exists to handle.
                # Sampled in the SAME local frame `placement_rect` above is
                # already in, then carried into world space by
                # `seated_cell_world_poses` - the identical local-then-world
                # split `sample_obstructions`/`obstruction_world_poses` use,
                # so a seated cell turns WITH its cartridge instead of
                # staying axis-aligned to the world while the bay rotates
                # under it.
                #
                # `local_poses` (the obstructions just sampled above, in the
                # SAME local frame) are rasterised into a forbidden-cell grid
                # and handed to the packer, so a seated cell cannot land on
                # top of adhesive/foam/tape/a label - physically impossible,
                # and it is the exact case obstructions exist to teach the
                # segmenter to avoid. A densely obstructed bay legitimately
                # seats fewer cells, or none; seated_cell_poses returns
                # whatever fits with no error.
                if rng.random() < cfg.layout.p_seated:
                    cell_w, cell_h = W.SEAT_CELL_FOOTPRINT_M
                    cap = max(1, int(
                        (placement_rect[2] - placement_rect[0]) *
                        (placement_rect[3] - placement_rect[1]) /
                        (cell_w * cell_h)))
                    want = max(1, int(
                        cap * rng.uniform(*cfg.layout.seated_frac)))
                    forbidden = B.obstruction_forbidden_mask(
                        local_poses, placement_rect, B.SEAT_MM_PER_CELL)
                    local_seats = B.seated_cell_poses(
                        placement_rect, cell_w, cell_h, want, rng,
                        forbidden_mask=forbidden,
                        mm_per_cell=B.SEAT_MM_PER_CELL)
                    world_seats = B.seated_cell_world_poses(
                        local_seats, item.rot_deg, item.placed_xy)
                    item.seated_objects = W.seat_cells(
                        library, item.asset, world_seats, hi.z, rng,
                        cfg, backdrop_luma=backdrop_luma)
                    meta.setdefault("seated_cells", []).append(
                        {"asset": item.asset, "n": len(item.seated_objects)})

    # ---- pass indices ----------------------------------------------------- #
    for o in bpy.data.objects:
        o.pass_index = 0
    id_meta: Dict[int, dict] = {}
    groups: Dict[int, str] = {}
    objects_by_id: Dict[int, list] = {}
    pid = 0
    for gi, item in enumerate(items):
        gid = f"item{gi}"
        if item.merge:
            _check_group_homogeneous(item, gid)
        for obj in item.objects:
            pid += 1
            obj.pass_index = pid
            id_meta[pid] = {"class": item.labels.get(obj.name),
                            "asset": item.asset, "variant": item.variant,
                            "role": role_of(obj.name)}
            objects_by_id[pid] = [obj]
            if item.merge:
                groups[pid] = gid

        # The electronics module (world.build_pcb's board) is scene
        # content built here rather than a CAD sub-part, so it is not in
        # item.objects/labels and gets its own pid instead of going
        # through the loop above. Its gold ports and copper inductor are
        # given the SAME pid as the board rather than staying at the 0
        # build_pcb left them on: annotate.boxes_from_mask keys the
        # instance mask by pass_index, so a different (or zero) id on a
        # child would cut a hole out of the board's mask exactly where its
        # most recognisable features are - the opposite of what "label it"
        # is for. A shared id also needs no merge-group entry: one id
        # already renders as one connected mask, unlike the per-sub-part
        # ids the case/cell loop above merges by box only.
        if item.module_object is not None:
            pid += 1
            item.module_object.pass_index = pid
            children = list(item.module_object.children)
            for child in children:
                child.pass_index = pid
            id_meta[pid] = {"class": "electronics_module", "asset": item.asset,
                            "variant": item.variant, "role": "module"}
            objects_by_id[pid] = [item.module_object] + children

        # The placement_area proxy is scene content built here too, not a
        # CAD sub-part, so it gets its own pid for the same reason the
        # module does. Unlike the module it has no children to fold in - it
        # is a single plane.
        if item.bay_object is not None:
            pid += 1
            item.bay_object.pass_index = pid
            id_meta[pid] = {"class": "placement_area", "asset": item.asset,
                            "variant": item.variant, "role": "placement_area"}
            objects_by_id[pid] = [item.bay_object]

        # Obstructions sitting on the bay proxy: adhesive, foam, tape and
        # labels. Each is its own instance - the segmenter is meant to
        # learn individual blob/pad/strip shapes, not one glued-together
        # region - so each gets its OWN pid rather than sharing the bay's
        # or each other's.
        for obj in getattr(item, "obstruction_objects", None) or []:
            pid += 1
            obj.pass_index = pid
            id_meta[pid] = {"class": "obstruction", "asset": item.asset,
                            "variant": item.variant, "role": "obstruction"}
            objects_by_id[pid] = [obj]

        # Cells seated in the bay: real 18650 geometry, so they carry the
        # SAME "battery" class as a loose cell - the deployed camera cannot
        # tell a seated cell from a loose one, and neither should the
        # segmenter's label set. Each is its own instance, same reasoning
        # as obstructions above (no shared/merged id).
        for obj in getattr(item, "seated_objects", None) or []:
            pid += 1
            obj.pass_index = pid
            id_meta[pid] = {"class": "battery", "asset": item.asset,
                            "variant": item.variant, "role": "cell"}
            objects_by_id[pid] = [obj]

    # ---- backdrop, camera, lighting --------------------------------------- #
    # Built AFTER the jig so it can be sunk below the plate. Pocket floors are
    # cut below z = 0; a backdrop left at z = 0 sits in front of them, so every
    # recess rendered as flat plate colour. Scatter scenes keep z = 0, which is
    # the plane their parts were dropped onto.
    plane_size = max(cfg.layout.area) * 6.0
    backdrop_z = meta.get("jig", {}).get("backdrop_z", 0.0)
    _, bd = W.build_backdrop(params["backdrop"], rng, cfg, size=plane_size,
                             z=backdrop_z)
    meta["backdrop"] = bd

    all_objs = [o for it in items for o in it.objects]
    _, top = A.group_bbox(all_objs)
    cam, cam_meta = W.setup_camera(cfg.camera, cfg.layout, cfg.render.res,
                                   rng, top_z=top.z,
                                   zoom=params.get("zoom", 1.0))
    meta["camera"] = cam_meta
    meta["lighting"] = W.setup_lighting(params["lighting"], rng, cam.location,
                                        cfg)
    meta["objects_by_id"] = {k: [o.name for o in v]
                             for k, v in objects_by_id.items()}

    # The backdrop is created after the reset loop above and sets its own
    # pass_index to 0; this is the belt-and-braces check that nothing else did.
    _check_id_meta_covers_scene(id_meta)

    return id_meta, groups, meta
