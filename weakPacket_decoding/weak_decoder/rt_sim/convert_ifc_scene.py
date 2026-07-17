#!/usr/bin/env python3
"""Convert the hospital IFC model into a compact Sionna RT scene."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import trimesh


DEFAULT_IFC = Path(__file__).resolve().parents[2] / "Buildings" / "B" / "DeKH_B_ICU.ifc"
DEFAULT_OUTPUT = DEFAULT_IFC.parent / "sionna_scene"

# Initial low-UHF material assumptions. Concrete is the ITU-R P.2040 law
# extrapolated slightly below its documented 1 GHz lower bound. These values
# must be treated as sensitivity parameters until measurements are available.
MATERIAL_MODELS = {
    "concrete": {
        "itu_coefficients": (5.24, 0.0, 0.0462, 0.7822),
        "thickness": 0.16,
        "color": (0.60, 0.62, 0.65),
        "extrapolated": True,
    },
    "wood": {
        "itu_coefficients": (1.99, 0.0, 0.0047, 1.0718),
        "thickness": 0.04,
        "color": (0.55, 0.36, 0.20),
        "extrapolated": False,
    },
    "glass": {
        "itu_coefficients": (6.31, 0.0, 0.0036, 1.3394),
        "thickness": 0.008,
        "color": (0.35, 0.65, 0.85),
        "extrapolated": False,
    },
}

TYPE_TO_GROUP = {
    "IfcWall": "concrete",
    "IfcSlab": "concrete",
    "IfcColumn": "concrete",
    "IfcDoor": "wood",
    "IfcWindow": "glass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", type=float, default=487.7e6)
    return parser.parse_args()


def _material_properties(frequency: float) -> dict[str, dict[str, object]]:
    frequency_ghz = float(frequency) / 1e9
    if frequency_ghz <= 0.0:
        raise ValueError("frequency must be positive")
    materials: dict[str, dict[str, object]] = {}
    for name, model in MATERIAL_MODELS.items():
        a, b, c, d = model["itu_coefficients"]
        materials[name] = {
            "relative_permittivity": float(a * frequency_ghz**b),
            "conductivity": float(c * frequency_ghz**d),
            "thickness": float(model["thickness"]),
            "color": tuple(model["color"]),
            "extrapolated": bool(model["extrapolated"]),
        }
    return materials


def _shape_mesh(settings: ifcopenshell.geom.settings, product: object) -> tuple[np.ndarray, np.ndarray]:
    shape = ifcopenshell.geom.create_shape(settings, product)
    vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _merge_meshes(parts: list[tuple[np.ndarray, np.ndarray]]) -> trimesh.Trimesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for part_vertices, part_faces in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + vertex_offset)
        vertex_offset += part_vertices.shape[0]
    mesh = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def _scene_xml(mesh_names: list[str], materials: dict[str, dict[str, object]]) -> str:
    lines = ['<scene version="2.1.0">', "", "    <!-- Radio materials -->"]
    for name, material in materials.items():
        r, g, b = material["color"]
        lines.extend(
            [
                f'    <bsdf type="radio-material" id="mat-custom-{escape(name)}">',
                f'        <float name="relative_permittivity" value="{material["relative_permittivity"]}"/>',
                f'        <float name="conductivity" value="{material["conductivity"]}"/>',
                f'        <float name="thickness" value="{material["thickness"]}"/>',
                '        <float name="scattering_coefficient" value="0.0"/>',
                f'        <rgb name="color" value="{r}, {g}, {b}"/>',
                "    </bsdf>",
                "",
            ]
        )
    lines.append("    <!-- IFC geometry merged by radio material -->")
    for name in mesh_names:
        lines.extend(
            [
                f'    <shape type="ply" id="mesh-{escape(name)}">',
                f'        <string name="filename" value="meshes/{escape(name)}.ply"/>',
                '        <boolean name="face_normals" value="true"/>',
                f'        <ref id="mat-custom-{escape(name)}" name="bsdf"/>',
                "    </shape>",
                "",
            ]
        )
    lines.append("</scene>")
    return "\n".join(lines) + "\n"


def convert(ifc_path: Path, output_dir: Path, frequency: float) -> dict[str, object]:
    ifc_path = ifc_path.resolve()
    output_dir = output_dir.resolve()
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    materials = _material_properties(frequency)

    model = ifcopenshell.open(str(ifc_path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    entity_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []

    for ifc_type, group in TYPE_TO_GROUP.items():
        for product in model.by_type(ifc_type):
            if getattr(product, "Representation", None) is None:
                continue
            try:
                grouped[group].append(_shape_mesh(settings, product))
                entity_counts[ifc_type] += 1
            except Exception as exc:  # pragma: no cover - geometry kernels vary
                skipped.append(
                    {
                        "ifc_type": ifc_type,
                        "name": str(getattr(product, "Name", "")),
                        "error": str(exc),
                    }
                )

    exported: dict[str, dict[str, object]] = {}
    scene_bounds: list[np.ndarray] = []
    for group in sorted(grouped):
        if not grouped[group]:
            continue
        mesh = _merge_meshes(grouped[group])
        mesh_path = mesh_dir / f"{group}.ply"
        mesh.export(mesh_path, file_type="ply", encoding="binary_little_endian")
        scene_bounds.extend([mesh.bounds[0], mesh.bounds[1]])
        exported[group] = {
            "path": str(mesh_path.relative_to(output_dir)),
            "vertices": int(mesh.vertices.shape[0]),
            "triangles": int(mesh.faces.shape[0]),
            "bounds_min": mesh.bounds[0].tolist(),
            "bounds_max": mesh.bounds[1].tolist(),
            "material": materials[group],
        }

    if not exported:
        raise RuntimeError("IFC conversion produced no radio-scene meshes")

    xml_path = output_dir / "hospital.xml"
    xml_path.write_text(_scene_xml(sorted(exported), materials), encoding="utf-8")

    room_centers: list[dict[str, object]] = []
    for space in model.by_type("IfcSpace"):
        if getattr(space, "Representation", None) is None:
            continue
        try:
            vertices, _ = _shape_mesh(settings, space)
        except Exception:
            continue
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        # Place antennas roughly 1.2 m above each room floor.
        center[2] = bounds_min[2] + min(1.2, max(0.2, 0.5 * (bounds_max[2] - bounds_min[2])))
        room_centers.append(
            {
                "name": str(getattr(space, "Name", "")),
                "global_id": str(getattr(space, "GlobalId", "")),
                "position": center.tolist(),
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
            }
        )

    bounds_array = np.vstack(scene_bounds)
    metadata = {
        "source_ifc": str(ifc_path),
        "scene_xml": str(xml_path),
        "material_frequency_hz": float(frequency),
        "entity_counts": dict(sorted(entity_counts.items())),
        "meshes": exported,
        "scene_bounds_min": bounds_array.min(axis=0).tolist(),
        "scene_bounds_max": bounds_array.max(axis=0).tolist(),
        "room_centers": room_centers,
        "skipped": skipped,
        "material_note": (
            f"Initial {float(frequency) / 1e6:.3f} MHz assumptions. Concrete uses "
            "an extrapolation of ITU-R P.2040 below 1 GHz; calibrate material parameters before "
            "treating absolute path loss as predictive."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    metadata = convert(args.ifc, args.output, float(args.frequency))
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
