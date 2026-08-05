"""Blender-based synthetic detection-dataset generator.

Submodules deliberately are NOT included here: config, catalog, layout and
annotate must stay usable without Blender, while assets, materials,
world, render and scene require bpy. Import what you need directly.
"""

__version__ = "1.0.0"
