# Urban Blend Sketch Implementation

This note documents `topology/sketch_zspace_urban_blend.cpp`.

## Purpose

- Separate active sketch for testing typology blending on the input mesh faces.
- The original `sketch_zspace_urban_codex_loop.cpp` is kept inactive.
- No green/open-space assignment is used in this sketch; every mesh face is a building plot.

## Active Anchor Blend

- Top left anchor: Type C, edge fraction `1.0`.
- Bottom left anchor: Type B, edge fraction `0.5`, internal edge fraction `0.5`.
- Bottom right anchor: Type D.
- Top right anchor: Type A, edge fraction `0.6`.

## Display

- Buildings are generated as black per-plot isomeshes.
- Effective building centerline graphs are drawn in magenta above the building mesh.
- Base mesh, height map, and field mesh remain toggleable.

## Notes

- Plot typology is blended from anchors using plot graph distance over the actual input mesh faces, not a predefined UV grid.
- This sketch is intended as the current sandbox for improving graph blending before reconnecting it to VLM-driven plot assignment.
