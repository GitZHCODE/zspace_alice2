# Urban Codex Loop Sketch Implementation

This note documents the current implementation in `topology/sketch_zspace_urban_codex_loop.cpp`.

## Display Layers

- Base mesh: neutral plot/parcel face display, toggleable with `Base Mesh`.
- Height map: duplicate face-only display of the input mesh, toggleable with `Height Map`.
- Height map color domain: blue to magenta.
- Height map value: attractor-based gradient evaluated at each face center.
- Street field mesh: debug scalar field display, toggleable with `Field Mesh`.
- Street contour: grey level-0 street SDF contour.
- Buildings: black per-plot isomeshes.
- Building center graphs: magenta effective graph display.

## Plot Assignment

- Each input mesh face is treated as one plot.
- Each plot stores:
  - face id and center
  - ordered plot vertices
  - boundary edges
  - boundary edge type: primary road, secondary road, tertiary road, or plot split line
  - plot use: `Building` or `Green`
  - building type weights for Type A, B, C, and D
- Current plot-use assignment is deterministic-random.
- Typology anchor plots are forced to `Building`.
- Future VLM critique can replace the deterministic assignment with explicit per-plot use and typology weights.

## Streets

- Street hierarchy is assigned from selected mesh edges.
- Primary width is controlled by slider `p`.
- Secondary and tertiary widths are derived from `p`.
- Street SDF is generated from the classified street edge network.
- The level-0 street contour is drawn as grey geometry.

## Building Width Controls

- `minW` slider controls minimum building width in meters.
- `maxW` slider controls maximum building width in meters.
- Width values are clamped so `maxW` stays greater than `minW`.
- Current model conversion uses `modelUnitsPerMeter = 1.0` and global scale `0.1`.

## Building SDF Generation

- Building SDFs are generated per plot.
- Each building plot creates a local scalar field based on plot bounds.
- Green plots skip building graph and SDF generation.
- Plot setback clipping uses inward boundary half-planes.
- Buildings are extracted as isomeshes and drawn black.

## Building Types

### Type A

- Uses the plot centerline graph.
- Generates L-shaped or loop-like graph segments from selected graph corners.
- Edge length fraction controls how much of the adjacent graph edges is used.

### Type B

- Uses an S-shaped graph derived from two opposite centerline graph corners and their parallel edges.
- `X + Y = 1` logic is retained through the type B fractions.
- Internal edge fraction controls how much of the connecting edge is active.

### Type C

- Uses two parallel centerline graph edges.
- Edge fraction controls how much of those parallel edges is active.

### Type D

- Uses the centerline graph as a closed polygon.
- The building shape is a polygon SDF of that centerline graph.
- The polygon SDF is clipped by the same plot setback half-planes used by the other building types.
- The effective graph display for Type D is the closed centerline loop.

## Typology Blending

- Typology anchors hold weighted parameters for Type A, B, C, and D.
- Plot typology genes are blended from anchors using plot graph distances.
- A, B, and C contribute graph segments to the effective graph overlay.
- D is handled as a polygon SDF when it is the dominant type weight.

## Future VLM Hooks

- Plot use assignment: building, green, or later other civic/open-space categories.
- Typology weights: Type A/B/C/D per plot.
- Building width range.
- Type-specific parameters such as edge fractions, S-graph internal fraction, and orientation.
- Height-field attractor position and intensity.
