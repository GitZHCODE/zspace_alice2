# Current Agent Plan

Status: in progress

## User Intent

Create a Codex-orchestrated urban design iteration loop in alice2/zSpace. The sketch starts from a clean, deliberately underdeveloped base urban grid mesh with neutral faces and simple black massing only. A fixed camera captures screenshots. A local Qwen VLM only describes and critiques each screenshot as an urban designer and assigns a score out of 10. Codex reads that critique, decides the next design/code change, updates the C++ sketch, rebuilds, reruns, logs the iteration, and commits locally.

## Key Constraint

The VLM is a critic only. It must not update code, generate patches, or be treated as the design authority. Codex owns all code changes and design moves.

The local VLM interface is Ollama.

## Participating Agents

- `zspace_agent`: zSpace mesh loading, face traversal, offset geometry, and later SDF-style field logic.
- `alice2_agent`: sketch lifecycle, fixed camera, screenshot capture, display conventions, and run behavior.
- `code_agent`: C++ sketch structure, iteration parameters, logging helpers, and orchestration scripts.
- `build_agent`: build and runtime checks using `alice2\build_with_zspace.bat`.
- `document_agent`: maintain the markdown iteration log when the loop runs.

## Proposed Changes

- Add a new clean sketch, without carrying over urban-evaluator typology logic:
  - `alice2/userSrc/zspace/topology/sketch_zspace_urban_codex_loop.cpp`
- Keep the older urban evaluator intact as a reference.
- Deactivate the previous SDF isocontour sketch so alice2 starts directly in the Codex loop sketch.
- Use `data/input_grid_01.obj` as the initial base mesh.
- Start with simple non-SDF baseline parameters:
  - neutral base mesh color
  - simple massing coverage step
  - simple massing footprint scale
  - minimum building length/depth and maximum aspect ratio
- Do not start with greenery, open-space hierarchy, colored gradients, or SDF fields.
- Add SDF methods later only when Codex interprets VLM critique and decides they are the next design move.
- Street methods should be parametric:
  - treat each mesh face as a plot
  - select only some plot edges as street centerlines; non-selected edges remain plot divisions
  - classify actual mesh edges as primary, secondary, or tertiary
  - draw only the level-0 street contour as zGRAY; keep red primary, blue secondary, green tertiary as internal edge classes
  - derive a zSpace mesh scalar-field SDF from the selected classified edges
  - extract level-0 street geometry with `zFnMeshScalarField::getIsocontour`
  - keep the sampled field mesh behind a `Field Mesh` UI toggle, off by default
  - use parametric road widths in meters: primary roads `p`, secondary `2/3 p`, tertiary `1/3 p`; default `p = 12m`
  - apply global parameter scale `1.0` when converting dimensional parameters to model units
  - expose a live `p` slider from 0m to 100m for width tuning while keeping the selected street topology stable
  - derive secondary and tertiary street classes from the selected primary network
- Building typology SDF methods should use the `plot` class:
  - one `plot` is one mesh face / plot
  - each plot stores ordered vertices and boundary edges
  - boundary edges are tagged as primary road, secondary road, tertiary road, or plot split line
  - each plot owns a connected Type A centerline graph
  - graph vertex count equals plot vertex count
  - each graph edge stores its offset distance from the matching boundary edge
  - graph offset = setback distance + half building width + half road width for primary/secondary/tertiary frontages
  - Type A building width range is 15m to 25m; width is a per-plot parameter
  - Type A edge length is a per-plot parameter constrained to 0.25-0.75; if above 0.75, set it to 1.0
  - current sketch assigns Type A width and edge length randomly per plot until VLM critique drives explicit values
  - Type A setback is 5m on primary/secondary roads and 2m on tertiary/plot-line edges
  - centerline graph geometry is stored as a zSpace `zObjectGraph` on each `plot`
  - Type A SDF A: square at two opposite centerline graph corners, side length `1.2 * building width`
  - Type A SDF B: building-width strips starting from those same corners along incident graph edges, length controlled by the per-plot edge parameter
  - Type A SDF C: per-plot subtractive setback zone from continuous inset boundary half-planes, using variable setback plus road half-width where applicable
  - Type A result SDF = `(A union B) subtract C`, extracted as a level-0 iso-contour; two L-shapes when `edge < 1.0`, full offset center graph when `edge = 1.0`
  - future building typologies should choose frontage/setback/open-side rules from these edge tags and graph edges
- Make the sketch deterministic:
  - fixed top-down orthographic camera
  - white background
  - neutral base mesh
  - hidden grid/axes
  - automatic screenshot after camera settles
  - automatic exit after screenshot
- Add an orchestration script in a later step to:
  - build
  - run
  - find latest screenshot
  - call local Qwen VLM
  - parse critique and score
  - append markdown log
  - let Codex update the sketch
  - locally commit each iteration

## VLM Critique Contract

Qwen should return structured critique only:

```json
{
  "description": "...",
  "urban_design_critique": "...",
  "score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggested_design_directions": ["..."]
}
```

Codex may use `suggested_design_directions` as input, but Codex decides what code changes to make.

## Iteration Log

Create or append:

```text
alice2/output/urban_codex_vlm_iterations.md
```

Each iteration should include:

- iteration number
- screenshot path
- VLM description
- VLM critique
- score out of 10
- Codex interpretation
- key C++ update snippet
- local commit hash

## Build Command

```bat
alice2\build_with_zspace.bat
```

## Run Command

```bat
alice2\run_with_zspace.bat
```

## Acceptance Checks

- New sketch exists separately from the old urban evaluator.
- New sketch is active with `#define __MAIN__`.
- It loads the base grid mesh.
- It sets a deterministic fixed camera.
- It captures a screenshot automatically.
- It exits after screenshot capture.
- Qwen is used only for visual critique.
- Codex remains responsible for all C++ updates.
- Codex preserves minimum building dimensions and acceptable footprint proportions while adding future methods.
- Iteration logs include screenshot, critique, score, Codex decision, key code snippet, and commit hash.
- Each plot stores a deterministic random building type assignment; only the assigned typology is drawn/generated for that plot.
- Building Type A stores per-plot width and edge length parameters; width range is 15m to 25m, edge length varies from 0.25 to 0.75 and snaps to 1.0 above 0.75.
- Building Type B stores per-plot S-graph parameters `X` and `Y`; `X + Y = 1.0`, with current random X from 0.25 to 0.75 and Y computed from X.
- Building Type B stores a per-plot internal-edge parameter from 0.0 to 0.5; it controls how much of the middle S edge is used from both internal vertices, giving two L shapes at 0.0 and the full S connector at 0.5.
- Building Type B displays its S graph in zMagenta and generates an SDF by offsetting the whole graph as exact edge rectangles plus compact vertex fills, then clipping it with the same continuous inset boundary half-planes as Type A.

## Implementation Status

- [x] Replace copied evaluator with a clean simple SDF baseline sketch.
- [x] Enable automatic screenshot and exit.
- [x] Build and fix compile issues.
- [x] Add first-run markdown log scaffold.
- [x] Deactivate previous SDF sketch as the active startup sketch.
- [x] Add VLM critique prompt to the iteration log.
- [x] Record Ollama as the local VLM interface.
- [x] Remove starting gradient/green/open-space logic for neutral baseline.
- [x] Add minimum building dimension and proportion constraints.
- [ ] Add Ollama critique script.
- [ ] Add local git commit step for each iteration.
