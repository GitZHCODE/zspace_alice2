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
