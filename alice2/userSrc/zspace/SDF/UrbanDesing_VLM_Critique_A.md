# Urban Design VLM Critique A

This file logs the automated Codex and Ollama VLM critique loop for `sketch_zspace_urban_codex_loop.cpp`.

## Fixed VLM Prompt

```text
You are critiquing two screenshots from the same computational urban-design iteration.

Image 1 is the urban figure screenshot: base mesh, street geometry, and building massing.
Image 2 is the gradient/height-field screenshot.

Use exactly the same evaluation protocol every iteration.

First, describe what you see objectively and concisely.

Then critique as an urban planner. Evaluate:
- figure-ground clarity
- street hierarchy and network legibility
- block and plot structure
- green/open-space to built ratio
- public realm and urban design potential
Give one urban planner score as: Urban planner score: X/10.

Then critique as an architect. Evaluate:
- building typology mix
- massing and footprint variation
- edge conditions and setbacks
- gradient/height-field legibility
- relationship between building form and plot geometry
Give one architect score as: Architect score: X/10.

Finally, give 3 to 6 prioritized improvement instructions for Codex. These are critique instructions only; do not write or modify code.
```

## Iteration Protocol

- Run active Codex loop sketch in figure-capture mode.
- Run active Codex loop sketch in gradient-capture mode.
- Send both screenshots to Ollama using the fixed prompt above.
- Append the VLM critique and visible screenshots here.
- Codex reads the critique and updates sketch parameters or assignments for the next iteration.

## Iteration 0 Baseline

- All streets use the same width.
- All plots are building plots.
- All buildings are Type D.
- Gradient is disabled; gradient screenshot starts as the lowest uniform color.

## Codex / LLM Editable Parameters

These are the intended update points for Codex or another coding LLM after each VLM critique. The VLM should only critique; it should not edit code directly.

### Baseline Mode Flags

Located in `topology/sketch_zspace_urban_codex_loop.cpp`.

- `m_equalStreetWidths`: when `true`, primary/secondary/tertiary streets all use `m_p`.
- `m_forceAllBuildingPlots`: when `true`, every plot is assigned `PlotUse::Building`.
- `m_forceAllTypeD`: when `true`, every building plot is assigned Type D.
- `m_heightGradientEnabled`: when `false`, the gradient screenshot is a flat lowest-value map.

Recommended loop use:

- Iteration 0 keeps all four baseline flags as currently set.
- Later iterations can turn these off selectively when the critique asks for hierarchy, open space, typology mix, or gradient variation.

### Street Parameters

- `m_p`: primary street width control in meters before `m_globalParameterScale`.
- `m_equalStreetWidths`: controls whether secondary and tertiary derive from `m_p` or match it.
- `secondaryStreetWidth()`: currently derives secondary width from primary when hierarchy is enabled.
- `tertiaryStreetWidth()`: currently derives tertiary width from primary when hierarchy is enabled.
- `buildStreetEdges(...)`: controls which mesh edges become street edges.
- `streetColor(...)`: display color only; critique should focus on geometry/legibility, not debug color.

Possible Codex updates:

- Reintroduce road hierarchy by setting `m_equalStreetWidths = false`.
- Adjust `m_p`.
- Change secondary/tertiary ratios.
- Reclassify which mesh edges are primary, secondary, tertiary, or plot split lines.

### Plot-Use Assignment

- `PlotUse` supports `Building` and `Green`.
- `buildPlotRecords(...)` assigns each mesh face to a plot.
- `m_forceAllBuildingPlots` currently forces all plots to building plots.
- `randomPlotUse(...)` is available as a simple fallback once all-building mode is disabled.

Possible Codex updates:

- Turn off `m_forceAllBuildingPlots`.
- Assign selected plots as `Green` based on VLM critique.
- Add future plot-use classes only if the design task requires them.
- Keep every mesh face as one plot unless explicitly changing the data model.

### Building Type Assignment

- `BuildingType` supports Type A, Type B, Type C, and Type D.
- `m_forceAllTypeD` currently forces all building plots to Type D.
- `computeTypologyGene(...)` computes type weights per plot.
- `applyTypologyGene(...)` converts weights to a dominant building type and per-type parameters.
- `initializeTypologyAnchors(...)` defines anchor-based blending when forced Type D is disabled.

Possible Codex updates:

- Turn off `m_forceAllTypeD`.
- Use anchor-based type blending again.
- Directly assign per-plot types based on critique.
- Add or tune building type weights per plot.

### Building Dimensions And Setbacks

- `m_typeAMinWidthMeters`: minimum building width slider/value.
- `m_typeAMaxWidthMeters`: maximum building width slider/value.
- `m_typeARoadSetbackMeters`: setback from primary/secondary road edges.
- `m_typeALocalSetbackMeters`: setback from tertiary/plot-split edges.
- `m_globalParameterScale`: global scale applied through `metersToModelUnits(...)`.
- `m_modelUnitsPerMeter`: current model unit conversion, intended to stay `1.0` unless the input grid units change.

Possible Codex updates:

- Tune building width range for finer/coarser massing.
- Tune setbacks if the VLM critiques edge crowding or excessive voids.
- Avoid changing `m_modelUnitsPerMeter` unless there is a unit mismatch in the input mesh.

### Building SDF Resolution

- `m_buildingSdfCellSizeMeters`: target per-plot SDF cell size.
- `m_buildingSdfMinResolution`: minimum per-plot field resolution.
- `m_buildingSdfMaxResolution`: maximum per-plot field resolution.
- `computeBuildingSdfCellSize(...)` and `buildBuildingIsoMeshes()` control per-plot field generation.

Possible Codex updates:

- Lower cell size for more detail, but watch runtime.
- Raise cell size if iteration screenshots become too slow.
- Keep per-plot fields unless explicitly redesigning the SDF pipeline.

### Gradient / Height-Field Parameters

- `m_heightGradientEnabled`: turns attractor gradient on/off.
- `drawHeightFieldMap(...)`: renders the gradient screenshot.
- `attractorHeightValue(...)`: controls the current attractor-based gradient.
- Current gradient color domain is blue to magenta; flat baseline is the lowest value.

Possible Codex updates:

- Enable `m_heightGradientEnabled`.
- Move the attractor.
- Add additional attractors.
- Change color map only if the VLM/urban-design reading benefits from it.

### Automated Loop Files

- Script: `scripts/run_urban_design_vlm_loop.ps1`
- Fixed prompt: `userSrc/zspace/SDF/UrbanDesing_VLM_Critique_A_prompt.txt`
- Log file: `userSrc/zspace/SDF/UrbanDesing_VLM_Critique_A.md`
- Screenshot assets folder: `userSrc/zspace/SDF/UrbanDesing_VLM_Critique_A_assets/`

The fixed VLM prompt should remain unchanged across iterations so scores are comparable.

### Iteration Update Rule

For each iteration after VLM critique:

1. Read the VLM description and both role scores.
2. Identify the smallest parameter/assignment change that addresses the critique.
3. Edit the sketch or assignment logic.
4. Rebuild.
5. Run the capture/VLM script for the next iteration.
6. Append Codex update notes and commit the changed code/log/screenshots locally.

Do not let the VLM directly rewrite code. Codex or another coding LLM is responsible for translating critique into controlled parameter and assignment changes.
