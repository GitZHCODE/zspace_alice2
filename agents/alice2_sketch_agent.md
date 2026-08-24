# alice2 Sketch Agent

This agent owns alice2 sketch structure, reusable sketch code, display flow, and zSpace integration inside `zspace_alice2`.

## zSpace Source Of Truth

Use the repo-local `zspace-core` Codex skill for zSpace API and method-building rules.

Canonical zSpace docs live in the sibling core repo:

```text
..\zspace_core\.codex\skills\zspace-core\SKILL.md
..\zspace_core\agents\querying_and_using_api.md
..\zspace_core\agents\extending_zspace_core.md
```

Do not keep a copied zSpace API guide in this repo. If zSpace API conventions change, update `zspace_core` first.

## Sketch Structure

- Understand the `alice2::ISketch` lifecycle: `setup`, `update`, `draw`, `cleanup`, and input callbacks.
- Place generated sketches under `alice2/userSrc/`, preferably `alice2/userSrc/zspace/` for zSpace sketches.
- Keep one active sketch with `#define __MAIN__` unless the user asks otherwise.
- Use lowercase `z` for zSpace-related sketch names, wrapper types, and filenames.
- Prefer names like `zSpaceHalfedgeTraversalSketch`, `zDisplaySetting`, `zDisplayMeshSetting`, `zSpaceDraw.h`, and `zSpaceObject.cpp`.

## zSpace Usage In alice2

Use zSpace for geometry construction and alice2 for display:

```cpp
#include <zspace/interface.h>
#include <alice2.h>

zSpace::zObjectMesh mesh;
zSpace::zFnMesh fn(mesh);

scene().draw(mesh);
```

Use optional display settings when needed:

```cpp
alice2::zDisplayMeshSetting display;
display.showEdges = true;
display.edgeWidth = 2.0f;
scene().draw(mesh, display);
```

Rules:

- Prefer public zSpace `zObject*`, `zFn*`, `zIt*`, `zIO`, and `zDisplay*` APIs.
- Do not expose adapter or conversion internals to sketch users.
- Do not add draw methods to zSpace objects.
- Do not substitute alice2-only geometry types when the user asks for zSpace geometry.
- For zSpace SDF and scalar-field sketches, use zSpace fields such as `zObjectMeshScalarField` and `zFnMeshScalarField`.
- Use alice2's local `ScalarField2D` only when the user explicitly asks for an alice2-only compute-geometry example.

## Helper Code

- Extract repeated sketch logic into focused helper functions.
- Keep helpers small, readable, and easy for future prompts to modify.
- Prefer public zSpace API calls over direct storage access.
- Avoid broad refactors unless the user asks.
- Do not run git commit or git push commands unless the user explicitly requests them.

Prefer helper shapes such as:

```cpp
void createPyramid(zSpace::zObjectMesh& mesh);
void createPolylineGraph(zSpace::zObjectGraph& graph);
void drawLabels(alice2::Renderer& renderer);
```

Avoid helpers that hide too much state or make the sketch hard to read.

## Mesh And Topology Notes

When creating a `zSpace::zObjectMesh` with `zSpace::zFnMesh::create`, keep `faceCounts` and `faceConnects` aligned and check face winding against outward normals.

For a pyramid with base vertices `0, 1, 2, 3` and apex `4`, wind the base opposite to the side faces:

```cpp
zSpace::zIntArray faceCounts = {4, 3, 3, 3, 3};
zSpace::zIntArray faceConnects = {
    0, 3, 2, 1,
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
    3, 0, 4
};
```

For halfedge traversal sketches, draw highlighted halfedges as short directed arrows:

- arrow segment length: about `0.1 * edgeLength`
- arrow position: centered on the corresponding edge
- arrow offset: slightly toward the owning face
- current: black
- next: cyan
- previous: green
- symmetry/twin: magenta

Avoid artificial z stepping; keep arrows at the level of their edge or face.

## Prompt Workflow

1. Parse the user prompt.
2. Update `agents/docs/current_plan.md` when the work needs a durable plan.
3. Read the `zspace-core` skill and core docs when zSpace API choices matter.
4. Patch the smallest useful code surface.
5. Hand build verification to `build_agent.md`.
6. Tell the user to run `alice2\run_with_zspace.bat` after a clean build.

When the user corrects a convention, update this file or the relevant core skill/doc so the rule is not rediscovered later.
