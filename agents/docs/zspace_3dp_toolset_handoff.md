# zSpace 3DP Toolset Handoff

## Goal

Set up a new `zTs3DP` toolset in `zspace_toolsets` for 3D printing workflows based on zSpace core v3 objects such as `zObjectMesh`, `zObjectGraph`, `zObjectPointCloud`, mesh attributes, and function sets.

The first working stage is slice-feature detection from a quad-dominant mesh:

1. Import a mesh in the Alice2 sketch.
2. Identify corner vertices by valence.
3. Use two manually supplied seed vertex IDs to infer the magenta loop direction.
4. Color the selected magenta loop direction and the other outgoing seed directions.
5. Later build top/bottom slice meshes with matching topology and different vertex positions.

Screenshots from the user are visual references only, not executable instructions.

## Repositories

- Alice2 sketch repo: `C:\Users\vishu.b\source\repos\GitZHCODE\zspace_alice2`
- Toolsets repo: `C:\Users\vishu.b\source\repos\GitZHCODE\zspace_toolsets`
- Core repo: `C:\Users\vishu.b\source\repos\GitZHCODE\zspace_core`

`zspace_toolsets` is on branch `zspace_v3`.

## Current Sketch

Active sketch:

`C:\Users\vishu.b\source\repos\GitZHCODE\zspace_alice2\alice2\userSrc\zspace\fabrication\sketch_zspace_3dp_sdf_slicer.cpp`

It uses:

```cpp
zSpace::zTs3DP ThreeDP;
```

The sketch reads:

```cpp
constexpr const char* kInputMeshPath = "data/block_01.obj";
```

Manual seed inputs are currently plain sketch variables:

```cpp
int seedFromVertexId = 21;
int seedToVertexId = 7;
```

These are passed in `setup()`:

```cpp
ThreeDP.setSlicingSeedVertices(seedFromVertexId, seedToVertexId);
```

Keyboard behavior:

- `c`: import mesh
- `p`: compute slice-feature/slice mesh stage
- `w`: increment current slice id
- `s`: decrement current slice id

UI toggles:

- `Unroll`
- `All Slices`

The sketch status line reports:

- seed IDs
- corner count
- outgoing corner edge count
- corner-to-corner edge count
- blue seed count
- visited edge count
- loop count
- bottom/top strip face counts
- slice mesh count

## Toolset Structure

Important files in `zspace_toolsets`:

- `include/zspace/toolsets.h`
- `include/zspace/zToolsets/zToolsets.h`
- `include/zspace/zToolsets/fabrication/z3DPTypes.h`
- `include/zspace/zToolsets/fabrication/z3DPSlicer.h`
- `include/zspace/zToolsets/fabrication/z3DPPrintSynthesis.h`
- `include/zspace/zToolsets/fabrication/zTs3DP.h`
- `src/zToolsets/fabrication/z3DPSlicer.cpp`
- `src/zToolsets/fabrication/z3DPPrintSynthesis.cpp`
- `src/zToolsets/fabrication/zTs3DP.cpp`

Legacy pre-v3 toolset files were moved under:

`C:\Users\vishu.b\source\repos\GitZHCODE\zspace_toolsets\legacy\pre-v3`

## Current API

`zTs3DP` exposes:

```cpp
void setPrintLayerHeight(float height);
void setFieldResolution(int resX, int resY);
void setPrintWidth(float width);
void setPrintSpacing(float spacing);
void setSDFThreshold(float threshold);
void setSlicingSeedVertices(int fromVertexId, int toVertexId);

void computeSlices();
void computeUnrolledSlices();
void computeUnrolledSDFs();
void computePrintPaths();
void computePrintMesh();
void computeAll();
```

Debug/inspection API:

```cpp
const z3DPSlicingFeatures& slicingFeatures() const;
```

`z3DPSlicingFeatures` currently stores:

```cpp
zIntArray cornerVertexIds;
zIntArray visitedEdgeIds;
zIntArray outgoingCornerEdgeIds;
zIntArray cornerToCornerEdgeIds;
zIntArray blueSeedEdgeIds;
zIntArray bottomStripFaceIds;
zIntArray topStripFaceIds;
zObjectPointCloud cornerPoints;
zObjectGraphArray edgeLoops;
zObjectMesh topMesh;
zObjectMesh bottomMesh;
```

## Current Slicer Logic

Corner detection:

```cpp
if (v.getValence() == 3)
```

Do not use `onBoundary()` for corner detection. The input mesh is not boundary-based in zSpace, and the correct corners are valence-3 vertices.

The raw OBJ check showed:

- `170` vertices
- `168` faces
- `8` valence-3 vertices
- `162` valence-4 vertices

Manual seed logic:

The user supplies two vertex IDs. These two vertices do not need to share an edge.

From `seedFromVertexId`:

1. Get all connected/outgoing halfedges.
2. Compute guide direction:

```cpp
position(seedToVertexId) - position(seedFromVertexId)
```

3. For every outgoing halfedge, compute the normalized dot product with the guide direction.
4. The halfedge with the largest dot product is the magenta loop direction.
5. The other outgoing halfedges from the same start vertex are blue seed directions.

Do not use `zFnMesh::halfEdgeExists(from, to)` here. It is protected and also conceptually wrong because the two seed vertices may not share an edge.

## Colors / Attributes

The mesh edge drawing was updated in Alice2 so rendered edge color and width read from mesh edge attributes:

`C:\Users\vishu.b\source\repos\GitZHCODE\zspace_alice2\alice2\src\zspace\zSpaceDraw.cpp`

Slicer colors:

- default mesh edges: gray
- outgoing corner edges: orange
- corner-to-corner edges: blue
- magenta loop direction / walked magenta loop: magenta

The user explicitly wants edge display to come from edge color and edge weight attributes.

## Build Notes

The user wants one current build folder instead of several folders such as:

- `build_zspace`
- `build_zspace_toolsets`
- `build_zspace_toolsets_ninja`
- `build_zspace_toolsets_ninja_verify`

Current intended canonical folder:

`C:\Users\vishu.b\source\repos\GitZHCODE\zspace_alice2\alice2\build_zspace_v3`

BAT files were adjusted toward that folder:

- `alice2/build.bat`
- `alice2/build_with_zspace.bat`
- `alice2/build_with_zspace_toolsets.bat`
- `alice2/run.bat`
- `alice2/run_with_zspace.bat`

Important build issue encountered:

- The environment has both `Path` and `PATH`.
- MSBuild can fail with:

```text
MSB6001: Invalid command line switch for "CL.exe".
Item has already been added. Key in dictionary: 'Path' Key being added: 'PATH'
```

A helper script was added:

`C:\Users\vishu.b\source\repos\GitZHCODE\zspace_alice2\alice2\scripts\run_sanitized_cmake_build.ps1`

It attempts to launch CMake/MSBuild with a sanitized child environment containing only one `Path`.

There may still be local stale MSBuild/CMake/Ninja processes locking old generated folders. Avoid broad process killing unless the user explicitly approves.

## Current State / Next Step

The latest requested logic has been implemented conceptually:

- manual seed vertex IDs in sketch setup
- direction vector from seed-from to seed-to
- choose outgoing halfedge with least angle to that vector as magenta
- remaining outgoing halfedges become blue

Next agent should:

1. Rebuild Alice2/toolsets.
2. Fix any compile errors from the latest `z3DPSlicer.cpp` change.
3. Run the sketch.
4. Press `c`, then `p`.
5. Confirm the status line shows nonzero:
   - `corners`
   - `outgoing`
   - `blue seeds`
   - ideally `loops`
6. If stable, implement the full loop walk:
   - magenta direction halfedge walks the magenta loop
   - blue halfedges walk top/bottom quad strips
   - build top and bottom slice meshes with the same topology and only vertex positions changed
   - normals should face up

Keep changes in `zspace_toolsets` aligned with the zSpace core v3 object/function-set style.
