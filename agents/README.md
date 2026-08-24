# alice2 Agent Workflow

This folder defines the working agreement for agents helping users create alice2 sketches with zSpace geometry.

Keep the top-level agent set intentionally small:

- `alice2_sketch_agent.md`: sketch lifecycle, helper code, display flow, zSpace integration, and local topology notes.
- `build_agent.md`: zSpace-enabled alice2 build verification and compiler/linker triage.
- `document_agent.md`: sketch documentation only when the user explicitly asks for docs, writeups, notes, screenshots, or explanations.
- `docs/`: durable plans, sketch writeups, and images.

## zSpace API Source

Do not maintain a copied zSpace API guide in this repo.

Use the `zspace-core` skill and source docs from the sibling core repo:

```text
..\zspace_core\.codex\skills\zspace-core\SKILL.md
..\zspace_core\agents\querying_and_using_api.md
..\zspace_core\agents\extending_zspace_core.md
```

## Working Loop

1. Read `alice2_sketch_agent.md` for sketch and helper-code conventions.
2. Read the `zspace-core` skill when zSpace API or data-structure choices matter.
3. Update `agents/docs/current_plan.md` when a durable plan is useful.
4. Implement the smallest coherent change.
5. Run the build through `build_agent.md`.
6. Fix errors and rebuild until clean, unless an external dependency blocks the build.

Build command:

```bat
alice2\build_with_zspace.bat
```

After a clean build, tell the user to run:

```bat
alice2\run_with_zspace.bat
```

## Refinement Rule

When a prompt reveals a new convention, mistake, or preferred workflow, update the smallest relevant guide. zSpace API rules belong in `zspace_core`; alice2 sketch rules belong here.
