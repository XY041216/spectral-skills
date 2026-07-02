# Spectral Skills Install

This project has three release-facing shapes:

- `skills/` and `spectral_core/` are the development source.
- `plugins/spectral-skills/` is the shared distributable skill/plugin image.
- `.agents/plugins/marketplace.json` exposes that image to Codex, while
  `.claude-plugin/` exposes the same bundle to Claude-compatible agents.

Users should install the plugin image, not a single skill folder. The
`spectral-workflow` skill depends on `spectral-reader`, `spectral-qc`,
`spectral-splitter`, `spectral-preprocess`, `spectral-feature`,
`spectral-modeling`, `spectral-optimizer`, `spectral-report`, and
`spectral_core`.

## Release Packaging

Run these commands from the repository root:

```powershell
python install\build_codex_plugin.py --clean --json
python install\check_codex_plugin.py --json
```

The release artifact is:

```text
plugins/spectral-skills/
```

That folder contains:

```text
plugins/spectral-skills/
  .codex-plugin/plugin.json
  .mcp.json
  skills/
    spectral-reader/
    spectral-qc/
    spectral-splitter/
    spectral-preprocess/
    spectral-feature/
    spectral-modeling/
    spectral-optimizer/
    spectral-report/
    spectral-workflow/
  spectral_core/
  scripts/
  shared/
  requirements.txt
```

The repository root also contains Claude-compatible metadata:

```text
.claude-plugin/
  plugin.json
  marketplace.json
```

Before publishing a new release, keep the semantic plugin version in
`install/build_codex_plugin.py` and all skill manifests aligned. Use the PEP 440
equivalent in `pyproject.toml`/`uv.lock` (for example, plugin
`0.1.0-beta.1` maps to Python package `0.1.0b1`). Rebuild the plugin image and
let `check_codex_plugin.py` verify the generated `plugin.json` version.

## Codex Plugin Install

Codex users should install the repository as a plugin marketplace:

Before importing or enabling the plugin, validate the user-level Codex config:

```bash
python install/check_codex_config.py --json
```

If Codex reports an error such as:

```text
Could not load config.toml ... unclosed table, expected ]
```

the failing file is `~/.codex/config.toml` or `%USERPROFILE%\.codex\config.toml`.
It is not a Spectral Skills runtime error. Importing or enabling a plugin makes
Codex reload its global configuration, so a historical malformed entry can
surface at that moment. The most common cause is an old `[projects.'...']`
table whose path was truncated or lost the closing quote and `]`. Fix or remove
the malformed table, rerun the preflight, and only then retry plugin import.

```bash
codex plugin marketplace add https://github.com/XY041216/spectral-skills.git --ref main
codex plugin add spectral-skills@spectral-skills-local-marketplace
```

If the Codex CLI cannot run on Windows, or if `config.toml` contains the
marketplace entry but the plugin does not appear under
`%USERPROFILE%\.codex\plugins\cache`, run the repository installer from the
clone root:

```bash
python install/install_codex_plugin.py --json
```

This installer does both required local steps:

- upserts the `spectral-skills-local-marketplace` and
  `spectral-skills@spectral-skills-local-marketplace` entries in
  `config.toml`, after writing a backup;
- materializes the built plugin image into
  `%USERPROFILE%\.codex\plugins\cache\spectral-skills-local-marketplace\spectral-skills\<version>\`.

Run the verifier after installation:

```bash
python install/check_codex_plugin.py --json
python install/check_codex_config.py --json
```

Codex Desktop users can add the same repository as a custom plugin marketplace:

```text
Marketplace source: https://github.com/XY041216/spectral-skills.git
Branch/ref: main
Plugin: spectral-skills
```

The repository root now also contains `.codex-plugin/plugin.json` and `.mcp.json`
as a GitHub plugin-import entrypoint. That root manifest delegates the Codex
skills path to `plugins/spectral-skills/skills`, which is the built release
image. This prevents a generic GitHub import flow from treating the development
source under root `skills/` as the release unit.

For purely local Codex use, give the user a release bundle that contains both:

```text
.agents/plugins/marketplace.json
plugins/spectral-skills/
```

After cloning this repository, that bundle root is:

```text
<path-to-clone>/spectral-skills
```

The plugin image itself is inside that root:

```text
<path-to-clone>/spectral-skills/plugins/spectral-skills
```

The repository already writes a local marketplace file:

```text
.agents/plugins/marketplace.json
```

It points to:

```json
{
  "source": {
    "source": "local",
    "path": "./plugins/spectral-skills"
  }
}
```

So the user should add the bundle root, not the inner plugin folder, as a local
marketplace source in their Agent/Codex configuration.

For a local Windows Codex setup, the resulting config shape is:

```toml
[marketplaces.spectral-skills-local-marketplace]
source_type = "local"
source = 'C:\path\to\spectral-skills'

[plugins."spectral-skills@spectral-skills-local-marketplace"]
enabled = true
```

After the Agent loads that marketplace/plugin, or after
`install/install_codex_plugin.py` materializes the local cache, a typical
installed path looks like:

```text
C:\Users\<USER>\.codex\plugins\cache\spectral-skills-local-marketplace\spectral-skills\0.1.0-beta.1\
```

Users normally do not hand-edit that cache folder. If Desktop does not create
it automatically, use `python install/install_codex_plugin.py --json` so the
cache copy is created from the release image and validated. Then restart Codex
in a new thread and invoke the skills by name. The main full-chain entry is:

```text
$spectral-skills:spectral-workflow
```

Example user request:

```text
Use $spectral-skills:spectral-workflow to read Tablet_ext_0-3.csv, run QC,
split 6:2:2, apply SNV preprocessing, use no feature reduction, and train a
random_forest_classifier.
```

## Do Not Use Direct GitHub Skill Installer

Spectral Skills is a plugin release unit, not a set of independent GitHub skill
folders. If an import log shows `skill-installer` or
`install-skill-from-github.py --repo XY041216/spectral-skills`, stop that flow
and install the plugin marketplace instead.

Direct skill installation copies isolated folders into `~/.codex/skills` and can
drop shared runtime files, plugin metadata, MCP configuration, and release
checks. Even when individual folders appear to install successfully, the result
is not the supported Spectral Skills distribution. Use:

```bash
codex plugin marketplace add https://github.com/XY041216/spectral-skills.git --ref main
codex plugin add spectral-skills@spectral-skills-local-marketplace
```

## Claude-Compatible Local Install

For Claude-compatible agents that understand Claude plugin metadata, install
from GitHub:

```bash
claude plugin marketplace add XY041216/spectral-skills
claude plugin install spectral-skills@spectral-skills
```

For local use, give the user the same repository/bundle root. The Claude-facing
metadata is:

```text
.claude-plugin/plugin.json
.claude-plugin/marketplace.json
```

The Claude marketplace entry follows the `nature-skills-reference` pattern and
uses:

```json
{
  "source": "./"
}
```

That source points at the bundle root so the agent can discover the
Claude-compatible metadata while reusing the same `plugins/spectral-skills/`
skill image.

## Direct Script Smoke Test

Users can also verify the plugin from the plugin root:

```powershell
cd E:\GPskill\skill8\spectral-skills\plugins\spectral-skills
python skills\spectral-workflow\scripts\run_spectral_workflow.py --input path\to\data.csv --output-dir outputs\workflow_demo --task-goal classification --split-ratio 6:2:2 --preprocess-methods snv --feature-method none --models random_forest_classifier --json
```

For Python dependencies, install the package requirements in the environment
used by Codex or by the direct script:

```powershell
pip install -r requirements.txt
```

Keep `numpy`, `scipy`, and `scikit-learn` versions compatible. If Python raises
a NumPy ABI error while importing SciPy or scikit-learn, rebuild or reinstall
those packages as a matched set before running modeling workflows.

## Mental Model

Think of the release as:

```text
developer source -> build_codex_plugin.py -> plugin metadata + plugins/spectral-skills -> user installs in Codex or Claude-compatible agent -> user calls spectral-workflow
```

`spectral-workflow` is the user-facing full-chain skill. `spectral-skills` is
the release/install unit.
