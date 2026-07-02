# Reader Execution Boundary

Reader may:

- preview files and folders to understand layout;
- execute a confirmed read plan;
- transpose confirmed samples-as-columns matrices;
- align confirmed external label files by sample ID;
- standardize X, y, sample IDs, band axis, and metadata;
- run minimal structural assertions;
- write only standard output files and `data_contract.json`.

Reader must not:

- remove samples or bands;
- split data;
- preprocess spectra;
- create features;
- train models;
- optimize pipelines;
- generate final reports;
- write debug archives, package manifests, summaries, or audit directories as
  final outputs.

Runtime validation is a reliability guard. Passing assertions do not become
user-facing reports; failures return concise `blocked` reasons.
