# Mode Selection

- User asks "recommend methods" or gives only data size/profile:
  `recommend_from_profile`.
- User names one method but asks for best parameter:
  `tune_method`.
- User asks which feature/modeling/preprocess step is best:
  `compare_step`.
- User asks for fully automatic best pipeline:
  `optimize_pipeline` with a confirmed trial budget.

Before executing `compare_step`, ask the user to confirm the full comparison
card: candidates, fixed stages, validator model, primary metric, auxiliary
metrics, and max_trials. The validator model is not neutral; for example,
preprocess comparison under SVM means "best under the confirmed SVM validator",
not "globally best for every model".

For modeling comparison, preview must also prove that every validation-only
trial has a complete locked parameter set. Candidate-list input should receive
the optimizer's deterministic fixed defaults automatically. A custom space
with missing fields must return every missing field and recommended value in
`needs_confirmation`; do not materialize or execute the plan.

Before that confirmation, do not write files. Do not create a fixed SNV package,
candidate-space file, optimizer output directory, or trial manifest. If a
machine-readable card is useful, call optimizer with `--preview-only`; the
result must report `files_written=[]` and `directories_created=[]`.

If any candidate method expands parameters, confirm the parameter grid in the
same card before execution. Confirming method names alone is not enough:
`PCA / PLS / VIP / KBest / SPA` must also show the exact component or `top_k`
grid and the expanded trial count. Use `--confirm-parameter-grid` only after
that confirmation.

For optimizer execution, offer four explicit budget profiles:

- `quick`: smoke/interactive search; one point per traditional feature method.
- `regular`: recommended bounded traditional grid; backward-compatible alias
  `recommended` is accepted.
- `extended`: broader traditional grids plus repeated-holdout follow-up.
- `deep`: opt-in deep embedding plus classifier search. It requires explicit
  device, dimension, epoch, augmentation, seed, metric, and expanded-trial
  confirmation and is never selected automatically.

If the user fixes an upstream preprocess step such as SNV for feature
comparison, pass it to optimizer as `--fixed-preprocess-methods snv`. Do not
run `spectral-preprocess` yourself. The optimizer executor must prepare SNV
inside each trial after confirmation.

If a later comparison reuses the validator model from an earlier comparison,
either ask whether to reuse it or record
`validator_model.source=inherited_from_previous_user_confirmed_comparison` and
the previous confirmation stage in the optimizer contract.

Before executing `optimize_pipeline`, ask the user to confirm the candidate
space policy and the trial budget. Do not pass `--confirm-budget` unless the
user explicitly confirmed the expanded trial count.

For automatic compact search, show included and excluded methods before
execution. Say when `pls_latent_variables`, SPA/CARS/UVE, optional boosting, or
experimental small-sample models are excluded from the compact default and that
an extended/custom search is available.

The compact classification default includes `pls_latent_variables` because
PLS-LV is a core spectral feature extraction method. If budget pressure removes
it from a custom compact card, say so explicitly and offer extended/custom
search.

If the user already specified every method and parameter, do not invoke
optimizer; route directly through `spectral-workflow`.

After a small-sample single-holdout search selects a best pipeline, recommend a
locked stratified repeated-holdout check with 5 or 10 repeats and mean/std
Macro-F1. Treat near-perfect train accuracy as an additional overfit signal.
