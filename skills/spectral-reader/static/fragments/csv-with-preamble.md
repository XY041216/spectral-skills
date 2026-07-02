# CSV With Preamble

Some CSV/TXT exports contain leading description lines before the true table.

Common preamble markers include `#`, `//`, `;`, `[Header]`, instrument names,
export timestamps, and operator notes.

The Agent should inspect `leading_preamble_candidates`,
`delimiter_candidates`, and `header_row_candidates`. Identify the likely real
header row from column-like tokens, delimiter consistency, and following
numeric rows. Use `skiprows` and `header_row` as internal read settings.

Skipping leading lines changes interpretation and must be confirmed. Do not
allow scripts to blindly treat the first line as the table header.

## Reading Semantics

This scene usually affects `skiprows`, `header_row`, `delimiter`, and
`data_start_row`.

Add `confirm_preamble_skip` to `required_confirmations` unless the user has
already confirmed the skipped lines. If no plausible table header exists after
the preamble, return `blocked` with a header-related reason.
