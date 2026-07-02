# Column Role Cases

These examples describe role evidence for spectral data reading. They are not
hard rules.

## Column Role Recognition

- Sample ID evidence maps to `sample_id.column`.
- Classification evidence maps to `label.column` and `task_hint=classification`.
- Regression evidence maps to `target.column` and `task_hint=regression`.
- Metadata evidence maps to `metadata.columns`.
- Spectral boundary evidence maps to `spectral_columns.columns` or
  `spectral_columns.start` / `spectral_columns.end`.
- Ambiguous roles should ask for the smallest necessary confirmation before
  reading.

## Sample ID Candidates

`sample_id`, `sample`, `id`, `name`, `Unnamed: 0`, `样本编号`, and `编号` may be
sample IDs when values are unique and not numeric sequence-only.

## Label Candidates

`label`, `class`, `category`, `type`, `group`, `类别`, `等级`, and `品种` may be
classification labels.

## Target Candidates

`target`, `y`, `value`, `content`, `concentration`, `含量`, `浓度`, and analyte
names may be regression targets.

## Metadata Candidates

`No.`, `index`, `remark`, `batch`, `date`, `operator`, `仪器`, `批次`, `备注`, and
`序号` should not enter X unless explicitly confirmed.

## Spectral Boundaries

Spectral columns are often contiguous band-like names. Boundaries are uncertain
when numeric metadata sits next to spectral numeric headers.
