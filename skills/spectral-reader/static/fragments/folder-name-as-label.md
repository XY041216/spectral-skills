# Folder Or File Name Labels

Use this fragment when one file represents one sample and labels are encoded in
folder names or file names.

## Folder Name Labels

- Parent folder names may be class labels when the folder structure groups
  samples by class.
- Use folder labels only when requested or clearly confirmed.
- File stems become sample IDs unless another sample ID rule is provided.

## File Name Labels

- File prefixes before the first underscore may be class labels, such as
  `A_s001.csv` and `B_s003.csv`.
- Use file-name labels only when requested or clearly confirmed.
- If a file name cannot be parsed by the selected simple rule, return
  `blocked`.

## Sample Files

Each sample file must provide a band-axis column and a value column, or the user
must specify them. Band axes must match across sample files. Do not resample or
interpolate inside the reader.
