# Band Quality

Band quality checks identify risky wavelength or wavenumber positions.

## Candidate Checks

- constant bands;
- low-variance bands;
- high missing-rate bands;
- nonmonotonic or irregular band axis warnings inherited from reader;
- abnormal band profile candidates.

## Action

Do not remove bands by default. Band removal changes model evidence and must be
confirmed by the user.

If bands are removed, rewrite `X.csv` and `band_axis.csv` together and preserve
alignment. Then update `data_contract.json` with compact QC action metadata.
