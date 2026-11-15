# SNP ball-tracking correction workspace

This repository preserves the ZIP-based SNP `ball-tracking` source snapshot,
the local correction scripts, and generated review deliverables.

- `work/source/SNP-ball-tracking/`: original extracted source snapshot.
- `work/*.py`: local extraction, projection, ranking, pilot correction, export,
  and verification scripts used in this workspace.
- `outputs/`: coordinate exports, reports, and the 180-second correction videos.
- `outputs/ball-correction-3min/README_KO.md`: pilot method, scope and limitations.

The source arrived as `SNP-ball-tracking.zip`, without Git history. This local
repository does not claim ancestry with the upstream SNP repository. Its
`origin` URL identifies the requested upstream, but no remote push has been
performed and upstream access was unavailable when this repository was created.

Generated video, compressed coordinate exports and full pilot decision records
use Git LFS. Runtime environments, downloaded weights, temporary caches and raw
source video are excluded. The underlying original assets and first-half
checkpoint store remain under `H:\data`; these are not duplicated in Git.

The pilot scripts currently contain this machine's local asset paths and the
selected 180-second range. They are a reproducible record of this run, not yet a
general-purpose whole-match command. Full-run adaptation must remove pilot-only
range assertions and the explicitly identified one-frame visual override.

Automated candidate selection is not measured ground-truth accuracy. Missing or
uncertain observations remain explicitly unresolved rather than interpolated.
