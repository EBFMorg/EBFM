<!--
SPDX-FileCopyrightText: 2025 EBFM Authors

SPDX-License-Identifier: BSD-3-Clause
-->

# Guide for releasing a new version of EBFM

Creating a new tag starting with `v*` on `main` will automatically create a new release with the specified version.

In order to create a release, please follow these steps:

* [Draft a new release.](https://github.com/EBFMorg/EBFM/releases/new)
  * The release Tag should be set according to the rules of semantic versioning. Please refer to the [existing tags](https://github.com/EBFMorg/EBFM/tags).
  * Note: It might be a good idea to create a release candidate (tag ending with `rcX`, e.g. `v0.0.1rc5`) before doing the real release. Please also tick the box to mark the release as a pre-release in this case.
  * Use the version number as release title (i.e. the same as your tag).
  * Briefly summarize the content of the release.
* If everything looks good you can publish the release.
* Please check the [release workflow on github](https://github.com/EBFMorg/EBFM/actions/workflows/release.yaml) and the [history on PyPI](https://pypi.org/project/EBFM/#history).

## Notes on the release workflow and settings

The following rules exist to ensure that releases are not created by accident:

* The branch `main` is protected by the [branch protection rule](https://github.com/EBFMorg/EBFM/settings/rules) "Protect main".
* The creation and deletion of release tags (`v*`) is restricted via the [tag protection rule](https://github.com/EBFMorg/EBFM/settings/rules) "Protect version tag".
* The release workflow may only be triggered on the protected `main` branch due to the [envorinment `pypi`](https://github.com/EBFMorg/EBFM/settings/environments). This requires the rule defined under ["branches"](https://github.com/EBFMorg/EBFM/settings/branches)
