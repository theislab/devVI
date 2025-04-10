# devVI

[![Build][badge-build]][build]
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Codecov][badge-codecov]][codecov]

[badge-build]: https://img.shields.io/github/actions/workflow/status/theislab/devVI/build.yaml?branch=main&style=flat&logo=github&label=Build%20checks
[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/devVI/test.yaml?branch=main&style=flat&logo=github&label=Tests
[badge-docs]: https://img.shields.io/readthedocs/devvi/latest.svg?label=Read%20the%20Docs
[badge-codecov]: https://codecov.io/gh/theislab/devVI/graph/badge.svg?token=fDsBzRodRK

Integration of developmental scRNA-seq data. Inspired by scPoli.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install devVI:

<!--
1) Install the latest release of `devVI` from [PyPI][]:

```bash
pip install devVI
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/devVI.git@main
```

<!--
## Release notes

See the [changelog][].
-->

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/theislab/devVI/issues
[tests]: https://github.com/theislab/devVI/actions/workflows/test.yaml
[build]: https://github.com/theislab/devVI/actions/workflows/build.yaml
[documentation]: https://devVI.readthedocs.io
[changelog]: https://devVI.readthedocs.io/en/latest/changelog.html
[api documentation]: https://devVI.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/devVI
[codecov]: https://codecov.io/gh/theislab/devVI
