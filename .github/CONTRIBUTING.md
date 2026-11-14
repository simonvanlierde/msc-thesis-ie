# Contributing

Report bugs and propose changes through the [issue forms](https://github.com/simonvanlierde/msc-thesis-ie/issues/new/choose).
For a bug, include the commit, command, expected result and error output. Remove personal data and API keys.
Read the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities as [SECURITY.md](SECURITY.md) describes.

## Development

Follow the [development guide](../docs/development.md) for setup, checks and hooks.
`just check` runs lint, formatting, tests and the dashboard build. `just types` runs the type checker.
A behaviour change needs a test; a bug fix needs a test that fails before the fix.
Use Ruff for Python and Biome for TypeScript. `just fmt` applies their fixes.

Keep pull requests focused. Discuss a large change in an issue first.
Use Conventional Commits, such as `fix: validate building geometry`, for commits and pull request titles.
Sign off each commit with `git commit -s` to confirm your right to contribute under the project licence.
Fork pull requests have their sign-offs checked in CI.

## Releases

Record user-visible changes under `Unreleased` in [CHANGELOG.md](../CHANGELOG.md).
The maintainer reviews the full release, updates citation metadata and publishes the release.
