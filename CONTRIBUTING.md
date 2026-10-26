# Contributing

Thank you for helping with PaminMemory. Several people and their coding agents work here at the same
time, so changes are coordinated before code is written.

## External contributors

1. Pick an open issue and comment `/claim` on it. The issue is assigned to you for 7 days; comment
   `/unclaim` if you stop.
2. Open a pull request that closes the issue (`Closes #<number>` in the description).
3. CI checks that the pull request closes an issue you claimed. Claims without a pull request expire
   after 7 days.

Please follow [AGENTS.md](AGENTS.md) for code quality, tests and measurements. Commits and pull request
titles use [Conventional Commits](https://www.conventionalcommits.org/) and are written in English.

## Pamin Labs members

Your agent declares an intent with the `pamin-labs-dev` plugin before editing. Pull requests name it
with a line `Intent: I-<number>` in the description, from a branch named `<type>/i-<number>-<slug>`.
