# Security

## Report a vulnerability

Report it privately, through
[a GitHub security advisory](https://github.com/simonvanlierde/msc-thesis-ie/security/advisories/new). Don't open a public issue or pull request for it:
anyone can read those before there is a fix.

Please include:

- What the problem is and where it is: a file, a function or a dependency.
- Steps to reproduce it, or a proof of concept.
- The impact you expect, and the release or commit you saw it in.

I acknowledge a report within 14 days, then tell you whether I can reproduce it and when I
expect a fix.

## Fixes and disclosure

A fix ships in a new release. When it is out, I publish the GitHub security advisory for
it, with a CVE where one applies, and credit you unless you'd rather not be named. The
CHANGELOG entry for that release links the advisory.

## Scope

<!-- Say what this project is and what it is not: a server, stored user data, a
     published package. Keep the lists below in line with it. -->

In scope:

- The code in this repository and what it builds.
- The CI and release workflows in `.github/workflows`.
- The dependencies in the lockfiles, when the problem affects this project.

Out of scope:

- A problem in a dependency that doesn't affect this project. Report it upstream.
