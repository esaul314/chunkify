# Public repository publication privacy gate

This is a public project. Every contributor and agent must keep private information
and unrelated operational material out of its commits, branches, tags, pull requests,
release assets and exported patches. A local commit or green test suite does not
establish that a branch is safe to publish.

## Publication is blocked until review is complete

Before publishing current local work, complete and record the review below. An
unresolved finding or uninspected attachment keeps publication blocked. Continue
local development separately; do not treat a clean newest commit as clearance for
its ancestors. Do not push all branches or tags to bypass the reviewed scope.

Public source should contain reusable implementation, portable documentation and
synthetic or explicitly approved redistributable fixtures. Keep the following in
private storage outside tracked source:

- Actual processed documents, excerpts, borrowed titles/author names, annotations,
  rendered pages, extracted text, source manifests and private evaluation datasets.
- Identifiable source/generation fingerprints, retrieval queries/results, review
  questions/answers, run receipts and material-specific acceptance reports.
- Local usernames, absolute personal paths, hostnames, network addresses, deployment
  commands, service inventory and unrelated operational history.
- Credentials, tokens, cookies, private keys, authentication exports and secrets in
  configuration, logs, metadata or generated files.

Use invented names and generated documents for regressions. Preserve accurate
maintainer attribution, license information and ordinary public project metadata;
these are distinct from identifying people or documents processed privately.
Ignoring a file prevents accidental new staging; it does not remove tracked data
or erase data already present in Git history.

## Required depth of review

1. **Freeze scope.** Verify the actual destination remote and its current public
   base with a bounded read-only query. Record full base/tip object IDs and all
   branches/tags intended for publication privately. Do not rely only on stale
   tracking refs or the last few log entries.
2. **Inspect complete outgoing history.** Review every commit, commit message and
   introduced path/blob between that base and each proposed tip, including merges,
   renamed/deleted files and intermediate content removed by later commits. Inspect
   author/committer metadata and release notes. A tip-only diff is insufficient.
3. **Check inherited content.** Review the resulting tracked tree and recent public
   history for inherited private or unrelated content. A range scanner that excludes
   its base cannot certify the base or an already-published artifact. State those
   exclusions explicitly and investigate any finding privately.
4. **Review content, not only patterns.** Combine secret/private-identifier scans
   with a manual review of source, comments, tests, fixtures, agent instructions,
   documentation, configuration, notebooks and generated outputs. Keep actual
   identifying search terms and detailed findings in a private review policy/report.
   Absence of regex matches is not proof of absence of private information.
5. **Open attachments safely.** Inventory binaries, Base64 fixtures, embedded data,
   archives, submodules and large-file pointers. Inspect their decoded content and
   metadata in a private bounded environment. Unreadable or unreviewed content
   prevents clearance; do not silently classify it as clean.
6. **Review exports.** Inspect the complete patch/PR/release payload, including diff
   context, deleted lines, headers, filenames and attachments. Removing a private
   line can still disclose it in the exported deletion hunk.
7. **Bind the result.** Record scope, checks, manual dispositions, remaining unknowns
   and final artifact hashes privately. Verify reviewed refs and destination have
   not changed immediately before publishing. New commits, amended history or new
   attachments require renewed review of the affected scope.

## Findings and remediation

Stop publication when private or irrelevant material is found. Preserve the local
originals and review evidence privately. Prefer a separate reviewed public branch
or backport from the verified public base, carrying only generic changes and
synthetic fixtures. Review required dependency commits as carefully as the feature.
A later cleanup commit does not sanitize earlier history, and its patch may expose
the removed content again.

Do not rewrite shared history, force-push, publish sensitive findings in issues,
or delete source/evidence as an automatic remediation. If content is already
public, record the exposure privately and obtain direction for the appropriate
containment or history repair. Handle any exposed credentials through their
normal incident process. Do not claim that local cleanup retracts a disclosure.

## Completion record

A release review is complete only when all intended outgoing history and payloads
have a documented disposition, every attachment is reviewed, unresolved findings
are zero, and the exact final refs/artifacts match the reviewed identities. Retain
a private receipt; public documentation must not repeat the sensitive findings.
This gate is a required review procedure, not an installed Git hook or automatic
technical prevention mechanism.
