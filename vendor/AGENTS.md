# AGENTS.md — `vendor/`

Repo-wide contracts: [`../AGENTS.md`](../AGENTS.md) (§10 for the security
model, §12 for what is deliberately absent).

**Vendored third-party code. Do not edit anything here** to "fix" warnings,
style, or behaviour — changes are overwritten on the next vendor update and
make future upgrades unreviewable. If you believe a vendored file is broken,
raise it in the PR description instead of patching it silently.

| Directory | Upstream | Why it is here |
|---|---|---|
| `cpp-httplib/` | [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) | header+source HTTP server for `tts-server`. Built via `add_subdirectory(vendor/cpp-httplib)` in the root `CMakeLists.txt`. |
| `yyjson/` | [ibireme/yyjson](https://github.com/ibireme/yyjson) | JSON parse/write for `tts-server`, compiled as a static lib in the root `CMakeLists.txt` with warnings suppressed (`/W0` or `-w`) — the suppression is deliberate, not an oversight. |

Two contract notes that live with these files:

- **cpp-httplib is built without SSL.** No OpenSSL is linked anywhere in this
  repo, so `tts-server` can never speak HTTPS on its own — the security model
  is "bind `127.0.0.1` (the default) or put an authenticating TLS reverse
  proxy in front". Do not add TLS by editing `httplib.h`; the build has no
  way to satisfy it.
- `tts-server` caps request bodies with `set_payload_max_length(32 * 1024 *
  1024)` and `set_read_timeout(60)` — those are *our* limits configured in
  `src/tts-server.h`, not defaults of the vendored library.

Licenses: `cpp-httplib/LICENSE` (MIT) and yyjson's own MIT header. Keep both
intact when updating.
