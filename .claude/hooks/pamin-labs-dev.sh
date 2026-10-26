#!/bin/sh
# Repository fallback for pamin-labs-dev (InternalDocs pamin-labs-dev/architecture/0.1.0/p3-design.md, section 4).
# - Cloud sessions load no plugins: run the same hook with the binary served by the Worker; the
#   environment's network-secrets proxy adds the Access service token.
# - Locally the organization plugin does the work; if it is not running in this session, deny edits.
# Usage: pamin-labs-dev.sh session-start|pre-tool-use|stop|session-end, with the hook input on stdin.
hook="$1"
input="$(cat)"
home="${PAMIN_LABS_DEV_HOME:-$HOME/.pamin-labs-dev}"

if [ "$CLAUDE_CODE_REMOTE" = true ]; then
  server="https://dev.paminlabs.com/v1/raw/releases"
  tag="$(curl -fsS "$server/LATEST" 2> /dev/null)" || exit 0
  case "$(uname -m)" in
    aarch64 | arm64) asset=pamin-labs-dev-aarch64-unknown-linux-musl ;;
    *) asset=pamin-labs-dev-x86_64-unknown-linux-musl ;;
  esac
  bin="$home/bin/$tag/pamin-labs-dev"
  if [ ! -x "$bin" ]; then
    tmp="$(mktemp -d)" && trap 'rm -rf "$tmp"' EXIT
    curl -fsS -o "$tmp/$asset" "$server/$tag/$asset" && curl -fsS -o "$tmp/SHA256SUMS" "$server/$tag/SHA256SUMS" || exit 0
    (cd "$tmp" && grep " $asset\$" SHA256SUMS | sha256sum -c - > /dev/null) || exit 0
    mkdir -p "$(dirname "$bin")" && chmod +x "$tmp/$asset" && mv "$tmp/$asset" "$bin" || exit 0
  fi
  # No plugin monitor runs here: watch keeps the cache live and sends the heartbeat, and exits with Claude Code.
  [ "$hook" = session-start ] && "$bin" watch > /dev/null 2>&1 &
  printf '%s' "$input" | exec "$bin" hook "$hook"
fi

[ "$hook" = pre-tool-use ] || exit 0
field() { printf '%s' "$input" | sed -n "s/.*\"$1\" *: *\"\([^\"]*\)\".*/\1/p"; }
case "$(field tool_name)" in
  Edit | Write | MultiEdit | NotebookEdit) ;;
  *) exit 0 ;;
esac
session="$(field session_id)"
# The plugin's SessionStart records this session; then it guards edits itself.
grep -qs "\"session_id\":\"$session\"" "$home"/sessions/*.json && exit 0
cat << 'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"pamin-labs-dev is not running in this session, so edits here are not coordinated. Enable the organization plugin pamin-labs-dev (it is Required for Pamin Labs members), make sure the GitHub CLI is signed in, and start a new session."}}
JSON
