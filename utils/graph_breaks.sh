#!/usr/bin/env bash
# Catalogue torch.compile graph breaks / recompiles across run logs.
#   usage: ./graph_breaks.sh [logdir]     (default: logs)
DIR="${1:-logs}"
printf '%6s  %s\n' COUNT 'SITE :: REASON'
find "$DIR" \( -name '*.out' -o -name '*.log' \) -print0 2>/dev/null |
xargs -0 awk '
  function flush(){ if (why!="" && site!="") n[site " :: " why]++; why=""; site="" }
  # a reason line: dynamo Explanation:, a raw exc.<Name>:, or "Reason:"
  /Explanation: |torch\._dynamo\.exc\.[A-Za-z]+: |^[[:space:]]*Reason: / {
      flush(); w=$0
      sub(/^.*Explanation: /,"",w); sub(/^.*torch\._dynamo\.exc\./,"",w)
      sub(/^[[:space:]]*Reason: /,"",w)
      sub(/ Adding a graph break\..*$/,"",w); sub(/[[:space:]]*$/,"",w)
      why=w; next }
  # "Graph break in user code at <file>:<line>" carries its own site
  /Graph break in user code at / {
      s=$0; sub(/^.*at /,"",s); sub(/^.*GTagger-experiments\//,"",s)
      sub(/[[:space:]]*$/,"",s); site=s; if (why=="") why="graph break"; flush(); next }
  # otherwise the first REPO file after a reason is the site
  why!="" && site=="" && /File "/ {
      f=$0; sub(/^.*File "/,"",f); sub(/".*$/,"",f)
      l=$0; sub(/^.*, line /,"",l); sub(/[^0-9].*$/,"",l)
      if (f !~ /(dist|site)-packages\/torch\//) {
          sub(/^.*GTagger-experiments\//,"",f); sub(/^.*(dist|site)-packages\//,"",f); site=f ":" l; flush() } }
  END { flush(); for (k in n) printf "%6d  %s\n", n[k], k }
' | sort -k1,1nr
