#!/usr/bin/env bash
# Catalogue non-finite-gradient events across run logs.
#   usage: ./nan_scan.sh [logdir]        (default: logs)
# One row per affected log: model, frames, first/last affected iteration, total
# count, worst consecutive run, whether the run hit the 50-consecutive abort, and
# any dumped batch file. Portable awk (no ENDFILE -- mawk ignores it silently).
DIR="${1:-logs}"
printf '%-32s %-26s %-16s %8s %8s %6s %6s  %s\n' \
       LOG MODEL FRAMES FIRST LAST COUNT MAXRUN OUTCOME
find "$DIR" \( -name '*.out' -o -name '*.log' \) -print0 2>/dev/null |
xargs -0 awk '
  function flush(){
      if (n>0) printf "%-32s %-26s %-16s %8s %8s %6d %6d  %s%s\n",
                 substr(file,1,32), substr(model,1,26), substr(frames,1,16),
                 first, last, n, mx, out, (dump==""?"":"  dump:" dump)
      model="?"; frames="-"; first=""; last=""; n=0; mx=0; out="ok"; dump="" }
  FNR==1 { flush(); file=FILENAME; sub(/^.*\//,"",file) }
  /Instantiated model / { model=$0; sub(/^.*Instantiated model /,"",model); sub(/ with .*$/,"",model) }
  /Frames approach: /   { frames=$0; sub(/^.*Frames approach: /,"",frames)
                          sub(/[ (].*$/,"",frames) }
  /gradient norm is non-finite/ {
      it=$0; sub(/^.*Skipping iteration /,"",it); sub(/[^0-9].*$/,"",it)
      c=$0;  sub(/^.*\[/,"",c);  sub(/[^0-9].*$/,"",c)
      if (first=="") first=it
      last=it; n++; if (c+0 > mx) mx=c+0 }
  /gradient norm .* exceeds maximum/ { n++; if (out=="ok") out="clipped" }
  /non-finite batch written to/ { dump=$0; sub(/^.*written to /,"",dump); sub(/^.*\//,"",dump) }
  /consecutive non-finite gradient norms/ { out="ABORTED" }
  END { flush() }
' | sort -k6,6nr
