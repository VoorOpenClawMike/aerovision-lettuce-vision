#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f "PROJECT_SPEC.md" ]; then
  echo "FOUT: PROJECT_SPEC.md niet gevonden in $(pwd). Ben je in de juiste map?"
  exit 1
fi

echo "=== Preflight-check wordt (opnieuw) uitgevoerd voor de zekerheid ==="
if [ -x "./preflight-check.sh" ]; then
  ./preflight-check.sh || { echo "FOUT: preflight-check faalde. Los dit eerst op (zie FASE-3)."; exit 1; }
else
  echo "WAARSCHUWING: preflight-check.sh niet gevonden of niet uitvoerbaar — doorgaan zonder check."
fi

echo
echo "=== Autonome build start nu — geen tussentijdse vragen, log gaat naar build.log ==="
echo "Je kunt de voortgang in een tweede terminal volgen met: tail -f build.log"
echo

claude -p "$(cat PROJECT_SPEC.md)" \
  --dangerously-skip-permissions \
  --max-turns 500 \
  --output-format stream-json \
  --verbose 2>&1 | tee build.log

echo
echo "=== Autonome build klaar (of gestopt) — zie build.log en RESULTS.md ==="
echo "Ga nu naar FASE-5-resultaat-controleren.md"
