#!/usr/bin/env bash
# Build the Java 8 JBWAPI fat JAR (in Docker Maven so no host JDK 8/Maven is needed)
# and install it into ~/.scbw/bots/DiffusionProtoss for scbw / watch.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="${HOME}/.scbw/bots/DiffusionProtoss"

docker run --rm \
    -v "${SCRIPT_DIR}:/work" \
    -v "${HOME}/.m2:/root/.m2" \
    -w /work \
    maven:3.9-eclipse-temurin-8 \
    mvn -q package

mkdir -p "${BOT_DIR}/AI" "${BOT_DIR}/read"
cp "${SCRIPT_DIR}/target/DiffusionProtoss.jar" "${BOT_DIR}/AI/DiffusionProtoss.jar"

# Reuse BWAPI 4.4.0 DLL if present in another installed bot (e.g. Stardust).
if [ ! -f "${BOT_DIR}/BWAPI.dll" ]; then
    for candidate in "${HOME}/.scbw/bots/Stardust/BWAPI.dll" "${HOME}/.scbw/bots/"*/BWAPI.dll; do
        if [ -f "${candidate}" ]; then
            cp "${candidate}" "${BOT_DIR}/BWAPI.dll"
            break
        fi
    done
fi

cat > "${BOT_DIR}/bot.json" <<'EOF'
{
  "name": "DiffusionProtoss",
  "race": "Protoss",
  "botType": "JAVA_MIRROR",
  "description": "Protoss Zealot + Dragoon bot with S0 JBWAPI + DiffusionGemma S1/S2"
}
EOF

echo "Installed DiffusionProtoss to ${BOT_DIR}"
