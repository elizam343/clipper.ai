#!/bin/bash
# Build Clipper.app: a small app bundle that runs app/main.py from this folder with the
# Python that has clipper's packages. Re-run it after moving the clipper folder.
#
#   bash app/make_app.sh                 # -> ~/Applications/Clipper.app
#   bash app/make_app.sh /Applications   # somewhere else
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # the clipper folder
DEST="${1:-$HOME/Applications}"
PY="$(command -v python3)"
APP="$DEST/Clipper.app"

"$PY" -c "import flask, webview" 2>/dev/null || {
  echo "Missing packages for $PY: run  pip install flask pywebview" >&2; exit 1; }

mkdir -p "$DEST"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# icon: assets/icon.png -> Clipper.icns
TMP="$(mktemp -d)"
ICONSET="$TMP/Clipper.iconset"
mkdir "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$HERE/assets/icon.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$HERE/assets/icon.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Clipper.icns"
rm -rf "$TMP"

cat > "$APP/Contents/MacOS/Clipper" <<EOF
#!/bin/bash
# Logs: ~/Library/Logs/Clipper.log
exec "$PY" "$HERE/app/main.py" >> "\$HOME/Library/Logs/Clipper.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/Clipper"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Clipper</string>
  <key>CFBundleDisplayName</key><string>Clipper</string>
  <key>CFBundleIdentifier</key><string>local.clipper.app</string>
  <key>CFBundleExecutable</key><string>Clipper</string>
  <key>CFBundleIconFile</key><string>Clipper</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

touch "$APP"   # so Finder picks up the icon
echo "Built $APP"
