#!/bin/sh
# Build a one-click DSV4.1 launcher app (and its .dmg) for this Mac.
#
#   sh launcher/pack_macos.sh single|dual [--out <folder>] [--python <python3.12>] [--stack-root <dir>] [--dock] [--icon-python <python with Pillow>]
#
# Reads launcher/config/<mode>.json plus your private launcher/config/<mode>.local.json,
# bakes the Mac section into "<out>/DSV41 Single.app" (or "DSV41 Dual.app"), signs it
# ad hoc, and wraps it in "<out>/DSV41 Single.dmg". The app opens a Terminal window as its
# status console. --dock adds the app to the Dock (the Dock plist is backed up first).
# Uses only tools that ship with macOS: codesign, hdiutil and the Dock defaults.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:?usage: pack_macos.sh single|dual [--out DIR] [--python PY] [--stack-root DIR] [--dock]}
shift
out="$here/dist"; python=""; stack=""; dock=0; iconpy=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out=$2; shift 2 ;;
    --icon-python) iconpy=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --stack-root) stack=$2; shift 2 ;;
    --dock) dock=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
case "$mode" in single) name="DSV41 Single" ;; dual) name="DSV41 Dual" ;; *) echo "mode must be single or dual" >&2; exit 2 ;; esac
[ -n "$python" ] || python=$(command -v python3.12 || ls /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -n 1 || true)
[ -x "$python" ] || { echo "Python 3.12 not found; pass --python" >&2; exit 1; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
app="$out/$name.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
res="$app/Contents/Resources"

set -- --mode "$mode" --side mac --python "$python" --out "$res/launcher.json"
[ -z "$stack" ] || set -- "$@" --stack-root "$stack"
"$python" -B "$here/app/bake.py" "$@"
cp "$here/app/dsv41_launcher.py" "$res/dsv41_launcher.py"
if [ ! -f "$here/icons/built/$mode.icns" ]; then
  # Quick Look renders the SVG; Pillow writes the .icns (the stack's core venv has it).
  [ -n "$iconpy" ] || iconpy="$here/dist/runtime/core-venv/bin/python"
  [ -x "$iconpy" ] || iconpy="$python"
  "$iconpy" -B "$here/icons/build_icons.py" "$mode"
fi
cp "$here/icons/built/$mode.icns" "$res/AppIcon.icns"
cat > "$res/run.command" <<EOF
#!/bin/sh
# Status console for $name (opened in Terminal by the app).
cd "\$(dirname "\$0")"
exec "$python" -B dsv41_launcher.py --config launcher.json
EOF
cat > "$app/Contents/MacOS/launcher" <<'EOF'
#!/bin/sh
# Open the launcher's status console in Terminal; the console runs the launcher.
exec /usr/bin/open -a Terminal "$(cd "$(dirname "$0")/../Resources" && pwd)/run.command"
EOF
chmod 755 "$res/run.command" "$app/Contents/MacOS/launcher"
id=$(printf '%s' "$mode" | tr -cd 'a-z')
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundleIdentifier</key><string>com.fronteagle.dsv41-launcher.$id</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict></plist>
EOF
codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"
echo "Built and ad-hoc signed $app"

hdiutil create -quiet -volname "$name" -srcfolder "$app" -ov -format UDZO "$out/$name.dmg"
echo "Built $out/$name.dmg"

if [ "$dock" = 1 ]; then
  if defaults read com.apple.dock persistent-apps 2>/dev/null | grep -q "$(basename "$app")"; then
    echo "Dock already has $name"
  else
    backup="$HOME/Library/Preferences/com.apple.dock.backup-$(date +%Y%m%d-%H%M%S).plist"
    cp "$HOME/Library/Preferences/com.apple.dock.plist" "$backup"
    echo "Dock plist backed up to $backup"
    defaults write com.apple.dock persistent-apps -array-add "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>file://$(printf '%s' "$app/" | sed 's/ /%20/g')</string><key>_CFURLStringType</key><integer>15</integer></dict></dict></dict>"
    killall Dock
    echo "Added $name to the Dock"
  fi
fi
