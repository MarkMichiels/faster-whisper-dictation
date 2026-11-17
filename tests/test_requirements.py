"""Real disposable venv metadata, offline pip/systemd doubles; never live HOME."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[1]


class RequirementsTests(unittest.TestCase):
    def test_drift_repair_and_idempotence(self):
        with tempfile.TemporaryDirectory(prefix="wp58-") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in ("install.sh", "requirements_check.py"):
                shutil.copyfile(ROOT / name, source / name)
            for name in ("llm_log.py", "research.py"):
                (source / name).write_text("raise SystemExit(0)\n")
            (source / "requirements.txt").write_text("demo-base==1.0\n")
            (source / "requirements-wayland.txt").write_text("demo-wayland>=2,<3\n")
            extension = source / "gnome-extension/dictation-helper@faster-whisper-dictation"
            extension.mkdir(parents=True)
            for name in ("metadata.json", "extension.js"):
                (extension / name).write_text("{}")
            virtual = source / "venv"
            venv.create(virtual, with_pip=True)
            python = virtual / "bin/python3"
            site = Path(subprocess.check_output([str(python), "-c",
                        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())

            def metadata(name, version):
                directory = site / f"{name}-{version}.dist-info"
                directory.mkdir(exist_ok=True)
                (directory / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")

            (site / "evdev.py").write_text("# offline Wayland import double\n")
            metadata("demo_base", "1.0")  # old requirements are satisfied
            metadata("demo_wayland", "1.0")  # wrong version of Wayland addition
            commands = root / "commands"
            commands.mkdir()
            # Only pip is doubled: its install result is real distribution metadata.
            python.rename(virtual / "bin/real-python3")
            python.write_text('''#!/bin/bash
set -euo pipefail
if [[ "${1:-} ${2:-}" == '-m pip' ]]; then
    echo "$*" >> "$PIP_LOG"
    for arg in "$@"; do [[ "$arg" != *requirements-wayland.txt ]] || touch "$WAYLAND_SEEN"; done
    exec "$(dirname "$0")/real-python3" "$REPAIR_SCRIPT"
fi
exec "$(dirname "$0")/real-python3" "$@"
''')
            python.chmod(0o755)
            repair = root / "repair.py"
            repair.write_text(f'''from pathlib import Path
import os
site = Path({str(site)!r})
packages = [('demo_base', '1.0'), ('demo_added', '3.0')]
if Path(os.environ['WAYLAND_SEEN']).exists():
    for old in site.glob('demo_wayland-*.dist-info/METADATA'):
        old.unlink()
    packages.append(('demo_wayland', '2.0'))
for name, version in packages:
    directory = site / f'{{name}}-{{version}}.dist-info'
    directory.mkdir(exist_ok=True)
    (directory / 'METADATA').write_text(f'Name: {{name}}\\nVersion: {{version}}\\n')
''')
            for name, body in {
                "systemctl": '''if [[ "$*" == *enable* ]]; then
 dir="$HOME/.config/systemd/user"
 mkdir -p "$dir/graphical-session.target.wants"
 ln -sf "$dir/spraakherkenning.service" "$dir/graphical-session.target.wants/spraakherkenning.service"
fi
''',
                "id": 'echo input\n',
                "gnome-extensions": 'echo "State: ENABLED"\n',
            }.items():
                command = commands / name
                command.write_text("#!/bin/bash\nset -eu\n" + body)
                command.chmod(0o755)
            env = dict(os.environ, HOME=str(root / "home"),
                       XDG_DATA_HOME=str(root / "data"), XDG_SESSION_TYPE="x11",
                       PATH=f"{commands}:/usr/bin:/bin", PIP_LOG=str(root / "pip.log"),
                       REPAIR_SCRIPT=str(repair), WAYLAND_SEEN=str(root / "wayland-seen"))

            def run(*args):
                result = subprocess.run(["bash", str(source / "install.sh"), *args],
                                        env=env, capture_output=True, text=True, timeout=30)
                print(f"install.sh {' '.join(args)}: exit={result.returncode}\n"
                      f"{result.stdout}{result.stderr}")
                return result

            self.assertEqual(run().returncode, 0)  # old base, no pip
            self.assertFalse((root / "pip.log").exists())
            self.assertEqual(run("--check").returncode, 0)
            with (source / "requirements.txt").open("a") as stream:
                stream.write("demo-added==3.0\n")
            drift = run("--check")
            self.assertEqual(drift.returncode, 1)
            self.assertIn("demo-added==3.0: missing", drift.stderr)
            self.assertNotIn("CURRENT: revision", drift.stdout)
            self.assertFalse((root / "pip.log").exists())
            self.assertEqual(run().returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(len((root / "pip.log").read_text().splitlines()), 1)
            # Install helper first to avoid the existing extension gate hiding drift.
            target = root / "data/gnome-shell/extensions" / extension.name
            shutil.copytree(extension, target)
            drift = run("--check", "--wayland")
            self.assertEqual(drift.returncode, 1)
            self.assertIn("installed 1.0", drift.stderr)
            self.assertEqual(run("--wayland").returncode, 0)
            self.assertTrue((root / "wayland-seen").exists())
            self.assertEqual(run("--wayland").returncode, 0)
            self.assertEqual(len((root / "pip.log").read_text().splitlines()), 2)
            self.assertEqual(run("--check", "--wayland").returncode, 0)
            # Removal after a successful install is detected (no stale hash stamp).
            (site / "demo_added-3.0.dist-info/METADATA").unlink()
            self.assertEqual(run("--check").returncode, 1)


if __name__ == "__main__":
    unittest.main()
