"""Offline installer regression tests: no real pip, desktop or systemd writes.

INSTALLER_UNDER_TEST may point to an older install.sh for red/green checks.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


INSTALLER = Path(os.environ.get(
    "INSTALLER_UNDER_TEST", Path(__file__).resolve().parents[1] / "install.sh"
)).read_text()
FUNCTIONS = INSTALLER[INSTALLER.index("detect_ld_path() {"):INSTALLER.index("unit_is_current() {")]


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + text)
    path.chmod(0o755)


class InstallerTests(unittest.TestCase):
    def run_shell(self, code, env=None):
        return subprocess.run(["bash", "-c", "set -euo pipefail\n" + code],
                              env=env, text=True, capture_output=True, timeout=15)

    def test_missing_venv_is_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_shell(f'VENV_DIR={shlex.quote(tmp)}/absent\n' + FUNCTIONS +
                                    '\ndetect_ld_path\nprintf "<%s>" "$LD_PATH"')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "<>")

    def test_actual_interpreter_paths_and_optional_libraries(self):
        for libraries in ((), ("cudnn",), ("cublas",), ("cudnn", "cublas")):
            with self.subTest(libraries=libraries), tempfile.TemporaryDirectory() as tmp:
                venv = Path(tmp) / "venv"
                site = venv / "lib/python3.13/site-packages"
                executable(venv / "bin/python3", f'printf "%s\\n" {shlex.quote(str(site))}\n')
                for library in libraries:
                    (site / "nvidia" / library / "lib").mkdir(parents=True)
                # Stale paths from another Python must not leak into the unit.
                (venv / "lib/python3.12/site-packages/nvidia/cublas/lib").mkdir(parents=True)
                result = self.run_shell(f'VENV_DIR={shlex.quote(str(venv))}\n' + FUNCTIONS +
                                        '\ndetect_ld_path\nprintf "%s" "$LD_PATH"')
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = ":".join(str(site / "nvidia" / lib / "lib") for lib in libraries)
                self.assertEqual(result.stdout, expected)

    def test_first_install_then_check_and_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "source/install.sh"
            script.parent.mkdir()
            script.write_text(INSTALLER)
            commands = root / "commands"
            commands.mkdir()
            log = root / "calls"
            # Emulate venv creation and pip's CUDA wheel directories, no network.
            executable(commands / "python3", '''
[[ "$1 $2" == '-m venv' ]]
venv="$3"
mkdir -p "$venv/bin"
printf '#!/bin/bash\\nprintf "%%s\\n" "%s/lib/python3.13/site-packages"\\n' "$venv" > "$venv/bin/python3"
chmod +x "$venv/bin/python3"
printf 'export PATH="%s/bin:$PATH"\\n' "$venv" > "$venv/bin/activate"
cat > "$venv/bin/pip" <<'PIP'
#!/bin/bash
set -euo pipefail
echo pip >> "$CALL_LOG"
site="$(dirname "$0")/../lib/python3.13/site-packages"
mkdir -p "$site/nvidia/cudnn/lib" "$site/nvidia/cublas/lib"
PIP
chmod +x "$venv/bin/pip"
''')
            executable(commands / "systemctl", '''
echo "systemctl $*" >> "$CALL_LOG"
if [[ "$*" == *is-active* ]]; then exit "${ACTIVE_STATUS:-1}"; fi
if [[ "$*" == *enable* ]]; then
    dir="$HOME/.config/systemd/user"
    mkdir -p "$dir/graphical-session.target.wants"
    ln -sf "$dir/spraakherkenning.service" "$dir/graphical-session.target.wants/spraakherkenning.service"
fi
''')
            env = dict(os.environ, HOME=str(root / "home"), CALL_LOG=str(log),
                       PATH=f"{commands}:/usr/bin:/bin")
            # --check without a venv is a normal negative, with a useful reason.
            checked = subprocess.run(["bash", str(script), "--check"], env=env,
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(checked.returncode, 1)
            self.assertIn("virtual environment is missing", checked.stderr)
            self.assertFalse(log.exists())
            for _ in range(2):
                installed = subprocess.run(["bash", str(script)], env=env,
                                           capture_output=True, text=True, timeout=15)
                self.assertEqual(installed.returncode, 0, installed.stderr)
            unit = (root / "home/.config/systemd/user/spraakherkenning.service").read_text()
            site = script.parent / "venv/lib/python3.13/site-packages/nvidia"
            self.assertIn(f"Environment=LD_LIBRARY_PATH={site}/cudnn/lib:{site}/cublas/lib\n", unit)
            self.assertEqual(log.read_text().splitlines().count("pip"), 1)
            before = log.read_text()
            checked = subprocess.run(["bash", str(script), "--check"], env=env,
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(checked.returncode, 1)
            self.assertIn("service is not active", checked.stderr)
            self.assertEqual(log.read_text()[len(before):],
                             "systemctl --user is-active --quiet spraakherkenning.service\n")
            checked = subprocess.run(["bash", str(script), "--check"],
                                     env=dict(env, ACTIVE_STATUS="0"),
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn("CURRENT: revision", checked.stdout)


if __name__ == "__main__":
    unittest.main()
