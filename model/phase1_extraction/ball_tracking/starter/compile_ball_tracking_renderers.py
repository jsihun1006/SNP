"""Build the two retained local AVFoundation renderers; never export a video."""
from pathlib import Path
import argparse
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['ball', 'unified', 'all'], default='ball')
    parser.add_argument('--sdk', type=Path,
        default=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk'))
    args = parser.parse_args()
    if not args.sdk.is_dir():
        raise FileNotFoundError('Pass an installed macOS SDK directory with --sdk.')
    pairs = {
        'ball': ('starter/render_ball_atd_native.m', 'outputs/ball_tracking_runtime/render_atd_native'),
        'unified': ('starter/render_unified_tracking_native_v1.m', 'outputs/unified_tracking_v1/render_assets/render_native'),
    }
    for key, (source, target) in pairs.items():
        if args.target not in ['all', key]:
            continue
        if key == 'unified' and not (ROOT / 'outputs/unified_tracking_v1/tracking_all_frames.csv').is_file():
            raise FileNotFoundError('Build unified_tracking_v1 first, then compile its renderer.')
        dest = ROOT / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ['/usr/bin/clang', '-isysroot', str(args.sdk), '-mmacosx-version-min=13.0',
            '-fobjc-arc', str(ROOT / source), '-o', str(dest)]
        for framework in ['Foundation', 'AVFoundation', 'CoreMedia', 'QuartzCore', 'AppKit']:
            cmd.extend(['-framework', framework])
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
