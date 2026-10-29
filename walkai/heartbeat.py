"""Every N minutes, one KakaoTalk message about the walking training runs (standai's kakao_notify.send_message, only the message is custom).

  WALK_GATE_CROSS=1 ../standai/.venv/bin/python -I walkai/heartbeat.py --runs walk15a walk15b [--interval-minutes 30] [--once] [--dry]
Per run: training progress, a quick evaluation of the newest checkpoint (6 standing starts, 8 s, command 0.4 m/s) and the training log's latest
numbers. Stops after one last message when no train_walk.py process is running any more. The token lives in ../standai/checkpoints/kakao_token.json.
"""
import argparse
import glob
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "standai" / "scripts"))      # the REAL standai checkout (it holds checkpoints/kakao_token.json)

TOTAL = 40_000_000
PY = str(ROOT.parent / "standai" / ".venv" / "bin" / "python")


def latest_steps(run):
    cks = glob.glob(str(ROOT / "checkpoints" / "autosave" / f"{run}_*_steps.zip"))
    if not cks:
        return None
    return max(int(re.search(r"_(\d+)_steps", c).group(1)) for c in cks)


def train_numbers(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dirs = sorted(glob.glob(str(ROOT / "checkpoints" / "tb" / f"{run}_*")), key=os.path.getmtime)
    if not dirs:
        return {}
    ea = EventAccumulator(dirs[-1], size_guidance={"scalars": 0}); ea.Reload()
    out = {}
    for k, tag in (("fall", "walk/fall_rate"), ("track", "walk/track_ratio"), ("vmax", "walk/vmax")):
        if tag in ea.Tags()["scalars"]:
            vals = [e.value for e in ea.Scalars(tag)][-4:]
            out[k] = sum(vals) / len(vals)
    return out


def evaluate(run):
    r = subprocess.run([PY, "-I", str(HERE / "report.py"), run, "--cmd", "0.4", "--seeds", "6"], capture_output=True, text=True, cwd=str(ROOT))
    line = next((l for l in r.stdout.splitlines() if l.startswith(run)), "")
    m = re.search(r"upright\s+(\d+)%.*moved\s+([+-][\d.]+) m.*real steps ([\d.]+) \(rear foot passes ([\d.]+)\).*single-support\s+(\d+)%.*quality ([\d.]+)", line)
    return m.groups() if m else None


def describe(runs):
    blocks = []
    for run in runs:
        steps = latest_steps(run)
        if steps is None:
            blocks.append(f"🚶 {run}\n▸ 시작 중(체크포인트 아직 없음)")
            continue
        lines = [f"🚶 걷기 학습 {run}", f"▸ 진행 {steps / 1e6:.0f}M / {TOTAL / 1e6:.0f}M ({100 * steps / TOTAL:.0f}%)"]
        ev = evaluate(run)
        if ev:
            up, mv, st, cr, ss, q = ev
            lines.append(f"▸ 평가(0.4m/s) 서있음 {up}% · 8초 이동 {mv}m · 발걸음 {st}")
            lines.append(f"▸ ✨뒷발 추월 {cr}번 · 한발지지 {ss}% · 품질 {q}")
        tn = train_numbers(run)
        if tn:
            lines.append("▸ 학습 " + " · ".join(x for x in (f"쓰러짐 {tn['fall'] * 100:.0f}%" if 'fall' in tn else None,
                                                    f"속도추종 {tn['track']:.2f}" if 'track' in tn else None,
                                                    f"속도상한 {tn['vmax']:.1f}" if 'vmax' in tn else None) if x))
        blocks.append("\n".join(lines))
    return blocks


def alive():
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    return any("train_walk.py" in l for l in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--interval-minutes", type=float, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    from kakao_notify import send_message
    while True:
        running = alive()
        msg = f"⏱ {time.strftime('%H:%M')} 경과\n━━━━━━━━━━\n" + "\n\n".join(describe(a.runs)) + ("" if running else "\n\n🏁 학습이 모두 끝났어요")
        print(msg, flush=True)
        if not a.dry:
            try:
                send_message(msg)
            except Exception as e:
                print("send failed:", e, flush=True)
        if a.once or not running:
            break
        time.sleep(a.interval_minutes * 60)


if __name__ == "__main__":
    main()
