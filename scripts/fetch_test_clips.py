"""Download the Mixkit test clips (private people / stock actors) used by the Android parity suite, SHA-checked."""
import hashlib, sys, urllib.request, os
CLIPS = {
    'mixkit_48205.mp4': ('https://assets.mixkit.co/videos/48205/48205-720.mp4', '900b21ea270cc3eb8a9f297e5186301ff751422f0dc886eae5db90cbc5e515d4'),
    'mixkit_49656.mp4': ('https://assets.mixkit.co/videos/49656/49656-720.mp4', '98eb047efe185017b651b0821cb8b5aa072f3a9951e87dc5040d6e3edc883889'),
}
out = sys.argv[1] if len(sys.argv) > 1 else 'clips'
os.makedirs(out, exist_ok=True)
for name, (url, sha) in CLIPS.items():
    p = os.path.join(out, name)
    if not (os.path.exists(p) and hashlib.sha256(open(p, 'rb').read()).hexdigest() == sha):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (FaceFusionStudio CI)'})
        with urllib.request.urlopen(req, timeout=120) as r, open(p, 'wb') as f: f.write(r.read())
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert got == sha, f'{name}: sha mismatch {got}'
    print(name, os.path.getsize(p), 'OK')
