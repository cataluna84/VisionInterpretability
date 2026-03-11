"""Patch StreamingOpen.islocal() to recognize Windows drive letters.

Root cause: StreamingOpen (cache.py line 212) checks
  `parsed.scheme in ["", "file"]`
but urlparse("c:\\path") gives scheme="c", which isn't recognized.
It falls to the gopen branch, which doesn't return a proper stream-like
object that tarfile can iterate.

Fix: Also patch the `islocal` function in cache.py to recognize
single-letter schemes as local paths (Windows drive letters).
"""
import json
from pathlib import Path

nb_path = Path(r"C:\Users\cataluna84\Documents\Workspace\VisionInterpretability\notebooks\Segment_3_canonical.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))

    if 'gopen_schemes[_letter] = gopen_file' in src:
        # Find the monkey-patch block and extend it to also patch islocal
        # and StreamingOpen
        new_source = []
        for line in cell["source"]:
            new_source.append(line)
            if '        gopen_schemes[_letter.upper()] = gopen_file' in line:
                # After registering gopen schemes, also patch islocal
                # and StreamingOpen to recognize drive letters as local
                new_source.append(
                    '    # Also patch islocal() and StreamingOpen to\n'
                )
                new_source.append(
                    '    # recognise single-letter (drive) schemes as local.\n'
                )
                new_source.append(
                    '    import webdataset.cache as _wds_cache\n'
                )
                new_source.append(
                    '    _orig_islocal = _wds_cache.islocal\n'
                )
                new_source.append(
                    '    def _patched_islocal(url):\n'
                )
                new_source.append(
                    '        from urllib.parse import urlparse as _up\n'
                )
                new_source.append(
                    '        s = _up(url).scheme\n'
                )
                new_source.append(
                    '        if len(s) == 1 and s.isalpha():\n'
                )
                new_source.append(
                    '            return True\n'
                )
                new_source.append(
                    '        return _orig_islocal(url)\n'
                )
                new_source.append(
                    '    _wds_cache.islocal = _patched_islocal\n'
                )
                new_source.append(
                    '\n'
                )
                new_source.append(
                    '    # Patch StreamingOpen.__call__ to open drive-letter\n'
                )
                new_source.append(
                    '    # paths directly with open() instead of gopen().\n'
                )
                new_source.append(
                    '    _OrigStreamingOpen = _wds_cache.StreamingOpen\n'
                )
                new_source.append(
                    '    _orig_call = _OrigStreamingOpen.__call__\n'
                )
                new_source.append(
                    '    def _patched_streaming_call(self, urls):\n'
                )
                new_source.append(
                    '        for url in urls:\n'
                )
                new_source.append(
                    '            if isinstance(url, dict):\n'
                )
                new_source.append(
                    '                url = url["url"]\n'
                )
                new_source.append(
                    '            try:\n'
                )
                new_source.append(
                    '                stream = open(url, "rb")\n'
                )
                new_source.append(
                    '                yield dict(url=url, stream=stream, local_path=url)\n'
                )
                new_source.append(
                    '            except Exception as exn:\n'
                )
                new_source.append(
                    '                if self.handler(exn):\n'
                )
                new_source.append(
                    '                    continue\n'
                )
                new_source.append(
                    '                else:\n'
                )
                new_source.append(
                    '                    break\n'
                )
                new_source.append(
                    '    _OrigStreamingOpen.__call__ = _patched_streaming_call\n'
                )
        cell["source"] = new_source
        print("Patched StreamingOpen to open drive-letter paths directly")
        break

nb_path.write_text(
    json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("Done.")
