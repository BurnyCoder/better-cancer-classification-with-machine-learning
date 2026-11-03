"""Tests for the CAMELYON17 slide-header reader behind results/wilds_scanners.json (no network: `_range` is patched)."""

import struct

from pcam_hc import wilds


def _tiff(entries: list[tuple[int, int, int, bytes]]) -> bytes:
    """Little-endian classic TIFF with one image directory. entries: (tag, type, count, value bytes); values longer
    than 4 bytes are stored after the directory and referenced by offset (TIFF 6.0, section 2)."""
    n = len(entries)
    ifd_off = 8
    data_off = ifd_off + 2 + 12 * n + 4
    ifd, data = struct.pack("<H", n), b""
    for tag, typ, count, value in entries:
        if len(value) <= 4:
            ifd += struct.pack("<HHI", tag, typ, count) + value.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, typ, count, data_off + len(data))
            data += value
    return b"II*\0" + struct.pack("<I", ifd_off) + ifd + b"\0\0\0\0" + data


def _serve(monkeypatch, blob: bytes):
    monkeypatch.setattr(wilds, "_range", lambda url, start, n: blob[start:start + n])


def test_resolution_tags_give_microns_per_pixel(monkeypatch):
    # 41,136.38 px/cm (ResolutionUnit 3 = cm) is the 0.2431 um/px of the RUMC 3DHistech slides.
    blob = _tiff([(282, 5, 1, struct.pack("<II", 4113638, 100)), (296, 3, 1, struct.pack("<H", 3)),
                  (305, 2, 4, b"ASA\0")])
    _serve(monkeypatch, blob)
    out = wilds.tiff_scanner_tags("x")
    assert out == {"um_per_px": 0.2431, "manufacturer": None, "software": "ASA"}


def test_philips_xml_gives_spacing_and_maker(monkeypatch):
    xml = (b'<DataObject><Attribute Name="DICOM_MANUFACTURER" Group="0x0008">PHILIPS</Attribute>'
           b'<Attribute Name="DICOM_PIXEL_SPACING" Group="0x0028">&quot;0.00025&quot; &quot;0.00025&quot;</Attribute>'
           b'</DataObject>\0')
    blob = _tiff([(270, 2, len(xml), xml), (305, 2, 16, b"Philips DP v1.0\0")])
    _serve(monkeypatch, blob)
    out = wilds.tiff_scanner_tags("x")
    assert out == {"um_per_px": 0.25, "manufacturer": "PHILIPS", "software": "Philips DP v1.0"}
