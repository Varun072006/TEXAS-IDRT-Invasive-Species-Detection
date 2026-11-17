"""Lossless EXIF orientation reset for JPEG files (stdlib only).

Sets the Orientation tag (0x0112) in the Exif APP1 block of IFD0 to a new value (default 1 = "as stored") by
patching exactly two bytes. Pixel data, quantisation tables and every other byte stay identical, so there is no
re-encoding loss. After this, EXIF-aware readers (browsers, PIL exif_transpose, GIS tools) show the stored pixel
grid, which is the grid Labelbox boxes refer to.
"""
from __future__ import annotations


def set_exif_orientation(data: bytes, value: int = 1) -> tuple[bytes, int | None, int | None]:
    """Return (new_bytes, old_orientation, patched_byte_offset). Orientation None = no tag present (nothing changed)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    n = len(data)
    while i + 4 <= n and data[i] == 0xFF:
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):  # EOI / start of scan: no more metadata segments
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = int.from_bytes(data[i + 2:i + 4], "big")
        if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            tiff = i + 10
            endian = "little" if data[tiff:tiff + 2] == b"II" else "big"
            ifd0 = int.from_bytes(data[tiff + 4:tiff + 8], endian)
            count = int.from_bytes(data[tiff + ifd0:tiff + ifd0 + 2], endian)
            for k in range(count):
                e = tiff + ifd0 + 2 + 12 * k
                if int.from_bytes(data[e:e + 2], endian) == 0x0112:
                    typ = int.from_bytes(data[e + 2:e + 4], endian)
                    if typ != 3:
                        raise ValueError(f"unexpected Orientation type {typ}")
                    old = int.from_bytes(data[e + 8:e + 10], endian)
                    out = bytearray(data)
                    out[e + 8:e + 10] = int(value).to_bytes(2, endian)
                    return bytes(out), old, e + 8
        i += 2 + seglen
    return data, None, None


def read_exif_orientation(data: bytes) -> int | None:
    return set_exif_orientation(data, 1)[1]
