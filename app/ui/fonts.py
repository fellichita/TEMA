"""Load bundled Open Sans into this process without installing system fonts."""

import ctypes
import ctypes.util
import sys
from pathlib import Path

_LOADED = False


def load_fonts():
    global _LOADED
    if _LOADED:
        return
    paths = sorted((Path(__file__).parent / 'assets' / 'fonts').glob('*.ttf'))
    if sys.platform == 'darwin':
        cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        ct = ctypes.CDLL('/System/Library/Frameworks/CoreText.framework/CoreText')
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
        cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        ct.CTFontManagerRegisterFontsForURL.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
        for path in paths:
            data = bytes(path.resolve())
            url = cf.CFURLCreateFromFileSystemRepresentation(None, data, len(data), False)
            if url:
                try:
                    ct.CTFontManagerRegisterFontsForURL(url, 1, None)  # process scope
                finally:
                    cf.CFRelease(url)
    elif sys.platform == 'win32':
        add = ctypes.windll.gdi32.AddFontResourceExW
        add.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
        add.restype = ctypes.c_int
        for path in paths:
            add(str(path.resolve()), 0x10, None)  # FR_PRIVATE
    else:
        library = ctypes.util.find_library('fontconfig')
        if library:
            fc = ctypes.CDLL(library)
            fc.FcConfigGetCurrent.restype = ctypes.c_void_p
            fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            fc.FcConfigAppFontAddFile.restype = ctypes.c_int
            config = fc.FcConfigGetCurrent()
            for path in paths:
                fc.FcConfigAppFontAddFile(config, bytes(path.resolve()))
    _LOADED = True
