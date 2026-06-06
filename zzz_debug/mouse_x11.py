"""Linux X11 鼠标按键与指针位置（绕过 Motrix Input）。"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass

Button1Mask = 1 << 8
Button2Mask = 1 << 9
Button3Mask = 1 << 10


@dataclass
class MouseButtonState:
    left: bool = False
    middle: bool = False
    right: bool = False


@dataclass
class MousePointerState:
    held: MouseButtonState
    pressed: MouseButtonState
    released: MouseButtonState
    root_x: int = 0
    root_y: int = 0
    window_id: int = 0
    win_x: int = 0
    win_y: int = 0
    window_w: int = 0
    window_h: int = 0


class _XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("c_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_void_p),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class _X11MouseReader:
    def __init__(self) -> None:
        lib_path = ctypes.util.find_library("X11")
        if lib_path is None:
            raise OSError("libX11 not found")
        self._x11 = ctypes.CDLL(lib_path)
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise OSError("XOpenDisplay failed")
        self._x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self._x11.XDefaultRootWindow.restype = ctypes.c_ulong
        self._x11.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self._x11.XQueryTree.restype = ctypes.c_int
        self._x11.XTranslateCoordinates.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self._x11.XTranslateCoordinates.restype = ctypes.c_int
        self._root = int(self._x11.XDefaultRootWindow(self._display))
        self._prev = MouseButtonState()
        self._app_window_id: int | None = None
        self._viewport_window_id: int | None = None

    def close(self) -> None:
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None

    def _window_size(self, window_id: int) -> tuple[int, int]:
        attrs = _XWindowAttributes()
        ok = self._x11.XGetWindowAttributes(
            self._display,
            ctypes.c_ulong(window_id),
            ctypes.byref(attrs),
        )
        if not ok:
            return 0, 0
        return int(attrs.width), int(attrs.height)

    def _parent_window(self, window_id: int) -> int:
        root_return = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        child_count = ctypes.c_uint()
        ok = self._x11.XQueryTree(
            self._display,
            ctypes.c_ulong(window_id),
            ctypes.byref(root_return),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(child_count),
        )
        if not ok:
            return 0
        return int(parent.value)

    def _resolve_app_window(self, leaf_window_id: int) -> int:
        window_id = leaf_window_id
        best_id = leaf_window_id
        best_area = 0
        while window_id not in (0, self._root):
            width, height = self._window_size(window_id)
            area = width * height
            if width >= 800 and height >= 500 and area > best_area:
                best_id = window_id
                best_area = area
            window_id = self._parent_window(window_id)
        return best_id

    def _resolve_viewport_window(self, app_window_id: int) -> int:
        root_return = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        child_count = ctypes.c_uint()
        ok = self._x11.XQueryTree(
            self._display,
            ctypes.c_ulong(app_window_id),
            ctypes.byref(root_return),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(child_count),
        )
        if not ok or child_count.value == 0:
            return app_window_id

        best_id = app_window_id
        best_area = 0
        for idx in range(int(child_count.value)):
            child_id = int(children[idx])
            width, height = self._window_size(child_id)
            area = width * height
            if area > best_area:
                best_id = child_id
                best_area = area
        return best_id

    def _to_window_local(self, window_id: int, root_x: int, root_y: int) -> tuple[int, int]:
        dest_x = ctypes.c_int()
        dest_y = ctypes.c_int()
        child = ctypes.c_ulong()
        ok = self._x11.XTranslateCoordinates(
            self._display,
            ctypes.c_ulong(self._root),
            ctypes.c_ulong(window_id),
            int(root_x),
            int(root_y),
            ctypes.byref(dest_x),
            ctypes.byref(dest_y),
            ctypes.byref(child),
        )
        if not ok:
            return int(root_x), int(root_y)
        return int(dest_x.value), int(dest_y.value)

    def poll(self) -> MousePointerState:
        root_return = ctypes.c_ulong()
        child_return = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()
        ok = self._x11.XQueryPointer(
            self._display,
            ctypes.c_ulong(self._root),
            ctypes.byref(root_return),
            ctypes.byref(child_return),
            ctypes.byref(root_x),
            ctypes.byref(root_y),
            ctypes.byref(win_x),
            ctypes.byref(win_y),
            ctypes.byref(mask),
        )
        if not ok:
            return MousePointerState(
                held=MouseButtonState(),
                pressed=MouseButtonState(),
                released=MouseButtonState(),
            )

        m = int(mask.value)
        held = MouseButtonState(
            left=bool(m & Button1Mask),
            middle=bool(m & Button2Mask),
            right=bool(m & Button3Mask),
        )
        pressed = MouseButtonState(
            left=held.left and not self._prev.left,
            middle=held.middle and not self._prev.middle,
            right=held.right and not self._prev.right,
        )
        released = MouseButtonState(
            left=not held.left and self._prev.left,
            middle=not held.middle and self._prev.middle,
            right=not held.right and self._prev.right,
        )
        self._prev = held

        leaf_window_id = int(child_return.value) if int(child_return.value) else int(root_return.value)
        if self._app_window_id is None and leaf_window_id not in (0, self._root):
            self._app_window_id = self._resolve_app_window(leaf_window_id)
        if self._app_window_id is not None and self._viewport_window_id is None:
            self._viewport_window_id = self._resolve_viewport_window(self._app_window_id)

        viewport_id = self._viewport_window_id if self._viewport_window_id is not None else leaf_window_id
        viewport_w, viewport_h = self._window_size(viewport_id)
        local_x, local_y = self._to_window_local(viewport_id, int(root_x.value), int(root_y.value))

        return MousePointerState(
            held=held,
            pressed=pressed,
            released=released,
            root_x=int(root_x.value),
            root_y=int(root_y.value),
            window_id=viewport_id,
            win_x=local_x,
            win_y=local_y,
            window_w=viewport_w,
            window_h=viewport_h,
        )


_reader: _X11MouseReader | None = None


def x11_mouse_available() -> bool:
    return sys.platform.startswith("linux")


def poll_mouse_pointer() -> MousePointerState | None:
    global _reader
    if not x11_mouse_available():
        return None
    if _reader is None:
        try:
            _reader = _X11MouseReader()
        except OSError:
            return None
    return _reader.poll()


def close_mouse_reader() -> None:
    global _reader
    if _reader is not None:
        _reader.close()
        _reader = None
