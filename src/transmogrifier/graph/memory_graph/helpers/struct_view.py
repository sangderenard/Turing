import collections
import ctypes
from importlib import abc
import itertools
import math
import random
import re
import sys
import threading
from uuid import uuid4

from ....cells.simulator import Simulator
from ....cells.cellsim.api.saline import SalinePressureAPI as SalineHydraulicSystem
from ....cells.cell_consts import Cell

import json

# Mirror LinearCells flag used by older region manager
IMMUTABLE = 1 << 5
class StructView:

    def make_view(self, raw_ptr: memoryview, Template: type,
                  *, data_key_guess=("node_data","edge_data")):
        """
        Return a live Mapping over *raw_ptr* cast as *Template*.
        Works for NodeEntry, EdgeEntry, MetaGraphEdge, or any future template.

            >>> off = graph.n_start
            >>> raw = graph.hard_memory.view(off, ctypes.sizeof(NodeEntry))
            >>> n   = units.make_view(raw, NodeEntry)
            >>> n['bit_width'] = 16          # writes straight into the buffer
        """
        field_names = [n for n,_ in Template._fields_]
        data_field  = next((k for k in data_key_guess if k in field_names), None)
        T_size      = ctypes.sizeof(Template)

        # cast the slice as a live ctypes object (zero-copy)
        entry = Template.from_buffer(raw_ptr)

        # ---------- dynamic proxy class (one per call, cheap) ----------
        # Using a closure keeps a reference to *entry* and *raw_ptr*
        class _Proxy(abc.MutableMapping):
            __slots__ = ("_e","_kv","_blob_mv")
            def __init__(self):                      # bind outer-scope vars
                self._e   = entry
                self._kv  = None                     # lazy-parse dict
                if data_field:
                    start = getattr(Template, data_field).offset
                    length= getattr(Template, data_field).size
                    self._blob_mv = raw_ptr[start:start+length]
                else:
                    self._blob_mv = None

            # ---- helper: parse / flush --------------------------------
            def _ensure_cache(self):
                if self._kv is not None or self._blob_mv is None:
                    return
                raw = bytes(self._blob_mv).rstrip(b"\0")
                if not raw:
                    self._kv = {}
                else:
                    try: self._kv = json.loads(raw.decode())
                    except Exception:
                        kv={}
                        for tok in raw.decode().split(";"):
                            if "=" in tok:
                                k,v=tok.split("=",1)
                                kv[k]=v
                        self._kv=kv

            def _flush(self):
                if self._kv is None or self._blob_mv is None:
                    return
                blob = json.dumps(self._kv, separators=(",",":")).encode()
                blob = blob[:len(self._blob_mv)].ljust(len(self._blob_mv), b"\0")
                self._blob_mv[:] = blob     # in-place write

            # ---- mapping interface -----------------------------------
            def __getitem__(self, k):
                if hasattr(self._e, k):        return getattr(self._e, k)
                self._ensure_cache();          return self._kv[k]

            def __setitem__(self, k, v):
                if hasattr(self._e, k):        setattr(self._e, k, v)
                else:                          self._ensure_cache(); self._kv[k]=v; self._flush()

            def __delitem__(self, k):
                if hasattr(self._e, k):        raise TypeError("cannot delete fixed field")
                self._ensure_cache();          del self._kv[k]; self._flush()

            def __iter__(self):
                yield from field_names
                if data_field:
                    self._ensure_cache();  yield from self._kv

            def __len__(self):
                self._ensure_cache();      return len(field_names)+len(self._kv or {})

            # optional convenience
            def __repr__(self):
                d={k:self[k] for k in self};   return f"<{Template.__name__}View {d}>"

        return _Proxy()
