"""A/B wrapper: python ab_wrap.py <pin 0|1> <gc 0|1> <camera.py args>. Switches the mallopt pin and the gc freeze/collect without editing camera.py."""
import sys

from ai.board import camera

pin, gc_on = sys.argv[1] == "1", sys.argv[2] == "1"
if not pin:
    camera.pin_mmap_threshold = lambda: False
if not gc_on:
    camera.gc.collect = lambda *a, **k: 0
    camera.gc.freeze = lambda: None
sys.argv = ["camera"] + sys.argv[3:]
camera.main()
