from src.rendering.ascii_render import AsciiRenderer
import time

print("[ascii_render] Test: ASCII diff printing demo")
renderer = AsciiRenderer(80, 30, depth=1)
renderer.clear(0)
renderer.line(0, 0, 79, 29, value=255)
print("First diff (should show a diagonal line):")
diff1 = renderer.to_ascii_diff()
print(diff1 if diff1 else "(no diff output)")
time.sleep(0.5)
#renderer.clear(0)
renderer.line(0, 29, 79, 0, value=255)
print("Second diff (should show the other diagonal):")
diff2 = renderer.to_ascii_diff()
print(diff2 if diff2 else "(no diff output)")
