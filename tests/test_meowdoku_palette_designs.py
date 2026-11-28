"""The color game's designs: palettes of 1 to 6 colors, a token per page element, harmony scores and what is worn."""
import collections
import random
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "examples")]
from meowdoku import palette_designs as designs  # noqa: E402

# The patterns palette-theme.js accepts (VARIABLE, COLOR, LENGTH, SHADOW).
VARIABLE = re.compile(r"^--palette-[a-z0-9]+(-[a-z0-9]+){0,3}$")
COLOR = re.compile(r"^(#[0-9a-f]{6}|light-dark\(#[0-9a-f]{6}, #[0-9a-f]{6}\))$", re.I)
LENGTH = re.compile(r"^\d{1,3}(\.\d{1,2})?px$")
SHADOW = re.compile(r"^[-0-9a-z.,()% ]{1,240}$", re.I)


def _palette(*names):
    return [{"name": name, "color": designs.NAMED[name]} for name in names]


def test_palette_sizes_follow_a_curve_centred_on_four():
    rng = random.Random(11)
    sizes = collections.Counter(len(designs.random_design(rng)["colors"]) for _ in range(3000))
    assert set(sizes) <= set(designs.SIZES)
    assert sizes.most_common(1)[0][0] == 4
    assert sizes[3] > sizes[2] > sizes[1] and sizes[5] > sizes[6]


def test_every_drawn_design_is_valid_and_has_its_feature_count():
    rng = random.Random(5)
    for _ in range(200):
        design = designs.random_design(rng)
        assert designs.valid(design), design
        row = designs.features(design)
        assert len(row) == designs.FEATURES and all(-1e-9 <= v <= 1 + 1e-9 for v in row)


def test_harmony_scores_name_the_arrangement():
    index = {"monochrome": 1, "complementary": 3, "triadic": 4, "tetradic": 6}
    mono = designs.harmony(_palette("navy", "mediumblue", "royalblue"))
    complementary = designs.harmony(_palette("blue", "orange"))
    triadic = designs.harmony(_palette("red", "lime", "blue"))
    assert mono[index["monochrome"]] > complementary[index["monochrome"]]
    assert complementary[index["complementary"]] > mono[index["complementary"]]
    assert triadic[index["triadic"]] > complementary[index["triadic"]]
    neutrals = designs.harmony(_palette("white", "gray", "black"))
    assert neutrals[9] == 1.0, "a palette of neutrals is all neutral"


def test_worn_variables_are_ones_the_page_accepts_and_text_is_readable():
    rng = random.Random(9)
    for _ in range(100):
        design = designs.random_design(rng)
        worn = designs.wear(design)
        for name, value in worn["variables"].items():
            assert VARIABLE.match(name), name
            pattern = LENGTH if name.endswith(("-radius", "-border")) else SHADOW if name.endswith("-shadow") else COLOR
            assert pattern.match(value), (name, value)
        view = designs.preview(design)
        assert designs.contrast(view["text"], view["surface"]) >= 4.5
        assert designs.contrast(view["muted"], view["surface"]) >= 3.0
        assert designs.contrast(view["link"], view["surface"]) >= 4.5


def test_designs_from_before_palettes_are_read_as_palettes():
    old = {"name": "tomato", "color": "#ff6347", "bevel": 2.0, "light": 1.0, "shadow_x": 1.0, "shadow_y": 3.0,
           "blur": 5.0, "spread": 0.0, "alpha": 0.3}
    secondary = dict(old, name="navy", color="#000080")
    upgraded = designs.upgrade(old, secondary)
    assert designs.valid(upgraded) and designs.names(upgraded) == ["tomato", "navy"]
    assert len(designs.features(old, secondary)) == designs.FEATURES
    assert designs.upgrade(old, None)["colors"] == [{"name": "tomato", "color": "#ff6347"}]
