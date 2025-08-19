import datetime as _dt
import math
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from .time_units import TimeUnit
from .theme_manager import ThemeManager
from .render_backend import RenderingBackend

class ClockRenderer:
    @staticmethod
    def render_analog(
        time_obj: _dt.datetime | _dt.timedelta,
        units: Optional[List[TimeUnit]] = None,
        canvas_size_px: Optional[int] = None,
        bounding_size: Optional[Tuple[int, int]] = None,  # (width, height) to auto‐size/center
        backdrop_image_path: Optional[str] = None,
        theme_manager: Optional[ThemeManager] = None,
        render_backend: Optional[RenderingBackend] = None,
        **params: Any
    ) -> Image.Image:
        """
        Draw an analog clock face.
        If bounding_size=(w,h) is provided and canvas_size_px is None,
        the clock diameter = min(w,h) and is centered in that box.
        """
        # --- determine canvas dimensions & center ---
        if bounding_size:
            bw, bh = bounding_size
            dia = canvas_size_px or min(bw, bh)
            base = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            cx, cy = bw // 2, bh // 2
        else:
            dia = canvas_size_px or 120
            # load or new square canvas
            if backdrop_image_path:
                try:
                    base = Image.open(backdrop_image_path).convert("RGBA")
                    base = base.resize((dia, dia), Image.Resampling.LANCZOS)
                except Exception:
                    base = Image.new("RGBA", (dia, dia), (0, 0, 0, 0))
            else:
                base = Image.new("RGBA", (dia, dia), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            cx = cy = dia // 2
        radius = dia // 2 - dia // 20

        # --- draw face outline ---
        stroke = max(1, dia // 50)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=params.get("face_color", (70, 70, 70, 255)), width=stroke
        )

        # --- marks ---
        major_w = max(1, dia // 60)
        minor_w = max(1, dia // 100)
        for i in range(12):
            ang = math.radians(i * 30 - 90)
            outer = radius
            inner = radius * (0.88 if i % 3 == 0 else 0.92)
            x1, y1 = cx + inner * math.cos(ang), cy + inner * math.sin(ang)
            x2, y2 = cx + outer * math.cos(ang), cy + outer * math.sin(ang)
            draw.line((x1, y1, x2, y2),
                      fill=params.get("marks_color", (220, 220, 200, 255)),
                      width=(major_w if i % 3 == 0 else minor_w))

        # --- hands ---
        if units is None:
            units = []
        hand_default = (200, 200, 200, 255)
        for u in units:
            val = u.get_value(time_obj)
            ang = math.radians((val / u.max_value) * 360 - 90)
            length = radius * u.hand_length_factor
            width = max(1, int(dia * u.hand_width_factor))
            color = hand_default
            # fetch per-unit theme color if available
            if theme_manager and u.hand_color_key in theme_manager.current_theme.palette:
                color = tuple(theme_manager.current_theme.palette[u.hand_color_key])
            x2 = cx + length * math.cos(ang)
            y2 = cy + length * math.sin(ang)
            draw.line((cx, cy, x2, y2), fill=color, width=width)

        # --- center dot ---
        dot_r = max(2, dia // 30)
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                     fill=params.get("center_dot_color", (255, 255, 255, 255)))

        # --- post-processing ---
        img = base
        if render_backend:
            img = render_backend.process(img)
        elif theme_manager:
            img = theme_manager.apply_effects(img)
            img = theme_manager.apply_theme(img)
        return img

    @staticmethod
    def render_digital(
        time_obj: _dt.datetime | _dt.timedelta,
        format_str: str = "{hours:02d}:{minutes:02d}:{seconds:02d}",
        image_size: Tuple[int, int] = (200, 50),
        font_path: Optional[str] = None,
        font_size: int = 40,
        backdrop_image_path: Optional[str] = None,
        theme_manager: Optional[ThemeManager] = None,
        render_backend: Optional[RenderingBackend] = None,
        **params: Any
    ) -> Image.Image:
        """
        Draw a digital clock face to a PIL RGBA image.
        'format_str' can use keys hours, minutes, seconds, milliseconds.
        Other text params (colors/shadow) in **params.
        """
        # --- compute time fields ---
        if isinstance(time_obj, _dt.datetime):
            h, m, s = time_obj.hour, time_obj.minute, time_obj.second
            ms = int(time_obj.microsecond / 1000)
        else:
            total_ms = int(time_obj.total_seconds() * 1000)
            h, rem = divmod(total_ms, 3600*1000)
            m, rem = divmod(rem, 60*1000)
            s, ms = divmod(rem, 1000)
        text = format_str.format(hours=h, minutes=m, seconds=s, milliseconds=ms, time=time_obj)

        # --- defaults & override ---
        defaults: Dict[str, Any] = {
            "text_color": (255, 255, 255, 255),
            "outline_color": (0, 0, 0, 200),
            "outline_width": 2,
            "shadow_color": (0, 0, 0, 100),
            "shadow_offset": (2, 2),
        }
        defaults.update(params)

        # --- load or new canvas ---
        w, h_px = image_size
        if backdrop_image_path:
            try:
                base = Image.open(backdrop_image_path).convert("RGBA")
                base = base.resize((w, h_px), Image.Resampling.LANCZOS)
            except Exception:
                base = Image.new("RGBA", (w, h_px), (0, 0, 0, 0))
        else:
            base = Image.new("RGBA", (w, h_px), (0, 0, 0, 0))
        draw = ImageDraw.Draw(base)

        # --- load font ---
        try:
            font = ImageFont.truetype(font_path or "DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # --- measure & center ---
        bbox = draw.textbbox((0, 0), text, font=font,
                             stroke_width=defaults["outline_width"])
        tx = (w - (bbox[2] - bbox[0])) // 2 - bbox[0]
        ty = (h_px - (bbox[3] - bbox[1])) // 2 - bbox[1]

        # --- draw shadow ---
        sx = tx + defaults["shadow_offset"][0]
        sy = ty + defaults["shadow_offset"][1]
        draw.text((sx, sy), text, font=font, fill=defaults["shadow_color"])

        # --- draw main text with outline ---
        draw.text((tx, ty), text, font=font,
                  fill=defaults["text_color"],
                  stroke_width=defaults["outline_width"],
                  stroke_fill=defaults["outline_color"])

        # --- post-processing ---
        img = base
        if render_backend:
            img = render_backend.process(img)
        elif theme_manager:
            img = theme_manager.apply_effects(img)
            img = theme_manager.apply_theme(img)
        return img
