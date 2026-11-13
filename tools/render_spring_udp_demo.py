"""Render recorded NFAB UDP positions from the native cloth demo as a GIF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(source: Path, target: Path, pins: tuple[int,int] = (0,8),
           material: tuple[str, tuple[float,float,float]] | None = None,
           table: tuple[float,float,float,float,float] | None = None) -> None:
    frames: dict[int, dict[int, tuple[float, float, float]]] = {}
    frozen_frames: set[int] = set()
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, node = int(row["frame"]), int(row["node"])
            frames.setdefault(frame, {})[node] = tuple(float(row[k]) for k in ("x", "y", "z"))
            if int(row.get('frozen',0)):
                frozen_frames.add(frame)
    if len(frames) != 101 or any(len(nodes) != 81 for nodes in frames.values()):
        raise ValueError("expected 101 complete UDP frames with 81 nodes each")
    depth_y=.42 if table else .18
    projected = [(x + .30*z, -y + depth_y*z) for nodes in frames.values()
                 for x, y, z in nodes.values()]
    if table:
        top,min_x,max_x,min_z,max_z=table
        projected.extend((x+.30*z,-top+depth_y*z) for x in (min_x,max_x) for z in (min_z,max_z))
    low_x=min(point[0] for point in projected);high_x=max(point[0] for point in projected)
    low_y=min(point[1] for point in projected);high_y=max(point[1] for point in projected)
    scale=min(610/max(high_x-low_x,1e-6),390/max(high_y-low_y,1e-6))
    left=360-(high_x-low_x)*scale/2
    font = ImageFont.load_default()
    images = []
    for frame in range(0, 101, 2):
        image = Image.new("RGB", (720, 540), "#08111b")
        draw = ImageDraw.Draw(image)
        positions = frames[frame]

        def project(node: int) -> tuple[int, int]:
            x, y, z = positions[node]
            return project_xyz(x,y,z)

        def project_xyz(x:float,y:float,z:float) -> tuple[int,int]:
            return (round(left+(x+.30*z-low_x)*scale),round(88+(-y+depth_y*z-low_y)*scale))

        if table:
            top,min_x,max_x,min_z,max_z=table
            slab=[project_xyz(x,top,z) for x,z in ((min_x,min_z),(max_x,min_z),
                                                    (max_x,max_z),(min_x,max_z))]
            draw.polygon(slab,fill='#263640',outline='#6b8490',width=3)

        if material:
            _,albedo=material
            encoded=tuple(round(255*(12.92*v if v<=.0031308 else 1.055*v**(1/2.4)-.055))
                          for v in albedo)
            for y in range(8):
                for x in range(8):
                    i=y*9+x
                    shade=.76+.08*((x+y)%2)
                    fill=tuple(round(min(255,max(0,c*shade))) for c in encoded)
                    draw.polygon([project(n) for n in (i,i+1,i+10,i+9)],fill=fill)
        for node in range(81):
            x, y = node % 9, node // 9
            for adjacent in ((node + 1,) if x < 8 else ()) + ((node + 9,) if y < 8 else ()):
                draw.line((project(node), project(adjacent)), fill="#34c9da" if not material else '#456e79', width=2)
        for node in range(81):
            px, py = project(node)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill="#b4ffff" if not material else '#7aabb2')
        for node in pins:
            px, py = project(node)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#ffc56c")
        if table:
            top,min_x,max_x,min_z,max_z=table
            for x in (min_x+.22,max_x-.22):
                a=project_xyz(x,top,max_z-.15);b=project_xyz(x,top-1.25,max_z-.15)
                draw.line((a,b),fill='#526674',width=7)
            front=[project_xyz(x,top,max_z) for x in (min_x,max_x)]
            draw.line(front,fill='#9aabb1',width=4)
        draw.text((24, 20), "Nodus cloth / native SymPy -> AbstractTensor -> LLVM", font=font, fill="#e7faff")
        draw.text((24, 42), f"fabric 20261112  |  UDP frame {frame + 1:03d}  |  t = {frame * .01:.2f} s", font=font, fill="#8ab3c5")
        if frame in frozen_frames:
            draw.text((615, 42), 'FROZEN', font=font, fill='#ffc56c')
        finish=f'  /  {material[0]}' if material else ''
        draw.text((24, 513), f"9 x 9 spring sheet  /  top corners pinned  /  81 UDP pairs{finish}", font=font, fill="#8ab3c5")
        images.append(image)
    target.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(target, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--pins", default="0,8", help="two pinned node IDs, e.g. 72,80 for a Studio wall")
    parser.add_argument('--assets',type=Path,help='Studio exported assets.json for material color')
    parser.add_argument('--material',help='name of a material in --assets')
    parser.add_argument('--contact',type=Path,help='Studio table contact CSV for drawing the actual tabletop')
    args = parser.parse_args()
    pins=tuple(int(value) for value in args.pins.split(','))
    if len(pins)!=2 or any(node<0 or node>=81 for node in pins):
        parser.error('--pins needs two node IDs from 0 to 80')
    material=None
    if args.material:
        if not args.assets:
            parser.error('--material requires --assets')
        assets=json.loads(args.assets.read_text(encoding='utf-8'))
        index=assets['material_index'][args.material]
        table=assets['materials']['pbr'];stride=table['floats_per_record']
        material=(args.material,tuple(float(v) for v in table['data'][index*stride:index*stride+3]))
    table=None
    if args.contact:
        with args.contact.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.reader(stream))
        if len(rows)!=2 or rows[0]!=['topY','minX','maxX','minZ','maxZ']:
            parser.error('--contact needs the exported Studio table contact CSV')
        table=tuple(float(value) for value in rows[1])
    render(args.source, args.target, pins, material, table)
    print(args.target)


if __name__ == "__main__":
    main()
