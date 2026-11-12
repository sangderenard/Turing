# Batched spring sheet and rope row

The numerical step is authored in `src/compiler/spring_network_law.py` as SymPy equations. Turing lowers it through its AbstractTensor Python stage and repository SSA to a native LLVM kernel. The Nodus plugin source in `spring_network_nodus_row.cpp.in` owns state and topology only; it gathers neighbors from its previous positions and calls the compiled kernel once for the full batch. The plugin keeps its position and velocity arrays private. No dt-system clock is added: a scalar tick input advances exactly one admitted step.

Build in a user-owned directory:

```powershell
python tools/build_spring_nodus_row.py --output C:\Users\alber\nodus-build\spring-row --nodus-root C:\Users\alber\nodus-build\free-runtime-src --nodus-build C:\Users\alber\nodus-build\free-runtime-build --accept
```

The builder refuses changes to the compiled kernel's pointer order, dtype or batch shape. The `--accept` run loads the actual plugin, configures one rope, advances it, checks pinned and padded nodes, and receives a UDP packet. The resulting DLL can be loaded by Nodus's ordinary plugin loader under tool id `spring.network.batched`.

## Row input

One AbstractTensor input per execution:

- Configuration: float64 `[8, 256, 38]`. This resets the row's private state and emits its initial positions.
- Tick: float64 scalar containing seconds. This advances one symplectic Euler step with implicit linear drag and emits new positions as float64 `[8, 256, 3]`.
- Freeze/thaw: float64 `[2]` containing `[1, 1]` to freeze or `[1, 0]` to thaw. Both return the held positions. While frozen, scalar ticks return the same positions without advancing velocity or sending redundant frames. Thaw resumes the stored velocity; neither command resets the fabric.

The 38 configuration fields per node are:

| Indices | Meaning |
|---|---|
| 0–2 | Initial xyz position |
| 3–5 | Initial xyz velocity |
| 6 | Occupied mask, 0 or 1 |
| 7 | Pinned mask, 0 or 1 |
| 8–10 | Pin anchor xyz |
| 11 | Inverse mass |
| 12 | Spring stiffness |
| 13 | Drag |
| 14–16 | External acceleration xyz |
| 17–20 | Four neighbor node indices within the same network |
| 21–32 | Four signed rest offsets, xyz each |
| 33–36 | Four edge masks, 0 or 1 |
| 37 | Fabric identity; read from node zero of each network |

For a grid sheet, use the four neighbor slots for left/right/up/down. For a rope, use two slots for previous/next and mask the other two. All networks and nodes occupy the fixed batch; unoccupied nodes retain their state, and missing edges contribute zero force. An active fabric must have a nonzero integral identity no greater than `2^53-1`, unique among the active fabrics in this row. Other fields of column 37 are ignored. An unstable tick above the row's configured conservative spring limit is refused; the caller can deliver smaller ticks.

## UDP publication

After each successful configure or tick, the row sends `NFAB` v1 UDP datagrams to `239.255.78.68:48768` by default. `NODUS_FABRIC_UDP_HOST` and `NODUS_FABRIC_UDP_PORT` can select a different multicast or unicast destination before row initialization. Multicast TTL is four. Network routing must permit multicast for receivers beyond the local segment.

Every datagram is little endian: `magic:u32`, `version:u16`, `header_bytes:u16` (32), `frame:u64`, `fabric_id:u64`, `chunk:u16`, `chunk_count:u16`, `pair_count:u16`, `state:u16` (bit 0 frozen), followed by up to 48 pairs of `node_id:u32, x:f32, y:f32, z:f32`. A receiver joins the destination group, groups packets by `(fabric_id, frame)`, and presents a frame only after all chunks arrive. The identity of a vector is `(fabric_id, node_id)`; `frame` is a row-local publication sequence. UDP is best effort, so dropped chunks should cause that frame to be skipped.

## Measured speed

On the local Windows Nodus build, five 100-step runs of the 81-node sheet measured median row times of 59.1, 59.7, 61.7, 62.1 and 64.5 microseconds. This includes the fixed 8×256×3 batch kernel, AbstractTensor handoff, output publication and two UDP sends. Median UDP receive plus CSV writing was about 111 microseconds. These are local measurements, not a deadline guarantee or live-server result.

## Local cloth demo

After the build above, run the native plugin driver in the same shell. It configures a 9×9 sheet with its top corners pinned, advances 100 steps, and records only the received UDP packets. The renderer reads that recording; it does not simulate cloth.

```powershell
$env:PATH = 'C:\Users\alber\nodus-build\free-runtime-build\Release;' + $env:PATH
& 'C:\Users\alber\nodus-build\spring-row\build\bin\spring_nodus_row_demo.exe' 'C:\Users\alber\nodus-build\spring-row\build\bin\spring_network_nodus_row.dll' 'C:\Users\alber\nodus-build\spring-row\cloth_udp_frames.csv'
python tools/render_spring_udp_demo.py C:\Users\alber\nodus-build\spring-row\cloth_udp_frames.csv C:\Users\alber\nodus-build\spring-row\cloth_udp_demo.gif
```

To use Studio's actual tabletop backdrop wall as the source mesh, export its material assets first, then:

```powershell
node tools/export_backdrop_fabric.mjs build/meowdoku/web/assets.json C:\Users\alber\nodus-build\spring-row\studio_backdrop_mesh.csv
& 'C:\Users\alber\nodus-build\spring-row\build\bin\spring_nodus_row_demo.exe' 'C:\Users\alber\nodus-build\spring-row\build\bin\spring_network_nodus_row.dll' 'C:\Users\alber\nodus-build\spring-row\studio_backdrop_udp.csv' 'C:\Users\alber\nodus-build\spring-row\studio_backdrop_mesh.csv' freeze
python tools/render_spring_udp_demo.py C:\Users\alber\nodus-build\spring-row\studio_backdrop_udp.csv C:\Users\alber\nodus-build\spring-row\studio_backdrop_fabric.gif --pins 72,80
```

`marble_backdrop_fabric.js` samples the two wall triangles from `backdropFaces` into 81 nodes and 128 triangles without changing their material ID. The same helper packs the fixed Nodus row configuration and accepts returned positions when the Studio scene is rebuilt. The live Studio server does not yet connect a canvas fabric row or subscribe to its UDP stream, so its fabric control currently changes the mesh and saved surface behavior, while the live motion/freeze path is proven by the local native demo. Ellipse, grid, artwork and curved sweep fabric are outside this first wall-mesh lane; cardstock retains those existing shapes.
