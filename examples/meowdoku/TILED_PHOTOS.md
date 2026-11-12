# Tiled and oversampled Studio photographs

The existing PhotoJobs owner and Spectral marble_preview_worker execute this mode.
No additional server, renderer, GPU owner, or cross-job scheduler is installed.
The HTTP/agent caller only supplies settings and polls the existing job.

- Existing preset sizes through 4096 retain the ordinary path by default.
- Large/custom dimensions and oversampling automatically use serial sensor tiles.
- `photo.resolution` is the final square size; `photo.width` and `photo.height`
  choose a custom rectangular final size together.
- `photo.oversampling` is an integer linear multiplier from 1 through 4. Thus
  resolution 8192 and oversampling 2 trace a 16384-square sensor and average each
  2x2 block of linear sensor means before the existing color/display transform.
- `photo.tiled = {"tile_size":4096}` explicitly requests 4K tiles. Automatic size
  is 1024. Tile edges may be partial. Per-edge sensor limit 32768; maximum 4096 tiles.
- The exposure time is per tile. A tile finishes at a complete sensor pass,
  never halfway through rows, then divides accumulated radiance by its sample
  count. Preparation, minimum full passes and assembly add time.
- The forward photon field is prepared once and frozen across tiles so later
  windows do not receive a progressively brighter field. Ordinary/live
  exposure behavior is unchanged.
- GPU textures are tile-sized. Full-image raw/profiled HDR assembly is disk-backed,
  exact global 99th percentile is computed on a disk-backed luminance array,
  and PNG/NPZ encoding and authorized downloads are streamed. Admission reserves
  the internal sensor, final HDR/PNG, peak scratch stores and 512 MiB headroom
  before launching a GPU owner. Supersampling does not reserve final-size
  output buffers at the larger internal resolution.
- RGB still photographs are supported. Live, orbit and device intersections
  remain explicit incompatible modes. The UI turns off device participation
  for tiles. Interactive HDR retouch is unavailable for tiled products; download
  the HDR or choose gamma before exposing.
- Existing ownership, active-job refusal and explicit cancellation remain in
  force. Job status reports tile count/current window and assembly state; this
  is a queue inside one photograph, not a queue of unrelated submissions.

Requested 8K/2x/4K example: resolution 8192, oversampling 2,
`tiled:{"tile_size":4096}` gives 16 tiles. At 30 seconds per tile, requested exposure is
8 minutes plus preparation, complete-pass overhead and disk assembly.

Validation: CPU coordinate/sample parity, odd edge coverage, exact monolithic
linear-downsample/display comparison at 1x/2x/3x/4x, invalid modes/resource bounds,
serial worker control flow and failure cleanup, photo ownership/cancel/download
routes, UI settings and existing retouch lifecycle. A real GPU/output render is
recorded separately when performed; CPU tests are not claimed as GPU proof.
