const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addSharedLibrary("cgpt", "cgpt.c", .{});
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.linkLibC();

    b.installArtifact(exe);
}
