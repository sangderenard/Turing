import assert from "node:assert/strict";
import { previewMotionAngle } from "../examples/meowdoku/web/marble_preview_motion.js";

const near = (a,b) => assert.ok(Math.abs(a-b)<1e-9, `${a} != ${b}`);
near(previewMotionAngle("off",1,0,60,2),0);
near(previewMotionAngle("rotate",0.5,90,60,1),3*Math.PI/2);
near(previewMotionAngle("rock",1,90,60,0),Math.PI/6);
near(previewMotionAngle("rock",1,270,60,0),-Math.PI/6);
near(previewMotionAngle("flip",1,0,60,0),0);
near(previewMotionAngle("flip",1,0,60,0.35),Math.PI/2);
near(previewMotionAngle("flip",1,0,60,0.7),Math.PI);
near(previewMotionAngle("flip",1,0,60,0.99),Math.PI);
near(previewMotionAngle("flip",1,0,60,1),Math.PI);
near(previewMotionAngle("flip",1,180,60,0),previewMotionAngle("flip",1,0,60,0.5));
console.log("Marble preview motion: OK");
