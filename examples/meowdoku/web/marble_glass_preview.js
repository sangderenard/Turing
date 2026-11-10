// Raster-only draw planning. Saved geometry and ray material tables are untouched.
export function isOpaquePreviewMaterial(transmission, opacity) {
  // Conservative at grazing angles: only fully non-transmissive surfaces
  // write depth. A partially reflective glass surface stays in sorted draws.
  return opacity >= 0.999 && transmission <= 0;
}

export function triangleCentres(faces) {
  return faces.map(([a,b,c])=>a.map((x,k)=>(x+b[k]+c[k])/3));
}

export function transparentTriangleOrder(centres, modelView) {
  // OpenGL view space looks down -Z: more negative is farther away.
  // A centroid sort improves crossing shells, but is not order-independent
  // transparency for intersecting triangles and cannot reproduce refraction.
  const depths=centres.map(([x,y,z])=>modelView[2]*x+modelView[6]*y+modelView[10]*z);
  const triangles=Array.from({length:centres.length},(_,i)=>i)
    .sort((a,b)=>depths[a]-depths[b]||a-b);
  const indices=new Uint32Array(triangles.length*3);
  triangles.forEach((i,k)=>indices.set([3*i,3*i+1,3*i+2],3*k));
  return indices;
}
