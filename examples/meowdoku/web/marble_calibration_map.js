// Sparse measurements from the authored exposure chart. The QR code fixes a
// projective image-to-card map; every measurement uses the same map and the
// actual sensor pixels, never a brightness threshold or whole-frame mean.
import {studioExposureTargetArtwork} from './marble_studio_grid.js';
import {qrMatrix} from './tipjar.js';
import {qrWallLayout} from './qr_wall.js';

const {reference}=studioExposureTargetArtwork();
const QR=reference.qr;
const REGION_NAMES=['blackCenter','gray18','whiteRing','blackOuter',
  'red','green','blue','yellow'];

function solve8(rows, values) {
  const a=rows.map((row,i)=>[...row,values[i]]);
  for(let c=0;c<8;c++){
    let pivot=c;
    for(let r=c+1;r<8;r++)if(Math.abs(a[r][c])>Math.abs(a[pivot][c]))pivot=r;
    if(Math.abs(a[pivot][c])<1e-10)throw Error('Calibration QR corners are degenerate');
    [a[c],a[pivot]]=[a[pivot],a[c]];
    const scale=a[c][c];for(let j=c;j<=8;j++)a[c][j]/=scale;
    for(let r=0;r<8;r++)if(r!==c){const f=a[r][c];for(let j=c;j<=8;j++)a[r][j]-=f*a[c][j];}
  }
  return a.map(row=>row[8]);
}

// Corner order is top-left, top-right, bottom-right, bottom-left of the QR's
// complete quiet-zone square. BarcodeDetector reports the same order.
export function calibrationImageMap(corners,width,height) {
  if(!Array.isArray(corners)||corners.length!==4||!Number.isSafeInteger(width)||
     !Number.isSafeInteger(height)||width<1||height<1)throw Error('Calibration needs four QR corners and image dimensions');
  const uv=[[0,0],[1,0],[1,1],[0,1]],rows=[],values=[];
  for(let i=0;i<4;i++){
    const {x,y}=corners[i];
    if(!Number.isFinite(x)||!Number.isFinite(y))throw Error('Invalid QR corner');
    rows.push([x,y,1,0,0,0,-uv[i][0]*x,-uv[i][0]*y]);values.push(uv[i][0]);
    rows.push([0,0,0,x,y,1,-uv[i][1]*x,-uv[i][1]*y]);values.push(uv[i][1]);
  }
  const h=solve8(rows,values),pixels=new Int16Array(width*height);
  pixels.fill(-1);
  const [left,bottom,right,top]=QR.bounds,size=right-left;
  for(let py=0;py<height;py++)for(let px=0;px<width;px++){
    const x=px+.5,y=py+.5,den=h[6]*x+h[7]*y+1;
    if(Math.abs(den)<1e-10)continue;
    const u=(h[0]*x+h[1]*y+h[2])/den,v=(h[3]*x+h[4]*y+h[5])/den;
    const cx=left+u*size,cy=top-v*size,r=Math.hypot(cx,cy);
    if(Math.abs(cx)>.5||Math.abs(cy)>.5)continue;
    // Inset from cell/ring boundaries to keep mixed pixels out of the fit.
    let region=-1;
    if(r<.09)region=0;
    else if(r>.14&&r<.20)region=1;
    else if(r>.26&&r<.31)region=2;
    else if(r>.45&&r<.47)region=3;
    else if(Math.abs(cy)<.025&&cx>.36&&cx<.40)region=4;
    else if(Math.abs(cy)<.025&&cx<-.36&&cx>-.40)region=5;
    else if(Math.abs(cx)<.025&&cy>.36&&cy<.40)region=6;
    else if(Math.abs(cx)<.025&&cy<-.36&&cy>-.40)region=7;
    pixels[py*width+px]=region;
  }
  return {version:1,width,height,qrPayload:QR.payload,qrCorners:corners.map(({x,y})=>({x,y})),
    regions:REGION_NAMES,pixels};
}

export function measureCalibrationMap(map,array,{minimumPixels=3}={}) {
  const {shape,data}=array||{},channels=shape?.length===3?shape[2]:1;
  if(![1,3,4].includes(channels)||shape[0]!==map.height||shape[1]!==map.width||
     data?.length!==map.width*map.height*channels)
    throw Error('Calibration map and sensor image dimensions differ');
  const counts=Array(REGION_NAMES.length).fill(0),sums=REGION_NAMES.map(()=>Array(channels).fill(0));
  for(let p=0;p<map.pixels.length;p++){
    const region=map.pixels[p];if(region<0)continue;
    counts[region]++;
    for(let c=0;c<channels;c++){
      const value=data[p*channels+c];
      if(!Number.isFinite(value))throw Error('Calibration region contains nonfinite pixels');
      sums[region][c]+=value;
    }
  }
  return Object.fromEntries(REGION_NAMES.map((name,i)=>[name,{pixels:counts[i],
    mean:counts[i]>=minimumPixels?sums[i].map(v=>v/counts[i]):null,
    usable:counts[i]>=minimumPixels}]));
}

export function calibrationSparseRuns(map) {
  const regions=Object.fromEntries(REGION_NAMES.map(name=>[name,[]]));
  for(let y=0;y<map.height;y++)for(let x=0;x<map.width;){
    const index=y*map.width+x,region=map.pixels[index];
    if(region<0){x++;continue;}
    let end=x+1;
    while(end<map.width&&map.pixels[y*map.width+end]===region)end++;
    regions[REGION_NAMES[region]].push([index,end-x]);x=end;
  }
  return {version:1,width:map.width,height:map.height,regions};
}

export async function detectCalibrationQr(image,photoUrl=null) {
  if(photoUrl){
    const url=new URL(photoUrl,location.href),query=url.searchParams;
    const layout=qrWallLayout(qrMatrix(QR.payload));
    const modules=Array.from({length:layout.total},(_,y)=>
      layout.cells.slice(y*layout.total,(y+1)*layout.total).map(cell=>cell.dark?'1':'0').join(''));
    const response=await fetch('/api/marble_photo/calibration_qr',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({digest:query.get('digest'),name:query.get('name'),
        lane:Number(query.get('lane')??0),modules})});
    const result=await response.json();
    if(!response.ok)throw Error(result.error||`QR registration HTTP ${response.status}`);
    return result.corners;
  }
  if(typeof BarcodeDetector==='undefined')throw Error('The Phong canvas needs browser QR detection');
  const detector=new BarcodeDetector({formats:['qr_code']});
  const codes=await detector.detect(image);
  const code=codes.find(item=>item.rawValue===QR.payload);
  if(!code||code.cornerPoints?.length!==4)throw Error('Exposure target QR was not found at readable scale');
  return code.cornerPoints.map(({x,y})=>({x,y}));
}
