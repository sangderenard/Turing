// Draw grouping only: retain existing geometry, index order, and material rows.
export const FIELD_CATEGORY_BITS=Object.freeze({subject:1,stage:2,room:4});
export function fieldCategoryEnabled(mask,category) {
  return !!(mask & (FIELD_CATEGORY_BITS[category]??0));
}
export function withSceneCategory(faces,category) {
  if(!FIELD_CATEGORY_BITS[category])throw Error('Unknown scene receiver category.');
  return faces.map(face=>Object.assign([...face],{sceneCategory:category},
    face.responseSource?{responseSource:{...face.responseSource}}:{}));
}
export function fieldCategoryRuns(categories,indices=null) {
  const count=indices?indices.length:categories.length*3,runs=[];
  if(count%3)throw Error('Receiver lineage requires complete triangles.');
  for(let first=0;first<count;first+=3) {
    const vertex=indices?indices[first]:first,triangle=vertex/3;
    if(!Number.isInteger(triangle)||triangle<0||triangle>=categories.length||
       (indices&&(indices[first+1]!==vertex+1||indices[first+2]!==vertex+2)))
      throw Error('Receiver lineage no longer matches triangle order.');
    const category=FIELD_CATEGORY_BITS[categories[triangle]]?categories[triangle]:null;
    const last=runs.at(-1);
    if(last?.category===category)last.count+=3;
    else runs.push({category,first,count:3});
  }
  return runs;
}
