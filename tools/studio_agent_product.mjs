// Local worker entry. Trusted startup chooses catalog/default files; stdin is
// one bounded JSON request and stdout is one JSON blob result. No listener.
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {buildStudioAgentProduct} from '../examples/meowdoku/web/marble_studio_agent.js';
const MAX=32*1024*1024;
try{
  const args=process.argv.slice(2),options={};
  for(let i=0;i<args.length;i+=2){
    if(!['--assets','--defaults'].includes(args[i])||!args[i+1]||options[args[i]])throw Error('Use --assets FILE [--defaults FILE]');
    options[args[i]]=args[i+1];
  }
  if(!options['--assets'])throw Error('--assets is required');
  const chunks=[];let size=0;
  for await(const chunk of process.stdin){size+=chunk.length;if(size>MAX)throw Error('Studio request exceeds 32 MiB');chunks.push(chunk);}
  const request=JSON.parse(Buffer.concat(chunks).toString('utf8'));
  const assets=JSON.parse(await readFile(options['--assets'],'utf8'));
  const defaults=options['--defaults']?JSON.parse(await readFile(options['--defaults'],'utf8')):undefined;
  const product=await buildStudioAgentProduct(request,{assets,defaults});
  if(product.bytes.length>MAX)throw Error('Studio result exceeds 32 MiB');
  process.stdout.write(JSON.stringify({content_type:product.contentType,encoding:'base64',
    sha256:createHash('sha256').update(product.bytes).digest('hex'),blob:Buffer.from(product.bytes).toString('base64')}));
}catch(error){process.stderr.write(String(error.message||error)+'\n');process.exitCode=1;}
