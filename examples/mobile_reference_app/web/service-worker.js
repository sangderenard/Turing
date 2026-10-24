const CACHE='turing-mobile-v1';
const FILES=['./','index.html','turing-wasm-shell.js','manifest.webmanifest','program.abi.json','mobile_main.wasm','application.json'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(FILES))));
self.addEventListener('fetch',event=>event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request))));
