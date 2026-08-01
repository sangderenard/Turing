(module ;; render_chunk0__0
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32) (param $out12 i32) (param $out13 i32) (param $out14 i32) (param $out15 i32) (param $out16 i32) (param $out17 i32) (param $out18 i32) (param $out19 i32) (param $out20 i32) (param $out21 i32) (param $out22 i32) (param $out23 i32) (param $out24 i32) (param $out25 i32) (param $out26 i32) (param $out27 i32) (param $out28 i32) (param $out29 i32) (param $out30 i32) (param $out31 i32) (param $out32 i32) (param $out33 i32) (param $out34 i32) (param $out35 i32) (param $out36 i32) (param $out37 i32) (param $out38 i32) (param $out39 i32) (param $out40 i32) (param $out41 i32) (param $out42 i32) (param $out43 i32) (param $out44 i32) (param $out45 i32) (param $out46 i32) (param $out47 i32) (param $out48 i32) (param $out49 i32) (param $out50 i32) (param $out51 i32) (param $out52 i32) (param $out53 i32) (param $out54 i32) (param $out55 i32) (param $out56 i32) (param $out57 i32) (param $out58 i32) (param $out59 i32) (param $out60 i32) (param $out61 i32) (param $out62 i32) (param $out63 i32) (param $out64 i32) (param $out65 i32) (param $out66 i32) (param $out67 i32) (param $out68 i32) (param $out69 i32) (param $out70 i32) (param $out71 i32) (param $out72 i32) (param $out73 i32) (param $out74 i32) (param $out75 i32) (param $out76 i32) (param $out77 i32) (param $out78 i32) (param $out79 i32) (param $out80 i32) (param $out81 i32) (param $out82 i32) (param $out83 i32) (param $out84 i32) (param $out85 i32) (param $out86 i32) (param $out87 i32) (param $out88 i32) (param $out89 i32) (param $out90 i32) (param $out91 i32) (param $out92 i32) (param $out93 i32) (param $out94 i32) (param $out95 i32) (param $out96 i32) (param $out97 i32) (param $out98 i32) (param $out99 i32) (param $out100 i32) (param $out101 i32) (param $out102 i32) (param $out103 i32) (param $out104 i32) (param $out105 i32) (param $out106 i32) (param $out107 i32) (param $out108 i32) (param $out109 i32) (param $out110 i32) (param $out111 i32) (param $out112 i32) (param $out113 i32) (param $out114 i32) (param $out115 i32) (param $out116 i32) (param $out117 i32) (param $out118 i32) (param $out119 i32) (param $out120 i32) (param $out121 i32) (param $out122 i32) (param $out123 i32) (param $out124 i32) (param $out125 i32) (param $out126 i32) (param $out127 i32) (param $out128 i32) (param $out129 i32) (param $out130 i32) (param $out131 i32) (param $out132 i32) (param $out133 i32) (param $out134 i32) (param $out135 i32) (param $out136 i32) (param $out137 i32) (param $out138 i32) (param $out139 i32) (param $out140 i32) (param $out141 i32) (param $out142 i32) (param $out143 i32) (param $out144 i32) (param $out145 i32) (param $out146 i32) (param $out147 i32) (param $out148 i32) (param $out149 i32) (param $out150 i32) (param $out151 i32) (param $out152 i32) (param $out153 i32) (param $out154 i32) (param $out155 i32) (param $out156 i32) (param $out157 i32) (param $out158 i32) (param $out159 i32) (param $out160 i32) (param $out161 i32) (param $out162 i32) (param $out163 i32) (param $out164 i32) (param $out165 i32) (param $out166 i32) (param $out167 i32) (param $out168 i32) (param $out169 i32) (param $out170 i32) (param $out171 i32) (param $out172 i32) (param $out173 i32) (param $out174 i32) (param $out175 i32) (param $out176 i32) (param $out177 i32) (param $out178 i32) (param $out179 i32) (param $out180 i32) (param $out181 i32) (param $out182 i32) (param $out183 i32) (param $out184 i32) (param $out185 i32) (param $out186 i32) (param $out187 i32) (param $out188 i32) (param $out189 i32) (param $out190 i32) (param $out191 i32) (param $out192 i32) (param $out193 i32) (param $out194 i32) (param $out195 i32) (param $out196 i32) (param $out197 i32) (param $out198 i32) (param $out199 i32) (param $out200 i32) (param $out201 i32) (param $out202 i32) (param $out203 i32) (param $out204 i32) (param $out205 i32) (param $out206 i32) (param $out207 i32) (param $out208 i32) (param $out209 i32) (param $out210 i32) (param $out211 i32) (param $out212 i32) (param $out213 i32) (param $out214 i32) (param $out215 i32) (param $out216 i32) (param $out217 i32) (param $out218 i32) (param $out219 i32) (param $out220 i32) (param $out221 i32) (param $out222 i32) (param $out223 i32) (param $out224 i32) (param $out225 i32) (param $out226 i32) (param $out227 i32) (param $out228 i32) (param $out229 i32) (param $out230 i32) (param $out231 i32) (param $out232 i32) (param $out233 i32) (param $out234 i32) (param $out235 i32) (param $out236 i32) (param $out237 i32) (param $out238 i32) (param $out239 i32) (param $out240 i32) (param $out241 i32) (param $out242 i32) (param $out243 i32) (param $out244 i32) (param $out245 i32) (param $out246 i32) (param $out247 i32) (param $out248 i32) (param $out249 i32) (param $out250 i32) (param $out251 i32) (param $out252 i32) (param $out253 i32) (param $out254 i32) (param $out255 i32) (param $out256 i32) (param $out257 i32) (param $out258 i32) (param $out259 i32) (param $out260 i32) (param $out261 i32) (param $out262 i32) (param $out263 i32) (param $out264 i32) (param $out265 i32) (param $out266 i32) (param $out267 i32) (param $out268 i32) (param $out269 i32) (param $out270 i32) (param $out271 i32) (param $out272 i32) (param $out273 i32) (param $out274 i32) (param $out275 i32) (param $out276 i32) (param $out277 i32) (param $out278 i32) (param $out279 i32) (param $out280 i32) (param $out281 i32) (param $out282 i32) (param $out283 i32) (param $out284 i32) (param $out285 i32) (param $out286 i32) (param $out287 i32) (param $out288 i32) (param $out289 i32) (param $out290 i32) (param $out291 i32) (param $out292 i32) (param $out293 i32) (param $out294 i32) (param $out295 i32) (param $out296 i32) (param $out297 i32) (param $out298 i32) (param $out299 i32) (param $out300 i32) (param $out301 i32) (param $out302 i32) (param $out303 i32) (param $out304 i32) (param $out305 i32) (param $out306 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $v0
      f64.const 0.83
      f64.mul
      local.set $v4
      local.get $v1
      f64.const 0.97
      f64.mul
      local.set $v5
      local.get $v1
      f64.const 2.17
      f64.mul
      local.set $v6
      local.get $v1
      f64.const 0.83
      f64.mul
      local.set $v7
      local.get $v1
      f64.const 1.67
      f64.mul
      local.set $v8
      local.get $v1
      f64.const 0.24
      f64.mul
      local.set $v9
      local.get $v1
      f64.const 0.31
      f64.mul
      local.set $v10
      local.get $v0
      f64.const -0.36
      f64.mul
      local.set $v11
      ;; cos via baked lookup table (see the .wasm)
      local.get $v10
      local.set $v12
      ;; sin via baked lookup table (see the .wasm)
      local.get $v8
      local.set $v13
      local.get $v5
      f64.const 0.61
      f64.add
      local.set $v14
      ;; sin via baked lookup table (see the .wasm)
      local.get $v9
      local.set $v15
      local.get $v2
      f64.const 0.91
      f64.mul
      local.set $v16
      local.get $v11
      local.get $v16
      f64.add
      local.set $v17
      local.get $v2
      f64.const -0.41
      f64.mul
      local.set $v18
      local.get $v4
      local.get $v18
      f64.add
      local.set $v19
      local.get $v3
      f64.const 0.31
      f64.mul
      local.set $v20
      local.get $v3
      f64.const 0.72
      f64.mul
      local.set $v21
      local.get $v19
      local.get $v21
      f64.add
      local.set $v22
      local.get $v22
      f64.const 0.15
      f64.add
      local.set $v23
      local.get $v3
      f64.const 0.48
      f64.mul
      local.set $v24
      local.get $v17
      local.get $v24
      f64.add
      local.set $v25
      local.get $v25
      f64.const 0.08
      f64.sub
      local.set $v26
      ;; tanh via baked lookup table (see the .wasm)
      local.get $v23
      local.set $v27
      ;; tanh via baked lookup table (see the .wasm)
      local.get $v26
      local.set $v28
      local.get $v27
      f64.const 0.62
      f64.mul
      local.set $v29
      local.get $v28
      f64.const -0.57
      f64.mul
      local.set $v30
      local.get $v29
      local.get $v30
      f64.add
      local.set $v31
      local.get $v31
      local.get $v20
      f64.add
      local.set $v32
      ;; tanh via baked lookup table (see the .wasm)
      local.get $v32
      local.set $v33
      local.get $v33
      f64.const -0.003
      f64.mul
      local.set $v34
      local.get $v2
      local.get $v34
      f64.add
      local.set $v35
      local.get $v33
      f64.const 0.004
      f64.mul
      local.set $v36
      local.get $v0
      local.get $v36
      f64.add
      local.set $v37
      ;; sin via baked lookup table (see the .wasm)
      local.get $v6
      local.set $v38
      local.get $v38
      f64.const 0.22
      f64.mul
      local.set $v39
      local.get $v12
      f64.const 0.58
      f64.mul
      local.set $v40
      local.get $v40
      f64.const 0.5
      f64.mul
      local.set $v41
      ;; sin via baked lookup table (see the .wasm)
      local.get $v14
      local.set $v42
      local.get $v42
      f64.const 0.48
      f64.mul
      local.set $v43
      local.get $v15
      f64.const 0.5
      f64.mul
      local.set $v44
      local.get $v13
      f64.const 0.19
      f64.mul
      local.set $v45
      local.get $v43
      local.get $v45
      f64.add
      local.set $v46
      local.get $v46
      f64.const 0.004
      f64.mul
      local.set $v47
      local.get $v47
      f64.const 0.0011
      f64.sub
      local.set $v48
      local.get $v48
      f64.const 0.131825904
      f64.add
      local.set $v49
      local.get $v40
      local.get $v40
      f64.mul
      local.set $v50
      local.get $v44
      f64.const 0.5
      f64.add
      local.set $v51
      local.get $v51
      f64.const 0.18
      f64.mul
      local.set $v52
      local.get $v52
      f64.const 0.04
      f64.add
      local.set $v53
      local.get $v53
      f64.const -1.0
      f64.mul
      local.set $v54
      local.get $v54
      f64.const 1.0
      f64.add
      local.set $v55
      local.get $v1
      f64.const 0.71
      f64.mul
      local.set $v56
      ;; sin via baked lookup table (see the .wasm)
      local.get $v56
      local.set $v57
      local.get $v57
      f64.const 1.25
      f64.mul
      local.set $v58
      ;; sin via baked lookup table (see the .wasm)
      local.get $v7
      local.set $v59
      local.get $v59
      f64.const 0.58
      f64.mul
      local.set $v60
      local.get $v60
      local.get $v39
      f64.add
      local.set $v61
      local.get $v61
      f64.const 0.004
      f64.mul
      local.set $v62
      local.get $v62
      f64.const 0.743643887
      f64.sub
      local.set $v63
      local.get $v1
      f64.const 0.31
      f64.mul
      local.set $v64
      ;; sin via baked lookup table (see the .wasm)
      local.get $v64
      local.set $v65
      local.get $v65
      f64.const 0.58
      f64.mul
      local.set $v66
      local.get $v66
      f64.const 0.5
      f64.mul
      local.set $v67
      local.get $v66
      local.get $v66
      f64.mul
      local.set $v68
      local.get $v50
      local.get $v68
      f64.sub
      local.set $v69
      local.get $v69
      f64.const 0.25
      f64.mul
      local.set $v70
      local.get $v41
      local.get $v70
      f64.sub
      local.set $v71
      local.get $v71
      local.get $v53
      f64.mul
      local.set $v72
      local.get $v63
      local.get $v72
      f64.sub
      local.set $v73
      local.get $v73
      local.get $v55
      f64.div
      local.set $v74
      local.get $v40
      local.get $v66
      f64.mul
      local.set $v75
      local.get $v75
      f64.const 2.0
      f64.mul
      local.set $v76
      local.get $v76
      f64.const 0.25
      f64.mul
      local.set $v77
      local.get $v67
      local.get $v77
      f64.sub
      local.set $v78
      local.get $v78
      local.get $v53
      f64.mul
      local.set $v79
      local.get $v49
      local.get $v79
      f64.sub
      local.set $v80
      local.get $v80
      local.get $v55
      f64.div
      local.set $v81
      local.get $v1
      f64.const 1.93
      f64.mul
      local.set $v82
      ;; sin via baked lookup table (see the .wasm)
      local.get $v82
      local.set $v83
      local.get $v83
      f64.const 0.45
      f64.mul
      local.set $v84
      local.get $v58
      local.get $v84
      f64.add
      local.set $v85
      ;; exp via baked lookup table (see the .wasm)
      local.get $v85
      local.set $v86
      local.get $v86
      f64.const 0.004
      f64.mul
      local.set $v87
      local.get $v87
      local.get $v55
      f64.div
      local.set $v88
      local.get $v37
      local.get $v88
      f64.mul
      local.set $v89
      local.get $v74
      local.get $v89
      f64.add
      local.set $v90
      local.get $v90
      local.get $v53
      f64.mul
      local.set $v91
      local.get $v71
      local.get $v90
      f64.sub
      local.set $v92
      local.get $v35
      local.get $v88
      f64.mul
      local.set $v93
      local.get $v81
      local.get $v93
      f64.add
      local.set $v94
      local.get $v94
      local.get $v53
      f64.mul
      local.set $v95
      local.get $v78
      local.get $v94
      f64.sub
      local.set $v96
      local.get $v91
      f64.const 2.0
      f64.mul
      local.set $v97
      local.get $v53
      local.get $v96
      f64.mul
      local.set $v98
      local.get $v91
      local.get $v91
      f64.mul
      local.set $v99
      local.get $v90
      f64.const 0.0
      f64.mul
      local.set $v100
      local.get $v53
      local.get $v92
      f64.mul
      local.set $v101
      local.get $v90
      local.get $v101
      f64.add
      local.set $v102
      local.get $v90
      f64.const 0.0
      f64.mul
      local.set $v103
      local.get $v103
      f64.const 1e+18
      f64.add
      local.set $v104
      local.get $v104
      f64.neg
      local.set $v105
      local.get $v104
      f64.neg
      local.set $v106
      local.get $v104
      f64.neg
      local.set $v107
      local.get $v104
      f64.neg
      local.set $v108
      local.get $v104
      f64.neg
      local.set $v109
      local.get $v104
      f64.neg
      local.set $v110
      local.get $v104
      f64.neg
      local.set $v111
      local.get $v104
      f64.neg
      local.set $v112
      local.get $v104
      f64.neg
      local.set $v113
      local.get $v104
      f64.neg
      local.set $v114
      local.get $v104
      f64.neg
      local.set $v115
      local.get $v104
      f64.neg
      local.set $v116
      local.get $v104
      f64.neg
      local.set $v117
      local.get $v104
      f64.neg
      local.set $v118
      local.get $v104
      f64.neg
      local.set $v119
      local.get $v104
      f64.neg
      local.set $v120
      local.get $v104
      f64.neg
      local.set $v121
      local.get $v104
      f64.neg
      local.set $v122
      local.get $v104
      f64.neg
      local.set $v123
      local.get $v104
      f64.neg
      local.set $v124
      local.get $v104
      f64.neg
      local.set $v125
      local.get $v104
      f64.neg
      local.set $v126
      local.get $v104
      f64.neg
      local.set $v127
      local.get $v104
      f64.neg
      local.set $v128
      local.get $v104
      f64.neg
      local.set $v129
      local.get $v104
      f64.neg
      local.set $v130
      local.get $v104
      f64.neg
      local.set $v131
      local.get $v104
      f64.neg
      local.set $v132
      local.get $v104
      f64.neg
      local.set $v133
      local.get $v104
      f64.neg
      local.set $v134
      local.get $v104
      f64.neg
      local.set $v135
      local.get $v104
      f64.neg
      local.set $v136
      local.get $v104
      f64.neg
      local.set $v137
      local.get $v104
      f64.neg
      local.set $v138
      local.get $v104
      f64.neg
      local.set $v139
      local.get $v104
      f64.neg
      local.set $v140
      local.get $v104
      f64.neg
      local.set $v141
      local.get $v104
      f64.neg
      local.set $v142
      local.get $v104
      f64.neg
      local.set $v143
      local.get $v104
      f64.neg
      local.set $v144
      local.get $v104
      f64.neg
      local.set $v145
      local.get $v104
      f64.neg
      local.set $v146
      local.get $v104
      f64.neg
      local.set $v147
      local.get $v104
      f64.neg
      local.set $v148
      local.get $v104
      f64.neg
      local.set $v149
      local.get $v104
      f64.neg
      local.set $v150
      local.get $v104
      f64.neg
      local.set $v151
      local.get $v104
      f64.neg
      local.set $v152
      local.get $v104
      f64.neg
      local.set $v153
      local.get $v104
      f64.neg
      local.set $v154
      local.get $v104
      f64.neg
      local.set $v155
      local.get $v104
      f64.neg
      local.set $v156
      local.get $v104
      f64.neg
      local.set $v157
      local.get $v104
      f64.neg
      local.set $v158
      local.get $v104
      f64.neg
      local.set $v159
      local.get $v104
      f64.neg
      local.set $v160
      local.get $v104
      f64.neg
      local.set $v161
      local.get $v104
      f64.neg
      local.set $v162
      local.get $v104
      f64.neg
      local.set $v163
      local.get $v104
      f64.neg
      local.set $v164
      local.get $v104
      f64.neg
      local.set $v165
      local.get $v104
      f64.neg
      local.set $v166
      local.get $v104
      f64.neg
      local.set $v167
      local.get $v104
      f64.neg
      local.set $v168
      local.get $v104
      f64.neg
      local.set $v169
      local.get $v104
      f64.neg
      local.set $v170
      local.get $v104
      f64.neg
      local.set $v171
      local.get $v104
      f64.neg
      local.set $v172
      local.get $v104
      f64.neg
      local.set $v173
      local.get $v104
      f64.neg
      local.set $v174
      local.get $v104
      f64.neg
      local.set $v175
      local.get $v104
      f64.neg
      local.set $v176
      local.get $v104
      f64.neg
      local.set $v177
      local.get $v104
      f64.neg
      local.set $v178
      local.get $v104
      f64.neg
      local.set $v179
      local.get $v104
      f64.neg
      local.set $v180
      local.get $v104
      f64.neg
      local.set $v181
      local.get $v104
      f64.neg
      local.set $v182
      local.get $v104
      f64.neg
      local.set $v183
      local.get $v104
      f64.neg
      local.set $v184
      local.get $v104
      f64.neg
      local.set $v185
      local.get $v104
      f64.neg
      local.set $v186
      local.get $v104
      f64.neg
      local.set $v187
      local.get $v104
      f64.neg
      local.set $v188
      local.get $v104
      f64.neg
      local.set $v189
      local.get $v104
      f64.neg
      local.set $v190
      local.get $v104
      f64.neg
      local.set $v191
      local.get $v104
      f64.neg
      local.set $v192
      local.get $v104
      f64.neg
      local.set $v193
      local.get $v104
      f64.neg
      local.set $v194
      local.get $v104
      f64.neg
      local.set $v195
      local.get $v104
      f64.neg
      local.set $v196
      local.get $v104
      f64.neg
      local.set $v197
      local.get $v104
      f64.neg
      local.set $v198
      local.get $v104
      f64.neg
      local.set $v199
      local.get $v104
      f64.neg
      local.set $v200
      local.get $v104
      f64.neg
      local.set $v201
      local.get $v104
      f64.neg
      local.set $v202
      local.get $v104
      f64.neg
      local.set $v203
      local.get $v104
      f64.neg
      local.set $v204
      local.get $v104
      f64.neg
      local.set $v205
      local.get $v104
      f64.neg
      local.set $v206
      local.get $v104
      f64.neg
      local.set $v207
      local.get $v104
      f64.neg
      local.set $v208
      local.get $v104
      f64.neg
      local.set $v209
      local.get $v104
      f64.neg
      local.set $v210
      local.get $v104
      f64.neg
      local.set $v211
      local.get $v104
      f64.neg
      local.set $v212
      local.get $v104
      f64.neg
      local.set $v213
      local.get $v104
      f64.neg
      local.set $v214
      local.get $v104
      f64.neg
      local.set $v215
      local.get $v104
      f64.neg
      local.set $v216
      local.get $v104
      f64.neg
      local.set $v217
      local.get $v104
      f64.neg
      local.set $v218
      local.get $v104
      f64.neg
      local.set $v219
      local.get $v104
      f64.neg
      local.set $v220
      local.get $v104
      f64.neg
      local.set $v221
      local.get $v104
      f64.neg
      local.set $v222
      local.get $v104
      f64.neg
      local.set $v223
      local.get $v104
      f64.neg
      local.set $v224
      local.get $v104
      f64.neg
      local.set $v225
      local.get $v104
      f64.neg
      local.set $v226
      local.get $v104
      f64.neg
      local.set $v227
      local.get $v104
      f64.neg
      local.set $v228
      local.get $v104
      f64.neg
      local.set $v229
      local.get $v104
      f64.neg
      local.set $v230
      local.get $v104
      f64.neg
      local.set $v231
      local.get $v104
      f64.neg
      local.set $v232
      local.get $v104
      f64.neg
      local.set $v233
      local.get $v104
      f64.neg
      local.set $v234
      local.get $v104
      f64.neg
      local.set $v235
      local.get $v104
      f64.neg
      local.set $v236
      local.get $v104
      f64.neg
      local.set $v237
      local.get $v104
      f64.neg
      local.set $v238
      local.get $v104
      f64.neg
      local.set $v239
      local.get $v104
      f64.neg
      local.set $v240
      local.get $v104
      f64.neg
      local.set $v241
      local.get $v104
      f64.neg
      local.set $v242
      local.get $v104
      f64.neg
      local.set $v243
      local.get $v104
      f64.neg
      local.set $v244
      local.get $v104
      f64.neg
      local.set $v245
      local.get $v104
      f64.neg
      local.set $v246
      local.get $v104
      f64.neg
      local.set $v247
      local.get $v104
      f64.neg
      local.set $v248
      local.get $v104
      f64.neg
      local.set $v249
      local.get $v104
      f64.neg
      local.set $v250
      local.get $v104
      f64.neg
      local.set $v251
      local.get $v104
      f64.neg
      local.set $v252
      local.get $v104
      f64.neg
      local.set $v253
      local.get $v104
      f64.neg
      local.set $v254
      local.get $v104
      f64.neg
      local.set $v255
      local.get $v104
      f64.neg
      local.set $v256
      local.get $v104
      f64.neg
      local.set $v257
      local.get $v104
      f64.neg
      local.set $v258
      local.get $v104
      f64.neg
      local.set $v259
      local.get $v104
      f64.neg
      local.set $v260
      local.get $v104
      f64.neg
      local.set $v261
      local.get $v104
      f64.neg
      local.set $v262
      local.get $v104
      f64.neg
      local.set $v263
      local.get $v104
      f64.neg
      local.set $v264
      local.get $v104
      f64.neg
      local.set $v265
      local.get $v104
      f64.neg
      local.set $v266
      local.get $v104
      f64.neg
      local.set $v267
      local.get $v104
      f64.neg
      local.set $v268
      local.get $v104
      f64.neg
      local.set $v269
      local.get $v104
      f64.neg
      local.set $v270
      local.get $v104
      f64.neg
      local.set $v271
      local.get $v104
      f64.neg
      local.set $v272
      local.get $v104
      f64.neg
      local.set $v273
      local.get $v104
      f64.neg
      local.set $v274
      local.get $v104
      f64.neg
      local.set $v275
      local.get $v104
      f64.neg
      local.set $v276
      local.get $v104
      f64.neg
      local.set $v277
      local.get $v104
      f64.neg
      local.set $v278
      local.get $v104
      f64.neg
      local.set $v279
      local.get $v104
      f64.neg
      local.set $v280
      local.get $v104
      f64.neg
      local.set $v281
      local.get $v104
      f64.neg
      local.set $v282
      local.get $v104
      f64.neg
      local.set $v283
      local.get $v104
      f64.neg
      local.set $v284
      local.get $v104
      f64.neg
      local.set $v285
      local.get $v104
      f64.neg
      local.set $v286
      local.get $v104
      f64.neg
      local.set $v287
      local.get $v104
      f64.neg
      local.set $v288
      local.get $v104
      f64.neg
      local.set $v289
      local.get $v104
      f64.neg
      local.set $v290
      local.get $v104
      f64.neg
      local.set $v291
      local.get $v104
      f64.neg
      local.set $v292
      local.get $v104
      f64.neg
      local.set $v293
      local.get $v104
      f64.neg
      local.set $v294
      local.get $v104
      f64.neg
      local.set $v295
      local.get $v104
      f64.neg
      local.set $v296
      local.get $v104
      f64.neg
      local.set $v297
      local.get $v104
      f64.neg
      local.set $v298
      local.get $v104
      f64.neg
      local.set $v299
      local.get $v104
      f64.neg
      local.set $v300
      local.get $v104
      f64.neg
      local.set $v301
      local.get $v104
      f64.neg
      local.set $v302
      local.get $v104
      f64.neg
      local.set $v303
      local.get $v104
      f64.neg
      local.set $v304
      local.get $v104
      f64.neg
      local.set $v305
      local.get $v104
      f64.neg
      local.set $v306
      local.get $v104
      f64.neg
      local.set $v307
      local.get $v104
      f64.neg
      local.set $v308
      local.get $v104
      f64.neg
      local.set $v309
      local.get $v104
      f64.neg
      local.set $v310
      local.get $v104
      f64.neg
      local.set $v311
      local.get $v104
      f64.neg
      local.set $v312
      local.get $v104
      f64.neg
      local.set $v313
      local.get $v104
      f64.neg
      local.set $v314
      local.get $v104
      f64.neg
      local.set $v315
      local.get $v104
      f64.neg
      local.set $v316
      local.get $v104
      f64.neg
      local.set $v317
      local.get $v104
      f64.neg
      local.set $v318
      local.get $v104
      f64.neg
      local.set $v319
      local.get $v104
      f64.neg
      local.set $v320
      local.get $v104
      f64.neg
      local.set $v321
      local.get $v104
      f64.neg
      local.set $v322
      local.get $v104
      f64.neg
      local.set $v323
      local.get $v104
      f64.neg
      local.set $v324
      local.get $v104
      f64.neg
      local.set $v325
      local.get $v104
      f64.neg
      local.set $v326
      local.get $v104
      f64.neg
      local.set $v327
      local.get $v104
      f64.neg
      local.set $v328
      local.get $v104
      f64.neg
      local.set $v329
      local.get $v104
      f64.neg
      local.set $v330
      local.get $v104
      f64.neg
      local.set $v331
      local.get $v104
      f64.neg
      local.set $v332
      local.get $v104
      f64.neg
      local.set $v333
      local.get $v104
      f64.neg
      local.set $v334
      local.get $v104
      f64.neg
      local.set $v335
      local.get $v104
      f64.neg
      local.set $v336
      local.get $v104
      f64.neg
      local.set $v337
      local.get $v104
      f64.neg
      local.set $v338
      local.get $v104
      f64.neg
      local.set $v339
      local.get $v104
      f64.neg
      local.set $v340
      local.get $v104
      f64.neg
      local.set $v341
      local.get $v104
      f64.neg
      local.set $v342
      local.get $v104
      f64.neg
      local.set $v343
      local.get $v104
      f64.neg
      local.set $v344
      local.get $v104
      f64.neg
      local.set $v345
      local.get $v104
      f64.neg
      local.set $v346
      local.get $v104
      f64.neg
      local.set $v347
      local.get $v104
      f64.neg
      local.set $v348
      local.get $v104
      f64.neg
      local.set $v349
      local.get $v104
      f64.neg
      local.set $v350
      local.get $v104
      f64.neg
      local.set $v351
      local.get $v104
      f64.neg
      local.set $v352
      local.get $v104
      f64.neg
      local.set $v353
      local.get $v104
      f64.neg
      local.set $v354
      local.get $v104
      f64.neg
      local.set $v355
      local.get $v104
      f64.neg
      local.set $v356
      local.get $v104
      f64.neg
      local.set $v357
      local.get $v104
      f64.neg
      local.set $v358
      local.get $v104
      f64.neg
      local.set $v359
      local.get $v104
      f64.neg
      local.set $v360
      local.get $v104
      f64.neg
      local.set $v361
      local.get $v104
      f64.neg
      local.set $v362
      local.get $v104
      f64.neg
      local.set $v363
      local.get $v104
      f64.neg
      local.set $v364
      local.get $v104
      f64.neg
      local.set $v365
      local.get $v104
      f64.neg
      local.set $v366
      local.get $v104
      f64.neg
      local.set $v367
      local.get $v104
      f64.neg
      local.set $v368
      local.get $v104
      f64.neg
      local.set $v369
      local.get $v104
      f64.neg
      local.set $v370
      local.get $v104
      f64.neg
      local.set $v371
      local.get $v104
      f64.neg
      local.set $v372
      local.get $v104
      f64.neg
      local.set $v373
      local.get $v104
      f64.neg
      local.set $v374
      local.get $v104
      f64.neg
      local.set $v375
      local.get $v104
      f64.neg
      local.set $v376
      local.get $v104
      f64.neg
      local.set $v377
      local.get $v104
      f64.neg
      local.set $v378
      local.get $v104
      f64.neg
      local.set $v379
      local.get $v104
      f64.neg
      local.set $v380
      local.get $v104
      f64.neg
      local.set $v381
      local.get $v104
      f64.neg
      local.set $v382
      local.get $v104
      f64.neg
      local.set $v383
      local.get $v104
      f64.neg
      local.set $v384
      local.get $v104
      f64.neg
      local.set $v385
      local.get $v104
      f64.neg
      local.set $v386
      local.get $v104
      f64.neg
      local.set $v387
      local.get $v104
      f64.neg
      local.set $v388
      local.get $v104
      f64.neg
      local.set $v389
      local.get $v104
      f64.neg
      local.set $v390
      local.get $v104
      f64.neg
      local.set $v391
      local.get $v104
      f64.neg
      local.set $v392
      local.get $v104
      f64.neg
      local.set $v393
      local.get $v104
      f64.neg
      local.set $v394
      local.get $v104
      f64.neg
      local.set $v395
      local.get $v104
      f64.neg
      local.set $v396
      local.get $v104
      f64.neg
      local.set $v397
      local.get $v104
      f64.neg
      local.set $v398
      local.get $v104
      f64.neg
      local.set $v399
      local.get $v104
      f64.neg
      local.set $v400
      local.get $v104
      f64.neg
      local.set $v401
      local.get $v104
      f64.neg
      local.set $v402
      local.get $v104
      f64.neg
      local.set $v403
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v94
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v95
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v97
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v98
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v99
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v100
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v102
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v104
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v105
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v106
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v107
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v108
      f64.store
      local.get $out12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v109
      f64.store
      local.get $out13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v110
      f64.store
      local.get $out14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v111
      f64.store
      local.get $out15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v112
      f64.store
      local.get $out16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v113
      f64.store
      local.get $out17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v114
      f64.store
      local.get $out18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v115
      f64.store
      local.get $out19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v116
      f64.store
      local.get $out20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v117
      f64.store
      local.get $out21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v118
      f64.store
      local.get $out22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v119
      f64.store
      local.get $out23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v120
      f64.store
      local.get $out24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v121
      f64.store
      local.get $out25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v122
      f64.store
      local.get $out26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v123
      f64.store
      local.get $out27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v124
      f64.store
      local.get $out28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v125
      f64.store
      local.get $out29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v126
      f64.store
      local.get $out30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v127
      f64.store
      local.get $out31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v128
      f64.store
      local.get $out32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v129
      f64.store
      local.get $out33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v130
      f64.store
      local.get $out34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v131
      f64.store
      local.get $out35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v132
      f64.store
      local.get $out36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v133
      f64.store
      local.get $out37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v134
      f64.store
      local.get $out38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v135
      f64.store
      local.get $out39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v136
      f64.store
      local.get $out40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v137
      f64.store
      local.get $out41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v138
      f64.store
      local.get $out42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v139
      f64.store
      local.get $out43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v140
      f64.store
      local.get $out44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v141
      f64.store
      local.get $out45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v142
      f64.store
      local.get $out46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v143
      f64.store
      local.get $out47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v144
      f64.store
      local.get $out48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v145
      f64.store
      local.get $out49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v146
      f64.store
      local.get $out50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v147
      f64.store
      local.get $out51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v148
      f64.store
      local.get $out52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v149
      f64.store
      local.get $out53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v150
      f64.store
      local.get $out54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v151
      f64.store
      local.get $out55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v152
      f64.store
      local.get $out56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v153
      f64.store
      local.get $out57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v154
      f64.store
      local.get $out58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v155
      f64.store
      local.get $out59
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v156
      f64.store
      local.get $out60
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v157
      f64.store
      local.get $out61
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v158
      f64.store
      local.get $out62
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v159
      f64.store
      local.get $out63
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v160
      f64.store
      local.get $out64
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v161
      f64.store
      local.get $out65
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v162
      f64.store
      local.get $out66
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v163
      f64.store
      local.get $out67
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v164
      f64.store
      local.get $out68
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v165
      f64.store
      local.get $out69
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v166
      f64.store
      local.get $out70
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v167
      f64.store
      local.get $out71
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v168
      f64.store
      local.get $out72
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v169
      f64.store
      local.get $out73
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v170
      f64.store
      local.get $out74
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v171
      f64.store
      local.get $out75
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v172
      f64.store
      local.get $out76
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v173
      f64.store
      local.get $out77
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v174
      f64.store
      local.get $out78
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v175
      f64.store
      local.get $out79
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v176
      f64.store
      local.get $out80
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v177
      f64.store
      local.get $out81
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v178
      f64.store
      local.get $out82
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v179
      f64.store
      local.get $out83
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v180
      f64.store
      local.get $out84
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v181
      f64.store
      local.get $out85
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v182
      f64.store
      local.get $out86
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v183
      f64.store
      local.get $out87
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v184
      f64.store
      local.get $out88
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v185
      f64.store
      local.get $out89
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v186
      f64.store
      local.get $out90
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v187
      f64.store
      local.get $out91
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v188
      f64.store
      local.get $out92
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v189
      f64.store
      local.get $out93
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v190
      f64.store
      local.get $out94
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v191
      f64.store
      local.get $out95
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v192
      f64.store
      local.get $out96
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v193
      f64.store
      local.get $out97
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v194
      f64.store
      local.get $out98
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v195
      f64.store
      local.get $out99
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v196
      f64.store
      local.get $out100
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v197
      f64.store
      local.get $out101
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v198
      f64.store
      local.get $out102
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v199
      f64.store
      local.get $out103
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v200
      f64.store
      local.get $out104
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v201
      f64.store
      local.get $out105
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v202
      f64.store
      local.get $out106
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v203
      f64.store
      local.get $out107
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v204
      f64.store
      local.get $out108
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v205
      f64.store
      local.get $out109
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v206
      f64.store
      local.get $out110
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v207
      f64.store
      local.get $out111
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v208
      f64.store
      local.get $out112
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v209
      f64.store
      local.get $out113
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v210
      f64.store
      local.get $out114
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v211
      f64.store
      local.get $out115
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v212
      f64.store
      local.get $out116
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v213
      f64.store
      local.get $out117
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v214
      f64.store
      local.get $out118
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v215
      f64.store
      local.get $out119
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v216
      f64.store
      local.get $out120
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v217
      f64.store
      local.get $out121
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v218
      f64.store
      local.get $out122
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v219
      f64.store
      local.get $out123
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v220
      f64.store
      local.get $out124
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v221
      f64.store
      local.get $out125
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v222
      f64.store
      local.get $out126
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v223
      f64.store
      local.get $out127
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v224
      f64.store
      local.get $out128
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v225
      f64.store
      local.get $out129
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v226
      f64.store
      local.get $out130
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v227
      f64.store
      local.get $out131
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v228
      f64.store
      local.get $out132
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v229
      f64.store
      local.get $out133
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v230
      f64.store
      local.get $out134
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v231
      f64.store
      local.get $out135
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v232
      f64.store
      local.get $out136
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v233
      f64.store
      local.get $out137
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v234
      f64.store
      local.get $out138
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v235
      f64.store
      local.get $out139
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v236
      f64.store
      local.get $out140
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v237
      f64.store
      local.get $out141
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v238
      f64.store
      local.get $out142
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v239
      f64.store
      local.get $out143
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v240
      f64.store
      local.get $out144
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v241
      f64.store
      local.get $out145
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v242
      f64.store
      local.get $out146
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v243
      f64.store
      local.get $out147
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v244
      f64.store
      local.get $out148
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v245
      f64.store
      local.get $out149
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v246
      f64.store
      local.get $out150
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v247
      f64.store
      local.get $out151
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v248
      f64.store
      local.get $out152
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v249
      f64.store
      local.get $out153
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v250
      f64.store
      local.get $out154
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v251
      f64.store
      local.get $out155
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v252
      f64.store
      local.get $out156
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v253
      f64.store
      local.get $out157
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v254
      f64.store
      local.get $out158
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v255
      f64.store
      local.get $out159
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v256
      f64.store
      local.get $out160
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v257
      f64.store
      local.get $out161
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v258
      f64.store
      local.get $out162
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v259
      f64.store
      local.get $out163
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v260
      f64.store
      local.get $out164
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v261
      f64.store
      local.get $out165
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v262
      f64.store
      local.get $out166
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v263
      f64.store
      local.get $out167
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v264
      f64.store
      local.get $out168
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v265
      f64.store
      local.get $out169
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v266
      f64.store
      local.get $out170
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v267
      f64.store
      local.get $out171
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v268
      f64.store
      local.get $out172
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v269
      f64.store
      local.get $out173
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v270
      f64.store
      local.get $out174
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v271
      f64.store
      local.get $out175
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v272
      f64.store
      local.get $out176
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v273
      f64.store
      local.get $out177
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v274
      f64.store
      local.get $out178
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v275
      f64.store
      local.get $out179
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v276
      f64.store
      local.get $out180
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v277
      f64.store
      local.get $out181
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v278
      f64.store
      local.get $out182
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v279
      f64.store
      local.get $out183
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v280
      f64.store
      local.get $out184
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v281
      f64.store
      local.get $out185
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v282
      f64.store
      local.get $out186
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v283
      f64.store
      local.get $out187
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v284
      f64.store
      local.get $out188
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v285
      f64.store
      local.get $out189
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v286
      f64.store
      local.get $out190
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v287
      f64.store
      local.get $out191
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v288
      f64.store
      local.get $out192
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v289
      f64.store
      local.get $out193
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v290
      f64.store
      local.get $out194
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v291
      f64.store
      local.get $out195
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v292
      f64.store
      local.get $out196
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v293
      f64.store
      local.get $out197
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v294
      f64.store
      local.get $out198
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v295
      f64.store
      local.get $out199
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v296
      f64.store
      local.get $out200
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v297
      f64.store
      local.get $out201
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v298
      f64.store
      local.get $out202
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v299
      f64.store
      local.get $out203
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v300
      f64.store
      local.get $out204
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v301
      f64.store
      local.get $out205
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v302
      f64.store
      local.get $out206
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v303
      f64.store
      local.get $out207
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v304
      f64.store
      local.get $out208
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v305
      f64.store
      local.get $out209
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v306
      f64.store
      local.get $out210
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v307
      f64.store
      local.get $out211
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v308
      f64.store
      local.get $out212
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v309
      f64.store
      local.get $out213
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v310
      f64.store
      local.get $out214
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v311
      f64.store
      local.get $out215
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v312
      f64.store
      local.get $out216
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v313
      f64.store
      local.get $out217
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v314
      f64.store
      local.get $out218
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v315
      f64.store
      local.get $out219
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v316
      f64.store
      local.get $out220
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v317
      f64.store
      local.get $out221
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v318
      f64.store
      local.get $out222
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v319
      f64.store
      local.get $out223
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v320
      f64.store
      local.get $out224
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v321
      f64.store
      local.get $out225
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v322
      f64.store
      local.get $out226
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v323
      f64.store
      local.get $out227
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v324
      f64.store
      local.get $out228
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v325
      f64.store
      local.get $out229
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v326
      f64.store
      local.get $out230
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v327
      f64.store
      local.get $out231
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v328
      f64.store
      local.get $out232
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v329
      f64.store
      local.get $out233
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v330
      f64.store
      local.get $out234
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v331
      f64.store
      local.get $out235
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v332
      f64.store
      local.get $out236
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v333
      f64.store
      local.get $out237
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v334
      f64.store
      local.get $out238
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v335
      f64.store
      local.get $out239
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v336
      f64.store
      local.get $out240
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v337
      f64.store
      local.get $out241
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v338
      f64.store
      local.get $out242
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v339
      f64.store
      local.get $out243
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v340
      f64.store
      local.get $out244
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v341
      f64.store
      local.get $out245
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v342
      f64.store
      local.get $out246
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v343
      f64.store
      local.get $out247
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v344
      f64.store
      local.get $out248
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v345
      f64.store
      local.get $out249
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v346
      f64.store
      local.get $out250
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v347
      f64.store
      local.get $out251
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v348
      f64.store
      local.get $out252
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v349
      f64.store
      local.get $out253
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v350
      f64.store
      local.get $out254
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v351
      f64.store
      local.get $out255
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v352
      f64.store
      local.get $out256
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v353
      f64.store
      local.get $out257
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v354
      f64.store
      local.get $out258
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v355
      f64.store
      local.get $out259
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v356
      f64.store
      local.get $out260
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v357
      f64.store
      local.get $out261
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v358
      f64.store
      local.get $out262
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v359
      f64.store
      local.get $out263
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v360
      f64.store
      local.get $out264
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v361
      f64.store
      local.get $out265
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v362
      f64.store
      local.get $out266
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v363
      f64.store
      local.get $out267
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v364
      f64.store
      local.get $out268
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v365
      f64.store
      local.get $out269
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v366
      f64.store
      local.get $out270
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v367
      f64.store
      local.get $out271
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v368
      f64.store
      local.get $out272
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v369
      f64.store
      local.get $out273
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v370
      f64.store
      local.get $out274
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v371
      f64.store
      local.get $out275
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v372
      f64.store
      local.get $out276
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v373
      f64.store
      local.get $out277
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v374
      f64.store
      local.get $out278
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v375
      f64.store
      local.get $out279
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v376
      f64.store
      local.get $out280
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v377
      f64.store
      local.get $out281
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v378
      f64.store
      local.get $out282
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v379
      f64.store
      local.get $out283
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v380
      f64.store
      local.get $out284
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v381
      f64.store
      local.get $out285
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v382
      f64.store
      local.get $out286
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v383
      f64.store
      local.get $out287
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v384
      f64.store
      local.get $out288
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v385
      f64.store
      local.get $out289
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v386
      f64.store
      local.get $out290
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v387
      f64.store
      local.get $out291
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v388
      f64.store
      local.get $out292
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v389
      f64.store
      local.get $out293
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v390
      f64.store
      local.get $out294
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v391
      f64.store
      local.get $out295
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v392
      f64.store
      local.get $out296
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v393
      f64.store
      local.get $out297
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v394
      f64.store
      local.get $out298
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v395
      f64.store
      local.get $out299
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v396
      f64.store
      local.get $out300
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v397
      f64.store
      local.get $out301
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v398
      f64.store
      local.get $out302
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v399
      f64.store
      local.get $out303
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v400
      f64.store
      local.get $out304
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v401
      f64.store
      local.get $out305
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v402
      f64.store
      local.get $out306
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v403
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render_chunk1__1
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $feed49 i32) (param $feed50 i32) (param $feed51 i32) (param $feed52 i32) (param $feed53 i32) (param $feed54 i32) (param $feed55 i32) (param $feed56 i32) (param $feed57 i32) (param $feed58 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32) (param $out12 i32) (param $out13 i32) (param $out14 i32) (param $out15 i32) (param $out16 i32) (param $out17 i32) (param $out18 i32) (param $out19 i32) (param $out20 i32) (param $out21 i32) (param $out22 i32) (param $out23 i32) (param $out24 i32) (param $out25 i32) (param $out26 i32) (param $out27 i32) (param $out28 i32) (param $out29 i32) (param $out30 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $feed49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v49
      local.get $feed50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v50
      local.get $feed51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v51
      local.get $feed52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v52
      local.get $feed53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v53
      local.get $feed54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v54
      local.get $feed55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v55
      local.get $feed56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v56
      local.get $feed57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v57
      local.get $feed58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v58
      local.get $v0
      f64.neg
      local.set $v59
      local.get $v0
      f64.neg
      local.set $v60
      local.get $v0
      f64.neg
      local.set $v61
      local.get $v0
      f64.neg
      local.set $v62
      local.get $v0
      f64.neg
      local.set $v63
      local.get $v0
      f64.neg
      local.set $v64
      local.get $v0
      f64.neg
      local.set $v65
      local.get $v0
      f64.neg
      local.set $v66
      local.get $v0
      f64.neg
      local.set $v67
      local.get $v0
      f64.neg
      local.set $v68
      local.get $v0
      f64.neg
      local.set $v69
      local.get $v0
      f64.neg
      local.set $v70
      local.get $v0
      f64.neg
      local.set $v71
      local.get $v0
      f64.neg
      local.set $v72
      local.get $v0
      f64.neg
      local.set $v73
      local.get $v0
      f64.neg
      local.set $v74
      local.get $v1
      local.get $v2
      f64.mul
      local.set $v75
      local.get $v3
      local.get $v4
      f64.add
      local.set $v76
      local.get $v75
      local.get $v76
      f64.add
      local.set $v77
      local.get $v77
      local.get $v0
      f64.min
      local.set $v78
      local.get $v78
      local.get $v73
      f64.max
      local.set $v79
      local.get $v79
      local.get $v79
      f64.mul
      local.set $v80
      local.get $v2
      local.get $v2
      f64.mul
      local.set $v81
      local.get $v5
      local.get $v81
      f64.sub
      local.set $v82
      local.get $v82
      local.get $v6
      f64.add
      local.set $v83
      local.get $v83
      local.get $v0
      f64.min
      local.set $v84
      local.get $v84
      local.get $v74
      f64.max
      local.set $v85
      local.get $v85
      f64.const 2.0
      f64.mul
      local.set $v86
      local.get $v86
      local.get $v79
      f64.mul
      local.set $v87
      local.get $v87
      local.get $v76
      f64.add
      local.set $v88
      local.get $v88
      local.get $v0
      f64.min
      local.set $v89
      local.get $v89
      local.get $v7
      f64.max
      local.set $v90
      local.get $v85
      local.get $v85
      f64.mul
      local.set $v91
      local.get $v91
      local.get $v80
      f64.add
      local.set $v92
      local.get $v91
      local.get $v80
      f64.sub
      local.set $v93
      local.get $v93
      local.get $v6
      f64.add
      local.set $v94
      local.get $v94
      local.get $v0
      f64.min
      local.set $v95
      local.get $v95
      local.get $v8
      f64.max
      local.set $v96
      local.get $v92
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v97
      local.get $v96
      f64.const 2.0
      f64.mul
      local.set $v98
      local.get $v98
      local.get $v90
      f64.mul
      local.set $v99
      local.get $v99
      local.get $v76
      f64.add
      local.set $v100
      local.get $v100
      local.get $v0
      f64.min
      local.set $v101
      local.get $v101
      local.get $v9
      f64.max
      local.set $v102
      local.get $v102
      local.get $v102
      f64.mul
      local.set $v103
      local.get $v96
      local.get $v96
      f64.mul
      local.set $v104
      local.get $v90
      local.get $v90
      f64.mul
      local.set $v105
      local.get $v104
      local.get $v105
      f64.add
      local.set $v106
      local.get $v106
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v107
      local.get $v104
      local.get $v105
      f64.sub
      local.set $v108
      local.get $v108
      local.get $v6
      f64.add
      local.set $v109
      local.get $v109
      local.get $v0
      f64.min
      local.set $v110
      local.get $v110
      local.get $v10
      f64.max
      local.set $v111
      local.get $v111
      local.get $v111
      f64.mul
      local.set $v112
      local.get $v112
      local.get $v103
      f64.add
      local.set $v113
      local.get $v112
      local.get $v103
      f64.sub
      local.set $v114
      local.get $v114
      local.get $v6
      f64.add
      local.set $v115
      local.get $v113
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v116
      local.get $v115
      local.get $v0
      f64.min
      local.set $v117
      local.get $v117
      local.get $v11
      f64.max
      local.set $v118
      local.get $v118
      local.get $v118
      f64.mul
      local.set $v119
      local.get $v118
      f64.const 2.0
      f64.mul
      local.set $v120
      local.get $v111
      f64.const 2.0
      f64.mul
      local.set $v121
      local.get $v121
      local.get $v102
      f64.mul
      local.set $v122
      local.get $v122
      local.get $v76
      f64.add
      local.set $v123
      local.get $v123
      local.get $v0
      f64.min
      local.set $v124
      local.get $v124
      local.get $v12
      f64.max
      local.set $v125
      local.get $v125
      local.get $v125
      f64.mul
      local.set $v126
      local.get $v120
      local.get $v125
      f64.mul
      local.set $v127
      local.get $v127
      local.get $v76
      f64.add
      local.set $v128
      local.get $v128
      local.get $v0
      f64.min
      local.set $v129
      local.get $v129
      local.get $v13
      f64.max
      local.set $v130
      local.get $v119
      local.get $v126
      f64.add
      local.set $v131
      local.get $v131
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v132
      local.get $v119
      local.get $v126
      f64.sub
      local.set $v133
      local.get $v133
      local.get $v6
      f64.add
      local.set $v134
      local.get $v134
      local.get $v0
      f64.min
      local.set $v135
      local.get $v135
      local.get $v14
      f64.max
      local.set $v136
      local.get $v136
      f64.const 2.0
      f64.mul
      local.set $v137
      local.get $v136
      local.get $v136
      f64.mul
      local.set $v138
      local.get $v130
      local.get $v130
      f64.mul
      local.set $v139
      local.get $v138
      local.get $v139
      f64.add
      local.set $v140
      local.get $v140
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v141
      local.get $v137
      local.get $v130
      f64.mul
      local.set $v142
      local.get $v142
      local.get $v76
      f64.add
      local.set $v143
      local.get $v143
      local.get $v0
      f64.min
      local.set $v144
      local.get $v138
      local.get $v139
      f64.sub
      local.set $v145
      local.get $v145
      local.get $v6
      f64.add
      local.set $v146
      local.get $v144
      local.get $v15
      f64.max
      local.set $v147
      local.get $v146
      local.get $v0
      f64.min
      local.set $v148
      local.get $v148
      local.get $v16
      f64.max
      local.set $v149
      local.get $v149
      local.get $v149
      f64.mul
      local.set $v150
      local.get $v149
      f64.const 2.0
      f64.mul
      local.set $v151
      local.get $v147
      local.get $v147
      f64.mul
      local.set $v152
      local.get $v151
      local.get $v147
      f64.mul
      local.set $v153
      local.get $v153
      local.get $v76
      f64.add
      local.set $v154
      local.get $v154
      local.get $v0
      f64.min
      local.set $v155
      local.get $v150
      local.get $v152
      f64.add
      local.set $v156
      local.get $v150
      local.get $v152
      f64.sub
      local.set $v157
      local.get $v155
      local.get $v17
      f64.max
      local.set $v158
      local.get $v158
      local.get $v158
      f64.mul
      local.set $v159
      local.get $v157
      local.get $v6
      f64.add
      local.set $v160
      local.get $v160
      local.get $v0
      f64.min
      local.set $v161
      local.get $v161
      local.get $v18
      f64.max
      local.set $v162
      local.get $v162
      local.get $v162
      f64.mul
      local.set $v163
      local.get $v163
      local.get $v159
      f64.sub
      local.set $v164
      local.get $v164
      local.get $v6
      f64.add
      local.set $v165
      local.get $v165
      local.get $v0
      f64.min
      local.set $v166
      local.get $v166
      local.get $v19
      f64.max
      local.set $v167
      local.get $v167
      local.get $v167
      f64.mul
      local.set $v168
      local.get $v167
      f64.const 2.0
      f64.mul
      local.set $v169
      local.get $v162
      f64.const 2.0
      f64.mul
      local.set $v170
      local.get $v170
      local.get $v158
      f64.mul
      local.set $v171
      local.get $v171
      local.get $v76
      f64.add
      local.set $v172
      local.get $v172
      local.get $v0
      f64.min
      local.set $v173
      local.get $v173
      local.get $v20
      f64.max
      local.set $v174
      local.get $v174
      local.get $v174
      f64.mul
      local.set $v175
      local.get $v168
      local.get $v175
      f64.add
      local.set $v176
      local.get $v176
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v177
      local.get $v169
      local.get $v174
      f64.mul
      local.set $v178
      local.get $v178
      local.get $v76
      f64.add
      local.set $v179
      local.get $v179
      local.get $v0
      f64.min
      local.set $v180
      local.get $v168
      local.get $v175
      f64.sub
      local.set $v181
      local.get $v181
      local.get $v6
      f64.add
      local.set $v182
      local.get $v182
      local.get $v0
      f64.min
      local.set $v183
      local.get $v183
      local.get $v21
      f64.max
      local.set $v184
      local.get $v184
      f64.const 2.0
      f64.mul
      local.set $v185
      local.get $v163
      local.get $v159
      f64.add
      local.set $v186
      local.get $v186
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v187
      local.get $v180
      local.get $v22
      f64.max
      local.set $v188
      local.get $v184
      local.get $v184
      f64.mul
      local.set $v189
      local.get $v188
      local.get $v188
      f64.mul
      local.set $v190
      local.get $v189
      local.get $v190
      f64.sub
      local.set $v191
      local.get $v191
      local.get $v6
      f64.add
      local.set $v192
      local.get $v192
      local.get $v0
      f64.min
      local.set $v193
      local.get $v193
      local.get $v23
      f64.max
      local.set $v194
      local.get $v194
      f64.const 2.0
      f64.mul
      local.set $v195
      local.get $v185
      local.get $v188
      f64.mul
      local.set $v196
      local.get $v196
      local.get $v76
      f64.add
      local.set $v197
      local.get $v197
      local.get $v0
      f64.min
      local.set $v198
      local.get $v189
      local.get $v190
      f64.add
      local.set $v199
      local.get $v199
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v200
      local.get $v156
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v201
      local.get $v5
      local.get $v81
      f64.add
      local.set $v202
      local.get $v202
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v203
      local.get $v24
      local.get $v203
      f64.add
      local.set $v204
      local.get $v204
      local.get $v97
      f64.add
      local.set $v205
      local.get $v205
      local.get $v107
      f64.add
      local.set $v206
      local.get $v206
      local.get $v116
      f64.add
      local.set $v207
      local.get $v207
      local.get $v132
      f64.add
      local.set $v208
      local.get $v208
      local.get $v141
      f64.add
      local.set $v209
      local.get $v209
      local.get $v201
      f64.add
      local.set $v210
      local.get $v210
      local.get $v187
      f64.add
      local.set $v211
      local.get $v211
      local.get $v177
      f64.add
      local.set $v212
      local.get $v212
      local.get $v200
      f64.add
      local.set $v213
      local.get $v194
      local.get $v194
      f64.mul
      local.set $v214
      local.get $v0
      f64.neg
      local.set $v215
      local.get $v0
      f64.neg
      local.set $v216
      local.get $v198
      local.get $v25
      f64.max
      local.set $v217
      local.get $v195
      local.get $v217
      f64.mul
      local.set $v218
      local.get $v217
      local.get $v217
      f64.mul
      local.set $v219
      local.get $v214
      local.get $v219
      f64.sub
      local.set $v220
      local.get $v220
      local.get $v6
      f64.add
      local.set $v221
      local.get $v221
      local.get $v0
      f64.min
      local.set $v222
      local.get $v214
      local.get $v219
      f64.add
      local.set $v223
      local.get $v222
      local.get $v26
      f64.max
      local.set $v224
      local.get $v224
      f64.const 2.0
      f64.mul
      local.set $v225
      local.get $v224
      local.get $v224
      f64.mul
      local.set $v226
      local.get $v223
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v227
      local.get $v213
      local.get $v227
      f64.add
      local.set $v228
      local.get $v218
      local.get $v76
      f64.add
      local.set $v229
      local.get $v229
      local.get $v0
      f64.min
      local.set $v230
      local.get $v230
      local.get $v27
      f64.max
      local.set $v231
      local.get $v225
      local.get $v231
      f64.mul
      local.set $v232
      local.get $v231
      local.get $v231
      f64.mul
      local.set $v233
      local.get $v226
      local.get $v233
      f64.add
      local.set $v234
      local.get $v226
      local.get $v233
      f64.sub
      local.set $v235
      local.get $v234
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v236
      local.get $v232
      local.get $v76
      f64.add
      local.set $v237
      local.get $v235
      local.get $v6
      f64.add
      local.set $v238
      local.get $v228
      local.get $v236
      f64.add
      local.set $v239
      local.get $v238
      local.get $v0
      f64.min
      local.set $v240
      local.get $v237
      local.get $v0
      f64.min
      local.set $v241
      local.get $v240
      local.get $v28
      f64.max
      local.set $v242
      local.get $v242
      local.get $v242
      f64.mul
      local.set $v243
      local.get $v241
      local.get $v215
      f64.max
      local.set $v244
      local.get $v0
      f64.neg
      local.set $v245
      local.get $v242
      f64.const 2.0
      f64.mul
      local.set $v246
      local.get $v244
      local.get $v244
      f64.mul
      local.set $v247
      local.get $v246
      local.get $v244
      f64.mul
      local.set $v248
      local.get $v248
      local.get $v76
      f64.add
      local.set $v249
      local.get $v249
      local.get $v0
      f64.min
      local.set $v250
      local.get $v250
      local.get $v216
      f64.max
      local.set $v251
      local.get $v251
      local.get $v251
      f64.mul
      local.set $v252
      local.get $v243
      local.get $v247
      f64.add
      local.set $v253
      local.get $v243
      local.get $v247
      f64.sub
      local.set $v254
      local.get $v254
      local.get $v6
      f64.add
      local.set $v255
      local.get $v255
      local.get $v0
      f64.min
      local.set $v256
      local.get $v256
      local.get $v245
      f64.max
      local.set $v257
      local.get $v257
      local.get $v257
      f64.mul
      local.set $v258
      local.get $v258
      local.get $v252
      f64.add
      local.set $v259
      local.get $v258
      local.get $v252
      f64.sub
      local.set $v260
      local.get $v260
      local.get $v6
      f64.add
      local.set $v261
      local.get $v259
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v262
      local.get $v261
      local.get $v0
      f64.min
      local.set $v263
      local.get $v263
      local.get $v29
      f64.max
      local.set $v264
      local.get $v264
      f64.const 2.0
      f64.mul
      local.set $v265
      local.get $v264
      local.get $v264
      f64.mul
      local.set $v266
      local.get $v257
      f64.const 2.0
      f64.mul
      local.set $v267
      local.get $v267
      local.get $v251
      f64.mul
      local.set $v268
      local.get $v268
      local.get $v76
      f64.add
      local.set $v269
      local.get $v269
      local.get $v0
      f64.min
      local.set $v270
      local.get $v270
      local.get $v30
      f64.max
      local.set $v271
      local.get $v271
      local.get $v271
      f64.mul
      local.set $v272
      local.get $v265
      local.get $v271
      f64.mul
      local.set $v273
      local.get $v266
      local.get $v272
      f64.sub
      local.set $v274
      local.get $v274
      local.get $v6
      f64.add
      local.set $v275
      local.get $v275
      local.get $v0
      f64.min
      local.set $v276
      local.get $v276
      local.get $v31
      f64.max
      local.set $v277
      local.get $v277
      local.get $v277
      f64.mul
      local.set $v278
      local.get $v277
      f64.const 2.0
      f64.mul
      local.set $v279
      local.get $v266
      local.get $v272
      f64.add
      local.set $v280
      local.get $v280
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v281
      local.get $v273
      local.get $v76
      f64.add
      local.set $v282
      local.get $v282
      local.get $v0
      f64.min
      local.set $v283
      local.get $v283
      local.get $v32
      f64.max
      local.set $v284
      local.get $v284
      local.get $v284
      f64.mul
      local.set $v285
      local.get $v279
      local.get $v284
      f64.mul
      local.set $v286
      local.get $v286
      local.get $v76
      f64.add
      local.set $v287
      local.get $v287
      local.get $v0
      f64.min
      local.set $v288
      local.get $v288
      local.get $v33
      f64.max
      local.set $v289
      local.get $v289
      local.get $v289
      f64.mul
      local.set $v290
      local.get $v278
      local.get $v285
      f64.add
      local.set $v291
      local.get $v291
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v292
      local.get $v278
      local.get $v285
      f64.sub
      local.set $v293
      local.get $v293
      local.get $v6
      f64.add
      local.set $v294
      local.get $v294
      local.get $v0
      f64.min
      local.set $v295
      local.get $v295
      local.get $v34
      f64.max
      local.set $v296
      local.get $v296
      f64.const 2.0
      f64.mul
      local.set $v297
      local.get $v297
      local.get $v289
      f64.mul
      local.set $v298
      local.get $v298
      local.get $v76
      f64.add
      local.set $v299
      local.get $v299
      local.get $v0
      f64.min
      local.set $v300
      local.get $v300
      local.get $v35
      f64.max
      local.set $v301
      local.get $v301
      local.get $v301
      f64.mul
      local.set $v302
      local.get $v296
      local.get $v296
      f64.mul
      local.set $v303
      local.get $v303
      local.get $v290
      f64.add
      local.set $v304
      local.get $v303
      local.get $v290
      f64.sub
      local.set $v305
      local.get $v305
      local.get $v6
      f64.add
      local.set $v306
      local.get $v304
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v307
      local.get $v306
      local.get $v0
      f64.min
      local.set $v308
      local.get $v308
      local.get $v36
      f64.max
      local.set $v309
      local.get $v309
      f64.const 2.0
      f64.mul
      local.set $v310
      local.get $v310
      local.get $v301
      f64.mul
      local.set $v311
      local.get $v309
      local.get $v309
      f64.mul
      local.set $v312
      local.get $v312
      local.get $v302
      f64.add
      local.set $v313
      local.get $v312
      local.get $v302
      f64.sub
      local.set $v314
      local.get $v314
      local.get $v6
      f64.add
      local.set $v315
      local.get $v313
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v316
      local.get $v253
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v317
      local.get $v239
      local.get $v317
      f64.add
      local.set $v318
      local.get $v318
      local.get $v262
      f64.add
      local.set $v319
      local.get $v319
      local.get $v281
      f64.add
      local.set $v320
      local.get $v320
      local.get $v292
      f64.add
      local.set $v321
      local.get $v321
      local.get $v307
      f64.add
      local.set $v322
      local.get $v311
      local.get $v76
      f64.add
      local.set $v323
      local.get $v322
      local.get $v316
      f64.add
      local.set $v324
      local.get $v315
      local.get $v0
      f64.min
      local.set $v325
      local.get $v323
      local.get $v0
      f64.min
      local.set $v326
      local.get $v325
      local.get $v37
      f64.max
      local.set $v327
      local.get $v326
      local.get $v38
      f64.max
      local.set $v328
      local.get $v327
      local.get $v327
      f64.mul
      local.set $v329
      local.get $v327
      f64.const 2.0
      f64.mul
      local.set $v330
      local.get $v328
      local.get $v328
      f64.mul
      local.set $v331
      local.get $v329
      local.get $v331
      f64.sub
      local.set $v332
      local.get $v332
      local.get $v6
      f64.add
      local.set $v333
      local.get $v333
      local.get $v0
      f64.min
      local.set $v334
      local.get $v334
      local.get $v39
      f64.max
      local.set $v335
      local.get $v335
      local.get $v335
      f64.mul
      local.set $v336
      local.get $v335
      f64.const 2.0
      f64.mul
      local.set $v337
      local.get $v330
      local.get $v328
      f64.mul
      local.set $v338
      local.get $v338
      local.get $v76
      f64.add
      local.set $v339
      local.get $v339
      local.get $v0
      f64.min
      local.set $v340
      local.get $v340
      local.get $v40
      f64.max
      local.set $v341
      local.get $v341
      local.get $v341
      f64.mul
      local.set $v342
      local.get $v337
      local.get $v341
      f64.mul
      local.set $v343
      local.get $v336
      local.get $v342
      f64.add
      local.set $v344
      local.get $v344
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v345
      local.get $v343
      local.get $v76
      f64.add
      local.set $v346
      local.get $v336
      local.get $v342
      f64.sub
      local.set $v347
      local.get $v347
      local.get $v6
      f64.add
      local.set $v348
      local.get $v348
      local.get $v0
      f64.min
      local.set $v349
      local.get $v346
      local.get $v0
      f64.min
      local.set $v350
      local.get $v349
      local.get $v41
      f64.max
      local.set $v351
      local.get $v350
      local.get $v42
      f64.max
      local.set $v352
      local.get $v352
      local.get $v352
      f64.mul
      local.set $v353
      local.get $v351
      f64.const 2.0
      f64.mul
      local.set $v354
      local.get $v354
      local.get $v352
      f64.mul
      local.set $v355
      local.get $v355
      local.get $v76
      f64.add
      local.set $v356
      local.get $v356
      local.get $v0
      f64.min
      local.set $v357
      local.get $v357
      local.get $v43
      f64.max
      local.set $v358
      local.get $v358
      local.get $v358
      f64.mul
      local.set $v359
      local.get $v351
      local.get $v351
      f64.mul
      local.set $v360
      local.get $v360
      local.get $v353
      f64.sub
      local.set $v361
      local.get $v361
      local.get $v6
      f64.add
      local.set $v362
      local.get $v362
      local.get $v0
      f64.min
      local.set $v363
      local.get $v363
      local.get $v44
      f64.max
      local.set $v364
      local.get $v364
      local.get $v364
      f64.mul
      local.set $v365
      local.get $v365
      local.get $v359
      f64.sub
      local.set $v366
      local.get $v366
      local.get $v6
      f64.add
      local.set $v367
      local.get $v367
      local.get $v0
      f64.min
      local.set $v368
      local.get $v368
      local.get $v45
      f64.max
      local.set $v369
      local.get $v369
      local.get $v369
      f64.mul
      local.set $v370
      local.get $v369
      f64.const 2.0
      f64.mul
      local.set $v371
      local.get $v364
      f64.const 2.0
      f64.mul
      local.set $v372
      local.get $v372
      local.get $v358
      f64.mul
      local.set $v373
      local.get $v373
      local.get $v76
      f64.add
      local.set $v374
      local.get $v374
      local.get $v0
      f64.min
      local.set $v375
      local.get $v375
      local.get $v46
      f64.max
      local.set $v376
      local.get $v376
      local.get $v376
      f64.mul
      local.set $v377
      local.get $v371
      local.get $v376
      f64.mul
      local.set $v378
      local.get $v370
      local.get $v377
      f64.add
      local.set $v379
      local.get $v370
      local.get $v377
      f64.sub
      local.set $v380
      local.get $v379
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v381
      local.get $v380
      local.get $v6
      f64.add
      local.set $v382
      local.get $v378
      local.get $v76
      f64.add
      local.set $v383
      local.get $v383
      local.get $v0
      f64.min
      local.set $v384
      local.get $v384
      local.get $v47
      f64.max
      local.set $v385
      local.get $v385
      local.get $v385
      f64.mul
      local.set $v386
      local.get $v382
      local.get $v0
      f64.min
      local.set $v387
      local.get $v387
      local.get $v48
      f64.max
      local.set $v388
      local.get $v388
      local.get $v388
      f64.mul
      local.set $v389
      local.get $v389
      local.get $v386
      f64.sub
      local.set $v390
      local.get $v390
      local.get $v6
      f64.add
      local.set $v391
      local.get $v389
      local.get $v386
      f64.add
      local.set $v392
      local.get $v392
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v393
      local.get $v388
      f64.const 2.0
      f64.mul
      local.set $v394
      local.get $v394
      local.get $v385
      f64.mul
      local.set $v395
      local.get $v395
      local.get $v76
      f64.add
      local.set $v396
      local.get $v391
      local.get $v0
      f64.min
      local.set $v397
      local.get $v397
      local.get $v49
      f64.max
      local.set $v398
      local.get $v398
      local.get $v398
      f64.mul
      local.set $v399
      local.get $v365
      local.get $v359
      f64.add
      local.set $v400
      local.get $v400
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v401
      local.get $v398
      f64.const 2.0
      f64.mul
      local.set $v402
      local.get $v396
      local.get $v0
      f64.min
      local.set $v403
      local.get $v403
      local.get $v50
      f64.max
      local.set $v404
      local.get $v404
      local.get $v404
      f64.mul
      local.set $v405
      local.get $v399
      local.get $v405
      f64.sub
      local.set $v406
      local.get $v399
      local.get $v405
      f64.add
      local.set $v407
      local.get $v406
      local.get $v6
      f64.add
      local.set $v408
      local.get $v407
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v409
      local.get $v408
      local.get $v0
      f64.min
      local.set $v410
      local.get $v410
      local.get $v51
      f64.max
      local.set $v411
      local.get $v411
      f64.const 2.0
      f64.mul
      local.set $v412
      local.get $v411
      local.get $v411
      f64.mul
      local.set $v413
      local.get $v402
      local.get $v404
      f64.mul
      local.set $v414
      local.get $v414
      local.get $v76
      f64.add
      local.set $v415
      local.get $v415
      local.get $v0
      f64.min
      local.set $v416
      local.get $v416
      local.get $v52
      f64.max
      local.set $v417
      local.get $v417
      local.get $v417
      f64.mul
      local.set $v418
      local.get $v413
      local.get $v418
      f64.sub
      local.set $v419
      local.get $v419
      local.get $v6
      f64.add
      local.set $v420
      local.get $v420
      local.get $v0
      f64.min
      local.set $v421
      local.get $v412
      local.get $v417
      f64.mul
      local.set $v422
      local.get $v422
      local.get $v76
      f64.add
      local.set $v423
      local.get $v423
      local.get $v0
      f64.min
      local.set $v424
      local.get $v413
      local.get $v418
      f64.add
      local.set $v425
      local.get $v425
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v426
      local.get $v421
      local.get $v53
      f64.max
      local.set $v427
      local.get $v427
      local.get $v427
      f64.mul
      local.set $v428
      local.get $v424
      local.get $v54
      f64.max
      local.set $v429
      local.get $v429
      local.get $v429
      f64.mul
      local.set $v430
      local.get $v428
      local.get $v430
      f64.add
      local.set $v431
      local.get $v428
      local.get $v430
      f64.sub
      local.set $v432
      local.get $v432
      local.get $v6
      f64.add
      local.set $v433
      local.get $v433
      local.get $v0
      f64.min
      local.set $v434
      local.get $v434
      local.get $v55
      f64.max
      local.set $v435
      local.get $v435
      f64.const 2.0
      f64.mul
      local.set $v436
      local.get $v435
      local.get $v435
      f64.mul
      local.set $v437
      local.get $v427
      f64.const 2.0
      f64.mul
      local.set $v438
      local.get $v438
      local.get $v429
      f64.mul
      local.set $v439
      local.get $v439
      local.get $v76
      f64.add
      local.set $v440
      local.get $v440
      local.get $v0
      f64.min
      local.set $v441
      local.get $v441
      local.get $v56
      f64.max
      local.set $v442
      local.get $v436
      local.get $v442
      f64.mul
      local.set $v443
      local.get $v442
      local.get $v442
      f64.mul
      local.set $v444
      local.get $v437
      local.get $v444
      f64.add
      local.set $v445
      local.get $v445
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v446
      local.get $v437
      local.get $v444
      f64.sub
      local.set $v447
      local.get $v447
      local.get $v6
      f64.add
      local.set $v448
      local.get $v448
      local.get $v0
      f64.min
      local.set $v449
      local.get $v449
      local.get $v57
      f64.max
      local.set $v450
      local.get $v450
      local.get $v450
      f64.mul
      local.set $v451
      local.get $v450
      f64.const 2.0
      f64.mul
      local.set $v452
      local.get $v443
      local.get $v76
      f64.add
      local.set $v453
      local.get $v453
      local.get $v0
      f64.min
      local.set $v454
      local.get $v454
      local.get $v58
      f64.max
      local.set $v455
      local.get $v455
      local.get $v455
      f64.mul
      local.set $v456
      local.get $v452
      local.get $v455
      f64.mul
      local.set $v457
      local.get $v457
      local.get $v76
      f64.add
      local.set $v458
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v59
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v60
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v61
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v62
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v63
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v64
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v65
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v66
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v67
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v68
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v69
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v70
      f64.store
      local.get $out12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v71
      f64.store
      local.get $out13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v72
      f64.store
      local.get $out14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v76
      f64.store
      local.get $out15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v324
      f64.store
      local.get $out16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v329
      f64.store
      local.get $out17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v331
      f64.store
      local.get $out18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v345
      f64.store
      local.get $out19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v353
      f64.store
      local.get $out20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v360
      f64.store
      local.get $out21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v381
      f64.store
      local.get $out22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v393
      f64.store
      local.get $out23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v401
      f64.store
      local.get $out24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v409
      f64.store
      local.get $out25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v426
      f64.store
      local.get $out26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v431
      f64.store
      local.get $out27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v446
      f64.store
      local.get $out28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v451
      f64.store
      local.get $out29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v456
      f64.store
      local.get $out30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v458
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render_chunk2__2
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $feed49 i32) (param $feed50 i32) (param $feed51 i32) (param $feed52 i32) (param $feed53 i32) (param $feed54 i32) (param $feed55 i32) (param $feed56 i32) (param $feed57 i32) (param $feed58 i32) (param $feed59 i32) (param $feed60 i32) (param $feed61 i32) (param $feed62 i32) (param $feed63 i32) (param $feed64 i32) (param $feed65 i32) (param $feed66 i32) (param $feed67 i32) (param $feed68 i32) (param $feed69 i32) (param $feed70 i32) (param $feed71 i32) (param $feed72 i32) (param $feed73 i32) (param $feed74 i32) (param $feed75 i32) (param $feed76 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32) (param $out12 i32) (param $out13 i32) (param $out14 i32) (param $out15 i32) (param $out16 i32) (param $out17 i32) (param $out18 i32) (param $out19 i32) (param $out20 i32) (param $out21 i32) (param $out22 i32) (param $out23 i32) (param $out24 i32) (param $out25 i32) (param $out26 i32) (param $out27 i32) (param $out28 i32) (param $out29 i32) (param $out30 i32) (param $out31 i32) (param $out32 i32) (param $out33 i32) (param $out34 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (local $v459 f64)
    (local $v460 f64)
    (local $v461 f64)
    (local $v462 f64)
    (local $v463 f64)
    (local $v464 f64)
    (local $v465 f64)
    (local $v466 f64)
    (local $v467 f64)
    (local $v468 f64)
    (local $v469 f64)
    (local $v470 f64)
    (local $v471 f64)
    (local $v472 f64)
    (local $v473 f64)
    (local $v474 f64)
    (local $v475 f64)
    (local $v476 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $feed49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v49
      local.get $feed50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v50
      local.get $feed51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v51
      local.get $feed52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v52
      local.get $feed53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v53
      local.get $feed54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v54
      local.get $feed55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v55
      local.get $feed56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v56
      local.get $feed57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v57
      local.get $feed58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v58
      local.get $feed59
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v59
      local.get $feed60
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v60
      local.get $feed61
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v61
      local.get $feed62
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v62
      local.get $feed63
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v63
      local.get $feed64
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v64
      local.get $feed65
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v65
      local.get $feed66
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v66
      local.get $feed67
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v67
      local.get $feed68
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v68
      local.get $feed69
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v69
      local.get $feed70
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v70
      local.get $feed71
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v71
      local.get $feed72
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v72
      local.get $feed73
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v73
      local.get $feed74
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v74
      local.get $feed75
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v75
      local.get $feed76
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v76
      local.get $v0
      local.get $v1
      f64.add
      local.set $v77
      local.get $v77
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v78
      local.get $v0
      local.get $v1
      f64.sub
      local.set $v79
      local.get $v79
      local.get $v2
      f64.add
      local.set $v80
      local.get $v80
      local.get $v3
      f64.min
      local.set $v81
      local.get $v81
      local.get $v4
      f64.max
      local.set $v82
      local.get $v82
      local.get $v82
      f64.mul
      local.set $v83
      local.get $v82
      f64.const 2.0
      f64.mul
      local.set $v84
      local.get $v5
      local.get $v3
      f64.min
      local.set $v85
      local.get $v85
      local.get $v6
      f64.max
      local.set $v86
      local.get $v7
      local.get $v8
      f64.add
      local.set $v87
      local.get $v87
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v88
      local.get $v9
      local.get $v10
      f64.add
      local.set $v89
      local.get $v89
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v90
      local.get $v11
      local.get $v90
      f64.add
      local.set $v91
      local.get $v91
      local.get $v12
      f64.add
      local.set $v92
      local.get $v92
      local.get $v88
      f64.add
      local.set $v93
      local.get $v93
      local.get $v13
      f64.add
      local.set $v94
      local.get $v94
      local.get $v14
      f64.add
      local.set $v95
      local.get $v95
      local.get $v15
      f64.add
      local.set $v96
      local.get $v96
      local.get $v16
      f64.add
      local.set $v97
      local.get $v97
      local.get $v17
      f64.add
      local.set $v98
      local.get $v86
      local.get $v86
      f64.mul
      local.set $v99
      local.get $v83
      local.get $v99
      f64.add
      local.set $v100
      local.get $v83
      local.get $v99
      f64.sub
      local.set $v101
      local.get $v101
      local.get $v2
      f64.add
      local.set $v102
      local.get $v100
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v103
      local.get $v102
      local.get $v3
      f64.min
      local.set $v104
      local.get $v104
      local.get $v18
      f64.max
      local.set $v105
      local.get $v105
      local.get $v105
      f64.mul
      local.set $v106
      local.get $v105
      f64.const 2.0
      f64.mul
      local.set $v107
      local.get $v84
      local.get $v86
      f64.mul
      local.set $v108
      local.get $v108
      local.get $v19
      f64.add
      local.set $v109
      local.get $v109
      local.get $v3
      f64.min
      local.set $v110
      local.get $v110
      local.get $v20
      f64.max
      local.set $v111
      local.get $v111
      local.get $v111
      f64.mul
      local.set $v112
      local.get $v106
      local.get $v112
      f64.sub
      local.set $v113
      local.get $v106
      local.get $v112
      f64.add
      local.set $v114
      local.get $v113
      local.get $v2
      f64.add
      local.set $v115
      local.get $v114
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v116
      local.get $v115
      local.get $v3
      f64.min
      local.set $v117
      local.get $v117
      local.get $v21
      f64.max
      local.set $v118
      local.get $v118
      f64.const 2.0
      f64.mul
      local.set $v119
      local.get $v118
      local.get $v118
      f64.mul
      local.set $v120
      local.get $v107
      local.get $v111
      f64.mul
      local.set $v121
      local.get $v121
      local.get $v19
      f64.add
      local.set $v122
      local.get $v122
      local.get $v3
      f64.min
      local.set $v123
      local.get $v123
      local.get $v22
      f64.max
      local.set $v124
      local.get $v124
      local.get $v124
      f64.mul
      local.set $v125
      local.get $v119
      local.get $v124
      f64.mul
      local.set $v126
      local.get $v120
      local.get $v125
      f64.sub
      local.set $v127
      local.get $v120
      local.get $v125
      f64.add
      local.set $v128
      local.get $v127
      local.get $v2
      f64.add
      local.set $v129
      local.get $v129
      local.get $v3
      f64.min
      local.set $v130
      local.get $v130
      local.get $v23
      f64.max
      local.set $v131
      local.get $v131
      local.get $v131
      f64.mul
      local.set $v132
      local.get $v126
      local.get $v19
      f64.add
      local.set $v133
      local.get $v133
      local.get $v3
      f64.min
      local.set $v134
      local.get $v134
      local.get $v24
      f64.max
      local.set $v135
      local.get $v128
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v136
      local.get $v131
      f64.const 2.0
      f64.mul
      local.set $v137
      local.get $v135
      local.get $v135
      f64.mul
      local.set $v138
      local.get $v132
      local.get $v138
      f64.add
      local.set $v139
      local.get $v139
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v140
      local.get $v132
      local.get $v138
      f64.sub
      local.set $v141
      local.get $v137
      local.get $v135
      f64.mul
      local.set $v142
      local.get $v142
      local.get $v19
      f64.add
      local.set $v143
      local.get $v143
      local.get $v3
      f64.min
      local.set $v144
      local.get $v144
      local.get $v25
      f64.max
      local.set $v145
      local.get $v145
      local.get $v145
      f64.mul
      local.set $v146
      local.get $v141
      local.get $v2
      f64.add
      local.set $v147
      local.get $v147
      local.get $v3
      f64.min
      local.set $v148
      local.get $v148
      local.get $v26
      f64.max
      local.set $v149
      local.get $v149
      local.get $v149
      f64.mul
      local.set $v150
      local.get $v150
      local.get $v146
      f64.sub
      local.set $v151
      local.get $v151
      local.get $v2
      f64.add
      local.set $v152
      local.get $v152
      local.get $v3
      f64.min
      local.set $v153
      local.get $v153
      local.get $v27
      f64.max
      local.set $v154
      local.get $v154
      local.get $v154
      f64.mul
      local.set $v155
      local.get $v150
      local.get $v146
      f64.add
      local.set $v156
      local.get $v156
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v157
      local.get $v149
      f64.const 2.0
      f64.mul
      local.set $v158
      local.get $v158
      local.get $v145
      f64.mul
      local.set $v159
      local.get $v159
      local.get $v19
      f64.add
      local.set $v160
      local.get $v160
      local.get $v3
      f64.min
      local.set $v161
      local.get $v161
      local.get $v28
      f64.max
      local.set $v162
      local.get $v154
      f64.const 2.0
      f64.mul
      local.set $v163
      local.get $v162
      local.get $v162
      f64.mul
      local.set $v164
      local.get $v155
      local.get $v164
      f64.sub
      local.set $v165
      local.get $v165
      local.get $v2
      f64.add
      local.set $v166
      local.get $v166
      local.get $v3
      f64.min
      local.set $v167
      local.get $v167
      local.get $v29
      f64.max
      local.set $v168
      local.get $v168
      f64.const 2.0
      f64.mul
      local.set $v169
      local.get $v168
      local.get $v168
      f64.mul
      local.set $v170
      local.get $v155
      local.get $v164
      f64.add
      local.set $v171
      local.get $v171
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v172
      local.get $v163
      local.get $v162
      f64.mul
      local.set $v173
      local.get $v173
      local.get $v19
      f64.add
      local.set $v174
      local.get $v174
      local.get $v3
      f64.min
      local.set $v175
      local.get $v175
      local.get $v30
      f64.max
      local.set $v176
      local.get $v176
      local.get $v176
      f64.mul
      local.set $v177
      local.get $v170
      local.get $v177
      f64.sub
      local.set $v178
      local.get $v170
      local.get $v177
      f64.add
      local.set $v179
      local.get $v178
      local.get $v2
      f64.add
      local.set $v180
      local.get $v169
      local.get $v176
      f64.mul
      local.set $v181
      local.get $v179
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v182
      local.get $v180
      local.get $v3
      f64.min
      local.set $v183
      local.get $v183
      local.get $v31
      f64.max
      local.set $v184
      local.get $v184
      f64.const 2.0
      f64.mul
      local.set $v185
      local.get $v184
      local.get $v184
      f64.mul
      local.set $v186
      local.get $v181
      local.get $v19
      f64.add
      local.set $v187
      local.get $v187
      local.get $v3
      f64.min
      local.set $v188
      local.get $v188
      local.get $v32
      f64.max
      local.set $v189
      local.get $v189
      local.get $v189
      f64.mul
      local.set $v190
      local.get $v185
      local.get $v189
      f64.mul
      local.set $v191
      local.get $v191
      local.get $v19
      f64.add
      local.set $v192
      local.get $v192
      local.get $v3
      f64.min
      local.set $v193
      local.get $v193
      local.get $v33
      f64.max
      local.set $v194
      local.get $v194
      local.get $v194
      f64.mul
      local.set $v195
      local.get $v186
      local.get $v190
      f64.add
      local.set $v196
      local.get $v196
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v197
      local.get $v186
      local.get $v190
      f64.sub
      local.set $v198
      local.get $v198
      local.get $v2
      f64.add
      local.set $v199
      local.get $v199
      local.get $v3
      f64.min
      local.set $v200
      local.get $v200
      local.get $v34
      f64.max
      local.set $v201
      local.get $v201
      local.get $v201
      f64.mul
      local.set $v202
      local.get $v202
      local.get $v195
      f64.sub
      local.set $v203
      local.get $v203
      local.get $v2
      f64.add
      local.set $v204
      local.get $v204
      local.get $v3
      f64.min
      local.set $v205
      local.get $v205
      local.get $v35
      f64.max
      local.set $v206
      local.get $v206
      f64.const 2.0
      f64.mul
      local.set $v207
      local.get $v206
      local.get $v206
      f64.mul
      local.set $v208
      local.get $v202
      local.get $v195
      f64.add
      local.set $v209
      local.get $v209
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v210
      local.get $v201
      f64.const 2.0
      f64.mul
      local.set $v211
      local.get $v211
      local.get $v194
      f64.mul
      local.set $v212
      local.get $v212
      local.get $v19
      f64.add
      local.set $v213
      local.get $v213
      local.get $v3
      f64.min
      local.set $v214
      local.get $v214
      local.get $v36
      f64.max
      local.set $v215
      local.get $v215
      local.get $v215
      f64.mul
      local.set $v216
      local.get $v207
      local.get $v215
      f64.mul
      local.set $v217
      local.get $v217
      local.get $v19
      f64.add
      local.set $v218
      local.get $v218
      local.get $v3
      f64.min
      local.set $v219
      local.get $v219
      local.get $v37
      f64.max
      local.set $v220
      local.get $v208
      local.get $v216
      f64.add
      local.set $v221
      local.get $v221
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v222
      local.get $v208
      local.get $v216
      f64.sub
      local.set $v223
      local.get $v223
      local.get $v2
      f64.add
      local.set $v224
      local.get $v224
      local.get $v3
      f64.min
      local.set $v225
      local.get $v225
      local.get $v38
      f64.max
      local.set $v226
      local.get $v226
      local.get $v226
      f64.mul
      local.set $v227
      local.get $v226
      f64.const 2.0
      f64.mul
      local.set $v228
      local.get $v220
      local.get $v220
      f64.mul
      local.set $v229
      local.get $v227
      local.get $v229
      f64.sub
      local.set $v230
      local.get $v230
      local.get $v2
      f64.add
      local.set $v231
      local.get $v231
      local.get $v3
      f64.min
      local.set $v232
      local.get $v232
      local.get $v39
      f64.max
      local.set $v233
      local.get $v233
      f64.const 2.0
      f64.mul
      local.set $v234
      local.get $v227
      local.get $v229
      f64.add
      local.set $v235
      local.get $v235
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v236
      local.get $v228
      local.get $v220
      f64.mul
      local.set $v237
      local.get $v237
      local.get $v19
      f64.add
      local.set $v238
      local.get $v238
      local.get $v3
      f64.min
      local.set $v239
      local.get $v233
      local.get $v233
      f64.mul
      local.set $v240
      local.get $v239
      local.get $v40
      f64.max
      local.set $v241
      local.get $v241
      local.get $v241
      f64.mul
      local.set $v242
      local.get $v240
      local.get $v242
      f64.sub
      local.set $v243
      local.get $v243
      local.get $v2
      f64.add
      local.set $v244
      local.get $v234
      local.get $v241
      f64.mul
      local.set $v245
      local.get $v244
      local.get $v3
      f64.min
      local.set $v246
      local.get $v246
      local.get $v41
      f64.max
      local.set $v247
      local.get $v247
      f64.const 2.0
      f64.mul
      local.set $v248
      local.get $v247
      local.get $v247
      f64.mul
      local.set $v249
      local.get $v245
      local.get $v19
      f64.add
      local.set $v250
      local.get $v250
      local.get $v3
      f64.min
      local.set $v251
      local.get $v251
      local.get $v42
      f64.max
      local.set $v252
      local.get $v248
      local.get $v252
      f64.mul
      local.set $v253
      local.get $v253
      local.get $v19
      f64.add
      local.set $v254
      local.get $v252
      local.get $v252
      f64.mul
      local.set $v255
      local.get $v249
      local.get $v255
      f64.sub
      local.set $v256
      local.get $v249
      local.get $v255
      f64.add
      local.set $v257
      local.get $v257
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v258
      local.get $v254
      local.get $v3
      f64.min
      local.set $v259
      local.get $v259
      local.get $v43
      f64.max
      local.set $v260
      local.get $v260
      local.get $v260
      f64.mul
      local.set $v261
      local.get $v240
      local.get $v242
      f64.add
      local.set $v262
      local.get $v262
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v263
      local.get $v256
      local.get $v2
      f64.add
      local.set $v264
      local.get $v264
      local.get $v3
      f64.min
      local.set $v265
      local.get $v265
      local.get $v44
      f64.max
      local.set $v266
      local.get $v266
      local.get $v266
      f64.mul
      local.set $v267
      local.get $v267
      local.get $v261
      f64.sub
      local.set $v268
      local.get $v267
      local.get $v261
      f64.add
      local.set $v269
      local.get $v266
      f64.const 2.0
      f64.mul
      local.set $v270
      local.get $v270
      local.get $v260
      f64.mul
      local.set $v271
      local.get $v268
      local.get $v2
      f64.add
      local.set $v272
      local.get $v271
      local.get $v19
      f64.add
      local.set $v273
      local.get $v273
      local.get $v3
      f64.min
      local.set $v274
      local.get $v274
      local.get $v45
      f64.max
      local.set $v275
      local.get $v275
      local.get $v275
      f64.mul
      local.set $v276
      local.get $v269
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v277
      local.get $v272
      local.get $v3
      f64.min
      local.set $v278
      local.get $v278
      local.get $v46
      f64.max
      local.set $v279
      local.get $v279
      local.get $v279
      f64.mul
      local.set $v280
      local.get $v280
      local.get $v276
      f64.add
      local.set $v281
      local.get $v281
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v282
      local.get $v280
      local.get $v276
      f64.sub
      local.set $v283
      local.get $v283
      local.get $v2
      f64.add
      local.set $v284
      local.get $v279
      f64.const 2.0
      f64.mul
      local.set $v285
      local.get $v285
      local.get $v275
      f64.mul
      local.set $v286
      local.get $v286
      local.get $v19
      f64.add
      local.set $v287
      local.get $v287
      local.get $v3
      f64.min
      local.set $v288
      local.get $v288
      local.get $v47
      f64.max
      local.set $v289
      local.get $v289
      local.get $v289
      f64.mul
      local.set $v290
      local.get $v284
      local.get $v3
      f64.min
      local.set $v291
      local.get $v291
      local.get $v48
      f64.max
      local.set $v292
      local.get $v292
      f64.const 2.0
      f64.mul
      local.set $v293
      local.get $v293
      local.get $v289
      f64.mul
      local.set $v294
      local.get $v294
      local.get $v19
      f64.add
      local.set $v295
      local.get $v295
      local.get $v3
      f64.min
      local.set $v296
      local.get $v296
      local.get $v49
      f64.max
      local.set $v297
      local.get $v292
      local.get $v292
      f64.mul
      local.set $v298
      local.get $v298
      local.get $v290
      f64.add
      local.set $v299
      local.get $v299
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v300
      local.get $v298
      local.get $v290
      f64.sub
      local.set $v301
      local.get $v301
      local.get $v2
      f64.add
      local.set $v302
      local.get $v302
      local.get $v3
      f64.min
      local.set $v303
      local.get $v303
      local.get $v50
      f64.max
      local.set $v304
      local.get $v304
      local.get $v304
      f64.mul
      local.set $v305
      local.get $v304
      f64.const 2.0
      f64.mul
      local.set $v306
      local.get $v297
      local.get $v297
      f64.mul
      local.set $v307
      local.get $v305
      local.get $v307
      f64.sub
      local.set $v308
      local.get $v308
      local.get $v2
      f64.add
      local.set $v309
      local.get $v305
      local.get $v307
      f64.add
      local.set $v310
      local.get $v309
      local.get $v3
      f64.min
      local.set $v311
      local.get $v311
      local.get $v51
      f64.max
      local.set $v312
      local.get $v312
      f64.const 2.0
      f64.mul
      local.set $v313
      local.get $v312
      local.get $v312
      f64.mul
      local.set $v314
      local.get $v310
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v315
      local.get $v306
      local.get $v297
      f64.mul
      local.set $v316
      local.get $v316
      local.get $v19
      f64.add
      local.set $v317
      local.get $v317
      local.get $v3
      f64.min
      local.set $v318
      local.get $v318
      local.get $v52
      f64.max
      local.set $v319
      local.get $v313
      local.get $v319
      f64.mul
      local.set $v320
      local.get $v320
      local.get $v19
      f64.add
      local.set $v321
      local.get $v321
      local.get $v3
      f64.min
      local.set $v322
      local.get $v319
      local.get $v319
      f64.mul
      local.set $v323
      local.get $v314
      local.get $v323
      f64.sub
      local.set $v324
      local.get $v314
      local.get $v323
      f64.add
      local.set $v325
      local.get $v322
      local.get $v53
      f64.max
      local.set $v326
      local.get $v326
      local.get $v326
      f64.mul
      local.set $v327
      local.get $v324
      local.get $v2
      f64.add
      local.set $v328
      local.get $v328
      local.get $v3
      f64.min
      local.set $v329
      local.get $v329
      local.get $v54
      f64.max
      local.set $v330
      local.get $v330
      local.get $v330
      f64.mul
      local.set $v331
      local.get $v331
      local.get $v327
      f64.add
      local.set $v332
      local.get $v332
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v333
      local.get $v331
      local.get $v327
      f64.sub
      local.set $v334
      local.get $v330
      f64.const 2.0
      f64.mul
      local.set $v335
      local.get $v335
      local.get $v326
      f64.mul
      local.set $v336
      local.get $v334
      local.get $v2
      f64.add
      local.set $v337
      local.get $v337
      local.get $v3
      f64.min
      local.set $v338
      local.get $v338
      local.get $v55
      f64.max
      local.set $v339
      local.get $v339
      local.get $v339
      f64.mul
      local.set $v340
      local.get $v339
      f64.const 2.0
      f64.mul
      local.set $v341
      local.get $v336
      local.get $v19
      f64.add
      local.set $v342
      local.get $v342
      local.get $v3
      f64.min
      local.set $v343
      local.get $v343
      local.get $v56
      f64.max
      local.set $v344
      local.get $v344
      local.get $v344
      f64.mul
      local.set $v345
      local.get $v340
      local.get $v345
      f64.add
      local.set $v346
      local.get $v346
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v347
      local.get $v341
      local.get $v344
      f64.mul
      local.set $v348
      local.get $v340
      local.get $v345
      f64.sub
      local.set $v349
      local.get $v349
      local.get $v2
      f64.add
      local.set $v350
      local.get $v350
      local.get $v3
      f64.min
      local.set $v351
      local.get $v351
      local.get $v57
      f64.max
      local.set $v352
      local.get $v352
      f64.const 2.0
      f64.mul
      local.set $v353
      local.get $v352
      local.get $v352
      f64.mul
      local.set $v354
      local.get $v325
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v355
      local.get $v348
      local.get $v19
      f64.add
      local.set $v356
      local.get $v356
      local.get $v3
      f64.min
      local.set $v357
      local.get $v357
      local.get $v58
      f64.max
      local.set $v358
      local.get $v358
      local.get $v358
      f64.mul
      local.set $v359
      local.get $v354
      local.get $v359
      f64.sub
      local.set $v360
      local.get $v354
      local.get $v359
      f64.add
      local.set $v361
      local.get $v361
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v362
      local.get $v353
      local.get $v358
      f64.mul
      local.set $v363
      local.get $v360
      local.get $v2
      f64.add
      local.set $v364
      local.get $v364
      local.get $v3
      f64.min
      local.set $v365
      local.get $v365
      local.get $v59
      f64.max
      local.set $v366
      local.get $v363
      local.get $v19
      f64.add
      local.set $v367
      local.get $v367
      local.get $v3
      f64.min
      local.set $v368
      local.get $v368
      local.get $v60
      f64.max
      local.set $v369
      local.get $v369
      local.get $v369
      f64.mul
      local.set $v370
      local.get $v366
      f64.const 2.0
      f64.mul
      local.set $v371
      local.get $v371
      local.get $v369
      f64.mul
      local.set $v372
      local.get $v366
      local.get $v366
      f64.mul
      local.set $v373
      local.get $v373
      local.get $v370
      f64.sub
      local.set $v374
      local.get $v374
      local.get $v2
      f64.add
      local.set $v375
      local.get $v375
      local.get $v3
      f64.min
      local.set $v376
      local.get $v376
      local.get $v61
      f64.max
      local.set $v377
      local.get $v377
      local.get $v377
      f64.mul
      local.set $v378
      local.get $v377
      f64.const 2.0
      f64.mul
      local.set $v379
      local.get $v373
      local.get $v370
      f64.add
      local.set $v380
      local.get $v380
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v381
      local.get $v372
      local.get $v19
      f64.add
      local.set $v382
      local.get $v382
      local.get $v3
      f64.min
      local.set $v383
      local.get $v383
      local.get $v62
      f64.max
      local.set $v384
      local.get $v379
      local.get $v384
      f64.mul
      local.set $v385
      local.get $v385
      local.get $v19
      f64.add
      local.set $v386
      local.get $v384
      local.get $v384
      f64.mul
      local.set $v387
      local.get $v378
      local.get $v387
      f64.sub
      local.set $v388
      local.get $v378
      local.get $v387
      f64.add
      local.set $v389
      local.get $v389
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v390
      local.get $v388
      local.get $v2
      f64.add
      local.set $v391
      local.get $v391
      local.get $v3
      f64.min
      local.set $v392
      local.get $v392
      local.get $v63
      f64.max
      local.set $v393
      local.get $v393
      local.get $v393
      f64.mul
      local.set $v394
      local.get $v393
      f64.const 2.0
      f64.mul
      local.set $v395
      local.get $v386
      local.get $v3
      f64.min
      local.set $v396
      local.get $v396
      local.get $v64
      f64.max
      local.set $v397
      local.get $v395
      local.get $v397
      f64.mul
      local.set $v398
      local.get $v397
      local.get $v397
      f64.mul
      local.set $v399
      local.get $v394
      local.get $v399
      f64.sub
      local.set $v400
      local.get $v400
      local.get $v2
      f64.add
      local.set $v401
      local.get $v394
      local.get $v399
      f64.add
      local.set $v402
      local.get $v402
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v403
      local.get $v401
      local.get $v3
      f64.min
      local.set $v404
      local.get $v404
      local.get $v65
      f64.max
      local.set $v405
      local.get $v405
      local.get $v405
      f64.mul
      local.set $v406
      local.get $v405
      f64.const 2.0
      f64.mul
      local.set $v407
      local.get $v398
      local.get $v19
      f64.add
      local.set $v408
      local.get $v408
      local.get $v3
      f64.min
      local.set $v409
      local.get $v409
      local.get $v66
      f64.max
      local.set $v410
      local.get $v407
      local.get $v410
      f64.mul
      local.set $v411
      local.get $v410
      local.get $v410
      f64.mul
      local.set $v412
      local.get $v406
      local.get $v412
      f64.sub
      local.set $v413
      local.get $v406
      local.get $v412
      f64.add
      local.set $v414
      local.get $v414
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v415
      local.get $v413
      local.get $v2
      f64.add
      local.set $v416
      local.get $v416
      local.get $v3
      f64.min
      local.set $v417
      local.get $v411
      local.get $v19
      f64.add
      local.set $v418
      local.get $v418
      local.get $v3
      f64.min
      local.set $v419
      local.get $v419
      local.get $v67
      f64.max
      local.set $v420
      local.get $v420
      local.get $v420
      f64.mul
      local.set $v421
      local.get $v417
      local.get $v68
      f64.max
      local.set $v422
      local.get $v422
      f64.const 2.0
      f64.mul
      local.set $v423
      local.get $v423
      local.get $v420
      f64.mul
      local.set $v424
      local.get $v422
      local.get $v422
      f64.mul
      local.set $v425
      local.get $v425
      local.get $v421
      f64.sub
      local.set $v426
      local.get $v425
      local.get $v421
      f64.add
      local.set $v427
      local.get $v427
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v428
      local.get $v424
      local.get $v19
      f64.add
      local.set $v429
      local.get $v429
      local.get $v3
      f64.min
      local.set $v430
      local.get $v430
      local.get $v69
      f64.max
      local.set $v431
      local.get $v431
      local.get $v431
      f64.mul
      local.set $v432
      local.get $v426
      local.get $v2
      f64.add
      local.set $v433
      local.get $v433
      local.get $v3
      f64.min
      local.set $v434
      local.get $v434
      local.get $v70
      f64.max
      local.set $v435
      local.get $v435
      f64.const 2.0
      f64.mul
      local.set $v436
      local.get $v436
      local.get $v431
      f64.mul
      local.set $v437
      local.get $v435
      local.get $v435
      f64.mul
      local.set $v438
      local.get $v438
      local.get $v432
      f64.sub
      local.set $v439
      local.get $v439
      local.get $v2
      f64.add
      local.set $v440
      local.get $v438
      local.get $v432
      f64.add
      local.set $v441
      local.get $v441
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v442
      local.get $v440
      local.get $v3
      f64.min
      local.set $v443
      local.get $v443
      local.get $v71
      f64.max
      local.set $v444
      local.get $v444
      local.get $v444
      f64.mul
      local.set $v445
      local.get $v444
      f64.const 2.0
      f64.mul
      local.set $v446
      local.get $v437
      local.get $v19
      f64.add
      local.set $v447
      local.get $v447
      local.get $v3
      f64.min
      local.set $v448
      local.get $v448
      local.get $v72
      f64.max
      local.set $v449
      local.get $v449
      local.get $v449
      f64.mul
      local.set $v450
      local.get $v445
      local.get $v450
      f64.add
      local.set $v451
      local.get $v451
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v452
      local.get $v446
      local.get $v449
      f64.mul
      local.set $v453
      local.get $v453
      local.get $v19
      f64.add
      local.set $v454
      local.get $v454
      local.get $v3
      f64.min
      local.set $v455
      local.get $v455
      local.get $v73
      f64.max
      local.set $v456
      local.get $v456
      local.get $v456
      f64.mul
      local.set $v457
      local.get $v445
      local.get $v450
      f64.sub
      local.set $v458
      local.get $v458
      local.get $v2
      f64.add
      local.set $v459
      local.get $v459
      local.get $v3
      f64.min
      local.set $v460
      local.get $v460
      local.get $v74
      f64.max
      local.set $v461
      local.get $v461
      local.get $v461
      f64.mul
      local.set $v462
      local.get $v462
      local.get $v457
      f64.sub
      local.set $v463
      local.get $v462
      local.get $v457
      f64.add
      local.set $v464
      local.get $v464
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v465
      local.get $v461
      f64.const 2.0
      f64.mul
      local.set $v466
      local.get $v466
      local.get $v456
      f64.mul
      local.set $v467
      local.get $v467
      local.get $v19
      f64.add
      local.set $v468
      local.get $v468
      local.get $v3
      f64.min
      local.set $v469
      local.get $v469
      local.get $v75
      f64.max
      local.set $v470
      local.get $v470
      local.get $v470
      f64.mul
      local.set $v471
      local.get $v463
      local.get $v2
      f64.add
      local.set $v472
      local.get $v472
      local.get $v3
      f64.min
      local.set $v473
      local.get $v473
      local.get $v76
      f64.max
      local.set $v474
      local.get $v474
      local.get $v474
      f64.mul
      local.set $v475
      local.get $v474
      f64.const 2.0
      f64.mul
      local.set $v476
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v78
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v98
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v103
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v116
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v136
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v140
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v157
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v172
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v182
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v197
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v210
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v222
      f64.store
      local.get $out12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v236
      f64.store
      local.get $out13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v258
      f64.store
      local.get $out14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v263
      f64.store
      local.get $out15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v277
      f64.store
      local.get $out16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v282
      f64.store
      local.get $out17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v300
      f64.store
      local.get $out18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v315
      f64.store
      local.get $out19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v333
      f64.store
      local.get $out20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v347
      f64.store
      local.get $out21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v355
      f64.store
      local.get $out22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v362
      f64.store
      local.get $out23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v381
      f64.store
      local.get $out24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v390
      f64.store
      local.get $out25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v403
      f64.store
      local.get $out26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v415
      f64.store
      local.get $out27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v428
      f64.store
      local.get $out28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v442
      f64.store
      local.get $out29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v452
      f64.store
      local.get $out30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v465
      f64.store
      local.get $out31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v470
      f64.store
      local.get $out32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v471
      f64.store
      local.get $out33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v475
      f64.store
      local.get $out34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v476
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render_chunk3__3
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $feed49 i32) (param $feed50 i32) (param $feed51 i32) (param $feed52 i32) (param $feed53 i32) (param $feed54 i32) (param $feed55 i32) (param $feed56 i32) (param $feed57 i32) (param $feed58 i32) (param $feed59 i32) (param $feed60 i32) (param $feed61 i32) (param $feed62 i32) (param $feed63 i32) (param $feed64 i32) (param $feed65 i32) (param $feed66 i32) (param $feed67 i32) (param $feed68 i32) (param $feed69 i32) (param $feed70 i32) (param $feed71 i32) (param $feed72 i32) (param $feed73 i32) (param $feed74 i32) (param $feed75 i32) (param $feed76 i32) (param $feed77 i32) (param $feed78 i32) (param $feed79 i32) (param $feed80 i32) (param $feed81 i32) (param $feed82 i32) (param $feed83 i32) (param $feed84 i32) (param $feed85 i32) (param $feed86 i32) (param $feed87 i32) (param $feed88 i32) (param $feed89 i32) (param $feed90 i32) (param $feed91 i32) (param $feed92 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (local $v459 f64)
    (local $v460 f64)
    (local $v461 f64)
    (local $v462 f64)
    (local $v463 f64)
    (local $v464 f64)
    (local $v465 f64)
    (local $v466 f64)
    (local $v467 f64)
    (local $v468 f64)
    (local $v469 f64)
    (local $v470 f64)
    (local $v471 f64)
    (local $v472 f64)
    (local $v473 f64)
    (local $v474 f64)
    (local $v475 f64)
    (local $v476 f64)
    (local $v477 f64)
    (local $v478 f64)
    (local $v479 f64)
    (local $v480 f64)
    (local $v481 f64)
    (local $v482 f64)
    (local $v483 f64)
    (local $v484 f64)
    (local $v485 f64)
    (local $v486 f64)
    (local $v487 f64)
    (local $v488 f64)
    (local $v489 f64)
    (local $v490 f64)
    (local $v491 f64)
    (local $v492 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $feed49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v49
      local.get $feed50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v50
      local.get $feed51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v51
      local.get $feed52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v52
      local.get $feed53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v53
      local.get $feed54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v54
      local.get $feed55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v55
      local.get $feed56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v56
      local.get $feed57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v57
      local.get $feed58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v58
      local.get $feed59
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v59
      local.get $feed60
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v60
      local.get $feed61
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v61
      local.get $feed62
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v62
      local.get $feed63
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v63
      local.get $feed64
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v64
      local.get $feed65
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v65
      local.get $feed66
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v66
      local.get $feed67
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v67
      local.get $feed68
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v68
      local.get $feed69
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v69
      local.get $feed70
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v70
      local.get $feed71
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v71
      local.get $feed72
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v72
      local.get $feed73
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v73
      local.get $feed74
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v74
      local.get $feed75
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v75
      local.get $feed76
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v76
      local.get $feed77
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v77
      local.get $feed78
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v78
      local.get $feed79
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v79
      local.get $feed80
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v80
      local.get $feed81
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v81
      local.get $feed82
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v82
      local.get $feed83
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v83
      local.get $feed84
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v84
      local.get $feed85
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v85
      local.get $feed86
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v86
      local.get $feed87
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v87
      local.get $feed88
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v88
      local.get $feed89
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v89
      local.get $feed90
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v90
      local.get $feed91
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v91
      local.get $feed92
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v92
      local.get $v0
      local.get $v1
      f64.mul
      local.set $v93
      local.get $v2
      local.get $v3
      f64.add
      local.set $v94
      local.get $v94
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v95
      local.get $v93
      local.get $v4
      f64.add
      local.set $v96
      local.get $v96
      local.get $v5
      f64.min
      local.set $v97
      local.get $v97
      local.get $v6
      f64.max
      local.set $v98
      local.get $v98
      local.get $v98
      f64.mul
      local.set $v99
      local.get $v2
      local.get $v3
      f64.sub
      local.set $v100
      local.get $v100
      local.get $v7
      f64.add
      local.set $v101
      local.get $v101
      local.get $v5
      f64.min
      local.set $v102
      local.get $v102
      local.get $v8
      f64.max
      local.set $v103
      local.get $v103
      local.get $v103
      f64.mul
      local.set $v104
      local.get $v104
      local.get $v99
      f64.sub
      local.set $v105
      local.get $v103
      f64.const 2.0
      f64.mul
      local.set $v106
      local.get $v106
      local.get $v98
      f64.mul
      local.set $v107
      local.get $v107
      local.get $v4
      f64.add
      local.set $v108
      local.get $v108
      local.get $v5
      f64.min
      local.set $v109
      local.get $v109
      local.get $v9
      f64.max
      local.set $v110
      local.get $v110
      local.get $v110
      f64.mul
      local.set $v111
      local.get $v105
      local.get $v7
      f64.add
      local.set $v112
      local.get $v112
      local.get $v5
      f64.min
      local.set $v113
      local.get $v104
      local.get $v99
      f64.add
      local.set $v114
      local.get $v114
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v115
      local.get $v113
      local.get $v10
      f64.max
      local.set $v116
      local.get $v116
      f64.const 2.0
      f64.mul
      local.set $v117
      local.get $v116
      local.get $v116
      f64.mul
      local.set $v118
      local.get $v118
      local.get $v111
      f64.add
      local.set $v119
      local.get $v118
      local.get $v111
      f64.sub
      local.set $v120
      local.get $v117
      local.get $v110
      f64.mul
      local.set $v121
      local.get $v121
      local.get $v4
      f64.add
      local.set $v122
      local.get $v122
      local.get $v5
      f64.min
      local.set $v123
      local.get $v119
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v124
      local.get $v120
      local.get $v7
      f64.add
      local.set $v125
      local.get $v125
      local.get $v5
      f64.min
      local.set $v126
      local.get $v126
      local.get $v11
      f64.max
      local.set $v127
      local.get $v127
      f64.const 2.0
      f64.mul
      local.set $v128
      local.get $v127
      local.get $v127
      f64.mul
      local.set $v129
      local.get $v123
      local.get $v12
      f64.max
      local.set $v130
      local.get $v130
      local.get $v130
      f64.mul
      local.set $v131
      local.get $v129
      local.get $v131
      f64.add
      local.set $v132
      local.get $v132
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v133
      local.get $v129
      local.get $v131
      f64.sub
      local.set $v134
      local.get $v134
      local.get $v7
      f64.add
      local.set $v135
      local.get $v135
      local.get $v5
      f64.min
      local.set $v136
      local.get $v128
      local.get $v130
      f64.mul
      local.set $v137
      local.get $v137
      local.get $v4
      f64.add
      local.set $v138
      local.get $v138
      local.get $v5
      f64.min
      local.set $v139
      local.get $v139
      local.get $v13
      f64.max
      local.set $v140
      local.get $v140
      local.get $v140
      f64.mul
      local.set $v141
      local.get $v136
      local.get $v14
      f64.max
      local.set $v142
      local.get $v142
      f64.const 2.0
      f64.mul
      local.set $v143
      local.get $v143
      local.get $v140
      f64.mul
      local.set $v144
      local.get $v142
      local.get $v142
      f64.mul
      local.set $v145
      local.get $v145
      local.get $v141
      f64.sub
      local.set $v146
      local.get $v145
      local.get $v141
      f64.add
      local.set $v147
      local.get $v147
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v148
      local.get $v146
      local.get $v7
      f64.add
      local.set $v149
      local.get $v149
      local.get $v5
      f64.min
      local.set $v150
      local.get $v150
      local.get $v15
      f64.max
      local.set $v151
      local.get $v151
      local.get $v151
      f64.mul
      local.set $v152
      local.get $v151
      f64.const 2.0
      f64.mul
      local.set $v153
      local.get $v144
      local.get $v4
      f64.add
      local.set $v154
      local.get $v154
      local.get $v5
      f64.min
      local.set $v155
      local.get $v155
      local.get $v16
      f64.max
      local.set $v156
      local.get $v156
      local.get $v156
      f64.mul
      local.set $v157
      local.get $v152
      local.get $v157
      f64.sub
      local.set $v158
      local.get $v152
      local.get $v157
      f64.add
      local.set $v159
      local.get $v158
      local.get $v7
      f64.add
      local.set $v160
      local.get $v160
      local.get $v5
      f64.min
      local.set $v161
      local.get $v161
      local.get $v17
      f64.max
      local.set $v162
      local.get $v153
      local.get $v156
      f64.mul
      local.set $v163
      local.get $v163
      local.get $v4
      f64.add
      local.set $v164
      local.get $v164
      local.get $v5
      f64.min
      local.set $v165
      local.get $v165
      local.get $v18
      f64.max
      local.set $v166
      local.get $v162
      f64.const 2.0
      f64.mul
      local.set $v167
      local.get $v167
      local.get $v166
      f64.mul
      local.set $v168
      local.get $v168
      local.get $v4
      f64.add
      local.set $v169
      local.get $v162
      local.get $v162
      f64.mul
      local.set $v170
      local.get $v166
      local.get $v166
      f64.mul
      local.set $v171
      local.get $v170
      local.get $v171
      f64.sub
      local.set $v172
      local.get $v172
      local.get $v7
      f64.add
      local.set $v173
      local.get $v170
      local.get $v171
      f64.add
      local.set $v174
      local.get $v174
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v175
      local.get $v169
      local.get $v5
      f64.min
      local.set $v176
      local.get $v176
      local.get $v19
      f64.max
      local.set $v177
      local.get $v177
      local.get $v177
      f64.mul
      local.set $v178
      local.get $v173
      local.get $v5
      f64.min
      local.set $v179
      local.get $v179
      local.get $v20
      f64.max
      local.set $v180
      local.get $v180
      f64.const 2.0
      f64.mul
      local.set $v181
      local.get $v181
      local.get $v177
      f64.mul
      local.set $v182
      local.get $v180
      local.get $v180
      f64.mul
      local.set $v183
      local.get $v183
      local.get $v178
      f64.sub
      local.set $v184
      local.get $v183
      local.get $v178
      f64.add
      local.set $v185
      local.get $v182
      local.get $v4
      f64.add
      local.set $v186
      local.get $v186
      local.get $v5
      f64.min
      local.set $v187
      local.get $v187
      local.get $v21
      f64.max
      local.set $v188
      local.get $v188
      local.get $v188
      f64.mul
      local.set $v189
      local.get $v185
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v190
      local.get $v184
      local.get $v7
      f64.add
      local.set $v191
      local.get $v191
      local.get $v5
      f64.min
      local.set $v192
      local.get $v192
      local.get $v22
      f64.max
      local.set $v193
      local.get $v193
      local.get $v193
      f64.mul
      local.set $v194
      local.get $v194
      local.get $v189
      f64.sub
      local.set $v195
      local.get $v194
      local.get $v189
      f64.add
      local.set $v196
      local.get $v196
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v197
      local.get $v195
      local.get $v7
      f64.add
      local.set $v198
      local.get $v198
      local.get $v5
      f64.min
      local.set $v199
      local.get $v199
      local.get $v23
      f64.max
      local.set $v200
      local.get $v200
      f64.const 2.0
      f64.mul
      local.set $v201
      local.get $v200
      local.get $v200
      f64.mul
      local.set $v202
      local.get $v193
      f64.const 2.0
      f64.mul
      local.set $v203
      local.get $v203
      local.get $v188
      f64.mul
      local.set $v204
      local.get $v204
      local.get $v4
      f64.add
      local.set $v205
      local.get $v205
      local.get $v5
      f64.min
      local.set $v206
      local.get $v206
      local.get $v24
      f64.max
      local.set $v207
      local.get $v207
      local.get $v207
      f64.mul
      local.set $v208
      local.get $v202
      local.get $v208
      f64.sub
      local.set $v209
      local.get $v209
      local.get $v7
      f64.add
      local.set $v210
      local.get $v202
      local.get $v208
      f64.add
      local.set $v211
      local.get $v211
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v212
      local.get $v201
      local.get $v207
      f64.mul
      local.set $v213
      local.get $v210
      local.get $v5
      f64.min
      local.set $v214
      local.get $v214
      local.get $v25
      f64.max
      local.set $v215
      local.get $v215
      local.get $v215
      f64.mul
      local.set $v216
      local.get $v215
      f64.const 2.0
      f64.mul
      local.set $v217
      local.get $v213
      local.get $v4
      f64.add
      local.set $v218
      local.get $v218
      local.get $v5
      f64.min
      local.set $v219
      local.get $v219
      local.get $v26
      f64.max
      local.set $v220
      local.get $v217
      local.get $v220
      f64.mul
      local.set $v221
      local.get $v221
      local.get $v4
      f64.add
      local.set $v222
      local.get $v222
      local.get $v5
      f64.min
      local.set $v223
      local.get $v223
      local.get $v27
      f64.max
      local.set $v224
      local.get $v220
      local.get $v220
      f64.mul
      local.set $v225
      local.get $v216
      local.get $v225
      f64.sub
      local.set $v226
      local.get $v226
      local.get $v7
      f64.add
      local.set $v227
      local.get $v227
      local.get $v5
      f64.min
      local.set $v228
      local.get $v228
      local.get $v28
      f64.max
      local.set $v229
      local.get $v229
      f64.const 2.0
      f64.mul
      local.set $v230
      local.get $v224
      local.get $v224
      f64.mul
      local.set $v231
      local.get $v230
      local.get $v224
      f64.mul
      local.set $v232
      local.get $v232
      local.get $v4
      f64.add
      local.set $v233
      local.get $v233
      local.get $v5
      f64.min
      local.set $v234
      local.get $v234
      local.get $v29
      f64.max
      local.set $v235
      local.get $v235
      local.get $v235
      f64.mul
      local.set $v236
      local.get $v229
      local.get $v229
      f64.mul
      local.set $v237
      local.get $v237
      local.get $v231
      f64.add
      local.set $v238
      local.get $v238
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v239
      local.get $v237
      local.get $v231
      f64.sub
      local.set $v240
      local.get $v240
      local.get $v7
      f64.add
      local.set $v241
      local.get $v241
      local.get $v5
      f64.min
      local.set $v242
      local.get $v242
      local.get $v30
      f64.max
      local.set $v243
      local.get $v243
      local.get $v243
      f64.mul
      local.set $v244
      local.get $v244
      local.get $v236
      f64.add
      local.set $v245
      local.get $v243
      f64.const 2.0
      f64.mul
      local.set $v246
      local.get $v246
      local.get $v235
      f64.mul
      local.set $v247
      local.get $v247
      local.get $v4
      f64.add
      local.set $v248
      local.get $v248
      local.get $v5
      f64.min
      local.set $v249
      local.get $v244
      local.get $v236
      f64.sub
      local.set $v250
      local.get $v250
      local.get $v7
      f64.add
      local.set $v251
      local.get $v251
      local.get $v5
      f64.min
      local.set $v252
      local.get $v245
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v253
      local.get $v249
      local.get $v31
      f64.max
      local.set $v254
      local.get $v254
      local.get $v254
      f64.mul
      local.set $v255
      local.get $v252
      local.get $v32
      f64.max
      local.set $v256
      local.get $v256
      local.get $v256
      f64.mul
      local.set $v257
      local.get $v257
      local.get $v255
      f64.sub
      local.set $v258
      local.get $v257
      local.get $v255
      f64.add
      local.set $v259
      local.get $v259
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v260
      local.get $v258
      local.get $v7
      f64.add
      local.set $v261
      local.get $v256
      f64.const 2.0
      f64.mul
      local.set $v262
      local.get $v262
      local.get $v254
      f64.mul
      local.set $v263
      local.get $v263
      local.get $v4
      f64.add
      local.set $v264
      local.get $v264
      local.get $v5
      f64.min
      local.set $v265
      local.get $v265
      local.get $v33
      f64.max
      local.set $v266
      local.get $v261
      local.get $v5
      f64.min
      local.set $v267
      local.get $v267
      local.get $v34
      f64.max
      local.set $v268
      local.get $v268
      f64.const 2.0
      f64.mul
      local.set $v269
      local.get $v268
      local.get $v268
      f64.mul
      local.set $v270
      local.get $v269
      local.get $v266
      f64.mul
      local.set $v271
      local.get $v271
      local.get $v4
      f64.add
      local.set $v272
      local.get $v272
      local.get $v5
      f64.min
      local.set $v273
      local.get $v273
      local.get $v35
      f64.max
      local.set $v274
      local.get $v274
      local.get $v274
      f64.mul
      local.set $v275
      local.get $v266
      local.get $v266
      f64.mul
      local.set $v276
      local.get $v270
      local.get $v276
      f64.add
      local.set $v277
      local.get $v277
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v278
      local.get $v270
      local.get $v276
      f64.sub
      local.set $v279
      local.get $v279
      local.get $v7
      f64.add
      local.set $v280
      local.get $v280
      local.get $v5
      f64.min
      local.set $v281
      local.get $v281
      local.get $v36
      f64.max
      local.set $v282
      local.get $v282
      local.get $v282
      f64.mul
      local.set $v283
      local.get $v283
      local.get $v275
      f64.sub
      local.set $v284
      local.get $v284
      local.get $v7
      f64.add
      local.set $v285
      local.get $v282
      f64.const 2.0
      f64.mul
      local.set $v286
      local.get $v283
      local.get $v275
      f64.add
      local.set $v287
      local.get $v287
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v288
      local.get $v285
      local.get $v5
      f64.min
      local.set $v289
      local.get $v289
      local.get $v37
      f64.max
      local.set $v290
      local.get $v290
      f64.const 2.0
      f64.mul
      local.set $v291
      local.get $v290
      local.get $v290
      f64.mul
      local.set $v292
      local.get $v286
      local.get $v274
      f64.mul
      local.set $v293
      local.get $v293
      local.get $v4
      f64.add
      local.set $v294
      local.get $v294
      local.get $v5
      f64.min
      local.set $v295
      local.get $v295
      local.get $v38
      f64.max
      local.set $v296
      local.get $v296
      local.get $v296
      f64.mul
      local.set $v297
      local.get $v292
      local.get $v297
      f64.add
      local.set $v298
      local.get $v291
      local.get $v296
      f64.mul
      local.set $v299
      local.get $v299
      local.get $v4
      f64.add
      local.set $v300
      local.get $v300
      local.get $v5
      f64.min
      local.set $v301
      local.get $v301
      local.get $v39
      f64.max
      local.set $v302
      local.get $v302
      local.get $v302
      f64.mul
      local.set $v303
      local.get $v298
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v304
      local.get $v292
      local.get $v297
      f64.sub
      local.set $v305
      local.get $v305
      local.get $v7
      f64.add
      local.set $v306
      local.get $v306
      local.get $v5
      f64.min
      local.set $v307
      local.get $v307
      local.get $v40
      f64.max
      local.set $v308
      local.get $v308
      local.get $v308
      f64.mul
      local.set $v309
      local.get $v309
      local.get $v303
      f64.add
      local.set $v310
      local.get $v309
      local.get $v303
      f64.sub
      local.set $v311
      local.get $v310
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v312
      local.get $v311
      local.get $v7
      f64.add
      local.set $v313
      local.get $v313
      local.get $v5
      f64.min
      local.set $v314
      local.get $v314
      local.get $v41
      f64.max
      local.set $v315
      local.get $v315
      f64.const 2.0
      f64.mul
      local.set $v316
      local.get $v315
      local.get $v315
      f64.mul
      local.set $v317
      local.get $v308
      f64.const 2.0
      f64.mul
      local.set $v318
      local.get $v318
      local.get $v302
      f64.mul
      local.set $v319
      local.get $v319
      local.get $v4
      f64.add
      local.set $v320
      local.get $v320
      local.get $v5
      f64.min
      local.set $v321
      local.get $v321
      local.get $v42
      f64.max
      local.set $v322
      local.get $v322
      local.get $v322
      f64.mul
      local.set $v323
      local.get $v316
      local.get $v322
      f64.mul
      local.set $v324
      local.get $v317
      local.get $v323
      f64.sub
      local.set $v325
      local.get $v325
      local.get $v7
      f64.add
      local.set $v326
      local.get $v326
      local.get $v5
      f64.min
      local.set $v327
      local.get $v327
      local.get $v43
      f64.max
      local.set $v328
      local.get $v328
      local.get $v328
      f64.mul
      local.set $v329
      local.get $v328
      f64.const 2.0
      f64.mul
      local.set $v330
      local.get $v324
      local.get $v4
      f64.add
      local.set $v331
      local.get $v331
      local.get $v5
      f64.min
      local.set $v332
      local.get $v332
      local.get $v44
      f64.max
      local.set $v333
      local.get $v330
      local.get $v333
      f64.mul
      local.set $v334
      local.get $v334
      local.get $v4
      f64.add
      local.set $v335
      local.get $v335
      local.get $v5
      f64.min
      local.set $v336
      local.get $v336
      local.get $v45
      f64.max
      local.set $v337
      local.get $v337
      local.get $v337
      f64.mul
      local.set $v338
      local.get $v333
      local.get $v333
      f64.mul
      local.set $v339
      local.get $v329
      local.get $v339
      f64.add
      local.set $v340
      local.get $v340
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v341
      local.get $v329
      local.get $v339
      f64.sub
      local.set $v342
      local.get $v342
      local.get $v7
      f64.add
      local.set $v343
      local.get $v343
      local.get $v5
      f64.min
      local.set $v344
      local.get $v344
      local.get $v46
      f64.max
      local.set $v345
      local.get $v345
      local.get $v345
      f64.mul
      local.set $v346
      local.get $v346
      local.get $v338
      f64.add
      local.set $v347
      local.get $v345
      f64.const 2.0
      f64.mul
      local.set $v348
      local.get $v348
      local.get $v337
      f64.mul
      local.set $v349
      local.get $v349
      local.get $v4
      f64.add
      local.set $v350
      local.get $v317
      local.get $v323
      f64.add
      local.set $v351
      local.get $v351
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v352
      local.get $v216
      local.get $v225
      f64.add
      local.set $v353
      local.get $v353
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v354
      local.get $v159
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v355
      local.get $v47
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v356
      local.get $v48
      local.get $v356
      f64.add
      local.set $v357
      local.get $v357
      local.get $v49
      f64.add
      local.set $v358
      local.get $v358
      local.get $v50
      f64.add
      local.set $v359
      local.get $v359
      local.get $v51
      f64.add
      local.set $v360
      local.get $v360
      local.get $v52
      f64.add
      local.set $v361
      local.get $v361
      local.get $v53
      f64.add
      local.set $v362
      local.get $v362
      local.get $v54
      f64.add
      local.set $v363
      local.get $v363
      local.get $v55
      f64.add
      local.set $v364
      local.get $v364
      local.get $v56
      f64.add
      local.set $v365
      local.get $v365
      local.get $v57
      f64.add
      local.set $v366
      local.get $v366
      local.get $v58
      f64.add
      local.set $v367
      local.get $v367
      local.get $v59
      f64.add
      local.set $v368
      local.get $v368
      local.get $v60
      f64.add
      local.set $v369
      local.get $v369
      local.get $v61
      f64.add
      local.set $v370
      local.get $v370
      local.get $v62
      f64.add
      local.set $v371
      local.get $v371
      local.get $v63
      f64.add
      local.set $v372
      local.get $v372
      local.get $v64
      f64.add
      local.set $v373
      local.get $v373
      local.get $v65
      f64.add
      local.set $v374
      local.get $v374
      local.get $v66
      f64.add
      local.set $v375
      local.get $v375
      local.get $v67
      f64.add
      local.set $v376
      local.get $v376
      local.get $v68
      f64.add
      local.set $v377
      local.get $v377
      local.get $v69
      f64.add
      local.set $v378
      local.get $v378
      local.get $v70
      f64.add
      local.set $v379
      local.get $v379
      local.get $v71
      f64.add
      local.set $v380
      local.get $v380
      local.get $v72
      f64.add
      local.set $v381
      local.get $v381
      local.get $v73
      f64.add
      local.set $v382
      local.get $v382
      local.get $v74
      f64.add
      local.set $v383
      local.get $v383
      local.get $v75
      f64.add
      local.set $v384
      local.get $v384
      local.get $v76
      f64.add
      local.set $v385
      local.get $v385
      local.get $v77
      f64.add
      local.set $v386
      local.get $v386
      local.get $v78
      f64.add
      local.set $v387
      local.get $v387
      local.get $v79
      f64.add
      local.set $v388
      local.get $v388
      local.get $v95
      f64.add
      local.set $v389
      local.get $v389
      local.get $v115
      f64.add
      local.set $v390
      local.get $v390
      local.get $v124
      f64.add
      local.set $v391
      local.get $v391
      local.get $v133
      f64.add
      local.set $v392
      local.get $v392
      local.get $v148
      f64.add
      local.set $v393
      local.get $v393
      local.get $v355
      f64.add
      local.set $v394
      local.get $v394
      local.get $v175
      f64.add
      local.set $v395
      local.get $v395
      local.get $v190
      f64.add
      local.set $v396
      local.get $v396
      local.get $v197
      f64.add
      local.set $v397
      local.get $v397
      local.get $v212
      f64.add
      local.set $v398
      local.get $v398
      local.get $v354
      f64.add
      local.set $v399
      local.get $v399
      local.get $v239
      f64.add
      local.set $v400
      local.get $v400
      local.get $v253
      f64.add
      local.set $v401
      local.get $v401
      local.get $v260
      f64.add
      local.set $v402
      local.get $v402
      local.get $v278
      f64.add
      local.set $v403
      local.get $v403
      local.get $v288
      f64.add
      local.set $v404
      local.get $v404
      local.get $v304
      f64.add
      local.set $v405
      local.get $v405
      local.get $v312
      f64.add
      local.set $v406
      local.get $v406
      local.get $v352
      f64.add
      local.set $v407
      local.get $v407
      local.get $v341
      f64.add
      local.set $v408
      local.get $v347
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v409
      local.get $v408
      local.get $v409
      f64.add
      local.set $v410
      local.get $v346
      local.get $v338
      f64.sub
      local.set $v411
      local.get $v411
      local.get $v7
      f64.add
      local.set $v412
      local.get $v412
      local.get $v5
      f64.min
      local.set $v413
      local.get $v350
      local.get $v5
      f64.min
      local.set $v414
      local.get $v414
      local.get $v80
      f64.max
      local.set $v415
      local.get $v415
      local.get $v415
      f64.mul
      local.set $v416
      local.get $v413
      local.get $v81
      f64.max
      local.set $v417
      local.get $v417
      f64.const 2.0
      f64.mul
      local.set $v418
      local.get $v418
      local.get $v415
      f64.mul
      local.set $v419
      local.get $v417
      local.get $v417
      f64.mul
      local.set $v420
      local.get $v420
      local.get $v416
      f64.add
      local.set $v421
      local.get $v420
      local.get $v416
      f64.sub
      local.set $v422
      local.get $v419
      local.get $v4
      f64.add
      local.set $v423
      local.get $v423
      local.get $v5
      f64.min
      local.set $v424
      local.get $v424
      local.get $v82
      f64.max
      local.set $v425
      local.get $v425
      local.get $v425
      f64.mul
      local.set $v426
      local.get $v422
      local.get $v7
      f64.add
      local.set $v427
      local.get $v427
      local.get $v5
      f64.min
      local.set $v428
      local.get $v428
      local.get $v83
      f64.max
      local.set $v429
      local.get $v421
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v430
      local.get $v410
      local.get $v430
      f64.add
      local.set $v431
      local.get $v429
      f64.const 2.0
      f64.mul
      local.set $v432
      local.get $v432
      local.get $v425
      f64.mul
      local.set $v433
      local.get $v429
      local.get $v429
      f64.mul
      local.set $v434
      local.get $v434
      local.get $v426
      f64.add
      local.set $v435
      local.get $v434
      local.get $v426
      f64.sub
      local.set $v436
      local.get $v435
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v437
      local.get $v433
      local.get $v4
      f64.add
      local.set $v438
      local.get $v438
      local.get $v5
      f64.min
      local.set $v439
      local.get $v439
      local.get $v84
      f64.max
      local.set $v440
      local.get $v431
      local.get $v437
      f64.add
      local.set $v441
      local.get $v436
      local.get $v7
      f64.add
      local.set $v442
      local.get $v442
      local.get $v5
      f64.min
      local.set $v443
      local.get $v443
      local.get $v85
      f64.max
      local.set $v444
      local.get $v444
      f64.const 2.0
      f64.mul
      local.set $v445
      local.get $v444
      local.get $v444
      f64.mul
      local.set $v446
      local.get $v440
      local.get $v440
      f64.mul
      local.set $v447
      local.get $v446
      local.get $v447
      f64.sub
      local.set $v448
      local.get $v448
      local.get $v7
      f64.add
      local.set $v449
      local.get $v446
      local.get $v447
      f64.add
      local.set $v450
      local.get $v450
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v451
      local.get $v441
      local.get $v451
      f64.add
      local.set $v452
      local.get $v449
      local.get $v5
      f64.min
      local.set $v453
      local.get $v445
      local.get $v440
      f64.mul
      local.set $v454
      local.get $v454
      local.get $v4
      f64.add
      local.set $v455
      local.get $v455
      local.get $v5
      f64.min
      local.set $v456
      local.get $v453
      local.get $v86
      f64.max
      local.set $v457
      local.get $v457
      local.get $v457
      f64.mul
      local.set $v458
      local.get $v457
      f64.const 2.0
      f64.mul
      local.set $v459
      local.get $v456
      local.get $v87
      f64.max
      local.set $v460
      local.get $v460
      local.get $v460
      f64.mul
      local.set $v461
      local.get $v459
      local.get $v460
      f64.mul
      local.set $v462
      local.get $v458
      local.get $v461
      f64.sub
      local.set $v463
      local.get $v463
      local.get $v7
      f64.add
      local.set $v464
      local.get $v458
      local.get $v461
      f64.add
      local.set $v465
      local.get $v465
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v466
      local.get $v462
      local.get $v4
      f64.add
      local.set $v467
      local.get $v452
      local.get $v466
      f64.add
      local.set $v468
      local.get $v464
      local.get $v5
      f64.min
      local.set $v469
      local.get $v469
      local.get $v88
      f64.max
      local.set $v470
      local.get $v470
      f64.const 2.0
      f64.mul
      local.set $v471
      local.get $v470
      local.get $v470
      f64.mul
      local.set $v472
      local.get $v467
      local.get $v5
      f64.min
      local.set $v473
      local.get $v473
      local.get $v89
      f64.max
      local.set $v474
      local.get $v474
      local.get $v474
      f64.mul
      local.set $v475
      local.get $v471
      local.get $v474
      f64.mul
      local.set $v476
      local.get $v476
      local.get $v4
      f64.add
      local.set $v477
      local.get $v477
      local.get $v5
      f64.min
      local.set $v478
      local.get $v472
      local.get $v475
      f64.add
      local.set $v479
      local.get $v478
      local.get $v90
      f64.max
      local.set $v480
      local.get $v479
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v481
      local.get $v468
      local.get $v481
      f64.add
      local.set $v482
      local.get $v472
      local.get $v475
      f64.sub
      local.set $v483
      local.get $v483
      local.get $v7
      f64.add
      local.set $v484
      local.get $v484
      local.get $v5
      f64.min
      local.set $v485
      local.get $v485
      local.get $v91
      f64.max
      local.set $v486
      local.get $v486
      f64.const 2.0
      f64.mul
      local.set $v487
      local.get $v487
      local.get $v480
      f64.mul
      local.set $v488
      local.get $v488
      local.get $v4
      f64.add
      local.set $v489
      local.get $v489
      local.get $v5
      f64.min
      local.set $v490
      local.get $v490
      local.get $v92
      f64.max
      local.set $v491
      local.get $v486
      local.get $v486
      f64.mul
      local.set $v492
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v480
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v482
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v491
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v492
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render_chunk4__4
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $feed49 i32) (param $feed50 i32) (param $feed51 i32) (param $feed52 i32) (param $feed53 i32) (param $feed54 i32) (param $feed55 i32) (param $feed56 i32) (param $feed57 i32) (param $feed58 i32) (param $feed59 i32) (param $feed60 i32) (param $feed61 i32) (param $feed62 i32) (param $feed63 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (local $v459 f64)
    (local $v460 f64)
    (local $v461 f64)
    (local $v462 f64)
    (local $v463 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $feed49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v49
      local.get $feed50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v50
      local.get $feed51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v51
      local.get $feed52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v52
      local.get $feed53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v53
      local.get $feed54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v54
      local.get $feed55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v55
      local.get $feed56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v56
      local.get $feed57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v57
      local.get $feed58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v58
      local.get $feed59
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v59
      local.get $feed60
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v60
      local.get $feed61
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v61
      local.get $feed62
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v62
      local.get $feed63
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v63
      local.get $v0
      local.get $v0
      f64.mul
      local.set $v64
      local.get $v1
      local.get $v64
      f64.add
      local.set $v65
      local.get $v65
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v66
      local.get $v2
      local.get $v66
      f64.add
      local.set $v67
      local.get $v1
      local.get $v64
      f64.sub
      local.set $v68
      local.get $v68
      local.get $v3
      f64.add
      local.set $v69
      local.get $v69
      local.get $v4
      f64.min
      local.set $v70
      local.get $v70
      local.get $v5
      f64.max
      local.set $v71
      local.get $v71
      local.get $v71
      f64.mul
      local.set $v72
      local.get $v71
      f64.const 2.0
      f64.mul
      local.set $v73
      local.get $v6
      local.get $v6
      f64.mul
      local.set $v74
      local.get $v72
      local.get $v74
      f64.sub
      local.set $v75
      local.get $v72
      local.get $v74
      f64.add
      local.set $v76
      local.get $v76
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v77
      local.get $v75
      local.get $v3
      f64.add
      local.set $v78
      local.get $v73
      local.get $v6
      f64.mul
      local.set $v79
      local.get $v79
      local.get $v7
      f64.add
      local.set $v80
      local.get $v80
      local.get $v4
      f64.min
      local.set $v81
      local.get $v81
      local.get $v8
      f64.max
      local.set $v82
      local.get $v78
      local.get $v4
      f64.min
      local.set $v83
      local.get $v83
      local.get $v9
      f64.max
      local.set $v84
      local.get $v84
      local.get $v84
      f64.mul
      local.set $v85
      local.get $v84
      f64.const 2.0
      f64.mul
      local.set $v86
      local.get $v67
      local.get $v77
      f64.add
      local.set $v87
      local.get $v82
      local.get $v82
      f64.mul
      local.set $v88
      local.get $v85
      local.get $v88
      f64.sub
      local.set $v89
      local.get $v85
      local.get $v88
      f64.add
      local.set $v90
      local.get $v90
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v91
      local.get $v86
      local.get $v82
      f64.mul
      local.set $v92
      local.get $v92
      local.get $v7
      f64.add
      local.set $v93
      local.get $v89
      local.get $v3
      f64.add
      local.set $v94
      local.get $v87
      local.get $v91
      f64.add
      local.set $v95
      local.get $v93
      local.get $v4
      f64.min
      local.set $v96
      local.get $v96
      local.get $v10
      f64.max
      local.set $v97
      local.get $v97
      local.get $v97
      f64.mul
      local.set $v98
      local.get $v94
      local.get $v4
      f64.min
      local.set $v99
      local.get $v99
      local.get $v11
      f64.max
      local.set $v100
      local.get $v100
      f64.const 2.0
      f64.mul
      local.set $v101
      local.get $v100
      local.get $v100
      f64.mul
      local.set $v102
      local.get $v102
      local.get $v98
      f64.sub
      local.set $v103
      local.get $v103
      local.get $v3
      f64.add
      local.set $v104
      local.get $v102
      local.get $v98
      f64.add
      local.set $v105
      local.get $v105
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v106
      local.get $v95
      local.get $v106
      f64.add
      local.set $v107
      local.get $v101
      local.get $v97
      f64.mul
      local.set $v108
      local.get $v108
      local.get $v7
      f64.add
      local.set $v109
      local.get $v109
      local.get $v4
      f64.min
      local.set $v110
      local.get $v110
      local.get $v12
      f64.max
      local.set $v111
      local.get $v111
      local.get $v111
      f64.mul
      local.set $v112
      local.get $v104
      local.get $v4
      f64.min
      local.set $v113
      local.get $v113
      local.get $v13
      f64.max
      local.set $v114
      local.get $v114
      local.get $v114
      f64.mul
      local.set $v115
      local.get $v114
      f64.const 2.0
      f64.mul
      local.set $v116
      local.get $v116
      local.get $v111
      f64.mul
      local.set $v117
      local.get $v115
      local.get $v112
      f64.sub
      local.set $v118
      local.get $v118
      local.get $v3
      f64.add
      local.set $v119
      local.get $v119
      local.get $v4
      f64.min
      local.set $v120
      local.get $v120
      local.get $v14
      f64.max
      local.set $v121
      local.get $v121
      local.get $v121
      f64.mul
      local.set $v122
      local.get $v121
      f64.const 2.0
      f64.mul
      local.set $v123
      local.get $v115
      local.get $v112
      f64.add
      local.set $v124
      local.get $v124
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v125
      local.get $v107
      local.get $v125
      f64.add
      local.set $v126
      local.get $v117
      local.get $v7
      f64.add
      local.set $v127
      local.get $v127
      local.get $v4
      f64.min
      local.set $v128
      local.get $v128
      local.get $v15
      f64.max
      local.set $v129
      local.get $v129
      local.get $v129
      f64.mul
      local.set $v130
      local.get $v123
      local.get $v129
      f64.mul
      local.set $v131
      local.get $v131
      local.get $v7
      f64.add
      local.set $v132
      local.get $v132
      local.get $v4
      f64.min
      local.set $v133
      local.get $v133
      local.get $v16
      f64.max
      local.set $v134
      local.get $v134
      local.get $v134
      f64.mul
      local.set $v135
      local.get $v122
      local.get $v130
      f64.sub
      local.set $v136
      local.get $v136
      local.get $v3
      f64.add
      local.set $v137
      local.get $v137
      local.get $v4
      f64.min
      local.set $v138
      local.get $v138
      local.get $v17
      f64.max
      local.set $v139
      local.get $v139
      local.get $v139
      f64.mul
      local.set $v140
      local.get $v140
      local.get $v135
      f64.add
      local.set $v141
      local.get $v140
      local.get $v135
      f64.sub
      local.set $v142
      local.get $v142
      local.get $v3
      f64.add
      local.set $v143
      local.get $v143
      local.get $v4
      f64.min
      local.set $v144
      local.get $v144
      local.get $v18
      f64.max
      local.set $v145
      local.get $v145
      local.get $v145
      f64.mul
      local.set $v146
      local.get $v139
      f64.const 2.0
      f64.mul
      local.set $v147
      local.get $v147
      local.get $v134
      f64.mul
      local.set $v148
      local.get $v148
      local.get $v7
      f64.add
      local.set $v149
      local.get $v149
      local.get $v4
      f64.min
      local.set $v150
      local.get $v150
      local.get $v19
      f64.max
      local.set $v151
      local.get $v141
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v152
      local.get $v151
      local.get $v151
      f64.mul
      local.set $v153
      local.get $v146
      local.get $v153
      f64.add
      local.set $v154
      local.get $v146
      local.get $v153
      f64.sub
      local.set $v155
      local.get $v154
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v156
      local.get $v155
      local.get $v3
      f64.add
      local.set $v157
      local.get $v122
      local.get $v130
      f64.add
      local.set $v158
      local.get $v158
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v159
      local.get $v157
      local.get $v4
      f64.min
      local.set $v160
      local.get $v160
      local.get $v20
      f64.max
      local.set $v161
      local.get $v161
      f64.const 2.0
      f64.mul
      local.set $v162
      local.get $v161
      local.get $v161
      f64.mul
      local.set $v163
      local.get $v126
      local.get $v159
      f64.add
      local.set $v164
      local.get $v164
      local.get $v152
      f64.add
      local.set $v165
      local.get $v165
      local.get $v156
      f64.add
      local.set $v166
      local.get $v145
      f64.const 2.0
      f64.mul
      local.set $v167
      local.get $v167
      local.get $v151
      f64.mul
      local.set $v168
      local.get $v168
      local.get $v7
      f64.add
      local.set $v169
      local.get $v169
      local.get $v4
      f64.min
      local.set $v170
      local.get $v170
      local.get $v21
      f64.max
      local.set $v171
      local.get $v171
      local.get $v171
      f64.mul
      local.set $v172
      local.get $v163
      local.get $v172
      f64.sub
      local.set $v173
      local.get $v173
      local.get $v3
      f64.add
      local.set $v174
      local.get $v163
      local.get $v172
      f64.add
      local.set $v175
      local.get $v162
      local.get $v171
      f64.mul
      local.set $v176
      local.get $v176
      local.get $v7
      f64.add
      local.set $v177
      local.get $v177
      local.get $v4
      f64.min
      local.set $v178
      local.get $v178
      local.get $v22
      f64.max
      local.set $v179
      local.get $v179
      local.get $v179
      f64.mul
      local.set $v180
      local.get $v175
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v181
      local.get $v166
      local.get $v181
      f64.add
      local.set $v182
      local.get $v174
      local.get $v4
      f64.min
      local.set $v183
      local.get $v183
      local.get $v23
      f64.max
      local.set $v184
      local.get $v184
      local.get $v184
      f64.mul
      local.set $v185
      local.get $v185
      local.get $v180
      f64.sub
      local.set $v186
      local.get $v185
      local.get $v180
      f64.add
      local.set $v187
      local.get $v187
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v188
      local.get $v186
      local.get $v3
      f64.add
      local.set $v189
      local.get $v182
      local.get $v188
      f64.add
      local.set $v190
      local.get $v184
      f64.const 2.0
      f64.mul
      local.set $v191
      local.get $v191
      local.get $v179
      f64.mul
      local.set $v192
      local.get $v192
      local.get $v7
      f64.add
      local.set $v193
      local.get $v193
      local.get $v4
      f64.min
      local.set $v194
      local.get $v189
      local.get $v4
      f64.min
      local.set $v195
      local.get $v195
      local.get $v24
      f64.max
      local.set $v196
      local.get $v196
      f64.const 2.0
      f64.mul
      local.set $v197
      local.get $v196
      local.get $v196
      f64.mul
      local.set $v198
      local.get $v194
      local.get $v25
      f64.max
      local.set $v199
      local.get $v199
      local.get $v199
      f64.mul
      local.set $v200
      local.get $v198
      local.get $v200
      f64.add
      local.set $v201
      local.get $v201
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v202
      local.get $v198
      local.get $v200
      f64.sub
      local.set $v203
      local.get $v203
      local.get $v3
      f64.add
      local.set $v204
      local.get $v204
      local.get $v4
      f64.min
      local.set $v205
      local.get $v190
      local.get $v202
      f64.add
      local.set $v206
      local.get $v205
      local.get $v26
      f64.max
      local.set $v207
      local.get $v207
      local.get $v207
      f64.mul
      local.set $v208
      local.get $v207
      f64.const 2.0
      f64.mul
      local.set $v209
      local.get $v197
      local.get $v199
      f64.mul
      local.set $v210
      local.get $v210
      local.get $v7
      f64.add
      local.set $v211
      local.get $v211
      local.get $v4
      f64.min
      local.set $v212
      local.get $v212
      local.get $v27
      f64.max
      local.set $v213
      local.get $v209
      local.get $v213
      f64.mul
      local.set $v214
      local.get $v213
      local.get $v213
      f64.mul
      local.set $v215
      local.get $v208
      local.get $v215
      f64.sub
      local.set $v216
      local.get $v208
      local.get $v215
      f64.add
      local.set $v217
      local.get $v217
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v218
      local.get $v216
      local.get $v3
      f64.add
      local.set $v219
      local.get $v206
      local.get $v218
      f64.add
      local.set $v220
      local.get $v214
      local.get $v7
      f64.add
      local.set $v221
      local.get $v219
      local.get $v4
      f64.min
      local.set $v222
      local.get $v222
      local.get $v28
      f64.max
      local.set $v223
      local.get $v223
      local.get $v223
      f64.mul
      local.set $v224
      local.get $v223
      f64.const 2.0
      f64.mul
      local.set $v225
      local.get $v221
      local.get $v4
      f64.min
      local.set $v226
      local.get $v226
      local.get $v29
      f64.max
      local.set $v227
      local.get $v225
      local.get $v227
      f64.mul
      local.set $v228
      local.get $v228
      local.get $v7
      f64.add
      local.set $v229
      local.get $v229
      local.get $v4
      f64.min
      local.set $v230
      local.get $v227
      local.get $v227
      f64.mul
      local.set $v231
      local.get $v224
      local.get $v231
      f64.add
      local.set $v232
      local.get $v224
      local.get $v231
      f64.sub
      local.set $v233
      local.get $v233
      local.get $v3
      f64.add
      local.set $v234
      local.get $v232
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v235
      local.get $v220
      local.get $v235
      f64.add
      local.set $v236
      local.get $v234
      local.get $v4
      f64.min
      local.set $v237
      local.get $v237
      local.get $v30
      f64.max
      local.set $v238
      local.get $v238
      local.get $v238
      f64.mul
      local.set $v239
      local.get $v238
      f64.const 2.0
      f64.mul
      local.set $v240
      local.get $v230
      local.get $v31
      f64.max
      local.set $v241
      local.get $v240
      local.get $v241
      f64.mul
      local.set $v242
      local.get $v241
      local.get $v241
      f64.mul
      local.set $v243
      local.get $v239
      local.get $v243
      f64.add
      local.set $v244
      local.get $v244
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v245
      local.get $v239
      local.get $v243
      f64.sub
      local.set $v246
      local.get $v236
      local.get $v245
      f64.add
      local.set $v247
      local.get $v246
      local.get $v3
      f64.add
      local.set $v248
      local.get $v248
      local.get $v4
      f64.min
      local.set $v249
      local.get $v242
      local.get $v7
      f64.add
      local.set $v250
      local.get $v250
      local.get $v4
      f64.min
      local.set $v251
      local.get $v251
      local.get $v32
      f64.max
      local.set $v252
      local.get $v252
      local.get $v252
      f64.mul
      local.set $v253
      local.get $v249
      local.get $v33
      f64.max
      local.set $v254
      local.get $v254
      local.get $v254
      f64.mul
      local.set $v255
      local.get $v255
      local.get $v253
      f64.add
      local.set $v256
      local.get $v255
      local.get $v253
      f64.sub
      local.set $v257
      local.get $v254
      f64.const 2.0
      f64.mul
      local.set $v258
      local.get $v258
      local.get $v252
      f64.mul
      local.set $v259
      local.get $v259
      local.get $v7
      f64.add
      local.set $v260
      local.get $v260
      local.get $v4
      f64.min
      local.set $v261
      local.get $v257
      local.get $v3
      f64.add
      local.set $v262
      local.get $v262
      local.get $v4
      f64.min
      local.set $v263
      local.get $v263
      local.get $v34
      f64.max
      local.set $v264
      local.get $v264
      f64.const 2.0
      f64.mul
      local.set $v265
      local.get $v264
      local.get $v264
      f64.mul
      local.set $v266
      local.get $v256
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v267
      local.get $v247
      local.get $v267
      f64.add
      local.set $v268
      local.get $v261
      local.get $v35
      f64.max
      local.set $v269
      local.get $v265
      local.get $v269
      f64.mul
      local.set $v270
      local.get $v270
      local.get $v7
      f64.add
      local.set $v271
      local.get $v269
      local.get $v269
      f64.mul
      local.set $v272
      local.get $v266
      local.get $v272
      f64.add
      local.set $v273
      local.get $v273
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v274
      local.get $v268
      local.get $v274
      f64.add
      local.set $v275
      local.get $v266
      local.get $v272
      f64.sub
      local.set $v276
      local.get $v276
      local.get $v3
      f64.add
      local.set $v277
      local.get $v277
      local.get $v4
      f64.min
      local.set $v278
      local.get $v278
      local.get $v36
      f64.max
      local.set $v279
      local.get $v279
      f64.const 2.0
      f64.mul
      local.set $v280
      local.get $v279
      local.get $v279
      f64.mul
      local.set $v281
      local.get $v271
      local.get $v4
      f64.min
      local.set $v282
      local.get $v282
      local.get $v37
      f64.max
      local.set $v283
      local.get $v283
      local.get $v283
      f64.mul
      local.set $v284
      local.get $v280
      local.get $v283
      f64.mul
      local.set $v285
      local.get $v285
      local.get $v7
      f64.add
      local.set $v286
      local.get $v286
      local.get $v4
      f64.min
      local.set $v287
      local.get $v281
      local.get $v284
      f64.add
      local.set $v288
      local.get $v288
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v289
      local.get $v275
      local.get $v289
      f64.add
      local.set $v290
      local.get $v287
      local.get $v38
      f64.max
      local.set $v291
      local.get $v291
      local.get $v291
      f64.mul
      local.set $v292
      local.get $v281
      local.get $v284
      f64.sub
      local.set $v293
      local.get $v293
      local.get $v3
      f64.add
      local.set $v294
      local.get $v294
      local.get $v4
      f64.min
      local.set $v295
      local.get $v295
      local.get $v39
      f64.max
      local.set $v296
      local.get $v296
      local.get $v296
      f64.mul
      local.set $v297
      local.get $v297
      local.get $v292
      f64.add
      local.set $v298
      local.get $v298
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v299
      local.get $v290
      local.get $v299
      f64.add
      local.set $v300
      local.get $v297
      local.get $v292
      f64.sub
      local.set $v301
      local.get $v296
      f64.const 2.0
      f64.mul
      local.set $v302
      local.get $v302
      local.get $v291
      f64.mul
      local.set $v303
      local.get $v303
      local.get $v7
      f64.add
      local.set $v304
      local.get $v304
      local.get $v4
      f64.min
      local.set $v305
      local.get $v305
      local.get $v40
      f64.max
      local.set $v306
      local.get $v306
      local.get $v306
      f64.mul
      local.set $v307
      local.get $v301
      local.get $v3
      f64.add
      local.set $v308
      local.get $v308
      local.get $v4
      f64.min
      local.set $v309
      local.get $v309
      local.get $v41
      f64.max
      local.set $v310
      local.get $v310
      local.get $v310
      f64.mul
      local.set $v311
      local.get $v311
      local.get $v307
      f64.add
      local.set $v312
      local.get $v312
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v313
      local.get $v300
      local.get $v313
      f64.add
      local.set $v314
      local.get $v311
      local.get $v307
      f64.sub
      local.set $v315
      local.get $v315
      local.get $v3
      f64.add
      local.set $v316
      local.get $v316
      local.get $v4
      f64.min
      local.set $v317
      local.get $v317
      local.get $v42
      f64.max
      local.set $v318
      local.get $v318
      local.get $v318
      f64.mul
      local.set $v319
      local.get $v318
      f64.const 2.0
      f64.mul
      local.set $v320
      local.get $v310
      f64.const 2.0
      f64.mul
      local.set $v321
      local.get $v321
      local.get $v306
      f64.mul
      local.set $v322
      local.get $v322
      local.get $v7
      f64.add
      local.set $v323
      local.get $v323
      local.get $v4
      f64.min
      local.set $v324
      local.get $v324
      local.get $v43
      f64.max
      local.set $v325
      local.get $v320
      local.get $v325
      f64.mul
      local.set $v326
      local.get $v325
      local.get $v325
      f64.mul
      local.set $v327
      local.get $v319
      local.get $v327
      f64.add
      local.set $v328
      local.get $v328
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v329
      local.get $v314
      local.get $v329
      f64.add
      local.set $v330
      local.get $v326
      local.get $v7
      f64.add
      local.set $v331
      local.get $v331
      local.get $v4
      f64.min
      local.set $v332
      local.get $v319
      local.get $v327
      f64.sub
      local.set $v333
      local.get $v333
      local.get $v3
      f64.add
      local.set $v334
      local.get $v334
      local.get $v4
      f64.min
      local.set $v335
      local.get $v335
      local.get $v44
      f64.max
      local.set $v336
      local.get $v336
      local.get $v336
      f64.mul
      local.set $v337
      local.get $v336
      f64.const 2.0
      f64.mul
      local.set $v338
      local.get $v332
      local.get $v45
      f64.max
      local.set $v339
      local.get $v339
      local.get $v339
      f64.mul
      local.set $v340
      local.get $v337
      local.get $v340
      f64.add
      local.set $v341
      local.get $v341
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v342
      local.get $v338
      local.get $v339
      f64.mul
      local.set $v343
      local.get $v343
      local.get $v7
      f64.add
      local.set $v344
      local.get $v344
      local.get $v4
      f64.min
      local.set $v345
      local.get $v345
      local.get $v46
      f64.max
      local.set $v346
      local.get $v346
      local.get $v346
      f64.mul
      local.set $v347
      local.get $v337
      local.get $v340
      f64.sub
      local.set $v348
      local.get $v330
      local.get $v342
      f64.add
      local.set $v349
      local.get $v348
      local.get $v3
      f64.add
      local.set $v350
      local.get $v350
      local.get $v4
      f64.min
      local.set $v351
      local.get $v351
      local.get $v47
      f64.max
      local.set $v352
      local.get $v352
      f64.const 2.0
      f64.mul
      local.set $v353
      local.get $v353
      local.get $v346
      f64.mul
      local.set $v354
      local.get $v354
      local.get $v7
      f64.add
      local.set $v355
      local.get $v352
      local.get $v352
      f64.mul
      local.set $v356
      local.get $v356
      local.get $v347
      f64.add
      local.set $v357
      local.get $v357
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v358
      local.get $v349
      local.get $v358
      f64.add
      local.set $v359
      local.get $v356
      local.get $v347
      f64.sub
      local.set $v360
      local.get $v360
      local.get $v3
      f64.add
      local.set $v361
      local.get $v355
      local.get $v4
      f64.min
      local.set $v362
      local.get $v362
      local.get $v48
      f64.max
      local.set $v363
      local.get $v363
      local.get $v363
      f64.mul
      local.set $v364
      local.get $v361
      local.get $v4
      f64.min
      local.set $v365
      local.get $v365
      local.get $v49
      f64.max
      local.set $v366
      local.get $v366
      f64.const 2.0
      f64.mul
      local.set $v367
      local.get $v367
      local.get $v363
      f64.mul
      local.set $v368
      local.get $v368
      local.get $v7
      f64.add
      local.set $v369
      local.get $v369
      local.get $v4
      f64.min
      local.set $v370
      local.get $v366
      local.get $v366
      f64.mul
      local.set $v371
      local.get $v371
      local.get $v364
      f64.add
      local.set $v372
      local.get $v371
      local.get $v364
      f64.sub
      local.set $v373
      local.get $v373
      local.get $v3
      f64.add
      local.set $v374
      local.get $v372
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v375
      local.get $v370
      local.get $v50
      f64.max
      local.set $v376
      local.get $v374
      local.get $v4
      f64.min
      local.set $v377
      local.get $v377
      local.get $v51
      f64.max
      local.set $v378
      local.get $v378
      local.get $v378
      f64.mul
      local.set $v379
      local.get $v359
      local.get $v375
      f64.add
      local.set $v380
      local.get $v376
      local.get $v376
      f64.mul
      local.set $v381
      local.get $v379
      local.get $v381
      f64.add
      local.set $v382
      local.get $v379
      local.get $v381
      f64.sub
      local.set $v383
      local.get $v378
      f64.const 2.0
      f64.mul
      local.set $v384
      local.get $v384
      local.get $v376
      f64.mul
      local.set $v385
      local.get $v385
      local.get $v7
      f64.add
      local.set $v386
      local.get $v383
      local.get $v3
      f64.add
      local.set $v387
      local.get $v386
      local.get $v4
      f64.min
      local.set $v388
      local.get $v388
      local.get $v52
      f64.max
      local.set $v389
      local.get $v389
      local.get $v389
      f64.mul
      local.set $v390
      local.get $v382
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v391
      local.get $v380
      local.get $v391
      f64.add
      local.set $v392
      local.get $v387
      local.get $v4
      f64.min
      local.set $v393
      local.get $v393
      local.get $v53
      f64.max
      local.set $v394
      local.get $v394
      local.get $v394
      f64.mul
      local.set $v395
      local.get $v394
      f64.const 2.0
      f64.mul
      local.set $v396
      local.get $v396
      local.get $v389
      f64.mul
      local.set $v397
      local.get $v397
      local.get $v7
      f64.add
      local.set $v398
      local.get $v398
      local.get $v4
      f64.min
      local.set $v399
      local.get $v399
      local.get $v54
      f64.max
      local.set $v400
      local.get $v400
      local.get $v400
      f64.mul
      local.set $v401
      local.get $v395
      local.get $v390
      f64.sub
      local.set $v402
      local.get $v402
      local.get $v3
      f64.add
      local.set $v403
      local.get $v395
      local.get $v390
      f64.add
      local.set $v404
      local.get $v404
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v405
      local.get $v403
      local.get $v4
      f64.min
      local.set $v406
      local.get $v406
      local.get $v55
      f64.max
      local.set $v407
      local.get $v407
      local.get $v407
      f64.mul
      local.set $v408
      local.get $v408
      local.get $v401
      f64.sub
      local.set $v409
      local.get $v392
      local.get $v405
      f64.add
      local.set $v410
      local.get $v408
      local.get $v401
      f64.add
      local.set $v411
      local.get $v407
      f64.const 2.0
      f64.mul
      local.set $v412
      local.get $v412
      local.get $v400
      f64.mul
      local.set $v413
      local.get $v411
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v414
      local.get $v413
      local.get $v7
      f64.add
      local.set $v415
      local.get $v415
      local.get $v4
      f64.min
      local.set $v416
      local.get $v416
      local.get $v56
      f64.max
      local.set $v417
      local.get $v417
      local.get $v417
      f64.mul
      local.set $v418
      local.get $v409
      local.get $v3
      f64.add
      local.set $v419
      local.get $v419
      local.get $v4
      f64.min
      local.set $v420
      local.get $v420
      local.get $v57
      f64.max
      local.set $v421
      local.get $v421
      local.get $v421
      f64.mul
      local.set $v422
      local.get $v422
      local.get $v418
      f64.sub
      local.set $v423
      local.get $v423
      local.get $v3
      f64.add
      local.set $v424
      local.get $v410
      local.get $v414
      f64.add
      local.set $v425
      local.get $v421
      f64.const 2.0
      f64.mul
      local.set $v426
      local.get $v426
      local.get $v417
      f64.mul
      local.set $v427
      local.get $v424
      local.get $v4
      f64.min
      local.set $v428
      local.get $v428
      local.get $v58
      f64.max
      local.set $v429
      local.get $v422
      local.get $v418
      f64.add
      local.set $v430
      local.get $v430
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v431
      local.get $v425
      local.get $v431
      f64.add
      local.set $v432
      local.get $v427
      local.get $v7
      f64.add
      local.set $v433
      local.get $v433
      local.get $v4
      f64.min
      local.set $v434
      local.get $v434
      local.get $v59
      f64.max
      local.set $v435
      local.get $v429
      local.get $v429
      f64.mul
      local.set $v436
      local.get $v435
      local.get $v435
      f64.mul
      local.set $v437
      local.get $v436
      local.get $v437
      f64.sub
      local.set $v438
      local.get $v438
      local.get $v3
      f64.add
      local.set $v439
      local.get $v439
      local.get $v4
      f64.min
      local.set $v440
      local.get $v429
      f64.const 2.0
      f64.mul
      local.set $v441
      local.get $v441
      local.get $v435
      f64.mul
      local.set $v442
      local.get $v442
      local.get $v7
      f64.add
      local.set $v443
      local.get $v440
      local.get $v60
      f64.max
      local.set $v444
      local.get $v444
      local.get $v444
      f64.mul
      local.set $v445
      local.get $v444
      f64.const 2.0
      f64.mul
      local.set $v446
      local.get $v443
      local.get $v4
      f64.min
      local.set $v447
      local.get $v447
      local.get $v61
      f64.max
      local.set $v448
      local.get $v448
      local.get $v448
      f64.mul
      local.set $v449
      local.get $v436
      local.get $v437
      f64.add
      local.set $v450
      local.get $v445
      local.get $v449
      f64.sub
      local.set $v451
      local.get $v450
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v452
      local.get $v432
      local.get $v452
      f64.add
      local.set $v453
      local.get $v451
      local.get $v3
      f64.add
      local.set $v454
      local.get $v446
      local.get $v448
      f64.mul
      local.set $v455
      local.get $v455
      local.get $v7
      f64.add
      local.set $v456
      local.get $v456
      local.get $v4
      f64.min
      local.set $v457
      local.get $v457
      local.get $v62
      f64.max
      local.set $v458
      local.get $v454
      local.get $v4
      f64.min
      local.set $v459
      local.get $v459
      local.get $v63
      f64.max
      local.set $v460
      local.get $v445
      local.get $v449
      f64.add
      local.set $v461
      local.get $v461
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v462
      local.get $v453
      local.get $v462
      f64.add
      local.set $v463
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v458
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v460
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v463
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render_chunk5__5
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $feed49 i32) (param $feed50 i32) (param $feed51 i32) (param $feed52 i32) (param $feed53 i32) (param $feed54 i32) (param $feed55 i32) (param $feed56 i32) (param $feed57 i32) (param $feed58 i32) (param $feed59 i32) (param $feed60 i32) (param $feed61 i32) (param $feed62 i32) (param $feed63 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (local $v344 f64)
    (local $v345 f64)
    (local $v346 f64)
    (local $v347 f64)
    (local $v348 f64)
    (local $v349 f64)
    (local $v350 f64)
    (local $v351 f64)
    (local $v352 f64)
    (local $v353 f64)
    (local $v354 f64)
    (local $v355 f64)
    (local $v356 f64)
    (local $v357 f64)
    (local $v358 f64)
    (local $v359 f64)
    (local $v360 f64)
    (local $v361 f64)
    (local $v362 f64)
    (local $v363 f64)
    (local $v364 f64)
    (local $v365 f64)
    (local $v366 f64)
    (local $v367 f64)
    (local $v368 f64)
    (local $v369 f64)
    (local $v370 f64)
    (local $v371 f64)
    (local $v372 f64)
    (local $v373 f64)
    (local $v374 f64)
    (local $v375 f64)
    (local $v376 f64)
    (local $v377 f64)
    (local $v378 f64)
    (local $v379 f64)
    (local $v380 f64)
    (local $v381 f64)
    (local $v382 f64)
    (local $v383 f64)
    (local $v384 f64)
    (local $v385 f64)
    (local $v386 f64)
    (local $v387 f64)
    (local $v388 f64)
    (local $v389 f64)
    (local $v390 f64)
    (local $v391 f64)
    (local $v392 f64)
    (local $v393 f64)
    (local $v394 f64)
    (local $v395 f64)
    (local $v396 f64)
    (local $v397 f64)
    (local $v398 f64)
    (local $v399 f64)
    (local $v400 f64)
    (local $v401 f64)
    (local $v402 f64)
    (local $v403 f64)
    (local $v404 f64)
    (local $v405 f64)
    (local $v406 f64)
    (local $v407 f64)
    (local $v408 f64)
    (local $v409 f64)
    (local $v410 f64)
    (local $v411 f64)
    (local $v412 f64)
    (local $v413 f64)
    (local $v414 f64)
    (local $v415 f64)
    (local $v416 f64)
    (local $v417 f64)
    (local $v418 f64)
    (local $v419 f64)
    (local $v420 f64)
    (local $v421 f64)
    (local $v422 f64)
    (local $v423 f64)
    (local $v424 f64)
    (local $v425 f64)
    (local $v426 f64)
    (local $v427 f64)
    (local $v428 f64)
    (local $v429 f64)
    (local $v430 f64)
    (local $v431 f64)
    (local $v432 f64)
    (local $v433 f64)
    (local $v434 f64)
    (local $v435 f64)
    (local $v436 f64)
    (local $v437 f64)
    (local $v438 f64)
    (local $v439 f64)
    (local $v440 f64)
    (local $v441 f64)
    (local $v442 f64)
    (local $v443 f64)
    (local $v444 f64)
    (local $v445 f64)
    (local $v446 f64)
    (local $v447 f64)
    (local $v448 f64)
    (local $v449 f64)
    (local $v450 f64)
    (local $v451 f64)
    (local $v452 f64)
    (local $v453 f64)
    (local $v454 f64)
    (local $v455 f64)
    (local $v456 f64)
    (local $v457 f64)
    (local $v458 f64)
    (local $v459 f64)
    (local $v460 f64)
    (local $v461 f64)
    (local $v462 f64)
    (local $v463 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $feed49
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v49
      local.get $feed50
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v50
      local.get $feed51
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v51
      local.get $feed52
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v52
      local.get $feed53
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v53
      local.get $feed54
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v54
      local.get $feed55
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v55
      local.get $feed56
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v56
      local.get $feed57
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v57
      local.get $feed58
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v58
      local.get $feed59
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v59
      local.get $feed60
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v60
      local.get $feed61
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v61
      local.get $feed62
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v62
      local.get $feed63
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v63
      local.get $v0
      local.get $v0
      f64.mul
      local.set $v64
      local.get $v1
      local.get $v1
      f64.mul
      local.set $v65
      local.get $v64
      local.get $v65
      f64.sub
      local.set $v66
      local.get $v64
      local.get $v65
      f64.add
      local.set $v67
      local.get $v0
      f64.const 2.0
      f64.mul
      local.set $v68
      local.get $v68
      local.get $v1
      f64.mul
      local.set $v69
      local.get $v69
      local.get $v2
      f64.add
      local.set $v70
      local.get $v70
      local.get $v3
      f64.min
      local.set $v71
      local.get $v71
      local.get $v4
      f64.max
      local.set $v72
      local.get $v72
      local.get $v72
      f64.mul
      local.set $v73
      local.get $v66
      local.get $v5
      f64.add
      local.set $v74
      local.get $v74
      local.get $v3
      f64.min
      local.set $v75
      local.get $v67
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v76
      local.get $v6
      local.get $v76
      f64.add
      local.set $v77
      local.get $v75
      local.get $v7
      f64.max
      local.set $v78
      local.get $v78
      local.get $v78
      f64.mul
      local.set $v79
      local.get $v79
      local.get $v73
      f64.add
      local.set $v80
      local.get $v80
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v81
      local.get $v78
      f64.const 2.0
      f64.mul
      local.set $v82
      local.get $v79
      local.get $v73
      f64.sub
      local.set $v83
      local.get $v83
      local.get $v5
      f64.add
      local.set $v84
      local.get $v82
      local.get $v72
      f64.mul
      local.set $v85
      local.get $v85
      local.get $v2
      f64.add
      local.set $v86
      local.get $v86
      local.get $v3
      f64.min
      local.set $v87
      local.get $v84
      local.get $v3
      f64.min
      local.set $v88
      local.get $v88
      local.get $v8
      f64.max
      local.set $v89
      local.get $v87
      local.get $v9
      f64.max
      local.set $v90
      local.get $v90
      local.get $v90
      f64.mul
      local.set $v91
      local.get $v77
      local.get $v81
      f64.add
      local.set $v92
      local.get $v89
      f64.const 2.0
      f64.mul
      local.set $v93
      local.get $v93
      local.get $v90
      f64.mul
      local.set $v94
      local.get $v94
      local.get $v2
      f64.add
      local.set $v95
      local.get $v95
      local.get $v3
      f64.min
      local.set $v96
      local.get $v96
      local.get $v10
      f64.max
      local.set $v97
      local.get $v97
      local.get $v97
      f64.mul
      local.set $v98
      local.get $v89
      local.get $v89
      f64.mul
      local.set $v99
      local.get $v99
      local.get $v91
      f64.add
      local.set $v100
      local.get $v100
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v101
      local.get $v92
      local.get $v101
      f64.add
      local.set $v102
      local.get $v99
      local.get $v91
      f64.sub
      local.set $v103
      local.get $v103
      local.get $v5
      f64.add
      local.set $v104
      local.get $v104
      local.get $v3
      f64.min
      local.set $v105
      local.get $v105
      local.get $v11
      f64.max
      local.set $v106
      local.get $v106
      f64.const 2.0
      f64.mul
      local.set $v107
      local.get $v107
      local.get $v97
      f64.mul
      local.set $v108
      local.get $v108
      local.get $v2
      f64.add
      local.set $v109
      local.get $v109
      local.get $v3
      f64.min
      local.set $v110
      local.get $v110
      local.get $v12
      f64.max
      local.set $v111
      local.get $v106
      local.get $v106
      f64.mul
      local.set $v112
      local.get $v112
      local.get $v98
      f64.add
      local.set $v113
      local.get $v113
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v114
      local.get $v112
      local.get $v98
      f64.sub
      local.set $v115
      local.get $v115
      local.get $v5
      f64.add
      local.set $v116
      local.get $v102
      local.get $v114
      f64.add
      local.set $v117
      local.get $v111
      local.get $v111
      f64.mul
      local.set $v118
      local.get $v116
      local.get $v3
      f64.min
      local.set $v119
      local.get $v119
      local.get $v13
      f64.max
      local.set $v120
      local.get $v120
      local.get $v120
      f64.mul
      local.set $v121
      local.get $v121
      local.get $v118
      f64.sub
      local.set $v122
      local.get $v120
      f64.const 2.0
      f64.mul
      local.set $v123
      local.get $v123
      local.get $v111
      f64.mul
      local.set $v124
      local.get $v124
      local.get $v2
      f64.add
      local.set $v125
      local.get $v125
      local.get $v3
      f64.min
      local.set $v126
      local.get $v122
      local.get $v5
      f64.add
      local.set $v127
      local.get $v127
      local.get $v3
      f64.min
      local.set $v128
      local.get $v128
      local.get $v14
      f64.max
      local.set $v129
      local.get $v129
      local.get $v129
      f64.mul
      local.set $v130
      local.get $v121
      local.get $v118
      f64.add
      local.set $v131
      local.get $v131
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v132
      local.get $v117
      local.get $v132
      f64.add
      local.set $v133
      local.get $v129
      f64.const 2.0
      f64.mul
      local.set $v134
      local.get $v126
      local.get $v15
      f64.max
      local.set $v135
      local.get $v135
      local.get $v135
      f64.mul
      local.set $v136
      local.get $v130
      local.get $v136
      f64.add
      local.set $v137
      local.get $v137
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v138
      local.get $v134
      local.get $v135
      f64.mul
      local.set $v139
      local.get $v139
      local.get $v2
      f64.add
      local.set $v140
      local.get $v140
      local.get $v3
      f64.min
      local.set $v141
      local.get $v141
      local.get $v16
      f64.max
      local.set $v142
      local.get $v142
      local.get $v142
      f64.mul
      local.set $v143
      local.get $v133
      local.get $v138
      f64.add
      local.set $v144
      local.get $v130
      local.get $v136
      f64.sub
      local.set $v145
      local.get $v145
      local.get $v5
      f64.add
      local.set $v146
      local.get $v146
      local.get $v3
      f64.min
      local.set $v147
      local.get $v147
      local.get $v17
      f64.max
      local.set $v148
      local.get $v148
      local.get $v148
      f64.mul
      local.set $v149
      local.get $v149
      local.get $v143
      f64.add
      local.set $v150
      local.get $v150
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v151
      local.get $v149
      local.get $v143
      f64.sub
      local.set $v152
      local.get $v148
      f64.const 2.0
      f64.mul
      local.set $v153
      local.get $v153
      local.get $v142
      f64.mul
      local.set $v154
      local.get $v152
      local.get $v5
      f64.add
      local.set $v155
      local.get $v155
      local.get $v3
      f64.min
      local.set $v156
      local.get $v156
      local.get $v18
      f64.max
      local.set $v157
      local.get $v157
      local.get $v157
      f64.mul
      local.set $v158
      local.get $v157
      f64.const 2.0
      f64.mul
      local.set $v159
      local.get $v144
      local.get $v151
      f64.add
      local.set $v160
      local.get $v154
      local.get $v2
      f64.add
      local.set $v161
      local.get $v161
      local.get $v3
      f64.min
      local.set $v162
      local.get $v162
      local.get $v19
      f64.max
      local.set $v163
      local.get $v159
      local.get $v163
      f64.mul
      local.set $v164
      local.get $v164
      local.get $v2
      f64.add
      local.set $v165
      local.get $v165
      local.get $v3
      f64.min
      local.set $v166
      local.get $v166
      local.get $v20
      f64.max
      local.set $v167
      local.get $v167
      local.get $v167
      f64.mul
      local.set $v168
      local.get $v163
      local.get $v163
      f64.mul
      local.set $v169
      local.get $v158
      local.get $v169
      f64.sub
      local.set $v170
      local.get $v158
      local.get $v169
      f64.add
      local.set $v171
      local.get $v171
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v172
      local.get $v160
      local.get $v172
      f64.add
      local.set $v173
      local.get $v170
      local.get $v5
      f64.add
      local.set $v174
      local.get $v174
      local.get $v3
      f64.min
      local.set $v175
      local.get $v175
      local.get $v21
      f64.max
      local.set $v176
      local.get $v176
      f64.const 2.0
      f64.mul
      local.set $v177
      local.get $v176
      local.get $v176
      f64.mul
      local.set $v178
      local.get $v178
      local.get $v168
      f64.sub
      local.set $v179
      local.get $v178
      local.get $v168
      f64.add
      local.set $v180
      local.get $v180
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v181
      local.get $v173
      local.get $v181
      f64.add
      local.set $v182
      local.get $v177
      local.get $v167
      f64.mul
      local.set $v183
      local.get $v183
      local.get $v2
      f64.add
      local.set $v184
      local.get $v184
      local.get $v3
      f64.min
      local.set $v185
      local.get $v185
      local.get $v22
      f64.max
      local.set $v186
      local.get $v186
      local.get $v186
      f64.mul
      local.set $v187
      local.get $v179
      local.get $v5
      f64.add
      local.set $v188
      local.get $v188
      local.get $v3
      f64.min
      local.set $v189
      local.get $v189
      local.get $v23
      f64.max
      local.set $v190
      local.get $v190
      local.get $v190
      f64.mul
      local.set $v191
      local.get $v191
      local.get $v187
      f64.sub
      local.set $v192
      local.get $v192
      local.get $v5
      f64.add
      local.set $v193
      local.get $v193
      local.get $v3
      f64.min
      local.set $v194
      local.get $v194
      local.get $v24
      f64.max
      local.set $v195
      local.get $v195
      f64.const 2.0
      f64.mul
      local.set $v196
      local.get $v195
      local.get $v195
      f64.mul
      local.set $v197
      local.get $v191
      local.get $v187
      f64.add
      local.set $v198
      local.get $v198
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v199
      local.get $v190
      f64.const 2.0
      f64.mul
      local.set $v200
      local.get $v200
      local.get $v186
      f64.mul
      local.set $v201
      local.get $v201
      local.get $v2
      f64.add
      local.set $v202
      local.get $v182
      local.get $v199
      f64.add
      local.set $v203
      local.get $v202
      local.get $v3
      f64.min
      local.set $v204
      local.get $v204
      local.get $v25
      f64.max
      local.set $v205
      local.get $v196
      local.get $v205
      f64.mul
      local.set $v206
      local.get $v206
      local.get $v2
      f64.add
      local.set $v207
      local.get $v205
      local.get $v205
      f64.mul
      local.set $v208
      local.get $v197
      local.get $v208
      f64.sub
      local.set $v209
      local.get $v209
      local.get $v5
      f64.add
      local.set $v210
      local.get $v210
      local.get $v3
      f64.min
      local.set $v211
      local.get $v197
      local.get $v208
      f64.add
      local.set $v212
      local.get $v207
      local.get $v3
      f64.min
      local.set $v213
      local.get $v212
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v214
      local.get $v203
      local.get $v214
      f64.add
      local.set $v215
      local.get $v211
      local.get $v26
      f64.max
      local.set $v216
      local.get $v216
      local.get $v216
      f64.mul
      local.set $v217
      local.get $v216
      f64.const 2.0
      f64.mul
      local.set $v218
      local.get $v213
      local.get $v27
      f64.max
      local.set $v219
      local.get $v218
      local.get $v219
      f64.mul
      local.set $v220
      local.get $v219
      local.get $v219
      f64.mul
      local.set $v221
      local.get $v217
      local.get $v221
      f64.sub
      local.set $v222
      local.get $v217
      local.get $v221
      f64.add
      local.set $v223
      local.get $v220
      local.get $v2
      f64.add
      local.set $v224
      local.get $v224
      local.get $v3
      f64.min
      local.set $v225
      local.get $v225
      local.get $v28
      f64.max
      local.set $v226
      local.get $v226
      local.get $v226
      f64.mul
      local.set $v227
      local.get $v223
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v228
      local.get $v215
      local.get $v228
      f64.add
      local.set $v229
      local.get $v222
      local.get $v5
      f64.add
      local.set $v230
      local.get $v230
      local.get $v3
      f64.min
      local.set $v231
      local.get $v231
      local.get $v29
      f64.max
      local.set $v232
      local.get $v232
      f64.const 2.0
      f64.mul
      local.set $v233
      local.get $v232
      local.get $v232
      f64.mul
      local.set $v234
      local.get $v234
      local.get $v227
      f64.sub
      local.set $v235
      local.get $v234
      local.get $v227
      f64.add
      local.set $v236
      local.get $v235
      local.get $v5
      f64.add
      local.set $v237
      local.get $v237
      local.get $v3
      f64.min
      local.set $v238
      local.get $v238
      local.get $v30
      f64.max
      local.set $v239
      local.get $v239
      local.get $v239
      f64.mul
      local.set $v240
      local.get $v239
      f64.const 2.0
      f64.mul
      local.set $v241
      local.get $v236
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v242
      local.get $v229
      local.get $v242
      f64.add
      local.set $v243
      local.get $v233
      local.get $v226
      f64.mul
      local.set $v244
      local.get $v244
      local.get $v2
      f64.add
      local.set $v245
      local.get $v245
      local.get $v3
      f64.min
      local.set $v246
      local.get $v246
      local.get $v31
      f64.max
      local.set $v247
      local.get $v247
      local.get $v247
      f64.mul
      local.set $v248
      local.get $v240
      local.get $v248
      f64.sub
      local.set $v249
      local.get $v241
      local.get $v247
      f64.mul
      local.set $v250
      local.get $v249
      local.get $v5
      f64.add
      local.set $v251
      local.get $v240
      local.get $v248
      f64.add
      local.set $v252
      local.get $v252
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v253
      local.get $v250
      local.get $v2
      f64.add
      local.set $v254
      local.get $v254
      local.get $v3
      f64.min
      local.set $v255
      local.get $v255
      local.get $v32
      f64.max
      local.set $v256
      local.get $v256
      local.get $v256
      f64.mul
      local.set $v257
      local.get $v243
      local.get $v253
      f64.add
      local.set $v258
      local.get $v251
      local.get $v3
      f64.min
      local.set $v259
      local.get $v259
      local.get $v33
      f64.max
      local.set $v260
      local.get $v260
      f64.const 2.0
      f64.mul
      local.set $v261
      local.get $v261
      local.get $v256
      f64.mul
      local.set $v262
      local.get $v262
      local.get $v2
      f64.add
      local.set $v263
      local.get $v260
      local.get $v260
      f64.mul
      local.set $v264
      local.get $v264
      local.get $v257
      f64.add
      local.set $v265
      local.get $v265
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v266
      local.get $v264
      local.get $v257
      f64.sub
      local.set $v267
      local.get $v267
      local.get $v5
      f64.add
      local.set $v268
      local.get $v268
      local.get $v3
      f64.min
      local.set $v269
      local.get $v269
      local.get $v34
      f64.max
      local.set $v270
      local.get $v270
      f64.const 2.0
      f64.mul
      local.set $v271
      local.get $v263
      local.get $v3
      f64.min
      local.set $v272
      local.get $v272
      local.get $v35
      f64.max
      local.set $v273
      local.get $v273
      local.get $v273
      f64.mul
      local.set $v274
      local.get $v271
      local.get $v273
      f64.mul
      local.set $v275
      local.get $v258
      local.get $v266
      f64.add
      local.set $v276
      local.get $v275
      local.get $v2
      f64.add
      local.set $v277
      local.get $v277
      local.get $v3
      f64.min
      local.set $v278
      local.get $v278
      local.get $v36
      f64.max
      local.set $v279
      local.get $v270
      local.get $v270
      f64.mul
      local.set $v280
      local.get $v280
      local.get $v274
      f64.add
      local.set $v281
      local.get $v281
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v282
      local.get $v280
      local.get $v274
      f64.sub
      local.set $v283
      local.get $v283
      local.get $v5
      f64.add
      local.set $v284
      local.get $v284
      local.get $v3
      f64.min
      local.set $v285
      local.get $v285
      local.get $v37
      f64.max
      local.set $v286
      local.get $v276
      local.get $v282
      f64.add
      local.set $v287
      local.get $v286
      local.get $v286
      f64.mul
      local.set $v288
      local.get $v286
      f64.const 2.0
      f64.mul
      local.set $v289
      local.get $v289
      local.get $v279
      f64.mul
      local.set $v290
      local.get $v279
      local.get $v279
      f64.mul
      local.set $v291
      local.get $v288
      local.get $v291
      f64.add
      local.set $v292
      local.get $v292
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v293
      local.get $v288
      local.get $v291
      f64.sub
      local.set $v294
      local.get $v294
      local.get $v5
      f64.add
      local.set $v295
      local.get $v295
      local.get $v3
      f64.min
      local.set $v296
      local.get $v296
      local.get $v38
      f64.max
      local.set $v297
      local.get $v290
      local.get $v2
      f64.add
      local.set $v298
      local.get $v298
      local.get $v3
      f64.min
      local.set $v299
      local.get $v287
      local.get $v293
      f64.add
      local.set $v300
      local.get $v299
      local.get $v39
      f64.max
      local.set $v301
      local.get $v301
      local.get $v301
      f64.mul
      local.set $v302
      local.get $v297
      f64.const 2.0
      f64.mul
      local.set $v303
      local.get $v303
      local.get $v301
      f64.mul
      local.set $v304
      local.get $v297
      local.get $v297
      f64.mul
      local.set $v305
      local.get $v305
      local.get $v302
      f64.add
      local.set $v306
      local.get $v306
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v307
      local.get $v305
      local.get $v302
      f64.sub
      local.set $v308
      local.get $v308
      local.get $v5
      f64.add
      local.set $v309
      local.get $v309
      local.get $v3
      f64.min
      local.set $v310
      local.get $v304
      local.get $v2
      f64.add
      local.set $v311
      local.get $v310
      local.get $v40
      f64.max
      local.set $v312
      local.get $v312
      local.get $v312
      f64.mul
      local.set $v313
      local.get $v312
      f64.const 2.0
      f64.mul
      local.set $v314
      local.get $v300
      local.get $v307
      f64.add
      local.set $v315
      local.get $v311
      local.get $v3
      f64.min
      local.set $v316
      local.get $v316
      local.get $v41
      f64.max
      local.set $v317
      local.get $v317
      local.get $v317
      f64.mul
      local.set $v318
      local.get $v313
      local.get $v318
      f64.sub
      local.set $v319
      local.get $v319
      local.get $v5
      f64.add
      local.set $v320
      local.get $v314
      local.get $v317
      f64.mul
      local.set $v321
      local.get $v313
      local.get $v318
      f64.add
      local.set $v322
      local.get $v322
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v323
      local.get $v320
      local.get $v3
      f64.min
      local.set $v324
      local.get $v324
      local.get $v42
      f64.max
      local.set $v325
      local.get $v325
      f64.const 2.0
      f64.mul
      local.set $v326
      local.get $v325
      local.get $v325
      f64.mul
      local.set $v327
      local.get $v315
      local.get $v323
      f64.add
      local.set $v328
      local.get $v321
      local.get $v2
      f64.add
      local.set $v329
      local.get $v329
      local.get $v3
      f64.min
      local.set $v330
      local.get $v330
      local.get $v43
      f64.max
      local.set $v331
      local.get $v331
      local.get $v331
      f64.mul
      local.set $v332
      local.get $v326
      local.get $v331
      f64.mul
      local.set $v333
      local.get $v327
      local.get $v332
      f64.sub
      local.set $v334
      local.get $v334
      local.get $v5
      f64.add
      local.set $v335
      local.get $v335
      local.get $v3
      f64.min
      local.set $v336
      local.get $v336
      local.get $v44
      f64.max
      local.set $v337
      local.get $v337
      local.get $v337
      f64.mul
      local.set $v338
      local.get $v337
      f64.const 2.0
      f64.mul
      local.set $v339
      local.get $v327
      local.get $v332
      f64.add
      local.set $v340
      local.get $v340
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v341
      local.get $v328
      local.get $v341
      f64.add
      local.set $v342
      local.get $v333
      local.get $v2
      f64.add
      local.set $v343
      local.get $v343
      local.get $v3
      f64.min
      local.set $v344
      local.get $v344
      local.get $v45
      f64.max
      local.set $v345
      local.get $v339
      local.get $v345
      f64.mul
      local.set $v346
      local.get $v346
      local.get $v2
      f64.add
      local.set $v347
      local.get $v345
      local.get $v345
      f64.mul
      local.set $v348
      local.get $v338
      local.get $v348
      f64.add
      local.set $v349
      local.get $v338
      local.get $v348
      f64.sub
      local.set $v350
      local.get $v350
      local.get $v5
      f64.add
      local.set $v351
      local.get $v351
      local.get $v3
      f64.min
      local.set $v352
      local.get $v352
      local.get $v46
      f64.max
      local.set $v353
      local.get $v353
      local.get $v353
      f64.mul
      local.set $v354
      local.get $v349
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v355
      local.get $v342
      local.get $v355
      f64.add
      local.set $v356
      local.get $v347
      local.get $v3
      f64.min
      local.set $v357
      local.get $v357
      local.get $v47
      f64.max
      local.set $v358
      local.get $v358
      local.get $v358
      f64.mul
      local.set $v359
      local.get $v354
      local.get $v359
      f64.add
      local.set $v360
      local.get $v354
      local.get $v359
      f64.sub
      local.set $v361
      local.get $v361
      local.get $v5
      f64.add
      local.set $v362
      local.get $v362
      local.get $v3
      f64.min
      local.set $v363
      local.get $v363
      local.get $v48
      f64.max
      local.set $v364
      local.get $v364
      local.get $v364
      f64.mul
      local.set $v365
      local.get $v364
      f64.const 2.0
      f64.mul
      local.set $v366
      local.get $v353
      f64.const 2.0
      f64.mul
      local.set $v367
      local.get $v367
      local.get $v358
      f64.mul
      local.set $v368
      local.get $v368
      local.get $v2
      f64.add
      local.set $v369
      local.get $v369
      local.get $v3
      f64.min
      local.set $v370
      local.get $v370
      local.get $v49
      f64.max
      local.set $v371
      local.get $v366
      local.get $v371
      f64.mul
      local.set $v372
      local.get $v372
      local.get $v2
      f64.add
      local.set $v373
      local.get $v373
      local.get $v3
      f64.min
      local.set $v374
      local.get $v371
      local.get $v371
      f64.mul
      local.set $v375
      local.get $v365
      local.get $v375
      f64.sub
      local.set $v376
      local.get $v376
      local.get $v5
      f64.add
      local.set $v377
      local.get $v365
      local.get $v375
      f64.add
      local.set $v378
      local.get $v374
      local.get $v50
      f64.max
      local.set $v379
      local.get $v379
      local.get $v379
      f64.mul
      local.set $v380
      local.get $v378
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v381
      local.get $v377
      local.get $v3
      f64.min
      local.set $v382
      local.get $v382
      local.get $v51
      f64.max
      local.set $v383
      local.get $v383
      f64.const 2.0
      f64.mul
      local.set $v384
      local.get $v384
      local.get $v379
      f64.mul
      local.set $v385
      local.get $v383
      local.get $v383
      f64.mul
      local.set $v386
      local.get $v386
      local.get $v380
      f64.add
      local.set $v387
      local.get $v387
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v388
      local.get $v385
      local.get $v2
      f64.add
      local.set $v389
      local.get $v386
      local.get $v380
      f64.sub
      local.set $v390
      local.get $v390
      local.get $v5
      f64.add
      local.set $v391
      local.get $v391
      local.get $v3
      f64.min
      local.set $v392
      local.get $v392
      local.get $v52
      f64.max
      local.set $v393
      local.get $v393
      local.get $v393
      f64.mul
      local.set $v394
      local.get $v393
      f64.const 2.0
      f64.mul
      local.set $v395
      local.get $v389
      local.get $v3
      f64.min
      local.set $v396
      local.get $v396
      local.get $v53
      f64.max
      local.set $v397
      local.get $v397
      local.get $v397
      f64.mul
      local.set $v398
      local.get $v394
      local.get $v398
      f64.sub
      local.set $v399
      local.get $v394
      local.get $v398
      f64.add
      local.set $v400
      local.get $v399
      local.get $v5
      f64.add
      local.set $v401
      local.get $v400
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v402
      local.get $v395
      local.get $v397
      f64.mul
      local.set $v403
      local.get $v403
      local.get $v2
      f64.add
      local.set $v404
      local.get $v404
      local.get $v3
      f64.min
      local.set $v405
      local.get $v401
      local.get $v3
      f64.min
      local.set $v406
      local.get $v406
      local.get $v54
      f64.max
      local.set $v407
      local.get $v407
      local.get $v407
      f64.mul
      local.set $v408
      local.get $v407
      f64.const 2.0
      f64.mul
      local.set $v409
      local.get $v405
      local.get $v55
      f64.max
      local.set $v410
      local.get $v409
      local.get $v410
      f64.mul
      local.set $v411
      local.get $v411
      local.get $v2
      f64.add
      local.set $v412
      local.get $v412
      local.get $v3
      f64.min
      local.set $v413
      local.get $v413
      local.get $v56
      f64.max
      local.set $v414
      local.get $v414
      local.get $v414
      f64.mul
      local.set $v415
      local.get $v410
      local.get $v410
      f64.mul
      local.set $v416
      local.get $v408
      local.get $v416
      f64.add
      local.set $v417
      local.get $v417
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v418
      local.get $v408
      local.get $v416
      f64.sub
      local.set $v419
      local.get $v419
      local.get $v5
      f64.add
      local.set $v420
      local.get $v420
      local.get $v3
      f64.min
      local.set $v421
      local.get $v421
      local.get $v57
      f64.max
      local.set $v422
      local.get $v422
      f64.const 2.0
      f64.mul
      local.set $v423
      local.get $v422
      local.get $v422
      f64.mul
      local.set $v424
      local.get $v424
      local.get $v415
      f64.sub
      local.set $v425
      local.get $v425
      local.get $v5
      f64.add
      local.set $v426
      local.get $v426
      local.get $v3
      f64.min
      local.set $v427
      local.get $v427
      local.get $v58
      f64.max
      local.set $v428
      local.get $v428
      f64.const 2.0
      f64.mul
      local.set $v429
      local.get $v428
      local.get $v428
      f64.mul
      local.set $v430
      local.get $v423
      local.get $v414
      f64.mul
      local.set $v431
      local.get $v431
      local.get $v2
      f64.add
      local.set $v432
      local.get $v432
      local.get $v3
      f64.min
      local.set $v433
      local.get $v433
      local.get $v59
      f64.max
      local.set $v434
      local.get $v434
      local.get $v434
      f64.mul
      local.set $v435
      local.get $v429
      local.get $v434
      f64.mul
      local.set $v436
      local.get $v436
      local.get $v2
      f64.add
      local.set $v437
      local.get $v437
      local.get $v3
      f64.min
      local.set $v438
      local.get $v438
      local.get $v60
      f64.max
      local.set $v439
      local.get $v439
      local.get $v439
      f64.mul
      local.set $v440
      local.get $v430
      local.get $v435
      f64.add
      local.set $v441
      local.get $v441
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v442
      local.get $v430
      local.get $v435
      f64.sub
      local.set $v443
      local.get $v443
      local.get $v5
      f64.add
      local.set $v444
      local.get $v444
      local.get $v3
      f64.min
      local.set $v445
      local.get $v445
      local.get $v61
      f64.max
      local.set $v446
      local.get $v446
      local.get $v446
      f64.mul
      local.set $v447
      local.get $v446
      f64.const 2.0
      f64.mul
      local.set $v448
      local.get $v447
      local.get $v440
      f64.sub
      local.set $v449
      local.get $v449
      local.get $v5
      f64.add
      local.set $v450
      local.get $v447
      local.get $v440
      f64.add
      local.set $v451
      local.get $v451
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v452
      local.get $v424
      local.get $v415
      f64.add
      local.set $v453
      local.get $v453
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v454
      local.get $v448
      local.get $v439
      f64.mul
      local.set $v455
      local.get $v455
      local.get $v2
      f64.add
      local.set $v456
      local.get $v456
      local.get $v3
      f64.min
      local.set $v457
      local.get $v457
      local.get $v62
      f64.max
      local.set $v458
      local.get $v458
      local.get $v458
      f64.mul
      local.set $v459
      local.get $v450
      local.get $v3
      f64.min
      local.set $v460
      local.get $v460
      local.get $v63
      f64.max
      local.set $v461
      local.get $v461
      f64.const 2.0
      f64.mul
      local.set $v462
      local.get $v462
      local.get $v458
      f64.mul
      local.set $v463
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v356
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v360
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v381
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v388
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v402
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v418
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v442
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v452
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v454
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v459
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v461
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v463
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)


(module ;; render__6
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (import "env" "memory" (memory 1))
  (func (export "run") (param $count i32) (param $feed0 i32) (param $feed1 i32) (param $feed2 i32) (param $feed3 i32) (param $feed4 i32) (param $feed5 i32) (param $feed6 i32) (param $feed7 i32) (param $feed8 i32) (param $feed9 i32) (param $feed10 i32) (param $feed11 i32) (param $feed12 i32) (param $feed13 i32) (param $feed14 i32) (param $feed15 i32) (param $feed16 i32) (param $feed17 i32) (param $feed18 i32) (param $feed19 i32) (param $feed20 i32) (param $feed21 i32) (param $feed22 i32) (param $feed23 i32) (param $feed24 i32) (param $feed25 i32) (param $feed26 i32) (param $feed27 i32) (param $feed28 i32) (param $feed29 i32) (param $feed30 i32) (param $feed31 i32) (param $feed32 i32) (param $feed33 i32) (param $feed34 i32) (param $feed35 i32) (param $feed36 i32) (param $feed37 i32) (param $feed38 i32) (param $feed39 i32) (param $feed40 i32) (param $feed41 i32) (param $feed42 i32) (param $feed43 i32) (param $feed44 i32) (param $feed45 i32) (param $feed46 i32) (param $feed47 i32) (param $feed48 i32) (param $out0 i32) (param $out1 i32) (param $out2 i32)
    (local $i i32)
    (local $addr i32)
    (local $v0 f64)
    (local $v1 f64)
    (local $v2 f64)
    (local $v3 f64)
    (local $v4 f64)
    (local $v5 f64)
    (local $v6 f64)
    (local $v7 f64)
    (local $v8 f64)
    (local $v9 f64)
    (local $v10 f64)
    (local $v11 f64)
    (local $v12 f64)
    (local $v13 f64)
    (local $v14 f64)
    (local $v15 f64)
    (local $v16 f64)
    (local $v17 f64)
    (local $v18 f64)
    (local $v19 f64)
    (local $v20 f64)
    (local $v21 f64)
    (local $v22 f64)
    (local $v23 f64)
    (local $v24 f64)
    (local $v25 f64)
    (local $v26 f64)
    (local $v27 f64)
    (local $v28 f64)
    (local $v29 f64)
    (local $v30 f64)
    (local $v31 f64)
    (local $v32 f64)
    (local $v33 f64)
    (local $v34 f64)
    (local $v35 f64)
    (local $v36 f64)
    (local $v37 f64)
    (local $v38 f64)
    (local $v39 f64)
    (local $v40 f64)
    (local $v41 f64)
    (local $v42 f64)
    (local $v43 f64)
    (local $v44 f64)
    (local $v45 f64)
    (local $v46 f64)
    (local $v47 f64)
    (local $v48 f64)
    (local $v49 f64)
    (local $v50 f64)
    (local $v51 f64)
    (local $v52 f64)
    (local $v53 f64)
    (local $v54 f64)
    (local $v55 f64)
    (local $v56 f64)
    (local $v57 f64)
    (local $v58 f64)
    (local $v59 f64)
    (local $v60 f64)
    (local $v61 f64)
    (local $v62 f64)
    (local $v63 f64)
    (local $v64 f64)
    (local $v65 f64)
    (local $v66 f64)
    (local $v67 f64)
    (local $v68 f64)
    (local $v69 f64)
    (local $v70 f64)
    (local $v71 f64)
    (local $v72 f64)
    (local $v73 f64)
    (local $v74 f64)
    (local $v75 f64)
    (local $v76 f64)
    (local $v77 f64)
    (local $v78 f64)
    (local $v79 f64)
    (local $v80 f64)
    (local $v81 f64)
    (local $v82 f64)
    (local $v83 f64)
    (local $v84 f64)
    (local $v85 f64)
    (local $v86 f64)
    (local $v87 f64)
    (local $v88 f64)
    (local $v89 f64)
    (local $v90 f64)
    (local $v91 f64)
    (local $v92 f64)
    (local $v93 f64)
    (local $v94 f64)
    (local $v95 f64)
    (local $v96 f64)
    (local $v97 f64)
    (local $v98 f64)
    (local $v99 f64)
    (local $v100 f64)
    (local $v101 f64)
    (local $v102 f64)
    (local $v103 f64)
    (local $v104 f64)
    (local $v105 f64)
    (local $v106 f64)
    (local $v107 f64)
    (local $v108 f64)
    (local $v109 f64)
    (local $v110 f64)
    (local $v111 f64)
    (local $v112 f64)
    (local $v113 f64)
    (local $v114 f64)
    (local $v115 f64)
    (local $v116 f64)
    (local $v117 f64)
    (local $v118 f64)
    (local $v119 f64)
    (local $v120 f64)
    (local $v121 f64)
    (local $v122 f64)
    (local $v123 f64)
    (local $v124 f64)
    (local $v125 f64)
    (local $v126 f64)
    (local $v127 f64)
    (local $v128 f64)
    (local $v129 f64)
    (local $v130 f64)
    (local $v131 f64)
    (local $v132 f64)
    (local $v133 f64)
    (local $v134 f64)
    (local $v135 f64)
    (local $v136 f64)
    (local $v137 f64)
    (local $v138 f64)
    (local $v139 f64)
    (local $v140 f64)
    (local $v141 f64)
    (local $v142 f64)
    (local $v143 f64)
    (local $v144 f64)
    (local $v145 f64)
    (local $v146 f64)
    (local $v147 f64)
    (local $v148 f64)
    (local $v149 f64)
    (local $v150 f64)
    (local $v151 f64)
    (local $v152 f64)
    (local $v153 f64)
    (local $v154 f64)
    (local $v155 f64)
    (local $v156 f64)
    (local $v157 f64)
    (local $v158 f64)
    (local $v159 f64)
    (local $v160 f64)
    (local $v161 f64)
    (local $v162 f64)
    (local $v163 f64)
    (local $v164 f64)
    (local $v165 f64)
    (local $v166 f64)
    (local $v167 f64)
    (local $v168 f64)
    (local $v169 f64)
    (local $v170 f64)
    (local $v171 f64)
    (local $v172 f64)
    (local $v173 f64)
    (local $v174 f64)
    (local $v175 f64)
    (local $v176 f64)
    (local $v177 f64)
    (local $v178 f64)
    (local $v179 f64)
    (local $v180 f64)
    (local $v181 f64)
    (local $v182 f64)
    (local $v183 f64)
    (local $v184 f64)
    (local $v185 f64)
    (local $v186 f64)
    (local $v187 f64)
    (local $v188 f64)
    (local $v189 f64)
    (local $v190 f64)
    (local $v191 f64)
    (local $v192 f64)
    (local $v193 f64)
    (local $v194 f64)
    (local $v195 f64)
    (local $v196 f64)
    (local $v197 f64)
    (local $v198 f64)
    (local $v199 f64)
    (local $v200 f64)
    (local $v201 f64)
    (local $v202 f64)
    (local $v203 f64)
    (local $v204 f64)
    (local $v205 f64)
    (local $v206 f64)
    (local $v207 f64)
    (local $v208 f64)
    (local $v209 f64)
    (local $v210 f64)
    (local $v211 f64)
    (local $v212 f64)
    (local $v213 f64)
    (local $v214 f64)
    (local $v215 f64)
    (local $v216 f64)
    (local $v217 f64)
    (local $v218 f64)
    (local $v219 f64)
    (local $v220 f64)
    (local $v221 f64)
    (local $v222 f64)
    (local $v223 f64)
    (local $v224 f64)
    (local $v225 f64)
    (local $v226 f64)
    (local $v227 f64)
    (local $v228 f64)
    (local $v229 f64)
    (local $v230 f64)
    (local $v231 f64)
    (local $v232 f64)
    (local $v233 f64)
    (local $v234 f64)
    (local $v235 f64)
    (local $v236 f64)
    (local $v237 f64)
    (local $v238 f64)
    (local $v239 f64)
    (local $v240 f64)
    (local $v241 f64)
    (local $v242 f64)
    (local $v243 f64)
    (local $v244 f64)
    (local $v245 f64)
    (local $v246 f64)
    (local $v247 f64)
    (local $v248 f64)
    (local $v249 f64)
    (local $v250 f64)
    (local $v251 f64)
    (local $v252 f64)
    (local $v253 f64)
    (local $v254 f64)
    (local $v255 f64)
    (local $v256 f64)
    (local $v257 f64)
    (local $v258 f64)
    (local $v259 f64)
    (local $v260 f64)
    (local $v261 f64)
    (local $v262 f64)
    (local $v263 f64)
    (local $v264 f64)
    (local $v265 f64)
    (local $v266 f64)
    (local $v267 f64)
    (local $v268 f64)
    (local $v269 f64)
    (local $v270 f64)
    (local $v271 f64)
    (local $v272 f64)
    (local $v273 f64)
    (local $v274 f64)
    (local $v275 f64)
    (local $v276 f64)
    (local $v277 f64)
    (local $v278 f64)
    (local $v279 f64)
    (local $v280 f64)
    (local $v281 f64)
    (local $v282 f64)
    (local $v283 f64)
    (local $v284 f64)
    (local $v285 f64)
    (local $v286 f64)
    (local $v287 f64)
    (local $v288 f64)
    (local $v289 f64)
    (local $v290 f64)
    (local $v291 f64)
    (local $v292 f64)
    (local $v293 f64)
    (local $v294 f64)
    (local $v295 f64)
    (local $v296 f64)
    (local $v297 f64)
    (local $v298 f64)
    (local $v299 f64)
    (local $v300 f64)
    (local $v301 f64)
    (local $v302 f64)
    (local $v303 f64)
    (local $v304 f64)
    (local $v305 f64)
    (local $v306 f64)
    (local $v307 f64)
    (local $v308 f64)
    (local $v309 f64)
    (local $v310 f64)
    (local $v311 f64)
    (local $v312 f64)
    (local $v313 f64)
    (local $v314 f64)
    (local $v315 f64)
    (local $v316 f64)
    (local $v317 f64)
    (local $v318 f64)
    (local $v319 f64)
    (local $v320 f64)
    (local $v321 f64)
    (local $v322 f64)
    (local $v323 f64)
    (local $v324 f64)
    (local $v325 f64)
    (local $v326 f64)
    (local $v327 f64)
    (local $v328 f64)
    (local $v329 f64)
    (local $v330 f64)
    (local $v331 f64)
    (local $v332 f64)
    (local $v333 f64)
    (local $v334 f64)
    (local $v335 f64)
    (local $v336 f64)
    (local $v337 f64)
    (local $v338 f64)
    (local $v339 f64)
    (local $v340 f64)
    (local $v341 f64)
    (local $v342 f64)
    (local $v343 f64)
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $feed0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $feed1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $feed2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $feed3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $feed4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $feed5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $feed6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $feed7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $feed8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $feed9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $feed10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $feed11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $feed12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $feed13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $feed14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $feed15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $feed16
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $feed17
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $feed18
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $feed19
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $feed20
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $feed21
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v21
      local.get $feed22
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v22
      local.get $feed23
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v23
      local.get $feed24
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v24
      local.get $feed25
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v25
      local.get $feed26
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v26
      local.get $feed27
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v27
      local.get $feed28
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v28
      local.get $feed29
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v29
      local.get $feed30
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v30
      local.get $feed31
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v31
      local.get $feed32
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v32
      local.get $feed33
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v33
      local.get $feed34
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v34
      local.get $feed35
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v35
      local.get $feed36
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v36
      local.get $feed37
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v37
      local.get $feed38
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v38
      local.get $feed39
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v39
      local.get $feed40
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v40
      local.get $feed41
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v41
      local.get $feed42
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v42
      local.get $feed43
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v43
      local.get $feed44
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v44
      local.get $feed45
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v45
      local.get $feed46
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v46
      local.get $feed47
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v47
      local.get $feed48
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v48
      local.get $v0
      local.get $v1
      f64.add
      local.set $v49
      local.get $v49
      local.get $v2
      f64.min
      local.set $v50
      local.get $v3
      local.get $v3
      f64.mul
      local.set $v51
      local.get $v51
      local.get $v4
      f64.add
      local.set $v52
      local.get $v51
      local.get $v4
      f64.sub
      local.set $v53
      local.get $v50
      local.get $v5
      f64.max
      local.set $v54
      local.get $v54
      local.get $v54
      f64.mul
      local.set $v55
      local.get $v52
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v56
      local.get $v53
      local.get $v6
      f64.add
      local.set $v57
      local.get $v57
      local.get $v2
      f64.min
      local.set $v58
      local.get $v58
      local.get $v7
      f64.max
      local.set $v59
      local.get $v59
      local.get $v59
      f64.mul
      local.set $v60
      local.get $v60
      local.get $v55
      f64.add
      local.set $v61
      local.get $v61
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v62
      local.get $v60
      local.get $v55
      f64.sub
      local.set $v63
      local.get $v63
      local.get $v6
      f64.add
      local.set $v64
      local.get $v59
      f64.const 2.0
      f64.mul
      local.set $v65
      local.get $v65
      local.get $v54
      f64.mul
      local.set $v66
      local.get $v66
      local.get $v1
      f64.add
      local.set $v67
      local.get $v67
      local.get $v2
      f64.min
      local.set $v68
      local.get $v64
      local.get $v2
      f64.min
      local.set $v69
      local.get $v69
      local.get $v8
      f64.max
      local.set $v70
      local.get $v70
      local.get $v70
      f64.mul
      local.set $v71
      local.get $v70
      f64.const 2.0
      f64.mul
      local.set $v72
      local.get $v9
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v73
      local.get $v10
      local.get $v73
      f64.add
      local.set $v74
      local.get $v74
      local.get $v11
      f64.add
      local.set $v75
      local.get $v75
      local.get $v12
      f64.add
      local.set $v76
      local.get $v76
      local.get $v13
      f64.add
      local.set $v77
      local.get $v77
      local.get $v14
      f64.add
      local.set $v78
      local.get $v78
      local.get $v15
      f64.add
      local.set $v79
      local.get $v79
      local.get $v16
      f64.add
      local.set $v80
      local.get $v80
      local.get $v17
      f64.add
      local.set $v81
      local.get $v81
      local.get $v56
      f64.add
      local.set $v82
      local.get $v82
      local.get $v62
      f64.add
      local.set $v83
      local.get $v68
      local.get $v18
      f64.max
      local.set $v84
      local.get $v84
      local.get $v84
      f64.mul
      local.set $v85
      local.get $v71
      local.get $v85
      f64.add
      local.set $v86
      local.get $v71
      local.get $v85
      f64.sub
      local.set $v87
      local.get $v86
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v88
      local.get $v83
      local.get $v88
      f64.add
      local.set $v89
      local.get $v72
      local.get $v84
      f64.mul
      local.set $v90
      local.get $v90
      local.get $v1
      f64.add
      local.set $v91
      local.get $v87
      local.get $v6
      f64.add
      local.set $v92
      local.get $v92
      local.get $v2
      f64.min
      local.set $v93
      local.get $v93
      local.get $v19
      f64.max
      local.set $v94
      local.get $v94
      local.get $v94
      f64.mul
      local.set $v95
      local.get $v94
      f64.const 2.0
      f64.mul
      local.set $v96
      local.get $v91
      local.get $v2
      f64.min
      local.set $v97
      local.get $v97
      local.get $v20
      f64.max
      local.set $v98
      local.get $v96
      local.get $v98
      f64.mul
      local.set $v99
      local.get $v98
      local.get $v98
      f64.mul
      local.set $v100
      local.get $v95
      local.get $v100
      f64.add
      local.set $v101
      local.get $v95
      local.get $v100
      f64.sub
      local.set $v102
      local.get $v102
      local.get $v6
      f64.add
      local.set $v103
      local.get $v103
      local.get $v2
      f64.min
      local.set $v104
      local.get $v104
      local.get $v21
      f64.max
      local.set $v105
      local.get $v99
      local.get $v1
      f64.add
      local.set $v106
      local.get $v101
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v107
      local.get $v89
      local.get $v107
      f64.add
      local.set $v108
      local.get $v106
      local.get $v2
      f64.min
      local.set $v109
      local.get $v105
      f64.const 2.0
      f64.mul
      local.set $v110
      local.get $v109
      local.get $v22
      f64.max
      local.set $v111
      local.get $v110
      local.get $v111
      f64.mul
      local.set $v112
      local.get $v112
      local.get $v1
      f64.add
      local.set $v113
      local.get $v111
      local.get $v111
      f64.mul
      local.set $v114
      local.get $v105
      local.get $v105
      f64.mul
      local.set $v115
      local.get $v115
      local.get $v114
      f64.sub
      local.set $v116
      local.get $v115
      local.get $v114
      f64.add
      local.set $v117
      local.get $v117
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v118
      local.get $v108
      local.get $v118
      f64.add
      local.set $v119
      local.get $v116
      local.get $v6
      f64.add
      local.set $v120
      local.get $v120
      local.get $v2
      f64.min
      local.set $v121
      local.get $v121
      local.get $v23
      f64.max
      local.set $v122
      local.get $v122
      local.get $v122
      f64.mul
      local.set $v123
      local.get $v122
      f64.const 2.0
      f64.mul
      local.set $v124
      local.get $v113
      local.get $v2
      f64.min
      local.set $v125
      local.get $v125
      local.get $v24
      f64.max
      local.set $v126
      local.get $v124
      local.get $v126
      f64.mul
      local.set $v127
      local.get $v127
      local.get $v1
      f64.add
      local.set $v128
      local.get $v128
      local.get $v2
      f64.min
      local.set $v129
      local.get $v129
      local.get $v25
      f64.max
      local.set $v130
      local.get $v130
      local.get $v130
      f64.mul
      local.set $v131
      local.get $v126
      local.get $v126
      f64.mul
      local.set $v132
      local.get $v123
      local.get $v132
      f64.add
      local.set $v133
      local.get $v123
      local.get $v132
      f64.sub
      local.set $v134
      local.get $v134
      local.get $v6
      f64.add
      local.set $v135
      local.get $v135
      local.get $v2
      f64.min
      local.set $v136
      local.get $v136
      local.get $v26
      f64.max
      local.set $v137
      local.get $v137
      local.get $v137
      f64.mul
      local.set $v138
      local.get $v137
      f64.const 2.0
      f64.mul
      local.set $v139
      local.get $v138
      local.get $v131
      f64.add
      local.set $v140
      local.get $v140
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v141
      local.get $v138
      local.get $v131
      f64.sub
      local.set $v142
      local.get $v142
      local.get $v6
      f64.add
      local.set $v143
      local.get $v143
      local.get $v2
      f64.min
      local.set $v144
      local.get $v144
      local.get $v27
      f64.max
      local.set $v145
      local.get $v145
      f64.const 2.0
      f64.mul
      local.set $v146
      local.get $v145
      local.get $v145
      f64.mul
      local.set $v147
      local.get $v139
      local.get $v130
      f64.mul
      local.set $v148
      local.get $v148
      local.get $v1
      f64.add
      local.set $v149
      local.get $v149
      local.get $v2
      f64.min
      local.set $v150
      local.get $v150
      local.get $v28
      f64.max
      local.set $v151
      local.get $v146
      local.get $v151
      f64.mul
      local.set $v152
      local.get $v151
      local.get $v151
      f64.mul
      local.set $v153
      local.get $v147
      local.get $v153
      f64.add
      local.set $v154
      local.get $v152
      local.get $v1
      f64.add
      local.set $v155
      local.get $v155
      local.get $v2
      f64.min
      local.set $v156
      local.get $v156
      local.get $v29
      f64.max
      local.set $v157
      local.get $v147
      local.get $v153
      f64.sub
      local.set $v158
      local.get $v158
      local.get $v6
      f64.add
      local.set $v159
      local.get $v154
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v160
      local.get $v159
      local.get $v2
      f64.min
      local.set $v161
      local.get $v157
      local.get $v157
      f64.mul
      local.set $v162
      local.get $v161
      local.get $v30
      f64.max
      local.set $v163
      local.get $v163
      f64.const 2.0
      f64.mul
      local.set $v164
      local.get $v163
      local.get $v163
      f64.mul
      local.set $v165
      local.get $v165
      local.get $v162
      f64.sub
      local.set $v166
      local.get $v166
      local.get $v6
      f64.add
      local.set $v167
      local.get $v165
      local.get $v162
      f64.add
      local.set $v168
      local.get $v168
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v169
      local.get $v164
      local.get $v157
      f64.mul
      local.set $v170
      local.get $v170
      local.get $v1
      f64.add
      local.set $v171
      local.get $v171
      local.get $v2
      f64.min
      local.set $v172
      local.get $v172
      local.get $v31
      f64.max
      local.set $v173
      local.get $v173
      local.get $v173
      f64.mul
      local.set $v174
      local.get $v167
      local.get $v2
      f64.min
      local.set $v175
      local.get $v175
      local.get $v32
      f64.max
      local.set $v176
      local.get $v176
      f64.const 2.0
      f64.mul
      local.set $v177
      local.get $v176
      local.get $v176
      f64.mul
      local.set $v178
      local.get $v177
      local.get $v173
      f64.mul
      local.set $v179
      local.get $v179
      local.get $v1
      f64.add
      local.set $v180
      local.get $v180
      local.get $v2
      f64.min
      local.set $v181
      local.get $v181
      local.get $v33
      f64.max
      local.set $v182
      local.get $v182
      local.get $v182
      f64.mul
      local.set $v183
      local.get $v178
      local.get $v174
      f64.sub
      local.set $v184
      local.get $v178
      local.get $v174
      f64.add
      local.set $v185
      local.get $v185
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v186
      local.get $v184
      local.get $v6
      f64.add
      local.set $v187
      local.get $v187
      local.get $v2
      f64.min
      local.set $v188
      local.get $v188
      local.get $v34
      f64.max
      local.set $v189
      local.get $v189
      f64.const 2.0
      f64.mul
      local.set $v190
      local.get $v190
      local.get $v182
      f64.mul
      local.set $v191
      local.get $v191
      local.get $v1
      f64.add
      local.set $v192
      local.get $v192
      local.get $v2
      f64.min
      local.set $v193
      local.get $v189
      local.get $v189
      f64.mul
      local.set $v194
      local.get $v194
      local.get $v183
      f64.sub
      local.set $v195
      local.get $v195
      local.get $v6
      f64.add
      local.set $v196
      local.get $v196
      local.get $v2
      f64.min
      local.set $v197
      local.get $v197
      local.get $v35
      f64.max
      local.set $v198
      local.get $v198
      local.get $v198
      f64.mul
      local.set $v199
      local.get $v194
      local.get $v183
      f64.add
      local.set $v200
      local.get $v200
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v201
      local.get $v198
      f64.const 2.0
      f64.mul
      local.set $v202
      local.get $v193
      local.get $v36
      f64.max
      local.set $v203
      local.get $v203
      local.get $v203
      f64.mul
      local.set $v204
      local.get $v199
      local.get $v204
      f64.sub
      local.set $v205
      local.get $v205
      local.get $v6
      f64.add
      local.set $v206
      local.get $v206
      local.get $v2
      f64.min
      local.set $v207
      local.get $v207
      local.get $v37
      f64.max
      local.set $v208
      local.get $v199
      local.get $v204
      f64.add
      local.set $v209
      local.get $v202
      local.get $v203
      f64.mul
      local.set $v210
      local.get $v210
      local.get $v1
      f64.add
      local.set $v211
      local.get $v208
      local.get $v208
      f64.mul
      local.set $v212
      local.get $v211
      local.get $v2
      f64.min
      local.set $v213
      local.get $v213
      local.get $v38
      f64.max
      local.set $v214
      local.get $v214
      local.get $v214
      f64.mul
      local.set $v215
      local.get $v212
      local.get $v215
      f64.add
      local.set $v216
      local.get $v216
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v217
      local.get $v208
      f64.const 2.0
      f64.mul
      local.set $v218
      local.get $v218
      local.get $v214
      f64.mul
      local.set $v219
      local.get $v219
      local.get $v1
      f64.add
      local.set $v220
      local.get $v212
      local.get $v215
      f64.sub
      local.set $v221
      local.get $v221
      local.get $v6
      f64.add
      local.set $v222
      local.get $v222
      local.get $v2
      f64.min
      local.set $v223
      local.get $v223
      local.get $v39
      f64.max
      local.set $v224
      local.get $v224
      local.get $v224
      f64.mul
      local.set $v225
      local.get $v224
      f64.const 2.0
      f64.mul
      local.set $v226
      local.get $v209
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v227
      local.get $v220
      local.get $v2
      f64.min
      local.set $v228
      local.get $v228
      local.get $v40
      f64.max
      local.set $v229
      local.get $v229
      local.get $v229
      f64.mul
      local.set $v230
      local.get $v225
      local.get $v230
      f64.sub
      local.set $v231
      local.get $v226
      local.get $v229
      f64.mul
      local.set $v232
      local.get $v232
      local.get $v1
      f64.add
      local.set $v233
      local.get $v225
      local.get $v230
      f64.add
      local.set $v234
      local.get $v234
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v235
      local.get $v231
      local.get $v6
      f64.add
      local.set $v236
      local.get $v236
      local.get $v2
      f64.min
      local.set $v237
      local.get $v237
      local.get $v41
      f64.max
      local.set $v238
      local.get $v238
      local.get $v238
      f64.mul
      local.set $v239
      local.get $v233
      local.get $v2
      f64.min
      local.set $v240
      local.get $v240
      local.get $v42
      f64.max
      local.set $v241
      local.get $v241
      local.get $v241
      f64.mul
      local.set $v242
      local.get $v239
      local.get $v242
      f64.add
      local.set $v243
      local.get $v243
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v244
      local.get $v238
      f64.const 2.0
      f64.mul
      local.set $v245
      local.get $v245
      local.get $v241
      f64.mul
      local.set $v246
      local.get $v246
      local.get $v1
      f64.add
      local.set $v247
      local.get $v247
      local.get $v2
      f64.min
      local.set $v248
      local.get $v239
      local.get $v242
      f64.sub
      local.set $v249
      local.get $v249
      local.get $v6
      f64.add
      local.set $v250
      local.get $v250
      local.get $v2
      f64.min
      local.set $v251
      local.get $v248
      local.get $v43
      f64.max
      local.set $v252
      local.get $v251
      local.get $v44
      f64.max
      local.set $v253
      local.get $v253
      f64.const 2.0
      f64.mul
      local.set $v254
      local.get $v253
      local.get $v253
      f64.mul
      local.set $v255
      local.get $v252
      local.get $v252
      f64.mul
      local.set $v256
      local.get $v255
      local.get $v256
      f64.sub
      local.set $v257
      local.get $v255
      local.get $v256
      f64.add
      local.set $v258
      local.get $v254
      local.get $v252
      f64.mul
      local.set $v259
      local.get $v133
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v260
      local.get $v119
      local.get $v260
      f64.add
      local.set $v261
      local.get $v261
      local.get $v141
      f64.add
      local.set $v262
      local.get $v262
      local.get $v160
      f64.add
      local.set $v263
      local.get $v263
      local.get $v169
      f64.add
      local.set $v264
      local.get $v264
      local.get $v186
      f64.add
      local.set $v265
      local.get $v265
      local.get $v201
      f64.add
      local.set $v266
      local.get $v266
      local.get $v227
      f64.add
      local.set $v267
      local.get $v267
      local.get $v217
      f64.add
      local.set $v268
      local.get $v268
      local.get $v235
      f64.add
      local.set $v269
      local.get $v269
      local.get $v244
      f64.add
      local.set $v270
      local.get $v259
      local.get $v1
      f64.add
      local.set $v271
      local.get $v271
      local.get $v2
      f64.min
      local.set $v272
      local.get $v272
      local.get $v45
      f64.max
      local.set $v273
      local.get $v273
      local.get $v273
      f64.mul
      local.set $v274
      local.get $v257
      local.get $v6
      f64.add
      local.set $v275
      local.get $v258
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v276
      local.get $v270
      local.get $v276
      f64.add
      local.set $v277
      local.get $v275
      local.get $v2
      f64.min
      local.set $v278
      local.get $v278
      local.get $v46
      f64.max
      local.set $v279
      local.get $v279
      local.get $v279
      f64.mul
      local.set $v280
      local.get $v279
      f64.const 2.0
      f64.mul
      local.set $v281
      local.get $v281
      local.get $v273
      f64.mul
      local.set $v282
      local.get $v280
      local.get $v274
      f64.sub
      local.set $v283
      local.get $v280
      local.get $v274
      f64.add
      local.set $v284
      local.get $v284
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v285
      local.get $v277
      local.get $v285
      f64.add
      local.set $v286
      local.get $v282
      local.get $v1
      f64.add
      local.set $v287
      local.get $v283
      local.get $v6
      f64.add
      local.set $v288
      local.get $v288
      local.get $v2
      f64.min
      local.set $v289
      local.get $v289
      local.get $v47
      f64.max
      local.set $v290
      local.get $v290
      local.get $v290
      f64.mul
      local.set $v291
      local.get $v287
      local.get $v2
      f64.min
      local.set $v292
      local.get $v292
      local.get $v48
      f64.max
      local.set $v293
      local.get $v293
      local.get $v293
      f64.mul
      local.set $v294
      local.get $v291
      local.get $v294
      f64.add
      local.set $v295
      local.get $v295
      f64.const 4.0
      f64.le
      f64.convert_i32_u
      local.set $v296
      local.get $v286
      local.get $v296
      f64.add
      local.set $v297
      local.get $v297
      f64.const 0.00625
      f64.mul
      local.set $v298
      local.get $v298
      f64.const 1.0
      f64.min
      local.set $v299
      local.get $v299
      f64.const 0.0
      f64.max
      local.set $v300
      local.get $v300
      f64.sqrt
      local.set $v301
      local.get $v301
      f64.const 0.0
      f64.add
      local.set $v302
      local.get $v302
      f64.const 6.283185307179586
      f64.mul
      local.set $v303
      ;; cos via baked lookup table (see the .wasm)
      local.get $v303
      local.set $v304
      local.get $v304
      f64.const 0.5
      f64.mul
      local.set $v305
      local.get $v305
      f64.const 0.5
      f64.add
      local.set $v306
      local.get $v306
      f64.sqrt
      local.set $v307
      local.get $v307
      f64.sqrt
      local.set $v308
      local.get $v308
      f64.sqrt
      local.set $v309
      local.get $v306
      local.get $v307
      f64.mul
      local.set $v310
      local.get $v310
      local.get $v309
      f64.mul
      local.set $v311
      local.get $v311
      f64.const 255.0
      f64.mul
      local.set $v312
      local.get $v312
      f64.const 0.5
      f64.add
      local.set $v313
      local.get $v313
      f64.const 255.0
      f64.min
      local.set $v314
      local.get $v314
      f64.const 0.0
      f64.max
      local.set $v315
      local.get $v301
      f64.const 0.21
      f64.add
      local.set $v316
      local.get $v301
      f64.const 0.43
      f64.add
      local.set $v317
      local.get $v316
      f64.const 6.283185307179586
      f64.mul
      local.set $v318
      ;; cos via baked lookup table (see the .wasm)
      local.get $v318
      local.set $v319
      local.get $v317
      f64.const 6.283185307179586
      f64.mul
      local.set $v320
      ;; cos via baked lookup table (see the .wasm)
      local.get $v320
      local.set $v321
      local.get $v321
      f64.const 0.5
      f64.mul
      local.set $v322
      local.get $v319
      f64.const 0.5
      f64.mul
      local.set $v323
      local.get $v323
      f64.const 0.5
      f64.add
      local.set $v324
      local.get $v324
      f64.sqrt
      local.set $v325
      local.get $v325
      f64.sqrt
      local.set $v326
      local.get $v326
      f64.sqrt
      local.set $v327
      local.get $v324
      local.get $v325
      f64.mul
      local.set $v328
      local.get $v328
      local.get $v327
      f64.mul
      local.set $v329
      local.get $v329
      f64.const 255.0
      f64.mul
      local.set $v330
      local.get $v330
      f64.const 0.5
      f64.add
      local.set $v331
      local.get $v331
      f64.const 255.0
      f64.min
      local.set $v332
      local.get $v332
      f64.const 0.0
      f64.max
      local.set $v333
      local.get $v322
      f64.const 0.5
      f64.add
      local.set $v334
      local.get $v334
      f64.sqrt
      local.set $v335
      local.get $v335
      f64.sqrt
      local.set $v336
      local.get $v336
      f64.sqrt
      local.set $v337
      local.get $v334
      local.get $v335
      f64.mul
      local.set $v338
      local.get $v338
      local.get $v337
      f64.mul
      local.set $v339
      local.get $v339
      f64.const 255.0
      f64.mul
      local.set $v340
      local.get $v340
      f64.const 0.5
      f64.add
      local.set $v341
      local.get $v341
      f64.const 255.0
      f64.min
      local.set $v342
      local.get $v342
      f64.const 0.0
      f64.max
      local.set $v343
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v315
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v333
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v343
      f64.store
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $body
      )
    )
  )
)
