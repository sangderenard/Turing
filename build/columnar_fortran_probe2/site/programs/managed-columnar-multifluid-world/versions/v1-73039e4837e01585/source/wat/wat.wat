(module ;; columnar_multifluid_rgb_step
  ;; The coordinator owns memory and passes byte offsets. A fused
  ;; elementwise program keeps no private tensor state.
  (memory (export "memory") 1)
  (func (export "run") (param $count i32) (param $managed_time i32) (param $dt i32) (param $audio_low i32) (param $audio_high i32) (param $audio_mid i32) (param $column_x i32) (param $column_y i32) (param $entity_y i32) (param $entity_x i32) (param $entity_velocity_x i32) (param $entity_velocity_y i32) (param $audio_level i32) (param $displacement_velocity i32) (param $ink_red i32) (param $ink_yellow i32) (param $ink_green i32) (param $ink_cyan i32) (param $ink_blue i32) (param $ink_magenta i32) (param $displacement i32) (param $rest_surface i32) (param $out0 i32) (param $out1 i32) (param $out2 i32) (param $out3 i32) (param $out4 i32) (param $out5 i32) (param $out6 i32) (param $out7 i32) (param $out8 i32) (param $out9 i32) (param $out10 i32) (param $out11 i32) (param $out12 i32) (param $out13 i32) (param $out14 i32) (param $out15 i32)
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
    i32.const 0
    local.set $i
    (block $sum_done_0
      (loop $sum_body_0
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_0
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $v6
      local.get $v7
      f64.sub
      local.set $v32
      local.get $v5
      local.get $v8
      f64.sub
      local.set $v34
      f64.const 0.18
      local.set $v49
      local.get $v34
      local.get $v34
      f64.mul
      local.set $v74
      local.get $v32
      local.get $v32
      f64.mul
      local.set $v75
      local.get $v74
      local.get $v75
      f64.add
      local.set $v93
      local.get $v93
      f64.neg
      local.set $v116
      local.get $v93
      local.get $v49
      f64.add
      local.set $v117
      f64.const 11.045000000000002
      local.set $v128
      local.get $v116
      local.get $v128
      f64.div
      local.set $v129
      ;; exp via baked lookup table (see the .wasm)
      local.get $v129
      local.set $v141
      local.get $v141
      local.get $v117
      f64.div
      local.set $v146
      local.get $v151
      local.get $v146
      f64.add
      local.set $v151
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_0
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_1
      (loop $sum_body_1
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_1
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $v2
      local.get $v3
      f64.sub
      local.set $v22
      f64.const 2.0
      local.set $v23
      local.get $v4
      local.get $v23
      f64.mul
      local.set $v24
      f64.const 0.61
      local.set $v25
      local.get $v5
      local.get $v25
      f64.mul
      local.set $v26
      f64.const 0.83
      local.set $v27
      local.get $v6
      local.get $v27
      f64.mul
      local.set $v28
      f64.const 0.37
      local.set $v29
      local.get $v5
      local.get $v29
      f64.mul
      local.set $v30
      f64.const 0.29
      local.set $v31
      local.get $v6
      local.get $v7
      f64.sub
      local.set $v32
      local.get $v6
      local.get $v31
      f64.mul
      local.set $v33
      local.get $v5
      local.get $v8
      f64.sub
      local.set $v34
      f64.const 0.72
      local.set $v40
      f64.const 0.18
      local.set $v49
      local.get $v30
      local.get $v33
      f64.sub
      local.set $v73
      local.get $v34
      local.get $v34
      f64.mul
      local.set $v74
      local.get $v32
      local.get $v32
      f64.mul
      local.set $v75
      local.get $v24
      local.get $v2
      f64.sub
      local.set $v77
      local.get $v22
      local.get $v22
      f64.mul
      local.set $v78
      local.get $v26
      local.get $v28
      f64.add
      local.set $v79
      local.get $v77
      local.get $v3
      f64.sub
      local.set $v91
      ;; sin via baked lookup table (see the .wasm)
      local.get $v73
      local.set $v92
      local.get $v74
      local.get $v75
      f64.add
      local.set $v93
      local.get $v91
      local.get $v91
      f64.mul
      local.set $v112
      local.get $v92
      local.get $v40
      f64.mul
      local.set $v113
      f64.const 0.12
      local.set $v114
      local.get $v93
      local.get $v114
      f64.add
      local.set $v115
      local.get $v93
      f64.neg
      local.set $v116
      local.get $v93
      local.get $v49
      f64.add
      local.set $v117
      local.get $v78
      local.get $v112
      f64.add
      local.set $v125
      local.get $v79
      local.get $v113
      f64.add
      local.set $v126
      local.get $v115
      f64.sqrt
      local.set $v127
      f64.const 11.045000000000002
      local.set $v128
      local.get $v116
      local.get $v128
      f64.div
      local.set $v129
      f64.const 1e-05
      local.set $v136
      local.get $v34
      local.get $v127
      f64.div
      local.set $v137
      local.get $v125
      local.get $v136
      f64.add
      local.set $v138
      ;; cos via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v139
      ;; sin via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v140
      ;; exp via baked lookup table (see the .wasm)
      local.get $v129
      local.set $v141
      local.get $v138
      f64.sqrt
      local.set $v145
      local.get $v141
      local.get $v117
      f64.div
      local.set $v146
      local.get $v22
      local.get $v145
      f64.div
      local.set $v148
      local.get $v91
      local.get $v145
      f64.div
      local.set $v150
      local.get $v140
      local.get $v150
      f64.mul
      local.set $v154
      local.get $v139
      local.get $v148
      f64.mul
      local.set $v155
      local.get $v155
      local.get $v154
      f64.add
      local.set $v158
      local.get $v146
      local.get $v158
      f64.mul
      local.set $v159
      local.get $v137
      local.get $v159
      f64.mul
      local.set $v160
      local.get $v162
      local.get $v160
      f64.add
      local.set $v162
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_1
      )
    )
    i32.const 0
    local.set $i
    (block $sum_done_2
      (loop $sum_body_2
        local.get $i
        local.get $count
        i32.ge_s
        br_if $sum_done_2
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $v2
      local.get $v3
      f64.sub
      local.set $v22
      f64.const 2.0
      local.set $v23
      local.get $v4
      local.get $v23
      f64.mul
      local.set $v24
      f64.const 0.61
      local.set $v25
      local.get $v5
      local.get $v25
      f64.mul
      local.set $v26
      f64.const 0.83
      local.set $v27
      local.get $v6
      local.get $v27
      f64.mul
      local.set $v28
      f64.const 0.37
      local.set $v29
      local.get $v5
      local.get $v29
      f64.mul
      local.set $v30
      f64.const 0.29
      local.set $v31
      local.get $v6
      local.get $v7
      f64.sub
      local.set $v32
      local.get $v6
      local.get $v31
      f64.mul
      local.set $v33
      local.get $v5
      local.get $v8
      f64.sub
      local.set $v34
      f64.const 0.72
      local.set $v40
      f64.const 0.18
      local.set $v49
      local.get $v30
      local.get $v33
      f64.sub
      local.set $v73
      local.get $v34
      local.get $v34
      f64.mul
      local.set $v74
      local.get $v32
      local.get $v32
      f64.mul
      local.set $v75
      local.get $v24
      local.get $v2
      f64.sub
      local.set $v77
      local.get $v22
      local.get $v22
      f64.mul
      local.set $v78
      local.get $v26
      local.get $v28
      f64.add
      local.set $v79
      local.get $v77
      local.get $v3
      f64.sub
      local.set $v91
      ;; sin via baked lookup table (see the .wasm)
      local.get $v73
      local.set $v92
      local.get $v74
      local.get $v75
      f64.add
      local.set $v93
      local.get $v91
      local.get $v91
      f64.mul
      local.set $v112
      local.get $v92
      local.get $v40
      f64.mul
      local.set $v113
      f64.const 0.12
      local.set $v114
      local.get $v93
      local.get $v114
      f64.add
      local.set $v115
      local.get $v93
      f64.neg
      local.set $v116
      local.get $v93
      local.get $v49
      f64.add
      local.set $v117
      local.get $v78
      local.get $v112
      f64.add
      local.set $v125
      local.get $v79
      local.get $v113
      f64.add
      local.set $v126
      local.get $v115
      f64.sqrt
      local.set $v127
      f64.const 11.045000000000002
      local.set $v128
      local.get $v116
      local.get $v128
      f64.div
      local.set $v129
      f64.const 1e-05
      local.set $v136
      local.get $v125
      local.get $v136
      f64.add
      local.set $v138
      ;; cos via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v139
      ;; sin via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v140
      ;; exp via baked lookup table (see the .wasm)
      local.get $v129
      local.set $v141
      local.get $v32
      local.get $v127
      f64.div
      local.set $v142
      local.get $v138
      f64.sqrt
      local.set $v145
      local.get $v141
      local.get $v117
      f64.div
      local.set $v146
      local.get $v22
      local.get $v145
      f64.div
      local.set $v148
      local.get $v91
      local.get $v145
      f64.div
      local.set $v150
      local.get $v140
      local.get $v150
      f64.mul
      local.set $v154
      local.get $v139
      local.get $v148
      f64.mul
      local.set $v155
      local.get $v155
      local.get $v154
      f64.add
      local.set $v158
      local.get $v146
      local.get $v158
      f64.mul
      local.set $v159
      local.get $v142
      local.get $v159
      f64.mul
      local.set $v161
      local.get $v163
      local.get $v161
      f64.add
      local.set $v163
        local.get $i
        i32.const 1
        i32.add
        local.set $i
        br $sum_body_2
      )
    )
    i32.const 0
    local.set $i
    (block $done
      (loop $body
        ;; while i < count
        local.get $i
        local.get $count
        i32.ge_s
        br_if $done
      local.get $managed_time
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v0
      local.get $dt
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v1
      local.get $audio_low
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v2
      local.get $audio_high
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v3
      local.get $audio_mid
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v4
      local.get $column_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v5
      local.get $column_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v6
      local.get $entity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v7
      local.get $entity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v8
      local.get $entity_velocity_x
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v9
      local.get $entity_velocity_y
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v10
      local.get $audio_level
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v11
      local.get $displacement_velocity
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v12
      local.get $ink_red
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v13
      local.get $ink_yellow
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v14
      local.get $ink_green
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v15
      local.get $ink_cyan
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v16
      local.get $ink_blue
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v17
      local.get $ink_magenta
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v18
      local.get $displacement
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v19
      local.get $rest_surface
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      f64.load
      local.set $v20
      local.get $v0
      local.get $v1
      f64.add
      local.set $v21
      local.get $v2
      local.get $v3
      f64.sub
      local.set $v22
      f64.const 2.0
      local.set $v23
      local.get $v4
      local.get $v23
      f64.mul
      local.set $v24
      f64.const 0.61
      local.set $v25
      local.get $v5
      local.get $v25
      f64.mul
      local.set $v26
      f64.const 0.83
      local.set $v27
      local.get $v6
      local.get $v27
      f64.mul
      local.set $v28
      f64.const 0.37
      local.set $v29
      local.get $v5
      local.get $v29
      f64.mul
      local.set $v30
      f64.const 0.29
      local.set $v31
      local.get $v6
      local.get $v7
      f64.sub
      local.set $v32
      local.get $v6
      local.get $v31
      f64.mul
      local.set $v33
      local.get $v5
      local.get $v8
      f64.sub
      local.set $v34
      f64.const 0.0
      local.set $v35
      local.get $v5
      local.get $v35
      f64.mul
      local.set $v36
      local.get $v6
      local.get $v35
      f64.mul
      local.set $v37
      f64.const 5.0
      local.set $v38
      local.get $v38
      local.get $v8
      f64.sub
      local.set $v39
      f64.const 0.72
      local.set $v40
      local.get $v9
      local.get $v40
      f64.mul
      local.set $v41
      f64.const 3.5
      local.set $v42
      local.get $v42
      local.get $v7
      f64.sub
      local.set $v43
      local.get $v10
      local.get $v40
      f64.mul
      local.set $v44
      f64.const 0.42
      local.set $v45
      local.get $v2
      local.get $v45
      f64.mul
      local.set $v46
      f64.const 0.34
      local.set $v47
      local.get $v4
      local.get $v47
      f64.mul
      local.set $v48
      f64.const 0.18
      local.set $v49
      local.get $v3
      local.get $v49
      f64.mul
      local.set $v50
      f64.const 0.35
      local.set $v51
      local.get $v11
      local.get $v51
      f64.mul
      local.set $v52
      f64.const 8.0
      local.set $v53
      local.get $v12
      local.get $v53
      f64.mul
      local.set $v54
      f64.const -0.05
      local.set $v55
      local.get $v1
      local.get $v55
      f64.mul
      local.set $v56
      f64.const 2.8
      local.set $v57
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v58
      f64.const -0.054
      local.set $v59
      local.get $v1
      local.get $v59
      f64.mul
      local.set $v60
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v61
      f64.const -0.058
      local.set $v62
      local.get $v1
      local.get $v62
      f64.mul
      local.set $v63
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v64
      f64.const -0.062
      local.set $v65
      local.get $v1
      local.get $v65
      f64.mul
      local.set $v66
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v67
      f64.const -0.066
      local.set $v68
      local.get $v1
      local.get $v68
      f64.mul
      local.set $v69
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v70
      f64.const -0.07
      local.set $v71
      local.get $v1
      local.get $v71
      f64.mul
      local.set $v72
      local.get $v30
      local.get $v33
      f64.sub
      local.set $v73
      local.get $v34
      local.get $v34
      f64.mul
      local.set $v74
      local.get $v32
      local.get $v32
      f64.mul
      local.set $v75
      local.get $v1
      local.get $v57
      f64.mul
      local.set $v76
      local.get $v24
      local.get $v2
      f64.sub
      local.set $v77
      local.get $v22
      local.get $v22
      f64.mul
      local.set $v78
      local.get $v26
      local.get $v28
      f64.add
      local.set $v79
      f64.const 0.1
      local.set $v80
      local.get $v39
      local.get $v80
      f64.mul
      local.set $v81
      local.get $v43
      local.get $v80
      f64.mul
      local.set $v82
      local.get $v46
      local.get $v48
      f64.add
      local.set $v83
      local.get $v21
      local.get $v45
      f64.mul
      local.set $v84
      ;; exp via baked lookup table (see the .wasm)
      local.get $v56
      local.set $v85
      ;; exp via baked lookup table (see the .wasm)
      local.get $v60
      local.set $v86
      ;; exp via baked lookup table (see the .wasm)
      local.get $v63
      local.set $v87
      ;; exp via baked lookup table (see the .wasm)
      local.get $v66
      local.set $v88
      ;; exp via baked lookup table (see the .wasm)
      local.get $v69
      local.set $v89
      ;; exp via baked lookup table (see the .wasm)
      local.get $v72
      local.set $v90
      local.get $v77
      local.get $v3
      f64.sub
      local.set $v91
      ;; sin via baked lookup table (see the .wasm)
      local.get $v73
      local.set $v92
      local.get $v74
      local.get $v75
      f64.add
      local.set $v93
      local.get $v83
      local.get $v50
      f64.add
      local.set $v94
      ;; cos via baked lookup table (see the .wasm)
      local.get $v84
      local.set $v95
      f64.const 1.0471975511965976
      local.set $v96
      local.get $v84
      local.get $v96
      f64.sub
      local.set $v97
      f64.const 2.0943951023931953
      local.set $v98
      local.get $v84
      local.get $v98
      f64.sub
      local.set $v99
      f64.const 3.141592653589793
      local.set $v100
      local.get $v84
      local.get $v100
      f64.sub
      local.set $v101
      f64.const 4.1887902047863905
      local.set $v102
      local.get $v84
      local.get $v102
      f64.sub
      local.set $v103
      f64.const 5.235987755982989
      local.set $v104
      local.get $v84
      local.get $v104
      f64.sub
      local.set $v105
      local.get $v13
      local.get $v85
      f64.mul
      local.set $v106
      local.get $v14
      local.get $v86
      f64.mul
      local.set $v107
      local.get $v15
      local.get $v87
      f64.mul
      local.set $v108
      local.get $v16
      local.get $v88
      f64.mul
      local.set $v109
      local.get $v17
      local.get $v89
      f64.mul
      local.set $v110
      local.get $v18
      local.get $v90
      f64.mul
      local.set $v111
      local.get $v91
      local.get $v91
      f64.mul
      local.set $v112
      local.get $v92
      local.get $v40
      f64.mul
      local.set $v113
      f64.const 0.12
      local.set $v114
      local.get $v93
      local.get $v114
      f64.add
      local.set $v115
      local.get $v93
      f64.neg
      local.set $v116
      local.get $v93
      local.get $v49
      f64.add
      local.set $v117
      local.get $v94
      local.get $v52
      f64.add
      local.set $v118
      local.get $v95
      local.get $v35
      f64.max
      local.set $v119
      ;; cos via baked lookup table (see the .wasm)
      local.get $v97
      local.set $v120
      ;; cos via baked lookup table (see the .wasm)
      local.get $v99
      local.set $v121
      ;; cos via baked lookup table (see the .wasm)
      local.get $v101
      local.set $v122
      ;; cos via baked lookup table (see the .wasm)
      local.get $v103
      local.set $v123
      ;; cos via baked lookup table (see the .wasm)
      local.get $v105
      local.set $v124
      local.get $v78
      local.get $v112
      f64.add
      local.set $v125
      local.get $v79
      local.get $v113
      f64.add
      local.set $v126
      local.get $v115
      f64.sqrt
      local.set $v127
      f64.const 11.045000000000002
      local.set $v128
      local.get $v116
      local.get $v128
      f64.div
      local.set $v129
      local.get $v118
      local.get $v35
      f64.max
      local.set $v130
      local.get $v120
      local.get $v35
      f64.max
      local.set $v131
      local.get $v121
      local.get $v35
      f64.max
      local.set $v132
      local.get $v122
      local.get $v35
      f64.max
      local.set $v133
      local.get $v123
      local.get $v35
      f64.max
      local.set $v134
      local.get $v124
      local.get $v35
      f64.max
      local.set $v135
      f64.const 1e-05
      local.set $v136
      local.get $v34
      local.get $v127
      f64.div
      local.set $v137
      local.get $v125
      local.get $v136
      f64.add
      local.set $v138
      ;; cos via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v139
      ;; sin via baked lookup table (see the .wasm)
      local.get $v126
      local.set $v140
      ;; exp via baked lookup table (see the .wasm)
      local.get $v129
      local.set $v141
      local.get $v32
      local.get $v127
      f64.div
      local.set $v142
      f64.const 1.0
      local.set $v143
      local.get $v130
      local.get $v143
      f64.min
      local.set $v144
      local.get $v138
      f64.sqrt
      local.set $v145
      local.get $v141
      local.get $v117
      f64.div
      local.set $v146
      f64.const 0.3
      local.set $v147
      local.get $v22
      local.get $v145
      f64.div
      local.set $v148
      local.get $v144
      local.get $v147
      f64.mul
      local.set $v149
      local.get $v91
      local.get $v145
      f64.div
      local.set $v150
      f64.const 0.32
      local.set $v152
      local.get $v149
      local.get $v152
      f64.add
      local.set $v153
      local.get $v140
      local.get $v150
      f64.mul
      local.set $v154
      local.get $v139
      local.get $v148
      f64.mul
      local.set $v155
      f64.const 1e-06
      local.set $v156
      local.get $v151
      local.get $v156
      f64.add
      local.set $v157
      local.get $v155
      local.get $v154
      f64.add
      local.set $v158
      local.get $v146
      local.get $v158
      f64.mul
      local.set $v159
      local.get $v137
      local.get $v159
      f64.mul
      local.set $v160
      local.get $v142
      local.get $v159
      f64.mul
      local.set $v161
      local.get $v163
      local.get $v157
      f64.div
      local.set $v164
      local.get $v162
      local.get $v157
      f64.div
      local.set $v165
      local.get $v36
      local.get $v165
      f64.add
      local.set $v166
      local.get $v37
      local.get $v164
      f64.add
      local.set $v167
      f64.const 1.85
      local.set $v168
      local.get $v166
      local.get $v168
      f64.mul
      local.set $v169
      local.get $v169
      local.get $v81
      f64.add
      local.set $v170
      local.get $v170
      local.get $v41
      f64.sub
      local.set $v171
      local.get $v171
      local.get $v1
      f64.mul
      local.set $v172
      local.get $v9
      local.get $v172
      f64.add
      local.set $v173
      local.get $v173
      local.get $v1
      f64.mul
      local.set $v174
      local.get $v8
      local.get $v174
      f64.add
      local.set $v175
      local.get $v167
      local.get $v168
      f64.mul
      local.set $v176
      local.get $v176
      local.get $v82
      f64.add
      local.set $v177
      local.get $v177
      local.get $v44
      f64.sub
      local.set $v178
      local.get $v178
      local.get $v1
      f64.mul
      local.set $v179
      local.get $v10
      local.get $v179
      f64.add
      local.set $v180
      local.get $v180
      local.get $v1
      f64.mul
      local.set $v181
      local.get $v7
      local.get $v181
      f64.add
      local.set $v182
      f64.const 0.65
      local.set $v183
      local.get $v175
      local.get $v183
      f64.max
      local.set $v184
      local.get $v182
      local.get $v183
      f64.max
      local.set $v185
      f64.const 9.35
      local.set $v186
      local.get $v184
      local.get $v186
      f64.min
      local.set $v187
      f64.const 6.35
      local.set $v188
      local.get $v185
      local.get $v188
      f64.min
      local.set $v189
      local.get $v6
      local.get $v189
      f64.sub
      local.set $v190
      local.get $v190
      f64.abs
      local.set $v191
      local.get $v153
      local.get $v191
      f64.sub
      local.set $v192
      local.get $v5
      local.get $v187
      f64.sub
      local.set $v193
      local.get $v5
      local.get $v187
      f64.sub
      local.set $v194
      local.get $v193
      local.get $v194
      f64.mul
      local.set $v195
      local.get $v6
      local.get $v189
      f64.sub
      local.set $v196
      local.get $v6
      local.get $v189
      f64.sub
      local.set $v197
      local.get $v196
      local.get $v197
      f64.mul
      local.set $v198
      local.get $v195
      local.get $v198
      f64.add
      local.set $v199
      local.get $v5
      local.get $v187
      f64.sub
      local.set $v200
      local.get $v200
      f64.abs
      local.set $v201
      local.get $v153
      local.get $v201
      f64.sub
      local.set $v202
      local.get $v202
      local.get $v35
      f64.max
      local.set $v203
      local.get $v199
      f64.neg
      local.set $v204
      local.get $v199
      f64.neg
      local.set $v205
      local.get $v199
      f64.neg
      local.set $v206
      local.get $v192
      local.get $v35
      f64.max
      local.set $v207
      local.get $v203
      local.get $v207
      f64.min
      local.set $v208
      local.get $v199
      f64.neg
      local.set $v209
      local.get $v199
      f64.neg
      local.set $v210
      local.get $v199
      f64.neg
      local.set $v211
      local.get $v199
      f64.neg
      local.set $v212
      f64.const 3.6450000000000005
      local.set $v213
      local.get $v210
      local.get $v213
      f64.div
      local.set $v214
      f64.const 0.3872
      local.set $v215
      local.get $v211
      local.get $v215
      f64.div
      local.set $v216
      f64.const 0.4608
      local.set $v217
      local.get $v212
      local.get $v217
      f64.div
      local.set $v218
      f64.const 0.5408000000000001
      local.set $v219
      local.get $v204
      local.get $v219
      f64.div
      local.set $v220
      f64.const 0.6272000000000001
      local.set $v221
      local.get $v205
      local.get $v221
      f64.div
      local.set $v222
      f64.const 0.72
      local.set $v223
      local.get $v206
      local.get $v223
      f64.div
      local.set $v224
      f64.const 0.8192
      local.set $v225
      ;; exp via baked lookup table (see the .wasm)
      local.get $v220
      local.set $v226
      local.get $v209
      local.get $v225
      f64.div
      local.set $v227
      ;; exp via baked lookup table (see the .wasm)
      local.get $v227
      local.set $v228
      ;; exp via baked lookup table (see the .wasm)
      local.get $v224
      local.set $v229
      ;; exp via baked lookup table (see the .wasm)
      local.get $v222
      local.set $v230
      local.get $v208
      local.get $v153
      f64.div
      local.set $v231
      ;; exp via baked lookup table (see the .wasm)
      local.get $v216
      local.set $v232
      ;; exp via baked lookup table (see the .wasm)
      local.get $v214
      local.set $v233
      ;; exp via baked lookup table (see the .wasm)
      local.get $v218
      local.set $v234
      f64.const -0.42
      local.set $v235
      local.get $v233
      local.get $v235
      f64.mul
      local.set $v236
      f64.const 0.22
      local.set $v237
      local.get $v231
      local.get $v237
      f64.mul
      local.set $v238
      f64.const 54.0
      local.set $v239
      local.get $v233
      local.get $v239
      f64.mul
      local.set $v240
      f64.const 30.0
      local.set $v241
      local.get $v233
      local.get $v241
      f64.mul
      local.set $v242
      f64.const 20.0
      local.set $v243
      local.get $v76
      local.get $v228
      f64.mul
      local.set $v244
      local.get $v244
      local.get $v135
      f64.mul
      local.set $v245
      local.get $v61
      local.get $v234
      f64.mul
      local.set $v246
      local.get $v246
      local.get $v131
      f64.mul
      local.set $v247
      local.get $v64
      local.get $v226
      f64.mul
      local.set $v248
      local.get $v248
      local.get $v132
      f64.mul
      local.set $v249
      local.get $v67
      local.get $v230
      f64.mul
      local.set $v250
      local.get $v250
      local.get $v133
      f64.mul
      local.set $v251
      local.get $v70
      local.get $v229
      f64.mul
      local.set $v252
      local.get $v252
      local.get $v134
      f64.mul
      local.set $v253
      local.get $v231
      local.get $v231
      f64.mul
      local.set $v254
      local.get $v58
      local.get $v232
      f64.mul
      local.set $v255
      local.get $v255
      local.get $v119
      f64.mul
      local.set $v256
      local.get $v238
      local.get $v231
      f64.mul
      local.set $v257
      local.get $v233
      local.get $v243
      f64.mul
      local.set $v258
      local.get $v143
      local.get $v254
      f64.sub
      local.set $v259
      f64.const 245.0
      local.set $v260
      local.get $v254
      local.get $v260
      f64.mul
      local.set $v261
      local.get $v143
      local.get $v254
      f64.sub
      local.set $v262
      f64.const 252.0
      local.set $v263
      local.get $v254
      local.get $v263
      f64.mul
      local.set $v264
      local.get $v143
      local.get $v254
      f64.sub
      local.set $v265
      f64.const 255.0
      local.set $v266
      local.get $v254
      local.get $v266
      f64.mul
      local.set $v267
      local.get $v108
      local.get $v249
      f64.add
      local.set $v268
      local.get $v106
      local.get $v256
      f64.add
      local.set $v269
      local.get $v109
      local.get $v251
      f64.add
      local.set $v270
      local.get $v107
      local.get $v247
      f64.add
      local.set $v271
      local.get $v110
      local.get $v253
      f64.add
      local.set $v272
      local.get $v111
      local.get $v245
      f64.add
      local.set $v273
      local.get $v236
      local.get $v257
      f64.sub
      local.set $v274
      local.get $v274
      local.get $v19
      f64.sub
      local.set $v275
      local.get $v269
      local.get $v143
      f64.min
      local.set $v276
      local.get $v271
      local.get $v143
      f64.min
      local.set $v277
      local.get $v268
      local.get $v143
      f64.min
      local.set $v278
      local.get $v270
      local.get $v143
      f64.min
      local.set $v279
      local.get $v272
      local.get $v143
      f64.min
      local.set $v280
      local.get $v273
      local.get $v143
      f64.min
      local.set $v281
      local.get $v277
      local.get $v278
      f64.add
      local.set $v282
      local.get $v282
      local.get $v279
      f64.add
      local.set $v283
      local.get $v276
      local.get $v277
      f64.add
      local.set $v284
      local.get $v284
      local.get $v281
      f64.add
      local.set $v285
      local.get $v279
      local.get $v280
      f64.add
      local.set $v286
      local.get $v286
      local.get $v281
      f64.add
      local.set $v287
      local.get $v275
      local.get $v243
      f64.mul
      local.set $v288
      local.get $v288
      local.get $v54
      f64.sub
      local.set $v289
      local.get $v289
      local.get $v1
      f64.mul
      local.set $v290
      local.get $v276
      local.get $v277
      f64.add
      local.set $v291
      local.get $v291
      local.get $v278
      f64.add
      local.set $v292
      local.get $v292
      local.get $v279
      f64.add
      local.set $v293
      local.get $v285
      local.get $v266
      f64.mul
      local.set $v294
      local.get $v283
      local.get $v266
      f64.mul
      local.set $v295
      local.get $v287
      local.get $v266
      f64.mul
      local.set $v296
      local.get $v12
      local.get $v290
      f64.add
      local.set $v297
      local.get $v297
      f64.abs
      local.set $v298
      local.get $v297
      local.get $v1
      f64.mul
      local.set $v299
      local.get $v19
      local.get $v299
      f64.add
      local.set $v300
      local.get $v293
      local.get $v280
      f64.add
      local.set $v301
      local.get $v301
      local.get $v281
      f64.add
      local.set $v302
      local.get $v298
      local.get $v143
      f64.min
      local.set $v303
      local.get $v20
      local.get $v300
      f64.add
      local.set $v304
      local.get $v302
      local.get $v156
      f64.max
      local.set $v305
      local.get $v300
      f64.neg
      local.set $v306
      local.get $v303
      local.get $v53
      f64.mul
      local.set $v307
      f64.const 0.88
      local.set $v308
      local.get $v294
      local.get $v305
      f64.div
      local.set $v309
      local.get $v305
      local.get $v308
      f64.min
      local.set $v310
      local.get $v295
      local.get $v305
      f64.div
      local.set $v311
      local.get $v296
      local.get $v305
      f64.div
      local.set $v312
      f64.const 0.5
      local.set $v313
      local.get $v304
      local.get $v313
      f64.sub
      local.set $v314
      local.get $v306
      local.get $v45
      f64.div
      local.set $v315
      local.get $v143
      local.get $v310
      f64.sub
      local.set $v316
      local.get $v309
      local.get $v310
      f64.mul
      local.set $v317
      local.get $v143
      local.get $v310
      f64.sub
      local.set $v318
      local.get $v311
      local.get $v310
      f64.mul
      local.set $v319
      local.get $v143
      local.get $v310
      f64.sub
      local.set $v320
      local.get $v312
      local.get $v310
      f64.mul
      local.set $v321
      local.get $v314
      local.get $v38
      f64.div
      local.set $v322
      local.get $v315
      local.get $v35
      f64.max
      local.set $v323
      local.get $v322
      local.get $v35
      f64.max
      local.set $v324
      local.get $v323
      local.get $v143
      f64.min
      local.set $v325
      local.get $v324
      local.get $v143
      f64.min
      local.set $v326
      f64.const 34.0
      local.set $v327
      local.get $v325
      local.get $v327
      f64.mul
      local.set $v328
      f64.const 21.0
      local.set $v329
      local.get $v325
      local.get $v329
      f64.mul
      local.set $v330
      f64.const 15.0
      local.set $v331
      local.get $v325
      local.get $v331
      f64.mul
      local.set $v332
      f64.const 27.0
      local.set $v333
      local.get $v326
      local.get $v333
      f64.mul
      local.set $v334
      f64.const 18.0
      local.set $v335
      local.get $v326
      local.get $v335
      f64.mul
      local.set $v336
      f64.const 16.0
      local.set $v337
      local.get $v326
      local.get $v337
      f64.mul
      local.set $v338
      f64.const 186.0
      local.set $v339
      local.get $v334
      local.get $v339
      f64.add
      local.set $v340
      f64.const 220.0
      local.set $v341
      local.get $v336
      local.get $v341
      f64.add
      local.set $v342
      f64.const 232.0
      local.set $v343
      local.get $v340
      local.get $v328
      f64.sub
      local.set $v344
      local.get $v344
      local.get $v240
      f64.add
      local.set $v345
      local.get $v342
      local.get $v330
      f64.sub
      local.set $v346
      local.get $v346
      local.get $v242
      f64.add
      local.set $v347
      local.get $v338
      local.get $v343
      f64.add
      local.set $v348
      local.get $v348
      local.get $v332
      f64.add
      local.set $v349
      local.get $v349
      local.get $v258
      f64.add
      local.set $v350
      local.get $v347
      local.get $v307
      f64.add
      local.set $v351
      local.get $v345
      local.get $v35
      f64.max
      local.set $v352
      local.get $v350
      local.get $v35
      f64.max
      local.set $v353
      local.get $v352
      local.get $v266
      f64.min
      local.set $v354
      local.get $v351
      local.get $v35
      f64.max
      local.set $v355
      local.get $v353
      local.get $v266
      f64.min
      local.set $v356
      local.get $v354
      local.get $v316
      f64.mul
      local.set $v357
      local.get $v357
      local.get $v317
      f64.add
      local.set $v358
      local.get $v358
      local.get $v259
      f64.mul
      local.set $v359
      local.get $v359
      local.get $v261
      f64.add
      local.set $v360
      local.get $v356
      local.get $v320
      f64.mul
      local.set $v361
      local.get $v361
      local.get $v321
      f64.add
      local.set $v362
      local.get $v362
      local.get $v265
      f64.mul
      local.set $v363
      local.get $v363
      local.get $v267
      f64.add
      local.set $v364
      local.get $v355
      local.get $v266
      f64.min
      local.set $v365
      local.get $v365
      local.get $v318
      f64.mul
      local.set $v366
      local.get $v366
      local.get $v319
      f64.add
      local.set $v367
      local.get $v367
      local.get $v262
      f64.mul
      local.set $v368
      local.get $v368
      local.get $v264
      f64.add
      local.set $v369
      local.get $out0
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v360
      f64.store
      local.get $out1
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v369
      f64.store
      local.get $out2
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v364
      f64.store
      local.get $out3
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v300
      f64.store
      local.get $out4
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v297
      f64.store
      local.get $out5
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v187
      f64.store
      local.get $out6
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v189
      f64.store
      local.get $out7
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v173
      f64.store
      local.get $out8
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v180
      f64.store
      local.get $out9
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v21
      f64.store
      local.get $out10
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v276
      f64.store
      local.get $out11
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v277
      f64.store
      local.get $out12
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v278
      f64.store
      local.get $out13
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v279
      f64.store
      local.get $out14
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v280
      f64.store
      local.get $out15
      local.get $i
      i32.const 8
      i32.mul
      i32.add
      local.get $v281
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
