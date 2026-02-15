// KERNEL 0
// Kind: simd
// ThreadCountScale Optional(3904)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_0(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(3904)) {
    int t89 = id;
    int t90 = t89 / 3904;
    uint _frameIndex = (uint)(t90);
    int t91 = t90 * 3904;
    int t92 = t89 - t91;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=19, axis=2, in=[61, 64, 3], out=[61, 64], inFA=false, outFA=false), value: empty) */
    float t93 = 0.0;
    int t94 = t92 / 64;
    int t95 = t94 * 64;
    int t96 = t92 - t95;
    int t97 = t96;
    int t98 = t97;
    int t99 = t96 - t98;
    int t100 = t94 * 192;
    int t101 = t100;
    int t102 = t97 * 3;
    int t103 = t101 + t102;
    for (uint t104 = 0; t104 < 3; t104++) {
      int t105 = t104;
      int t106 = t103 + t105;
      int t107 = t94 * 3;
      int t108 = t107 + t104;
      float t109 = memory[8706 + t108];
      int t110 = t97 * 3;
      int t111 = t110 + t104;
      int t112 = t111 / 3;
      int t113 = t112 * 3;
      int t114 = t111 - t113;
      int t115 = t114 * 64;
      int t116 = t112 + t115;
      float t117 = memory[0 + t116];
      float t118 = t109 * t117;
      float t119 = t93 + t118;
      t93 = t119;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t92] = t93;
    float t122 = memory[25460 + t92];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t123 = t92 / 64;
    int t124 = t123 * 64;
    int t125 = t92 - t124;
    int t126 = t125;
    float t127 = memory[192 + t126];
    float t128 = t122 + t127;
    memory[29364 + t92] = t128;
    float t130 = metal::tanh(t128);
    memory[33268 + t92] = t130;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 1
// Kind: simd
// ThreadCountScale Optional(3904)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_1(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(3904)) {
    int t132 = id;
    int t133 = t132 / 3904;
    uint _frameIndex = (uint)(t133);
    int t134 = t133 * 3904;
    int t135 = t132 - t134;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=26, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
    float t136 = 0.0;
    int t137 = t135 / 64;
    int t138 = t137 * 64;
    int t139 = t135 - t138;
    int t140 = t139;
    int t141 = t140;
    int t142 = t139 - t141;
    int t143 = t137 * 4096;
    int t144 = t143;
    int t145 = t140 * 64;
    int t146 = t144 + t145;
    for (uint t147 = 0; t147 < 64; t147++) {
      int t148 = t147;
      int t149 = t146 + t148;
      int t150 = t137 * 64;
      int t151 = t150 + t147;
      float t152 = memory[33268 + t151];
      int t153 = t140 * 64;
      int t154 = t153 + t147;
      int t155 = t154 / 64;
      int t156 = t155 * 64;
      int t157 = t154 - t156;
      int t158 = t157 * 64;
      int t159 = t155 + t158;
      float t160 = memory[256 + t159];
      float t161 = t152 * t160;
      float t162 = t136 + t161;
      t136 = t162;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t135] = t136;
    float t165 = memory[25460 + t135];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t166 = t135 / 64;
    int t167 = t166 * 64;
    int t168 = t135 - t167;
    int t169 = t168;
    float t170 = memory[4352 + t169];
    float t171 = t165 + t170;
    memory[41076 + t135] = t171;
    float t173 = metal::tanh(t171);
    memory[37172 + t135] = t173;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 2
// Kind: simd
// ThreadCountScale Optional(3904)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_2(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(3904)) {
    int t175 = id;
    int t176 = t175 / 3904;
    uint _frameIndex = (uint)(t176);
    int t177 = t176 * 3904;
    int t178 = t175 - t177;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=33, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
    float t179 = 0.0;
    int t180 = t178 / 64;
    int t181 = t180 * 64;
    int t182 = t178 - t181;
    int t183 = t182;
    int t184 = t183;
    int t185 = t182 - t184;
    int t186 = t180 * 4096;
    int t187 = t186;
    int t188 = t183 * 64;
    int t189 = t187 + t188;
    for (uint t190 = 0; t190 < 64; t190++) {
      int t191 = t190;
      int t192 = t189 + t191;
      int t193 = t180 * 64;
      int t194 = t193 + t190;
      float t195 = memory[37172 + t194];
      int t196 = t183 * 64;
      int t197 = t196 + t190;
      int t198 = t197 / 64;
      int t199 = t198 * 64;
      int t200 = t197 - t199;
      int t201 = t200 * 64;
      int t202 = t198 + t201;
      float t203 = memory[4416 + t202];
      float t204 = t195 * t203;
      float t205 = t179 + t204;
      t179 = t205;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t178] = t179;
    float t208 = memory[25460 + t178];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t209 = t178 / 64;
    int t210 = t209 * 64;
    int t211 = t178 - t210;
    int t212 = t211;
    float t213 = memory[8512 + t212];
    float t214 = t208 + t213;
    memory[48884 + t178] = t214;
    float t216 = t214 * -1.0;
    memory[44980 + t178] = t216;
    float t218 = metal::exp(t216);
    float t219 = 1.0 + t218;
    memory[56692 + t178] = t219;
    float t221 = 1.0 / t219;
    memory[52788 + t178] = t221;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 3
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_3(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(61)) {
    int t223 = id;
    int t224 = t223 / 61;
    uint _frameIndex = (uint)(t224);
    int t225 = t224 * 61;
    int t226 = t223 - t225;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=45, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
    float t227 = 0.0;
    int t228 = t226;
    int t229 = t228;
    int t230 = t226 - t229;
    int t231 = t230;
    int t232 = t231;
    int t233 = t230 - t232;
    int t234 = t228 * 64;
    int t235 = t234;
    int t236 = t231 * 64;
    int t237 = t235 + t236;
    for (uint t238 = 0; t238 < 64; t238++) {
      int t239 = t238;
      int t240 = t237 + t239;
      int t241 = t228 * 64;
      int t242 = t241 + t238;
      float t243 = memory[37172 + t242];
      int t244 = t238 / 64;
      int t245 = t244 * 64;
      int t246 = t238 - t245;
      float t247 = memory[8576 + t246];
      float t248 = t243 * t247;
      float t249 = t227 + t248;
      t227 = t249;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t226] = t227;
    float t252 = memory[25460 + t226];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t253 = t226;
    int t254 = t253;
    int t255 = t226 - t254;
    float t256 = memory[8640 + (int)0.0];
    float t257 = t252 + t256;
    memory[60788 + t226] = t257;
    float t259 = t257 * -1.0;
    memory[60724 + t226] = t259;
    float t261 = metal::exp(t259);
    float t262 = 1.0 + t261;
    memory[60660 + t226] = t262;
    float t264 = 1.0 / t262;
    memory[60596 + t226] = t264;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 4
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_4(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(61)) {
    int t266 = id;
    int t267 = t266 / 61;
    uint _frameIndex = (uint)(t267);
    int t268 = t267 * 61;
    int t269 = t266 - t268;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=57, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
    float t270 = 0.0;
    int t271 = t269;
    int t272 = t271;
    int t273 = t269 - t272;
    int t274 = t273;
    int t275 = t274;
    int t276 = t273 - t275;
    int t277 = t271 * 64;
    int t278 = t277;
    int t279 = t274 * 64;
    int t280 = t278 + t279;
    for (uint t281 = 0; t281 < 64; t281++) {
      int t282 = t281;
      int t283 = t280 + t282;
      int t284 = t271 * 64;
      int t285 = t284 + t281;
      float t286 = memory[37172 + t285];
      int t287 = t281 / 64;
      int t288 = t287 * 64;
      int t289 = t281 - t288;
      float t290 = memory[8641 + t289];
      float t291 = t286 * t290;
      float t292 = t270 + t291;
      t270 = t292;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t269] = t270;
    float t295 = memory[25460 + t269];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t296 = t269;
    int t297 = t296;
    int t298 = t269 - t297;
    float t299 = memory[8705 + (int)0.0];
    float t300 = t295 + t299;
    float t301 = t300 * -1.0;
    float t302 = metal::exp(t301);
    float t303 = 1.0 + t302;
    float t304 = 1.0 / t303;
    memory[60852 + t269] = t304;
  }
  #pragma clang diagnostic pop
}



// KERNEL 5
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_5(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(306), value: global(306)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[599948660];
      float t307 = t[0*frameCount + i] + 0.003662333;
      float t308 = metal::select(t307, 0.0, 0.0 > 0.0);
      float t309 = t308;
      float t310 = (t309 * 0.016666668);
      float t311 = metal::floor(t310);
      float t312 = t311 * 60.0;
      float t313 = t308 - t312;
      memory[599948660] = t313;
      float t315 = t313 >= 60.0;
      if (t315) {
        float t317 = t313 - 60.0;
        memory[599948660] = t317;
      }
      if (0.0) {
        memory[599948660] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 6
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_6(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(364), value: global(364)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(344), value: global(344)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(324), value: global(324)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(306) - handled in variable access */
    float t323 = metal::min(t[0*frameCount + id], 59.9999);
    t[1*frameCount + id] = metal::max(t323, 0.0);
    float t325 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t326 = t325 < 0.0;
    float t327 = t325 + 61.0;
    float t328 = metal::select(t325, t327, t326 > 0.0);
    float t329 = t328;
    float t330 = metal::floor(t329);
    float t331 = t329 - t330;
    float t332 = t330 + 1.0;
    float t333 = t332 >= 61.0;
    float t334 = metal::select(t332, 0.0, t333 > 0.0);
    int t335 = (int)t330;
    float t336 = memory[25273 + t335];
    int t337 = (int)t334;
    float t338 = memory[25273 + t337];
    float t339 = 1.0 - t331;
    float t340 = t336 * t339;
    float t341 = t338 * t331;
    float t342 = t340 + t341;
    float t343 = metal::max(t342, 20.0);
    t[2*frameCount + id] = metal::min(t343, 500.0);
    float t345 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t346 = t345 < 0.0;
    float t347 = t345 + 61.0;
    float t348 = metal::select(t345, t347, t346 > 0.0);
    float t349 = t348;
    float t350 = metal::floor(t349);
    float t351 = t349 - t350;
    float t352 = t350 + 1.0;
    float t353 = t352 >= 61.0;
    float t354 = metal::select(t352, 0.0, t353 > 0.0);
    int t355 = (int)t350;
    float t356 = memory[25334 + t355];
    int t357 = (int)t354;
    float t358 = memory[25334 + t357];
    float t359 = 1.0 - t351;
    float t360 = t356 * t359;
    float t361 = t358 * t351;
    float t362 = t360 + t361;
    float t363 = metal::min(t362, 1.0);
    t[3*frameCount + id] = metal::max(t363, 0.0);
  }
  #pragma clang diagnostic pop
}



// KERNEL 7
// Kind: simd
// ThreadCountScale Optional(64)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_7(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5735 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5735)) {
    /* loadGlobal(344) - handled in variable access */
    /* loadGlobal(324) - handled in variable access */
    int t365 = id;
    int t366 = t365 / 64;
    uint _frameIndex = (uint)(t366);
    int t367 = t366 * 64;
    int t368 = t365 - t367;
    float t369 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t370 = t369 < 0.0;
    float t371 = t369 + 61.0;
    float t372 = metal::select(t369, t371, t370 > 0.0);
    float t373 = metal::floor(t372);
    float t374 = t373 + 1.0;
    float t375 = t374 >= 61.0;
    float t376 = metal::select(t374, 0.0, t375 > 0.0);
    float t377 = t372 - t373;
    float t378 = 1.0 - t377;
    float t379 = t366 * 64.0;
    float t380 = (float)t368;
    float t381 = t373 * 64.0;
    float t382 = t381 + t380;
    int t383 = (int)t382;
    float t384 = memory[52788 + t383];
    float t385 = t376 * 64.0;
    float t386 = t385 + t380;
    int t387 = (int)t386;
    float t388 = memory[52788 + t387];
    float t389 = t378 * t384;
    float t390 = t377 * t388;
    float t391 = t389 + t390;
    float t392 = t379 + t380;
    int t393 = (int)t392;
    memory[60916 + t393] = t391;
    int t395 = (int)t392;
    memory[1109492 + t395] = t391;
    float t397 = memory[25395 + t368];
    float t398 = t397 * t[2*frameCount + _frameIndex];
    int t399 = _frameIndex;
    int t400 = t399 * 64;
    int t401 = t400 + t368;
    memory[2158068 + t401] = t398;
  }
  #pragma clang diagnostic pop
}



// KERNEL 8
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_8(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(64)) {
    for (uint i = 0; i < frameCount; i += 1) {
      int t403 = id;
      int t404 = i;
      int t405 = t404 * 64;
      int t406 = t405 + t403;
      float t407 = memory[2158068 + t406];
      float t408 = (t407 * 6.25e-05);
      float t409 = memory[25460 + t403];
      float t410 = t409 + t408;
      float t411 = metal::select(t410, 0.0, 0.0 > 0.0);
      float t412 = metal::floor(t411);
      float t413 = t411 - t412;
      float t414 = t413 >= 1.0;
      float t415 = t413 - 1.0;
      float t416 = metal::select(t413, t415, t414 > 0.0);
      float t417 = metal::select(t416, 0.0, 0.0 > 0.0);
      memory[25460 + t403] = t417;
      int t419 = i;
      int t420 = t419 * 64;
      int t421 = t420 + t403;
      memory[60916 + t421] = t409;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 9
// Kind: simd
// ThreadCountScale Optional(64)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_9(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5736 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5736)) {
    int t423 = id;
    int t424 = t423 / 64;
    uint _frameIndex = (uint)(t424);
    int t425 = t424 * 64;
    int t426 = t423 - t425;
    int t427 = _frameIndex;
    int t428 = t427 * 64;
    int t429 = t428 + t426;
    float t430 = memory[60916 + t429];
    float t431 = t430 * 6.283185;
    float t432 = metal::sin(t431);
    int t433 = _frameIndex;
    int t434 = t433 * 64;
    int t435 = t434 + t426;
    memory[3206644 + t435] = t432;
    int t437 = _frameIndex;
    int t438 = t437 * 64;
    int t439 = t438 + t426;
    float t440 = memory[1109492 + t439];
    float t441 = t432 * t440;
    int t442 = _frameIndex;
    int t443 = t442 * 64;
    int t444 = t443 + t426;
    memory[2158068 + t444] = t441;
  }
  #pragma clang diagnostic pop
}



// KERNEL 10
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_10(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(446), value: global(446)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    t[4*frameCount + id] = 0.0;
    for (uint t447 = 0; t447 < 64; t447++) {
      int t448 = id;
      int t449 = t448 * 64;
      int t450 = t449 + t447;
      float t451 = memory[2158068 + t450];
      float t452 = t[4*frameCount + id] + t451;
      t[4*frameCount + id] = t452;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 11
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_11(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(474), value: global(474)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(473), value: global(473)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(472), value: global(472)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(454), value: global(454)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(446) - handled in variable access */
    /* loadGlobal(364) - handled in variable access */
    /* loadGlobal(324) - handled in variable access */
    t[5*frameCount + id] = t[4*frameCount + id] * t[3*frameCount + id];
    float t455 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t456 = t455 < 0.0;
    float t457 = t455 + 61.0;
    float t458 = metal::select(t455, t457, t456 > 0.0);
    float t459 = t458;
    float t460 = metal::floor(t459);
    float t461 = t459 - t460;
    float t462 = t460 + 1.0;
    float t463 = t462 >= 61.0;
    float t464 = metal::select(t462, 0.0, t463 > 0.0);
    int t465 = (int)t460;
    float t466 = memory[60596 + t465];
    int t467 = (int)t464;
    float t468 = memory[60596 + t467];
    float t469 = 1.0 - t461;
    float t470 = t466 * t469;
    float t471 = t468 * t461;
    t[6*frameCount + id] = t470 + t471;
    t[7*frameCount + id] = t[5*frameCount + id] * t[6*frameCount + id];
    t[8*frameCount + id] = t[7*frameCount + id] * 0.015625;
    float t475 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t476 = t475 < 0.0;
    float t477 = t475 + 61.0;
    float t478 = metal::select(t475, t477, t476 > 0.0);
    float t479 = t478;
    float t480 = metal::floor(t479);
    float t481 = t479 - t480;
    float t482 = t480 + 1.0;
    float t483 = t482 >= 61.0;
    float t484 = metal::select(t482, 0.0, t483 > 0.0);
    int t485 = (int)t480;
    float t486 = memory[60852 + t485];
    int t487 = (int)t484;
    float t488 = memory[60852 + t487];
    float t489 = 1.0 - t481;
    float t490 = t486 * t489;
    float t491 = t488 * t481;
    float t492 = t490 + t491;
  }
  #pragma clang diagnostic pop
}



// KERNEL 12
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_12(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(493), value: global(493)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[9*frameCount + i] = memory[599948661];
      float t494 = t[9*frameCount + i] + 1.0;
      float t495 = metal::select(t494, 0.0, 0.0 > 0.0);
      float t496 = t495;
      float t497 = (t496 * 6.1035156e-05);
      float t498 = metal::floor(t497);
      float t499 = t498 * 16384.0;
      float t500 = t495 - t499;
      memory[599948661] = t500;
      float t502 = t500 >= 16384.0;
      if (t502) {
        float t504 = t500 - 16384.0;
        memory[599948661] = t504;
      }
      if (0.0) {
        memory[599948661] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 13
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_13(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(527), value: global(527)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(493) - handled in variable access */
    float t510 = (t[9*frameCount + id] - metal::floor(t[9*frameCount + id] / 16384.0) * 16384.0);
    float t511 = t510 < 0.0;
    float t512 = t510 + 16384.0;
    float t513 = metal::select(t510, t512, t511 > 0.0);
    float t514 = t513;
    float t515 = metal::floor(t514);
    float t516 = t514 - t515;
    float t517 = t515 + 1.0;
    float t518 = t517 >= 16384.0;
    float t519 = metal::select(t517, 0.0, t518 > 0.0);
    int t520 = (int)t515;
    float t521 = memory[8889 + t520];
    int t522 = (int)t519;
    float t523 = memory[8889 + t522];
    float t524 = 1.0 - t516;
    float t525 = t521 * t524;
    float t526 = t523 * t516;
    t[10*frameCount + id] = t525 + t526;
  }
  #pragma clang diagnostic pop
}



// KERNEL 14
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_14(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(528), value: global(528)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[11*frameCount + i] = memory[599948662];
      float t529 = t[11*frameCount + i] + 1.0;
      float t530 = metal::select(t529, 0.0, 0.0 > 0.0);
      float t531 = t530;
      float t532 = (t531 * 0.0078125);
      float t533 = metal::floor(t532);
      float t534 = t533 * 128.0;
      float t535 = t530 - t534;
      memory[599948662] = t535;
      float t537 = t535 >= 128.0;
      if (t537) {
        float t539 = t535 - 128.0;
        memory[599948662] = t539;
      }
      if (0.0) {
        memory[599948662] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 15
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_15(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(549), value: global(549)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(528) - handled in variable access */
    /* loadGlobal(527) - handled in variable access */
    /* loadGlobal(474) - handled in variable access */
    int t545 = id;
    int t546 = t545 * 1024;
    int t547 = t545 * 257;
    float t548 = t[11*frameCount + id] == 0.0;
    t[12*frameCount + id] = 0.0;
    if (t548) {
      for (uint _pr551 = 0; _pr551 < 512; _pr551++) {
        float t552 = (float)_pr551;
        float t553 = 6.283185 * t552;
        float t554 = (t553 * 0.0019569471);
        float t555 = metal::cos(t554);
        float t556 = 1.0 - t555;
        float t557 = 0.5 * t556;
        float t558 = (float)t545;
        float t559 = t558 - 511.0;
        float t560 = t559 + t552;
        float t561 = (t560 < 0 || t560 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t560];
        float t562 = (t560 < 0 || t560 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t560];
        int t563 = t546 + _pr551;
        float t564 = t561 * t557;
        memory[4255220 + t563] = t564;
        int t566 = t546 + _pr551;
        int t567 = t566 + 512;
        memory[4255220 + t567] = 0.0;
        int t569 = t546 + _pr551;
        float t570 = t562 * t557;
        memory[21032436 + t569] = t570;
        int t572 = t546 + _pr551;
        int t573 = t572 + 512;
        memory[21032436 + t573] = 0.0;
        memory[25460 + (int)_pr551] = t557;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t577 = 0; t577 < 512; t577++) {
        float t578 = (float)t577;
        float t579 = (t578 - metal::floor(t578 / 2.0) * 2.0);
        float t580 = t579;
        float t581 = (t578 * 0.5);
        float t582 = metal::floor(t581);
        float t583 = t580 * 2.0;
        float t584 = (t582 - metal::floor(t582 / 2.0) * 2.0);
        float t585 = t583 + t584;
        float t586 = (t582 * 0.5);
        float t587 = metal::floor(t586);
        float t588 = t585 * 2.0;
        float t589 = (t587 - metal::floor(t587 / 2.0) * 2.0);
        float t590 = t588 + t589;
        float t591 = (t587 * 0.5);
        float t592 = metal::floor(t591);
        float t593 = t590 * 2.0;
        float t594 = (t592 - metal::floor(t592 / 2.0) * 2.0);
        float t595 = t593 + t594;
        float t596 = (t592 * 0.5);
        float t597 = metal::floor(t596);
        float t598 = t595 * 2.0;
        float t599 = (t597 - metal::floor(t597 / 2.0) * 2.0);
        float t600 = t598 + t599;
        float t601 = (t597 * 0.5);
        float t602 = metal::floor(t601);
        float t603 = t600 * 2.0;
        float t604 = (t602 - metal::floor(t602 / 2.0) * 2.0);
        float t605 = t603 + t604;
        float t606 = (t602 * 0.5);
        float t607 = metal::floor(t606);
        float t608 = t605 * 2.0;
        float t609 = (t607 - metal::floor(t607 / 2.0) * 2.0);
        float t610 = t608 + t609;
        float t611 = (t607 * 0.5);
        float t612 = metal::floor(t611);
        float t613 = t610 * 2.0;
        float t614 = (t612 - metal::floor(t612 / 2.0) * 2.0);
        float t615 = t613 + t614;
        float t616 = (t612 * 0.5);
        float t617 = metal::floor(t616);
        float t618 = t615 * 2.0;
        float t619 = (t617 - metal::floor(t617 / 2.0) * 2.0);
        float t620 = t618 + t619;
        float t621 = (t617 * 0.5);
        float t622 = metal::floor(t621);
        float t623 = (float)t577;
        float t624 = t623 < t620;
        int t625 = (int)t620;
        int t626 = t546 + t577;
        float t627 = memory[4255220 + t626];
        int t628 = t546 + t577;
        int t629 = t628 + 512;
        float t630 = memory[4255220 + t629];
        int t631 = t546 + t625;
        float t632 = memory[4255220 + t631];
        int t633 = t546 + t625;
        int t634 = t633 + 512;
        float t635 = memory[4255220 + t634];
        float t636 = metal::select(t627, t632, t624 > 0.0);
        float t637 = metal::select(t630, t635, t624 > 0.0);
        float t638 = metal::select(t632, t627, t624 > 0.0);
        float t639 = metal::select(t635, t630, t624 > 0.0);
        int t640 = t546 + t577;
        memory[4255220 + t640] = t636;
        int t642 = t546 + t577;
        int t643 = t642 + 512;
        memory[4255220 + t643] = t637;
        int t645 = t546 + t625;
        memory[4255220 + t645] = t638;
        int t647 = t546 + t625;
        int t648 = t647 + 512;
        memory[4255220 + t648] = t639;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr651 = 0; _pr651 < 256; _pr651++) {
        float t652 = (float)_pr651;
        float t653 = t652;
        float t654 = metal::floor(t653);
        float t655 = t654;
        float t656 = t652 - t655;
        float t657 = t654 * 2.0;
        float t658 = t657 + t656;
        float t659 = t658 + 1.0;
        float t660 = -6.283185 * t656;
        float t661 = (t660 * 0.5);
        float t662 = metal::cos(t661);
        float t663 = metal::sin(t661);
        int t664 = (int)t658;
        int t665 = (int)t659;
        int t666 = t546 + t664;
        float t667 = memory[4255220 + t666];
        int t668 = t546 + t664;
        int t669 = t668 + 512;
        float t670 = memory[4255220 + t669];
        int t671 = t546 + t665;
        float t672 = memory[4255220 + t671];
        int t673 = t546 + t665;
        int t674 = t673 + 512;
        float t675 = memory[4255220 + t674];
        float t676 = t662 * t672;
        float t677 = t663 * t675;
        float t678 = t676 - t677;
        float t679 = t662 * t675;
        float t680 = t663 * t672;
        float t681 = t679 + t680;
        int t682 = t546 + t664;
        float t683 = t667 + t678;
        memory[4255220 + t682] = t683;
        int t685 = t546 + t664;
        int t686 = t685 + 512;
        float t687 = t670 + t681;
        memory[4255220 + t686] = t687;
        int t689 = t546 + t665;
        float t690 = t667 - t678;
        memory[4255220 + t689] = t690;
        int t692 = t546 + t665;
        int t693 = t692 + 512;
        float t694 = t670 - t681;
        memory[4255220 + t693] = t694;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr697 = 0; _pr697 < 256; _pr697++) {
        float t698 = (float)_pr697;
        float t699 = (t698 * 0.5);
        float t700 = metal::floor(t699);
        float t701 = t700 * 2.0;
        float t702 = t698 - t701;
        float t703 = t700 * 4.0;
        float t704 = t703 + t702;
        float t705 = t704 + 2.0;
        float t706 = -6.283185 * t702;
        float t707 = (t706 * 0.25);
        float t708 = metal::cos(t707);
        float t709 = metal::sin(t707);
        int t710 = (int)t704;
        int t711 = (int)t705;
        int t712 = t546 + t710;
        float t713 = memory[4255220 + t712];
        int t714 = t546 + t710;
        int t715 = t714 + 512;
        float t716 = memory[4255220 + t715];
        int t717 = t546 + t711;
        float t718 = memory[4255220 + t717];
        int t719 = t546 + t711;
        int t720 = t719 + 512;
        float t721 = memory[4255220 + t720];
        float t722 = t708 * t718;
        float t723 = t709 * t721;
        float t724 = t722 - t723;
        float t725 = t708 * t721;
        float t726 = t709 * t718;
        float t727 = t725 + t726;
        int t728 = t546 + t710;
        float t729 = t713 + t724;
        memory[4255220 + t728] = t729;
        int t731 = t546 + t710;
        int t732 = t731 + 512;
        float t733 = t716 + t727;
        memory[4255220 + t732] = t733;
        int t735 = t546 + t711;
        float t736 = t713 - t724;
        memory[4255220 + t735] = t736;
        int t738 = t546 + t711;
        int t739 = t738 + 512;
        float t740 = t716 - t727;
        memory[4255220 + t739] = t740;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr743 = 0; _pr743 < 256; _pr743++) {
        float t744 = (float)_pr743;
        float t745 = (t744 * 0.25);
        float t746 = metal::floor(t745);
        float t747 = t746 * 4.0;
        float t748 = t744 - t747;
        float t749 = t746 * 8.0;
        float t750 = t749 + t748;
        float t751 = t750 + 4.0;
        float t752 = -6.283185 * t748;
        float t753 = (t752 * 0.125);
        float t754 = metal::cos(t753);
        float t755 = metal::sin(t753);
        int t756 = (int)t750;
        int t757 = (int)t751;
        int t758 = t546 + t756;
        float t759 = memory[4255220 + t758];
        int t760 = t546 + t756;
        int t761 = t760 + 512;
        float t762 = memory[4255220 + t761];
        int t763 = t546 + t757;
        float t764 = memory[4255220 + t763];
        int t765 = t546 + t757;
        int t766 = t765 + 512;
        float t767 = memory[4255220 + t766];
        float t768 = t754 * t764;
        float t769 = t755 * t767;
        float t770 = t768 - t769;
        float t771 = t754 * t767;
        float t772 = t755 * t764;
        float t773 = t771 + t772;
        int t774 = t546 + t756;
        float t775 = t759 + t770;
        memory[4255220 + t774] = t775;
        int t777 = t546 + t756;
        int t778 = t777 + 512;
        float t779 = t762 + t773;
        memory[4255220 + t778] = t779;
        int t781 = t546 + t757;
        float t782 = t759 - t770;
        memory[4255220 + t781] = t782;
        int t784 = t546 + t757;
        int t785 = t784 + 512;
        float t786 = t762 - t773;
        memory[4255220 + t785] = t786;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr789 = 0; _pr789 < 256; _pr789++) {
        float t790 = (float)_pr789;
        float t791 = (t790 * 0.125);
        float t792 = metal::floor(t791);
        float t793 = t792 * 8.0;
        float t794 = t790 - t793;
        float t795 = t792 * 16.0;
        float t796 = t795 + t794;
        float t797 = t796 + 8.0;
        float t798 = -6.283185 * t794;
        float t799 = (t798 * 0.0625);
        float t800 = metal::cos(t799);
        float t801 = metal::sin(t799);
        int t802 = (int)t796;
        int t803 = (int)t797;
        int t804 = t546 + t802;
        float t805 = memory[4255220 + t804];
        int t806 = t546 + t802;
        int t807 = t806 + 512;
        float t808 = memory[4255220 + t807];
        int t809 = t546 + t803;
        float t810 = memory[4255220 + t809];
        int t811 = t546 + t803;
        int t812 = t811 + 512;
        float t813 = memory[4255220 + t812];
        float t814 = t800 * t810;
        float t815 = t801 * t813;
        float t816 = t814 - t815;
        float t817 = t800 * t813;
        float t818 = t801 * t810;
        float t819 = t817 + t818;
        int t820 = t546 + t802;
        float t821 = t805 + t816;
        memory[4255220 + t820] = t821;
        int t823 = t546 + t802;
        int t824 = t823 + 512;
        float t825 = t808 + t819;
        memory[4255220 + t824] = t825;
        int t827 = t546 + t803;
        float t828 = t805 - t816;
        memory[4255220 + t827] = t828;
        int t830 = t546 + t803;
        int t831 = t830 + 512;
        float t832 = t808 - t819;
        memory[4255220 + t831] = t832;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr835 = 0; _pr835 < 256; _pr835++) {
        float t836 = (float)_pr835;
        float t837 = (t836 * 0.0625);
        float t838 = metal::floor(t837);
        float t839 = t838 * 16.0;
        float t840 = t836 - t839;
        float t841 = t838 * 32.0;
        float t842 = t841 + t840;
        float t843 = t842 + 16.0;
        float t844 = -6.283185 * t840;
        float t845 = (t844 * 0.03125);
        float t846 = metal::cos(t845);
        float t847 = metal::sin(t845);
        int t848 = (int)t842;
        int t849 = (int)t843;
        int t850 = t546 + t848;
        float t851 = memory[4255220 + t850];
        int t852 = t546 + t848;
        int t853 = t852 + 512;
        float t854 = memory[4255220 + t853];
        int t855 = t546 + t849;
        float t856 = memory[4255220 + t855];
        int t857 = t546 + t849;
        int t858 = t857 + 512;
        float t859 = memory[4255220 + t858];
        float t860 = t846 * t856;
        float t861 = t847 * t859;
        float t862 = t860 - t861;
        float t863 = t846 * t859;
        float t864 = t847 * t856;
        float t865 = t863 + t864;
        int t866 = t546 + t848;
        float t867 = t851 + t862;
        memory[4255220 + t866] = t867;
        int t869 = t546 + t848;
        int t870 = t869 + 512;
        float t871 = t854 + t865;
        memory[4255220 + t870] = t871;
        int t873 = t546 + t849;
        float t874 = t851 - t862;
        memory[4255220 + t873] = t874;
        int t876 = t546 + t849;
        int t877 = t876 + 512;
        float t878 = t854 - t865;
        memory[4255220 + t877] = t878;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr881 = 0; _pr881 < 256; _pr881++) {
        float t882 = (float)_pr881;
        float t883 = (t882 * 0.03125);
        float t884 = metal::floor(t883);
        float t885 = t884 * 32.0;
        float t886 = t882 - t885;
        float t887 = t884 * 64.0;
        float t888 = t887 + t886;
        float t889 = t888 + 32.0;
        float t890 = -6.283185 * t886;
        float t891 = (t890 * 0.015625);
        float t892 = metal::cos(t891);
        float t893 = metal::sin(t891);
        int t894 = (int)t888;
        int t895 = (int)t889;
        int t896 = t546 + t894;
        float t897 = memory[4255220 + t896];
        int t898 = t546 + t894;
        int t899 = t898 + 512;
        float t900 = memory[4255220 + t899];
        int t901 = t546 + t895;
        float t902 = memory[4255220 + t901];
        int t903 = t546 + t895;
        int t904 = t903 + 512;
        float t905 = memory[4255220 + t904];
        float t906 = t892 * t902;
        float t907 = t893 * t905;
        float t908 = t906 - t907;
        float t909 = t892 * t905;
        float t910 = t893 * t902;
        float t911 = t909 + t910;
        int t912 = t546 + t894;
        float t913 = t897 + t908;
        memory[4255220 + t912] = t913;
        int t915 = t546 + t894;
        int t916 = t915 + 512;
        float t917 = t900 + t911;
        memory[4255220 + t916] = t917;
        int t919 = t546 + t895;
        float t920 = t897 - t908;
        memory[4255220 + t919] = t920;
        int t922 = t546 + t895;
        int t923 = t922 + 512;
        float t924 = t900 - t911;
        memory[4255220 + t923] = t924;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr927 = 0; _pr927 < 256; _pr927++) {
        float t928 = (float)_pr927;
        float t929 = (t928 * 0.015625);
        float t930 = metal::floor(t929);
        float t931 = t930 * 64.0;
        float t932 = t928 - t931;
        float t933 = t930 * 128.0;
        float t934 = t933 + t932;
        float t935 = t934 + 64.0;
        float t936 = -6.283185 * t932;
        float t937 = (t936 * 0.0078125);
        float t938 = metal::cos(t937);
        float t939 = metal::sin(t937);
        int t940 = (int)t934;
        int t941 = (int)t935;
        int t942 = t546 + t940;
        float t943 = memory[4255220 + t942];
        int t944 = t546 + t940;
        int t945 = t944 + 512;
        float t946 = memory[4255220 + t945];
        int t947 = t546 + t941;
        float t948 = memory[4255220 + t947];
        int t949 = t546 + t941;
        int t950 = t949 + 512;
        float t951 = memory[4255220 + t950];
        float t952 = t938 * t948;
        float t953 = t939 * t951;
        float t954 = t952 - t953;
        float t955 = t938 * t951;
        float t956 = t939 * t948;
        float t957 = t955 + t956;
        int t958 = t546 + t940;
        float t959 = t943 + t954;
        memory[4255220 + t958] = t959;
        int t961 = t546 + t940;
        int t962 = t961 + 512;
        float t963 = t946 + t957;
        memory[4255220 + t962] = t963;
        int t965 = t546 + t941;
        float t966 = t943 - t954;
        memory[4255220 + t965] = t966;
        int t968 = t546 + t941;
        int t969 = t968 + 512;
        float t970 = t946 - t957;
        memory[4255220 + t969] = t970;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr973 = 0; _pr973 < 256; _pr973++) {
        float t974 = (float)_pr973;
        float t975 = (t974 * 0.0078125);
        float t976 = metal::floor(t975);
        float t977 = t976 * 128.0;
        float t978 = t974 - t977;
        float t979 = t976 * 256.0;
        float t980 = t979 + t978;
        float t981 = t980 + 128.0;
        float t982 = -6.283185 * t978;
        float t983 = (t982 * 0.00390625);
        float t984 = metal::cos(t983);
        float t985 = metal::sin(t983);
        int t986 = (int)t980;
        int t987 = (int)t981;
        int t988 = t546 + t986;
        float t989 = memory[4255220 + t988];
        int t990 = t546 + t986;
        int t991 = t990 + 512;
        float t992 = memory[4255220 + t991];
        int t993 = t546 + t987;
        float t994 = memory[4255220 + t993];
        int t995 = t546 + t987;
        int t996 = t995 + 512;
        float t997 = memory[4255220 + t996];
        float t998 = t984 * t994;
        float t999 = t985 * t997;
        float t1000 = t998 - t999;
        float t1001 = t984 * t997;
        float t1002 = t985 * t994;
        float t1003 = t1001 + t1002;
        int t1004 = t546 + t986;
        float t1005 = t989 + t1000;
        memory[4255220 + t1004] = t1005;
        int t1007 = t546 + t986;
        int t1008 = t1007 + 512;
        float t1009 = t992 + t1003;
        memory[4255220 + t1008] = t1009;
        int t1011 = t546 + t987;
        float t1012 = t989 - t1000;
        memory[4255220 + t1011] = t1012;
        int t1014 = t546 + t987;
        int t1015 = t1014 + 512;
        float t1016 = t992 - t1003;
        memory[4255220 + t1015] = t1016;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1019 = 0; _pr1019 < 256; _pr1019++) {
        float t1020 = (float)_pr1019;
        float t1021 = (t1020 * 0.00390625);
        float t1022 = metal::floor(t1021);
        float t1023 = t1022 * 256.0;
        float t1024 = t1020 - t1023;
        float t1025 = t1022 * 512.0;
        float t1026 = t1025 + t1024;
        float t1027 = t1026 + 256.0;
        float t1028 = -6.283185 * t1024;
        float t1029 = (t1028 * 0.001953125);
        float t1030 = metal::cos(t1029);
        float t1031 = metal::sin(t1029);
        int t1032 = (int)t1026;
        int t1033 = (int)t1027;
        int t1034 = t546 + t1032;
        float t1035 = memory[4255220 + t1034];
        int t1036 = t546 + t1032;
        int t1037 = t1036 + 512;
        float t1038 = memory[4255220 + t1037];
        int t1039 = t546 + t1033;
        float t1040 = memory[4255220 + t1039];
        int t1041 = t546 + t1033;
        int t1042 = t1041 + 512;
        float t1043 = memory[4255220 + t1042];
        float t1044 = t1030 * t1040;
        float t1045 = t1031 * t1043;
        float t1046 = t1044 - t1045;
        float t1047 = t1030 * t1043;
        float t1048 = t1031 * t1040;
        float t1049 = t1047 + t1048;
        int t1050 = t546 + t1032;
        float t1051 = t1035 + t1046;
        memory[4255220 + t1050] = t1051;
        int t1053 = t546 + t1032;
        int t1054 = t1053 + 512;
        float t1055 = t1038 + t1049;
        memory[4255220 + t1054] = t1055;
        int t1057 = t546 + t1033;
        float t1058 = t1035 - t1046;
        memory[4255220 + t1057] = t1058;
        int t1060 = t546 + t1033;
        int t1061 = t1060 + 512;
        float t1062 = t1038 - t1049;
        memory[4255220 + t1061] = t1062;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1065 = 0; t1065 < 512; t1065++) {
        float t1066 = (float)t1065;
        float t1067 = (t1066 - metal::floor(t1066 / 2.0) * 2.0);
        float t1068 = t1067;
        float t1069 = (t1066 * 0.5);
        float t1070 = metal::floor(t1069);
        float t1071 = t1068 * 2.0;
        float t1072 = (t1070 - metal::floor(t1070 / 2.0) * 2.0);
        float t1073 = t1071 + t1072;
        float t1074 = (t1070 * 0.5);
        float t1075 = metal::floor(t1074);
        float t1076 = t1073 * 2.0;
        float t1077 = (t1075 - metal::floor(t1075 / 2.0) * 2.0);
        float t1078 = t1076 + t1077;
        float t1079 = (t1075 * 0.5);
        float t1080 = metal::floor(t1079);
        float t1081 = t1078 * 2.0;
        float t1082 = (t1080 - metal::floor(t1080 / 2.0) * 2.0);
        float t1083 = t1081 + t1082;
        float t1084 = (t1080 * 0.5);
        float t1085 = metal::floor(t1084);
        float t1086 = t1083 * 2.0;
        float t1087 = (t1085 - metal::floor(t1085 / 2.0) * 2.0);
        float t1088 = t1086 + t1087;
        float t1089 = (t1085 * 0.5);
        float t1090 = metal::floor(t1089);
        float t1091 = t1088 * 2.0;
        float t1092 = (t1090 - metal::floor(t1090 / 2.0) * 2.0);
        float t1093 = t1091 + t1092;
        float t1094 = (t1090 * 0.5);
        float t1095 = metal::floor(t1094);
        float t1096 = t1093 * 2.0;
        float t1097 = (t1095 - metal::floor(t1095 / 2.0) * 2.0);
        float t1098 = t1096 + t1097;
        float t1099 = (t1095 * 0.5);
        float t1100 = metal::floor(t1099);
        float t1101 = t1098 * 2.0;
        float t1102 = (t1100 - metal::floor(t1100 / 2.0) * 2.0);
        float t1103 = t1101 + t1102;
        float t1104 = (t1100 * 0.5);
        float t1105 = metal::floor(t1104);
        float t1106 = t1103 * 2.0;
        float t1107 = (t1105 - metal::floor(t1105 / 2.0) * 2.0);
        float t1108 = t1106 + t1107;
        float t1109 = (t1105 * 0.5);
        float t1110 = metal::floor(t1109);
        float t1111 = (float)t1065;
        float t1112 = t1111 < t1108;
        int t1113 = (int)t1108;
        int t1114 = t546 + t1065;
        float t1115 = memory[21032436 + t1114];
        int t1116 = t546 + t1065;
        int t1117 = t1116 + 512;
        float t1118 = memory[21032436 + t1117];
        int t1119 = t546 + t1113;
        float t1120 = memory[21032436 + t1119];
        int t1121 = t546 + t1113;
        int t1122 = t1121 + 512;
        float t1123 = memory[21032436 + t1122];
        float t1124 = metal::select(t1115, t1120, t1112 > 0.0);
        float t1125 = metal::select(t1118, t1123, t1112 > 0.0);
        float t1126 = metal::select(t1120, t1115, t1112 > 0.0);
        float t1127 = metal::select(t1123, t1118, t1112 > 0.0);
        int t1128 = t546 + t1065;
        memory[21032436 + t1128] = t1124;
        int t1130 = t546 + t1065;
        int t1131 = t1130 + 512;
        memory[21032436 + t1131] = t1125;
        int t1133 = t546 + t1113;
        memory[21032436 + t1133] = t1126;
        int t1135 = t546 + t1113;
        int t1136 = t1135 + 512;
        memory[21032436 + t1136] = t1127;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1139 = 0; _pr1139 < 256; _pr1139++) {
        float t1140 = (float)_pr1139;
        float t1141 = t1140;
        float t1142 = metal::floor(t1141);
        float t1143 = t1142;
        float t1144 = t1140 - t1143;
        float t1145 = t1142 * 2.0;
        float t1146 = t1145 + t1144;
        float t1147 = t1146 + 1.0;
        float t1148 = -6.283185 * t1144;
        float t1149 = (t1148 * 0.5);
        float t1150 = metal::cos(t1149);
        float t1151 = metal::sin(t1149);
        int t1152 = (int)t1146;
        int t1153 = (int)t1147;
        int t1154 = t546 + t1152;
        float t1155 = memory[21032436 + t1154];
        int t1156 = t546 + t1152;
        int t1157 = t1156 + 512;
        float t1158 = memory[21032436 + t1157];
        int t1159 = t546 + t1153;
        float t1160 = memory[21032436 + t1159];
        int t1161 = t546 + t1153;
        int t1162 = t1161 + 512;
        float t1163 = memory[21032436 + t1162];
        float t1164 = t1150 * t1160;
        float t1165 = t1151 * t1163;
        float t1166 = t1164 - t1165;
        float t1167 = t1150 * t1163;
        float t1168 = t1151 * t1160;
        float t1169 = t1167 + t1168;
        int t1170 = t546 + t1152;
        float t1171 = t1155 + t1166;
        memory[21032436 + t1170] = t1171;
        int t1173 = t546 + t1152;
        int t1174 = t1173 + 512;
        float t1175 = t1158 + t1169;
        memory[21032436 + t1174] = t1175;
        int t1177 = t546 + t1153;
        float t1178 = t1155 - t1166;
        memory[21032436 + t1177] = t1178;
        int t1180 = t546 + t1153;
        int t1181 = t1180 + 512;
        float t1182 = t1158 - t1169;
        memory[21032436 + t1181] = t1182;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1185 = 0; _pr1185 < 256; _pr1185++) {
        float t1186 = (float)_pr1185;
        float t1187 = (t1186 * 0.5);
        float t1188 = metal::floor(t1187);
        float t1189 = t1188 * 2.0;
        float t1190 = t1186 - t1189;
        float t1191 = t1188 * 4.0;
        float t1192 = t1191 + t1190;
        float t1193 = t1192 + 2.0;
        float t1194 = -6.283185 * t1190;
        float t1195 = (t1194 * 0.25);
        float t1196 = metal::cos(t1195);
        float t1197 = metal::sin(t1195);
        int t1198 = (int)t1192;
        int t1199 = (int)t1193;
        int t1200 = t546 + t1198;
        float t1201 = memory[21032436 + t1200];
        int t1202 = t546 + t1198;
        int t1203 = t1202 + 512;
        float t1204 = memory[21032436 + t1203];
        int t1205 = t546 + t1199;
        float t1206 = memory[21032436 + t1205];
        int t1207 = t546 + t1199;
        int t1208 = t1207 + 512;
        float t1209 = memory[21032436 + t1208];
        float t1210 = t1196 * t1206;
        float t1211 = t1197 * t1209;
        float t1212 = t1210 - t1211;
        float t1213 = t1196 * t1209;
        float t1214 = t1197 * t1206;
        float t1215 = t1213 + t1214;
        int t1216 = t546 + t1198;
        float t1217 = t1201 + t1212;
        memory[21032436 + t1216] = t1217;
        int t1219 = t546 + t1198;
        int t1220 = t1219 + 512;
        float t1221 = t1204 + t1215;
        memory[21032436 + t1220] = t1221;
        int t1223 = t546 + t1199;
        float t1224 = t1201 - t1212;
        memory[21032436 + t1223] = t1224;
        int t1226 = t546 + t1199;
        int t1227 = t1226 + 512;
        float t1228 = t1204 - t1215;
        memory[21032436 + t1227] = t1228;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1231 = 0; _pr1231 < 256; _pr1231++) {
        float t1232 = (float)_pr1231;
        float t1233 = (t1232 * 0.25);
        float t1234 = metal::floor(t1233);
        float t1235 = t1234 * 4.0;
        float t1236 = t1232 - t1235;
        float t1237 = t1234 * 8.0;
        float t1238 = t1237 + t1236;
        float t1239 = t1238 + 4.0;
        float t1240 = -6.283185 * t1236;
        float t1241 = (t1240 * 0.125);
        float t1242 = metal::cos(t1241);
        float t1243 = metal::sin(t1241);
        int t1244 = (int)t1238;
        int t1245 = (int)t1239;
        int t1246 = t546 + t1244;
        float t1247 = memory[21032436 + t1246];
        int t1248 = t546 + t1244;
        int t1249 = t1248 + 512;
        float t1250 = memory[21032436 + t1249];
        int t1251 = t546 + t1245;
        float t1252 = memory[21032436 + t1251];
        int t1253 = t546 + t1245;
        int t1254 = t1253 + 512;
        float t1255 = memory[21032436 + t1254];
        float t1256 = t1242 * t1252;
        float t1257 = t1243 * t1255;
        float t1258 = t1256 - t1257;
        float t1259 = t1242 * t1255;
        float t1260 = t1243 * t1252;
        float t1261 = t1259 + t1260;
        int t1262 = t546 + t1244;
        float t1263 = t1247 + t1258;
        memory[21032436 + t1262] = t1263;
        int t1265 = t546 + t1244;
        int t1266 = t1265 + 512;
        float t1267 = t1250 + t1261;
        memory[21032436 + t1266] = t1267;
        int t1269 = t546 + t1245;
        float t1270 = t1247 - t1258;
        memory[21032436 + t1269] = t1270;
        int t1272 = t546 + t1245;
        int t1273 = t1272 + 512;
        float t1274 = t1250 - t1261;
        memory[21032436 + t1273] = t1274;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1277 = 0; _pr1277 < 256; _pr1277++) {
        float t1278 = (float)_pr1277;
        float t1279 = (t1278 * 0.125);
        float t1280 = metal::floor(t1279);
        float t1281 = t1280 * 8.0;
        float t1282 = t1278 - t1281;
        float t1283 = t1280 * 16.0;
        float t1284 = t1283 + t1282;
        float t1285 = t1284 + 8.0;
        float t1286 = -6.283185 * t1282;
        float t1287 = (t1286 * 0.0625);
        float t1288 = metal::cos(t1287);
        float t1289 = metal::sin(t1287);
        int t1290 = (int)t1284;
        int t1291 = (int)t1285;
        int t1292 = t546 + t1290;
        float t1293 = memory[21032436 + t1292];
        int t1294 = t546 + t1290;
        int t1295 = t1294 + 512;
        float t1296 = memory[21032436 + t1295];
        int t1297 = t546 + t1291;
        float t1298 = memory[21032436 + t1297];
        int t1299 = t546 + t1291;
        int t1300 = t1299 + 512;
        float t1301 = memory[21032436 + t1300];
        float t1302 = t1288 * t1298;
        float t1303 = t1289 * t1301;
        float t1304 = t1302 - t1303;
        float t1305 = t1288 * t1301;
        float t1306 = t1289 * t1298;
        float t1307 = t1305 + t1306;
        int t1308 = t546 + t1290;
        float t1309 = t1293 + t1304;
        memory[21032436 + t1308] = t1309;
        int t1311 = t546 + t1290;
        int t1312 = t1311 + 512;
        float t1313 = t1296 + t1307;
        memory[21032436 + t1312] = t1313;
        int t1315 = t546 + t1291;
        float t1316 = t1293 - t1304;
        memory[21032436 + t1315] = t1316;
        int t1318 = t546 + t1291;
        int t1319 = t1318 + 512;
        float t1320 = t1296 - t1307;
        memory[21032436 + t1319] = t1320;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1323 = 0; _pr1323 < 256; _pr1323++) {
        float t1324 = (float)_pr1323;
        float t1325 = (t1324 * 0.0625);
        float t1326 = metal::floor(t1325);
        float t1327 = t1326 * 16.0;
        float t1328 = t1324 - t1327;
        float t1329 = t1326 * 32.0;
        float t1330 = t1329 + t1328;
        float t1331 = t1330 + 16.0;
        float t1332 = -6.283185 * t1328;
        float t1333 = (t1332 * 0.03125);
        float t1334 = metal::cos(t1333);
        float t1335 = metal::sin(t1333);
        int t1336 = (int)t1330;
        int t1337 = (int)t1331;
        int t1338 = t546 + t1336;
        float t1339 = memory[21032436 + t1338];
        int t1340 = t546 + t1336;
        int t1341 = t1340 + 512;
        float t1342 = memory[21032436 + t1341];
        int t1343 = t546 + t1337;
        float t1344 = memory[21032436 + t1343];
        int t1345 = t546 + t1337;
        int t1346 = t1345 + 512;
        float t1347 = memory[21032436 + t1346];
        float t1348 = t1334 * t1344;
        float t1349 = t1335 * t1347;
        float t1350 = t1348 - t1349;
        float t1351 = t1334 * t1347;
        float t1352 = t1335 * t1344;
        float t1353 = t1351 + t1352;
        int t1354 = t546 + t1336;
        float t1355 = t1339 + t1350;
        memory[21032436 + t1354] = t1355;
        int t1357 = t546 + t1336;
        int t1358 = t1357 + 512;
        float t1359 = t1342 + t1353;
        memory[21032436 + t1358] = t1359;
        int t1361 = t546 + t1337;
        float t1362 = t1339 - t1350;
        memory[21032436 + t1361] = t1362;
        int t1364 = t546 + t1337;
        int t1365 = t1364 + 512;
        float t1366 = t1342 - t1353;
        memory[21032436 + t1365] = t1366;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1369 = 0; _pr1369 < 256; _pr1369++) {
        float t1370 = (float)_pr1369;
        float t1371 = (t1370 * 0.03125);
        float t1372 = metal::floor(t1371);
        float t1373 = t1372 * 32.0;
        float t1374 = t1370 - t1373;
        float t1375 = t1372 * 64.0;
        float t1376 = t1375 + t1374;
        float t1377 = t1376 + 32.0;
        float t1378 = -6.283185 * t1374;
        float t1379 = (t1378 * 0.015625);
        float t1380 = metal::cos(t1379);
        float t1381 = metal::sin(t1379);
        int t1382 = (int)t1376;
        int t1383 = (int)t1377;
        int t1384 = t546 + t1382;
        float t1385 = memory[21032436 + t1384];
        int t1386 = t546 + t1382;
        int t1387 = t1386 + 512;
        float t1388 = memory[21032436 + t1387];
        int t1389 = t546 + t1383;
        float t1390 = memory[21032436 + t1389];
        int t1391 = t546 + t1383;
        int t1392 = t1391 + 512;
        float t1393 = memory[21032436 + t1392];
        float t1394 = t1380 * t1390;
        float t1395 = t1381 * t1393;
        float t1396 = t1394 - t1395;
        float t1397 = t1380 * t1393;
        float t1398 = t1381 * t1390;
        float t1399 = t1397 + t1398;
        int t1400 = t546 + t1382;
        float t1401 = t1385 + t1396;
        memory[21032436 + t1400] = t1401;
        int t1403 = t546 + t1382;
        int t1404 = t1403 + 512;
        float t1405 = t1388 + t1399;
        memory[21032436 + t1404] = t1405;
        int t1407 = t546 + t1383;
        float t1408 = t1385 - t1396;
        memory[21032436 + t1407] = t1408;
        int t1410 = t546 + t1383;
        int t1411 = t1410 + 512;
        float t1412 = t1388 - t1399;
        memory[21032436 + t1411] = t1412;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1415 = 0; _pr1415 < 256; _pr1415++) {
        float t1416 = (float)_pr1415;
        float t1417 = (t1416 * 0.015625);
        float t1418 = metal::floor(t1417);
        float t1419 = t1418 * 64.0;
        float t1420 = t1416 - t1419;
        float t1421 = t1418 * 128.0;
        float t1422 = t1421 + t1420;
        float t1423 = t1422 + 64.0;
        float t1424 = -6.283185 * t1420;
        float t1425 = (t1424 * 0.0078125);
        float t1426 = metal::cos(t1425);
        float t1427 = metal::sin(t1425);
        int t1428 = (int)t1422;
        int t1429 = (int)t1423;
        int t1430 = t546 + t1428;
        float t1431 = memory[21032436 + t1430];
        int t1432 = t546 + t1428;
        int t1433 = t1432 + 512;
        float t1434 = memory[21032436 + t1433];
        int t1435 = t546 + t1429;
        float t1436 = memory[21032436 + t1435];
        int t1437 = t546 + t1429;
        int t1438 = t1437 + 512;
        float t1439 = memory[21032436 + t1438];
        float t1440 = t1426 * t1436;
        float t1441 = t1427 * t1439;
        float t1442 = t1440 - t1441;
        float t1443 = t1426 * t1439;
        float t1444 = t1427 * t1436;
        float t1445 = t1443 + t1444;
        int t1446 = t546 + t1428;
        float t1447 = t1431 + t1442;
        memory[21032436 + t1446] = t1447;
        int t1449 = t546 + t1428;
        int t1450 = t1449 + 512;
        float t1451 = t1434 + t1445;
        memory[21032436 + t1450] = t1451;
        int t1453 = t546 + t1429;
        float t1454 = t1431 - t1442;
        memory[21032436 + t1453] = t1454;
        int t1456 = t546 + t1429;
        int t1457 = t1456 + 512;
        float t1458 = t1434 - t1445;
        memory[21032436 + t1457] = t1458;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1461 = 0; _pr1461 < 256; _pr1461++) {
        float t1462 = (float)_pr1461;
        float t1463 = (t1462 * 0.0078125);
        float t1464 = metal::floor(t1463);
        float t1465 = t1464 * 128.0;
        float t1466 = t1462 - t1465;
        float t1467 = t1464 * 256.0;
        float t1468 = t1467 + t1466;
        float t1469 = t1468 + 128.0;
        float t1470 = -6.283185 * t1466;
        float t1471 = (t1470 * 0.00390625);
        float t1472 = metal::cos(t1471);
        float t1473 = metal::sin(t1471);
        int t1474 = (int)t1468;
        int t1475 = (int)t1469;
        int t1476 = t546 + t1474;
        float t1477 = memory[21032436 + t1476];
        int t1478 = t546 + t1474;
        int t1479 = t1478 + 512;
        float t1480 = memory[21032436 + t1479];
        int t1481 = t546 + t1475;
        float t1482 = memory[21032436 + t1481];
        int t1483 = t546 + t1475;
        int t1484 = t1483 + 512;
        float t1485 = memory[21032436 + t1484];
        float t1486 = t1472 * t1482;
        float t1487 = t1473 * t1485;
        float t1488 = t1486 - t1487;
        float t1489 = t1472 * t1485;
        float t1490 = t1473 * t1482;
        float t1491 = t1489 + t1490;
        int t1492 = t546 + t1474;
        float t1493 = t1477 + t1488;
        memory[21032436 + t1492] = t1493;
        int t1495 = t546 + t1474;
        int t1496 = t1495 + 512;
        float t1497 = t1480 + t1491;
        memory[21032436 + t1496] = t1497;
        int t1499 = t546 + t1475;
        float t1500 = t1477 - t1488;
        memory[21032436 + t1499] = t1500;
        int t1502 = t546 + t1475;
        int t1503 = t1502 + 512;
        float t1504 = t1480 - t1491;
        memory[21032436 + t1503] = t1504;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1507 = 0; _pr1507 < 256; _pr1507++) {
        float t1508 = (float)_pr1507;
        float t1509 = (t1508 * 0.00390625);
        float t1510 = metal::floor(t1509);
        float t1511 = t1510 * 256.0;
        float t1512 = t1508 - t1511;
        float t1513 = t1510 * 512.0;
        float t1514 = t1513 + t1512;
        float t1515 = t1514 + 256.0;
        float t1516 = -6.283185 * t1512;
        float t1517 = (t1516 * 0.001953125);
        float t1518 = metal::cos(t1517);
        float t1519 = metal::sin(t1517);
        int t1520 = (int)t1514;
        int t1521 = (int)t1515;
        int t1522 = t546 + t1520;
        float t1523 = memory[21032436 + t1522];
        int t1524 = t546 + t1520;
        int t1525 = t1524 + 512;
        float t1526 = memory[21032436 + t1525];
        int t1527 = t546 + t1521;
        float t1528 = memory[21032436 + t1527];
        int t1529 = t546 + t1521;
        int t1530 = t1529 + 512;
        float t1531 = memory[21032436 + t1530];
        float t1532 = t1518 * t1528;
        float t1533 = t1519 * t1531;
        float t1534 = t1532 - t1533;
        float t1535 = t1518 * t1531;
        float t1536 = t1519 * t1528;
        float t1537 = t1535 + t1536;
        int t1538 = t546 + t1520;
        float t1539 = t1523 + t1534;
        memory[21032436 + t1538] = t1539;
        int t1541 = t546 + t1520;
        int t1542 = t1541 + 512;
        float t1543 = t1526 + t1537;
        memory[21032436 + t1542] = t1543;
        int t1545 = t546 + t1521;
        float t1546 = t1523 - t1534;
        memory[21032436 + t1545] = t1546;
        int t1548 = t546 + t1521;
        int t1549 = t1548 + 512;
        float t1550 = t1526 - t1537;
        memory[21032436 + t1549] = t1550;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1553 = 0; _pr1553 < 257; _pr1553++) {
        int t1554 = t546 + _pr1553;
        float t1555 = memory[4255220 + t1554];
        int t1556 = t546 + _pr1553;
        int t1557 = t1556 + 512;
        float t1558 = memory[4255220 + t1557];
        float t1559 = t1555 * t1555;
        float t1560 = t1558 * t1558;
        float t1561 = t1559 + t1560;
        float t1562 = metal::sqrt(t1561);
        int t1563 = t547 + _pr1553;
        memory[37809652 + t1563] = t1562;
        int t1565 = t546 + _pr1553;
        float t1566 = memory[21032436 + t1565];
        int t1567 = t546 + _pr1553;
        int t1568 = t1567 + 512;
        float t1569 = memory[21032436 + t1568];
        float t1570 = t1566 * t1566;
        float t1571 = t1569 * t1569;
        float t1572 = t1570 + t1571;
        float t1573 = metal::sqrt(t1572);
        int t1574 = t547 + _pr1553;
        memory[42020340 + t1574] = t1573;
        float t1576 = t1562 - t1573;
        int t1577 = t547 + _pr1553;
        float t1578 = t1576 * t1576;
        memory[46231028 + t1577] = t1578;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1581 = 0; t1581 < 257; t1581++) {
        int t1582 = t547 + t1581;
        float t1583 = memory[46231028 + t1582];
        float t1584 = t[12*frameCount + id] + t1583;
        t[12*frameCount + id] = t1584;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 16
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_16(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1588), value: global(1588)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(549) - handled in variable access */
    float t1587 = (t[12*frameCount + id] * 6.1035156e-05);
    t[13*frameCount + id] = t1587;
  }
  #pragma clang diagnostic pop
}



// KERNEL 17
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_17(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1589), value: global(1589)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[14*frameCount + i] = memory[599948663];
      float t1590 = t[14*frameCount + i] + 1.0;
      float t1591 = metal::select(t1590, 0.0, 0.0 > 0.0);
      float t1592 = t1591;
      float t1593 = (t1592 * 0.00390625);
      float t1594 = metal::floor(t1593);
      float t1595 = t1594 * 256.0;
      float t1596 = t1591 - t1595;
      memory[599948663] = t1596;
      float t1598 = t1596 >= 256.0;
      if (t1598) {
        float t1600 = t1596 - 256.0;
        memory[599948663] = t1600;
      }
      if (0.0) {
        memory[599948663] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 18
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_18(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1610), value: global(1610)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1589) - handled in variable access */
    /* loadGlobal(527) - handled in variable access */
    /* loadGlobal(474) - handled in variable access */
    int t1606 = id;
    int t1607 = t1606 * 2048;
    int t1608 = t1606 * 513;
    float t1609 = t[14*frameCount + id] == 0.0;
    t[15*frameCount + id] = 0.0;
    if (t1609) {
      for (uint _pr1612 = 0; _pr1612 < 1024; _pr1612++) {
        float t1613 = (float)_pr1612;
        float t1614 = 6.283185 * t1613;
        float t1615 = (t1614 * 0.0009775171);
        float t1616 = metal::cos(t1615);
        float t1617 = 1.0 - t1616;
        float t1618 = 0.5 * t1617;
        float t1619 = (float)t1606;
        float t1620 = t1619 - 1023.0;
        float t1621 = t1620 + t1613;
        float t1622 = (t1621 < 0 || t1621 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t1621];
        float t1623 = (t1621 < 0 || t1621 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t1621];
        int t1624 = t1607 + _pr1612;
        float t1625 = t1622 * t1618;
        memory[50441716 + t1624] = t1625;
        int t1627 = t1607 + _pr1612;
        int t1628 = t1627 + 1024;
        memory[50441716 + t1628] = 0.0;
        int t1630 = t1607 + _pr1612;
        float t1631 = t1623 * t1618;
        memory[83996148 + t1630] = t1631;
        int t1633 = t1607 + _pr1612;
        int t1634 = t1633 + 1024;
        memory[83996148 + t1634] = 0.0;
        memory[52788 + (int)_pr1612] = t1618;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1638 = 0; t1638 < 1024; t1638++) {
        float t1639 = (float)t1638;
        float t1640 = (t1639 - metal::floor(t1639 / 2.0) * 2.0);
        float t1641 = t1640;
        float t1642 = (t1639 * 0.5);
        float t1643 = metal::floor(t1642);
        float t1644 = t1641 * 2.0;
        float t1645 = (t1643 - metal::floor(t1643 / 2.0) * 2.0);
        float t1646 = t1644 + t1645;
        float t1647 = (t1643 * 0.5);
        float t1648 = metal::floor(t1647);
        float t1649 = t1646 * 2.0;
        float t1650 = (t1648 - metal::floor(t1648 / 2.0) * 2.0);
        float t1651 = t1649 + t1650;
        float t1652 = (t1648 * 0.5);
        float t1653 = metal::floor(t1652);
        float t1654 = t1651 * 2.0;
        float t1655 = (t1653 - metal::floor(t1653 / 2.0) * 2.0);
        float t1656 = t1654 + t1655;
        float t1657 = (t1653 * 0.5);
        float t1658 = metal::floor(t1657);
        float t1659 = t1656 * 2.0;
        float t1660 = (t1658 - metal::floor(t1658 / 2.0) * 2.0);
        float t1661 = t1659 + t1660;
        float t1662 = (t1658 * 0.5);
        float t1663 = metal::floor(t1662);
        float t1664 = t1661 * 2.0;
        float t1665 = (t1663 - metal::floor(t1663 / 2.0) * 2.0);
        float t1666 = t1664 + t1665;
        float t1667 = (t1663 * 0.5);
        float t1668 = metal::floor(t1667);
        float t1669 = t1666 * 2.0;
        float t1670 = (t1668 - metal::floor(t1668 / 2.0) * 2.0);
        float t1671 = t1669 + t1670;
        float t1672 = (t1668 * 0.5);
        float t1673 = metal::floor(t1672);
        float t1674 = t1671 * 2.0;
        float t1675 = (t1673 - metal::floor(t1673 / 2.0) * 2.0);
        float t1676 = t1674 + t1675;
        float t1677 = (t1673 * 0.5);
        float t1678 = metal::floor(t1677);
        float t1679 = t1676 * 2.0;
        float t1680 = (t1678 - metal::floor(t1678 / 2.0) * 2.0);
        float t1681 = t1679 + t1680;
        float t1682 = (t1678 * 0.5);
        float t1683 = metal::floor(t1682);
        float t1684 = t1681 * 2.0;
        float t1685 = (t1683 - metal::floor(t1683 / 2.0) * 2.0);
        float t1686 = t1684 + t1685;
        float t1687 = (t1683 * 0.5);
        float t1688 = metal::floor(t1687);
        float t1689 = (float)t1638;
        float t1690 = t1689 < t1686;
        int t1691 = (int)t1686;
        int t1692 = t1607 + t1638;
        float t1693 = memory[50441716 + t1692];
        int t1694 = t1607 + t1638;
        int t1695 = t1694 + 1024;
        float t1696 = memory[50441716 + t1695];
        int t1697 = t1607 + t1691;
        float t1698 = memory[50441716 + t1697];
        int t1699 = t1607 + t1691;
        int t1700 = t1699 + 1024;
        float t1701 = memory[50441716 + t1700];
        float t1702 = metal::select(t1693, t1698, t1690 > 0.0);
        float t1703 = metal::select(t1696, t1701, t1690 > 0.0);
        float t1704 = metal::select(t1698, t1693, t1690 > 0.0);
        float t1705 = metal::select(t1701, t1696, t1690 > 0.0);
        int t1706 = t1607 + t1638;
        memory[50441716 + t1706] = t1702;
        int t1708 = t1607 + t1638;
        int t1709 = t1708 + 1024;
        memory[50441716 + t1709] = t1703;
        int t1711 = t1607 + t1691;
        memory[50441716 + t1711] = t1704;
        int t1713 = t1607 + t1691;
        int t1714 = t1713 + 1024;
        memory[50441716 + t1714] = t1705;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1717 = 0; _pr1717 < 512; _pr1717++) {
        float t1718 = (float)_pr1717;
        float t1719 = t1718;
        float t1720 = metal::floor(t1719);
        float t1721 = t1720;
        float t1722 = t1718 - t1721;
        float t1723 = t1720 * 2.0;
        float t1724 = t1723 + t1722;
        float t1725 = t1724 + 1.0;
        float t1726 = -6.283185 * t1722;
        float t1727 = (t1726 * 0.5);
        float t1728 = metal::cos(t1727);
        float t1729 = metal::sin(t1727);
        int t1730 = (int)t1724;
        int t1731 = (int)t1725;
        int t1732 = t1607 + t1730;
        float t1733 = memory[50441716 + t1732];
        int t1734 = t1607 + t1730;
        int t1735 = t1734 + 1024;
        float t1736 = memory[50441716 + t1735];
        int t1737 = t1607 + t1731;
        float t1738 = memory[50441716 + t1737];
        int t1739 = t1607 + t1731;
        int t1740 = t1739 + 1024;
        float t1741 = memory[50441716 + t1740];
        float t1742 = t1728 * t1738;
        float t1743 = t1729 * t1741;
        float t1744 = t1742 - t1743;
        float t1745 = t1728 * t1741;
        float t1746 = t1729 * t1738;
        float t1747 = t1745 + t1746;
        int t1748 = t1607 + t1730;
        float t1749 = t1733 + t1744;
        memory[50441716 + t1748] = t1749;
        int t1751 = t1607 + t1730;
        int t1752 = t1751 + 1024;
        float t1753 = t1736 + t1747;
        memory[50441716 + t1752] = t1753;
        int t1755 = t1607 + t1731;
        float t1756 = t1733 - t1744;
        memory[50441716 + t1755] = t1756;
        int t1758 = t1607 + t1731;
        int t1759 = t1758 + 1024;
        float t1760 = t1736 - t1747;
        memory[50441716 + t1759] = t1760;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1763 = 0; _pr1763 < 512; _pr1763++) {
        float t1764 = (float)_pr1763;
        float t1765 = (t1764 * 0.5);
        float t1766 = metal::floor(t1765);
        float t1767 = t1766 * 2.0;
        float t1768 = t1764 - t1767;
        float t1769 = t1766 * 4.0;
        float t1770 = t1769 + t1768;
        float t1771 = t1770 + 2.0;
        float t1772 = -6.283185 * t1768;
        float t1773 = (t1772 * 0.25);
        float t1774 = metal::cos(t1773);
        float t1775 = metal::sin(t1773);
        int t1776 = (int)t1770;
        int t1777 = (int)t1771;
        int t1778 = t1607 + t1776;
        float t1779 = memory[50441716 + t1778];
        int t1780 = t1607 + t1776;
        int t1781 = t1780 + 1024;
        float t1782 = memory[50441716 + t1781];
        int t1783 = t1607 + t1777;
        float t1784 = memory[50441716 + t1783];
        int t1785 = t1607 + t1777;
        int t1786 = t1785 + 1024;
        float t1787 = memory[50441716 + t1786];
        float t1788 = t1774 * t1784;
        float t1789 = t1775 * t1787;
        float t1790 = t1788 - t1789;
        float t1791 = t1774 * t1787;
        float t1792 = t1775 * t1784;
        float t1793 = t1791 + t1792;
        int t1794 = t1607 + t1776;
        float t1795 = t1779 + t1790;
        memory[50441716 + t1794] = t1795;
        int t1797 = t1607 + t1776;
        int t1798 = t1797 + 1024;
        float t1799 = t1782 + t1793;
        memory[50441716 + t1798] = t1799;
        int t1801 = t1607 + t1777;
        float t1802 = t1779 - t1790;
        memory[50441716 + t1801] = t1802;
        int t1804 = t1607 + t1777;
        int t1805 = t1804 + 1024;
        float t1806 = t1782 - t1793;
        memory[50441716 + t1805] = t1806;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1809 = 0; _pr1809 < 512; _pr1809++) {
        float t1810 = (float)_pr1809;
        float t1811 = (t1810 * 0.25);
        float t1812 = metal::floor(t1811);
        float t1813 = t1812 * 4.0;
        float t1814 = t1810 - t1813;
        float t1815 = t1812 * 8.0;
        float t1816 = t1815 + t1814;
        float t1817 = t1816 + 4.0;
        float t1818 = -6.283185 * t1814;
        float t1819 = (t1818 * 0.125);
        float t1820 = metal::cos(t1819);
        float t1821 = metal::sin(t1819);
        int t1822 = (int)t1816;
        int t1823 = (int)t1817;
        int t1824 = t1607 + t1822;
        float t1825 = memory[50441716 + t1824];
        int t1826 = t1607 + t1822;
        int t1827 = t1826 + 1024;
        float t1828 = memory[50441716 + t1827];
        int t1829 = t1607 + t1823;
        float t1830 = memory[50441716 + t1829];
        int t1831 = t1607 + t1823;
        int t1832 = t1831 + 1024;
        float t1833 = memory[50441716 + t1832];
        float t1834 = t1820 * t1830;
        float t1835 = t1821 * t1833;
        float t1836 = t1834 - t1835;
        float t1837 = t1820 * t1833;
        float t1838 = t1821 * t1830;
        float t1839 = t1837 + t1838;
        int t1840 = t1607 + t1822;
        float t1841 = t1825 + t1836;
        memory[50441716 + t1840] = t1841;
        int t1843 = t1607 + t1822;
        int t1844 = t1843 + 1024;
        float t1845 = t1828 + t1839;
        memory[50441716 + t1844] = t1845;
        int t1847 = t1607 + t1823;
        float t1848 = t1825 - t1836;
        memory[50441716 + t1847] = t1848;
        int t1850 = t1607 + t1823;
        int t1851 = t1850 + 1024;
        float t1852 = t1828 - t1839;
        memory[50441716 + t1851] = t1852;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1855 = 0; _pr1855 < 512; _pr1855++) {
        float t1856 = (float)_pr1855;
        float t1857 = (t1856 * 0.125);
        float t1858 = metal::floor(t1857);
        float t1859 = t1858 * 8.0;
        float t1860 = t1856 - t1859;
        float t1861 = t1858 * 16.0;
        float t1862 = t1861 + t1860;
        float t1863 = t1862 + 8.0;
        float t1864 = -6.283185 * t1860;
        float t1865 = (t1864 * 0.0625);
        float t1866 = metal::cos(t1865);
        float t1867 = metal::sin(t1865);
        int t1868 = (int)t1862;
        int t1869 = (int)t1863;
        int t1870 = t1607 + t1868;
        float t1871 = memory[50441716 + t1870];
        int t1872 = t1607 + t1868;
        int t1873 = t1872 + 1024;
        float t1874 = memory[50441716 + t1873];
        int t1875 = t1607 + t1869;
        float t1876 = memory[50441716 + t1875];
        int t1877 = t1607 + t1869;
        int t1878 = t1877 + 1024;
        float t1879 = memory[50441716 + t1878];
        float t1880 = t1866 * t1876;
        float t1881 = t1867 * t1879;
        float t1882 = t1880 - t1881;
        float t1883 = t1866 * t1879;
        float t1884 = t1867 * t1876;
        float t1885 = t1883 + t1884;
        int t1886 = t1607 + t1868;
        float t1887 = t1871 + t1882;
        memory[50441716 + t1886] = t1887;
        int t1889 = t1607 + t1868;
        int t1890 = t1889 + 1024;
        float t1891 = t1874 + t1885;
        memory[50441716 + t1890] = t1891;
        int t1893 = t1607 + t1869;
        float t1894 = t1871 - t1882;
        memory[50441716 + t1893] = t1894;
        int t1896 = t1607 + t1869;
        int t1897 = t1896 + 1024;
        float t1898 = t1874 - t1885;
        memory[50441716 + t1897] = t1898;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1901 = 0; _pr1901 < 512; _pr1901++) {
        float t1902 = (float)_pr1901;
        float t1903 = (t1902 * 0.0625);
        float t1904 = metal::floor(t1903);
        float t1905 = t1904 * 16.0;
        float t1906 = t1902 - t1905;
        float t1907 = t1904 * 32.0;
        float t1908 = t1907 + t1906;
        float t1909 = t1908 + 16.0;
        float t1910 = -6.283185 * t1906;
        float t1911 = (t1910 * 0.03125);
        float t1912 = metal::cos(t1911);
        float t1913 = metal::sin(t1911);
        int t1914 = (int)t1908;
        int t1915 = (int)t1909;
        int t1916 = t1607 + t1914;
        float t1917 = memory[50441716 + t1916];
        int t1918 = t1607 + t1914;
        int t1919 = t1918 + 1024;
        float t1920 = memory[50441716 + t1919];
        int t1921 = t1607 + t1915;
        float t1922 = memory[50441716 + t1921];
        int t1923 = t1607 + t1915;
        int t1924 = t1923 + 1024;
        float t1925 = memory[50441716 + t1924];
        float t1926 = t1912 * t1922;
        float t1927 = t1913 * t1925;
        float t1928 = t1926 - t1927;
        float t1929 = t1912 * t1925;
        float t1930 = t1913 * t1922;
        float t1931 = t1929 + t1930;
        int t1932 = t1607 + t1914;
        float t1933 = t1917 + t1928;
        memory[50441716 + t1932] = t1933;
        int t1935 = t1607 + t1914;
        int t1936 = t1935 + 1024;
        float t1937 = t1920 + t1931;
        memory[50441716 + t1936] = t1937;
        int t1939 = t1607 + t1915;
        float t1940 = t1917 - t1928;
        memory[50441716 + t1939] = t1940;
        int t1942 = t1607 + t1915;
        int t1943 = t1942 + 1024;
        float t1944 = t1920 - t1931;
        memory[50441716 + t1943] = t1944;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1947 = 0; _pr1947 < 512; _pr1947++) {
        float t1948 = (float)_pr1947;
        float t1949 = (t1948 * 0.03125);
        float t1950 = metal::floor(t1949);
        float t1951 = t1950 * 32.0;
        float t1952 = t1948 - t1951;
        float t1953 = t1950 * 64.0;
        float t1954 = t1953 + t1952;
        float t1955 = t1954 + 32.0;
        float t1956 = -6.283185 * t1952;
        float t1957 = (t1956 * 0.015625);
        float t1958 = metal::cos(t1957);
        float t1959 = metal::sin(t1957);
        int t1960 = (int)t1954;
        int t1961 = (int)t1955;
        int t1962 = t1607 + t1960;
        float t1963 = memory[50441716 + t1962];
        int t1964 = t1607 + t1960;
        int t1965 = t1964 + 1024;
        float t1966 = memory[50441716 + t1965];
        int t1967 = t1607 + t1961;
        float t1968 = memory[50441716 + t1967];
        int t1969 = t1607 + t1961;
        int t1970 = t1969 + 1024;
        float t1971 = memory[50441716 + t1970];
        float t1972 = t1958 * t1968;
        float t1973 = t1959 * t1971;
        float t1974 = t1972 - t1973;
        float t1975 = t1958 * t1971;
        float t1976 = t1959 * t1968;
        float t1977 = t1975 + t1976;
        int t1978 = t1607 + t1960;
        float t1979 = t1963 + t1974;
        memory[50441716 + t1978] = t1979;
        int t1981 = t1607 + t1960;
        int t1982 = t1981 + 1024;
        float t1983 = t1966 + t1977;
        memory[50441716 + t1982] = t1983;
        int t1985 = t1607 + t1961;
        float t1986 = t1963 - t1974;
        memory[50441716 + t1985] = t1986;
        int t1988 = t1607 + t1961;
        int t1989 = t1988 + 1024;
        float t1990 = t1966 - t1977;
        memory[50441716 + t1989] = t1990;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1993 = 0; _pr1993 < 512; _pr1993++) {
        float t1994 = (float)_pr1993;
        float t1995 = (t1994 * 0.015625);
        float t1996 = metal::floor(t1995);
        float t1997 = t1996 * 64.0;
        float t1998 = t1994 - t1997;
        float t1999 = t1996 * 128.0;
        float t2000 = t1999 + t1998;
        float t2001 = t2000 + 64.0;
        float t2002 = -6.283185 * t1998;
        float t2003 = (t2002 * 0.0078125);
        float t2004 = metal::cos(t2003);
        float t2005 = metal::sin(t2003);
        int t2006 = (int)t2000;
        int t2007 = (int)t2001;
        int t2008 = t1607 + t2006;
        float t2009 = memory[50441716 + t2008];
        int t2010 = t1607 + t2006;
        int t2011 = t2010 + 1024;
        float t2012 = memory[50441716 + t2011];
        int t2013 = t1607 + t2007;
        float t2014 = memory[50441716 + t2013];
        int t2015 = t1607 + t2007;
        int t2016 = t2015 + 1024;
        float t2017 = memory[50441716 + t2016];
        float t2018 = t2004 * t2014;
        float t2019 = t2005 * t2017;
        float t2020 = t2018 - t2019;
        float t2021 = t2004 * t2017;
        float t2022 = t2005 * t2014;
        float t2023 = t2021 + t2022;
        int t2024 = t1607 + t2006;
        float t2025 = t2009 + t2020;
        memory[50441716 + t2024] = t2025;
        int t2027 = t1607 + t2006;
        int t2028 = t2027 + 1024;
        float t2029 = t2012 + t2023;
        memory[50441716 + t2028] = t2029;
        int t2031 = t1607 + t2007;
        float t2032 = t2009 - t2020;
        memory[50441716 + t2031] = t2032;
        int t2034 = t1607 + t2007;
        int t2035 = t2034 + 1024;
        float t2036 = t2012 - t2023;
        memory[50441716 + t2035] = t2036;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2039 = 0; _pr2039 < 512; _pr2039++) {
        float t2040 = (float)_pr2039;
        float t2041 = (t2040 * 0.0078125);
        float t2042 = metal::floor(t2041);
        float t2043 = t2042 * 128.0;
        float t2044 = t2040 - t2043;
        float t2045 = t2042 * 256.0;
        float t2046 = t2045 + t2044;
        float t2047 = t2046 + 128.0;
        float t2048 = -6.283185 * t2044;
        float t2049 = (t2048 * 0.00390625);
        float t2050 = metal::cos(t2049);
        float t2051 = metal::sin(t2049);
        int t2052 = (int)t2046;
        int t2053 = (int)t2047;
        int t2054 = t1607 + t2052;
        float t2055 = memory[50441716 + t2054];
        int t2056 = t1607 + t2052;
        int t2057 = t2056 + 1024;
        float t2058 = memory[50441716 + t2057];
        int t2059 = t1607 + t2053;
        float t2060 = memory[50441716 + t2059];
        int t2061 = t1607 + t2053;
        int t2062 = t2061 + 1024;
        float t2063 = memory[50441716 + t2062];
        float t2064 = t2050 * t2060;
        float t2065 = t2051 * t2063;
        float t2066 = t2064 - t2065;
        float t2067 = t2050 * t2063;
        float t2068 = t2051 * t2060;
        float t2069 = t2067 + t2068;
        int t2070 = t1607 + t2052;
        float t2071 = t2055 + t2066;
        memory[50441716 + t2070] = t2071;
        int t2073 = t1607 + t2052;
        int t2074 = t2073 + 1024;
        float t2075 = t2058 + t2069;
        memory[50441716 + t2074] = t2075;
        int t2077 = t1607 + t2053;
        float t2078 = t2055 - t2066;
        memory[50441716 + t2077] = t2078;
        int t2080 = t1607 + t2053;
        int t2081 = t2080 + 1024;
        float t2082 = t2058 - t2069;
        memory[50441716 + t2081] = t2082;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2085 = 0; _pr2085 < 512; _pr2085++) {
        float t2086 = (float)_pr2085;
        float t2087 = (t2086 * 0.00390625);
        float t2088 = metal::floor(t2087);
        float t2089 = t2088 * 256.0;
        float t2090 = t2086 - t2089;
        float t2091 = t2088 * 512.0;
        float t2092 = t2091 + t2090;
        float t2093 = t2092 + 256.0;
        float t2094 = -6.283185 * t2090;
        float t2095 = (t2094 * 0.001953125);
        float t2096 = metal::cos(t2095);
        float t2097 = metal::sin(t2095);
        int t2098 = (int)t2092;
        int t2099 = (int)t2093;
        int t2100 = t1607 + t2098;
        float t2101 = memory[50441716 + t2100];
        int t2102 = t1607 + t2098;
        int t2103 = t2102 + 1024;
        float t2104 = memory[50441716 + t2103];
        int t2105 = t1607 + t2099;
        float t2106 = memory[50441716 + t2105];
        int t2107 = t1607 + t2099;
        int t2108 = t2107 + 1024;
        float t2109 = memory[50441716 + t2108];
        float t2110 = t2096 * t2106;
        float t2111 = t2097 * t2109;
        float t2112 = t2110 - t2111;
        float t2113 = t2096 * t2109;
        float t2114 = t2097 * t2106;
        float t2115 = t2113 + t2114;
        int t2116 = t1607 + t2098;
        float t2117 = t2101 + t2112;
        memory[50441716 + t2116] = t2117;
        int t2119 = t1607 + t2098;
        int t2120 = t2119 + 1024;
        float t2121 = t2104 + t2115;
        memory[50441716 + t2120] = t2121;
        int t2123 = t1607 + t2099;
        float t2124 = t2101 - t2112;
        memory[50441716 + t2123] = t2124;
        int t2126 = t1607 + t2099;
        int t2127 = t2126 + 1024;
        float t2128 = t2104 - t2115;
        memory[50441716 + t2127] = t2128;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2131 = 0; _pr2131 < 512; _pr2131++) {
        float t2132 = (float)_pr2131;
        float t2133 = (t2132 * 0.001953125);
        float t2134 = metal::floor(t2133);
        float t2135 = t2134 * 512.0;
        float t2136 = t2132 - t2135;
        float t2137 = t2134 * 1024.0;
        float t2138 = t2137 + t2136;
        float t2139 = t2138 + 512.0;
        float t2140 = -6.283185 * t2136;
        float t2141 = (t2140 * 0.0009765625);
        float t2142 = metal::cos(t2141);
        float t2143 = metal::sin(t2141);
        int t2144 = (int)t2138;
        int t2145 = (int)t2139;
        int t2146 = t1607 + t2144;
        float t2147 = memory[50441716 + t2146];
        int t2148 = t1607 + t2144;
        int t2149 = t2148 + 1024;
        float t2150 = memory[50441716 + t2149];
        int t2151 = t1607 + t2145;
        float t2152 = memory[50441716 + t2151];
        int t2153 = t1607 + t2145;
        int t2154 = t2153 + 1024;
        float t2155 = memory[50441716 + t2154];
        float t2156 = t2142 * t2152;
        float t2157 = t2143 * t2155;
        float t2158 = t2156 - t2157;
        float t2159 = t2142 * t2155;
        float t2160 = t2143 * t2152;
        float t2161 = t2159 + t2160;
        int t2162 = t1607 + t2144;
        float t2163 = t2147 + t2158;
        memory[50441716 + t2162] = t2163;
        int t2165 = t1607 + t2144;
        int t2166 = t2165 + 1024;
        float t2167 = t2150 + t2161;
        memory[50441716 + t2166] = t2167;
        int t2169 = t1607 + t2145;
        float t2170 = t2147 - t2158;
        memory[50441716 + t2169] = t2170;
        int t2172 = t1607 + t2145;
        int t2173 = t2172 + 1024;
        float t2174 = t2150 - t2161;
        memory[50441716 + t2173] = t2174;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2177 = 0; t2177 < 1024; t2177++) {
        float t2178 = (float)t2177;
        float t2179 = (t2178 - metal::floor(t2178 / 2.0) * 2.0);
        float t2180 = t2179;
        float t2181 = (t2178 * 0.5);
        float t2182 = metal::floor(t2181);
        float t2183 = t2180 * 2.0;
        float t2184 = (t2182 - metal::floor(t2182 / 2.0) * 2.0);
        float t2185 = t2183 + t2184;
        float t2186 = (t2182 * 0.5);
        float t2187 = metal::floor(t2186);
        float t2188 = t2185 * 2.0;
        float t2189 = (t2187 - metal::floor(t2187 / 2.0) * 2.0);
        float t2190 = t2188 + t2189;
        float t2191 = (t2187 * 0.5);
        float t2192 = metal::floor(t2191);
        float t2193 = t2190 * 2.0;
        float t2194 = (t2192 - metal::floor(t2192 / 2.0) * 2.0);
        float t2195 = t2193 + t2194;
        float t2196 = (t2192 * 0.5);
        float t2197 = metal::floor(t2196);
        float t2198 = t2195 * 2.0;
        float t2199 = (t2197 - metal::floor(t2197 / 2.0) * 2.0);
        float t2200 = t2198 + t2199;
        float t2201 = (t2197 * 0.5);
        float t2202 = metal::floor(t2201);
        float t2203 = t2200 * 2.0;
        float t2204 = (t2202 - metal::floor(t2202 / 2.0) * 2.0);
        float t2205 = t2203 + t2204;
        float t2206 = (t2202 * 0.5);
        float t2207 = metal::floor(t2206);
        float t2208 = t2205 * 2.0;
        float t2209 = (t2207 - metal::floor(t2207 / 2.0) * 2.0);
        float t2210 = t2208 + t2209;
        float t2211 = (t2207 * 0.5);
        float t2212 = metal::floor(t2211);
        float t2213 = t2210 * 2.0;
        float t2214 = (t2212 - metal::floor(t2212 / 2.0) * 2.0);
        float t2215 = t2213 + t2214;
        float t2216 = (t2212 * 0.5);
        float t2217 = metal::floor(t2216);
        float t2218 = t2215 * 2.0;
        float t2219 = (t2217 - metal::floor(t2217 / 2.0) * 2.0);
        float t2220 = t2218 + t2219;
        float t2221 = (t2217 * 0.5);
        float t2222 = metal::floor(t2221);
        float t2223 = t2220 * 2.0;
        float t2224 = (t2222 - metal::floor(t2222 / 2.0) * 2.0);
        float t2225 = t2223 + t2224;
        float t2226 = (t2222 * 0.5);
        float t2227 = metal::floor(t2226);
        float t2228 = (float)t2177;
        float t2229 = t2228 < t2225;
        int t2230 = (int)t2225;
        int t2231 = t1607 + t2177;
        float t2232 = memory[83996148 + t2231];
        int t2233 = t1607 + t2177;
        int t2234 = t2233 + 1024;
        float t2235 = memory[83996148 + t2234];
        int t2236 = t1607 + t2230;
        float t2237 = memory[83996148 + t2236];
        int t2238 = t1607 + t2230;
        int t2239 = t2238 + 1024;
        float t2240 = memory[83996148 + t2239];
        float t2241 = metal::select(t2232, t2237, t2229 > 0.0);
        float t2242 = metal::select(t2235, t2240, t2229 > 0.0);
        float t2243 = metal::select(t2237, t2232, t2229 > 0.0);
        float t2244 = metal::select(t2240, t2235, t2229 > 0.0);
        int t2245 = t1607 + t2177;
        memory[83996148 + t2245] = t2241;
        int t2247 = t1607 + t2177;
        int t2248 = t2247 + 1024;
        memory[83996148 + t2248] = t2242;
        int t2250 = t1607 + t2230;
        memory[83996148 + t2250] = t2243;
        int t2252 = t1607 + t2230;
        int t2253 = t2252 + 1024;
        memory[83996148 + t2253] = t2244;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2256 = 0; _pr2256 < 512; _pr2256++) {
        float t2257 = (float)_pr2256;
        float t2258 = t2257;
        float t2259 = metal::floor(t2258);
        float t2260 = t2259;
        float t2261 = t2257 - t2260;
        float t2262 = t2259 * 2.0;
        float t2263 = t2262 + t2261;
        float t2264 = t2263 + 1.0;
        float t2265 = -6.283185 * t2261;
        float t2266 = (t2265 * 0.5);
        float t2267 = metal::cos(t2266);
        float t2268 = metal::sin(t2266);
        int t2269 = (int)t2263;
        int t2270 = (int)t2264;
        int t2271 = t1607 + t2269;
        float t2272 = memory[83996148 + t2271];
        int t2273 = t1607 + t2269;
        int t2274 = t2273 + 1024;
        float t2275 = memory[83996148 + t2274];
        int t2276 = t1607 + t2270;
        float t2277 = memory[83996148 + t2276];
        int t2278 = t1607 + t2270;
        int t2279 = t2278 + 1024;
        float t2280 = memory[83996148 + t2279];
        float t2281 = t2267 * t2277;
        float t2282 = t2268 * t2280;
        float t2283 = t2281 - t2282;
        float t2284 = t2267 * t2280;
        float t2285 = t2268 * t2277;
        float t2286 = t2284 + t2285;
        int t2287 = t1607 + t2269;
        float t2288 = t2272 + t2283;
        memory[83996148 + t2287] = t2288;
        int t2290 = t1607 + t2269;
        int t2291 = t2290 + 1024;
        float t2292 = t2275 + t2286;
        memory[83996148 + t2291] = t2292;
        int t2294 = t1607 + t2270;
        float t2295 = t2272 - t2283;
        memory[83996148 + t2294] = t2295;
        int t2297 = t1607 + t2270;
        int t2298 = t2297 + 1024;
        float t2299 = t2275 - t2286;
        memory[83996148 + t2298] = t2299;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2302 = 0; _pr2302 < 512; _pr2302++) {
        float t2303 = (float)_pr2302;
        float t2304 = (t2303 * 0.5);
        float t2305 = metal::floor(t2304);
        float t2306 = t2305 * 2.0;
        float t2307 = t2303 - t2306;
        float t2308 = t2305 * 4.0;
        float t2309 = t2308 + t2307;
        float t2310 = t2309 + 2.0;
        float t2311 = -6.283185 * t2307;
        float t2312 = (t2311 * 0.25);
        float t2313 = metal::cos(t2312);
        float t2314 = metal::sin(t2312);
        int t2315 = (int)t2309;
        int t2316 = (int)t2310;
        int t2317 = t1607 + t2315;
        float t2318 = memory[83996148 + t2317];
        int t2319 = t1607 + t2315;
        int t2320 = t2319 + 1024;
        float t2321 = memory[83996148 + t2320];
        int t2322 = t1607 + t2316;
        float t2323 = memory[83996148 + t2322];
        int t2324 = t1607 + t2316;
        int t2325 = t2324 + 1024;
        float t2326 = memory[83996148 + t2325];
        float t2327 = t2313 * t2323;
        float t2328 = t2314 * t2326;
        float t2329 = t2327 - t2328;
        float t2330 = t2313 * t2326;
        float t2331 = t2314 * t2323;
        float t2332 = t2330 + t2331;
        int t2333 = t1607 + t2315;
        float t2334 = t2318 + t2329;
        memory[83996148 + t2333] = t2334;
        int t2336 = t1607 + t2315;
        int t2337 = t2336 + 1024;
        float t2338 = t2321 + t2332;
        memory[83996148 + t2337] = t2338;
        int t2340 = t1607 + t2316;
        float t2341 = t2318 - t2329;
        memory[83996148 + t2340] = t2341;
        int t2343 = t1607 + t2316;
        int t2344 = t2343 + 1024;
        float t2345 = t2321 - t2332;
        memory[83996148 + t2344] = t2345;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2348 = 0; _pr2348 < 512; _pr2348++) {
        float t2349 = (float)_pr2348;
        float t2350 = (t2349 * 0.25);
        float t2351 = metal::floor(t2350);
        float t2352 = t2351 * 4.0;
        float t2353 = t2349 - t2352;
        float t2354 = t2351 * 8.0;
        float t2355 = t2354 + t2353;
        float t2356 = t2355 + 4.0;
        float t2357 = -6.283185 * t2353;
        float t2358 = (t2357 * 0.125);
        float t2359 = metal::cos(t2358);
        float t2360 = metal::sin(t2358);
        int t2361 = (int)t2355;
        int t2362 = (int)t2356;
        int t2363 = t1607 + t2361;
        float t2364 = memory[83996148 + t2363];
        int t2365 = t1607 + t2361;
        int t2366 = t2365 + 1024;
        float t2367 = memory[83996148 + t2366];
        int t2368 = t1607 + t2362;
        float t2369 = memory[83996148 + t2368];
        int t2370 = t1607 + t2362;
        int t2371 = t2370 + 1024;
        float t2372 = memory[83996148 + t2371];
        float t2373 = t2359 * t2369;
        float t2374 = t2360 * t2372;
        float t2375 = t2373 - t2374;
        float t2376 = t2359 * t2372;
        float t2377 = t2360 * t2369;
        float t2378 = t2376 + t2377;
        int t2379 = t1607 + t2361;
        float t2380 = t2364 + t2375;
        memory[83996148 + t2379] = t2380;
        int t2382 = t1607 + t2361;
        int t2383 = t2382 + 1024;
        float t2384 = t2367 + t2378;
        memory[83996148 + t2383] = t2384;
        int t2386 = t1607 + t2362;
        float t2387 = t2364 - t2375;
        memory[83996148 + t2386] = t2387;
        int t2389 = t1607 + t2362;
        int t2390 = t2389 + 1024;
        float t2391 = t2367 - t2378;
        memory[83996148 + t2390] = t2391;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2394 = 0; _pr2394 < 512; _pr2394++) {
        float t2395 = (float)_pr2394;
        float t2396 = (t2395 * 0.125);
        float t2397 = metal::floor(t2396);
        float t2398 = t2397 * 8.0;
        float t2399 = t2395 - t2398;
        float t2400 = t2397 * 16.0;
        float t2401 = t2400 + t2399;
        float t2402 = t2401 + 8.0;
        float t2403 = -6.283185 * t2399;
        float t2404 = (t2403 * 0.0625);
        float t2405 = metal::cos(t2404);
        float t2406 = metal::sin(t2404);
        int t2407 = (int)t2401;
        int t2408 = (int)t2402;
        int t2409 = t1607 + t2407;
        float t2410 = memory[83996148 + t2409];
        int t2411 = t1607 + t2407;
        int t2412 = t2411 + 1024;
        float t2413 = memory[83996148 + t2412];
        int t2414 = t1607 + t2408;
        float t2415 = memory[83996148 + t2414];
        int t2416 = t1607 + t2408;
        int t2417 = t2416 + 1024;
        float t2418 = memory[83996148 + t2417];
        float t2419 = t2405 * t2415;
        float t2420 = t2406 * t2418;
        float t2421 = t2419 - t2420;
        float t2422 = t2405 * t2418;
        float t2423 = t2406 * t2415;
        float t2424 = t2422 + t2423;
        int t2425 = t1607 + t2407;
        float t2426 = t2410 + t2421;
        memory[83996148 + t2425] = t2426;
        int t2428 = t1607 + t2407;
        int t2429 = t2428 + 1024;
        float t2430 = t2413 + t2424;
        memory[83996148 + t2429] = t2430;
        int t2432 = t1607 + t2408;
        float t2433 = t2410 - t2421;
        memory[83996148 + t2432] = t2433;
        int t2435 = t1607 + t2408;
        int t2436 = t2435 + 1024;
        float t2437 = t2413 - t2424;
        memory[83996148 + t2436] = t2437;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2440 = 0; _pr2440 < 512; _pr2440++) {
        float t2441 = (float)_pr2440;
        float t2442 = (t2441 * 0.0625);
        float t2443 = metal::floor(t2442);
        float t2444 = t2443 * 16.0;
        float t2445 = t2441 - t2444;
        float t2446 = t2443 * 32.0;
        float t2447 = t2446 + t2445;
        float t2448 = t2447 + 16.0;
        float t2449 = -6.283185 * t2445;
        float t2450 = (t2449 * 0.03125);
        float t2451 = metal::cos(t2450);
        float t2452 = metal::sin(t2450);
        int t2453 = (int)t2447;
        int t2454 = (int)t2448;
        int t2455 = t1607 + t2453;
        float t2456 = memory[83996148 + t2455];
        int t2457 = t1607 + t2453;
        int t2458 = t2457 + 1024;
        float t2459 = memory[83996148 + t2458];
        int t2460 = t1607 + t2454;
        float t2461 = memory[83996148 + t2460];
        int t2462 = t1607 + t2454;
        int t2463 = t2462 + 1024;
        float t2464 = memory[83996148 + t2463];
        float t2465 = t2451 * t2461;
        float t2466 = t2452 * t2464;
        float t2467 = t2465 - t2466;
        float t2468 = t2451 * t2464;
        float t2469 = t2452 * t2461;
        float t2470 = t2468 + t2469;
        int t2471 = t1607 + t2453;
        float t2472 = t2456 + t2467;
        memory[83996148 + t2471] = t2472;
        int t2474 = t1607 + t2453;
        int t2475 = t2474 + 1024;
        float t2476 = t2459 + t2470;
        memory[83996148 + t2475] = t2476;
        int t2478 = t1607 + t2454;
        float t2479 = t2456 - t2467;
        memory[83996148 + t2478] = t2479;
        int t2481 = t1607 + t2454;
        int t2482 = t2481 + 1024;
        float t2483 = t2459 - t2470;
        memory[83996148 + t2482] = t2483;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2486 = 0; _pr2486 < 512; _pr2486++) {
        float t2487 = (float)_pr2486;
        float t2488 = (t2487 * 0.03125);
        float t2489 = metal::floor(t2488);
        float t2490 = t2489 * 32.0;
        float t2491 = t2487 - t2490;
        float t2492 = t2489 * 64.0;
        float t2493 = t2492 + t2491;
        float t2494 = t2493 + 32.0;
        float t2495 = -6.283185 * t2491;
        float t2496 = (t2495 * 0.015625);
        float t2497 = metal::cos(t2496);
        float t2498 = metal::sin(t2496);
        int t2499 = (int)t2493;
        int t2500 = (int)t2494;
        int t2501 = t1607 + t2499;
        float t2502 = memory[83996148 + t2501];
        int t2503 = t1607 + t2499;
        int t2504 = t2503 + 1024;
        float t2505 = memory[83996148 + t2504];
        int t2506 = t1607 + t2500;
        float t2507 = memory[83996148 + t2506];
        int t2508 = t1607 + t2500;
        int t2509 = t2508 + 1024;
        float t2510 = memory[83996148 + t2509];
        float t2511 = t2497 * t2507;
        float t2512 = t2498 * t2510;
        float t2513 = t2511 - t2512;
        float t2514 = t2497 * t2510;
        float t2515 = t2498 * t2507;
        float t2516 = t2514 + t2515;
        int t2517 = t1607 + t2499;
        float t2518 = t2502 + t2513;
        memory[83996148 + t2517] = t2518;
        int t2520 = t1607 + t2499;
        int t2521 = t2520 + 1024;
        float t2522 = t2505 + t2516;
        memory[83996148 + t2521] = t2522;
        int t2524 = t1607 + t2500;
        float t2525 = t2502 - t2513;
        memory[83996148 + t2524] = t2525;
        int t2527 = t1607 + t2500;
        int t2528 = t2527 + 1024;
        float t2529 = t2505 - t2516;
        memory[83996148 + t2528] = t2529;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2532 = 0; _pr2532 < 512; _pr2532++) {
        float t2533 = (float)_pr2532;
        float t2534 = (t2533 * 0.015625);
        float t2535 = metal::floor(t2534);
        float t2536 = t2535 * 64.0;
        float t2537 = t2533 - t2536;
        float t2538 = t2535 * 128.0;
        float t2539 = t2538 + t2537;
        float t2540 = t2539 + 64.0;
        float t2541 = -6.283185 * t2537;
        float t2542 = (t2541 * 0.0078125);
        float t2543 = metal::cos(t2542);
        float t2544 = metal::sin(t2542);
        int t2545 = (int)t2539;
        int t2546 = (int)t2540;
        int t2547 = t1607 + t2545;
        float t2548 = memory[83996148 + t2547];
        int t2549 = t1607 + t2545;
        int t2550 = t2549 + 1024;
        float t2551 = memory[83996148 + t2550];
        int t2552 = t1607 + t2546;
        float t2553 = memory[83996148 + t2552];
        int t2554 = t1607 + t2546;
        int t2555 = t2554 + 1024;
        float t2556 = memory[83996148 + t2555];
        float t2557 = t2543 * t2553;
        float t2558 = t2544 * t2556;
        float t2559 = t2557 - t2558;
        float t2560 = t2543 * t2556;
        float t2561 = t2544 * t2553;
        float t2562 = t2560 + t2561;
        int t2563 = t1607 + t2545;
        float t2564 = t2548 + t2559;
        memory[83996148 + t2563] = t2564;
        int t2566 = t1607 + t2545;
        int t2567 = t2566 + 1024;
        float t2568 = t2551 + t2562;
        memory[83996148 + t2567] = t2568;
        int t2570 = t1607 + t2546;
        float t2571 = t2548 - t2559;
        memory[83996148 + t2570] = t2571;
        int t2573 = t1607 + t2546;
        int t2574 = t2573 + 1024;
        float t2575 = t2551 - t2562;
        memory[83996148 + t2574] = t2575;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2578 = 0; _pr2578 < 512; _pr2578++) {
        float t2579 = (float)_pr2578;
        float t2580 = (t2579 * 0.0078125);
        float t2581 = metal::floor(t2580);
        float t2582 = t2581 * 128.0;
        float t2583 = t2579 - t2582;
        float t2584 = t2581 * 256.0;
        float t2585 = t2584 + t2583;
        float t2586 = t2585 + 128.0;
        float t2587 = -6.283185 * t2583;
        float t2588 = (t2587 * 0.00390625);
        float t2589 = metal::cos(t2588);
        float t2590 = metal::sin(t2588);
        int t2591 = (int)t2585;
        int t2592 = (int)t2586;
        int t2593 = t1607 + t2591;
        float t2594 = memory[83996148 + t2593];
        int t2595 = t1607 + t2591;
        int t2596 = t2595 + 1024;
        float t2597 = memory[83996148 + t2596];
        int t2598 = t1607 + t2592;
        float t2599 = memory[83996148 + t2598];
        int t2600 = t1607 + t2592;
        int t2601 = t2600 + 1024;
        float t2602 = memory[83996148 + t2601];
        float t2603 = t2589 * t2599;
        float t2604 = t2590 * t2602;
        float t2605 = t2603 - t2604;
        float t2606 = t2589 * t2602;
        float t2607 = t2590 * t2599;
        float t2608 = t2606 + t2607;
        int t2609 = t1607 + t2591;
        float t2610 = t2594 + t2605;
        memory[83996148 + t2609] = t2610;
        int t2612 = t1607 + t2591;
        int t2613 = t2612 + 1024;
        float t2614 = t2597 + t2608;
        memory[83996148 + t2613] = t2614;
        int t2616 = t1607 + t2592;
        float t2617 = t2594 - t2605;
        memory[83996148 + t2616] = t2617;
        int t2619 = t1607 + t2592;
        int t2620 = t2619 + 1024;
        float t2621 = t2597 - t2608;
        memory[83996148 + t2620] = t2621;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2624 = 0; _pr2624 < 512; _pr2624++) {
        float t2625 = (float)_pr2624;
        float t2626 = (t2625 * 0.00390625);
        float t2627 = metal::floor(t2626);
        float t2628 = t2627 * 256.0;
        float t2629 = t2625 - t2628;
        float t2630 = t2627 * 512.0;
        float t2631 = t2630 + t2629;
        float t2632 = t2631 + 256.0;
        float t2633 = -6.283185 * t2629;
        float t2634 = (t2633 * 0.001953125);
        float t2635 = metal::cos(t2634);
        float t2636 = metal::sin(t2634);
        int t2637 = (int)t2631;
        int t2638 = (int)t2632;
        int t2639 = t1607 + t2637;
        float t2640 = memory[83996148 + t2639];
        int t2641 = t1607 + t2637;
        int t2642 = t2641 + 1024;
        float t2643 = memory[83996148 + t2642];
        int t2644 = t1607 + t2638;
        float t2645 = memory[83996148 + t2644];
        int t2646 = t1607 + t2638;
        int t2647 = t2646 + 1024;
        float t2648 = memory[83996148 + t2647];
        float t2649 = t2635 * t2645;
        float t2650 = t2636 * t2648;
        float t2651 = t2649 - t2650;
        float t2652 = t2635 * t2648;
        float t2653 = t2636 * t2645;
        float t2654 = t2652 + t2653;
        int t2655 = t1607 + t2637;
        float t2656 = t2640 + t2651;
        memory[83996148 + t2655] = t2656;
        int t2658 = t1607 + t2637;
        int t2659 = t2658 + 1024;
        float t2660 = t2643 + t2654;
        memory[83996148 + t2659] = t2660;
        int t2662 = t1607 + t2638;
        float t2663 = t2640 - t2651;
        memory[83996148 + t2662] = t2663;
        int t2665 = t1607 + t2638;
        int t2666 = t2665 + 1024;
        float t2667 = t2643 - t2654;
        memory[83996148 + t2666] = t2667;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2670 = 0; _pr2670 < 512; _pr2670++) {
        float t2671 = (float)_pr2670;
        float t2672 = (t2671 * 0.001953125);
        float t2673 = metal::floor(t2672);
        float t2674 = t2673 * 512.0;
        float t2675 = t2671 - t2674;
        float t2676 = t2673 * 1024.0;
        float t2677 = t2676 + t2675;
        float t2678 = t2677 + 512.0;
        float t2679 = -6.283185 * t2675;
        float t2680 = (t2679 * 0.0009765625);
        float t2681 = metal::cos(t2680);
        float t2682 = metal::sin(t2680);
        int t2683 = (int)t2677;
        int t2684 = (int)t2678;
        int t2685 = t1607 + t2683;
        float t2686 = memory[83996148 + t2685];
        int t2687 = t1607 + t2683;
        int t2688 = t2687 + 1024;
        float t2689 = memory[83996148 + t2688];
        int t2690 = t1607 + t2684;
        float t2691 = memory[83996148 + t2690];
        int t2692 = t1607 + t2684;
        int t2693 = t2692 + 1024;
        float t2694 = memory[83996148 + t2693];
        float t2695 = t2681 * t2691;
        float t2696 = t2682 * t2694;
        float t2697 = t2695 - t2696;
        float t2698 = t2681 * t2694;
        float t2699 = t2682 * t2691;
        float t2700 = t2698 + t2699;
        int t2701 = t1607 + t2683;
        float t2702 = t2686 + t2697;
        memory[83996148 + t2701] = t2702;
        int t2704 = t1607 + t2683;
        int t2705 = t2704 + 1024;
        float t2706 = t2689 + t2700;
        memory[83996148 + t2705] = t2706;
        int t2708 = t1607 + t2684;
        float t2709 = t2686 - t2697;
        memory[83996148 + t2708] = t2709;
        int t2711 = t1607 + t2684;
        int t2712 = t2711 + 1024;
        float t2713 = t2689 - t2700;
        memory[83996148 + t2712] = t2713;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2716 = 0; _pr2716 < 513; _pr2716++) {
        int t2717 = t1607 + _pr2716;
        float t2718 = memory[50441716 + t2717];
        int t2719 = t1607 + _pr2716;
        int t2720 = t2719 + 1024;
        float t2721 = memory[50441716 + t2720];
        float t2722 = t2718 * t2718;
        float t2723 = t2721 * t2721;
        float t2724 = t2722 + t2723;
        float t2725 = metal::sqrt(t2724);
        int t2726 = t1608 + _pr2716;
        memory[117550580 + t2726] = t2725;
        int t2728 = t1607 + _pr2716;
        float t2729 = memory[83996148 + t2728];
        int t2730 = t1607 + _pr2716;
        int t2731 = t2730 + 1024;
        float t2732 = memory[83996148 + t2731];
        float t2733 = t2729 * t2729;
        float t2734 = t2732 * t2732;
        float t2735 = t2733 + t2734;
        float t2736 = metal::sqrt(t2735);
        int t2737 = t1608 + _pr2716;
        memory[125955572 + t2737] = t2736;
        float t2739 = t2725 - t2736;
        int t2740 = t1608 + _pr2716;
        float t2741 = t2739 * t2739;
        memory[134360564 + t2740] = t2741;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2744 = 0; t2744 < 513; t2744++) {
        int t2745 = t1608 + t2744;
        float t2746 = memory[134360564 + t2745];
        float t2747 = t[15*frameCount + id] + t2746;
        t[15*frameCount + id] = t2747;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 19
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_19(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2758), value: global(2758)) */
  float t5737 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5737)) {
    /* loadGlobal(1610) - handled in variable access */
    /* loadGlobal(1588) - handled in variable access */
    int t2750 = id;
    int t2751 = t2750 / 61;
    uint _frameIndex = (uint)(t2751);
    int t2752 = t2751 * 61;
    int t2753 = t2750 - t2752;
    float t2754 = (t[15*frameCount + _frameIndex] * 6.1035156e-05);
    float t2755 = t[13*frameCount + _frameIndex] + t2754;
    float t2756 = t2755 * 0.5;
    float t2757 = t2756;
    t[16*frameCount + _frameIndex] = t2757;
    float t2759 = t2756;
    float t2760 = t2755;
    float t2761 = (t[15*frameCount + _frameIndex] * 3.7252903e-09);
    float t2762 = -0.5 * t2761;
  }
  #pragma clang diagnostic pop
}



// KERNEL 20
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_20(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1589) - handled in variable access */
    /* loadGlobal(527) - handled in variable access */
    /* loadGlobal(474) - handled in variable access */
    int t2763 = id;
    int t2764 = t2763 * 2048;
    int t2765 = t2763 * 513;
    int t2766 = t2763 * 2048;
    float t2767 = t[14*frameCount + id] == 0.0;
    if (t2767) {
      for (uint _pr2769 = 0; _pr2769 < 513; _pr2769++) {
        int t2770 = t2765 + _pr2769;
        float t2771 = memory[117550580 + t2770];
        int t2772 = t2765 + _pr2769;
        float t2773 = memory[125955572 + t2772];
        int t2774 = t2764 + _pr2769;
        float t2775 = memory[50441716 + t2774];
        int t2776 = t2764 + _pr2769;
        int t2777 = t2776 + 1024;
        float t2778 = memory[50441716 + t2777];
        int t2779 = t2764 + _pr2769;
        float t2780 = memory[83996148 + t2779];
        int t2781 = t2764 + _pr2769;
        int t2782 = t2781 + 1024;
        float t2783 = memory[83996148 + t2782];
        float t2784 = t2771 - t2773;
        float t2785 = 2.0 * t2784;
        float t2786 = t2785 * 3.0517578e-05;
        float t2787 = t2771 - t2773;
        float t2788 = -2.0 * t2787;
        float t2789 = t2788 * 3.0517578e-05;
        float t2790 = metal::max(t2771, 1e-08);
        float t2791 = metal::max(t2773, 1e-08);
        float t2792 = t2786 * t2775;
        float t2793 = t2792 / t2790;
        float t2794 = t2786 * t2778;
        float t2795 = t2794 / t2790;
        float t2796 = t2789 * t2780;
        float t2797 = t2796 / t2791;
        float t2798 = t2789 * t2783;
        float t2799 = t2798 / t2791;
        int t2800 = t2766 + _pr2769;
        memory[142765556 + t2800] = t2793;
        int t2802 = t2766 + _pr2769;
        int t2803 = t2802 + 1024;
        memory[142765556 + t2803] = t2795;
        int t2805 = t2766 + _pr2769;
        memory[176319988 + t2805] = t2797;
        int t2807 = t2766 + _pr2769;
        int t2808 = t2807 + 1024;
        memory[176319988 + t2808] = t2799;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2811 = 0; _pr2811 < 511; _pr2811++) {
        int t2812 = _pr2811 + 513;
        int t2813 = t2766 + t2812;
        memory[142765556 + t2813] = 0.0;
        int t2815 = t2766 + t2812;
        int t2816 = t2815 + 1024;
        memory[142765556 + t2816] = 0.0;
        int t2818 = t2766 + t2812;
        memory[176319988 + t2818] = 0.0;
        int t2820 = t2766 + t2812;
        int t2821 = t2820 + 1024;
        memory[176319988 + t2821] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 21
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_21(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1589) - handled in variable access */
    int t2825 = id;
    int t2826 = t2825 * 2048;
    int t2827 = t2825 * 1024;
    float t2828 = t[14*frameCount + id] == 0.0;
    if (t2828) {
      for (uint t2830 = 0; t2830 < 1024; t2830++) {
        float t2831 = (float)t2830;
        float t2832 = (t2831 - metal::floor(t2831 / 2.0) * 2.0);
        float t2833 = t2832;
        float t2834 = (t2831 * 0.5);
        float t2835 = metal::floor(t2834);
        float t2836 = t2833 * 2.0;
        float t2837 = (t2835 - metal::floor(t2835 / 2.0) * 2.0);
        float t2838 = t2836 + t2837;
        float t2839 = (t2835 * 0.5);
        float t2840 = metal::floor(t2839);
        float t2841 = t2838 * 2.0;
        float t2842 = (t2840 - metal::floor(t2840 / 2.0) * 2.0);
        float t2843 = t2841 + t2842;
        float t2844 = (t2840 * 0.5);
        float t2845 = metal::floor(t2844);
        float t2846 = t2843 * 2.0;
        float t2847 = (t2845 - metal::floor(t2845 / 2.0) * 2.0);
        float t2848 = t2846 + t2847;
        float t2849 = (t2845 * 0.5);
        float t2850 = metal::floor(t2849);
        float t2851 = t2848 * 2.0;
        float t2852 = (t2850 - metal::floor(t2850 / 2.0) * 2.0);
        float t2853 = t2851 + t2852;
        float t2854 = (t2850 * 0.5);
        float t2855 = metal::floor(t2854);
        float t2856 = t2853 * 2.0;
        float t2857 = (t2855 - metal::floor(t2855 / 2.0) * 2.0);
        float t2858 = t2856 + t2857;
        float t2859 = (t2855 * 0.5);
        float t2860 = metal::floor(t2859);
        float t2861 = t2858 * 2.0;
        float t2862 = (t2860 - metal::floor(t2860 / 2.0) * 2.0);
        float t2863 = t2861 + t2862;
        float t2864 = (t2860 * 0.5);
        float t2865 = metal::floor(t2864);
        float t2866 = t2863 * 2.0;
        float t2867 = (t2865 - metal::floor(t2865 / 2.0) * 2.0);
        float t2868 = t2866 + t2867;
        float t2869 = (t2865 * 0.5);
        float t2870 = metal::floor(t2869);
        float t2871 = t2868 * 2.0;
        float t2872 = (t2870 - metal::floor(t2870 / 2.0) * 2.0);
        float t2873 = t2871 + t2872;
        float t2874 = (t2870 * 0.5);
        float t2875 = metal::floor(t2874);
        float t2876 = t2873 * 2.0;
        float t2877 = (t2875 - metal::floor(t2875 / 2.0) * 2.0);
        float t2878 = t2876 + t2877;
        float t2879 = (t2875 * 0.5);
        float t2880 = metal::floor(t2879);
        float t2881 = (float)t2830;
        float t2882 = t2881 < t2878;
        int t2883 = (int)t2878;
        int t2884 = t2826 + t2830;
        float t2885 = memory[142765556 + t2884];
        int t2886 = t2826 + t2830;
        int t2887 = t2886 + 1024;
        float t2888 = memory[142765556 + t2887];
        int t2889 = t2826 + t2883;
        float t2890 = memory[142765556 + t2889];
        int t2891 = t2826 + t2883;
        int t2892 = t2891 + 1024;
        float t2893 = memory[142765556 + t2892];
        float t2894 = metal::select(t2885, t2890, t2882 > 0.0);
        float t2895 = metal::select(t2888, t2893, t2882 > 0.0);
        float t2896 = metal::select(t2890, t2885, t2882 > 0.0);
        float t2897 = metal::select(t2893, t2888, t2882 > 0.0);
        int t2898 = t2826 + t2830;
        memory[142765556 + t2898] = t2894;
        int t2900 = t2826 + t2830;
        int t2901 = t2900 + 1024;
        memory[142765556 + t2901] = t2895;
        int t2903 = t2826 + t2883;
        memory[142765556 + t2903] = t2896;
        int t2905 = t2826 + t2883;
        int t2906 = t2905 + 1024;
        memory[142765556 + t2906] = t2897;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2909 = 0; _pr2909 < 512; _pr2909++) {
        float t2910 = (float)_pr2909;
        float t2911 = t2910;
        float t2912 = metal::floor(t2911);
        float t2913 = t2912;
        float t2914 = t2910 - t2913;
        float t2915 = t2912 * 2.0;
        float t2916 = t2915 + t2914;
        float t2917 = t2916 + 1.0;
        float t2918 = 6.283185 * t2914;
        float t2919 = (t2918 * 0.5);
        float t2920 = metal::cos(t2919);
        float t2921 = metal::sin(t2919);
        int t2922 = (int)t2916;
        int t2923 = (int)t2917;
        int t2924 = t2826 + t2922;
        float t2925 = memory[142765556 + t2924];
        int t2926 = t2826 + t2922;
        int t2927 = t2926 + 1024;
        float t2928 = memory[142765556 + t2927];
        int t2929 = t2826 + t2923;
        float t2930 = memory[142765556 + t2929];
        int t2931 = t2826 + t2923;
        int t2932 = t2931 + 1024;
        float t2933 = memory[142765556 + t2932];
        float t2934 = t2920 * t2930;
        float t2935 = t2921 * t2933;
        float t2936 = t2934 - t2935;
        float t2937 = t2920 * t2933;
        float t2938 = t2921 * t2930;
        float t2939 = t2937 + t2938;
        int t2940 = t2826 + t2922;
        float t2941 = t2925 + t2936;
        memory[142765556 + t2940] = t2941;
        int t2943 = t2826 + t2922;
        int t2944 = t2943 + 1024;
        float t2945 = t2928 + t2939;
        memory[142765556 + t2944] = t2945;
        int t2947 = t2826 + t2923;
        float t2948 = t2925 - t2936;
        memory[142765556 + t2947] = t2948;
        int t2950 = t2826 + t2923;
        int t2951 = t2950 + 1024;
        float t2952 = t2928 - t2939;
        memory[142765556 + t2951] = t2952;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2955 = 0; _pr2955 < 512; _pr2955++) {
        float t2956 = (float)_pr2955;
        float t2957 = (t2956 * 0.5);
        float t2958 = metal::floor(t2957);
        float t2959 = t2958 * 2.0;
        float t2960 = t2956 - t2959;
        float t2961 = t2958 * 4.0;
        float t2962 = t2961 + t2960;
        float t2963 = t2962 + 2.0;
        float t2964 = 6.283185 * t2960;
        float t2965 = (t2964 * 0.25);
        float t2966 = metal::cos(t2965);
        float t2967 = metal::sin(t2965);
        int t2968 = (int)t2962;
        int t2969 = (int)t2963;
        int t2970 = t2826 + t2968;
        float t2971 = memory[142765556 + t2970];
        int t2972 = t2826 + t2968;
        int t2973 = t2972 + 1024;
        float t2974 = memory[142765556 + t2973];
        int t2975 = t2826 + t2969;
        float t2976 = memory[142765556 + t2975];
        int t2977 = t2826 + t2969;
        int t2978 = t2977 + 1024;
        float t2979 = memory[142765556 + t2978];
        float t2980 = t2966 * t2976;
        float t2981 = t2967 * t2979;
        float t2982 = t2980 - t2981;
        float t2983 = t2966 * t2979;
        float t2984 = t2967 * t2976;
        float t2985 = t2983 + t2984;
        int t2986 = t2826 + t2968;
        float t2987 = t2971 + t2982;
        memory[142765556 + t2986] = t2987;
        int t2989 = t2826 + t2968;
        int t2990 = t2989 + 1024;
        float t2991 = t2974 + t2985;
        memory[142765556 + t2990] = t2991;
        int t2993 = t2826 + t2969;
        float t2994 = t2971 - t2982;
        memory[142765556 + t2993] = t2994;
        int t2996 = t2826 + t2969;
        int t2997 = t2996 + 1024;
        float t2998 = t2974 - t2985;
        memory[142765556 + t2997] = t2998;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3001 = 0; _pr3001 < 512; _pr3001++) {
        float t3002 = (float)_pr3001;
        float t3003 = (t3002 * 0.25);
        float t3004 = metal::floor(t3003);
        float t3005 = t3004 * 4.0;
        float t3006 = t3002 - t3005;
        float t3007 = t3004 * 8.0;
        float t3008 = t3007 + t3006;
        float t3009 = t3008 + 4.0;
        float t3010 = 6.283185 * t3006;
        float t3011 = (t3010 * 0.125);
        float t3012 = metal::cos(t3011);
        float t3013 = metal::sin(t3011);
        int t3014 = (int)t3008;
        int t3015 = (int)t3009;
        int t3016 = t2826 + t3014;
        float t3017 = memory[142765556 + t3016];
        int t3018 = t2826 + t3014;
        int t3019 = t3018 + 1024;
        float t3020 = memory[142765556 + t3019];
        int t3021 = t2826 + t3015;
        float t3022 = memory[142765556 + t3021];
        int t3023 = t2826 + t3015;
        int t3024 = t3023 + 1024;
        float t3025 = memory[142765556 + t3024];
        float t3026 = t3012 * t3022;
        float t3027 = t3013 * t3025;
        float t3028 = t3026 - t3027;
        float t3029 = t3012 * t3025;
        float t3030 = t3013 * t3022;
        float t3031 = t3029 + t3030;
        int t3032 = t2826 + t3014;
        float t3033 = t3017 + t3028;
        memory[142765556 + t3032] = t3033;
        int t3035 = t2826 + t3014;
        int t3036 = t3035 + 1024;
        float t3037 = t3020 + t3031;
        memory[142765556 + t3036] = t3037;
        int t3039 = t2826 + t3015;
        float t3040 = t3017 - t3028;
        memory[142765556 + t3039] = t3040;
        int t3042 = t2826 + t3015;
        int t3043 = t3042 + 1024;
        float t3044 = t3020 - t3031;
        memory[142765556 + t3043] = t3044;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3047 = 0; _pr3047 < 512; _pr3047++) {
        float t3048 = (float)_pr3047;
        float t3049 = (t3048 * 0.125);
        float t3050 = metal::floor(t3049);
        float t3051 = t3050 * 8.0;
        float t3052 = t3048 - t3051;
        float t3053 = t3050 * 16.0;
        float t3054 = t3053 + t3052;
        float t3055 = t3054 + 8.0;
        float t3056 = 6.283185 * t3052;
        float t3057 = (t3056 * 0.0625);
        float t3058 = metal::cos(t3057);
        float t3059 = metal::sin(t3057);
        int t3060 = (int)t3054;
        int t3061 = (int)t3055;
        int t3062 = t2826 + t3060;
        float t3063 = memory[142765556 + t3062];
        int t3064 = t2826 + t3060;
        int t3065 = t3064 + 1024;
        float t3066 = memory[142765556 + t3065];
        int t3067 = t2826 + t3061;
        float t3068 = memory[142765556 + t3067];
        int t3069 = t2826 + t3061;
        int t3070 = t3069 + 1024;
        float t3071 = memory[142765556 + t3070];
        float t3072 = t3058 * t3068;
        float t3073 = t3059 * t3071;
        float t3074 = t3072 - t3073;
        float t3075 = t3058 * t3071;
        float t3076 = t3059 * t3068;
        float t3077 = t3075 + t3076;
        int t3078 = t2826 + t3060;
        float t3079 = t3063 + t3074;
        memory[142765556 + t3078] = t3079;
        int t3081 = t2826 + t3060;
        int t3082 = t3081 + 1024;
        float t3083 = t3066 + t3077;
        memory[142765556 + t3082] = t3083;
        int t3085 = t2826 + t3061;
        float t3086 = t3063 - t3074;
        memory[142765556 + t3085] = t3086;
        int t3088 = t2826 + t3061;
        int t3089 = t3088 + 1024;
        float t3090 = t3066 - t3077;
        memory[142765556 + t3089] = t3090;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3093 = 0; _pr3093 < 512; _pr3093++) {
        float t3094 = (float)_pr3093;
        float t3095 = (t3094 * 0.0625);
        float t3096 = metal::floor(t3095);
        float t3097 = t3096 * 16.0;
        float t3098 = t3094 - t3097;
        float t3099 = t3096 * 32.0;
        float t3100 = t3099 + t3098;
        float t3101 = t3100 + 16.0;
        float t3102 = 6.283185 * t3098;
        float t3103 = (t3102 * 0.03125);
        float t3104 = metal::cos(t3103);
        float t3105 = metal::sin(t3103);
        int t3106 = (int)t3100;
        int t3107 = (int)t3101;
        int t3108 = t2826 + t3106;
        float t3109 = memory[142765556 + t3108];
        int t3110 = t2826 + t3106;
        int t3111 = t3110 + 1024;
        float t3112 = memory[142765556 + t3111];
        int t3113 = t2826 + t3107;
        float t3114 = memory[142765556 + t3113];
        int t3115 = t2826 + t3107;
        int t3116 = t3115 + 1024;
        float t3117 = memory[142765556 + t3116];
        float t3118 = t3104 * t3114;
        float t3119 = t3105 * t3117;
        float t3120 = t3118 - t3119;
        float t3121 = t3104 * t3117;
        float t3122 = t3105 * t3114;
        float t3123 = t3121 + t3122;
        int t3124 = t2826 + t3106;
        float t3125 = t3109 + t3120;
        memory[142765556 + t3124] = t3125;
        int t3127 = t2826 + t3106;
        int t3128 = t3127 + 1024;
        float t3129 = t3112 + t3123;
        memory[142765556 + t3128] = t3129;
        int t3131 = t2826 + t3107;
        float t3132 = t3109 - t3120;
        memory[142765556 + t3131] = t3132;
        int t3134 = t2826 + t3107;
        int t3135 = t3134 + 1024;
        float t3136 = t3112 - t3123;
        memory[142765556 + t3135] = t3136;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3139 = 0; _pr3139 < 512; _pr3139++) {
        float t3140 = (float)_pr3139;
        float t3141 = (t3140 * 0.03125);
        float t3142 = metal::floor(t3141);
        float t3143 = t3142 * 32.0;
        float t3144 = t3140 - t3143;
        float t3145 = t3142 * 64.0;
        float t3146 = t3145 + t3144;
        float t3147 = t3146 + 32.0;
        float t3148 = 6.283185 * t3144;
        float t3149 = (t3148 * 0.015625);
        float t3150 = metal::cos(t3149);
        float t3151 = metal::sin(t3149);
        int t3152 = (int)t3146;
        int t3153 = (int)t3147;
        int t3154 = t2826 + t3152;
        float t3155 = memory[142765556 + t3154];
        int t3156 = t2826 + t3152;
        int t3157 = t3156 + 1024;
        float t3158 = memory[142765556 + t3157];
        int t3159 = t2826 + t3153;
        float t3160 = memory[142765556 + t3159];
        int t3161 = t2826 + t3153;
        int t3162 = t3161 + 1024;
        float t3163 = memory[142765556 + t3162];
        float t3164 = t3150 * t3160;
        float t3165 = t3151 * t3163;
        float t3166 = t3164 - t3165;
        float t3167 = t3150 * t3163;
        float t3168 = t3151 * t3160;
        float t3169 = t3167 + t3168;
        int t3170 = t2826 + t3152;
        float t3171 = t3155 + t3166;
        memory[142765556 + t3170] = t3171;
        int t3173 = t2826 + t3152;
        int t3174 = t3173 + 1024;
        float t3175 = t3158 + t3169;
        memory[142765556 + t3174] = t3175;
        int t3177 = t2826 + t3153;
        float t3178 = t3155 - t3166;
        memory[142765556 + t3177] = t3178;
        int t3180 = t2826 + t3153;
        int t3181 = t3180 + 1024;
        float t3182 = t3158 - t3169;
        memory[142765556 + t3181] = t3182;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3185 = 0; _pr3185 < 512; _pr3185++) {
        float t3186 = (float)_pr3185;
        float t3187 = (t3186 * 0.015625);
        float t3188 = metal::floor(t3187);
        float t3189 = t3188 * 64.0;
        float t3190 = t3186 - t3189;
        float t3191 = t3188 * 128.0;
        float t3192 = t3191 + t3190;
        float t3193 = t3192 + 64.0;
        float t3194 = 6.283185 * t3190;
        float t3195 = (t3194 * 0.0078125);
        float t3196 = metal::cos(t3195);
        float t3197 = metal::sin(t3195);
        int t3198 = (int)t3192;
        int t3199 = (int)t3193;
        int t3200 = t2826 + t3198;
        float t3201 = memory[142765556 + t3200];
        int t3202 = t2826 + t3198;
        int t3203 = t3202 + 1024;
        float t3204 = memory[142765556 + t3203];
        int t3205 = t2826 + t3199;
        float t3206 = memory[142765556 + t3205];
        int t3207 = t2826 + t3199;
        int t3208 = t3207 + 1024;
        float t3209 = memory[142765556 + t3208];
        float t3210 = t3196 * t3206;
        float t3211 = t3197 * t3209;
        float t3212 = t3210 - t3211;
        float t3213 = t3196 * t3209;
        float t3214 = t3197 * t3206;
        float t3215 = t3213 + t3214;
        int t3216 = t2826 + t3198;
        float t3217 = t3201 + t3212;
        memory[142765556 + t3216] = t3217;
        int t3219 = t2826 + t3198;
        int t3220 = t3219 + 1024;
        float t3221 = t3204 + t3215;
        memory[142765556 + t3220] = t3221;
        int t3223 = t2826 + t3199;
        float t3224 = t3201 - t3212;
        memory[142765556 + t3223] = t3224;
        int t3226 = t2826 + t3199;
        int t3227 = t3226 + 1024;
        float t3228 = t3204 - t3215;
        memory[142765556 + t3227] = t3228;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3231 = 0; _pr3231 < 512; _pr3231++) {
        float t3232 = (float)_pr3231;
        float t3233 = (t3232 * 0.0078125);
        float t3234 = metal::floor(t3233);
        float t3235 = t3234 * 128.0;
        float t3236 = t3232 - t3235;
        float t3237 = t3234 * 256.0;
        float t3238 = t3237 + t3236;
        float t3239 = t3238 + 128.0;
        float t3240 = 6.283185 * t3236;
        float t3241 = (t3240 * 0.00390625);
        float t3242 = metal::cos(t3241);
        float t3243 = metal::sin(t3241);
        int t3244 = (int)t3238;
        int t3245 = (int)t3239;
        int t3246 = t2826 + t3244;
        float t3247 = memory[142765556 + t3246];
        int t3248 = t2826 + t3244;
        int t3249 = t3248 + 1024;
        float t3250 = memory[142765556 + t3249];
        int t3251 = t2826 + t3245;
        float t3252 = memory[142765556 + t3251];
        int t3253 = t2826 + t3245;
        int t3254 = t3253 + 1024;
        float t3255 = memory[142765556 + t3254];
        float t3256 = t3242 * t3252;
        float t3257 = t3243 * t3255;
        float t3258 = t3256 - t3257;
        float t3259 = t3242 * t3255;
        float t3260 = t3243 * t3252;
        float t3261 = t3259 + t3260;
        int t3262 = t2826 + t3244;
        float t3263 = t3247 + t3258;
        memory[142765556 + t3262] = t3263;
        int t3265 = t2826 + t3244;
        int t3266 = t3265 + 1024;
        float t3267 = t3250 + t3261;
        memory[142765556 + t3266] = t3267;
        int t3269 = t2826 + t3245;
        float t3270 = t3247 - t3258;
        memory[142765556 + t3269] = t3270;
        int t3272 = t2826 + t3245;
        int t3273 = t3272 + 1024;
        float t3274 = t3250 - t3261;
        memory[142765556 + t3273] = t3274;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3277 = 0; _pr3277 < 512; _pr3277++) {
        float t3278 = (float)_pr3277;
        float t3279 = (t3278 * 0.00390625);
        float t3280 = metal::floor(t3279);
        float t3281 = t3280 * 256.0;
        float t3282 = t3278 - t3281;
        float t3283 = t3280 * 512.0;
        float t3284 = t3283 + t3282;
        float t3285 = t3284 + 256.0;
        float t3286 = 6.283185 * t3282;
        float t3287 = (t3286 * 0.001953125);
        float t3288 = metal::cos(t3287);
        float t3289 = metal::sin(t3287);
        int t3290 = (int)t3284;
        int t3291 = (int)t3285;
        int t3292 = t2826 + t3290;
        float t3293 = memory[142765556 + t3292];
        int t3294 = t2826 + t3290;
        int t3295 = t3294 + 1024;
        float t3296 = memory[142765556 + t3295];
        int t3297 = t2826 + t3291;
        float t3298 = memory[142765556 + t3297];
        int t3299 = t2826 + t3291;
        int t3300 = t3299 + 1024;
        float t3301 = memory[142765556 + t3300];
        float t3302 = t3288 * t3298;
        float t3303 = t3289 * t3301;
        float t3304 = t3302 - t3303;
        float t3305 = t3288 * t3301;
        float t3306 = t3289 * t3298;
        float t3307 = t3305 + t3306;
        int t3308 = t2826 + t3290;
        float t3309 = t3293 + t3304;
        memory[142765556 + t3308] = t3309;
        int t3311 = t2826 + t3290;
        int t3312 = t3311 + 1024;
        float t3313 = t3296 + t3307;
        memory[142765556 + t3312] = t3313;
        int t3315 = t2826 + t3291;
        float t3316 = t3293 - t3304;
        memory[142765556 + t3315] = t3316;
        int t3318 = t2826 + t3291;
        int t3319 = t3318 + 1024;
        float t3320 = t3296 - t3307;
        memory[142765556 + t3319] = t3320;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3323 = 0; _pr3323 < 512; _pr3323++) {
        float t3324 = (float)_pr3323;
        float t3325 = (t3324 * 0.001953125);
        float t3326 = metal::floor(t3325);
        float t3327 = t3326 * 512.0;
        float t3328 = t3324 - t3327;
        float t3329 = t3326 * 1024.0;
        float t3330 = t3329 + t3328;
        float t3331 = t3330 + 512.0;
        float t3332 = 6.283185 * t3328;
        float t3333 = (t3332 * 0.0009765625);
        float t3334 = metal::cos(t3333);
        float t3335 = metal::sin(t3333);
        int t3336 = (int)t3330;
        int t3337 = (int)t3331;
        int t3338 = t2826 + t3336;
        float t3339 = memory[142765556 + t3338];
        int t3340 = t2826 + t3336;
        int t3341 = t3340 + 1024;
        float t3342 = memory[142765556 + t3341];
        int t3343 = t2826 + t3337;
        float t3344 = memory[142765556 + t3343];
        int t3345 = t2826 + t3337;
        int t3346 = t3345 + 1024;
        float t3347 = memory[142765556 + t3346];
        float t3348 = t3334 * t3344;
        float t3349 = t3335 * t3347;
        float t3350 = t3348 - t3349;
        float t3351 = t3334 * t3347;
        float t3352 = t3335 * t3344;
        float t3353 = t3351 + t3352;
        int t3354 = t2826 + t3336;
        float t3355 = t3339 + t3350;
        memory[142765556 + t3354] = t3355;
        int t3357 = t2826 + t3336;
        int t3358 = t3357 + 1024;
        float t3359 = t3342 + t3353;
        memory[142765556 + t3358] = t3359;
        int t3361 = t2826 + t3337;
        float t3362 = t3339 - t3350;
        memory[142765556 + t3361] = t3362;
        int t3364 = t2826 + t3337;
        int t3365 = t3364 + 1024;
        float t3366 = t3342 - t3353;
        memory[142765556 + t3365] = t3366;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3369 = 0; _pr3369 < 1024; _pr3369++) {
        int t3370 = t2826 + _pr3369;
        float t3371 = memory[142765556 + t3370];
        float t3372 = t3371 * 1.9036306e-06;
        float t3373 = memory[52788 + (int)_pr3369];
        int t3374 = t2827 + _pr3369;
        float t3375 = t3372 * t3373;
        memory[50441716 + t3374] = t3375;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t3378 = 0; t3378 < 1024; t3378++) {
        float t3379 = (float)t3378;
        float t3380 = (t3379 - metal::floor(t3379 / 2.0) * 2.0);
        float t3381 = t3380;
        float t3382 = (t3379 * 0.5);
        float t3383 = metal::floor(t3382);
        float t3384 = t3381 * 2.0;
        float t3385 = (t3383 - metal::floor(t3383 / 2.0) * 2.0);
        float t3386 = t3384 + t3385;
        float t3387 = (t3383 * 0.5);
        float t3388 = metal::floor(t3387);
        float t3389 = t3386 * 2.0;
        float t3390 = (t3388 - metal::floor(t3388 / 2.0) * 2.0);
        float t3391 = t3389 + t3390;
        float t3392 = (t3388 * 0.5);
        float t3393 = metal::floor(t3392);
        float t3394 = t3391 * 2.0;
        float t3395 = (t3393 - metal::floor(t3393 / 2.0) * 2.0);
        float t3396 = t3394 + t3395;
        float t3397 = (t3393 * 0.5);
        float t3398 = metal::floor(t3397);
        float t3399 = t3396 * 2.0;
        float t3400 = (t3398 - metal::floor(t3398 / 2.0) * 2.0);
        float t3401 = t3399 + t3400;
        float t3402 = (t3398 * 0.5);
        float t3403 = metal::floor(t3402);
        float t3404 = t3401 * 2.0;
        float t3405 = (t3403 - metal::floor(t3403 / 2.0) * 2.0);
        float t3406 = t3404 + t3405;
        float t3407 = (t3403 * 0.5);
        float t3408 = metal::floor(t3407);
        float t3409 = t3406 * 2.0;
        float t3410 = (t3408 - metal::floor(t3408 / 2.0) * 2.0);
        float t3411 = t3409 + t3410;
        float t3412 = (t3408 * 0.5);
        float t3413 = metal::floor(t3412);
        float t3414 = t3411 * 2.0;
        float t3415 = (t3413 - metal::floor(t3413 / 2.0) * 2.0);
        float t3416 = t3414 + t3415;
        float t3417 = (t3413 * 0.5);
        float t3418 = metal::floor(t3417);
        float t3419 = t3416 * 2.0;
        float t3420 = (t3418 - metal::floor(t3418 / 2.0) * 2.0);
        float t3421 = t3419 + t3420;
        float t3422 = (t3418 * 0.5);
        float t3423 = metal::floor(t3422);
        float t3424 = t3421 * 2.0;
        float t3425 = (t3423 - metal::floor(t3423 / 2.0) * 2.0);
        float t3426 = t3424 + t3425;
        float t3427 = (t3423 * 0.5);
        float t3428 = metal::floor(t3427);
        float t3429 = (float)t3378;
        float t3430 = t3429 < t3426;
        int t3431 = (int)t3426;
        int t3432 = t2826 + t3378;
        float t3433 = memory[176319988 + t3432];
        int t3434 = t2826 + t3378;
        int t3435 = t3434 + 1024;
        float t3436 = memory[176319988 + t3435];
        int t3437 = t2826 + t3431;
        float t3438 = memory[176319988 + t3437];
        int t3439 = t2826 + t3431;
        int t3440 = t3439 + 1024;
        float t3441 = memory[176319988 + t3440];
        float t3442 = metal::select(t3433, t3438, t3430 > 0.0);
        float t3443 = metal::select(t3436, t3441, t3430 > 0.0);
        float t3444 = metal::select(t3438, t3433, t3430 > 0.0);
        float t3445 = metal::select(t3441, t3436, t3430 > 0.0);
        int t3446 = t2826 + t3378;
        memory[176319988 + t3446] = t3442;
        int t3448 = t2826 + t3378;
        int t3449 = t3448 + 1024;
        memory[176319988 + t3449] = t3443;
        int t3451 = t2826 + t3431;
        memory[176319988 + t3451] = t3444;
        int t3453 = t2826 + t3431;
        int t3454 = t3453 + 1024;
        memory[176319988 + t3454] = t3445;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3457 = 0; _pr3457 < 512; _pr3457++) {
        float t3458 = (float)_pr3457;
        float t3459 = t3458;
        float t3460 = metal::floor(t3459);
        float t3461 = t3460;
        float t3462 = t3458 - t3461;
        float t3463 = t3460 * 2.0;
        float t3464 = t3463 + t3462;
        float t3465 = t3464 + 1.0;
        float t3466 = 6.283185 * t3462;
        float t3467 = (t3466 * 0.5);
        float t3468 = metal::cos(t3467);
        float t3469 = metal::sin(t3467);
        int t3470 = (int)t3464;
        int t3471 = (int)t3465;
        int t3472 = t2826 + t3470;
        float t3473 = memory[176319988 + t3472];
        int t3474 = t2826 + t3470;
        int t3475 = t3474 + 1024;
        float t3476 = memory[176319988 + t3475];
        int t3477 = t2826 + t3471;
        float t3478 = memory[176319988 + t3477];
        int t3479 = t2826 + t3471;
        int t3480 = t3479 + 1024;
        float t3481 = memory[176319988 + t3480];
        float t3482 = t3468 * t3478;
        float t3483 = t3469 * t3481;
        float t3484 = t3482 - t3483;
        float t3485 = t3468 * t3481;
        float t3486 = t3469 * t3478;
        float t3487 = t3485 + t3486;
        int t3488 = t2826 + t3470;
        float t3489 = t3473 + t3484;
        memory[176319988 + t3488] = t3489;
        int t3491 = t2826 + t3470;
        int t3492 = t3491 + 1024;
        float t3493 = t3476 + t3487;
        memory[176319988 + t3492] = t3493;
        int t3495 = t2826 + t3471;
        float t3496 = t3473 - t3484;
        memory[176319988 + t3495] = t3496;
        int t3498 = t2826 + t3471;
        int t3499 = t3498 + 1024;
        float t3500 = t3476 - t3487;
        memory[176319988 + t3499] = t3500;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3503 = 0; _pr3503 < 512; _pr3503++) {
        float t3504 = (float)_pr3503;
        float t3505 = (t3504 * 0.5);
        float t3506 = metal::floor(t3505);
        float t3507 = t3506 * 2.0;
        float t3508 = t3504 - t3507;
        float t3509 = t3506 * 4.0;
        float t3510 = t3509 + t3508;
        float t3511 = t3510 + 2.0;
        float t3512 = 6.283185 * t3508;
        float t3513 = (t3512 * 0.25);
        float t3514 = metal::cos(t3513);
        float t3515 = metal::sin(t3513);
        int t3516 = (int)t3510;
        int t3517 = (int)t3511;
        int t3518 = t2826 + t3516;
        float t3519 = memory[176319988 + t3518];
        int t3520 = t2826 + t3516;
        int t3521 = t3520 + 1024;
        float t3522 = memory[176319988 + t3521];
        int t3523 = t2826 + t3517;
        float t3524 = memory[176319988 + t3523];
        int t3525 = t2826 + t3517;
        int t3526 = t3525 + 1024;
        float t3527 = memory[176319988 + t3526];
        float t3528 = t3514 * t3524;
        float t3529 = t3515 * t3527;
        float t3530 = t3528 - t3529;
        float t3531 = t3514 * t3527;
        float t3532 = t3515 * t3524;
        float t3533 = t3531 + t3532;
        int t3534 = t2826 + t3516;
        float t3535 = t3519 + t3530;
        memory[176319988 + t3534] = t3535;
        int t3537 = t2826 + t3516;
        int t3538 = t3537 + 1024;
        float t3539 = t3522 + t3533;
        memory[176319988 + t3538] = t3539;
        int t3541 = t2826 + t3517;
        float t3542 = t3519 - t3530;
        memory[176319988 + t3541] = t3542;
        int t3544 = t2826 + t3517;
        int t3545 = t3544 + 1024;
        float t3546 = t3522 - t3533;
        memory[176319988 + t3545] = t3546;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3549 = 0; _pr3549 < 512; _pr3549++) {
        float t3550 = (float)_pr3549;
        float t3551 = (t3550 * 0.25);
        float t3552 = metal::floor(t3551);
        float t3553 = t3552 * 4.0;
        float t3554 = t3550 - t3553;
        float t3555 = t3552 * 8.0;
        float t3556 = t3555 + t3554;
        float t3557 = t3556 + 4.0;
        float t3558 = 6.283185 * t3554;
        float t3559 = (t3558 * 0.125);
        float t3560 = metal::cos(t3559);
        float t3561 = metal::sin(t3559);
        int t3562 = (int)t3556;
        int t3563 = (int)t3557;
        int t3564 = t2826 + t3562;
        float t3565 = memory[176319988 + t3564];
        int t3566 = t2826 + t3562;
        int t3567 = t3566 + 1024;
        float t3568 = memory[176319988 + t3567];
        int t3569 = t2826 + t3563;
        float t3570 = memory[176319988 + t3569];
        int t3571 = t2826 + t3563;
        int t3572 = t3571 + 1024;
        float t3573 = memory[176319988 + t3572];
        float t3574 = t3560 * t3570;
        float t3575 = t3561 * t3573;
        float t3576 = t3574 - t3575;
        float t3577 = t3560 * t3573;
        float t3578 = t3561 * t3570;
        float t3579 = t3577 + t3578;
        int t3580 = t2826 + t3562;
        float t3581 = t3565 + t3576;
        memory[176319988 + t3580] = t3581;
        int t3583 = t2826 + t3562;
        int t3584 = t3583 + 1024;
        float t3585 = t3568 + t3579;
        memory[176319988 + t3584] = t3585;
        int t3587 = t2826 + t3563;
        float t3588 = t3565 - t3576;
        memory[176319988 + t3587] = t3588;
        int t3590 = t2826 + t3563;
        int t3591 = t3590 + 1024;
        float t3592 = t3568 - t3579;
        memory[176319988 + t3591] = t3592;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3595 = 0; _pr3595 < 512; _pr3595++) {
        float t3596 = (float)_pr3595;
        float t3597 = (t3596 * 0.125);
        float t3598 = metal::floor(t3597);
        float t3599 = t3598 * 8.0;
        float t3600 = t3596 - t3599;
        float t3601 = t3598 * 16.0;
        float t3602 = t3601 + t3600;
        float t3603 = t3602 + 8.0;
        float t3604 = 6.283185 * t3600;
        float t3605 = (t3604 * 0.0625);
        float t3606 = metal::cos(t3605);
        float t3607 = metal::sin(t3605);
        int t3608 = (int)t3602;
        int t3609 = (int)t3603;
        int t3610 = t2826 + t3608;
        float t3611 = memory[176319988 + t3610];
        int t3612 = t2826 + t3608;
        int t3613 = t3612 + 1024;
        float t3614 = memory[176319988 + t3613];
        int t3615 = t2826 + t3609;
        float t3616 = memory[176319988 + t3615];
        int t3617 = t2826 + t3609;
        int t3618 = t3617 + 1024;
        float t3619 = memory[176319988 + t3618];
        float t3620 = t3606 * t3616;
        float t3621 = t3607 * t3619;
        float t3622 = t3620 - t3621;
        float t3623 = t3606 * t3619;
        float t3624 = t3607 * t3616;
        float t3625 = t3623 + t3624;
        int t3626 = t2826 + t3608;
        float t3627 = t3611 + t3622;
        memory[176319988 + t3626] = t3627;
        int t3629 = t2826 + t3608;
        int t3630 = t3629 + 1024;
        float t3631 = t3614 + t3625;
        memory[176319988 + t3630] = t3631;
        int t3633 = t2826 + t3609;
        float t3634 = t3611 - t3622;
        memory[176319988 + t3633] = t3634;
        int t3636 = t2826 + t3609;
        int t3637 = t3636 + 1024;
        float t3638 = t3614 - t3625;
        memory[176319988 + t3637] = t3638;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3641 = 0; _pr3641 < 512; _pr3641++) {
        float t3642 = (float)_pr3641;
        float t3643 = (t3642 * 0.0625);
        float t3644 = metal::floor(t3643);
        float t3645 = t3644 * 16.0;
        float t3646 = t3642 - t3645;
        float t3647 = t3644 * 32.0;
        float t3648 = t3647 + t3646;
        float t3649 = t3648 + 16.0;
        float t3650 = 6.283185 * t3646;
        float t3651 = (t3650 * 0.03125);
        float t3652 = metal::cos(t3651);
        float t3653 = metal::sin(t3651);
        int t3654 = (int)t3648;
        int t3655 = (int)t3649;
        int t3656 = t2826 + t3654;
        float t3657 = memory[176319988 + t3656];
        int t3658 = t2826 + t3654;
        int t3659 = t3658 + 1024;
        float t3660 = memory[176319988 + t3659];
        int t3661 = t2826 + t3655;
        float t3662 = memory[176319988 + t3661];
        int t3663 = t2826 + t3655;
        int t3664 = t3663 + 1024;
        float t3665 = memory[176319988 + t3664];
        float t3666 = t3652 * t3662;
        float t3667 = t3653 * t3665;
        float t3668 = t3666 - t3667;
        float t3669 = t3652 * t3665;
        float t3670 = t3653 * t3662;
        float t3671 = t3669 + t3670;
        int t3672 = t2826 + t3654;
        float t3673 = t3657 + t3668;
        memory[176319988 + t3672] = t3673;
        int t3675 = t2826 + t3654;
        int t3676 = t3675 + 1024;
        float t3677 = t3660 + t3671;
        memory[176319988 + t3676] = t3677;
        int t3679 = t2826 + t3655;
        float t3680 = t3657 - t3668;
        memory[176319988 + t3679] = t3680;
        int t3682 = t2826 + t3655;
        int t3683 = t3682 + 1024;
        float t3684 = t3660 - t3671;
        memory[176319988 + t3683] = t3684;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3687 = 0; _pr3687 < 512; _pr3687++) {
        float t3688 = (float)_pr3687;
        float t3689 = (t3688 * 0.03125);
        float t3690 = metal::floor(t3689);
        float t3691 = t3690 * 32.0;
        float t3692 = t3688 - t3691;
        float t3693 = t3690 * 64.0;
        float t3694 = t3693 + t3692;
        float t3695 = t3694 + 32.0;
        float t3696 = 6.283185 * t3692;
        float t3697 = (t3696 * 0.015625);
        float t3698 = metal::cos(t3697);
        float t3699 = metal::sin(t3697);
        int t3700 = (int)t3694;
        int t3701 = (int)t3695;
        int t3702 = t2826 + t3700;
        float t3703 = memory[176319988 + t3702];
        int t3704 = t2826 + t3700;
        int t3705 = t3704 + 1024;
        float t3706 = memory[176319988 + t3705];
        int t3707 = t2826 + t3701;
        float t3708 = memory[176319988 + t3707];
        int t3709 = t2826 + t3701;
        int t3710 = t3709 + 1024;
        float t3711 = memory[176319988 + t3710];
        float t3712 = t3698 * t3708;
        float t3713 = t3699 * t3711;
        float t3714 = t3712 - t3713;
        float t3715 = t3698 * t3711;
        float t3716 = t3699 * t3708;
        float t3717 = t3715 + t3716;
        int t3718 = t2826 + t3700;
        float t3719 = t3703 + t3714;
        memory[176319988 + t3718] = t3719;
        int t3721 = t2826 + t3700;
        int t3722 = t3721 + 1024;
        float t3723 = t3706 + t3717;
        memory[176319988 + t3722] = t3723;
        int t3725 = t2826 + t3701;
        float t3726 = t3703 - t3714;
        memory[176319988 + t3725] = t3726;
        int t3728 = t2826 + t3701;
        int t3729 = t3728 + 1024;
        float t3730 = t3706 - t3717;
        memory[176319988 + t3729] = t3730;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3733 = 0; _pr3733 < 512; _pr3733++) {
        float t3734 = (float)_pr3733;
        float t3735 = (t3734 * 0.015625);
        float t3736 = metal::floor(t3735);
        float t3737 = t3736 * 64.0;
        float t3738 = t3734 - t3737;
        float t3739 = t3736 * 128.0;
        float t3740 = t3739 + t3738;
        float t3741 = t3740 + 64.0;
        float t3742 = 6.283185 * t3738;
        float t3743 = (t3742 * 0.0078125);
        float t3744 = metal::cos(t3743);
        float t3745 = metal::sin(t3743);
        int t3746 = (int)t3740;
        int t3747 = (int)t3741;
        int t3748 = t2826 + t3746;
        float t3749 = memory[176319988 + t3748];
        int t3750 = t2826 + t3746;
        int t3751 = t3750 + 1024;
        float t3752 = memory[176319988 + t3751];
        int t3753 = t2826 + t3747;
        float t3754 = memory[176319988 + t3753];
        int t3755 = t2826 + t3747;
        int t3756 = t3755 + 1024;
        float t3757 = memory[176319988 + t3756];
        float t3758 = t3744 * t3754;
        float t3759 = t3745 * t3757;
        float t3760 = t3758 - t3759;
        float t3761 = t3744 * t3757;
        float t3762 = t3745 * t3754;
        float t3763 = t3761 + t3762;
        int t3764 = t2826 + t3746;
        float t3765 = t3749 + t3760;
        memory[176319988 + t3764] = t3765;
        int t3767 = t2826 + t3746;
        int t3768 = t3767 + 1024;
        float t3769 = t3752 + t3763;
        memory[176319988 + t3768] = t3769;
        int t3771 = t2826 + t3747;
        float t3772 = t3749 - t3760;
        memory[176319988 + t3771] = t3772;
        int t3774 = t2826 + t3747;
        int t3775 = t3774 + 1024;
        float t3776 = t3752 - t3763;
        memory[176319988 + t3775] = t3776;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3779 = 0; _pr3779 < 512; _pr3779++) {
        float t3780 = (float)_pr3779;
        float t3781 = (t3780 * 0.0078125);
        float t3782 = metal::floor(t3781);
        float t3783 = t3782 * 128.0;
        float t3784 = t3780 - t3783;
        float t3785 = t3782 * 256.0;
        float t3786 = t3785 + t3784;
        float t3787 = t3786 + 128.0;
        float t3788 = 6.283185 * t3784;
        float t3789 = (t3788 * 0.00390625);
        float t3790 = metal::cos(t3789);
        float t3791 = metal::sin(t3789);
        int t3792 = (int)t3786;
        int t3793 = (int)t3787;
        int t3794 = t2826 + t3792;
        float t3795 = memory[176319988 + t3794];
        int t3796 = t2826 + t3792;
        int t3797 = t3796 + 1024;
        float t3798 = memory[176319988 + t3797];
        int t3799 = t2826 + t3793;
        float t3800 = memory[176319988 + t3799];
        int t3801 = t2826 + t3793;
        int t3802 = t3801 + 1024;
        float t3803 = memory[176319988 + t3802];
        float t3804 = t3790 * t3800;
        float t3805 = t3791 * t3803;
        float t3806 = t3804 - t3805;
        float t3807 = t3790 * t3803;
        float t3808 = t3791 * t3800;
        float t3809 = t3807 + t3808;
        int t3810 = t2826 + t3792;
        float t3811 = t3795 + t3806;
        memory[176319988 + t3810] = t3811;
        int t3813 = t2826 + t3792;
        int t3814 = t3813 + 1024;
        float t3815 = t3798 + t3809;
        memory[176319988 + t3814] = t3815;
        int t3817 = t2826 + t3793;
        float t3818 = t3795 - t3806;
        memory[176319988 + t3817] = t3818;
        int t3820 = t2826 + t3793;
        int t3821 = t3820 + 1024;
        float t3822 = t3798 - t3809;
        memory[176319988 + t3821] = t3822;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3825 = 0; _pr3825 < 512; _pr3825++) {
        float t3826 = (float)_pr3825;
        float t3827 = (t3826 * 0.00390625);
        float t3828 = metal::floor(t3827);
        float t3829 = t3828 * 256.0;
        float t3830 = t3826 - t3829;
        float t3831 = t3828 * 512.0;
        float t3832 = t3831 + t3830;
        float t3833 = t3832 + 256.0;
        float t3834 = 6.283185 * t3830;
        float t3835 = (t3834 * 0.001953125);
        float t3836 = metal::cos(t3835);
        float t3837 = metal::sin(t3835);
        int t3838 = (int)t3832;
        int t3839 = (int)t3833;
        int t3840 = t2826 + t3838;
        float t3841 = memory[176319988 + t3840];
        int t3842 = t2826 + t3838;
        int t3843 = t3842 + 1024;
        float t3844 = memory[176319988 + t3843];
        int t3845 = t2826 + t3839;
        float t3846 = memory[176319988 + t3845];
        int t3847 = t2826 + t3839;
        int t3848 = t3847 + 1024;
        float t3849 = memory[176319988 + t3848];
        float t3850 = t3836 * t3846;
        float t3851 = t3837 * t3849;
        float t3852 = t3850 - t3851;
        float t3853 = t3836 * t3849;
        float t3854 = t3837 * t3846;
        float t3855 = t3853 + t3854;
        int t3856 = t2826 + t3838;
        float t3857 = t3841 + t3852;
        memory[176319988 + t3856] = t3857;
        int t3859 = t2826 + t3838;
        int t3860 = t3859 + 1024;
        float t3861 = t3844 + t3855;
        memory[176319988 + t3860] = t3861;
        int t3863 = t2826 + t3839;
        float t3864 = t3841 - t3852;
        memory[176319988 + t3863] = t3864;
        int t3866 = t2826 + t3839;
        int t3867 = t3866 + 1024;
        float t3868 = t3844 - t3855;
        memory[176319988 + t3867] = t3868;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3871 = 0; _pr3871 < 512; _pr3871++) {
        float t3872 = (float)_pr3871;
        float t3873 = (t3872 * 0.001953125);
        float t3874 = metal::floor(t3873);
        float t3875 = t3874 * 512.0;
        float t3876 = t3872 - t3875;
        float t3877 = t3874 * 1024.0;
        float t3878 = t3877 + t3876;
        float t3879 = t3878 + 512.0;
        float t3880 = 6.283185 * t3876;
        float t3881 = (t3880 * 0.0009765625);
        float t3882 = metal::cos(t3881);
        float t3883 = metal::sin(t3881);
        int t3884 = (int)t3878;
        int t3885 = (int)t3879;
        int t3886 = t2826 + t3884;
        float t3887 = memory[176319988 + t3886];
        int t3888 = t2826 + t3884;
        int t3889 = t3888 + 1024;
        float t3890 = memory[176319988 + t3889];
        int t3891 = t2826 + t3885;
        float t3892 = memory[176319988 + t3891];
        int t3893 = t2826 + t3885;
        int t3894 = t3893 + 1024;
        float t3895 = memory[176319988 + t3894];
        float t3896 = t3882 * t3892;
        float t3897 = t3883 * t3895;
        float t3898 = t3896 - t3897;
        float t3899 = t3882 * t3895;
        float t3900 = t3883 * t3892;
        float t3901 = t3899 + t3900;
        int t3902 = t2826 + t3884;
        float t3903 = t3887 + t3898;
        memory[176319988 + t3902] = t3903;
        int t3905 = t2826 + t3884;
        int t3906 = t3905 + 1024;
        float t3907 = t3890 + t3901;
        memory[176319988 + t3906] = t3907;
        int t3909 = t2826 + t3885;
        float t3910 = t3887 - t3898;
        memory[176319988 + t3909] = t3910;
        int t3912 = t2826 + t3885;
        int t3913 = t3912 + 1024;
        float t3914 = t3890 - t3901;
        memory[176319988 + t3913] = t3914;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3917 = 0; _pr3917 < 1024; _pr3917++) {
        int t3918 = t2826 + _pr3917;
        float t3919 = memory[176319988 + t3918];
        float t3920 = t3919 * 1.9036306e-06;
        float t3921 = memory[52788 + (int)_pr3917];
        int t3922 = t2827 + _pr3917;
        float t3923 = t3920 * t3921;
        memory[83996148 + t3922] = t3923;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t3927 = t[14*frameCount + id] > 0.0;
    if (t3927) {
      for (uint _pr3929 = 0; _pr3929 < 1024; _pr3929++) {
        int t3930 = t2827 + _pr3929;
        memory[50441716 + t3930] = 0.0;
        int t3932 = t2827 + _pr3929;
        memory[83996148 + t3932] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 22
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_22(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3953), value: global(3953)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1589) - handled in variable access */
    int t3936 = id;
    float t3937 = 0.0;
    for (uint t3938 = 0; t3938 < 1024; t3938++) {
      float t3939 = (float)t3938;
      float t3940 = (float)t3936;
      float t3941 = t3940 + t3939;
      int t3942 = 1023 - t3938;
      float t3943 = frameCount - 1.0;
      float t3944 = metal::min(t3941, t3943);
      int t3945 = (int)t3944;
      int t3946 = t3945 * 1024;
      int t3947 = t3946 + t3942;
      float t3948 = memory[50441716 + t3947];
      float t3949 = t3941 < frameCount;
      float t3950 = metal::select(0.0, t3948, t3949 > 0.0);
      float t3951 = t3937 + t3950;
      t3937 = t3951;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[17*frameCount + id] = (t3937 * 0.0013797212);
  }
  #pragma clang diagnostic pop
}



// KERNEL 23
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_23(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3971), value: global(3971)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1589) - handled in variable access */
    int t3954 = id;
    float t3955 = 0.0;
    for (uint t3956 = 0; t3956 < 1024; t3956++) {
      float t3957 = (float)t3956;
      float t3958 = (float)t3954;
      float t3959 = t3958 + t3957;
      int t3960 = 1023 - t3956;
      float t3961 = frameCount - 1.0;
      float t3962 = metal::min(t3959, t3961);
      int t3963 = (int)t3962;
      int t3964 = t3963 * 1024;
      int t3965 = t3964 + t3960;
      float t3966 = memory[83996148 + t3965];
      float t3967 = t3959 < frameCount;
      float t3968 = metal::select(0.0, t3966, t3967 > 0.0);
      float t3969 = t3955 + t3968;
      t3955 = t3969;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[18*frameCount + id] = (t3955 * 0.0013797212);
  }
  #pragma clang diagnostic pop
}



// KERNEL 24
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_24(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5738 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5738)) {
    /* loadGlobal(549) - handled in variable access */
    int t3972 = id;
    int t3973 = t3972 / 61;
    uint _frameIndex = (uint)(t3973);
    int t3974 = t3973 * 61;
    int t3975 = t3972 - t3974;
    float t3976 = (t[12*frameCount + _frameIndex] * 3.7252903e-09);
    float t3977 = -0.5 * t3976;
  }
  #pragma clang diagnostic pop
}



// KERNEL 25
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_25(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(528) - handled in variable access */
    /* loadGlobal(527) - handled in variable access */
    /* loadGlobal(474) - handled in variable access */
    int t3978 = id;
    int t3979 = t3978 * 1024;
    int t3980 = t3978 * 257;
    int t3981 = t3978 * 1024;
    float t3982 = t[11*frameCount + id] == 0.0;
    if (t3982) {
      for (uint _pr3984 = 0; _pr3984 < 257; _pr3984++) {
        int t3985 = t3980 + _pr3984;
        float t3986 = memory[37809652 + t3985];
        int t3987 = t3980 + _pr3984;
        float t3988 = memory[42020340 + t3987];
        int t3989 = t3979 + _pr3984;
        float t3990 = memory[4255220 + t3989];
        int t3991 = t3979 + _pr3984;
        int t3992 = t3991 + 512;
        float t3993 = memory[4255220 + t3992];
        int t3994 = t3979 + _pr3984;
        float t3995 = memory[21032436 + t3994];
        int t3996 = t3979 + _pr3984;
        int t3997 = t3996 + 512;
        float t3998 = memory[21032436 + t3997];
        float t3999 = t3986 - t3988;
        float t4000 = 2.0 * t3999;
        float t4001 = t4000 * 3.0517578e-05;
        float t4002 = t3986 - t3988;
        float t4003 = -2.0 * t4002;
        float t4004 = t4003 * 3.0517578e-05;
        float t4005 = metal::max(t3986, 1e-08);
        float t4006 = metal::max(t3988, 1e-08);
        float t4007 = t4001 * t3990;
        float t4008 = t4007 / t4005;
        float t4009 = t4001 * t3993;
        float t4010 = t4009 / t4005;
        float t4011 = t4004 * t3995;
        float t4012 = t4011 / t4006;
        float t4013 = t4004 * t3998;
        float t4014 = t4013 / t4006;
        int t4015 = t3981 + _pr3984;
        memory[50441716 + t4015] = t4008;
        int t4017 = t3981 + _pr3984;
        int t4018 = t4017 + 512;
        memory[50441716 + t4018] = t4010;
        int t4020 = t3981 + _pr3984;
        memory[83996148 + t4020] = t4012;
        int t4022 = t3981 + _pr3984;
        int t4023 = t4022 + 512;
        memory[83996148 + t4023] = t4014;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4026 = 0; _pr4026 < 255; _pr4026++) {
        int t4027 = _pr4026 + 257;
        int t4028 = t3981 + t4027;
        memory[50441716 + t4028] = 0.0;
        int t4030 = t3981 + t4027;
        int t4031 = t4030 + 512;
        memory[50441716 + t4031] = 0.0;
        int t4033 = t3981 + t4027;
        memory[83996148 + t4033] = 0.0;
        int t4035 = t3981 + t4027;
        int t4036 = t4035 + 512;
        memory[83996148 + t4036] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 26
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_26(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(528) - handled in variable access */
    int t4040 = id;
    int t4041 = t4040 * 1024;
    int t4042 = t4040 * 512;
    float t4043 = t[11*frameCount + id] == 0.0;
    if (t4043) {
      for (uint t4045 = 0; t4045 < 512; t4045++) {
        float t4046 = (float)t4045;
        float t4047 = (t4046 - metal::floor(t4046 / 2.0) * 2.0);
        float t4048 = t4047;
        float t4049 = (t4046 * 0.5);
        float t4050 = metal::floor(t4049);
        float t4051 = t4048 * 2.0;
        float t4052 = (t4050 - metal::floor(t4050 / 2.0) * 2.0);
        float t4053 = t4051 + t4052;
        float t4054 = (t4050 * 0.5);
        float t4055 = metal::floor(t4054);
        float t4056 = t4053 * 2.0;
        float t4057 = (t4055 - metal::floor(t4055 / 2.0) * 2.0);
        float t4058 = t4056 + t4057;
        float t4059 = (t4055 * 0.5);
        float t4060 = metal::floor(t4059);
        float t4061 = t4058 * 2.0;
        float t4062 = (t4060 - metal::floor(t4060 / 2.0) * 2.0);
        float t4063 = t4061 + t4062;
        float t4064 = (t4060 * 0.5);
        float t4065 = metal::floor(t4064);
        float t4066 = t4063 * 2.0;
        float t4067 = (t4065 - metal::floor(t4065 / 2.0) * 2.0);
        float t4068 = t4066 + t4067;
        float t4069 = (t4065 * 0.5);
        float t4070 = metal::floor(t4069);
        float t4071 = t4068 * 2.0;
        float t4072 = (t4070 - metal::floor(t4070 / 2.0) * 2.0);
        float t4073 = t4071 + t4072;
        float t4074 = (t4070 * 0.5);
        float t4075 = metal::floor(t4074);
        float t4076 = t4073 * 2.0;
        float t4077 = (t4075 - metal::floor(t4075 / 2.0) * 2.0);
        float t4078 = t4076 + t4077;
        float t4079 = (t4075 * 0.5);
        float t4080 = metal::floor(t4079);
        float t4081 = t4078 * 2.0;
        float t4082 = (t4080 - metal::floor(t4080 / 2.0) * 2.0);
        float t4083 = t4081 + t4082;
        float t4084 = (t4080 * 0.5);
        float t4085 = metal::floor(t4084);
        float t4086 = t4083 * 2.0;
        float t4087 = (t4085 - metal::floor(t4085 / 2.0) * 2.0);
        float t4088 = t4086 + t4087;
        float t4089 = (t4085 * 0.5);
        float t4090 = metal::floor(t4089);
        float t4091 = (float)t4045;
        float t4092 = t4091 < t4088;
        int t4093 = (int)t4088;
        int t4094 = t4041 + t4045;
        float t4095 = memory[50441716 + t4094];
        int t4096 = t4041 + t4045;
        int t4097 = t4096 + 512;
        float t4098 = memory[50441716 + t4097];
        int t4099 = t4041 + t4093;
        float t4100 = memory[50441716 + t4099];
        int t4101 = t4041 + t4093;
        int t4102 = t4101 + 512;
        float t4103 = memory[50441716 + t4102];
        float t4104 = metal::select(t4095, t4100, t4092 > 0.0);
        float t4105 = metal::select(t4098, t4103, t4092 > 0.0);
        float t4106 = metal::select(t4100, t4095, t4092 > 0.0);
        float t4107 = metal::select(t4103, t4098, t4092 > 0.0);
        int t4108 = t4041 + t4045;
        memory[50441716 + t4108] = t4104;
        int t4110 = t4041 + t4045;
        int t4111 = t4110 + 512;
        memory[50441716 + t4111] = t4105;
        int t4113 = t4041 + t4093;
        memory[50441716 + t4113] = t4106;
        int t4115 = t4041 + t4093;
        int t4116 = t4115 + 512;
        memory[50441716 + t4116] = t4107;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4119 = 0; _pr4119 < 256; _pr4119++) {
        float t4120 = (float)_pr4119;
        float t4121 = t4120;
        float t4122 = metal::floor(t4121);
        float t4123 = t4122;
        float t4124 = t4120 - t4123;
        float t4125 = t4122 * 2.0;
        float t4126 = t4125 + t4124;
        float t4127 = t4126 + 1.0;
        float t4128 = 6.283185 * t4124;
        float t4129 = (t4128 * 0.5);
        float t4130 = metal::cos(t4129);
        float t4131 = metal::sin(t4129);
        int t4132 = (int)t4126;
        int t4133 = (int)t4127;
        int t4134 = t4041 + t4132;
        float t4135 = memory[50441716 + t4134];
        int t4136 = t4041 + t4132;
        int t4137 = t4136 + 512;
        float t4138 = memory[50441716 + t4137];
        int t4139 = t4041 + t4133;
        float t4140 = memory[50441716 + t4139];
        int t4141 = t4041 + t4133;
        int t4142 = t4141 + 512;
        float t4143 = memory[50441716 + t4142];
        float t4144 = t4130 * t4140;
        float t4145 = t4131 * t4143;
        float t4146 = t4144 - t4145;
        float t4147 = t4130 * t4143;
        float t4148 = t4131 * t4140;
        float t4149 = t4147 + t4148;
        int t4150 = t4041 + t4132;
        float t4151 = t4135 + t4146;
        memory[50441716 + t4150] = t4151;
        int t4153 = t4041 + t4132;
        int t4154 = t4153 + 512;
        float t4155 = t4138 + t4149;
        memory[50441716 + t4154] = t4155;
        int t4157 = t4041 + t4133;
        float t4158 = t4135 - t4146;
        memory[50441716 + t4157] = t4158;
        int t4160 = t4041 + t4133;
        int t4161 = t4160 + 512;
        float t4162 = t4138 - t4149;
        memory[50441716 + t4161] = t4162;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4165 = 0; _pr4165 < 256; _pr4165++) {
        float t4166 = (float)_pr4165;
        float t4167 = (t4166 * 0.5);
        float t4168 = metal::floor(t4167);
        float t4169 = t4168 * 2.0;
        float t4170 = t4166 - t4169;
        float t4171 = t4168 * 4.0;
        float t4172 = t4171 + t4170;
        float t4173 = t4172 + 2.0;
        float t4174 = 6.283185 * t4170;
        float t4175 = (t4174 * 0.25);
        float t4176 = metal::cos(t4175);
        float t4177 = metal::sin(t4175);
        int t4178 = (int)t4172;
        int t4179 = (int)t4173;
        int t4180 = t4041 + t4178;
        float t4181 = memory[50441716 + t4180];
        int t4182 = t4041 + t4178;
        int t4183 = t4182 + 512;
        float t4184 = memory[50441716 + t4183];
        int t4185 = t4041 + t4179;
        float t4186 = memory[50441716 + t4185];
        int t4187 = t4041 + t4179;
        int t4188 = t4187 + 512;
        float t4189 = memory[50441716 + t4188];
        float t4190 = t4176 * t4186;
        float t4191 = t4177 * t4189;
        float t4192 = t4190 - t4191;
        float t4193 = t4176 * t4189;
        float t4194 = t4177 * t4186;
        float t4195 = t4193 + t4194;
        int t4196 = t4041 + t4178;
        float t4197 = t4181 + t4192;
        memory[50441716 + t4196] = t4197;
        int t4199 = t4041 + t4178;
        int t4200 = t4199 + 512;
        float t4201 = t4184 + t4195;
        memory[50441716 + t4200] = t4201;
        int t4203 = t4041 + t4179;
        float t4204 = t4181 - t4192;
        memory[50441716 + t4203] = t4204;
        int t4206 = t4041 + t4179;
        int t4207 = t4206 + 512;
        float t4208 = t4184 - t4195;
        memory[50441716 + t4207] = t4208;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4211 = 0; _pr4211 < 256; _pr4211++) {
        float t4212 = (float)_pr4211;
        float t4213 = (t4212 * 0.25);
        float t4214 = metal::floor(t4213);
        float t4215 = t4214 * 4.0;
        float t4216 = t4212 - t4215;
        float t4217 = t4214 * 8.0;
        float t4218 = t4217 + t4216;
        float t4219 = t4218 + 4.0;
        float t4220 = 6.283185 * t4216;
        float t4221 = (t4220 * 0.125);
        float t4222 = metal::cos(t4221);
        float t4223 = metal::sin(t4221);
        int t4224 = (int)t4218;
        int t4225 = (int)t4219;
        int t4226 = t4041 + t4224;
        float t4227 = memory[50441716 + t4226];
        int t4228 = t4041 + t4224;
        int t4229 = t4228 + 512;
        float t4230 = memory[50441716 + t4229];
        int t4231 = t4041 + t4225;
        float t4232 = memory[50441716 + t4231];
        int t4233 = t4041 + t4225;
        int t4234 = t4233 + 512;
        float t4235 = memory[50441716 + t4234];
        float t4236 = t4222 * t4232;
        float t4237 = t4223 * t4235;
        float t4238 = t4236 - t4237;
        float t4239 = t4222 * t4235;
        float t4240 = t4223 * t4232;
        float t4241 = t4239 + t4240;
        int t4242 = t4041 + t4224;
        float t4243 = t4227 + t4238;
        memory[50441716 + t4242] = t4243;
        int t4245 = t4041 + t4224;
        int t4246 = t4245 + 512;
        float t4247 = t4230 + t4241;
        memory[50441716 + t4246] = t4247;
        int t4249 = t4041 + t4225;
        float t4250 = t4227 - t4238;
        memory[50441716 + t4249] = t4250;
        int t4252 = t4041 + t4225;
        int t4253 = t4252 + 512;
        float t4254 = t4230 - t4241;
        memory[50441716 + t4253] = t4254;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4257 = 0; _pr4257 < 256; _pr4257++) {
        float t4258 = (float)_pr4257;
        float t4259 = (t4258 * 0.125);
        float t4260 = metal::floor(t4259);
        float t4261 = t4260 * 8.0;
        float t4262 = t4258 - t4261;
        float t4263 = t4260 * 16.0;
        float t4264 = t4263 + t4262;
        float t4265 = t4264 + 8.0;
        float t4266 = 6.283185 * t4262;
        float t4267 = (t4266 * 0.0625);
        float t4268 = metal::cos(t4267);
        float t4269 = metal::sin(t4267);
        int t4270 = (int)t4264;
        int t4271 = (int)t4265;
        int t4272 = t4041 + t4270;
        float t4273 = memory[50441716 + t4272];
        int t4274 = t4041 + t4270;
        int t4275 = t4274 + 512;
        float t4276 = memory[50441716 + t4275];
        int t4277 = t4041 + t4271;
        float t4278 = memory[50441716 + t4277];
        int t4279 = t4041 + t4271;
        int t4280 = t4279 + 512;
        float t4281 = memory[50441716 + t4280];
        float t4282 = t4268 * t4278;
        float t4283 = t4269 * t4281;
        float t4284 = t4282 - t4283;
        float t4285 = t4268 * t4281;
        float t4286 = t4269 * t4278;
        float t4287 = t4285 + t4286;
        int t4288 = t4041 + t4270;
        float t4289 = t4273 + t4284;
        memory[50441716 + t4288] = t4289;
        int t4291 = t4041 + t4270;
        int t4292 = t4291 + 512;
        float t4293 = t4276 + t4287;
        memory[50441716 + t4292] = t4293;
        int t4295 = t4041 + t4271;
        float t4296 = t4273 - t4284;
        memory[50441716 + t4295] = t4296;
        int t4298 = t4041 + t4271;
        int t4299 = t4298 + 512;
        float t4300 = t4276 - t4287;
        memory[50441716 + t4299] = t4300;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4303 = 0; _pr4303 < 256; _pr4303++) {
        float t4304 = (float)_pr4303;
        float t4305 = (t4304 * 0.0625);
        float t4306 = metal::floor(t4305);
        float t4307 = t4306 * 16.0;
        float t4308 = t4304 - t4307;
        float t4309 = t4306 * 32.0;
        float t4310 = t4309 + t4308;
        float t4311 = t4310 + 16.0;
        float t4312 = 6.283185 * t4308;
        float t4313 = (t4312 * 0.03125);
        float t4314 = metal::cos(t4313);
        float t4315 = metal::sin(t4313);
        int t4316 = (int)t4310;
        int t4317 = (int)t4311;
        int t4318 = t4041 + t4316;
        float t4319 = memory[50441716 + t4318];
        int t4320 = t4041 + t4316;
        int t4321 = t4320 + 512;
        float t4322 = memory[50441716 + t4321];
        int t4323 = t4041 + t4317;
        float t4324 = memory[50441716 + t4323];
        int t4325 = t4041 + t4317;
        int t4326 = t4325 + 512;
        float t4327 = memory[50441716 + t4326];
        float t4328 = t4314 * t4324;
        float t4329 = t4315 * t4327;
        float t4330 = t4328 - t4329;
        float t4331 = t4314 * t4327;
        float t4332 = t4315 * t4324;
        float t4333 = t4331 + t4332;
        int t4334 = t4041 + t4316;
        float t4335 = t4319 + t4330;
        memory[50441716 + t4334] = t4335;
        int t4337 = t4041 + t4316;
        int t4338 = t4337 + 512;
        float t4339 = t4322 + t4333;
        memory[50441716 + t4338] = t4339;
        int t4341 = t4041 + t4317;
        float t4342 = t4319 - t4330;
        memory[50441716 + t4341] = t4342;
        int t4344 = t4041 + t4317;
        int t4345 = t4344 + 512;
        float t4346 = t4322 - t4333;
        memory[50441716 + t4345] = t4346;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4349 = 0; _pr4349 < 256; _pr4349++) {
        float t4350 = (float)_pr4349;
        float t4351 = (t4350 * 0.03125);
        float t4352 = metal::floor(t4351);
        float t4353 = t4352 * 32.0;
        float t4354 = t4350 - t4353;
        float t4355 = t4352 * 64.0;
        float t4356 = t4355 + t4354;
        float t4357 = t4356 + 32.0;
        float t4358 = 6.283185 * t4354;
        float t4359 = (t4358 * 0.015625);
        float t4360 = metal::cos(t4359);
        float t4361 = metal::sin(t4359);
        int t4362 = (int)t4356;
        int t4363 = (int)t4357;
        int t4364 = t4041 + t4362;
        float t4365 = memory[50441716 + t4364];
        int t4366 = t4041 + t4362;
        int t4367 = t4366 + 512;
        float t4368 = memory[50441716 + t4367];
        int t4369 = t4041 + t4363;
        float t4370 = memory[50441716 + t4369];
        int t4371 = t4041 + t4363;
        int t4372 = t4371 + 512;
        float t4373 = memory[50441716 + t4372];
        float t4374 = t4360 * t4370;
        float t4375 = t4361 * t4373;
        float t4376 = t4374 - t4375;
        float t4377 = t4360 * t4373;
        float t4378 = t4361 * t4370;
        float t4379 = t4377 + t4378;
        int t4380 = t4041 + t4362;
        float t4381 = t4365 + t4376;
        memory[50441716 + t4380] = t4381;
        int t4383 = t4041 + t4362;
        int t4384 = t4383 + 512;
        float t4385 = t4368 + t4379;
        memory[50441716 + t4384] = t4385;
        int t4387 = t4041 + t4363;
        float t4388 = t4365 - t4376;
        memory[50441716 + t4387] = t4388;
        int t4390 = t4041 + t4363;
        int t4391 = t4390 + 512;
        float t4392 = t4368 - t4379;
        memory[50441716 + t4391] = t4392;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4395 = 0; _pr4395 < 256; _pr4395++) {
        float t4396 = (float)_pr4395;
        float t4397 = (t4396 * 0.015625);
        float t4398 = metal::floor(t4397);
        float t4399 = t4398 * 64.0;
        float t4400 = t4396 - t4399;
        float t4401 = t4398 * 128.0;
        float t4402 = t4401 + t4400;
        float t4403 = t4402 + 64.0;
        float t4404 = 6.283185 * t4400;
        float t4405 = (t4404 * 0.0078125);
        float t4406 = metal::cos(t4405);
        float t4407 = metal::sin(t4405);
        int t4408 = (int)t4402;
        int t4409 = (int)t4403;
        int t4410 = t4041 + t4408;
        float t4411 = memory[50441716 + t4410];
        int t4412 = t4041 + t4408;
        int t4413 = t4412 + 512;
        float t4414 = memory[50441716 + t4413];
        int t4415 = t4041 + t4409;
        float t4416 = memory[50441716 + t4415];
        int t4417 = t4041 + t4409;
        int t4418 = t4417 + 512;
        float t4419 = memory[50441716 + t4418];
        float t4420 = t4406 * t4416;
        float t4421 = t4407 * t4419;
        float t4422 = t4420 - t4421;
        float t4423 = t4406 * t4419;
        float t4424 = t4407 * t4416;
        float t4425 = t4423 + t4424;
        int t4426 = t4041 + t4408;
        float t4427 = t4411 + t4422;
        memory[50441716 + t4426] = t4427;
        int t4429 = t4041 + t4408;
        int t4430 = t4429 + 512;
        float t4431 = t4414 + t4425;
        memory[50441716 + t4430] = t4431;
        int t4433 = t4041 + t4409;
        float t4434 = t4411 - t4422;
        memory[50441716 + t4433] = t4434;
        int t4436 = t4041 + t4409;
        int t4437 = t4436 + 512;
        float t4438 = t4414 - t4425;
        memory[50441716 + t4437] = t4438;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4441 = 0; _pr4441 < 256; _pr4441++) {
        float t4442 = (float)_pr4441;
        float t4443 = (t4442 * 0.0078125);
        float t4444 = metal::floor(t4443);
        float t4445 = t4444 * 128.0;
        float t4446 = t4442 - t4445;
        float t4447 = t4444 * 256.0;
        float t4448 = t4447 + t4446;
        float t4449 = t4448 + 128.0;
        float t4450 = 6.283185 * t4446;
        float t4451 = (t4450 * 0.00390625);
        float t4452 = metal::cos(t4451);
        float t4453 = metal::sin(t4451);
        int t4454 = (int)t4448;
        int t4455 = (int)t4449;
        int t4456 = t4041 + t4454;
        float t4457 = memory[50441716 + t4456];
        int t4458 = t4041 + t4454;
        int t4459 = t4458 + 512;
        float t4460 = memory[50441716 + t4459];
        int t4461 = t4041 + t4455;
        float t4462 = memory[50441716 + t4461];
        int t4463 = t4041 + t4455;
        int t4464 = t4463 + 512;
        float t4465 = memory[50441716 + t4464];
        float t4466 = t4452 * t4462;
        float t4467 = t4453 * t4465;
        float t4468 = t4466 - t4467;
        float t4469 = t4452 * t4465;
        float t4470 = t4453 * t4462;
        float t4471 = t4469 + t4470;
        int t4472 = t4041 + t4454;
        float t4473 = t4457 + t4468;
        memory[50441716 + t4472] = t4473;
        int t4475 = t4041 + t4454;
        int t4476 = t4475 + 512;
        float t4477 = t4460 + t4471;
        memory[50441716 + t4476] = t4477;
        int t4479 = t4041 + t4455;
        float t4480 = t4457 - t4468;
        memory[50441716 + t4479] = t4480;
        int t4482 = t4041 + t4455;
        int t4483 = t4482 + 512;
        float t4484 = t4460 - t4471;
        memory[50441716 + t4483] = t4484;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4487 = 0; _pr4487 < 256; _pr4487++) {
        float t4488 = (float)_pr4487;
        float t4489 = (t4488 * 0.00390625);
        float t4490 = metal::floor(t4489);
        float t4491 = t4490 * 256.0;
        float t4492 = t4488 - t4491;
        float t4493 = t4490 * 512.0;
        float t4494 = t4493 + t4492;
        float t4495 = t4494 + 256.0;
        float t4496 = 6.283185 * t4492;
        float t4497 = (t4496 * 0.001953125);
        float t4498 = metal::cos(t4497);
        float t4499 = metal::sin(t4497);
        int t4500 = (int)t4494;
        int t4501 = (int)t4495;
        int t4502 = t4041 + t4500;
        float t4503 = memory[50441716 + t4502];
        int t4504 = t4041 + t4500;
        int t4505 = t4504 + 512;
        float t4506 = memory[50441716 + t4505];
        int t4507 = t4041 + t4501;
        float t4508 = memory[50441716 + t4507];
        int t4509 = t4041 + t4501;
        int t4510 = t4509 + 512;
        float t4511 = memory[50441716 + t4510];
        float t4512 = t4498 * t4508;
        float t4513 = t4499 * t4511;
        float t4514 = t4512 - t4513;
        float t4515 = t4498 * t4511;
        float t4516 = t4499 * t4508;
        float t4517 = t4515 + t4516;
        int t4518 = t4041 + t4500;
        float t4519 = t4503 + t4514;
        memory[50441716 + t4518] = t4519;
        int t4521 = t4041 + t4500;
        int t4522 = t4521 + 512;
        float t4523 = t4506 + t4517;
        memory[50441716 + t4522] = t4523;
        int t4525 = t4041 + t4501;
        float t4526 = t4503 - t4514;
        memory[50441716 + t4525] = t4526;
        int t4528 = t4041 + t4501;
        int t4529 = t4528 + 512;
        float t4530 = t4506 - t4517;
        memory[50441716 + t4529] = t4530;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4533 = 0; _pr4533 < 512; _pr4533++) {
        int t4534 = t4041 + _pr4533;
        float t4535 = memory[50441716 + t4534];
        float t4536 = t4535 * 7.599708e-06;
        float t4537 = memory[25460 + (int)_pr4533];
        int t4538 = t4042 + _pr4533;
        float t4539 = t4536 * t4537;
        memory[117550580 + t4538] = t4539;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t4542 = 0; t4542 < 512; t4542++) {
        float t4543 = (float)t4542;
        float t4544 = (t4543 - metal::floor(t4543 / 2.0) * 2.0);
        float t4545 = t4544;
        float t4546 = (t4543 * 0.5);
        float t4547 = metal::floor(t4546);
        float t4548 = t4545 * 2.0;
        float t4549 = (t4547 - metal::floor(t4547 / 2.0) * 2.0);
        float t4550 = t4548 + t4549;
        float t4551 = (t4547 * 0.5);
        float t4552 = metal::floor(t4551);
        float t4553 = t4550 * 2.0;
        float t4554 = (t4552 - metal::floor(t4552 / 2.0) * 2.0);
        float t4555 = t4553 + t4554;
        float t4556 = (t4552 * 0.5);
        float t4557 = metal::floor(t4556);
        float t4558 = t4555 * 2.0;
        float t4559 = (t4557 - metal::floor(t4557 / 2.0) * 2.0);
        float t4560 = t4558 + t4559;
        float t4561 = (t4557 * 0.5);
        float t4562 = metal::floor(t4561);
        float t4563 = t4560 * 2.0;
        float t4564 = (t4562 - metal::floor(t4562 / 2.0) * 2.0);
        float t4565 = t4563 + t4564;
        float t4566 = (t4562 * 0.5);
        float t4567 = metal::floor(t4566);
        float t4568 = t4565 * 2.0;
        float t4569 = (t4567 - metal::floor(t4567 / 2.0) * 2.0);
        float t4570 = t4568 + t4569;
        float t4571 = (t4567 * 0.5);
        float t4572 = metal::floor(t4571);
        float t4573 = t4570 * 2.0;
        float t4574 = (t4572 - metal::floor(t4572 / 2.0) * 2.0);
        float t4575 = t4573 + t4574;
        float t4576 = (t4572 * 0.5);
        float t4577 = metal::floor(t4576);
        float t4578 = t4575 * 2.0;
        float t4579 = (t4577 - metal::floor(t4577 / 2.0) * 2.0);
        float t4580 = t4578 + t4579;
        float t4581 = (t4577 * 0.5);
        float t4582 = metal::floor(t4581);
        float t4583 = t4580 * 2.0;
        float t4584 = (t4582 - metal::floor(t4582 / 2.0) * 2.0);
        float t4585 = t4583 + t4584;
        float t4586 = (t4582 * 0.5);
        float t4587 = metal::floor(t4586);
        float t4588 = (float)t4542;
        float t4589 = t4588 < t4585;
        int t4590 = (int)t4585;
        int t4591 = t4041 + t4542;
        float t4592 = memory[83996148 + t4591];
        int t4593 = t4041 + t4542;
        int t4594 = t4593 + 512;
        float t4595 = memory[83996148 + t4594];
        int t4596 = t4041 + t4590;
        float t4597 = memory[83996148 + t4596];
        int t4598 = t4041 + t4590;
        int t4599 = t4598 + 512;
        float t4600 = memory[83996148 + t4599];
        float t4601 = metal::select(t4592, t4597, t4589 > 0.0);
        float t4602 = metal::select(t4595, t4600, t4589 > 0.0);
        float t4603 = metal::select(t4597, t4592, t4589 > 0.0);
        float t4604 = metal::select(t4600, t4595, t4589 > 0.0);
        int t4605 = t4041 + t4542;
        memory[83996148 + t4605] = t4601;
        int t4607 = t4041 + t4542;
        int t4608 = t4607 + 512;
        memory[83996148 + t4608] = t4602;
        int t4610 = t4041 + t4590;
        memory[83996148 + t4610] = t4603;
        int t4612 = t4041 + t4590;
        int t4613 = t4612 + 512;
        memory[83996148 + t4613] = t4604;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4616 = 0; _pr4616 < 256; _pr4616++) {
        float t4617 = (float)_pr4616;
        float t4618 = t4617;
        float t4619 = metal::floor(t4618);
        float t4620 = t4619;
        float t4621 = t4617 - t4620;
        float t4622 = t4619 * 2.0;
        float t4623 = t4622 + t4621;
        float t4624 = t4623 + 1.0;
        float t4625 = 6.283185 * t4621;
        float t4626 = (t4625 * 0.5);
        float t4627 = metal::cos(t4626);
        float t4628 = metal::sin(t4626);
        int t4629 = (int)t4623;
        int t4630 = (int)t4624;
        int t4631 = t4041 + t4629;
        float t4632 = memory[83996148 + t4631];
        int t4633 = t4041 + t4629;
        int t4634 = t4633 + 512;
        float t4635 = memory[83996148 + t4634];
        int t4636 = t4041 + t4630;
        float t4637 = memory[83996148 + t4636];
        int t4638 = t4041 + t4630;
        int t4639 = t4638 + 512;
        float t4640 = memory[83996148 + t4639];
        float t4641 = t4627 * t4637;
        float t4642 = t4628 * t4640;
        float t4643 = t4641 - t4642;
        float t4644 = t4627 * t4640;
        float t4645 = t4628 * t4637;
        float t4646 = t4644 + t4645;
        int t4647 = t4041 + t4629;
        float t4648 = t4632 + t4643;
        memory[83996148 + t4647] = t4648;
        int t4650 = t4041 + t4629;
        int t4651 = t4650 + 512;
        float t4652 = t4635 + t4646;
        memory[83996148 + t4651] = t4652;
        int t4654 = t4041 + t4630;
        float t4655 = t4632 - t4643;
        memory[83996148 + t4654] = t4655;
        int t4657 = t4041 + t4630;
        int t4658 = t4657 + 512;
        float t4659 = t4635 - t4646;
        memory[83996148 + t4658] = t4659;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4662 = 0; _pr4662 < 256; _pr4662++) {
        float t4663 = (float)_pr4662;
        float t4664 = (t4663 * 0.5);
        float t4665 = metal::floor(t4664);
        float t4666 = t4665 * 2.0;
        float t4667 = t4663 - t4666;
        float t4668 = t4665 * 4.0;
        float t4669 = t4668 + t4667;
        float t4670 = t4669 + 2.0;
        float t4671 = 6.283185 * t4667;
        float t4672 = (t4671 * 0.25);
        float t4673 = metal::cos(t4672);
        float t4674 = metal::sin(t4672);
        int t4675 = (int)t4669;
        int t4676 = (int)t4670;
        int t4677 = t4041 + t4675;
        float t4678 = memory[83996148 + t4677];
        int t4679 = t4041 + t4675;
        int t4680 = t4679 + 512;
        float t4681 = memory[83996148 + t4680];
        int t4682 = t4041 + t4676;
        float t4683 = memory[83996148 + t4682];
        int t4684 = t4041 + t4676;
        int t4685 = t4684 + 512;
        float t4686 = memory[83996148 + t4685];
        float t4687 = t4673 * t4683;
        float t4688 = t4674 * t4686;
        float t4689 = t4687 - t4688;
        float t4690 = t4673 * t4686;
        float t4691 = t4674 * t4683;
        float t4692 = t4690 + t4691;
        int t4693 = t4041 + t4675;
        float t4694 = t4678 + t4689;
        memory[83996148 + t4693] = t4694;
        int t4696 = t4041 + t4675;
        int t4697 = t4696 + 512;
        float t4698 = t4681 + t4692;
        memory[83996148 + t4697] = t4698;
        int t4700 = t4041 + t4676;
        float t4701 = t4678 - t4689;
        memory[83996148 + t4700] = t4701;
        int t4703 = t4041 + t4676;
        int t4704 = t4703 + 512;
        float t4705 = t4681 - t4692;
        memory[83996148 + t4704] = t4705;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4708 = 0; _pr4708 < 256; _pr4708++) {
        float t4709 = (float)_pr4708;
        float t4710 = (t4709 * 0.25);
        float t4711 = metal::floor(t4710);
        float t4712 = t4711 * 4.0;
        float t4713 = t4709 - t4712;
        float t4714 = t4711 * 8.0;
        float t4715 = t4714 + t4713;
        float t4716 = t4715 + 4.0;
        float t4717 = 6.283185 * t4713;
        float t4718 = (t4717 * 0.125);
        float t4719 = metal::cos(t4718);
        float t4720 = metal::sin(t4718);
        int t4721 = (int)t4715;
        int t4722 = (int)t4716;
        int t4723 = t4041 + t4721;
        float t4724 = memory[83996148 + t4723];
        int t4725 = t4041 + t4721;
        int t4726 = t4725 + 512;
        float t4727 = memory[83996148 + t4726];
        int t4728 = t4041 + t4722;
        float t4729 = memory[83996148 + t4728];
        int t4730 = t4041 + t4722;
        int t4731 = t4730 + 512;
        float t4732 = memory[83996148 + t4731];
        float t4733 = t4719 * t4729;
        float t4734 = t4720 * t4732;
        float t4735 = t4733 - t4734;
        float t4736 = t4719 * t4732;
        float t4737 = t4720 * t4729;
        float t4738 = t4736 + t4737;
        int t4739 = t4041 + t4721;
        float t4740 = t4724 + t4735;
        memory[83996148 + t4739] = t4740;
        int t4742 = t4041 + t4721;
        int t4743 = t4742 + 512;
        float t4744 = t4727 + t4738;
        memory[83996148 + t4743] = t4744;
        int t4746 = t4041 + t4722;
        float t4747 = t4724 - t4735;
        memory[83996148 + t4746] = t4747;
        int t4749 = t4041 + t4722;
        int t4750 = t4749 + 512;
        float t4751 = t4727 - t4738;
        memory[83996148 + t4750] = t4751;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4754 = 0; _pr4754 < 256; _pr4754++) {
        float t4755 = (float)_pr4754;
        float t4756 = (t4755 * 0.125);
        float t4757 = metal::floor(t4756);
        float t4758 = t4757 * 8.0;
        float t4759 = t4755 - t4758;
        float t4760 = t4757 * 16.0;
        float t4761 = t4760 + t4759;
        float t4762 = t4761 + 8.0;
        float t4763 = 6.283185 * t4759;
        float t4764 = (t4763 * 0.0625);
        float t4765 = metal::cos(t4764);
        float t4766 = metal::sin(t4764);
        int t4767 = (int)t4761;
        int t4768 = (int)t4762;
        int t4769 = t4041 + t4767;
        float t4770 = memory[83996148 + t4769];
        int t4771 = t4041 + t4767;
        int t4772 = t4771 + 512;
        float t4773 = memory[83996148 + t4772];
        int t4774 = t4041 + t4768;
        float t4775 = memory[83996148 + t4774];
        int t4776 = t4041 + t4768;
        int t4777 = t4776 + 512;
        float t4778 = memory[83996148 + t4777];
        float t4779 = t4765 * t4775;
        float t4780 = t4766 * t4778;
        float t4781 = t4779 - t4780;
        float t4782 = t4765 * t4778;
        float t4783 = t4766 * t4775;
        float t4784 = t4782 + t4783;
        int t4785 = t4041 + t4767;
        float t4786 = t4770 + t4781;
        memory[83996148 + t4785] = t4786;
        int t4788 = t4041 + t4767;
        int t4789 = t4788 + 512;
        float t4790 = t4773 + t4784;
        memory[83996148 + t4789] = t4790;
        int t4792 = t4041 + t4768;
        float t4793 = t4770 - t4781;
        memory[83996148 + t4792] = t4793;
        int t4795 = t4041 + t4768;
        int t4796 = t4795 + 512;
        float t4797 = t4773 - t4784;
        memory[83996148 + t4796] = t4797;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4800 = 0; _pr4800 < 256; _pr4800++) {
        float t4801 = (float)_pr4800;
        float t4802 = (t4801 * 0.0625);
        float t4803 = metal::floor(t4802);
        float t4804 = t4803 * 16.0;
        float t4805 = t4801 - t4804;
        float t4806 = t4803 * 32.0;
        float t4807 = t4806 + t4805;
        float t4808 = t4807 + 16.0;
        float t4809 = 6.283185 * t4805;
        float t4810 = (t4809 * 0.03125);
        float t4811 = metal::cos(t4810);
        float t4812 = metal::sin(t4810);
        int t4813 = (int)t4807;
        int t4814 = (int)t4808;
        int t4815 = t4041 + t4813;
        float t4816 = memory[83996148 + t4815];
        int t4817 = t4041 + t4813;
        int t4818 = t4817 + 512;
        float t4819 = memory[83996148 + t4818];
        int t4820 = t4041 + t4814;
        float t4821 = memory[83996148 + t4820];
        int t4822 = t4041 + t4814;
        int t4823 = t4822 + 512;
        float t4824 = memory[83996148 + t4823];
        float t4825 = t4811 * t4821;
        float t4826 = t4812 * t4824;
        float t4827 = t4825 - t4826;
        float t4828 = t4811 * t4824;
        float t4829 = t4812 * t4821;
        float t4830 = t4828 + t4829;
        int t4831 = t4041 + t4813;
        float t4832 = t4816 + t4827;
        memory[83996148 + t4831] = t4832;
        int t4834 = t4041 + t4813;
        int t4835 = t4834 + 512;
        float t4836 = t4819 + t4830;
        memory[83996148 + t4835] = t4836;
        int t4838 = t4041 + t4814;
        float t4839 = t4816 - t4827;
        memory[83996148 + t4838] = t4839;
        int t4841 = t4041 + t4814;
        int t4842 = t4841 + 512;
        float t4843 = t4819 - t4830;
        memory[83996148 + t4842] = t4843;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4846 = 0; _pr4846 < 256; _pr4846++) {
        float t4847 = (float)_pr4846;
        float t4848 = (t4847 * 0.03125);
        float t4849 = metal::floor(t4848);
        float t4850 = t4849 * 32.0;
        float t4851 = t4847 - t4850;
        float t4852 = t4849 * 64.0;
        float t4853 = t4852 + t4851;
        float t4854 = t4853 + 32.0;
        float t4855 = 6.283185 * t4851;
        float t4856 = (t4855 * 0.015625);
        float t4857 = metal::cos(t4856);
        float t4858 = metal::sin(t4856);
        int t4859 = (int)t4853;
        int t4860 = (int)t4854;
        int t4861 = t4041 + t4859;
        float t4862 = memory[83996148 + t4861];
        int t4863 = t4041 + t4859;
        int t4864 = t4863 + 512;
        float t4865 = memory[83996148 + t4864];
        int t4866 = t4041 + t4860;
        float t4867 = memory[83996148 + t4866];
        int t4868 = t4041 + t4860;
        int t4869 = t4868 + 512;
        float t4870 = memory[83996148 + t4869];
        float t4871 = t4857 * t4867;
        float t4872 = t4858 * t4870;
        float t4873 = t4871 - t4872;
        float t4874 = t4857 * t4870;
        float t4875 = t4858 * t4867;
        float t4876 = t4874 + t4875;
        int t4877 = t4041 + t4859;
        float t4878 = t4862 + t4873;
        memory[83996148 + t4877] = t4878;
        int t4880 = t4041 + t4859;
        int t4881 = t4880 + 512;
        float t4882 = t4865 + t4876;
        memory[83996148 + t4881] = t4882;
        int t4884 = t4041 + t4860;
        float t4885 = t4862 - t4873;
        memory[83996148 + t4884] = t4885;
        int t4887 = t4041 + t4860;
        int t4888 = t4887 + 512;
        float t4889 = t4865 - t4876;
        memory[83996148 + t4888] = t4889;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4892 = 0; _pr4892 < 256; _pr4892++) {
        float t4893 = (float)_pr4892;
        float t4894 = (t4893 * 0.015625);
        float t4895 = metal::floor(t4894);
        float t4896 = t4895 * 64.0;
        float t4897 = t4893 - t4896;
        float t4898 = t4895 * 128.0;
        float t4899 = t4898 + t4897;
        float t4900 = t4899 + 64.0;
        float t4901 = 6.283185 * t4897;
        float t4902 = (t4901 * 0.0078125);
        float t4903 = metal::cos(t4902);
        float t4904 = metal::sin(t4902);
        int t4905 = (int)t4899;
        int t4906 = (int)t4900;
        int t4907 = t4041 + t4905;
        float t4908 = memory[83996148 + t4907];
        int t4909 = t4041 + t4905;
        int t4910 = t4909 + 512;
        float t4911 = memory[83996148 + t4910];
        int t4912 = t4041 + t4906;
        float t4913 = memory[83996148 + t4912];
        int t4914 = t4041 + t4906;
        int t4915 = t4914 + 512;
        float t4916 = memory[83996148 + t4915];
        float t4917 = t4903 * t4913;
        float t4918 = t4904 * t4916;
        float t4919 = t4917 - t4918;
        float t4920 = t4903 * t4916;
        float t4921 = t4904 * t4913;
        float t4922 = t4920 + t4921;
        int t4923 = t4041 + t4905;
        float t4924 = t4908 + t4919;
        memory[83996148 + t4923] = t4924;
        int t4926 = t4041 + t4905;
        int t4927 = t4926 + 512;
        float t4928 = t4911 + t4922;
        memory[83996148 + t4927] = t4928;
        int t4930 = t4041 + t4906;
        float t4931 = t4908 - t4919;
        memory[83996148 + t4930] = t4931;
        int t4933 = t4041 + t4906;
        int t4934 = t4933 + 512;
        float t4935 = t4911 - t4922;
        memory[83996148 + t4934] = t4935;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4938 = 0; _pr4938 < 256; _pr4938++) {
        float t4939 = (float)_pr4938;
        float t4940 = (t4939 * 0.0078125);
        float t4941 = metal::floor(t4940);
        float t4942 = t4941 * 128.0;
        float t4943 = t4939 - t4942;
        float t4944 = t4941 * 256.0;
        float t4945 = t4944 + t4943;
        float t4946 = t4945 + 128.0;
        float t4947 = 6.283185 * t4943;
        float t4948 = (t4947 * 0.00390625);
        float t4949 = metal::cos(t4948);
        float t4950 = metal::sin(t4948);
        int t4951 = (int)t4945;
        int t4952 = (int)t4946;
        int t4953 = t4041 + t4951;
        float t4954 = memory[83996148 + t4953];
        int t4955 = t4041 + t4951;
        int t4956 = t4955 + 512;
        float t4957 = memory[83996148 + t4956];
        int t4958 = t4041 + t4952;
        float t4959 = memory[83996148 + t4958];
        int t4960 = t4041 + t4952;
        int t4961 = t4960 + 512;
        float t4962 = memory[83996148 + t4961];
        float t4963 = t4949 * t4959;
        float t4964 = t4950 * t4962;
        float t4965 = t4963 - t4964;
        float t4966 = t4949 * t4962;
        float t4967 = t4950 * t4959;
        float t4968 = t4966 + t4967;
        int t4969 = t4041 + t4951;
        float t4970 = t4954 + t4965;
        memory[83996148 + t4969] = t4970;
        int t4972 = t4041 + t4951;
        int t4973 = t4972 + 512;
        float t4974 = t4957 + t4968;
        memory[83996148 + t4973] = t4974;
        int t4976 = t4041 + t4952;
        float t4977 = t4954 - t4965;
        memory[83996148 + t4976] = t4977;
        int t4979 = t4041 + t4952;
        int t4980 = t4979 + 512;
        float t4981 = t4957 - t4968;
        memory[83996148 + t4980] = t4981;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4984 = 0; _pr4984 < 256; _pr4984++) {
        float t4985 = (float)_pr4984;
        float t4986 = (t4985 * 0.00390625);
        float t4987 = metal::floor(t4986);
        float t4988 = t4987 * 256.0;
        float t4989 = t4985 - t4988;
        float t4990 = t4987 * 512.0;
        float t4991 = t4990 + t4989;
        float t4992 = t4991 + 256.0;
        float t4993 = 6.283185 * t4989;
        float t4994 = (t4993 * 0.001953125);
        float t4995 = metal::cos(t4994);
        float t4996 = metal::sin(t4994);
        int t4997 = (int)t4991;
        int t4998 = (int)t4992;
        int t4999 = t4041 + t4997;
        float t5000 = memory[83996148 + t4999];
        int t5001 = t4041 + t4997;
        int t5002 = t5001 + 512;
        float t5003 = memory[83996148 + t5002];
        int t5004 = t4041 + t4998;
        float t5005 = memory[83996148 + t5004];
        int t5006 = t4041 + t4998;
        int t5007 = t5006 + 512;
        float t5008 = memory[83996148 + t5007];
        float t5009 = t4995 * t5005;
        float t5010 = t4996 * t5008;
        float t5011 = t5009 - t5010;
        float t5012 = t4995 * t5008;
        float t5013 = t4996 * t5005;
        float t5014 = t5012 + t5013;
        int t5015 = t4041 + t4997;
        float t5016 = t5000 + t5011;
        memory[83996148 + t5015] = t5016;
        int t5018 = t4041 + t4997;
        int t5019 = t5018 + 512;
        float t5020 = t5003 + t5014;
        memory[83996148 + t5019] = t5020;
        int t5022 = t4041 + t4998;
        float t5023 = t5000 - t5011;
        memory[83996148 + t5022] = t5023;
        int t5025 = t4041 + t4998;
        int t5026 = t5025 + 512;
        float t5027 = t5003 - t5014;
        memory[83996148 + t5026] = t5027;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr5030 = 0; _pr5030 < 512; _pr5030++) {
        int t5031 = t4041 + _pr5030;
        float t5032 = memory[83996148 + t5031];
        float t5033 = t5032 * 7.599708e-06;
        float t5034 = memory[25460 + (int)_pr5030];
        int t5035 = t4042 + _pr5030;
        float t5036 = t5033 * t5034;
        memory[125955572 + t5035] = t5036;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t5040 = t[11*frameCount + id] > 0.0;
    if (t5040) {
      for (uint _pr5042 = 0; _pr5042 < 512; _pr5042++) {
        int t5043 = t4042 + _pr5042;
        memory[117550580 + t5043] = 0.0;
        int t5045 = t4042 + _pr5042;
        memory[125955572 + t5045] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 27
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_27(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5066), value: global(5066)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(528) - handled in variable access */
    int t5049 = id;
    float t5050 = 0.0;
    for (uint t5051 = 0; t5051 < 512; t5051++) {
      float t5052 = (float)t5051;
      float t5053 = (float)t5049;
      float t5054 = t5053 + t5052;
      int t5055 = 511 - t5051;
      float t5056 = frameCount - 1.0;
      float t5057 = metal::min(t5054, t5056);
      int t5058 = (int)t5057;
      int t5059 = t5058 * 512;
      int t5060 = t5059 + t5055;
      float t5061 = memory[117550580 + t5060];
      float t5062 = t5054 < frameCount;
      float t5063 = metal::select(0.0, t5061, t5062 > 0.0);
      float t5064 = t5050 + t5063;
      t5050 = t5064;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[19*frameCount + id] = (t5050 * 0.0027567567);
  }
  #pragma clang diagnostic pop
}



// KERNEL 28
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_28(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5084), value: global(5084)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(528) - handled in variable access */
    int t5067 = id;
    float t5068 = 0.0;
    for (uint t5069 = 0; t5069 < 512; t5069++) {
      float t5070 = (float)t5069;
      float t5071 = (float)t5067;
      float t5072 = t5071 + t5070;
      int t5073 = 511 - t5069;
      float t5074 = frameCount - 1.0;
      float t5075 = metal::min(t5072, t5074);
      int t5076 = (int)t5075;
      int t5077 = t5076 * 512;
      int t5078 = t5077 + t5073;
      float t5079 = memory[125955572 + t5078];
      float t5080 = t5072 < frameCount;
      float t5081 = metal::select(0.0, t5079, t5080 > 0.0);
      float t5082 = t5068 + t5081;
      t5068 = t5082;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[20*frameCount + id] = (t5068 * 0.0027567567);
  }
  #pragma clang diagnostic pop
}



// KERNEL 29
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_29(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5109), value: global(5109)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5108), value: global(5108)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5093), value: global(5093)) */
  float t5739 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5739)) {
    /* loadGlobal(5084) - handled in variable access */
    /* loadGlobal(5066) - handled in variable access */
    /* loadGlobal(3971) - handled in variable access */
    /* loadGlobal(3953) - handled in variable access */
    /* loadGlobal(473) - handled in variable access */
    /* loadGlobal(472) - handled in variable access */
    /* loadGlobal(454) - handled in variable access */
    /* loadGlobal(324) - handled in variable access */
    int t5085 = id;
    int t5086 = t5085 / 61;
    uint _frameIndex = (uint)(t5086);
    int t5087 = t5086 * 61;
    int t5088 = t5085 - t5087;
    float t5089 = t[17*frameCount + _frameIndex] + t[19*frameCount + _frameIndex];
    float t5090 = t[18*frameCount + _frameIndex] + t[20*frameCount + _frameIndex];
    float t5091 = 0.015625 * t5089;
    float t5092 = t[7*frameCount + _frameIndex] * t5089;
    t[21*frameCount + _frameIndex] = t[6*frameCount + _frameIndex] * t5091;
    float t5094 = t[5*frameCount + _frameIndex] * t5091;
    float t5095 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5096 = t5095 < 0.0;
    float t5097 = t5095 + 61.0;
    float t5098 = metal::select(t5095, t5097, t5096 > 0.0);
    float t5099 = t5098;
    float t5100 = metal::floor(t5099);
    float t5101 = t5099 - t5100;
    float t5102 = t5100 + 1.0;
    float t5103 = t5102 >= 61.0;
    float t5104 = metal::select(t5102, 0.0, t5103 > 0.0);
    float t5105 = 1.0 - t5101;
    float t5106 = t5094 * t5105;
    float t5107 = t5094 * t5101;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5100], t5106, metal::memory_order_relaxed);
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5104], t5107, metal::memory_order_relaxed);
    float t5110 = memory[209874420 + t5088];
    float t5111 = memory[60660 + t5088];
    float t5112 = t5110 / t5111;
    float t5113 = memory[60660 + t5088];
    float t5114 = memory[60660 + t5088];
    float t5115 = t5113 * t5114;
    float t5116 = 1.0 / t5115;
    float t5117 = memory[209874420 + t5088];
    float t5118 = t5117 * -1.0;
    float t5119 = t5118 * t5116;
    float t5120 = t5112 + t5119;
    float t5121 = memory[60724 + t5088];
    float t5122 = metal::exp(t5121);
    float t5123 = t5122 * t5119;
    float t5124 = -1.0 * t5123;
    int t5125 = _frameIndex;
    int t5126 = t5125 * 61;
    int t5127 = t5126 + t5088;
    memory[2158068 + t5127] = t5124;
    float t5129 = memory[60788 + t5088];
    float t5130 = t5129 * t5123;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=201, axis=0, in=[61, 1], out=[1], inFA=true, outFA=true), value: empty) */
    float t5131 = 0.0;
    int t5132 = t5088;
    int t5133 = t5132;
    int t5134 = t5088 - t5133;
    int t5135 = t5132;
    int t5136 = t5135;
    for (uint t5137 = 0; t5137 < 61; t5137++) {
      int t5138 = t5137;
      int t5139 = t5136 + t5138;
      int t5140 = _frameIndex;
      int t5141 = t5140 * 61;
      int t5142 = t5141 + t5139;
      float t5143 = memory[2158068 + t5142];
      float t5144 = t5131 + t5143;
      t5131 = t5144;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t5146 = _frameIndex;
    int t5147 = t5146;
    int t5148 = t5147 + t5088;
    memory[60916 + t5148] = t5131;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 30
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_30(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5150 = 0; t5150 < 3904; t5150++) {
      /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=203, axis=2, in=[61, 1], out=[61, 1, 64], inFA=true, outFA=true), value: empty) */
      int t5151 = t5150 / 64;
      int t5152 = t5151 % 61;
      int t5153 = t5152 * 1;
      int t5154 = 0 + t5153;
      int t5155 = t5150 / 64;
      int t5156 = t5155 % 1;
      int t5157 = t5156 * 1;
      int t5158 = t5154 + t5157;
      float t5159 = (float)t5158;
      int t5160 = id;
      int t5161 = t5160 * 61;
      float t5162 = t5161 + t5159;
      int t5163 = (int)t5162;
      float t5164 = memory[2158068 + t5163];
      float t5165 = (float)t5150;
      int t5166 = id;
      int t5167 = t5166 * 3904;
      float t5168 = t5167 + t5165;
      int t5169 = (int)t5168;
      memory[273837620 + t5169] = t5164;
      int t5171 = t5150 / 64;
      int t5172 = t5171 * 64;
      int t5173 = t5150 - t5172;
      int t5174 = t5173 / 64;
      int t5175 = t5174 * 64;
      int t5176 = t5173 - t5175;
      int t5177 = t5176 / 64;
      int t5178 = t5177 * 64;
      int t5179 = t5176 - t5178;
      float t5180 = memory[8576 + t5179];
      int t5181 = id;
      int t5182 = t5181 * 3904;
      int t5183 = t5182 + t5150;
      float t5184 = memory[273837620 + t5183];
      float t5185 = t5180 * t5184;
      int t5186 = id;
      int t5187 = t5186 * 3904;
      int t5188 = t5187 + t5150;
      memory[209874484 + t5188] = t5185;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5190 = 0; t5190 < 64; t5190++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=206, axis=0, in=[61, 1, 64], out=[1, 64], inFA=true, outFA=true), value: empty) */
      float t5191 = 0.0;
      int t5192 = t5190 / 64;
      int t5193 = t5192 * 64;
      int t5194 = t5190 - t5193;
      int t5195 = t5194;
      int t5196 = t5195;
      int t5197 = t5194 - t5196;
      int t5198 = t5192 * 64;
      int t5199 = t5198;
      int t5200 = t5195;
      int t5201 = t5199 + t5200;
      for (uint t5202 = 0; t5202 < 61; t5202++) {
        int t5203 = t5202 * 64;
        int t5204 = t5201 + t5203;
        int t5205 = t5202 * 64;
        int t5206 = t5205 + t5195;
        float t5207 = memory[37172 + t5206];
        float t5208 = t5202 + 0.0;
        int t5209 = id;
        int t5210 = t5209 * 61;
        float t5211 = t5210 + t5208;
        int t5212 = (int)t5211;
        float t5213 = memory[2158068 + t5212];
        float t5214 = t5207 * t5213;
        float t5215 = t5191 + t5214;
        t5191 = t5215;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5217 = id;
      int t5218 = t5217 * 64;
      int t5219 = t5218 + t5190;
      memory[37809652 + t5219] = t5191;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 31
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_31(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5221), value: global(5221)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5093) - handled in variable access */
    /* loadGlobal(446) - handled in variable access */
    /* loadGlobal(364) - handled in variable access */
    t[24*frameCount + id] = t[3*frameCount + id] * t[21*frameCount + id];
    float t5222 = t[4*frameCount + id] * t[21*frameCount + id];
  }
  #pragma clang diagnostic pop
}



// KERNEL 32
// Kind: simd
// ThreadCountScale Optional(64)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_32(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5740 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5740)) {
    /* loadGlobal(5221) - handled in variable access */
    int t5223 = id;
    int t5224 = t5223 / 64;
    uint _frameIndex = (uint)(t5224);
    int t5225 = t5224 * 64;
    int t5226 = t5223 - t5225;
    int t5227 = t5224 * 64;
    int t5228 = t5227 + t5226;
    memory[42020340 + t5228] = t[24*frameCount + _frameIndex];
    int t5230 = _frameIndex;
    int t5231 = t5230 * 64;
    int t5232 = t5231 + t5226;
    float t5233 = memory[1109492 + t5232];
    int t5234 = _frameIndex;
    int t5235 = t5234 * 64;
    int t5236 = t5235 + t5226;
    float t5237 = memory[42020340 + t5236];
    float t5238 = t5233 * t5237;
    int t5239 = _frameIndex;
    int t5240 = t5239 * 64;
    int t5241 = t5240 + t5226;
    float t5242 = memory[3206644 + t5241];
    int t5243 = _frameIndex;
    int t5244 = t5243 * 64;
    int t5245 = t5244 + t5226;
    float t5246 = memory[42020340 + t5245];
    float t5247 = t5242 * t5246;
    int t5248 = _frameIndex;
    int t5249 = t5248 * 64;
    int t5250 = t5249 + t5226;
    memory[2158068 + t5250] = t5247;
  }
  #pragma clang diagnostic pop
}



// KERNEL 33
// Kind: simd
// ThreadCountScale Optional(3904)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_33(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5741 = frameCount * 3904.0;
  if (id >= 0 && id < (uint)(t5741)) {
    /* loadGlobal(324) - handled in variable access */
    int t5252 = id;
    int t5253 = t5252 / 3904;
    uint _frameIndex = (uint)(t5253);
    int t5254 = t5253 * 3904;
    int t5255 = t5252 - t5254;
    float t5256 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5257 = t5256 < 0.0;
    float t5258 = t5256 + 61.0;
    float t5259 = metal::select(t5256, t5258, t5257 > 0.0);
    float t5260 = metal::floor(t5259);
    float t5261 = t5260 + 1.0;
    float t5262 = t5261 >= 61.0;
    float t5263 = metal::select(t5261, 0.0, t5262 > 0.0);
    float t5264 = t5259 - t5260;
    int t5265 = _frameIndex;
    memory[42020340 + t5265] = t5260;
    memory[46231028 + t5265] = t5264;
    float t5268 = t5265 + 16384.0;
    int t5269 = (int)t5268;
    memory[42020340 + t5269] = t5263;
    float t5271 = 1.0 - t5264;
    float t5272 = t5265 * 64.0;
    for (uint _pr5273 = 0; _pr5273 < 64; _pr5273++) {
      float t5274 = (float)_pr5273;
      float t5275 = t5272 + t5274;
      int t5276 = (int)t5275;
      float t5277 = memory[2158068 + t5276];
      float t5278 = t5272 + t5274;
      float t5279 = t5277 * t5271;
      int t5280 = (int)t5278;
      memory[1109492 + t5280] = t5279;
      float t5282 = t5277 * t5264;
      int t5283 = (int)t5278;
      memory[3206644 + t5283] = t5282;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 34
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(3904)
kernel void kernel_34(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 3904) { uint _pr5286 = id;
    int t5287 = _pr5286 / 64;
    int t5288 = t5287 * 64;
    int t5289 = _pr5286 - t5288;
    float t5290 = (float)t5287;
    float t5291 = (float)t5289;
    float t5292 = 0.0;
    for (uint t5293 = 0; t5293 < 16384; t5293++) {
      float t5294 = (float)t5293;
      float t5295 = t5294 < frameCount;
      float t5296 = t5294 * 64.0;
      float t5297 = t5296 + t5291;
      float t5298 = memory[42020340 + (int)t5293];
      float t5299 = t5298 - t5290;
      float t5300 = metal::abs(t5299);
      float t5301 = t5300 < 0.5;
      int t5302 = (int)t5297;
      float t5303 = memory[1109492 + t5302];
      float t5304 = t5295 * t5301;
      float t5305 = t5304 > 0.0;
      float t5306 = metal::select(0.0, t5303, t5305 > 0.0);
      float t5307 = t5292 + t5306;
      t5292 = t5307;
      float t5308 = t5294 + 16384.0;
      int t5309 = (int)t5308;
      float t5310 = memory[42020340 + t5309];
      float t5311 = t5310 - t5290;
      float t5312 = metal::abs(t5311);
      float t5313 = t5312 < 0.5;
      int t5314 = (int)t5297;
      float t5315 = memory[3206644 + t5314];
      float t5316 = t5295 * t5313;
      float t5317 = t5316 > 0.0;
      float t5318 = metal::select(0.0, t5315, t5317 > 0.0);
      float t5319 = t5292 + t5318;
      t5292 = t5319;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t5321 = t5290 * 64.0;
    float t5322 = t5321 + t5291;
    int t5323 = (int)t5322;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[337800756 + t5323], t5292, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 35
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_35(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5326 = 0; t5326 < 3904; t5326++) {
      float t5327 = memory[337800756 + (int)t5326];
      float t5328 = memory[56692 + (int)t5326];
      float t5329 = t5327 / t5328;
      float t5330 = memory[56692 + (int)t5326];
      float t5331 = memory[56692 + (int)t5326];
      float t5332 = t5330 * t5331;
      float t5333 = 1.0 / t5332;
      float t5334 = memory[337800756 + (int)t5326];
      float t5335 = t5334 * -1.0;
      float t5336 = t5335 * t5333;
      float t5337 = t5329 + t5336;
      float t5338 = memory[44980 + (int)t5326];
      float t5339 = metal::exp(t5338);
      float t5340 = t5339 * t5336;
      float t5341 = -1.0 * t5340;
      int t5342 = id;
      int t5343 = t5342 * 3904;
      int t5344 = t5343 + t5326;
      memory[273837620 + t5344] = t5341;
      float t5346 = memory[48884 + (int)t5326];
      float t5347 = t5346 * t5340;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5348 = 0; t5348 < 64; t5348++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=232, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5349 = 0.0;
      int t5350 = t5348;
      int t5351 = t5350;
      int t5352 = t5348 - t5351;
      int t5353 = t5350;
      int t5354 = t5353;
      for (uint t5355 = 0; t5355 < 61; t5355++) {
        int t5356 = t5355 * 64;
        int t5357 = t5354 + t5356;
        int t5358 = id;
        int t5359 = t5358 * 3904;
        int t5360 = t5359 + t5357;
        float t5361 = memory[273837620 + t5360];
        float t5362 = t5349 + t5361;
        t5349 = t5362;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5364 = id;
      int t5365 = t5364 * 64;
      int t5366 = t5365 + t5348;
      memory[1109492 + t5366] = t5349;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 36
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_36(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5368 = 0; t5368 < 3904; t5368++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=237, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5369 = 0.0;
      int t5370 = t5368 / 64;
      int t5371 = t5370 * 64;
      int t5372 = t5368 - t5371;
      int t5373 = t5372;
      int t5374 = t5373;
      int t5375 = t5372 - t5374;
      int t5376 = t5370 * 4096;
      int t5377 = t5376;
      int t5378 = t5373;
      int t5379 = t5377 + t5378;
      for (uint t5380 = 0; t5380 < 64; t5380++) {
        int t5381 = t5380 * 64;
        int t5382 = t5379 + t5381;
        int t5383 = t5380 * 64;
        int t5384 = t5383 + t5373;
        int t5385 = t5384 / 64;
        int t5386 = t5385 * 64;
        int t5387 = t5384 - t5386;
        int t5388 = t5387 * 64;
        int t5389 = t5385 + t5388;
        float t5390 = memory[4416 + t5389];
        int t5391 = t5370 * 64;
        int t5392 = t5391 + t5380;
        int t5393 = id;
        int t5394 = t5393 * 3904;
        int t5395 = t5394 + t5392;
        float t5396 = memory[273837620 + t5395];
        float t5397 = t5390 * t5396;
        float t5398 = t5369 + t5397;
        t5369 = t5398;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5400 = id;
      int t5401 = t5400 * 3904;
      int t5402 = t5401 + t5368;
      memory[337804660 + t5402] = t5369;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5404 = 0; t5404 < 4096; t5404++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=239, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5405 = 0.0;
      int t5406 = t5404 / 64;
      int t5407 = t5406 * 64;
      int t5408 = t5404 - t5407;
      int t5409 = t5408;
      int t5410 = t5409;
      int t5411 = t5408 - t5410;
      int t5412 = t5406 * 64;
      int t5413 = t5412;
      int t5414 = t5409;
      int t5415 = t5413 + t5414;
      for (uint t5416 = 0; t5416 < 61; t5416++) {
        int t5417 = t5416 * 4096;
        int t5418 = t5415 + t5417;
        int t5419 = t5416 * 64;
        int t5420 = t5419 + t5409;
        float t5421 = memory[37172 + t5420];
        int t5422 = t5416 * 64;
        int t5423 = t5422 + t5406;
        int t5424 = id;
        int t5425 = t5424 * 3904;
        int t5426 = t5425 + t5423;
        float t5427 = memory[273837620 + t5426];
        float t5428 = t5421 * t5427;
        float t5429 = t5405 + t5428;
        t5405 = t5429;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5431 = id;
      int t5432 = t5431 * 4096;
      int t5433 = t5432 + t5404;
      memory[401767796 + t5433] = t5405;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 37
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_37(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5435 = 0; t5435 < 3904; t5435++) {
      int t5436 = id;
      int t5437 = t5436 * 3904;
      int t5438 = t5437 + t5435;
      float t5439 = memory[209874484 + t5438];
      int t5440 = id;
      int t5441 = t5440 * 3904;
      int t5442 = t5441 + t5435;
      float t5443 = memory[337804660 + t5442];
      float t5444 = t5439 + t5443;
      float t5445 = memory[41076 + (int)t5435];
      float t5446 = metal::tanh(t5445);
      float t5447 = t5446 * t5446;
      float t5448 = 1.0 - t5447;
      float t5449 = t5448 * t5444;
      int t5450 = id;
      int t5451 = t5450 * 3904;
      int t5452 = t5451 + t5435;
      memory[468876660 + t5452] = t5449;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5454 = 0; t5454 < 64; t5454++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=250, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5455 = 0.0;
      int t5456 = t5454;
      int t5457 = t5456;
      int t5458 = t5454 - t5457;
      int t5459 = t5456;
      int t5460 = t5459;
      for (uint t5461 = 0; t5461 < 61; t5461++) {
        int t5462 = t5461 * 64;
        int t5463 = t5460 + t5462;
        int t5464 = t5461 * 64;
        int t5465 = t5464 + t5456;
        float t5466 = memory[25460 + t5465];
        int t5467 = t5461 * 64;
        int t5468 = t5467 + t5456;
        int t5469 = id;
        int t5470 = t5469 * 3904;
        int t5471 = t5470 + t5468;
        float t5472 = memory[273837620 + t5471];
        float t5473 = t5466 * t5472;
        float t5474 = t5455 + t5473;
        t5455 = t5474;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5476 = id;
      int t5477 = t5476 * 64;
      int t5478 = t5477 + t5454;
      memory[2158068 + t5478] = t5455;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 38
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_38(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5480 = 0; t5480 < 3904; t5480++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=255, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5481 = 0.0;
      int t5482 = t5480 / 64;
      int t5483 = t5482 * 64;
      int t5484 = t5480 - t5483;
      int t5485 = t5484;
      int t5486 = t5485;
      int t5487 = t5484 - t5486;
      int t5488 = t5482 * 4096;
      int t5489 = t5488;
      int t5490 = t5485;
      int t5491 = t5489 + t5490;
      for (uint t5492 = 0; t5492 < 64; t5492++) {
        int t5493 = t5492 * 64;
        int t5494 = t5491 + t5493;
        int t5495 = t5492 * 64;
        int t5496 = t5495 + t5485;
        int t5497 = t5496 / 64;
        int t5498 = t5497 * 64;
        int t5499 = t5496 - t5498;
        int t5500 = t5499 * 64;
        int t5501 = t5497 + t5500;
        float t5502 = memory[256 + t5501];
        int t5503 = t5482 * 64;
        int t5504 = t5503 + t5492;
        int t5505 = id;
        int t5506 = t5505 * 3904;
        int t5507 = t5506 + t5504;
        float t5508 = memory[468876660 + t5507];
        float t5509 = t5502 * t5508;
        float t5510 = t5481 + t5509;
        t5481 = t5510;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5512 = id;
      int t5513 = t5512 * 3904;
      int t5514 = t5513 + t5480;
      memory[209874484 + t5514] = t5481;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5516 = 0; t5516 < 4096; t5516++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=257, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5517 = 0.0;
      int t5518 = t5516 / 64;
      int t5519 = t5518 * 64;
      int t5520 = t5516 - t5519;
      int t5521 = t5520;
      int t5522 = t5521;
      int t5523 = t5520 - t5522;
      int t5524 = t5518 * 64;
      int t5525 = t5524;
      int t5526 = t5521;
      int t5527 = t5525 + t5526;
      for (uint t5528 = 0; t5528 < 61; t5528++) {
        int t5529 = t5528 * 4096;
        int t5530 = t5527 + t5529;
        int t5531 = t5528 * 64;
        int t5532 = t5531 + t5521;
        float t5533 = memory[33268 + t5532];
        int t5534 = t5528 * 64;
        int t5535 = t5534 + t5518;
        int t5536 = id;
        int t5537 = t5536 * 3904;
        int t5538 = t5537 + t5535;
        float t5539 = memory[468876660 + t5538];
        float t5540 = t5533 * t5539;
        float t5541 = t5517 + t5540;
        t5517 = t5541;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5543 = id;
      int t5544 = t5543 * 4096;
      int t5545 = t5544 + t5516;
      memory[532839796 + t5545] = t5517;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 39
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_39(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5547 = 0; t5547 < 3904; t5547++) {
      float t5548 = memory[29364 + (int)t5547];
      float t5549 = metal::tanh(t5548);
      float t5550 = t5549 * t5549;
      float t5551 = 1.0 - t5550;
      int t5552 = id;
      int t5553 = t5552 * 3904;
      int t5554 = t5553 + t5547;
      float t5555 = memory[209874484 + t5554];
      float t5556 = t5551 * t5555;
      int t5557 = id;
      int t5558 = t5557 * 3904;
      int t5559 = t5558 + t5547;
      memory[273837620 + t5559] = t5556;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5561 = 0; t5561 < 64; t5561++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=267, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5562 = 0.0;
      int t5563 = t5561;
      int t5564 = t5563;
      int t5565 = t5561 - t5564;
      int t5566 = t5563;
      int t5567 = t5566;
      for (uint t5568 = 0; t5568 < 61; t5568++) {
        int t5569 = t5568 * 64;
        int t5570 = t5567 + t5569;
        int t5571 = t5568 * 64;
        int t5572 = t5571 + t5563;
        float t5573 = memory[25460 + t5572];
        int t5574 = t5568 * 64;
        int t5575 = t5574 + t5563;
        int t5576 = id;
        int t5577 = t5576 * 3904;
        int t5578 = t5577 + t5575;
        float t5579 = memory[209874484 + t5578];
        float t5580 = t5573 * t5579;
        float t5581 = t5562 + t5580;
        t5562 = t5581;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5583 = id;
      int t5584 = t5583 * 64;
      int t5585 = t5584 + t5561;
      memory[3206644 + t5585] = t5562;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 40
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_40(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5587 = 0; t5587 < 183; t5587++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=272, axis=1, in=[61, 64, 3], out=[61, 3], inFA=true, outFA=true), value: empty) */
      float t5588 = 0.0;
      int t5589 = t5587 / 3;
      int t5590 = t5589 * 3;
      int t5591 = t5587 - t5590;
      int t5592 = t5591;
      int t5593 = t5592;
      int t5594 = t5591 - t5593;
      int t5595 = t5589 * 192;
      int t5596 = t5595;
      int t5597 = t5592;
      int t5598 = t5596 + t5597;
      for (uint t5599 = 0; t5599 < 64; t5599++) {
        int t5600 = t5599 * 3;
        int t5601 = t5598 + t5600;
        int t5602 = t5599 * 3;
        int t5603 = t5602 + t5592;
        int t5604 = t5603 / 3;
        int t5605 = t5604 * 3;
        int t5606 = t5603 - t5605;
        int t5607 = t5606 * 64;
        int t5608 = t5604 + t5607;
        float t5609 = memory[0 + t5608];
        int t5610 = t5589 * 64;
        int t5611 = t5610 + t5599;
        int t5612 = id;
        int t5613 = t5612 * 3904;
        int t5614 = t5613 + t5611;
        float t5615 = memory[273837620 + t5614];
        float t5616 = t5609 * t5615;
        float t5617 = t5588 + t5616;
        t5588 = t5617;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5619 = id;
      int t5620 = t5619 * 183;
      int t5621 = t5620 + t5587;
      memory[42020340 + t5621] = t5588;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 3]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5623 = 0; t5623 < 192; t5623++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=274, axis=0, in=[61, 64, 3], out=[64, 3], inFA=true, outFA=true), value: empty) */
      float t5624 = 0.0;
      int t5625 = t5623 / 3;
      int t5626 = t5625 * 3;
      int t5627 = t5623 - t5626;
      int t5628 = t5627;
      int t5629 = t5628;
      int t5630 = t5627 - t5629;
      int t5631 = t5625 * 3;
      int t5632 = t5631;
      int t5633 = t5628;
      int t5634 = t5632 + t5633;
      for (uint t5635 = 0; t5635 < 61; t5635++) {
        int t5636 = t5635 * 192;
        int t5637 = t5634 + t5636;
        int t5638 = t5635 * 3;
        int t5639 = t5638 + t5628;
        float t5640 = memory[8706 + t5639];
        int t5641 = t5635 * 64;
        int t5642 = t5641 + t5625;
        int t5643 = id;
        int t5644 = t5643 * 3904;
        int t5645 = t5644 + t5642;
        float t5646 = memory[273837620 + t5645];
        float t5647 = t5640 * t5646;
        float t5648 = t5624 + t5647;
        t5624 = t5648;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5650 = id;
      int t5651 = t5650 * 192;
      int t5652 = t5651 + t5623;
      memory[46231028 + t5652] = t5624;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 3]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([64, 3]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 41
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(192)
kernel void kernel_41(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 192) { uint _pr5654 = id;
    float t5655 = 0.0;
    for (uint t5656 = 0; t5656 < 16384; t5656++) {
      int t5657 = t5656 * 192;
      int t5658 = t5657 + _pr5654;
      float t5659 = memory[46231028 + t5658];
      float t5660 = t5655 + t5659;
      t5655 = t5660;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5654] = t5655;
  }
  #pragma clang diagnostic pop
}



// KERNEL 42
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_42(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5664 = id;
    float t5665 = 0.0;
    for (uint t5666 = 0; t5666 < 16384; t5666++) {
      int t5667 = t5666 * 64;
      int t5668 = t5667 + _pr5664;
      float t5669 = memory[3206644 + t5668];
      float t5670 = t5665 + t5669;
      t5665 = t5670;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5664] = t5665;
  }
  #pragma clang diagnostic pop
}



// KERNEL 43
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(4096)
kernel void kernel_43(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 4096) { uint _pr5674 = id;
    float t5675 = 0.0;
    for (uint t5676 = 0; t5676 < 16384; t5676++) {
      int t5677 = t5676 * 4096;
      int t5678 = t5677 + _pr5674;
      float t5679 = memory[532839796 + t5678];
      float t5680 = t5675 + t5679;
      t5675 = t5680;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[3206644 + (int)_pr5674] = t5675;
  }
  #pragma clang diagnostic pop
}



// KERNEL 44
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_44(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5684 = id;
    float t5685 = 0.0;
    for (uint t5686 = 0; t5686 < 16384; t5686++) {
      int t5687 = t5686 * 64;
      int t5688 = t5687 + _pr5684;
      float t5689 = memory[2158068 + t5688];
      float t5690 = t5685 + t5689;
      t5685 = t5690;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5684] = t5685;
  }
  #pragma clang diagnostic pop
}



// KERNEL 45
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(4096)
kernel void kernel_45(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 4096) { uint _pr5694 = id;
    float t5695 = 0.0;
    for (uint t5696 = 0; t5696 < 16384; t5696++) {
      int t5697 = t5696 * 4096;
      int t5698 = t5697 + _pr5694;
      float t5699 = memory[401767796 + t5698];
      float t5700 = t5695 + t5699;
      t5695 = t5700;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[2158068 + (int)_pr5694] = t5695;
  }
  #pragma clang diagnostic pop
}



// KERNEL 46
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_46(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5704 = id;
    float t5705 = 0.0;
    for (uint t5706 = 0; t5706 < 16384; t5706++) {
      int t5707 = t5706 * 64;
      int t5708 = t5707 + _pr5704;
      float t5709 = memory[1109492 + t5708];
      float t5710 = t5705 + t5709;
      t5705 = t5710;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5704] = t5705;
  }
  #pragma clang diagnostic pop
}



// KERNEL 47
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_47(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5714 = id;
    float t5715 = 0.0;
    for (uint t5716 = 0; t5716 < 16384; t5716++) {
      int t5717 = t5716 * 64;
      int t5718 = t5717 + _pr5714;
      float t5719 = memory[37809652 + t5718];
      float t5720 = t5715 + t5719;
      t5715 = t5720;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5714] = t5715;
  }
  #pragma clang diagnostic pop
}



// KERNEL 48
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(1)
kernel void kernel_48(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 1) { uint _pr5724 = id;
    float t5725 = 0.0;
    for (uint t5726 = 0; t5726 < 16384; t5726++) {
      int t5727 = t5726;
      int t5728 = t5727 + _pr5724;
      float t5729 = memory[60916 + t5728];
      float t5730 = t5725 + t5729;
      t5725 = t5730;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[60596 + (int)_pr5724] = t5725;
  }
  #pragma clang diagnostic pop
}



// KERNEL 49
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_49(
    device float *outputs [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5109) - handled in variable access */
    /* loadGlobal(5108) - handled in variable access */
    /* loadGlobal(2758) - handled in variable access */
    outputs[0 * frameCount + id] = t[16*frameCount + id];
  }
  #pragma clang diagnostic pop
}

