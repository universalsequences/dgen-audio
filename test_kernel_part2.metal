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
    memory[33268 + t92] = t128;
    float t130 = metal::tanh(t128);
    memory[29364 + t92] = t130;
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
      float t152 = memory[29364 + t151];
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
    int t179 = t178 == 0.0;
    if (t179) {
    }
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=33, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
    float t182 = 0.0;
    int t183 = t178 / 64;
    int t184 = t183 * 64;
    int t185 = t178 - t184;
    int t186 = t185;
    int t187 = t186;
    int t188 = t185 - t187;
    int t189 = t183 * 4096;
    int t190 = t189;
    int t191 = t186 * 64;
    int t192 = t190 + t191;
    for (uint t193 = 0; t193 < 64; t193++) {
      int t194 = t193;
      int t195 = t192 + t194;
      int t196 = t183 * 64;
      int t197 = t196 + t193;
      float t198 = memory[37172 + t197];
      int t199 = t186 * 64;
      int t200 = t199 + t193;
      int t201 = t200 / 64;
      int t202 = t201 * 64;
      int t203 = t200 - t202;
      int t204 = t203 * 64;
      int t205 = t201 + t204;
      float t206 = memory[4416 + t205];
      float t207 = t198 * t206;
      float t208 = t182 + t207;
      t182 = t208;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t178] = t182;
    float t211 = memory[25460 + t178];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t212 = t178 / 64;
    int t213 = t212 * 64;
    int t214 = t178 - t213;
    int t215 = t214;
    float t216 = memory[8512 + t215];
    float t217 = t211 + t216;
    memory[44980 + t178] = t217;
    float t219 = t217 * -1.0;
    memory[56692 + t178] = t219;
    float t221 = metal::exp(t219);
    float t222 = 1.0 + t221;
    memory[48884 + t178] = t222;
    float t224 = 1.0 / t222;
    memory[52788 + t178] = t224;
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
    int t226 = id;
    int t227 = t226 / 61;
    uint _frameIndex = (uint)(t227);
    int t228 = t227 * 61;
    int t229 = t226 - t228;
    int t230 = t229 == 0.0;
    if (t230) {
    }
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=45, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
    float t233 = 0.0;
    int t234 = t229;
    int t235 = t234;
    int t236 = t229 - t235;
    int t237 = t236;
    int t238 = t237;
    int t239 = t236 - t238;
    int t240 = t234 * 64;
    int t241 = t240;
    int t242 = t237 * 64;
    int t243 = t241 + t242;
    for (uint t244 = 0; t244 < 64; t244++) {
      int t245 = t244;
      int t246 = t243 + t245;
      int t247 = t234 * 64;
      int t248 = t247 + t244;
      float t249 = memory[37172 + t248];
      int t250 = t244 / 64;
      int t251 = t250 * 64;
      int t252 = t244 - t251;
      float t253 = memory[8576 + t252];
      float t254 = t249 * t253;
      float t255 = t233 + t254;
      t233 = t255;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t229] = t233;
    float t258 = memory[25460 + t229];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t259 = t229;
    int t260 = t259;
    int t261 = t229 - t260;
    float t262 = memory[8640 + (int)0.0];
    float t263 = t258 + t262;
    memory[60788 + t229] = t263;
    float t265 = t263 * -1.0;
    memory[60724 + t229] = t265;
    float t267 = metal::exp(t265);
    float t268 = 1.0 + t267;
    memory[60596 + t229] = t268;
    float t270 = 1.0 / t268;
    memory[60660 + t229] = t270;
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
    int t272 = id;
    int t273 = t272 / 61;
    uint _frameIndex = (uint)(t273);
    int t274 = t273 * 61;
    int t275 = t272 - t274;
    int t276 = t275 == 0.0;
    if (t276) {
    }
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=57, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
    float t279 = 0.0;
    int t280 = t275;
    int t281 = t280;
    int t282 = t275 - t281;
    int t283 = t282;
    int t284 = t283;
    int t285 = t282 - t284;
    int t286 = t280 * 64;
    int t287 = t286;
    int t288 = t283 * 64;
    int t289 = t287 + t288;
    for (uint t290 = 0; t290 < 64; t290++) {
      int t291 = t290;
      int t292 = t289 + t291;
      int t293 = t280 * 64;
      int t294 = t293 + t290;
      float t295 = memory[37172 + t294];
      int t296 = t290 / 64;
      int t297 = t296 * 64;
      int t298 = t290 - t297;
      float t299 = memory[8641 + t298];
      float t300 = t295 * t299;
      float t301 = t279 + t300;
      t279 = t301;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + t275] = t279;
    float t304 = memory[25460 + t275];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t305 = t275;
    int t306 = t305;
    int t307 = t275 - t306;
    float t308 = memory[8705 + (int)0.0];
    float t309 = t304 + t308;
    float t310 = t309 * -1.0;
    float t311 = metal::exp(t310);
    float t312 = 1.0 + t311;
    float t313 = 1.0 / t312;
    memory[60852 + t275] = t313;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(315), value: global(315)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[599948660];
      float t316 = t[0*frameCount + i] + 0.003662333;
      float t317 = metal::select(t316, 0.0, 0.0 > 0.0);
      float t318 = t317;
      float t319 = (t318 * 0.016666668);
      float t320 = metal::floor(t319);
      float t321 = t320 * 60.0;
      float t322 = t317 - t321;
      memory[599948660] = t322;
      float t324 = t322 >= 60.0;
      if (t324) {
        float t326 = t322 - 60.0;
        memory[599948660] = t326;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(373), value: global(373)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(353), value: global(353)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(333), value: global(333)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(315) - handled in variable access */
    float t332 = metal::min(t[0*frameCount + id], 59.9999);
    t[1*frameCount + id] = metal::max(t332, 0.0);
    float t334 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t335 = t334 < 0.0;
    float t336 = t334 + 61.0;
    float t337 = metal::select(t334, t336, t335 > 0.0);
    float t338 = t337;
    float t339 = metal::floor(t338);
    float t340 = t338 - t339;
    float t341 = t339 + 1.0;
    float t342 = t341 >= 61.0;
    float t343 = metal::select(t341, 0.0, t342 > 0.0);
    int t344 = (int)t339;
    float t345 = memory[25273 + t344];
    int t346 = (int)t343;
    float t347 = memory[25273 + t346];
    float t348 = 1.0 - t340;
    float t349 = t345 * t348;
    float t350 = t347 * t340;
    float t351 = t349 + t350;
    float t352 = metal::max(t351, 20.0);
    t[2*frameCount + id] = metal::min(t352, 500.0);
    float t354 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t355 = t354 < 0.0;
    float t356 = t354 + 61.0;
    float t357 = metal::select(t354, t356, t355 > 0.0);
    float t358 = t357;
    float t359 = metal::floor(t358);
    float t360 = t358 - t359;
    float t361 = t359 + 1.0;
    float t362 = t361 >= 61.0;
    float t363 = metal::select(t361, 0.0, t362 > 0.0);
    int t364 = (int)t359;
    float t365 = memory[25334 + t364];
    int t366 = (int)t363;
    float t367 = memory[25334 + t366];
    float t368 = 1.0 - t360;
    float t369 = t365 * t368;
    float t370 = t367 * t360;
    float t371 = t369 + t370;
    float t372 = metal::min(t371, 1.0);
    t[3*frameCount + id] = metal::max(t372, 0.0);
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
  float t5747 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5747)) {
    /* loadGlobal(353) - handled in variable access */
    /* loadGlobal(333) - handled in variable access */
    int t374 = id;
    int t375 = t374 / 64;
    uint _frameIndex = (uint)(t375);
    int t376 = t375 * 64;
    int t377 = t374 - t376;
    float t378 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t379 = t378 < 0.0;
    float t380 = t378 + 61.0;
    float t381 = metal::select(t378, t380, t379 > 0.0);
    float t382 = metal::floor(t381);
    float t383 = t382 + 1.0;
    float t384 = t383 >= 61.0;
    float t385 = metal::select(t383, 0.0, t384 > 0.0);
    float t386 = t381 - t382;
    float t387 = 1.0 - t386;
    float t388 = t375 * 64.0;
    float t389 = (float)t377;
    float t390 = t382 * 64.0;
    float t391 = t390 + t389;
    int t392 = (int)t391;
    float t393 = memory[52788 + t392];
    float t394 = t385 * 64.0;
    float t395 = t394 + t389;
    int t396 = (int)t395;
    float t397 = memory[52788 + t396];
    float t398 = t387 * t393;
    float t399 = t386 * t397;
    float t400 = t398 + t399;
    float t401 = t388 + t389;
    int t402 = (int)t401;
    memory[60916 + t402] = t400;
    int t404 = (int)t401;
    memory[1109492 + t404] = t400;
    float t406 = memory[25395 + t377];
    float t407 = t406 * t[2*frameCount + _frameIndex];
    int t408 = _frameIndex;
    int t409 = t408 * 64;
    int t410 = t409 + t377;
    memory[2158068 + t410] = t407;
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
      int t412 = id;
      int t413 = i;
      int t414 = t413 * 64;
      int t415 = t414 + t412;
      float t416 = memory[2158068 + t415];
      float t417 = (t416 * 6.25e-05);
      float t418 = memory[25460 + t412];
      float t419 = t418 + t417;
      float t420 = metal::select(t419, 0.0, 0.0 > 0.0);
      float t421 = metal::floor(t420);
      float t422 = t420 - t421;
      float t423 = t422 >= 1.0;
      float t424 = t422 - 1.0;
      float t425 = metal::select(t422, t424, t423 > 0.0);
      float t426 = metal::select(t425, 0.0, 0.0 > 0.0);
      memory[25460 + t412] = t426;
      int t428 = i;
      int t429 = t428 * 64;
      int t430 = t429 + t412;
      memory[60916 + t430] = t418;
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
  float t5748 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5748)) {
    int t432 = id;
    int t433 = t432 / 64;
    uint _frameIndex = (uint)(t433);
    int t434 = t433 * 64;
    int t435 = t432 - t434;
    int t436 = _frameIndex;
    int t437 = t436 * 64;
    int t438 = t437 + t435;
    float t439 = memory[60916 + t438];
    float t440 = t439 * 6.283185;
    float t441 = metal::sin(t440);
    int t442 = _frameIndex;
    int t443 = t442 * 64;
    int t444 = t443 + t435;
    memory[3206644 + t444] = t441;
    int t446 = _frameIndex;
    int t447 = t446 * 64;
    int t448 = t447 + t435;
    float t449 = memory[1109492 + t448];
    float t450 = t441 * t449;
    int t451 = _frameIndex;
    int t452 = t451 * 64;
    int t453 = t452 + t435;
    memory[2158068 + t453] = t450;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(455), value: global(455)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    t[4*frameCount + id] = 0.0;
    for (uint t456 = 0; t456 < 64; t456++) {
      int t457 = id;
      int t458 = t457 * 64;
      int t459 = t458 + t456;
      float t460 = memory[2158068 + t459];
      float t461 = t[4*frameCount + id] + t460;
      t[4*frameCount + id] = t461;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(483), value: global(483)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(482), value: global(482)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(481), value: global(481)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(463), value: global(463)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(455) - handled in variable access */
    /* loadGlobal(373) - handled in variable access */
    /* loadGlobal(333) - handled in variable access */
    t[5*frameCount + id] = t[4*frameCount + id] * t[3*frameCount + id];
    float t464 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t465 = t464 < 0.0;
    float t466 = t464 + 61.0;
    float t467 = metal::select(t464, t466, t465 > 0.0);
    float t468 = t467;
    float t469 = metal::floor(t468);
    float t470 = t468 - t469;
    float t471 = t469 + 1.0;
    float t472 = t471 >= 61.0;
    float t473 = metal::select(t471, 0.0, t472 > 0.0);
    int t474 = (int)t469;
    float t475 = memory[60660 + t474];
    int t476 = (int)t473;
    float t477 = memory[60660 + t476];
    float t478 = 1.0 - t470;
    float t479 = t475 * t478;
    float t480 = t477 * t470;
    t[6*frameCount + id] = t479 + t480;
    t[7*frameCount + id] = t[5*frameCount + id] * t[6*frameCount + id];
    t[8*frameCount + id] = t[7*frameCount + id] * 0.015625;
    float t484 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t485 = t484 < 0.0;
    float t486 = t484 + 61.0;
    float t487 = metal::select(t484, t486, t485 > 0.0);
    float t488 = t487;
    float t489 = metal::floor(t488);
    float t490 = t488 - t489;
    float t491 = t489 + 1.0;
    float t492 = t491 >= 61.0;
    float t493 = metal::select(t491, 0.0, t492 > 0.0);
    int t494 = (int)t489;
    float t495 = memory[60852 + t494];
    int t496 = (int)t493;
    float t497 = memory[60852 + t496];
    float t498 = 1.0 - t490;
    float t499 = t495 * t498;
    float t500 = t497 * t490;
    float t501 = t499 + t500;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(502), value: global(502)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[9*frameCount + i] = memory[599948661];
      float t503 = t[9*frameCount + i] + 1.0;
      float t504 = metal::select(t503, 0.0, 0.0 > 0.0);
      float t505 = t504;
      float t506 = (t505 * 6.1035156e-05);
      float t507 = metal::floor(t506);
      float t508 = t507 * 16384.0;
      float t509 = t504 - t508;
      memory[599948661] = t509;
      float t511 = t509 >= 16384.0;
      if (t511) {
        float t513 = t509 - 16384.0;
        memory[599948661] = t513;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(536), value: global(536)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(502) - handled in variable access */
    float t519 = (t[9*frameCount + id] - metal::floor(t[9*frameCount + id] / 16384.0) * 16384.0);
    float t520 = t519 < 0.0;
    float t521 = t519 + 16384.0;
    float t522 = metal::select(t519, t521, t520 > 0.0);
    float t523 = t522;
    float t524 = metal::floor(t523);
    float t525 = t523 - t524;
    float t526 = t524 + 1.0;
    float t527 = t526 >= 16384.0;
    float t528 = metal::select(t526, 0.0, t527 > 0.0);
    int t529 = (int)t524;
    float t530 = memory[8889 + t529];
    int t531 = (int)t528;
    float t532 = memory[8889 + t531];
    float t533 = 1.0 - t525;
    float t534 = t530 * t533;
    float t535 = t532 * t525;
    t[10*frameCount + id] = t534 + t535;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(537), value: global(537)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[11*frameCount + i] = memory[599948662];
      float t538 = t[11*frameCount + i] + 1.0;
      float t539 = metal::select(t538, 0.0, 0.0 > 0.0);
      float t540 = t539;
      float t541 = (t540 * 0.0078125);
      float t542 = metal::floor(t541);
      float t543 = t542 * 128.0;
      float t544 = t539 - t543;
      memory[599948662] = t544;
      float t546 = t544 >= 128.0;
      if (t546) {
        float t548 = t544 - 128.0;
        memory[599948662] = t548;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(558), value: global(558)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(537) - handled in variable access */
    /* loadGlobal(536) - handled in variable access */
    /* loadGlobal(483) - handled in variable access */
    int t554 = id;
    int t555 = t554 * 1024;
    int t556 = t554 * 257;
    float t557 = t[11*frameCount + id] == 0.0;
    t[12*frameCount + id] = 0.0;
    if (t557) {
      for (uint _pr560 = 0; _pr560 < 512; _pr560++) {
        float t561 = (float)_pr560;
        float t562 = 6.283185 * t561;
        float t563 = (t562 * 0.0019569471);
        float t564 = metal::cos(t563);
        float t565 = 1.0 - t564;
        float t566 = 0.5 * t565;
        float t567 = (float)t554;
        float t568 = t567 - 511.0;
        float t569 = t568 + t561;
        float t570 = (t569 < 0 || t569 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t569];
        float t571 = (t569 < 0 || t569 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t569];
        int t572 = t555 + _pr560;
        float t573 = t570 * t566;
        memory[4255220 + t572] = t573;
        int t575 = t555 + _pr560;
        int t576 = t575 + 512;
        memory[4255220 + t576] = 0.0;
        int t578 = t555 + _pr560;
        float t579 = t571 * t566;
        memory[21032436 + t578] = t579;
        int t581 = t555 + _pr560;
        int t582 = t581 + 512;
        memory[21032436 + t582] = 0.0;
        memory[25460 + (int)_pr560] = t566;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t586 = 0; t586 < 512; t586++) {
        float t587 = (float)t586;
        float t588 = (t587 - metal::floor(t587 / 2.0) * 2.0);
        float t589 = t588;
        float t590 = (t587 * 0.5);
        float t591 = metal::floor(t590);
        float t592 = t589 * 2.0;
        float t593 = (t591 - metal::floor(t591 / 2.0) * 2.0);
        float t594 = t592 + t593;
        float t595 = (t591 * 0.5);
        float t596 = metal::floor(t595);
        float t597 = t594 * 2.0;
        float t598 = (t596 - metal::floor(t596 / 2.0) * 2.0);
        float t599 = t597 + t598;
        float t600 = (t596 * 0.5);
        float t601 = metal::floor(t600);
        float t602 = t599 * 2.0;
        float t603 = (t601 - metal::floor(t601 / 2.0) * 2.0);
        float t604 = t602 + t603;
        float t605 = (t601 * 0.5);
        float t606 = metal::floor(t605);
        float t607 = t604 * 2.0;
        float t608 = (t606 - metal::floor(t606 / 2.0) * 2.0);
        float t609 = t607 + t608;
        float t610 = (t606 * 0.5);
        float t611 = metal::floor(t610);
        float t612 = t609 * 2.0;
        float t613 = (t611 - metal::floor(t611 / 2.0) * 2.0);
        float t614 = t612 + t613;
        float t615 = (t611 * 0.5);
        float t616 = metal::floor(t615);
        float t617 = t614 * 2.0;
        float t618 = (t616 - metal::floor(t616 / 2.0) * 2.0);
        float t619 = t617 + t618;
        float t620 = (t616 * 0.5);
        float t621 = metal::floor(t620);
        float t622 = t619 * 2.0;
        float t623 = (t621 - metal::floor(t621 / 2.0) * 2.0);
        float t624 = t622 + t623;
        float t625 = (t621 * 0.5);
        float t626 = metal::floor(t625);
        float t627 = t624 * 2.0;
        float t628 = (t626 - metal::floor(t626 / 2.0) * 2.0);
        float t629 = t627 + t628;
        float t630 = (t626 * 0.5);
        float t631 = metal::floor(t630);
        float t632 = (float)t586;
        float t633 = t632 < t629;
        int t634 = (int)t629;
        int t635 = t555 + t586;
        float t636 = memory[4255220 + t635];
        int t637 = t555 + t586;
        int t638 = t637 + 512;
        float t639 = memory[4255220 + t638];
        int t640 = t555 + t634;
        float t641 = memory[4255220 + t640];
        int t642 = t555 + t634;
        int t643 = t642 + 512;
        float t644 = memory[4255220 + t643];
        float t645 = metal::select(t636, t641, t633 > 0.0);
        float t646 = metal::select(t639, t644, t633 > 0.0);
        float t647 = metal::select(t641, t636, t633 > 0.0);
        float t648 = metal::select(t644, t639, t633 > 0.0);
        int t649 = t555 + t586;
        memory[4255220 + t649] = t645;
        int t651 = t555 + t586;
        int t652 = t651 + 512;
        memory[4255220 + t652] = t646;
        int t654 = t555 + t634;
        memory[4255220 + t654] = t647;
        int t656 = t555 + t634;
        int t657 = t656 + 512;
        memory[4255220 + t657] = t648;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr660 = 0; _pr660 < 256; _pr660++) {
        float t661 = (float)_pr660;
        float t662 = t661;
        float t663 = metal::floor(t662);
        float t664 = t663;
        float t665 = t661 - t664;
        float t666 = t663 * 2.0;
        float t667 = t666 + t665;
        float t668 = t667 + 1.0;
        float t669 = -6.283185 * t665;
        float t670 = (t669 * 0.5);
        float t671 = metal::cos(t670);
        float t672 = metal::sin(t670);
        int t673 = (int)t667;
        int t674 = (int)t668;
        int t675 = t555 + t673;
        float t676 = memory[4255220 + t675];
        int t677 = t555 + t673;
        int t678 = t677 + 512;
        float t679 = memory[4255220 + t678];
        int t680 = t555 + t674;
        float t681 = memory[4255220 + t680];
        int t682 = t555 + t674;
        int t683 = t682 + 512;
        float t684 = memory[4255220 + t683];
        float t685 = t671 * t681;
        float t686 = t672 * t684;
        float t687 = t685 - t686;
        float t688 = t671 * t684;
        float t689 = t672 * t681;
        float t690 = t688 + t689;
        int t691 = t555 + t673;
        float t692 = t676 + t687;
        memory[4255220 + t691] = t692;
        int t694 = t555 + t673;
        int t695 = t694 + 512;
        float t696 = t679 + t690;
        memory[4255220 + t695] = t696;
        int t698 = t555 + t674;
        float t699 = t676 - t687;
        memory[4255220 + t698] = t699;
        int t701 = t555 + t674;
        int t702 = t701 + 512;
        float t703 = t679 - t690;
        memory[4255220 + t702] = t703;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr706 = 0; _pr706 < 256; _pr706++) {
        float t707 = (float)_pr706;
        float t708 = (t707 * 0.5);
        float t709 = metal::floor(t708);
        float t710 = t709 * 2.0;
        float t711 = t707 - t710;
        float t712 = t709 * 4.0;
        float t713 = t712 + t711;
        float t714 = t713 + 2.0;
        float t715 = -6.283185 * t711;
        float t716 = (t715 * 0.25);
        float t717 = metal::cos(t716);
        float t718 = metal::sin(t716);
        int t719 = (int)t713;
        int t720 = (int)t714;
        int t721 = t555 + t719;
        float t722 = memory[4255220 + t721];
        int t723 = t555 + t719;
        int t724 = t723 + 512;
        float t725 = memory[4255220 + t724];
        int t726 = t555 + t720;
        float t727 = memory[4255220 + t726];
        int t728 = t555 + t720;
        int t729 = t728 + 512;
        float t730 = memory[4255220 + t729];
        float t731 = t717 * t727;
        float t732 = t718 * t730;
        float t733 = t731 - t732;
        float t734 = t717 * t730;
        float t735 = t718 * t727;
        float t736 = t734 + t735;
        int t737 = t555 + t719;
        float t738 = t722 + t733;
        memory[4255220 + t737] = t738;
        int t740 = t555 + t719;
        int t741 = t740 + 512;
        float t742 = t725 + t736;
        memory[4255220 + t741] = t742;
        int t744 = t555 + t720;
        float t745 = t722 - t733;
        memory[4255220 + t744] = t745;
        int t747 = t555 + t720;
        int t748 = t747 + 512;
        float t749 = t725 - t736;
        memory[4255220 + t748] = t749;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr752 = 0; _pr752 < 256; _pr752++) {
        float t753 = (float)_pr752;
        float t754 = (t753 * 0.25);
        float t755 = metal::floor(t754);
        float t756 = t755 * 4.0;
        float t757 = t753 - t756;
        float t758 = t755 * 8.0;
        float t759 = t758 + t757;
        float t760 = t759 + 4.0;
        float t761 = -6.283185 * t757;
        float t762 = (t761 * 0.125);
        float t763 = metal::cos(t762);
        float t764 = metal::sin(t762);
        int t765 = (int)t759;
        int t766 = (int)t760;
        int t767 = t555 + t765;
        float t768 = memory[4255220 + t767];
        int t769 = t555 + t765;
        int t770 = t769 + 512;
        float t771 = memory[4255220 + t770];
        int t772 = t555 + t766;
        float t773 = memory[4255220 + t772];
        int t774 = t555 + t766;
        int t775 = t774 + 512;
        float t776 = memory[4255220 + t775];
        float t777 = t763 * t773;
        float t778 = t764 * t776;
        float t779 = t777 - t778;
        float t780 = t763 * t776;
        float t781 = t764 * t773;
        float t782 = t780 + t781;
        int t783 = t555 + t765;
        float t784 = t768 + t779;
        memory[4255220 + t783] = t784;
        int t786 = t555 + t765;
        int t787 = t786 + 512;
        float t788 = t771 + t782;
        memory[4255220 + t787] = t788;
        int t790 = t555 + t766;
        float t791 = t768 - t779;
        memory[4255220 + t790] = t791;
        int t793 = t555 + t766;
        int t794 = t793 + 512;
        float t795 = t771 - t782;
        memory[4255220 + t794] = t795;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr798 = 0; _pr798 < 256; _pr798++) {
        float t799 = (float)_pr798;
        float t800 = (t799 * 0.125);
        float t801 = metal::floor(t800);
        float t802 = t801 * 8.0;
        float t803 = t799 - t802;
        float t804 = t801 * 16.0;
        float t805 = t804 + t803;
        float t806 = t805 + 8.0;
        float t807 = -6.283185 * t803;
        float t808 = (t807 * 0.0625);
        float t809 = metal::cos(t808);
        float t810 = metal::sin(t808);
        int t811 = (int)t805;
        int t812 = (int)t806;
        int t813 = t555 + t811;
        float t814 = memory[4255220 + t813];
        int t815 = t555 + t811;
        int t816 = t815 + 512;
        float t817 = memory[4255220 + t816];
        int t818 = t555 + t812;
        float t819 = memory[4255220 + t818];
        int t820 = t555 + t812;
        int t821 = t820 + 512;
        float t822 = memory[4255220 + t821];
        float t823 = t809 * t819;
        float t824 = t810 * t822;
        float t825 = t823 - t824;
        float t826 = t809 * t822;
        float t827 = t810 * t819;
        float t828 = t826 + t827;
        int t829 = t555 + t811;
        float t830 = t814 + t825;
        memory[4255220 + t829] = t830;
        int t832 = t555 + t811;
        int t833 = t832 + 512;
        float t834 = t817 + t828;
        memory[4255220 + t833] = t834;
        int t836 = t555 + t812;
        float t837 = t814 - t825;
        memory[4255220 + t836] = t837;
        int t839 = t555 + t812;
        int t840 = t839 + 512;
        float t841 = t817 - t828;
        memory[4255220 + t840] = t841;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr844 = 0; _pr844 < 256; _pr844++) {
        float t845 = (float)_pr844;
        float t846 = (t845 * 0.0625);
        float t847 = metal::floor(t846);
        float t848 = t847 * 16.0;
        float t849 = t845 - t848;
        float t850 = t847 * 32.0;
        float t851 = t850 + t849;
        float t852 = t851 + 16.0;
        float t853 = -6.283185 * t849;
        float t854 = (t853 * 0.03125);
        float t855 = metal::cos(t854);
        float t856 = metal::sin(t854);
        int t857 = (int)t851;
        int t858 = (int)t852;
        int t859 = t555 + t857;
        float t860 = memory[4255220 + t859];
        int t861 = t555 + t857;
        int t862 = t861 + 512;
        float t863 = memory[4255220 + t862];
        int t864 = t555 + t858;
        float t865 = memory[4255220 + t864];
        int t866 = t555 + t858;
        int t867 = t866 + 512;
        float t868 = memory[4255220 + t867];
        float t869 = t855 * t865;
        float t870 = t856 * t868;
        float t871 = t869 - t870;
        float t872 = t855 * t868;
        float t873 = t856 * t865;
        float t874 = t872 + t873;
        int t875 = t555 + t857;
        float t876 = t860 + t871;
        memory[4255220 + t875] = t876;
        int t878 = t555 + t857;
        int t879 = t878 + 512;
        float t880 = t863 + t874;
        memory[4255220 + t879] = t880;
        int t882 = t555 + t858;
        float t883 = t860 - t871;
        memory[4255220 + t882] = t883;
        int t885 = t555 + t858;
        int t886 = t885 + 512;
        float t887 = t863 - t874;
        memory[4255220 + t886] = t887;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr890 = 0; _pr890 < 256; _pr890++) {
        float t891 = (float)_pr890;
        float t892 = (t891 * 0.03125);
        float t893 = metal::floor(t892);
        float t894 = t893 * 32.0;
        float t895 = t891 - t894;
        float t896 = t893 * 64.0;
        float t897 = t896 + t895;
        float t898 = t897 + 32.0;
        float t899 = -6.283185 * t895;
        float t900 = (t899 * 0.015625);
        float t901 = metal::cos(t900);
        float t902 = metal::sin(t900);
        int t903 = (int)t897;
        int t904 = (int)t898;
        int t905 = t555 + t903;
        float t906 = memory[4255220 + t905];
        int t907 = t555 + t903;
        int t908 = t907 + 512;
        float t909 = memory[4255220 + t908];
        int t910 = t555 + t904;
        float t911 = memory[4255220 + t910];
        int t912 = t555 + t904;
        int t913 = t912 + 512;
        float t914 = memory[4255220 + t913];
        float t915 = t901 * t911;
        float t916 = t902 * t914;
        float t917 = t915 - t916;
        float t918 = t901 * t914;
        float t919 = t902 * t911;
        float t920 = t918 + t919;
        int t921 = t555 + t903;
        float t922 = t906 + t917;
        memory[4255220 + t921] = t922;
        int t924 = t555 + t903;
        int t925 = t924 + 512;
        float t926 = t909 + t920;
        memory[4255220 + t925] = t926;
        int t928 = t555 + t904;
        float t929 = t906 - t917;
        memory[4255220 + t928] = t929;
        int t931 = t555 + t904;
        int t932 = t931 + 512;
        float t933 = t909 - t920;
        memory[4255220 + t932] = t933;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr936 = 0; _pr936 < 256; _pr936++) {
        float t937 = (float)_pr936;
        float t938 = (t937 * 0.015625);
        float t939 = metal::floor(t938);
        float t940 = t939 * 64.0;
        float t941 = t937 - t940;
        float t942 = t939 * 128.0;
        float t943 = t942 + t941;
        float t944 = t943 + 64.0;
        float t945 = -6.283185 * t941;
        float t946 = (t945 * 0.0078125);
        float t947 = metal::cos(t946);
        float t948 = metal::sin(t946);
        int t949 = (int)t943;
        int t950 = (int)t944;
        int t951 = t555 + t949;
        float t952 = memory[4255220 + t951];
        int t953 = t555 + t949;
        int t954 = t953 + 512;
        float t955 = memory[4255220 + t954];
        int t956 = t555 + t950;
        float t957 = memory[4255220 + t956];
        int t958 = t555 + t950;
        int t959 = t958 + 512;
        float t960 = memory[4255220 + t959];
        float t961 = t947 * t957;
        float t962 = t948 * t960;
        float t963 = t961 - t962;
        float t964 = t947 * t960;
        float t965 = t948 * t957;
        float t966 = t964 + t965;
        int t967 = t555 + t949;
        float t968 = t952 + t963;
        memory[4255220 + t967] = t968;
        int t970 = t555 + t949;
        int t971 = t970 + 512;
        float t972 = t955 + t966;
        memory[4255220 + t971] = t972;
        int t974 = t555 + t950;
        float t975 = t952 - t963;
        memory[4255220 + t974] = t975;
        int t977 = t555 + t950;
        int t978 = t977 + 512;
        float t979 = t955 - t966;
        memory[4255220 + t978] = t979;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr982 = 0; _pr982 < 256; _pr982++) {
        float t983 = (float)_pr982;
        float t984 = (t983 * 0.0078125);
        float t985 = metal::floor(t984);
        float t986 = t985 * 128.0;
        float t987 = t983 - t986;
        float t988 = t985 * 256.0;
        float t989 = t988 + t987;
        float t990 = t989 + 128.0;
        float t991 = -6.283185 * t987;
        float t992 = (t991 * 0.00390625);
        float t993 = metal::cos(t992);
        float t994 = metal::sin(t992);
        int t995 = (int)t989;
        int t996 = (int)t990;
        int t997 = t555 + t995;
        float t998 = memory[4255220 + t997];
        int t999 = t555 + t995;
        int t1000 = t999 + 512;
        float t1001 = memory[4255220 + t1000];
        int t1002 = t555 + t996;
        float t1003 = memory[4255220 + t1002];
        int t1004 = t555 + t996;
        int t1005 = t1004 + 512;
        float t1006 = memory[4255220 + t1005];
        float t1007 = t993 * t1003;
        float t1008 = t994 * t1006;
        float t1009 = t1007 - t1008;
        float t1010 = t993 * t1006;
        float t1011 = t994 * t1003;
        float t1012 = t1010 + t1011;
        int t1013 = t555 + t995;
        float t1014 = t998 + t1009;
        memory[4255220 + t1013] = t1014;
        int t1016 = t555 + t995;
        int t1017 = t1016 + 512;
        float t1018 = t1001 + t1012;
        memory[4255220 + t1017] = t1018;
        int t1020 = t555 + t996;
        float t1021 = t998 - t1009;
        memory[4255220 + t1020] = t1021;
        int t1023 = t555 + t996;
        int t1024 = t1023 + 512;
        float t1025 = t1001 - t1012;
        memory[4255220 + t1024] = t1025;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1028 = 0; _pr1028 < 256; _pr1028++) {
        float t1029 = (float)_pr1028;
        float t1030 = (t1029 * 0.00390625);
        float t1031 = metal::floor(t1030);
        float t1032 = t1031 * 256.0;
        float t1033 = t1029 - t1032;
        float t1034 = t1031 * 512.0;
        float t1035 = t1034 + t1033;
        float t1036 = t1035 + 256.0;
        float t1037 = -6.283185 * t1033;
        float t1038 = (t1037 * 0.001953125);
        float t1039 = metal::cos(t1038);
        float t1040 = metal::sin(t1038);
        int t1041 = (int)t1035;
        int t1042 = (int)t1036;
        int t1043 = t555 + t1041;
        float t1044 = memory[4255220 + t1043];
        int t1045 = t555 + t1041;
        int t1046 = t1045 + 512;
        float t1047 = memory[4255220 + t1046];
        int t1048 = t555 + t1042;
        float t1049 = memory[4255220 + t1048];
        int t1050 = t555 + t1042;
        int t1051 = t1050 + 512;
        float t1052 = memory[4255220 + t1051];
        float t1053 = t1039 * t1049;
        float t1054 = t1040 * t1052;
        float t1055 = t1053 - t1054;
        float t1056 = t1039 * t1052;
        float t1057 = t1040 * t1049;
        float t1058 = t1056 + t1057;
        int t1059 = t555 + t1041;
        float t1060 = t1044 + t1055;
        memory[4255220 + t1059] = t1060;
        int t1062 = t555 + t1041;
        int t1063 = t1062 + 512;
        float t1064 = t1047 + t1058;
        memory[4255220 + t1063] = t1064;
        int t1066 = t555 + t1042;
        float t1067 = t1044 - t1055;
        memory[4255220 + t1066] = t1067;
        int t1069 = t555 + t1042;
        int t1070 = t1069 + 512;
        float t1071 = t1047 - t1058;
        memory[4255220 + t1070] = t1071;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1074 = 0; t1074 < 512; t1074++) {
        float t1075 = (float)t1074;
        float t1076 = (t1075 - metal::floor(t1075 / 2.0) * 2.0);
        float t1077 = t1076;
        float t1078 = (t1075 * 0.5);
        float t1079 = metal::floor(t1078);
        float t1080 = t1077 * 2.0;
        float t1081 = (t1079 - metal::floor(t1079 / 2.0) * 2.0);
        float t1082 = t1080 + t1081;
        float t1083 = (t1079 * 0.5);
        float t1084 = metal::floor(t1083);
        float t1085 = t1082 * 2.0;
        float t1086 = (t1084 - metal::floor(t1084 / 2.0) * 2.0);
        float t1087 = t1085 + t1086;
        float t1088 = (t1084 * 0.5);
        float t1089 = metal::floor(t1088);
        float t1090 = t1087 * 2.0;
        float t1091 = (t1089 - metal::floor(t1089 / 2.0) * 2.0);
        float t1092 = t1090 + t1091;
        float t1093 = (t1089 * 0.5);
        float t1094 = metal::floor(t1093);
        float t1095 = t1092 * 2.0;
        float t1096 = (t1094 - metal::floor(t1094 / 2.0) * 2.0);
        float t1097 = t1095 + t1096;
        float t1098 = (t1094 * 0.5);
        float t1099 = metal::floor(t1098);
        float t1100 = t1097 * 2.0;
        float t1101 = (t1099 - metal::floor(t1099 / 2.0) * 2.0);
        float t1102 = t1100 + t1101;
        float t1103 = (t1099 * 0.5);
        float t1104 = metal::floor(t1103);
        float t1105 = t1102 * 2.0;
        float t1106 = (t1104 - metal::floor(t1104 / 2.0) * 2.0);
        float t1107 = t1105 + t1106;
        float t1108 = (t1104 * 0.5);
        float t1109 = metal::floor(t1108);
        float t1110 = t1107 * 2.0;
        float t1111 = (t1109 - metal::floor(t1109 / 2.0) * 2.0);
        float t1112 = t1110 + t1111;
        float t1113 = (t1109 * 0.5);
        float t1114 = metal::floor(t1113);
        float t1115 = t1112 * 2.0;
        float t1116 = (t1114 - metal::floor(t1114 / 2.0) * 2.0);
        float t1117 = t1115 + t1116;
        float t1118 = (t1114 * 0.5);
        float t1119 = metal::floor(t1118);
        float t1120 = (float)t1074;
        float t1121 = t1120 < t1117;
        int t1122 = (int)t1117;
        int t1123 = t555 + t1074;
        float t1124 = memory[21032436 + t1123];
        int t1125 = t555 + t1074;
        int t1126 = t1125 + 512;
        float t1127 = memory[21032436 + t1126];
        int t1128 = t555 + t1122;
        float t1129 = memory[21032436 + t1128];
        int t1130 = t555 + t1122;
        int t1131 = t1130 + 512;
        float t1132 = memory[21032436 + t1131];
        float t1133 = metal::select(t1124, t1129, t1121 > 0.0);
        float t1134 = metal::select(t1127, t1132, t1121 > 0.0);
        float t1135 = metal::select(t1129, t1124, t1121 > 0.0);
        float t1136 = metal::select(t1132, t1127, t1121 > 0.0);
        int t1137 = t555 + t1074;
        memory[21032436 + t1137] = t1133;
        int t1139 = t555 + t1074;
        int t1140 = t1139 + 512;
        memory[21032436 + t1140] = t1134;
        int t1142 = t555 + t1122;
        memory[21032436 + t1142] = t1135;
        int t1144 = t555 + t1122;
        int t1145 = t1144 + 512;
        memory[21032436 + t1145] = t1136;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1148 = 0; _pr1148 < 256; _pr1148++) {
        float t1149 = (float)_pr1148;
        float t1150 = t1149;
        float t1151 = metal::floor(t1150);
        float t1152 = t1151;
        float t1153 = t1149 - t1152;
        float t1154 = t1151 * 2.0;
        float t1155 = t1154 + t1153;
        float t1156 = t1155 + 1.0;
        float t1157 = -6.283185 * t1153;
        float t1158 = (t1157 * 0.5);
        float t1159 = metal::cos(t1158);
        float t1160 = metal::sin(t1158);
        int t1161 = (int)t1155;
        int t1162 = (int)t1156;
        int t1163 = t555 + t1161;
        float t1164 = memory[21032436 + t1163];
        int t1165 = t555 + t1161;
        int t1166 = t1165 + 512;
        float t1167 = memory[21032436 + t1166];
        int t1168 = t555 + t1162;
        float t1169 = memory[21032436 + t1168];
        int t1170 = t555 + t1162;
        int t1171 = t1170 + 512;
        float t1172 = memory[21032436 + t1171];
        float t1173 = t1159 * t1169;
        float t1174 = t1160 * t1172;
        float t1175 = t1173 - t1174;
        float t1176 = t1159 * t1172;
        float t1177 = t1160 * t1169;
        float t1178 = t1176 + t1177;
        int t1179 = t555 + t1161;
        float t1180 = t1164 + t1175;
        memory[21032436 + t1179] = t1180;
        int t1182 = t555 + t1161;
        int t1183 = t1182 + 512;
        float t1184 = t1167 + t1178;
        memory[21032436 + t1183] = t1184;
        int t1186 = t555 + t1162;
        float t1187 = t1164 - t1175;
        memory[21032436 + t1186] = t1187;
        int t1189 = t555 + t1162;
        int t1190 = t1189 + 512;
        float t1191 = t1167 - t1178;
        memory[21032436 + t1190] = t1191;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1194 = 0; _pr1194 < 256; _pr1194++) {
        float t1195 = (float)_pr1194;
        float t1196 = (t1195 * 0.5);
        float t1197 = metal::floor(t1196);
        float t1198 = t1197 * 2.0;
        float t1199 = t1195 - t1198;
        float t1200 = t1197 * 4.0;
        float t1201 = t1200 + t1199;
        float t1202 = t1201 + 2.0;
        float t1203 = -6.283185 * t1199;
        float t1204 = (t1203 * 0.25);
        float t1205 = metal::cos(t1204);
        float t1206 = metal::sin(t1204);
        int t1207 = (int)t1201;
        int t1208 = (int)t1202;
        int t1209 = t555 + t1207;
        float t1210 = memory[21032436 + t1209];
        int t1211 = t555 + t1207;
        int t1212 = t1211 + 512;
        float t1213 = memory[21032436 + t1212];
        int t1214 = t555 + t1208;
        float t1215 = memory[21032436 + t1214];
        int t1216 = t555 + t1208;
        int t1217 = t1216 + 512;
        float t1218 = memory[21032436 + t1217];
        float t1219 = t1205 * t1215;
        float t1220 = t1206 * t1218;
        float t1221 = t1219 - t1220;
        float t1222 = t1205 * t1218;
        float t1223 = t1206 * t1215;
        float t1224 = t1222 + t1223;
        int t1225 = t555 + t1207;
        float t1226 = t1210 + t1221;
        memory[21032436 + t1225] = t1226;
        int t1228 = t555 + t1207;
        int t1229 = t1228 + 512;
        float t1230 = t1213 + t1224;
        memory[21032436 + t1229] = t1230;
        int t1232 = t555 + t1208;
        float t1233 = t1210 - t1221;
        memory[21032436 + t1232] = t1233;
        int t1235 = t555 + t1208;
        int t1236 = t1235 + 512;
        float t1237 = t1213 - t1224;
        memory[21032436 + t1236] = t1237;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1240 = 0; _pr1240 < 256; _pr1240++) {
        float t1241 = (float)_pr1240;
        float t1242 = (t1241 * 0.25);
        float t1243 = metal::floor(t1242);
        float t1244 = t1243 * 4.0;
        float t1245 = t1241 - t1244;
        float t1246 = t1243 * 8.0;
        float t1247 = t1246 + t1245;
        float t1248 = t1247 + 4.0;
        float t1249 = -6.283185 * t1245;
        float t1250 = (t1249 * 0.125);
        float t1251 = metal::cos(t1250);
        float t1252 = metal::sin(t1250);
        int t1253 = (int)t1247;
        int t1254 = (int)t1248;
        int t1255 = t555 + t1253;
        float t1256 = memory[21032436 + t1255];
        int t1257 = t555 + t1253;
        int t1258 = t1257 + 512;
        float t1259 = memory[21032436 + t1258];
        int t1260 = t555 + t1254;
        float t1261 = memory[21032436 + t1260];
        int t1262 = t555 + t1254;
        int t1263 = t1262 + 512;
        float t1264 = memory[21032436 + t1263];
        float t1265 = t1251 * t1261;
        float t1266 = t1252 * t1264;
        float t1267 = t1265 - t1266;
        float t1268 = t1251 * t1264;
        float t1269 = t1252 * t1261;
        float t1270 = t1268 + t1269;
        int t1271 = t555 + t1253;
        float t1272 = t1256 + t1267;
        memory[21032436 + t1271] = t1272;
        int t1274 = t555 + t1253;
        int t1275 = t1274 + 512;
        float t1276 = t1259 + t1270;
        memory[21032436 + t1275] = t1276;
        int t1278 = t555 + t1254;
        float t1279 = t1256 - t1267;
        memory[21032436 + t1278] = t1279;
        int t1281 = t555 + t1254;
        int t1282 = t1281 + 512;
        float t1283 = t1259 - t1270;
        memory[21032436 + t1282] = t1283;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1286 = 0; _pr1286 < 256; _pr1286++) {
        float t1287 = (float)_pr1286;
        float t1288 = (t1287 * 0.125);
        float t1289 = metal::floor(t1288);
        float t1290 = t1289 * 8.0;
        float t1291 = t1287 - t1290;
        float t1292 = t1289 * 16.0;
        float t1293 = t1292 + t1291;
        float t1294 = t1293 + 8.0;
        float t1295 = -6.283185 * t1291;
        float t1296 = (t1295 * 0.0625);
        float t1297 = metal::cos(t1296);
        float t1298 = metal::sin(t1296);
        int t1299 = (int)t1293;
        int t1300 = (int)t1294;
        int t1301 = t555 + t1299;
        float t1302 = memory[21032436 + t1301];
        int t1303 = t555 + t1299;
        int t1304 = t1303 + 512;
        float t1305 = memory[21032436 + t1304];
        int t1306 = t555 + t1300;
        float t1307 = memory[21032436 + t1306];
        int t1308 = t555 + t1300;
        int t1309 = t1308 + 512;
        float t1310 = memory[21032436 + t1309];
        float t1311 = t1297 * t1307;
        float t1312 = t1298 * t1310;
        float t1313 = t1311 - t1312;
        float t1314 = t1297 * t1310;
        float t1315 = t1298 * t1307;
        float t1316 = t1314 + t1315;
        int t1317 = t555 + t1299;
        float t1318 = t1302 + t1313;
        memory[21032436 + t1317] = t1318;
        int t1320 = t555 + t1299;
        int t1321 = t1320 + 512;
        float t1322 = t1305 + t1316;
        memory[21032436 + t1321] = t1322;
        int t1324 = t555 + t1300;
        float t1325 = t1302 - t1313;
        memory[21032436 + t1324] = t1325;
        int t1327 = t555 + t1300;
        int t1328 = t1327 + 512;
        float t1329 = t1305 - t1316;
        memory[21032436 + t1328] = t1329;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1332 = 0; _pr1332 < 256; _pr1332++) {
        float t1333 = (float)_pr1332;
        float t1334 = (t1333 * 0.0625);
        float t1335 = metal::floor(t1334);
        float t1336 = t1335 * 16.0;
        float t1337 = t1333 - t1336;
        float t1338 = t1335 * 32.0;
        float t1339 = t1338 + t1337;
        float t1340 = t1339 + 16.0;
        float t1341 = -6.283185 * t1337;
        float t1342 = (t1341 * 0.03125);
        float t1343 = metal::cos(t1342);
        float t1344 = metal::sin(t1342);
        int t1345 = (int)t1339;
        int t1346 = (int)t1340;
        int t1347 = t555 + t1345;
        float t1348 = memory[21032436 + t1347];
        int t1349 = t555 + t1345;
        int t1350 = t1349 + 512;
        float t1351 = memory[21032436 + t1350];
        int t1352 = t555 + t1346;
        float t1353 = memory[21032436 + t1352];
        int t1354 = t555 + t1346;
        int t1355 = t1354 + 512;
        float t1356 = memory[21032436 + t1355];
        float t1357 = t1343 * t1353;
        float t1358 = t1344 * t1356;
        float t1359 = t1357 - t1358;
        float t1360 = t1343 * t1356;
        float t1361 = t1344 * t1353;
        float t1362 = t1360 + t1361;
        int t1363 = t555 + t1345;
        float t1364 = t1348 + t1359;
        memory[21032436 + t1363] = t1364;
        int t1366 = t555 + t1345;
        int t1367 = t1366 + 512;
        float t1368 = t1351 + t1362;
        memory[21032436 + t1367] = t1368;
        int t1370 = t555 + t1346;
        float t1371 = t1348 - t1359;
        memory[21032436 + t1370] = t1371;
        int t1373 = t555 + t1346;
        int t1374 = t1373 + 512;
        float t1375 = t1351 - t1362;
        memory[21032436 + t1374] = t1375;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1378 = 0; _pr1378 < 256; _pr1378++) {
        float t1379 = (float)_pr1378;
        float t1380 = (t1379 * 0.03125);
        float t1381 = metal::floor(t1380);
        float t1382 = t1381 * 32.0;
        float t1383 = t1379 - t1382;
        float t1384 = t1381 * 64.0;
        float t1385 = t1384 + t1383;
        float t1386 = t1385 + 32.0;
        float t1387 = -6.283185 * t1383;
        float t1388 = (t1387 * 0.015625);
        float t1389 = metal::cos(t1388);
        float t1390 = metal::sin(t1388);
        int t1391 = (int)t1385;
        int t1392 = (int)t1386;
        int t1393 = t555 + t1391;
        float t1394 = memory[21032436 + t1393];
        int t1395 = t555 + t1391;
        int t1396 = t1395 + 512;
        float t1397 = memory[21032436 + t1396];
        int t1398 = t555 + t1392;
        float t1399 = memory[21032436 + t1398];
        int t1400 = t555 + t1392;
        int t1401 = t1400 + 512;
        float t1402 = memory[21032436 + t1401];
        float t1403 = t1389 * t1399;
        float t1404 = t1390 * t1402;
        float t1405 = t1403 - t1404;
        float t1406 = t1389 * t1402;
        float t1407 = t1390 * t1399;
        float t1408 = t1406 + t1407;
        int t1409 = t555 + t1391;
        float t1410 = t1394 + t1405;
        memory[21032436 + t1409] = t1410;
        int t1412 = t555 + t1391;
        int t1413 = t1412 + 512;
        float t1414 = t1397 + t1408;
        memory[21032436 + t1413] = t1414;
        int t1416 = t555 + t1392;
        float t1417 = t1394 - t1405;
        memory[21032436 + t1416] = t1417;
        int t1419 = t555 + t1392;
        int t1420 = t1419 + 512;
        float t1421 = t1397 - t1408;
        memory[21032436 + t1420] = t1421;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1424 = 0; _pr1424 < 256; _pr1424++) {
        float t1425 = (float)_pr1424;
        float t1426 = (t1425 * 0.015625);
        float t1427 = metal::floor(t1426);
        float t1428 = t1427 * 64.0;
        float t1429 = t1425 - t1428;
        float t1430 = t1427 * 128.0;
        float t1431 = t1430 + t1429;
        float t1432 = t1431 + 64.0;
        float t1433 = -6.283185 * t1429;
        float t1434 = (t1433 * 0.0078125);
        float t1435 = metal::cos(t1434);
        float t1436 = metal::sin(t1434);
        int t1437 = (int)t1431;
        int t1438 = (int)t1432;
        int t1439 = t555 + t1437;
        float t1440 = memory[21032436 + t1439];
        int t1441 = t555 + t1437;
        int t1442 = t1441 + 512;
        float t1443 = memory[21032436 + t1442];
        int t1444 = t555 + t1438;
        float t1445 = memory[21032436 + t1444];
        int t1446 = t555 + t1438;
        int t1447 = t1446 + 512;
        float t1448 = memory[21032436 + t1447];
        float t1449 = t1435 * t1445;
        float t1450 = t1436 * t1448;
        float t1451 = t1449 - t1450;
        float t1452 = t1435 * t1448;
        float t1453 = t1436 * t1445;
        float t1454 = t1452 + t1453;
        int t1455 = t555 + t1437;
        float t1456 = t1440 + t1451;
        memory[21032436 + t1455] = t1456;
        int t1458 = t555 + t1437;
        int t1459 = t1458 + 512;
        float t1460 = t1443 + t1454;
        memory[21032436 + t1459] = t1460;
        int t1462 = t555 + t1438;
        float t1463 = t1440 - t1451;
        memory[21032436 + t1462] = t1463;
        int t1465 = t555 + t1438;
        int t1466 = t1465 + 512;
        float t1467 = t1443 - t1454;
        memory[21032436 + t1466] = t1467;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1470 = 0; _pr1470 < 256; _pr1470++) {
        float t1471 = (float)_pr1470;
        float t1472 = (t1471 * 0.0078125);
        float t1473 = metal::floor(t1472);
        float t1474 = t1473 * 128.0;
        float t1475 = t1471 - t1474;
        float t1476 = t1473 * 256.0;
        float t1477 = t1476 + t1475;
        float t1478 = t1477 + 128.0;
        float t1479 = -6.283185 * t1475;
        float t1480 = (t1479 * 0.00390625);
        float t1481 = metal::cos(t1480);
        float t1482 = metal::sin(t1480);
        int t1483 = (int)t1477;
        int t1484 = (int)t1478;
        int t1485 = t555 + t1483;
        float t1486 = memory[21032436 + t1485];
        int t1487 = t555 + t1483;
        int t1488 = t1487 + 512;
        float t1489 = memory[21032436 + t1488];
        int t1490 = t555 + t1484;
        float t1491 = memory[21032436 + t1490];
        int t1492 = t555 + t1484;
        int t1493 = t1492 + 512;
        float t1494 = memory[21032436 + t1493];
        float t1495 = t1481 * t1491;
        float t1496 = t1482 * t1494;
        float t1497 = t1495 - t1496;
        float t1498 = t1481 * t1494;
        float t1499 = t1482 * t1491;
        float t1500 = t1498 + t1499;
        int t1501 = t555 + t1483;
        float t1502 = t1486 + t1497;
        memory[21032436 + t1501] = t1502;
        int t1504 = t555 + t1483;
        int t1505 = t1504 + 512;
        float t1506 = t1489 + t1500;
        memory[21032436 + t1505] = t1506;
        int t1508 = t555 + t1484;
        float t1509 = t1486 - t1497;
        memory[21032436 + t1508] = t1509;
        int t1511 = t555 + t1484;
        int t1512 = t1511 + 512;
        float t1513 = t1489 - t1500;
        memory[21032436 + t1512] = t1513;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1516 = 0; _pr1516 < 256; _pr1516++) {
        float t1517 = (float)_pr1516;
        float t1518 = (t1517 * 0.00390625);
        float t1519 = metal::floor(t1518);
        float t1520 = t1519 * 256.0;
        float t1521 = t1517 - t1520;
        float t1522 = t1519 * 512.0;
        float t1523 = t1522 + t1521;
        float t1524 = t1523 + 256.0;
        float t1525 = -6.283185 * t1521;
        float t1526 = (t1525 * 0.001953125);
        float t1527 = metal::cos(t1526);
        float t1528 = metal::sin(t1526);
        int t1529 = (int)t1523;
        int t1530 = (int)t1524;
        int t1531 = t555 + t1529;
        float t1532 = memory[21032436 + t1531];
        int t1533 = t555 + t1529;
        int t1534 = t1533 + 512;
        float t1535 = memory[21032436 + t1534];
        int t1536 = t555 + t1530;
        float t1537 = memory[21032436 + t1536];
        int t1538 = t555 + t1530;
        int t1539 = t1538 + 512;
        float t1540 = memory[21032436 + t1539];
        float t1541 = t1527 * t1537;
        float t1542 = t1528 * t1540;
        float t1543 = t1541 - t1542;
        float t1544 = t1527 * t1540;
        float t1545 = t1528 * t1537;
        float t1546 = t1544 + t1545;
        int t1547 = t555 + t1529;
        float t1548 = t1532 + t1543;
        memory[21032436 + t1547] = t1548;
        int t1550 = t555 + t1529;
        int t1551 = t1550 + 512;
        float t1552 = t1535 + t1546;
        memory[21032436 + t1551] = t1552;
        int t1554 = t555 + t1530;
        float t1555 = t1532 - t1543;
        memory[21032436 + t1554] = t1555;
        int t1557 = t555 + t1530;
        int t1558 = t1557 + 512;
        float t1559 = t1535 - t1546;
        memory[21032436 + t1558] = t1559;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1562 = 0; _pr1562 < 257; _pr1562++) {
        int t1563 = t555 + _pr1562;
        float t1564 = memory[4255220 + t1563];
        int t1565 = t555 + _pr1562;
        int t1566 = t1565 + 512;
        float t1567 = memory[4255220 + t1566];
        float t1568 = t1564 * t1564;
        float t1569 = t1567 * t1567;
        float t1570 = t1568 + t1569;
        float t1571 = metal::sqrt(t1570);
        int t1572 = t556 + _pr1562;
        memory[37809652 + t1572] = t1571;
        int t1574 = t555 + _pr1562;
        float t1575 = memory[21032436 + t1574];
        int t1576 = t555 + _pr1562;
        int t1577 = t1576 + 512;
        float t1578 = memory[21032436 + t1577];
        float t1579 = t1575 * t1575;
        float t1580 = t1578 * t1578;
        float t1581 = t1579 + t1580;
        float t1582 = metal::sqrt(t1581);
        int t1583 = t556 + _pr1562;
        memory[42020340 + t1583] = t1582;
        float t1585 = t1571 - t1582;
        int t1586 = t556 + _pr1562;
        float t1587 = t1585 * t1585;
        memory[46231028 + t1586] = t1587;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1590 = 0; t1590 < 257; t1590++) {
        int t1591 = t556 + t1590;
        float t1592 = memory[46231028 + t1591];
        float t1593 = t[12*frameCount + id] + t1592;
        t[12*frameCount + id] = t1593;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1597), value: global(1597)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(558) - handled in variable access */
    float t1596 = (t[12*frameCount + id] * 6.1035156e-05);
    t[13*frameCount + id] = t1596;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1598), value: global(1598)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[14*frameCount + i] = memory[599948663];
      float t1599 = t[14*frameCount + i] + 1.0;
      float t1600 = metal::select(t1599, 0.0, 0.0 > 0.0);
      float t1601 = t1600;
      float t1602 = (t1601 * 0.00390625);
      float t1603 = metal::floor(t1602);
      float t1604 = t1603 * 256.0;
      float t1605 = t1600 - t1604;
      memory[599948663] = t1605;
      float t1607 = t1605 >= 256.0;
      if (t1607) {
        float t1609 = t1605 - 256.0;
        memory[599948663] = t1609;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1619), value: global(1619)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1598) - handled in variable access */
    /* loadGlobal(536) - handled in variable access */
    /* loadGlobal(483) - handled in variable access */
    int t1615 = id;
    int t1616 = t1615 * 2048;
    int t1617 = t1615 * 513;
    float t1618 = t[14*frameCount + id] == 0.0;
    t[15*frameCount + id] = 0.0;
    if (t1618) {
      for (uint _pr1621 = 0; _pr1621 < 1024; _pr1621++) {
        float t1622 = (float)_pr1621;
        float t1623 = 6.283185 * t1622;
        float t1624 = (t1623 * 0.0009775171);
        float t1625 = metal::cos(t1624);
        float t1626 = 1.0 - t1625;
        float t1627 = 0.5 * t1626;
        float t1628 = (float)t1615;
        float t1629 = t1628 - 1023.0;
        float t1630 = t1629 + t1622;
        float t1631 = (t1630 < 0 || t1630 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t1630];
        float t1632 = (t1630 < 0 || t1630 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t1630];
        int t1633 = t1616 + _pr1621;
        float t1634 = t1631 * t1627;
        memory[50441716 + t1633] = t1634;
        int t1636 = t1616 + _pr1621;
        int t1637 = t1636 + 1024;
        memory[50441716 + t1637] = 0.0;
        int t1639 = t1616 + _pr1621;
        float t1640 = t1632 * t1627;
        memory[83996148 + t1639] = t1640;
        int t1642 = t1616 + _pr1621;
        int t1643 = t1642 + 1024;
        memory[83996148 + t1643] = 0.0;
        memory[52788 + (int)_pr1621] = t1627;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1647 = 0; t1647 < 1024; t1647++) {
        float t1648 = (float)t1647;
        float t1649 = (t1648 - metal::floor(t1648 / 2.0) * 2.0);
        float t1650 = t1649;
        float t1651 = (t1648 * 0.5);
        float t1652 = metal::floor(t1651);
        float t1653 = t1650 * 2.0;
        float t1654 = (t1652 - metal::floor(t1652 / 2.0) * 2.0);
        float t1655 = t1653 + t1654;
        float t1656 = (t1652 * 0.5);
        float t1657 = metal::floor(t1656);
        float t1658 = t1655 * 2.0;
        float t1659 = (t1657 - metal::floor(t1657 / 2.0) * 2.0);
        float t1660 = t1658 + t1659;
        float t1661 = (t1657 * 0.5);
        float t1662 = metal::floor(t1661);
        float t1663 = t1660 * 2.0;
        float t1664 = (t1662 - metal::floor(t1662 / 2.0) * 2.0);
        float t1665 = t1663 + t1664;
        float t1666 = (t1662 * 0.5);
        float t1667 = metal::floor(t1666);
        float t1668 = t1665 * 2.0;
        float t1669 = (t1667 - metal::floor(t1667 / 2.0) * 2.0);
        float t1670 = t1668 + t1669;
        float t1671 = (t1667 * 0.5);
        float t1672 = metal::floor(t1671);
        float t1673 = t1670 * 2.0;
        float t1674 = (t1672 - metal::floor(t1672 / 2.0) * 2.0);
        float t1675 = t1673 + t1674;
        float t1676 = (t1672 * 0.5);
        float t1677 = metal::floor(t1676);
        float t1678 = t1675 * 2.0;
        float t1679 = (t1677 - metal::floor(t1677 / 2.0) * 2.0);
        float t1680 = t1678 + t1679;
        float t1681 = (t1677 * 0.5);
        float t1682 = metal::floor(t1681);
        float t1683 = t1680 * 2.0;
        float t1684 = (t1682 - metal::floor(t1682 / 2.0) * 2.0);
        float t1685 = t1683 + t1684;
        float t1686 = (t1682 * 0.5);
        float t1687 = metal::floor(t1686);
        float t1688 = t1685 * 2.0;
        float t1689 = (t1687 - metal::floor(t1687 / 2.0) * 2.0);
        float t1690 = t1688 + t1689;
        float t1691 = (t1687 * 0.5);
        float t1692 = metal::floor(t1691);
        float t1693 = t1690 * 2.0;
        float t1694 = (t1692 - metal::floor(t1692 / 2.0) * 2.0);
        float t1695 = t1693 + t1694;
        float t1696 = (t1692 * 0.5);
        float t1697 = metal::floor(t1696);
        float t1698 = (float)t1647;
        float t1699 = t1698 < t1695;
        int t1700 = (int)t1695;
        int t1701 = t1616 + t1647;
        float t1702 = memory[50441716 + t1701];
        int t1703 = t1616 + t1647;
        int t1704 = t1703 + 1024;
        float t1705 = memory[50441716 + t1704];
        int t1706 = t1616 + t1700;
        float t1707 = memory[50441716 + t1706];
        int t1708 = t1616 + t1700;
        int t1709 = t1708 + 1024;
        float t1710 = memory[50441716 + t1709];
        float t1711 = metal::select(t1702, t1707, t1699 > 0.0);
        float t1712 = metal::select(t1705, t1710, t1699 > 0.0);
        float t1713 = metal::select(t1707, t1702, t1699 > 0.0);
        float t1714 = metal::select(t1710, t1705, t1699 > 0.0);
        int t1715 = t1616 + t1647;
        memory[50441716 + t1715] = t1711;
        int t1717 = t1616 + t1647;
        int t1718 = t1717 + 1024;
        memory[50441716 + t1718] = t1712;
        int t1720 = t1616 + t1700;
        memory[50441716 + t1720] = t1713;
        int t1722 = t1616 + t1700;
        int t1723 = t1722 + 1024;
        memory[50441716 + t1723] = t1714;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1726 = 0; _pr1726 < 512; _pr1726++) {
        float t1727 = (float)_pr1726;
        float t1728 = t1727;
        float t1729 = metal::floor(t1728);
        float t1730 = t1729;
        float t1731 = t1727 - t1730;
        float t1732 = t1729 * 2.0;
        float t1733 = t1732 + t1731;
        float t1734 = t1733 + 1.0;
        float t1735 = -6.283185 * t1731;
        float t1736 = (t1735 * 0.5);
        float t1737 = metal::cos(t1736);
        float t1738 = metal::sin(t1736);
        int t1739 = (int)t1733;
        int t1740 = (int)t1734;
        int t1741 = t1616 + t1739;
        float t1742 = memory[50441716 + t1741];
        int t1743 = t1616 + t1739;
        int t1744 = t1743 + 1024;
        float t1745 = memory[50441716 + t1744];
        int t1746 = t1616 + t1740;
        float t1747 = memory[50441716 + t1746];
        int t1748 = t1616 + t1740;
        int t1749 = t1748 + 1024;
        float t1750 = memory[50441716 + t1749];
        float t1751 = t1737 * t1747;
        float t1752 = t1738 * t1750;
        float t1753 = t1751 - t1752;
        float t1754 = t1737 * t1750;
        float t1755 = t1738 * t1747;
        float t1756 = t1754 + t1755;
        int t1757 = t1616 + t1739;
        float t1758 = t1742 + t1753;
        memory[50441716 + t1757] = t1758;
        int t1760 = t1616 + t1739;
        int t1761 = t1760 + 1024;
        float t1762 = t1745 + t1756;
        memory[50441716 + t1761] = t1762;
        int t1764 = t1616 + t1740;
        float t1765 = t1742 - t1753;
        memory[50441716 + t1764] = t1765;
        int t1767 = t1616 + t1740;
        int t1768 = t1767 + 1024;
        float t1769 = t1745 - t1756;
        memory[50441716 + t1768] = t1769;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1772 = 0; _pr1772 < 512; _pr1772++) {
        float t1773 = (float)_pr1772;
        float t1774 = (t1773 * 0.5);
        float t1775 = metal::floor(t1774);
        float t1776 = t1775 * 2.0;
        float t1777 = t1773 - t1776;
        float t1778 = t1775 * 4.0;
        float t1779 = t1778 + t1777;
        float t1780 = t1779 + 2.0;
        float t1781 = -6.283185 * t1777;
        float t1782 = (t1781 * 0.25);
        float t1783 = metal::cos(t1782);
        float t1784 = metal::sin(t1782);
        int t1785 = (int)t1779;
        int t1786 = (int)t1780;
        int t1787 = t1616 + t1785;
        float t1788 = memory[50441716 + t1787];
        int t1789 = t1616 + t1785;
        int t1790 = t1789 + 1024;
        float t1791 = memory[50441716 + t1790];
        int t1792 = t1616 + t1786;
        float t1793 = memory[50441716 + t1792];
        int t1794 = t1616 + t1786;
        int t1795 = t1794 + 1024;
        float t1796 = memory[50441716 + t1795];
        float t1797 = t1783 * t1793;
        float t1798 = t1784 * t1796;
        float t1799 = t1797 - t1798;
        float t1800 = t1783 * t1796;
        float t1801 = t1784 * t1793;
        float t1802 = t1800 + t1801;
        int t1803 = t1616 + t1785;
        float t1804 = t1788 + t1799;
        memory[50441716 + t1803] = t1804;
        int t1806 = t1616 + t1785;
        int t1807 = t1806 + 1024;
        float t1808 = t1791 + t1802;
        memory[50441716 + t1807] = t1808;
        int t1810 = t1616 + t1786;
        float t1811 = t1788 - t1799;
        memory[50441716 + t1810] = t1811;
        int t1813 = t1616 + t1786;
        int t1814 = t1813 + 1024;
        float t1815 = t1791 - t1802;
        memory[50441716 + t1814] = t1815;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1818 = 0; _pr1818 < 512; _pr1818++) {
        float t1819 = (float)_pr1818;
        float t1820 = (t1819 * 0.25);
        float t1821 = metal::floor(t1820);
        float t1822 = t1821 * 4.0;
        float t1823 = t1819 - t1822;
        float t1824 = t1821 * 8.0;
        float t1825 = t1824 + t1823;
        float t1826 = t1825 + 4.0;
        float t1827 = -6.283185 * t1823;
        float t1828 = (t1827 * 0.125);
        float t1829 = metal::cos(t1828);
        float t1830 = metal::sin(t1828);
        int t1831 = (int)t1825;
        int t1832 = (int)t1826;
        int t1833 = t1616 + t1831;
        float t1834 = memory[50441716 + t1833];
        int t1835 = t1616 + t1831;
        int t1836 = t1835 + 1024;
        float t1837 = memory[50441716 + t1836];
        int t1838 = t1616 + t1832;
        float t1839 = memory[50441716 + t1838];
        int t1840 = t1616 + t1832;
        int t1841 = t1840 + 1024;
        float t1842 = memory[50441716 + t1841];
        float t1843 = t1829 * t1839;
        float t1844 = t1830 * t1842;
        float t1845 = t1843 - t1844;
        float t1846 = t1829 * t1842;
        float t1847 = t1830 * t1839;
        float t1848 = t1846 + t1847;
        int t1849 = t1616 + t1831;
        float t1850 = t1834 + t1845;
        memory[50441716 + t1849] = t1850;
        int t1852 = t1616 + t1831;
        int t1853 = t1852 + 1024;
        float t1854 = t1837 + t1848;
        memory[50441716 + t1853] = t1854;
        int t1856 = t1616 + t1832;
        float t1857 = t1834 - t1845;
        memory[50441716 + t1856] = t1857;
        int t1859 = t1616 + t1832;
        int t1860 = t1859 + 1024;
        float t1861 = t1837 - t1848;
        memory[50441716 + t1860] = t1861;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1864 = 0; _pr1864 < 512; _pr1864++) {
        float t1865 = (float)_pr1864;
        float t1866 = (t1865 * 0.125);
        float t1867 = metal::floor(t1866);
        float t1868 = t1867 * 8.0;
        float t1869 = t1865 - t1868;
        float t1870 = t1867 * 16.0;
        float t1871 = t1870 + t1869;
        float t1872 = t1871 + 8.0;
        float t1873 = -6.283185 * t1869;
        float t1874 = (t1873 * 0.0625);
        float t1875 = metal::cos(t1874);
        float t1876 = metal::sin(t1874);
        int t1877 = (int)t1871;
        int t1878 = (int)t1872;
        int t1879 = t1616 + t1877;
        float t1880 = memory[50441716 + t1879];
        int t1881 = t1616 + t1877;
        int t1882 = t1881 + 1024;
        float t1883 = memory[50441716 + t1882];
        int t1884 = t1616 + t1878;
        float t1885 = memory[50441716 + t1884];
        int t1886 = t1616 + t1878;
        int t1887 = t1886 + 1024;
        float t1888 = memory[50441716 + t1887];
        float t1889 = t1875 * t1885;
        float t1890 = t1876 * t1888;
        float t1891 = t1889 - t1890;
        float t1892 = t1875 * t1888;
        float t1893 = t1876 * t1885;
        float t1894 = t1892 + t1893;
        int t1895 = t1616 + t1877;
        float t1896 = t1880 + t1891;
        memory[50441716 + t1895] = t1896;
        int t1898 = t1616 + t1877;
        int t1899 = t1898 + 1024;
        float t1900 = t1883 + t1894;
        memory[50441716 + t1899] = t1900;
        int t1902 = t1616 + t1878;
        float t1903 = t1880 - t1891;
        memory[50441716 + t1902] = t1903;
        int t1905 = t1616 + t1878;
        int t1906 = t1905 + 1024;
        float t1907 = t1883 - t1894;
        memory[50441716 + t1906] = t1907;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1910 = 0; _pr1910 < 512; _pr1910++) {
        float t1911 = (float)_pr1910;
        float t1912 = (t1911 * 0.0625);
        float t1913 = metal::floor(t1912);
        float t1914 = t1913 * 16.0;
        float t1915 = t1911 - t1914;
        float t1916 = t1913 * 32.0;
        float t1917 = t1916 + t1915;
        float t1918 = t1917 + 16.0;
        float t1919 = -6.283185 * t1915;
        float t1920 = (t1919 * 0.03125);
        float t1921 = metal::cos(t1920);
        float t1922 = metal::sin(t1920);
        int t1923 = (int)t1917;
        int t1924 = (int)t1918;
        int t1925 = t1616 + t1923;
        float t1926 = memory[50441716 + t1925];
        int t1927 = t1616 + t1923;
        int t1928 = t1927 + 1024;
        float t1929 = memory[50441716 + t1928];
        int t1930 = t1616 + t1924;
        float t1931 = memory[50441716 + t1930];
        int t1932 = t1616 + t1924;
        int t1933 = t1932 + 1024;
        float t1934 = memory[50441716 + t1933];
        float t1935 = t1921 * t1931;
        float t1936 = t1922 * t1934;
        float t1937 = t1935 - t1936;
        float t1938 = t1921 * t1934;
        float t1939 = t1922 * t1931;
        float t1940 = t1938 + t1939;
        int t1941 = t1616 + t1923;
        float t1942 = t1926 + t1937;
        memory[50441716 + t1941] = t1942;
        int t1944 = t1616 + t1923;
        int t1945 = t1944 + 1024;
        float t1946 = t1929 + t1940;
        memory[50441716 + t1945] = t1946;
        int t1948 = t1616 + t1924;
        float t1949 = t1926 - t1937;
        memory[50441716 + t1948] = t1949;
        int t1951 = t1616 + t1924;
        int t1952 = t1951 + 1024;
        float t1953 = t1929 - t1940;
        memory[50441716 + t1952] = t1953;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1956 = 0; _pr1956 < 512; _pr1956++) {
        float t1957 = (float)_pr1956;
        float t1958 = (t1957 * 0.03125);
        float t1959 = metal::floor(t1958);
        float t1960 = t1959 * 32.0;
        float t1961 = t1957 - t1960;
        float t1962 = t1959 * 64.0;
        float t1963 = t1962 + t1961;
        float t1964 = t1963 + 32.0;
        float t1965 = -6.283185 * t1961;
        float t1966 = (t1965 * 0.015625);
        float t1967 = metal::cos(t1966);
        float t1968 = metal::sin(t1966);
        int t1969 = (int)t1963;
        int t1970 = (int)t1964;
        int t1971 = t1616 + t1969;
        float t1972 = memory[50441716 + t1971];
        int t1973 = t1616 + t1969;
        int t1974 = t1973 + 1024;
        float t1975 = memory[50441716 + t1974];
        int t1976 = t1616 + t1970;
        float t1977 = memory[50441716 + t1976];
        int t1978 = t1616 + t1970;
        int t1979 = t1978 + 1024;
        float t1980 = memory[50441716 + t1979];
        float t1981 = t1967 * t1977;
        float t1982 = t1968 * t1980;
        float t1983 = t1981 - t1982;
        float t1984 = t1967 * t1980;
        float t1985 = t1968 * t1977;
        float t1986 = t1984 + t1985;
        int t1987 = t1616 + t1969;
        float t1988 = t1972 + t1983;
        memory[50441716 + t1987] = t1988;
        int t1990 = t1616 + t1969;
        int t1991 = t1990 + 1024;
        float t1992 = t1975 + t1986;
        memory[50441716 + t1991] = t1992;
        int t1994 = t1616 + t1970;
        float t1995 = t1972 - t1983;
        memory[50441716 + t1994] = t1995;
        int t1997 = t1616 + t1970;
        int t1998 = t1997 + 1024;
        float t1999 = t1975 - t1986;
        memory[50441716 + t1998] = t1999;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2002 = 0; _pr2002 < 512; _pr2002++) {
        float t2003 = (float)_pr2002;
        float t2004 = (t2003 * 0.015625);
        float t2005 = metal::floor(t2004);
        float t2006 = t2005 * 64.0;
        float t2007 = t2003 - t2006;
        float t2008 = t2005 * 128.0;
        float t2009 = t2008 + t2007;
        float t2010 = t2009 + 64.0;
        float t2011 = -6.283185 * t2007;
        float t2012 = (t2011 * 0.0078125);
        float t2013 = metal::cos(t2012);
        float t2014 = metal::sin(t2012);
        int t2015 = (int)t2009;
        int t2016 = (int)t2010;
        int t2017 = t1616 + t2015;
        float t2018 = memory[50441716 + t2017];
        int t2019 = t1616 + t2015;
        int t2020 = t2019 + 1024;
        float t2021 = memory[50441716 + t2020];
        int t2022 = t1616 + t2016;
        float t2023 = memory[50441716 + t2022];
        int t2024 = t1616 + t2016;
        int t2025 = t2024 + 1024;
        float t2026 = memory[50441716 + t2025];
        float t2027 = t2013 * t2023;
        float t2028 = t2014 * t2026;
        float t2029 = t2027 - t2028;
        float t2030 = t2013 * t2026;
        float t2031 = t2014 * t2023;
        float t2032 = t2030 + t2031;
        int t2033 = t1616 + t2015;
        float t2034 = t2018 + t2029;
        memory[50441716 + t2033] = t2034;
        int t2036 = t1616 + t2015;
        int t2037 = t2036 + 1024;
        float t2038 = t2021 + t2032;
        memory[50441716 + t2037] = t2038;
        int t2040 = t1616 + t2016;
        float t2041 = t2018 - t2029;
        memory[50441716 + t2040] = t2041;
        int t2043 = t1616 + t2016;
        int t2044 = t2043 + 1024;
        float t2045 = t2021 - t2032;
        memory[50441716 + t2044] = t2045;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2048 = 0; _pr2048 < 512; _pr2048++) {
        float t2049 = (float)_pr2048;
        float t2050 = (t2049 * 0.0078125);
        float t2051 = metal::floor(t2050);
        float t2052 = t2051 * 128.0;
        float t2053 = t2049 - t2052;
        float t2054 = t2051 * 256.0;
        float t2055 = t2054 + t2053;
        float t2056 = t2055 + 128.0;
        float t2057 = -6.283185 * t2053;
        float t2058 = (t2057 * 0.00390625);
        float t2059 = metal::cos(t2058);
        float t2060 = metal::sin(t2058);
        int t2061 = (int)t2055;
        int t2062 = (int)t2056;
        int t2063 = t1616 + t2061;
        float t2064 = memory[50441716 + t2063];
        int t2065 = t1616 + t2061;
        int t2066 = t2065 + 1024;
        float t2067 = memory[50441716 + t2066];
        int t2068 = t1616 + t2062;
        float t2069 = memory[50441716 + t2068];
        int t2070 = t1616 + t2062;
        int t2071 = t2070 + 1024;
        float t2072 = memory[50441716 + t2071];
        float t2073 = t2059 * t2069;
        float t2074 = t2060 * t2072;
        float t2075 = t2073 - t2074;
        float t2076 = t2059 * t2072;
        float t2077 = t2060 * t2069;
        float t2078 = t2076 + t2077;
        int t2079 = t1616 + t2061;
        float t2080 = t2064 + t2075;
        memory[50441716 + t2079] = t2080;
        int t2082 = t1616 + t2061;
        int t2083 = t2082 + 1024;
        float t2084 = t2067 + t2078;
        memory[50441716 + t2083] = t2084;
        int t2086 = t1616 + t2062;
        float t2087 = t2064 - t2075;
        memory[50441716 + t2086] = t2087;
        int t2089 = t1616 + t2062;
        int t2090 = t2089 + 1024;
        float t2091 = t2067 - t2078;
        memory[50441716 + t2090] = t2091;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2094 = 0; _pr2094 < 512; _pr2094++) {
        float t2095 = (float)_pr2094;
        float t2096 = (t2095 * 0.00390625);
        float t2097 = metal::floor(t2096);
        float t2098 = t2097 * 256.0;
        float t2099 = t2095 - t2098;
        float t2100 = t2097 * 512.0;
        float t2101 = t2100 + t2099;
        float t2102 = t2101 + 256.0;
        float t2103 = -6.283185 * t2099;
        float t2104 = (t2103 * 0.001953125);
        float t2105 = metal::cos(t2104);
        float t2106 = metal::sin(t2104);
        int t2107 = (int)t2101;
        int t2108 = (int)t2102;
        int t2109 = t1616 + t2107;
        float t2110 = memory[50441716 + t2109];
        int t2111 = t1616 + t2107;
        int t2112 = t2111 + 1024;
        float t2113 = memory[50441716 + t2112];
        int t2114 = t1616 + t2108;
        float t2115 = memory[50441716 + t2114];
        int t2116 = t1616 + t2108;
        int t2117 = t2116 + 1024;
        float t2118 = memory[50441716 + t2117];
        float t2119 = t2105 * t2115;
        float t2120 = t2106 * t2118;
        float t2121 = t2119 - t2120;
        float t2122 = t2105 * t2118;
        float t2123 = t2106 * t2115;
        float t2124 = t2122 + t2123;
        int t2125 = t1616 + t2107;
        float t2126 = t2110 + t2121;
        memory[50441716 + t2125] = t2126;
        int t2128 = t1616 + t2107;
        int t2129 = t2128 + 1024;
        float t2130 = t2113 + t2124;
        memory[50441716 + t2129] = t2130;
        int t2132 = t1616 + t2108;
        float t2133 = t2110 - t2121;
        memory[50441716 + t2132] = t2133;
        int t2135 = t1616 + t2108;
        int t2136 = t2135 + 1024;
        float t2137 = t2113 - t2124;
        memory[50441716 + t2136] = t2137;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2140 = 0; _pr2140 < 512; _pr2140++) {
        float t2141 = (float)_pr2140;
        float t2142 = (t2141 * 0.001953125);
        float t2143 = metal::floor(t2142);
        float t2144 = t2143 * 512.0;
        float t2145 = t2141 - t2144;
        float t2146 = t2143 * 1024.0;
        float t2147 = t2146 + t2145;
        float t2148 = t2147 + 512.0;
        float t2149 = -6.283185 * t2145;
        float t2150 = (t2149 * 0.0009765625);
        float t2151 = metal::cos(t2150);
        float t2152 = metal::sin(t2150);
        int t2153 = (int)t2147;
        int t2154 = (int)t2148;
        int t2155 = t1616 + t2153;
        float t2156 = memory[50441716 + t2155];
        int t2157 = t1616 + t2153;
        int t2158 = t2157 + 1024;
        float t2159 = memory[50441716 + t2158];
        int t2160 = t1616 + t2154;
        float t2161 = memory[50441716 + t2160];
        int t2162 = t1616 + t2154;
        int t2163 = t2162 + 1024;
        float t2164 = memory[50441716 + t2163];
        float t2165 = t2151 * t2161;
        float t2166 = t2152 * t2164;
        float t2167 = t2165 - t2166;
        float t2168 = t2151 * t2164;
        float t2169 = t2152 * t2161;
        float t2170 = t2168 + t2169;
        int t2171 = t1616 + t2153;
        float t2172 = t2156 + t2167;
        memory[50441716 + t2171] = t2172;
        int t2174 = t1616 + t2153;
        int t2175 = t2174 + 1024;
        float t2176 = t2159 + t2170;
        memory[50441716 + t2175] = t2176;
        int t2178 = t1616 + t2154;
        float t2179 = t2156 - t2167;
        memory[50441716 + t2178] = t2179;
        int t2181 = t1616 + t2154;
        int t2182 = t2181 + 1024;
        float t2183 = t2159 - t2170;
        memory[50441716 + t2182] = t2183;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2186 = 0; t2186 < 1024; t2186++) {
        float t2187 = (float)t2186;
        float t2188 = (t2187 - metal::floor(t2187 / 2.0) * 2.0);
        float t2189 = t2188;
        float t2190 = (t2187 * 0.5);
        float t2191 = metal::floor(t2190);
        float t2192 = t2189 * 2.0;
        float t2193 = (t2191 - metal::floor(t2191 / 2.0) * 2.0);
        float t2194 = t2192 + t2193;
        float t2195 = (t2191 * 0.5);
        float t2196 = metal::floor(t2195);
        float t2197 = t2194 * 2.0;
        float t2198 = (t2196 - metal::floor(t2196 / 2.0) * 2.0);
        float t2199 = t2197 + t2198;
        float t2200 = (t2196 * 0.5);
        float t2201 = metal::floor(t2200);
        float t2202 = t2199 * 2.0;
        float t2203 = (t2201 - metal::floor(t2201 / 2.0) * 2.0);
        float t2204 = t2202 + t2203;
        float t2205 = (t2201 * 0.5);
        float t2206 = metal::floor(t2205);
        float t2207 = t2204 * 2.0;
        float t2208 = (t2206 - metal::floor(t2206 / 2.0) * 2.0);
        float t2209 = t2207 + t2208;
        float t2210 = (t2206 * 0.5);
        float t2211 = metal::floor(t2210);
        float t2212 = t2209 * 2.0;
        float t2213 = (t2211 - metal::floor(t2211 / 2.0) * 2.0);
        float t2214 = t2212 + t2213;
        float t2215 = (t2211 * 0.5);
        float t2216 = metal::floor(t2215);
        float t2217 = t2214 * 2.0;
        float t2218 = (t2216 - metal::floor(t2216 / 2.0) * 2.0);
        float t2219 = t2217 + t2218;
        float t2220 = (t2216 * 0.5);
        float t2221 = metal::floor(t2220);
        float t2222 = t2219 * 2.0;
        float t2223 = (t2221 - metal::floor(t2221 / 2.0) * 2.0);
        float t2224 = t2222 + t2223;
        float t2225 = (t2221 * 0.5);
        float t2226 = metal::floor(t2225);
        float t2227 = t2224 * 2.0;
        float t2228 = (t2226 - metal::floor(t2226 / 2.0) * 2.0);
        float t2229 = t2227 + t2228;
        float t2230 = (t2226 * 0.5);
        float t2231 = metal::floor(t2230);
        float t2232 = t2229 * 2.0;
        float t2233 = (t2231 - metal::floor(t2231 / 2.0) * 2.0);
        float t2234 = t2232 + t2233;
        float t2235 = (t2231 * 0.5);
        float t2236 = metal::floor(t2235);
        float t2237 = (float)t2186;
        float t2238 = t2237 < t2234;
        int t2239 = (int)t2234;
        int t2240 = t1616 + t2186;
        float t2241 = memory[83996148 + t2240];
        int t2242 = t1616 + t2186;
        int t2243 = t2242 + 1024;
        float t2244 = memory[83996148 + t2243];
        int t2245 = t1616 + t2239;
        float t2246 = memory[83996148 + t2245];
        int t2247 = t1616 + t2239;
        int t2248 = t2247 + 1024;
        float t2249 = memory[83996148 + t2248];
        float t2250 = metal::select(t2241, t2246, t2238 > 0.0);
        float t2251 = metal::select(t2244, t2249, t2238 > 0.0);
        float t2252 = metal::select(t2246, t2241, t2238 > 0.0);
        float t2253 = metal::select(t2249, t2244, t2238 > 0.0);
        int t2254 = t1616 + t2186;
        memory[83996148 + t2254] = t2250;
        int t2256 = t1616 + t2186;
        int t2257 = t2256 + 1024;
        memory[83996148 + t2257] = t2251;
        int t2259 = t1616 + t2239;
        memory[83996148 + t2259] = t2252;
        int t2261 = t1616 + t2239;
        int t2262 = t2261 + 1024;
        memory[83996148 + t2262] = t2253;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2265 = 0; _pr2265 < 512; _pr2265++) {
        float t2266 = (float)_pr2265;
        float t2267 = t2266;
        float t2268 = metal::floor(t2267);
        float t2269 = t2268;
        float t2270 = t2266 - t2269;
        float t2271 = t2268 * 2.0;
        float t2272 = t2271 + t2270;
        float t2273 = t2272 + 1.0;
        float t2274 = -6.283185 * t2270;
        float t2275 = (t2274 * 0.5);
        float t2276 = metal::cos(t2275);
        float t2277 = metal::sin(t2275);
        int t2278 = (int)t2272;
        int t2279 = (int)t2273;
        int t2280 = t1616 + t2278;
        float t2281 = memory[83996148 + t2280];
        int t2282 = t1616 + t2278;
        int t2283 = t2282 + 1024;
        float t2284 = memory[83996148 + t2283];
        int t2285 = t1616 + t2279;
        float t2286 = memory[83996148 + t2285];
        int t2287 = t1616 + t2279;
        int t2288 = t2287 + 1024;
        float t2289 = memory[83996148 + t2288];
        float t2290 = t2276 * t2286;
        float t2291 = t2277 * t2289;
        float t2292 = t2290 - t2291;
        float t2293 = t2276 * t2289;
        float t2294 = t2277 * t2286;
        float t2295 = t2293 + t2294;
        int t2296 = t1616 + t2278;
        float t2297 = t2281 + t2292;
        memory[83996148 + t2296] = t2297;
        int t2299 = t1616 + t2278;
        int t2300 = t2299 + 1024;
        float t2301 = t2284 + t2295;
        memory[83996148 + t2300] = t2301;
        int t2303 = t1616 + t2279;
        float t2304 = t2281 - t2292;
        memory[83996148 + t2303] = t2304;
        int t2306 = t1616 + t2279;
        int t2307 = t2306 + 1024;
        float t2308 = t2284 - t2295;
        memory[83996148 + t2307] = t2308;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2311 = 0; _pr2311 < 512; _pr2311++) {
        float t2312 = (float)_pr2311;
        float t2313 = (t2312 * 0.5);
        float t2314 = metal::floor(t2313);
        float t2315 = t2314 * 2.0;
        float t2316 = t2312 - t2315;
        float t2317 = t2314 * 4.0;
        float t2318 = t2317 + t2316;
        float t2319 = t2318 + 2.0;
        float t2320 = -6.283185 * t2316;
        float t2321 = (t2320 * 0.25);
        float t2322 = metal::cos(t2321);
        float t2323 = metal::sin(t2321);
        int t2324 = (int)t2318;
        int t2325 = (int)t2319;
        int t2326 = t1616 + t2324;
        float t2327 = memory[83996148 + t2326];
        int t2328 = t1616 + t2324;
        int t2329 = t2328 + 1024;
        float t2330 = memory[83996148 + t2329];
        int t2331 = t1616 + t2325;
        float t2332 = memory[83996148 + t2331];
        int t2333 = t1616 + t2325;
        int t2334 = t2333 + 1024;
        float t2335 = memory[83996148 + t2334];
        float t2336 = t2322 * t2332;
        float t2337 = t2323 * t2335;
        float t2338 = t2336 - t2337;
        float t2339 = t2322 * t2335;
        float t2340 = t2323 * t2332;
        float t2341 = t2339 + t2340;
        int t2342 = t1616 + t2324;
        float t2343 = t2327 + t2338;
        memory[83996148 + t2342] = t2343;
        int t2345 = t1616 + t2324;
        int t2346 = t2345 + 1024;
        float t2347 = t2330 + t2341;
        memory[83996148 + t2346] = t2347;
        int t2349 = t1616 + t2325;
        float t2350 = t2327 - t2338;
        memory[83996148 + t2349] = t2350;
        int t2352 = t1616 + t2325;
        int t2353 = t2352 + 1024;
        float t2354 = t2330 - t2341;
        memory[83996148 + t2353] = t2354;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2357 = 0; _pr2357 < 512; _pr2357++) {
        float t2358 = (float)_pr2357;
        float t2359 = (t2358 * 0.25);
        float t2360 = metal::floor(t2359);
        float t2361 = t2360 * 4.0;
        float t2362 = t2358 - t2361;
        float t2363 = t2360 * 8.0;
        float t2364 = t2363 + t2362;
        float t2365 = t2364 + 4.0;
        float t2366 = -6.283185 * t2362;
        float t2367 = (t2366 * 0.125);
        float t2368 = metal::cos(t2367);
        float t2369 = metal::sin(t2367);
        int t2370 = (int)t2364;
        int t2371 = (int)t2365;
        int t2372 = t1616 + t2370;
        float t2373 = memory[83996148 + t2372];
        int t2374 = t1616 + t2370;
        int t2375 = t2374 + 1024;
        float t2376 = memory[83996148 + t2375];
        int t2377 = t1616 + t2371;
        float t2378 = memory[83996148 + t2377];
        int t2379 = t1616 + t2371;
        int t2380 = t2379 + 1024;
        float t2381 = memory[83996148 + t2380];
        float t2382 = t2368 * t2378;
        float t2383 = t2369 * t2381;
        float t2384 = t2382 - t2383;
        float t2385 = t2368 * t2381;
        float t2386 = t2369 * t2378;
        float t2387 = t2385 + t2386;
        int t2388 = t1616 + t2370;
        float t2389 = t2373 + t2384;
        memory[83996148 + t2388] = t2389;
        int t2391 = t1616 + t2370;
        int t2392 = t2391 + 1024;
        float t2393 = t2376 + t2387;
        memory[83996148 + t2392] = t2393;
        int t2395 = t1616 + t2371;
        float t2396 = t2373 - t2384;
        memory[83996148 + t2395] = t2396;
        int t2398 = t1616 + t2371;
        int t2399 = t2398 + 1024;
        float t2400 = t2376 - t2387;
        memory[83996148 + t2399] = t2400;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2403 = 0; _pr2403 < 512; _pr2403++) {
        float t2404 = (float)_pr2403;
        float t2405 = (t2404 * 0.125);
        float t2406 = metal::floor(t2405);
        float t2407 = t2406 * 8.0;
        float t2408 = t2404 - t2407;
        float t2409 = t2406 * 16.0;
        float t2410 = t2409 + t2408;
        float t2411 = t2410 + 8.0;
        float t2412 = -6.283185 * t2408;
        float t2413 = (t2412 * 0.0625);
        float t2414 = metal::cos(t2413);
        float t2415 = metal::sin(t2413);
        int t2416 = (int)t2410;
        int t2417 = (int)t2411;
        int t2418 = t1616 + t2416;
        float t2419 = memory[83996148 + t2418];
        int t2420 = t1616 + t2416;
        int t2421 = t2420 + 1024;
        float t2422 = memory[83996148 + t2421];
        int t2423 = t1616 + t2417;
        float t2424 = memory[83996148 + t2423];
        int t2425 = t1616 + t2417;
        int t2426 = t2425 + 1024;
        float t2427 = memory[83996148 + t2426];
        float t2428 = t2414 * t2424;
        float t2429 = t2415 * t2427;
        float t2430 = t2428 - t2429;
        float t2431 = t2414 * t2427;
        float t2432 = t2415 * t2424;
        float t2433 = t2431 + t2432;
        int t2434 = t1616 + t2416;
        float t2435 = t2419 + t2430;
        memory[83996148 + t2434] = t2435;
        int t2437 = t1616 + t2416;
        int t2438 = t2437 + 1024;
        float t2439 = t2422 + t2433;
        memory[83996148 + t2438] = t2439;
        int t2441 = t1616 + t2417;
        float t2442 = t2419 - t2430;
        memory[83996148 + t2441] = t2442;
        int t2444 = t1616 + t2417;
        int t2445 = t2444 + 1024;
        float t2446 = t2422 - t2433;
        memory[83996148 + t2445] = t2446;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2449 = 0; _pr2449 < 512; _pr2449++) {
        float t2450 = (float)_pr2449;
        float t2451 = (t2450 * 0.0625);
        float t2452 = metal::floor(t2451);
        float t2453 = t2452 * 16.0;
        float t2454 = t2450 - t2453;
        float t2455 = t2452 * 32.0;
        float t2456 = t2455 + t2454;
        float t2457 = t2456 + 16.0;
        float t2458 = -6.283185 * t2454;
        float t2459 = (t2458 * 0.03125);
        float t2460 = metal::cos(t2459);
        float t2461 = metal::sin(t2459);
        int t2462 = (int)t2456;
        int t2463 = (int)t2457;
        int t2464 = t1616 + t2462;
        float t2465 = memory[83996148 + t2464];
        int t2466 = t1616 + t2462;
        int t2467 = t2466 + 1024;
        float t2468 = memory[83996148 + t2467];
        int t2469 = t1616 + t2463;
        float t2470 = memory[83996148 + t2469];
        int t2471 = t1616 + t2463;
        int t2472 = t2471 + 1024;
        float t2473 = memory[83996148 + t2472];
        float t2474 = t2460 * t2470;
        float t2475 = t2461 * t2473;
        float t2476 = t2474 - t2475;
        float t2477 = t2460 * t2473;
        float t2478 = t2461 * t2470;
        float t2479 = t2477 + t2478;
        int t2480 = t1616 + t2462;
        float t2481 = t2465 + t2476;
        memory[83996148 + t2480] = t2481;
        int t2483 = t1616 + t2462;
        int t2484 = t2483 + 1024;
        float t2485 = t2468 + t2479;
        memory[83996148 + t2484] = t2485;
        int t2487 = t1616 + t2463;
        float t2488 = t2465 - t2476;
        memory[83996148 + t2487] = t2488;
        int t2490 = t1616 + t2463;
        int t2491 = t2490 + 1024;
        float t2492 = t2468 - t2479;
        memory[83996148 + t2491] = t2492;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2495 = 0; _pr2495 < 512; _pr2495++) {
        float t2496 = (float)_pr2495;
        float t2497 = (t2496 * 0.03125);
        float t2498 = metal::floor(t2497);
        float t2499 = t2498 * 32.0;
        float t2500 = t2496 - t2499;
        float t2501 = t2498 * 64.0;
        float t2502 = t2501 + t2500;
        float t2503 = t2502 + 32.0;
        float t2504 = -6.283185 * t2500;
        float t2505 = (t2504 * 0.015625);
        float t2506 = metal::cos(t2505);
        float t2507 = metal::sin(t2505);
        int t2508 = (int)t2502;
        int t2509 = (int)t2503;
        int t2510 = t1616 + t2508;
        float t2511 = memory[83996148 + t2510];
        int t2512 = t1616 + t2508;
        int t2513 = t2512 + 1024;
        float t2514 = memory[83996148 + t2513];
        int t2515 = t1616 + t2509;
        float t2516 = memory[83996148 + t2515];
        int t2517 = t1616 + t2509;
        int t2518 = t2517 + 1024;
        float t2519 = memory[83996148 + t2518];
        float t2520 = t2506 * t2516;
        float t2521 = t2507 * t2519;
        float t2522 = t2520 - t2521;
        float t2523 = t2506 * t2519;
        float t2524 = t2507 * t2516;
        float t2525 = t2523 + t2524;
        int t2526 = t1616 + t2508;
        float t2527 = t2511 + t2522;
        memory[83996148 + t2526] = t2527;
        int t2529 = t1616 + t2508;
        int t2530 = t2529 + 1024;
        float t2531 = t2514 + t2525;
        memory[83996148 + t2530] = t2531;
        int t2533 = t1616 + t2509;
        float t2534 = t2511 - t2522;
        memory[83996148 + t2533] = t2534;
        int t2536 = t1616 + t2509;
        int t2537 = t2536 + 1024;
        float t2538 = t2514 - t2525;
        memory[83996148 + t2537] = t2538;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2541 = 0; _pr2541 < 512; _pr2541++) {
        float t2542 = (float)_pr2541;
        float t2543 = (t2542 * 0.015625);
        float t2544 = metal::floor(t2543);
        float t2545 = t2544 * 64.0;
        float t2546 = t2542 - t2545;
        float t2547 = t2544 * 128.0;
        float t2548 = t2547 + t2546;
        float t2549 = t2548 + 64.0;
        float t2550 = -6.283185 * t2546;
        float t2551 = (t2550 * 0.0078125);
        float t2552 = metal::cos(t2551);
        float t2553 = metal::sin(t2551);
        int t2554 = (int)t2548;
        int t2555 = (int)t2549;
        int t2556 = t1616 + t2554;
        float t2557 = memory[83996148 + t2556];
        int t2558 = t1616 + t2554;
        int t2559 = t2558 + 1024;
        float t2560 = memory[83996148 + t2559];
        int t2561 = t1616 + t2555;
        float t2562 = memory[83996148 + t2561];
        int t2563 = t1616 + t2555;
        int t2564 = t2563 + 1024;
        float t2565 = memory[83996148 + t2564];
        float t2566 = t2552 * t2562;
        float t2567 = t2553 * t2565;
        float t2568 = t2566 - t2567;
        float t2569 = t2552 * t2565;
        float t2570 = t2553 * t2562;
        float t2571 = t2569 + t2570;
        int t2572 = t1616 + t2554;
        float t2573 = t2557 + t2568;
        memory[83996148 + t2572] = t2573;
        int t2575 = t1616 + t2554;
        int t2576 = t2575 + 1024;
        float t2577 = t2560 + t2571;
        memory[83996148 + t2576] = t2577;
        int t2579 = t1616 + t2555;
        float t2580 = t2557 - t2568;
        memory[83996148 + t2579] = t2580;
        int t2582 = t1616 + t2555;
        int t2583 = t2582 + 1024;
        float t2584 = t2560 - t2571;
        memory[83996148 + t2583] = t2584;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2587 = 0; _pr2587 < 512; _pr2587++) {
        float t2588 = (float)_pr2587;
        float t2589 = (t2588 * 0.0078125);
        float t2590 = metal::floor(t2589);
        float t2591 = t2590 * 128.0;
        float t2592 = t2588 - t2591;
        float t2593 = t2590 * 256.0;
        float t2594 = t2593 + t2592;
        float t2595 = t2594 + 128.0;
        float t2596 = -6.283185 * t2592;
        float t2597 = (t2596 * 0.00390625);
        float t2598 = metal::cos(t2597);
        float t2599 = metal::sin(t2597);
        int t2600 = (int)t2594;
        int t2601 = (int)t2595;
        int t2602 = t1616 + t2600;
        float t2603 = memory[83996148 + t2602];
        int t2604 = t1616 + t2600;
        int t2605 = t2604 + 1024;
        float t2606 = memory[83996148 + t2605];
        int t2607 = t1616 + t2601;
        float t2608 = memory[83996148 + t2607];
        int t2609 = t1616 + t2601;
        int t2610 = t2609 + 1024;
        float t2611 = memory[83996148 + t2610];
        float t2612 = t2598 * t2608;
        float t2613 = t2599 * t2611;
        float t2614 = t2612 - t2613;
        float t2615 = t2598 * t2611;
        float t2616 = t2599 * t2608;
        float t2617 = t2615 + t2616;
        int t2618 = t1616 + t2600;
        float t2619 = t2603 + t2614;
        memory[83996148 + t2618] = t2619;
        int t2621 = t1616 + t2600;
        int t2622 = t2621 + 1024;
        float t2623 = t2606 + t2617;
        memory[83996148 + t2622] = t2623;
        int t2625 = t1616 + t2601;
        float t2626 = t2603 - t2614;
        memory[83996148 + t2625] = t2626;
        int t2628 = t1616 + t2601;
        int t2629 = t2628 + 1024;
        float t2630 = t2606 - t2617;
        memory[83996148 + t2629] = t2630;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2633 = 0; _pr2633 < 512; _pr2633++) {
        float t2634 = (float)_pr2633;
        float t2635 = (t2634 * 0.00390625);
        float t2636 = metal::floor(t2635);
        float t2637 = t2636 * 256.0;
        float t2638 = t2634 - t2637;
        float t2639 = t2636 * 512.0;
        float t2640 = t2639 + t2638;
        float t2641 = t2640 + 256.0;
        float t2642 = -6.283185 * t2638;
        float t2643 = (t2642 * 0.001953125);
        float t2644 = metal::cos(t2643);
        float t2645 = metal::sin(t2643);
        int t2646 = (int)t2640;
        int t2647 = (int)t2641;
        int t2648 = t1616 + t2646;
        float t2649 = memory[83996148 + t2648];
        int t2650 = t1616 + t2646;
        int t2651 = t2650 + 1024;
        float t2652 = memory[83996148 + t2651];
        int t2653 = t1616 + t2647;
        float t2654 = memory[83996148 + t2653];
        int t2655 = t1616 + t2647;
        int t2656 = t2655 + 1024;
        float t2657 = memory[83996148 + t2656];
        float t2658 = t2644 * t2654;
        float t2659 = t2645 * t2657;
        float t2660 = t2658 - t2659;
        float t2661 = t2644 * t2657;
        float t2662 = t2645 * t2654;
        float t2663 = t2661 + t2662;
        int t2664 = t1616 + t2646;
        float t2665 = t2649 + t2660;
        memory[83996148 + t2664] = t2665;
        int t2667 = t1616 + t2646;
        int t2668 = t2667 + 1024;
        float t2669 = t2652 + t2663;
        memory[83996148 + t2668] = t2669;
        int t2671 = t1616 + t2647;
        float t2672 = t2649 - t2660;
        memory[83996148 + t2671] = t2672;
        int t2674 = t1616 + t2647;
        int t2675 = t2674 + 1024;
        float t2676 = t2652 - t2663;
        memory[83996148 + t2675] = t2676;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2679 = 0; _pr2679 < 512; _pr2679++) {
        float t2680 = (float)_pr2679;
        float t2681 = (t2680 * 0.001953125);
        float t2682 = metal::floor(t2681);
        float t2683 = t2682 * 512.0;
        float t2684 = t2680 - t2683;
        float t2685 = t2682 * 1024.0;
        float t2686 = t2685 + t2684;
        float t2687 = t2686 + 512.0;
        float t2688 = -6.283185 * t2684;
        float t2689 = (t2688 * 0.0009765625);
        float t2690 = metal::cos(t2689);
        float t2691 = metal::sin(t2689);
        int t2692 = (int)t2686;
        int t2693 = (int)t2687;
        int t2694 = t1616 + t2692;
        float t2695 = memory[83996148 + t2694];
        int t2696 = t1616 + t2692;
        int t2697 = t2696 + 1024;
        float t2698 = memory[83996148 + t2697];
        int t2699 = t1616 + t2693;
        float t2700 = memory[83996148 + t2699];
        int t2701 = t1616 + t2693;
        int t2702 = t2701 + 1024;
        float t2703 = memory[83996148 + t2702];
        float t2704 = t2690 * t2700;
        float t2705 = t2691 * t2703;
        float t2706 = t2704 - t2705;
        float t2707 = t2690 * t2703;
        float t2708 = t2691 * t2700;
        float t2709 = t2707 + t2708;
        int t2710 = t1616 + t2692;
        float t2711 = t2695 + t2706;
        memory[83996148 + t2710] = t2711;
        int t2713 = t1616 + t2692;
        int t2714 = t2713 + 1024;
        float t2715 = t2698 + t2709;
        memory[83996148 + t2714] = t2715;
        int t2717 = t1616 + t2693;
        float t2718 = t2695 - t2706;
        memory[83996148 + t2717] = t2718;
        int t2720 = t1616 + t2693;
        int t2721 = t2720 + 1024;
        float t2722 = t2698 - t2709;
        memory[83996148 + t2721] = t2722;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2725 = 0; _pr2725 < 513; _pr2725++) {
        int t2726 = t1616 + _pr2725;
        float t2727 = memory[50441716 + t2726];
        int t2728 = t1616 + _pr2725;
        int t2729 = t2728 + 1024;
        float t2730 = memory[50441716 + t2729];
        float t2731 = t2727 * t2727;
        float t2732 = t2730 * t2730;
        float t2733 = t2731 + t2732;
        float t2734 = metal::sqrt(t2733);
        int t2735 = t1617 + _pr2725;
        memory[117550580 + t2735] = t2734;
        int t2737 = t1616 + _pr2725;
        float t2738 = memory[83996148 + t2737];
        int t2739 = t1616 + _pr2725;
        int t2740 = t2739 + 1024;
        float t2741 = memory[83996148 + t2740];
        float t2742 = t2738 * t2738;
        float t2743 = t2741 * t2741;
        float t2744 = t2742 + t2743;
        float t2745 = metal::sqrt(t2744);
        int t2746 = t1617 + _pr2725;
        memory[125955572 + t2746] = t2745;
        float t2748 = t2734 - t2745;
        int t2749 = t1617 + _pr2725;
        float t2750 = t2748 * t2748;
        memory[134360564 + t2749] = t2750;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2753 = 0; t2753 < 513; t2753++) {
        int t2754 = t1617 + t2753;
        float t2755 = memory[134360564 + t2754];
        float t2756 = t[15*frameCount + id] + t2755;
        t[15*frameCount + id] = t2756;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2767), value: global(2767)) */
  float t5749 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5749)) {
    /* loadGlobal(1619) - handled in variable access */
    /* loadGlobal(1597) - handled in variable access */
    int t2759 = id;
    int t2760 = t2759 / 61;
    uint _frameIndex = (uint)(t2760);
    int t2761 = t2760 * 61;
    int t2762 = t2759 - t2761;
    float t2763 = (t[15*frameCount + _frameIndex] * 6.1035156e-05);
    float t2764 = t[13*frameCount + _frameIndex] + t2763;
    float t2765 = t2764 * 0.5;
    float t2766 = t2765;
    t[16*frameCount + _frameIndex] = t2766;
    float t2768 = t2765;
    float t2769 = t2764;
    float t2770 = (t[15*frameCount + _frameIndex] * 3.7252903e-09);
    float t2771 = -0.5 * t2770;
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
    /* loadGlobal(1598) - handled in variable access */
    /* loadGlobal(536) - handled in variable access */
    /* loadGlobal(483) - handled in variable access */
    int t2772 = id;
    int t2773 = t2772 * 2048;
    int t2774 = t2772 * 513;
    int t2775 = t2772 * 2048;
    float t2776 = t[14*frameCount + id] == 0.0;
    if (t2776) {
      for (uint _pr2778 = 0; _pr2778 < 513; _pr2778++) {
        int t2779 = t2774 + _pr2778;
        float t2780 = memory[117550580 + t2779];
        int t2781 = t2774 + _pr2778;
        float t2782 = memory[125955572 + t2781];
        int t2783 = t2773 + _pr2778;
        float t2784 = memory[50441716 + t2783];
        int t2785 = t2773 + _pr2778;
        int t2786 = t2785 + 1024;
        float t2787 = memory[50441716 + t2786];
        int t2788 = t2773 + _pr2778;
        float t2789 = memory[83996148 + t2788];
        int t2790 = t2773 + _pr2778;
        int t2791 = t2790 + 1024;
        float t2792 = memory[83996148 + t2791];
        float t2793 = t2780 - t2782;
        float t2794 = 2.0 * t2793;
        float t2795 = t2794 * 3.0517578e-05;
        float t2796 = t2780 - t2782;
        float t2797 = -2.0 * t2796;
        float t2798 = t2797 * 3.0517578e-05;
        float t2799 = metal::max(t2780, 1e-08);
        float t2800 = metal::max(t2782, 1e-08);
        float t2801 = t2795 * t2784;
        float t2802 = t2801 / t2799;
        float t2803 = t2795 * t2787;
        float t2804 = t2803 / t2799;
        float t2805 = t2798 * t2789;
        float t2806 = t2805 / t2800;
        float t2807 = t2798 * t2792;
        float t2808 = t2807 / t2800;
        int t2809 = t2775 + _pr2778;
        memory[142765556 + t2809] = t2802;
        int t2811 = t2775 + _pr2778;
        int t2812 = t2811 + 1024;
        memory[142765556 + t2812] = t2804;
        int t2814 = t2775 + _pr2778;
        memory[176319988 + t2814] = t2806;
        int t2816 = t2775 + _pr2778;
        int t2817 = t2816 + 1024;
        memory[176319988 + t2817] = t2808;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2820 = 0; _pr2820 < 511; _pr2820++) {
        int t2821 = _pr2820 + 513;
        int t2822 = t2775 + t2821;
        memory[142765556 + t2822] = 0.0;
        int t2824 = t2775 + t2821;
        int t2825 = t2824 + 1024;
        memory[142765556 + t2825] = 0.0;
        int t2827 = t2775 + t2821;
        memory[176319988 + t2827] = 0.0;
        int t2829 = t2775 + t2821;
        int t2830 = t2829 + 1024;
        memory[176319988 + t2830] = 0.0;
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
    /* loadGlobal(1598) - handled in variable access */
    int t2834 = id;
    int t2835 = t2834 * 2048;
    int t2836 = t2834 * 1024;
    float t2837 = t[14*frameCount + id] == 0.0;
    if (t2837) {
      for (uint t2839 = 0; t2839 < 1024; t2839++) {
        float t2840 = (float)t2839;
        float t2841 = (t2840 - metal::floor(t2840 / 2.0) * 2.0);
        float t2842 = t2841;
        float t2843 = (t2840 * 0.5);
        float t2844 = metal::floor(t2843);
        float t2845 = t2842 * 2.0;
        float t2846 = (t2844 - metal::floor(t2844 / 2.0) * 2.0);
        float t2847 = t2845 + t2846;
        float t2848 = (t2844 * 0.5);
        float t2849 = metal::floor(t2848);
        float t2850 = t2847 * 2.0;
        float t2851 = (t2849 - metal::floor(t2849 / 2.0) * 2.0);
        float t2852 = t2850 + t2851;
        float t2853 = (t2849 * 0.5);
        float t2854 = metal::floor(t2853);
        float t2855 = t2852 * 2.0;
        float t2856 = (t2854 - metal::floor(t2854 / 2.0) * 2.0);
        float t2857 = t2855 + t2856;
        float t2858 = (t2854 * 0.5);
        float t2859 = metal::floor(t2858);
        float t2860 = t2857 * 2.0;
        float t2861 = (t2859 - metal::floor(t2859 / 2.0) * 2.0);
        float t2862 = t2860 + t2861;
        float t2863 = (t2859 * 0.5);
        float t2864 = metal::floor(t2863);
        float t2865 = t2862 * 2.0;
        float t2866 = (t2864 - metal::floor(t2864 / 2.0) * 2.0);
        float t2867 = t2865 + t2866;
        float t2868 = (t2864 * 0.5);
        float t2869 = metal::floor(t2868);
        float t2870 = t2867 * 2.0;
        float t2871 = (t2869 - metal::floor(t2869 / 2.0) * 2.0);
        float t2872 = t2870 + t2871;
        float t2873 = (t2869 * 0.5);
        float t2874 = metal::floor(t2873);
        float t2875 = t2872 * 2.0;
        float t2876 = (t2874 - metal::floor(t2874 / 2.0) * 2.0);
        float t2877 = t2875 + t2876;
        float t2878 = (t2874 * 0.5);
        float t2879 = metal::floor(t2878);
        float t2880 = t2877 * 2.0;
        float t2881 = (t2879 - metal::floor(t2879 / 2.0) * 2.0);
        float t2882 = t2880 + t2881;
        float t2883 = (t2879 * 0.5);
        float t2884 = metal::floor(t2883);
        float t2885 = t2882 * 2.0;
        float t2886 = (t2884 - metal::floor(t2884 / 2.0) * 2.0);
        float t2887 = t2885 + t2886;
        float t2888 = (t2884 * 0.5);
        float t2889 = metal::floor(t2888);
        float t2890 = (float)t2839;
        float t2891 = t2890 < t2887;
        int t2892 = (int)t2887;
        int t2893 = t2835 + t2839;
        float t2894 = memory[142765556 + t2893];
        int t2895 = t2835 + t2839;
        int t2896 = t2895 + 1024;
        float t2897 = memory[142765556 + t2896];
        int t2898 = t2835 + t2892;
        float t2899 = memory[142765556 + t2898];
        int t2900 = t2835 + t2892;
        int t2901 = t2900 + 1024;
        float t2902 = memory[142765556 + t2901];
        float t2903 = metal::select(t2894, t2899, t2891 > 0.0);
        float t2904 = metal::select(t2897, t2902, t2891 > 0.0);
        float t2905 = metal::select(t2899, t2894, t2891 > 0.0);
        float t2906 = metal::select(t2902, t2897, t2891 > 0.0);
        int t2907 = t2835 + t2839;
        memory[142765556 + t2907] = t2903;
        int t2909 = t2835 + t2839;
        int t2910 = t2909 + 1024;
        memory[142765556 + t2910] = t2904;
        int t2912 = t2835 + t2892;
        memory[142765556 + t2912] = t2905;
        int t2914 = t2835 + t2892;
        int t2915 = t2914 + 1024;
        memory[142765556 + t2915] = t2906;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2918 = 0; _pr2918 < 512; _pr2918++) {
        float t2919 = (float)_pr2918;
        float t2920 = t2919;
        float t2921 = metal::floor(t2920);
        float t2922 = t2921;
        float t2923 = t2919 - t2922;
        float t2924 = t2921 * 2.0;
        float t2925 = t2924 + t2923;
        float t2926 = t2925 + 1.0;
        float t2927 = 6.283185 * t2923;
        float t2928 = (t2927 * 0.5);
        float t2929 = metal::cos(t2928);
        float t2930 = metal::sin(t2928);
        int t2931 = (int)t2925;
        int t2932 = (int)t2926;
        int t2933 = t2835 + t2931;
        float t2934 = memory[142765556 + t2933];
        int t2935 = t2835 + t2931;
        int t2936 = t2935 + 1024;
        float t2937 = memory[142765556 + t2936];
        int t2938 = t2835 + t2932;
        float t2939 = memory[142765556 + t2938];
        int t2940 = t2835 + t2932;
        int t2941 = t2940 + 1024;
        float t2942 = memory[142765556 + t2941];
        float t2943 = t2929 * t2939;
        float t2944 = t2930 * t2942;
        float t2945 = t2943 - t2944;
        float t2946 = t2929 * t2942;
        float t2947 = t2930 * t2939;
        float t2948 = t2946 + t2947;
        int t2949 = t2835 + t2931;
        float t2950 = t2934 + t2945;
        memory[142765556 + t2949] = t2950;
        int t2952 = t2835 + t2931;
        int t2953 = t2952 + 1024;
        float t2954 = t2937 + t2948;
        memory[142765556 + t2953] = t2954;
        int t2956 = t2835 + t2932;
        float t2957 = t2934 - t2945;
        memory[142765556 + t2956] = t2957;
        int t2959 = t2835 + t2932;
        int t2960 = t2959 + 1024;
        float t2961 = t2937 - t2948;
        memory[142765556 + t2960] = t2961;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2964 = 0; _pr2964 < 512; _pr2964++) {
        float t2965 = (float)_pr2964;
        float t2966 = (t2965 * 0.5);
        float t2967 = metal::floor(t2966);
        float t2968 = t2967 * 2.0;
        float t2969 = t2965 - t2968;
        float t2970 = t2967 * 4.0;
        float t2971 = t2970 + t2969;
        float t2972 = t2971 + 2.0;
        float t2973 = 6.283185 * t2969;
        float t2974 = (t2973 * 0.25);
        float t2975 = metal::cos(t2974);
        float t2976 = metal::sin(t2974);
        int t2977 = (int)t2971;
        int t2978 = (int)t2972;
        int t2979 = t2835 + t2977;
        float t2980 = memory[142765556 + t2979];
        int t2981 = t2835 + t2977;
        int t2982 = t2981 + 1024;
        float t2983 = memory[142765556 + t2982];
        int t2984 = t2835 + t2978;
        float t2985 = memory[142765556 + t2984];
        int t2986 = t2835 + t2978;
        int t2987 = t2986 + 1024;
        float t2988 = memory[142765556 + t2987];
        float t2989 = t2975 * t2985;
        float t2990 = t2976 * t2988;
        float t2991 = t2989 - t2990;
        float t2992 = t2975 * t2988;
        float t2993 = t2976 * t2985;
        float t2994 = t2992 + t2993;
        int t2995 = t2835 + t2977;
        float t2996 = t2980 + t2991;
        memory[142765556 + t2995] = t2996;
        int t2998 = t2835 + t2977;
        int t2999 = t2998 + 1024;
        float t3000 = t2983 + t2994;
        memory[142765556 + t2999] = t3000;
        int t3002 = t2835 + t2978;
        float t3003 = t2980 - t2991;
        memory[142765556 + t3002] = t3003;
        int t3005 = t2835 + t2978;
        int t3006 = t3005 + 1024;
        float t3007 = t2983 - t2994;
        memory[142765556 + t3006] = t3007;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3010 = 0; _pr3010 < 512; _pr3010++) {
        float t3011 = (float)_pr3010;
        float t3012 = (t3011 * 0.25);
        float t3013 = metal::floor(t3012);
        float t3014 = t3013 * 4.0;
        float t3015 = t3011 - t3014;
        float t3016 = t3013 * 8.0;
        float t3017 = t3016 + t3015;
        float t3018 = t3017 + 4.0;
        float t3019 = 6.283185 * t3015;
        float t3020 = (t3019 * 0.125);
        float t3021 = metal::cos(t3020);
        float t3022 = metal::sin(t3020);
        int t3023 = (int)t3017;
        int t3024 = (int)t3018;
        int t3025 = t2835 + t3023;
        float t3026 = memory[142765556 + t3025];
        int t3027 = t2835 + t3023;
        int t3028 = t3027 + 1024;
        float t3029 = memory[142765556 + t3028];
        int t3030 = t2835 + t3024;
        float t3031 = memory[142765556 + t3030];
        int t3032 = t2835 + t3024;
        int t3033 = t3032 + 1024;
        float t3034 = memory[142765556 + t3033];
        float t3035 = t3021 * t3031;
        float t3036 = t3022 * t3034;
        float t3037 = t3035 - t3036;
        float t3038 = t3021 * t3034;
        float t3039 = t3022 * t3031;
        float t3040 = t3038 + t3039;
        int t3041 = t2835 + t3023;
        float t3042 = t3026 + t3037;
        memory[142765556 + t3041] = t3042;
        int t3044 = t2835 + t3023;
        int t3045 = t3044 + 1024;
        float t3046 = t3029 + t3040;
        memory[142765556 + t3045] = t3046;
        int t3048 = t2835 + t3024;
        float t3049 = t3026 - t3037;
        memory[142765556 + t3048] = t3049;
        int t3051 = t2835 + t3024;
        int t3052 = t3051 + 1024;
        float t3053 = t3029 - t3040;
        memory[142765556 + t3052] = t3053;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3056 = 0; _pr3056 < 512; _pr3056++) {
        float t3057 = (float)_pr3056;
        float t3058 = (t3057 * 0.125);
        float t3059 = metal::floor(t3058);
        float t3060 = t3059 * 8.0;
        float t3061 = t3057 - t3060;
        float t3062 = t3059 * 16.0;
        float t3063 = t3062 + t3061;
        float t3064 = t3063 + 8.0;
        float t3065 = 6.283185 * t3061;
        float t3066 = (t3065 * 0.0625);
        float t3067 = metal::cos(t3066);
        float t3068 = metal::sin(t3066);
        int t3069 = (int)t3063;
        int t3070 = (int)t3064;
        int t3071 = t2835 + t3069;
        float t3072 = memory[142765556 + t3071];
        int t3073 = t2835 + t3069;
        int t3074 = t3073 + 1024;
        float t3075 = memory[142765556 + t3074];
        int t3076 = t2835 + t3070;
        float t3077 = memory[142765556 + t3076];
        int t3078 = t2835 + t3070;
        int t3079 = t3078 + 1024;
        float t3080 = memory[142765556 + t3079];
        float t3081 = t3067 * t3077;
        float t3082 = t3068 * t3080;
        float t3083 = t3081 - t3082;
        float t3084 = t3067 * t3080;
        float t3085 = t3068 * t3077;
        float t3086 = t3084 + t3085;
        int t3087 = t2835 + t3069;
        float t3088 = t3072 + t3083;
        memory[142765556 + t3087] = t3088;
        int t3090 = t2835 + t3069;
        int t3091 = t3090 + 1024;
        float t3092 = t3075 + t3086;
        memory[142765556 + t3091] = t3092;
        int t3094 = t2835 + t3070;
        float t3095 = t3072 - t3083;
        memory[142765556 + t3094] = t3095;
        int t3097 = t2835 + t3070;
        int t3098 = t3097 + 1024;
        float t3099 = t3075 - t3086;
        memory[142765556 + t3098] = t3099;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3102 = 0; _pr3102 < 512; _pr3102++) {
        float t3103 = (float)_pr3102;
        float t3104 = (t3103 * 0.0625);
        float t3105 = metal::floor(t3104);
        float t3106 = t3105 * 16.0;
        float t3107 = t3103 - t3106;
        float t3108 = t3105 * 32.0;
        float t3109 = t3108 + t3107;
        float t3110 = t3109 + 16.0;
        float t3111 = 6.283185 * t3107;
        float t3112 = (t3111 * 0.03125);
        float t3113 = metal::cos(t3112);
        float t3114 = metal::sin(t3112);
        int t3115 = (int)t3109;
        int t3116 = (int)t3110;
        int t3117 = t2835 + t3115;
        float t3118 = memory[142765556 + t3117];
        int t3119 = t2835 + t3115;
        int t3120 = t3119 + 1024;
        float t3121 = memory[142765556 + t3120];
        int t3122 = t2835 + t3116;
        float t3123 = memory[142765556 + t3122];
        int t3124 = t2835 + t3116;
        int t3125 = t3124 + 1024;
        float t3126 = memory[142765556 + t3125];
        float t3127 = t3113 * t3123;
        float t3128 = t3114 * t3126;
        float t3129 = t3127 - t3128;
        float t3130 = t3113 * t3126;
        float t3131 = t3114 * t3123;
        float t3132 = t3130 + t3131;
        int t3133 = t2835 + t3115;
        float t3134 = t3118 + t3129;
        memory[142765556 + t3133] = t3134;
        int t3136 = t2835 + t3115;
        int t3137 = t3136 + 1024;
        float t3138 = t3121 + t3132;
        memory[142765556 + t3137] = t3138;
        int t3140 = t2835 + t3116;
        float t3141 = t3118 - t3129;
        memory[142765556 + t3140] = t3141;
        int t3143 = t2835 + t3116;
        int t3144 = t3143 + 1024;
        float t3145 = t3121 - t3132;
        memory[142765556 + t3144] = t3145;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3148 = 0; _pr3148 < 512; _pr3148++) {
        float t3149 = (float)_pr3148;
        float t3150 = (t3149 * 0.03125);
        float t3151 = metal::floor(t3150);
        float t3152 = t3151 * 32.0;
        float t3153 = t3149 - t3152;
        float t3154 = t3151 * 64.0;
        float t3155 = t3154 + t3153;
        float t3156 = t3155 + 32.0;
        float t3157 = 6.283185 * t3153;
        float t3158 = (t3157 * 0.015625);
        float t3159 = metal::cos(t3158);
        float t3160 = metal::sin(t3158);
        int t3161 = (int)t3155;
        int t3162 = (int)t3156;
        int t3163 = t2835 + t3161;
        float t3164 = memory[142765556 + t3163];
        int t3165 = t2835 + t3161;
        int t3166 = t3165 + 1024;
        float t3167 = memory[142765556 + t3166];
        int t3168 = t2835 + t3162;
        float t3169 = memory[142765556 + t3168];
        int t3170 = t2835 + t3162;
        int t3171 = t3170 + 1024;
        float t3172 = memory[142765556 + t3171];
        float t3173 = t3159 * t3169;
        float t3174 = t3160 * t3172;
        float t3175 = t3173 - t3174;
        float t3176 = t3159 * t3172;
        float t3177 = t3160 * t3169;
        float t3178 = t3176 + t3177;
        int t3179 = t2835 + t3161;
        float t3180 = t3164 + t3175;
        memory[142765556 + t3179] = t3180;
        int t3182 = t2835 + t3161;
        int t3183 = t3182 + 1024;
        float t3184 = t3167 + t3178;
        memory[142765556 + t3183] = t3184;
        int t3186 = t2835 + t3162;
        float t3187 = t3164 - t3175;
        memory[142765556 + t3186] = t3187;
        int t3189 = t2835 + t3162;
        int t3190 = t3189 + 1024;
        float t3191 = t3167 - t3178;
        memory[142765556 + t3190] = t3191;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3194 = 0; _pr3194 < 512; _pr3194++) {
        float t3195 = (float)_pr3194;
        float t3196 = (t3195 * 0.015625);
        float t3197 = metal::floor(t3196);
        float t3198 = t3197 * 64.0;
        float t3199 = t3195 - t3198;
        float t3200 = t3197 * 128.0;
        float t3201 = t3200 + t3199;
        float t3202 = t3201 + 64.0;
        float t3203 = 6.283185 * t3199;
        float t3204 = (t3203 * 0.0078125);
        float t3205 = metal::cos(t3204);
        float t3206 = metal::sin(t3204);
        int t3207 = (int)t3201;
        int t3208 = (int)t3202;
        int t3209 = t2835 + t3207;
        float t3210 = memory[142765556 + t3209];
        int t3211 = t2835 + t3207;
        int t3212 = t3211 + 1024;
        float t3213 = memory[142765556 + t3212];
        int t3214 = t2835 + t3208;
        float t3215 = memory[142765556 + t3214];
        int t3216 = t2835 + t3208;
        int t3217 = t3216 + 1024;
        float t3218 = memory[142765556 + t3217];
        float t3219 = t3205 * t3215;
        float t3220 = t3206 * t3218;
        float t3221 = t3219 - t3220;
        float t3222 = t3205 * t3218;
        float t3223 = t3206 * t3215;
        float t3224 = t3222 + t3223;
        int t3225 = t2835 + t3207;
        float t3226 = t3210 + t3221;
        memory[142765556 + t3225] = t3226;
        int t3228 = t2835 + t3207;
        int t3229 = t3228 + 1024;
        float t3230 = t3213 + t3224;
        memory[142765556 + t3229] = t3230;
        int t3232 = t2835 + t3208;
        float t3233 = t3210 - t3221;
        memory[142765556 + t3232] = t3233;
        int t3235 = t2835 + t3208;
        int t3236 = t3235 + 1024;
        float t3237 = t3213 - t3224;
        memory[142765556 + t3236] = t3237;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3240 = 0; _pr3240 < 512; _pr3240++) {
        float t3241 = (float)_pr3240;
        float t3242 = (t3241 * 0.0078125);
        float t3243 = metal::floor(t3242);
        float t3244 = t3243 * 128.0;
        float t3245 = t3241 - t3244;
        float t3246 = t3243 * 256.0;
        float t3247 = t3246 + t3245;
        float t3248 = t3247 + 128.0;
        float t3249 = 6.283185 * t3245;
        float t3250 = (t3249 * 0.00390625);
        float t3251 = metal::cos(t3250);
        float t3252 = metal::sin(t3250);
        int t3253 = (int)t3247;
        int t3254 = (int)t3248;
        int t3255 = t2835 + t3253;
        float t3256 = memory[142765556 + t3255];
        int t3257 = t2835 + t3253;
        int t3258 = t3257 + 1024;
        float t3259 = memory[142765556 + t3258];
        int t3260 = t2835 + t3254;
        float t3261 = memory[142765556 + t3260];
        int t3262 = t2835 + t3254;
        int t3263 = t3262 + 1024;
        float t3264 = memory[142765556 + t3263];
        float t3265 = t3251 * t3261;
        float t3266 = t3252 * t3264;
        float t3267 = t3265 - t3266;
        float t3268 = t3251 * t3264;
        float t3269 = t3252 * t3261;
        float t3270 = t3268 + t3269;
        int t3271 = t2835 + t3253;
        float t3272 = t3256 + t3267;
        memory[142765556 + t3271] = t3272;
        int t3274 = t2835 + t3253;
        int t3275 = t3274 + 1024;
        float t3276 = t3259 + t3270;
        memory[142765556 + t3275] = t3276;
        int t3278 = t2835 + t3254;
        float t3279 = t3256 - t3267;
        memory[142765556 + t3278] = t3279;
        int t3281 = t2835 + t3254;
        int t3282 = t3281 + 1024;
        float t3283 = t3259 - t3270;
        memory[142765556 + t3282] = t3283;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3286 = 0; _pr3286 < 512; _pr3286++) {
        float t3287 = (float)_pr3286;
        float t3288 = (t3287 * 0.00390625);
        float t3289 = metal::floor(t3288);
        float t3290 = t3289 * 256.0;
        float t3291 = t3287 - t3290;
        float t3292 = t3289 * 512.0;
        float t3293 = t3292 + t3291;
        float t3294 = t3293 + 256.0;
        float t3295 = 6.283185 * t3291;
        float t3296 = (t3295 * 0.001953125);
        float t3297 = metal::cos(t3296);
        float t3298 = metal::sin(t3296);
        int t3299 = (int)t3293;
        int t3300 = (int)t3294;
        int t3301 = t2835 + t3299;
        float t3302 = memory[142765556 + t3301];
        int t3303 = t2835 + t3299;
        int t3304 = t3303 + 1024;
        float t3305 = memory[142765556 + t3304];
        int t3306 = t2835 + t3300;
        float t3307 = memory[142765556 + t3306];
        int t3308 = t2835 + t3300;
        int t3309 = t3308 + 1024;
        float t3310 = memory[142765556 + t3309];
        float t3311 = t3297 * t3307;
        float t3312 = t3298 * t3310;
        float t3313 = t3311 - t3312;
        float t3314 = t3297 * t3310;
        float t3315 = t3298 * t3307;
        float t3316 = t3314 + t3315;
        int t3317 = t2835 + t3299;
        float t3318 = t3302 + t3313;
        memory[142765556 + t3317] = t3318;
        int t3320 = t2835 + t3299;
        int t3321 = t3320 + 1024;
        float t3322 = t3305 + t3316;
        memory[142765556 + t3321] = t3322;
        int t3324 = t2835 + t3300;
        float t3325 = t3302 - t3313;
        memory[142765556 + t3324] = t3325;
        int t3327 = t2835 + t3300;
        int t3328 = t3327 + 1024;
        float t3329 = t3305 - t3316;
        memory[142765556 + t3328] = t3329;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3332 = 0; _pr3332 < 512; _pr3332++) {
        float t3333 = (float)_pr3332;
        float t3334 = (t3333 * 0.001953125);
        float t3335 = metal::floor(t3334);
        float t3336 = t3335 * 512.0;
        float t3337 = t3333 - t3336;
        float t3338 = t3335 * 1024.0;
        float t3339 = t3338 + t3337;
        float t3340 = t3339 + 512.0;
        float t3341 = 6.283185 * t3337;
        float t3342 = (t3341 * 0.0009765625);
        float t3343 = metal::cos(t3342);
        float t3344 = metal::sin(t3342);
        int t3345 = (int)t3339;
        int t3346 = (int)t3340;
        int t3347 = t2835 + t3345;
        float t3348 = memory[142765556 + t3347];
        int t3349 = t2835 + t3345;
        int t3350 = t3349 + 1024;
        float t3351 = memory[142765556 + t3350];
        int t3352 = t2835 + t3346;
        float t3353 = memory[142765556 + t3352];
        int t3354 = t2835 + t3346;
        int t3355 = t3354 + 1024;
        float t3356 = memory[142765556 + t3355];
        float t3357 = t3343 * t3353;
        float t3358 = t3344 * t3356;
        float t3359 = t3357 - t3358;
        float t3360 = t3343 * t3356;
        float t3361 = t3344 * t3353;
        float t3362 = t3360 + t3361;
        int t3363 = t2835 + t3345;
        float t3364 = t3348 + t3359;
        memory[142765556 + t3363] = t3364;
        int t3366 = t2835 + t3345;
        int t3367 = t3366 + 1024;
        float t3368 = t3351 + t3362;
        memory[142765556 + t3367] = t3368;
        int t3370 = t2835 + t3346;
        float t3371 = t3348 - t3359;
        memory[142765556 + t3370] = t3371;
        int t3373 = t2835 + t3346;
        int t3374 = t3373 + 1024;
        float t3375 = t3351 - t3362;
        memory[142765556 + t3374] = t3375;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3378 = 0; _pr3378 < 1024; _pr3378++) {
        int t3379 = t2835 + _pr3378;
        float t3380 = memory[142765556 + t3379];
        float t3381 = t3380 * 1.9036306e-06;
        float t3382 = memory[52788 + (int)_pr3378];
        int t3383 = t2836 + _pr3378;
        float t3384 = t3381 * t3382;
        memory[50441716 + t3383] = t3384;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t3387 = 0; t3387 < 1024; t3387++) {
        float t3388 = (float)t3387;
        float t3389 = (t3388 - metal::floor(t3388 / 2.0) * 2.0);
        float t3390 = t3389;
        float t3391 = (t3388 * 0.5);
        float t3392 = metal::floor(t3391);
        float t3393 = t3390 * 2.0;
        float t3394 = (t3392 - metal::floor(t3392 / 2.0) * 2.0);
        float t3395 = t3393 + t3394;
        float t3396 = (t3392 * 0.5);
        float t3397 = metal::floor(t3396);
        float t3398 = t3395 * 2.0;
        float t3399 = (t3397 - metal::floor(t3397 / 2.0) * 2.0);
        float t3400 = t3398 + t3399;
        float t3401 = (t3397 * 0.5);
        float t3402 = metal::floor(t3401);
        float t3403 = t3400 * 2.0;
        float t3404 = (t3402 - metal::floor(t3402 / 2.0) * 2.0);
        float t3405 = t3403 + t3404;
        float t3406 = (t3402 * 0.5);
        float t3407 = metal::floor(t3406);
        float t3408 = t3405 * 2.0;
        float t3409 = (t3407 - metal::floor(t3407 / 2.0) * 2.0);
        float t3410 = t3408 + t3409;
        float t3411 = (t3407 * 0.5);
        float t3412 = metal::floor(t3411);
        float t3413 = t3410 * 2.0;
        float t3414 = (t3412 - metal::floor(t3412 / 2.0) * 2.0);
        float t3415 = t3413 + t3414;
        float t3416 = (t3412 * 0.5);
        float t3417 = metal::floor(t3416);
        float t3418 = t3415 * 2.0;
        float t3419 = (t3417 - metal::floor(t3417 / 2.0) * 2.0);
        float t3420 = t3418 + t3419;
        float t3421 = (t3417 * 0.5);
        float t3422 = metal::floor(t3421);
        float t3423 = t3420 * 2.0;
        float t3424 = (t3422 - metal::floor(t3422 / 2.0) * 2.0);
        float t3425 = t3423 + t3424;
        float t3426 = (t3422 * 0.5);
        float t3427 = metal::floor(t3426);
        float t3428 = t3425 * 2.0;
        float t3429 = (t3427 - metal::floor(t3427 / 2.0) * 2.0);
        float t3430 = t3428 + t3429;
        float t3431 = (t3427 * 0.5);
        float t3432 = metal::floor(t3431);
        float t3433 = t3430 * 2.0;
        float t3434 = (t3432 - metal::floor(t3432 / 2.0) * 2.0);
        float t3435 = t3433 + t3434;
        float t3436 = (t3432 * 0.5);
        float t3437 = metal::floor(t3436);
        float t3438 = (float)t3387;
        float t3439 = t3438 < t3435;
        int t3440 = (int)t3435;
        int t3441 = t2835 + t3387;
        float t3442 = memory[176319988 + t3441];
        int t3443 = t2835 + t3387;
        int t3444 = t3443 + 1024;
        float t3445 = memory[176319988 + t3444];
        int t3446 = t2835 + t3440;
        float t3447 = memory[176319988 + t3446];
        int t3448 = t2835 + t3440;
        int t3449 = t3448 + 1024;
        float t3450 = memory[176319988 + t3449];
        float t3451 = metal::select(t3442, t3447, t3439 > 0.0);
        float t3452 = metal::select(t3445, t3450, t3439 > 0.0);
        float t3453 = metal::select(t3447, t3442, t3439 > 0.0);
        float t3454 = metal::select(t3450, t3445, t3439 > 0.0);
        int t3455 = t2835 + t3387;
        memory[176319988 + t3455] = t3451;
        int t3457 = t2835 + t3387;
        int t3458 = t3457 + 1024;
        memory[176319988 + t3458] = t3452;
        int t3460 = t2835 + t3440;
        memory[176319988 + t3460] = t3453;
        int t3462 = t2835 + t3440;
        int t3463 = t3462 + 1024;
        memory[176319988 + t3463] = t3454;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3466 = 0; _pr3466 < 512; _pr3466++) {
        float t3467 = (float)_pr3466;
        float t3468 = t3467;
        float t3469 = metal::floor(t3468);
        float t3470 = t3469;
        float t3471 = t3467 - t3470;
        float t3472 = t3469 * 2.0;
        float t3473 = t3472 + t3471;
        float t3474 = t3473 + 1.0;
        float t3475 = 6.283185 * t3471;
        float t3476 = (t3475 * 0.5);
        float t3477 = metal::cos(t3476);
        float t3478 = metal::sin(t3476);
        int t3479 = (int)t3473;
        int t3480 = (int)t3474;
        int t3481 = t2835 + t3479;
        float t3482 = memory[176319988 + t3481];
        int t3483 = t2835 + t3479;
        int t3484 = t3483 + 1024;
        float t3485 = memory[176319988 + t3484];
        int t3486 = t2835 + t3480;
        float t3487 = memory[176319988 + t3486];
        int t3488 = t2835 + t3480;
        int t3489 = t3488 + 1024;
        float t3490 = memory[176319988 + t3489];
        float t3491 = t3477 * t3487;
        float t3492 = t3478 * t3490;
        float t3493 = t3491 - t3492;
        float t3494 = t3477 * t3490;
        float t3495 = t3478 * t3487;
        float t3496 = t3494 + t3495;
        int t3497 = t2835 + t3479;
        float t3498 = t3482 + t3493;
        memory[176319988 + t3497] = t3498;
        int t3500 = t2835 + t3479;
        int t3501 = t3500 + 1024;
        float t3502 = t3485 + t3496;
        memory[176319988 + t3501] = t3502;
        int t3504 = t2835 + t3480;
        float t3505 = t3482 - t3493;
        memory[176319988 + t3504] = t3505;
        int t3507 = t2835 + t3480;
        int t3508 = t3507 + 1024;
        float t3509 = t3485 - t3496;
        memory[176319988 + t3508] = t3509;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3512 = 0; _pr3512 < 512; _pr3512++) {
        float t3513 = (float)_pr3512;
        float t3514 = (t3513 * 0.5);
        float t3515 = metal::floor(t3514);
        float t3516 = t3515 * 2.0;
        float t3517 = t3513 - t3516;
        float t3518 = t3515 * 4.0;
        float t3519 = t3518 + t3517;
        float t3520 = t3519 + 2.0;
        float t3521 = 6.283185 * t3517;
        float t3522 = (t3521 * 0.25);
        float t3523 = metal::cos(t3522);
        float t3524 = metal::sin(t3522);
        int t3525 = (int)t3519;
        int t3526 = (int)t3520;
        int t3527 = t2835 + t3525;
        float t3528 = memory[176319988 + t3527];
        int t3529 = t2835 + t3525;
        int t3530 = t3529 + 1024;
        float t3531 = memory[176319988 + t3530];
        int t3532 = t2835 + t3526;
        float t3533 = memory[176319988 + t3532];
        int t3534 = t2835 + t3526;
        int t3535 = t3534 + 1024;
        float t3536 = memory[176319988 + t3535];
        float t3537 = t3523 * t3533;
        float t3538 = t3524 * t3536;
        float t3539 = t3537 - t3538;
        float t3540 = t3523 * t3536;
        float t3541 = t3524 * t3533;
        float t3542 = t3540 + t3541;
        int t3543 = t2835 + t3525;
        float t3544 = t3528 + t3539;
        memory[176319988 + t3543] = t3544;
        int t3546 = t2835 + t3525;
        int t3547 = t3546 + 1024;
        float t3548 = t3531 + t3542;
        memory[176319988 + t3547] = t3548;
        int t3550 = t2835 + t3526;
        float t3551 = t3528 - t3539;
        memory[176319988 + t3550] = t3551;
        int t3553 = t2835 + t3526;
        int t3554 = t3553 + 1024;
        float t3555 = t3531 - t3542;
        memory[176319988 + t3554] = t3555;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3558 = 0; _pr3558 < 512; _pr3558++) {
        float t3559 = (float)_pr3558;
        float t3560 = (t3559 * 0.25);
        float t3561 = metal::floor(t3560);
        float t3562 = t3561 * 4.0;
        float t3563 = t3559 - t3562;
        float t3564 = t3561 * 8.0;
        float t3565 = t3564 + t3563;
        float t3566 = t3565 + 4.0;
        float t3567 = 6.283185 * t3563;
        float t3568 = (t3567 * 0.125);
        float t3569 = metal::cos(t3568);
        float t3570 = metal::sin(t3568);
        int t3571 = (int)t3565;
        int t3572 = (int)t3566;
        int t3573 = t2835 + t3571;
        float t3574 = memory[176319988 + t3573];
        int t3575 = t2835 + t3571;
        int t3576 = t3575 + 1024;
        float t3577 = memory[176319988 + t3576];
        int t3578 = t2835 + t3572;
        float t3579 = memory[176319988 + t3578];
        int t3580 = t2835 + t3572;
        int t3581 = t3580 + 1024;
        float t3582 = memory[176319988 + t3581];
        float t3583 = t3569 * t3579;
        float t3584 = t3570 * t3582;
        float t3585 = t3583 - t3584;
        float t3586 = t3569 * t3582;
        float t3587 = t3570 * t3579;
        float t3588 = t3586 + t3587;
        int t3589 = t2835 + t3571;
        float t3590 = t3574 + t3585;
        memory[176319988 + t3589] = t3590;
        int t3592 = t2835 + t3571;
        int t3593 = t3592 + 1024;
        float t3594 = t3577 + t3588;
        memory[176319988 + t3593] = t3594;
        int t3596 = t2835 + t3572;
        float t3597 = t3574 - t3585;
        memory[176319988 + t3596] = t3597;
        int t3599 = t2835 + t3572;
        int t3600 = t3599 + 1024;
        float t3601 = t3577 - t3588;
        memory[176319988 + t3600] = t3601;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3604 = 0; _pr3604 < 512; _pr3604++) {
        float t3605 = (float)_pr3604;
        float t3606 = (t3605 * 0.125);
        float t3607 = metal::floor(t3606);
        float t3608 = t3607 * 8.0;
        float t3609 = t3605 - t3608;
        float t3610 = t3607 * 16.0;
        float t3611 = t3610 + t3609;
        float t3612 = t3611 + 8.0;
        float t3613 = 6.283185 * t3609;
        float t3614 = (t3613 * 0.0625);
        float t3615 = metal::cos(t3614);
        float t3616 = metal::sin(t3614);
        int t3617 = (int)t3611;
        int t3618 = (int)t3612;
        int t3619 = t2835 + t3617;
        float t3620 = memory[176319988 + t3619];
        int t3621 = t2835 + t3617;
        int t3622 = t3621 + 1024;
        float t3623 = memory[176319988 + t3622];
        int t3624 = t2835 + t3618;
        float t3625 = memory[176319988 + t3624];
        int t3626 = t2835 + t3618;
        int t3627 = t3626 + 1024;
        float t3628 = memory[176319988 + t3627];
        float t3629 = t3615 * t3625;
        float t3630 = t3616 * t3628;
        float t3631 = t3629 - t3630;
        float t3632 = t3615 * t3628;
        float t3633 = t3616 * t3625;
        float t3634 = t3632 + t3633;
        int t3635 = t2835 + t3617;
        float t3636 = t3620 + t3631;
        memory[176319988 + t3635] = t3636;
        int t3638 = t2835 + t3617;
        int t3639 = t3638 + 1024;
        float t3640 = t3623 + t3634;
        memory[176319988 + t3639] = t3640;
        int t3642 = t2835 + t3618;
        float t3643 = t3620 - t3631;
        memory[176319988 + t3642] = t3643;
        int t3645 = t2835 + t3618;
        int t3646 = t3645 + 1024;
        float t3647 = t3623 - t3634;
        memory[176319988 + t3646] = t3647;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3650 = 0; _pr3650 < 512; _pr3650++) {
        float t3651 = (float)_pr3650;
        float t3652 = (t3651 * 0.0625);
        float t3653 = metal::floor(t3652);
        float t3654 = t3653 * 16.0;
        float t3655 = t3651 - t3654;
        float t3656 = t3653 * 32.0;
        float t3657 = t3656 + t3655;
        float t3658 = t3657 + 16.0;
        float t3659 = 6.283185 * t3655;
        float t3660 = (t3659 * 0.03125);
        float t3661 = metal::cos(t3660);
        float t3662 = metal::sin(t3660);
        int t3663 = (int)t3657;
        int t3664 = (int)t3658;
        int t3665 = t2835 + t3663;
        float t3666 = memory[176319988 + t3665];
        int t3667 = t2835 + t3663;
        int t3668 = t3667 + 1024;
        float t3669 = memory[176319988 + t3668];
        int t3670 = t2835 + t3664;
        float t3671 = memory[176319988 + t3670];
        int t3672 = t2835 + t3664;
        int t3673 = t3672 + 1024;
        float t3674 = memory[176319988 + t3673];
        float t3675 = t3661 * t3671;
        float t3676 = t3662 * t3674;
        float t3677 = t3675 - t3676;
        float t3678 = t3661 * t3674;
        float t3679 = t3662 * t3671;
        float t3680 = t3678 + t3679;
        int t3681 = t2835 + t3663;
        float t3682 = t3666 + t3677;
        memory[176319988 + t3681] = t3682;
        int t3684 = t2835 + t3663;
        int t3685 = t3684 + 1024;
        float t3686 = t3669 + t3680;
        memory[176319988 + t3685] = t3686;
        int t3688 = t2835 + t3664;
        float t3689 = t3666 - t3677;
        memory[176319988 + t3688] = t3689;
        int t3691 = t2835 + t3664;
        int t3692 = t3691 + 1024;
        float t3693 = t3669 - t3680;
        memory[176319988 + t3692] = t3693;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3696 = 0; _pr3696 < 512; _pr3696++) {
        float t3697 = (float)_pr3696;
        float t3698 = (t3697 * 0.03125);
        float t3699 = metal::floor(t3698);
        float t3700 = t3699 * 32.0;
        float t3701 = t3697 - t3700;
        float t3702 = t3699 * 64.0;
        float t3703 = t3702 + t3701;
        float t3704 = t3703 + 32.0;
        float t3705 = 6.283185 * t3701;
        float t3706 = (t3705 * 0.015625);
        float t3707 = metal::cos(t3706);
        float t3708 = metal::sin(t3706);
        int t3709 = (int)t3703;
        int t3710 = (int)t3704;
        int t3711 = t2835 + t3709;
        float t3712 = memory[176319988 + t3711];
        int t3713 = t2835 + t3709;
        int t3714 = t3713 + 1024;
        float t3715 = memory[176319988 + t3714];
        int t3716 = t2835 + t3710;
        float t3717 = memory[176319988 + t3716];
        int t3718 = t2835 + t3710;
        int t3719 = t3718 + 1024;
        float t3720 = memory[176319988 + t3719];
        float t3721 = t3707 * t3717;
        float t3722 = t3708 * t3720;
        float t3723 = t3721 - t3722;
        float t3724 = t3707 * t3720;
        float t3725 = t3708 * t3717;
        float t3726 = t3724 + t3725;
        int t3727 = t2835 + t3709;
        float t3728 = t3712 + t3723;
        memory[176319988 + t3727] = t3728;
        int t3730 = t2835 + t3709;
        int t3731 = t3730 + 1024;
        float t3732 = t3715 + t3726;
        memory[176319988 + t3731] = t3732;
        int t3734 = t2835 + t3710;
        float t3735 = t3712 - t3723;
        memory[176319988 + t3734] = t3735;
        int t3737 = t2835 + t3710;
        int t3738 = t3737 + 1024;
        float t3739 = t3715 - t3726;
        memory[176319988 + t3738] = t3739;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3742 = 0; _pr3742 < 512; _pr3742++) {
        float t3743 = (float)_pr3742;
        float t3744 = (t3743 * 0.015625);
        float t3745 = metal::floor(t3744);
        float t3746 = t3745 * 64.0;
        float t3747 = t3743 - t3746;
        float t3748 = t3745 * 128.0;
        float t3749 = t3748 + t3747;
        float t3750 = t3749 + 64.0;
        float t3751 = 6.283185 * t3747;
        float t3752 = (t3751 * 0.0078125);
        float t3753 = metal::cos(t3752);
        float t3754 = metal::sin(t3752);
        int t3755 = (int)t3749;
        int t3756 = (int)t3750;
        int t3757 = t2835 + t3755;
        float t3758 = memory[176319988 + t3757];
        int t3759 = t2835 + t3755;
        int t3760 = t3759 + 1024;
        float t3761 = memory[176319988 + t3760];
        int t3762 = t2835 + t3756;
        float t3763 = memory[176319988 + t3762];
        int t3764 = t2835 + t3756;
        int t3765 = t3764 + 1024;
        float t3766 = memory[176319988 + t3765];
        float t3767 = t3753 * t3763;
        float t3768 = t3754 * t3766;
        float t3769 = t3767 - t3768;
        float t3770 = t3753 * t3766;
        float t3771 = t3754 * t3763;
        float t3772 = t3770 + t3771;
        int t3773 = t2835 + t3755;
        float t3774 = t3758 + t3769;
        memory[176319988 + t3773] = t3774;
        int t3776 = t2835 + t3755;
        int t3777 = t3776 + 1024;
        float t3778 = t3761 + t3772;
        memory[176319988 + t3777] = t3778;
        int t3780 = t2835 + t3756;
        float t3781 = t3758 - t3769;
        memory[176319988 + t3780] = t3781;
        int t3783 = t2835 + t3756;
        int t3784 = t3783 + 1024;
        float t3785 = t3761 - t3772;
        memory[176319988 + t3784] = t3785;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3788 = 0; _pr3788 < 512; _pr3788++) {
        float t3789 = (float)_pr3788;
        float t3790 = (t3789 * 0.0078125);
        float t3791 = metal::floor(t3790);
        float t3792 = t3791 * 128.0;
        float t3793 = t3789 - t3792;
        float t3794 = t3791 * 256.0;
        float t3795 = t3794 + t3793;
        float t3796 = t3795 + 128.0;
        float t3797 = 6.283185 * t3793;
        float t3798 = (t3797 * 0.00390625);
        float t3799 = metal::cos(t3798);
        float t3800 = metal::sin(t3798);
        int t3801 = (int)t3795;
        int t3802 = (int)t3796;
        int t3803 = t2835 + t3801;
        float t3804 = memory[176319988 + t3803];
        int t3805 = t2835 + t3801;
        int t3806 = t3805 + 1024;
        float t3807 = memory[176319988 + t3806];
        int t3808 = t2835 + t3802;
        float t3809 = memory[176319988 + t3808];
        int t3810 = t2835 + t3802;
        int t3811 = t3810 + 1024;
        float t3812 = memory[176319988 + t3811];
        float t3813 = t3799 * t3809;
        float t3814 = t3800 * t3812;
        float t3815 = t3813 - t3814;
        float t3816 = t3799 * t3812;
        float t3817 = t3800 * t3809;
        float t3818 = t3816 + t3817;
        int t3819 = t2835 + t3801;
        float t3820 = t3804 + t3815;
        memory[176319988 + t3819] = t3820;
        int t3822 = t2835 + t3801;
        int t3823 = t3822 + 1024;
        float t3824 = t3807 + t3818;
        memory[176319988 + t3823] = t3824;
        int t3826 = t2835 + t3802;
        float t3827 = t3804 - t3815;
        memory[176319988 + t3826] = t3827;
        int t3829 = t2835 + t3802;
        int t3830 = t3829 + 1024;
        float t3831 = t3807 - t3818;
        memory[176319988 + t3830] = t3831;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3834 = 0; _pr3834 < 512; _pr3834++) {
        float t3835 = (float)_pr3834;
        float t3836 = (t3835 * 0.00390625);
        float t3837 = metal::floor(t3836);
        float t3838 = t3837 * 256.0;
        float t3839 = t3835 - t3838;
        float t3840 = t3837 * 512.0;
        float t3841 = t3840 + t3839;
        float t3842 = t3841 + 256.0;
        float t3843 = 6.283185 * t3839;
        float t3844 = (t3843 * 0.001953125);
        float t3845 = metal::cos(t3844);
        float t3846 = metal::sin(t3844);
        int t3847 = (int)t3841;
        int t3848 = (int)t3842;
        int t3849 = t2835 + t3847;
        float t3850 = memory[176319988 + t3849];
        int t3851 = t2835 + t3847;
        int t3852 = t3851 + 1024;
        float t3853 = memory[176319988 + t3852];
        int t3854 = t2835 + t3848;
        float t3855 = memory[176319988 + t3854];
        int t3856 = t2835 + t3848;
        int t3857 = t3856 + 1024;
        float t3858 = memory[176319988 + t3857];
        float t3859 = t3845 * t3855;
        float t3860 = t3846 * t3858;
        float t3861 = t3859 - t3860;
        float t3862 = t3845 * t3858;
        float t3863 = t3846 * t3855;
        float t3864 = t3862 + t3863;
        int t3865 = t2835 + t3847;
        float t3866 = t3850 + t3861;
        memory[176319988 + t3865] = t3866;
        int t3868 = t2835 + t3847;
        int t3869 = t3868 + 1024;
        float t3870 = t3853 + t3864;
        memory[176319988 + t3869] = t3870;
        int t3872 = t2835 + t3848;
        float t3873 = t3850 - t3861;
        memory[176319988 + t3872] = t3873;
        int t3875 = t2835 + t3848;
        int t3876 = t3875 + 1024;
        float t3877 = t3853 - t3864;
        memory[176319988 + t3876] = t3877;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3880 = 0; _pr3880 < 512; _pr3880++) {
        float t3881 = (float)_pr3880;
        float t3882 = (t3881 * 0.001953125);
        float t3883 = metal::floor(t3882);
        float t3884 = t3883 * 512.0;
        float t3885 = t3881 - t3884;
        float t3886 = t3883 * 1024.0;
        float t3887 = t3886 + t3885;
        float t3888 = t3887 + 512.0;
        float t3889 = 6.283185 * t3885;
        float t3890 = (t3889 * 0.0009765625);
        float t3891 = metal::cos(t3890);
        float t3892 = metal::sin(t3890);
        int t3893 = (int)t3887;
        int t3894 = (int)t3888;
        int t3895 = t2835 + t3893;
        float t3896 = memory[176319988 + t3895];
        int t3897 = t2835 + t3893;
        int t3898 = t3897 + 1024;
        float t3899 = memory[176319988 + t3898];
        int t3900 = t2835 + t3894;
        float t3901 = memory[176319988 + t3900];
        int t3902 = t2835 + t3894;
        int t3903 = t3902 + 1024;
        float t3904 = memory[176319988 + t3903];
        float t3905 = t3891 * t3901;
        float t3906 = t3892 * t3904;
        float t3907 = t3905 - t3906;
        float t3908 = t3891 * t3904;
        float t3909 = t3892 * t3901;
        float t3910 = t3908 + t3909;
        int t3911 = t2835 + t3893;
        float t3912 = t3896 + t3907;
        memory[176319988 + t3911] = t3912;
        int t3914 = t2835 + t3893;
        int t3915 = t3914 + 1024;
        float t3916 = t3899 + t3910;
        memory[176319988 + t3915] = t3916;
        int t3918 = t2835 + t3894;
        float t3919 = t3896 - t3907;
        memory[176319988 + t3918] = t3919;
        int t3921 = t2835 + t3894;
        int t3922 = t3921 + 1024;
        float t3923 = t3899 - t3910;
        memory[176319988 + t3922] = t3923;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3926 = 0; _pr3926 < 1024; _pr3926++) {
        int t3927 = t2835 + _pr3926;
        float t3928 = memory[176319988 + t3927];
        float t3929 = t3928 * 1.9036306e-06;
        float t3930 = memory[52788 + (int)_pr3926];
        int t3931 = t2836 + _pr3926;
        float t3932 = t3929 * t3930;
        memory[83996148 + t3931] = t3932;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t3936 = t[14*frameCount + id] > 0.0;
    if (t3936) {
      for (uint _pr3938 = 0; _pr3938 < 1024; _pr3938++) {
        int t3939 = t2836 + _pr3938;
        memory[50441716 + t3939] = 0.0;
        int t3941 = t2836 + _pr3938;
        memory[83996148 + t3941] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3962), value: global(3962)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1598) - handled in variable access */
    int t3945 = id;
    float t3946 = 0.0;
    for (uint t3947 = 0; t3947 < 1024; t3947++) {
      float t3948 = (float)t3947;
      float t3949 = (float)t3945;
      float t3950 = t3949 + t3948;
      int t3951 = 1023 - t3947;
      float t3952 = frameCount - 1.0;
      float t3953 = metal::min(t3950, t3952);
      int t3954 = (int)t3953;
      int t3955 = t3954 * 1024;
      int t3956 = t3955 + t3951;
      float t3957 = memory[50441716 + t3956];
      float t3958 = t3950 < frameCount;
      float t3959 = metal::select(0.0, t3957, t3958 > 0.0);
      float t3960 = t3946 + t3959;
      t3946 = t3960;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[17*frameCount + id] = (t3946 * 0.0013797212);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3980), value: global(3980)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1598) - handled in variable access */
    int t3963 = id;
    float t3964 = 0.0;
    for (uint t3965 = 0; t3965 < 1024; t3965++) {
      float t3966 = (float)t3965;
      float t3967 = (float)t3963;
      float t3968 = t3967 + t3966;
      int t3969 = 1023 - t3965;
      float t3970 = frameCount - 1.0;
      float t3971 = metal::min(t3968, t3970);
      int t3972 = (int)t3971;
      int t3973 = t3972 * 1024;
      int t3974 = t3973 + t3969;
      float t3975 = memory[83996148 + t3974];
      float t3976 = t3968 < frameCount;
      float t3977 = metal::select(0.0, t3975, t3976 > 0.0);
      float t3978 = t3964 + t3977;
      t3964 = t3978;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[18*frameCount + id] = (t3964 * 0.0013797212);
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
  float t5750 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5750)) {
    /* loadGlobal(558) - handled in variable access */
    int t3981 = id;
    int t3982 = t3981 / 61;
    uint _frameIndex = (uint)(t3982);
    int t3983 = t3982 * 61;
    int t3984 = t3981 - t3983;
    float t3985 = (t[12*frameCount + _frameIndex] * 3.7252903e-09);
    float t3986 = -0.5 * t3985;
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
    /* loadGlobal(537) - handled in variable access */
    /* loadGlobal(536) - handled in variable access */
    /* loadGlobal(483) - handled in variable access */
    int t3987 = id;
    int t3988 = t3987 * 1024;
    int t3989 = t3987 * 257;
    int t3990 = t3987 * 1024;
    float t3991 = t[11*frameCount + id] == 0.0;
    if (t3991) {
      for (uint _pr3993 = 0; _pr3993 < 257; _pr3993++) {
        int t3994 = t3989 + _pr3993;
        float t3995 = memory[37809652 + t3994];
        int t3996 = t3989 + _pr3993;
        float t3997 = memory[42020340 + t3996];
        int t3998 = t3988 + _pr3993;
        float t3999 = memory[4255220 + t3998];
        int t4000 = t3988 + _pr3993;
        int t4001 = t4000 + 512;
        float t4002 = memory[4255220 + t4001];
        int t4003 = t3988 + _pr3993;
        float t4004 = memory[21032436 + t4003];
        int t4005 = t3988 + _pr3993;
        int t4006 = t4005 + 512;
        float t4007 = memory[21032436 + t4006];
        float t4008 = t3995 - t3997;
        float t4009 = 2.0 * t4008;
        float t4010 = t4009 * 3.0517578e-05;
        float t4011 = t3995 - t3997;
        float t4012 = -2.0 * t4011;
        float t4013 = t4012 * 3.0517578e-05;
        float t4014 = metal::max(t3995, 1e-08);
        float t4015 = metal::max(t3997, 1e-08);
        float t4016 = t4010 * t3999;
        float t4017 = t4016 / t4014;
        float t4018 = t4010 * t4002;
        float t4019 = t4018 / t4014;
        float t4020 = t4013 * t4004;
        float t4021 = t4020 / t4015;
        float t4022 = t4013 * t4007;
        float t4023 = t4022 / t4015;
        int t4024 = t3990 + _pr3993;
        memory[50441716 + t4024] = t4017;
        int t4026 = t3990 + _pr3993;
        int t4027 = t4026 + 512;
        memory[50441716 + t4027] = t4019;
        int t4029 = t3990 + _pr3993;
        memory[83996148 + t4029] = t4021;
        int t4031 = t3990 + _pr3993;
        int t4032 = t4031 + 512;
        memory[83996148 + t4032] = t4023;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4035 = 0; _pr4035 < 255; _pr4035++) {
        int t4036 = _pr4035 + 257;
        int t4037 = t3990 + t4036;
        memory[50441716 + t4037] = 0.0;
        int t4039 = t3990 + t4036;
        int t4040 = t4039 + 512;
        memory[50441716 + t4040] = 0.0;
        int t4042 = t3990 + t4036;
        memory[83996148 + t4042] = 0.0;
        int t4044 = t3990 + t4036;
        int t4045 = t4044 + 512;
        memory[83996148 + t4045] = 0.0;
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
    /* loadGlobal(537) - handled in variable access */
    int t4049 = id;
    int t4050 = t4049 * 1024;
    int t4051 = t4049 * 512;
    float t4052 = t[11*frameCount + id] == 0.0;
    if (t4052) {
      for (uint t4054 = 0; t4054 < 512; t4054++) {
        float t4055 = (float)t4054;
        float t4056 = (t4055 - metal::floor(t4055 / 2.0) * 2.0);
        float t4057 = t4056;
        float t4058 = (t4055 * 0.5);
        float t4059 = metal::floor(t4058);
        float t4060 = t4057 * 2.0;
        float t4061 = (t4059 - metal::floor(t4059 / 2.0) * 2.0);
        float t4062 = t4060 + t4061;
        float t4063 = (t4059 * 0.5);
        float t4064 = metal::floor(t4063);
        float t4065 = t4062 * 2.0;
        float t4066 = (t4064 - metal::floor(t4064 / 2.0) * 2.0);
        float t4067 = t4065 + t4066;
        float t4068 = (t4064 * 0.5);
        float t4069 = metal::floor(t4068);
        float t4070 = t4067 * 2.0;
        float t4071 = (t4069 - metal::floor(t4069 / 2.0) * 2.0);
        float t4072 = t4070 + t4071;
        float t4073 = (t4069 * 0.5);
        float t4074 = metal::floor(t4073);
        float t4075 = t4072 * 2.0;
        float t4076 = (t4074 - metal::floor(t4074 / 2.0) * 2.0);
        float t4077 = t4075 + t4076;
        float t4078 = (t4074 * 0.5);
        float t4079 = metal::floor(t4078);
        float t4080 = t4077 * 2.0;
        float t4081 = (t4079 - metal::floor(t4079 / 2.0) * 2.0);
        float t4082 = t4080 + t4081;
        float t4083 = (t4079 * 0.5);
        float t4084 = metal::floor(t4083);
        float t4085 = t4082 * 2.0;
        float t4086 = (t4084 - metal::floor(t4084 / 2.0) * 2.0);
        float t4087 = t4085 + t4086;
        float t4088 = (t4084 * 0.5);
        float t4089 = metal::floor(t4088);
        float t4090 = t4087 * 2.0;
        float t4091 = (t4089 - metal::floor(t4089 / 2.0) * 2.0);
        float t4092 = t4090 + t4091;
        float t4093 = (t4089 * 0.5);
        float t4094 = metal::floor(t4093);
        float t4095 = t4092 * 2.0;
        float t4096 = (t4094 - metal::floor(t4094 / 2.0) * 2.0);
        float t4097 = t4095 + t4096;
        float t4098 = (t4094 * 0.5);
        float t4099 = metal::floor(t4098);
        float t4100 = (float)t4054;
        float t4101 = t4100 < t4097;
        int t4102 = (int)t4097;
        int t4103 = t4050 + t4054;
        float t4104 = memory[50441716 + t4103];
        int t4105 = t4050 + t4054;
        int t4106 = t4105 + 512;
        float t4107 = memory[50441716 + t4106];
        int t4108 = t4050 + t4102;
        float t4109 = memory[50441716 + t4108];
        int t4110 = t4050 + t4102;
        int t4111 = t4110 + 512;
        float t4112 = memory[50441716 + t4111];
        float t4113 = metal::select(t4104, t4109, t4101 > 0.0);
        float t4114 = metal::select(t4107, t4112, t4101 > 0.0);
        float t4115 = metal::select(t4109, t4104, t4101 > 0.0);
        float t4116 = metal::select(t4112, t4107, t4101 > 0.0);
        int t4117 = t4050 + t4054;
        memory[50441716 + t4117] = t4113;
        int t4119 = t4050 + t4054;
        int t4120 = t4119 + 512;
        memory[50441716 + t4120] = t4114;
        int t4122 = t4050 + t4102;
        memory[50441716 + t4122] = t4115;
        int t4124 = t4050 + t4102;
        int t4125 = t4124 + 512;
        memory[50441716 + t4125] = t4116;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4128 = 0; _pr4128 < 256; _pr4128++) {
        float t4129 = (float)_pr4128;
        float t4130 = t4129;
        float t4131 = metal::floor(t4130);
        float t4132 = t4131;
        float t4133 = t4129 - t4132;
        float t4134 = t4131 * 2.0;
        float t4135 = t4134 + t4133;
        float t4136 = t4135 + 1.0;
        float t4137 = 6.283185 * t4133;
        float t4138 = (t4137 * 0.5);
        float t4139 = metal::cos(t4138);
        float t4140 = metal::sin(t4138);
        int t4141 = (int)t4135;
        int t4142 = (int)t4136;
        int t4143 = t4050 + t4141;
        float t4144 = memory[50441716 + t4143];
        int t4145 = t4050 + t4141;
        int t4146 = t4145 + 512;
        float t4147 = memory[50441716 + t4146];
        int t4148 = t4050 + t4142;
        float t4149 = memory[50441716 + t4148];
        int t4150 = t4050 + t4142;
        int t4151 = t4150 + 512;
        float t4152 = memory[50441716 + t4151];
        float t4153 = t4139 * t4149;
        float t4154 = t4140 * t4152;
        float t4155 = t4153 - t4154;
        float t4156 = t4139 * t4152;
        float t4157 = t4140 * t4149;
        float t4158 = t4156 + t4157;
        int t4159 = t4050 + t4141;
        float t4160 = t4144 + t4155;
        memory[50441716 + t4159] = t4160;
        int t4162 = t4050 + t4141;
        int t4163 = t4162 + 512;
        float t4164 = t4147 + t4158;
        memory[50441716 + t4163] = t4164;
        int t4166 = t4050 + t4142;
        float t4167 = t4144 - t4155;
        memory[50441716 + t4166] = t4167;
        int t4169 = t4050 + t4142;
        int t4170 = t4169 + 512;
        float t4171 = t4147 - t4158;
        memory[50441716 + t4170] = t4171;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4174 = 0; _pr4174 < 256; _pr4174++) {
        float t4175 = (float)_pr4174;
        float t4176 = (t4175 * 0.5);
        float t4177 = metal::floor(t4176);
        float t4178 = t4177 * 2.0;
        float t4179 = t4175 - t4178;
        float t4180 = t4177 * 4.0;
        float t4181 = t4180 + t4179;
        float t4182 = t4181 + 2.0;
        float t4183 = 6.283185 * t4179;
        float t4184 = (t4183 * 0.25);
        float t4185 = metal::cos(t4184);
        float t4186 = metal::sin(t4184);
        int t4187 = (int)t4181;
        int t4188 = (int)t4182;
        int t4189 = t4050 + t4187;
        float t4190 = memory[50441716 + t4189];
        int t4191 = t4050 + t4187;
        int t4192 = t4191 + 512;
        float t4193 = memory[50441716 + t4192];
        int t4194 = t4050 + t4188;
        float t4195 = memory[50441716 + t4194];
        int t4196 = t4050 + t4188;
        int t4197 = t4196 + 512;
        float t4198 = memory[50441716 + t4197];
        float t4199 = t4185 * t4195;
        float t4200 = t4186 * t4198;
        float t4201 = t4199 - t4200;
        float t4202 = t4185 * t4198;
        float t4203 = t4186 * t4195;
        float t4204 = t4202 + t4203;
        int t4205 = t4050 + t4187;
        float t4206 = t4190 + t4201;
        memory[50441716 + t4205] = t4206;
        int t4208 = t4050 + t4187;
        int t4209 = t4208 + 512;
        float t4210 = t4193 + t4204;
        memory[50441716 + t4209] = t4210;
        int t4212 = t4050 + t4188;
        float t4213 = t4190 - t4201;
        memory[50441716 + t4212] = t4213;
        int t4215 = t4050 + t4188;
        int t4216 = t4215 + 512;
        float t4217 = t4193 - t4204;
        memory[50441716 + t4216] = t4217;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4220 = 0; _pr4220 < 256; _pr4220++) {
        float t4221 = (float)_pr4220;
        float t4222 = (t4221 * 0.25);
        float t4223 = metal::floor(t4222);
        float t4224 = t4223 * 4.0;
        float t4225 = t4221 - t4224;
        float t4226 = t4223 * 8.0;
        float t4227 = t4226 + t4225;
        float t4228 = t4227 + 4.0;
        float t4229 = 6.283185 * t4225;
        float t4230 = (t4229 * 0.125);
        float t4231 = metal::cos(t4230);
        float t4232 = metal::sin(t4230);
        int t4233 = (int)t4227;
        int t4234 = (int)t4228;
        int t4235 = t4050 + t4233;
        float t4236 = memory[50441716 + t4235];
        int t4237 = t4050 + t4233;
        int t4238 = t4237 + 512;
        float t4239 = memory[50441716 + t4238];
        int t4240 = t4050 + t4234;
        float t4241 = memory[50441716 + t4240];
        int t4242 = t4050 + t4234;
        int t4243 = t4242 + 512;
        float t4244 = memory[50441716 + t4243];
        float t4245 = t4231 * t4241;
        float t4246 = t4232 * t4244;
        float t4247 = t4245 - t4246;
        float t4248 = t4231 * t4244;
        float t4249 = t4232 * t4241;
        float t4250 = t4248 + t4249;
        int t4251 = t4050 + t4233;
        float t4252 = t4236 + t4247;
        memory[50441716 + t4251] = t4252;
        int t4254 = t4050 + t4233;
        int t4255 = t4254 + 512;
        float t4256 = t4239 + t4250;
        memory[50441716 + t4255] = t4256;
        int t4258 = t4050 + t4234;
        float t4259 = t4236 - t4247;
        memory[50441716 + t4258] = t4259;
        int t4261 = t4050 + t4234;
        int t4262 = t4261 + 512;
        float t4263 = t4239 - t4250;
        memory[50441716 + t4262] = t4263;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4266 = 0; _pr4266 < 256; _pr4266++) {
        float t4267 = (float)_pr4266;
        float t4268 = (t4267 * 0.125);
        float t4269 = metal::floor(t4268);
        float t4270 = t4269 * 8.0;
        float t4271 = t4267 - t4270;
        float t4272 = t4269 * 16.0;
        float t4273 = t4272 + t4271;
        float t4274 = t4273 + 8.0;
        float t4275 = 6.283185 * t4271;
        float t4276 = (t4275 * 0.0625);
        float t4277 = metal::cos(t4276);
        float t4278 = metal::sin(t4276);
        int t4279 = (int)t4273;
        int t4280 = (int)t4274;
        int t4281 = t4050 + t4279;
        float t4282 = memory[50441716 + t4281];
        int t4283 = t4050 + t4279;
        int t4284 = t4283 + 512;
        float t4285 = memory[50441716 + t4284];
        int t4286 = t4050 + t4280;
        float t4287 = memory[50441716 + t4286];
        int t4288 = t4050 + t4280;
        int t4289 = t4288 + 512;
        float t4290 = memory[50441716 + t4289];
        float t4291 = t4277 * t4287;
        float t4292 = t4278 * t4290;
        float t4293 = t4291 - t4292;
        float t4294 = t4277 * t4290;
        float t4295 = t4278 * t4287;
        float t4296 = t4294 + t4295;
        int t4297 = t4050 + t4279;
        float t4298 = t4282 + t4293;
        memory[50441716 + t4297] = t4298;
        int t4300 = t4050 + t4279;
        int t4301 = t4300 + 512;
        float t4302 = t4285 + t4296;
        memory[50441716 + t4301] = t4302;
        int t4304 = t4050 + t4280;
        float t4305 = t4282 - t4293;
        memory[50441716 + t4304] = t4305;
        int t4307 = t4050 + t4280;
        int t4308 = t4307 + 512;
        float t4309 = t4285 - t4296;
        memory[50441716 + t4308] = t4309;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4312 = 0; _pr4312 < 256; _pr4312++) {
        float t4313 = (float)_pr4312;
        float t4314 = (t4313 * 0.0625);
        float t4315 = metal::floor(t4314);
        float t4316 = t4315 * 16.0;
        float t4317 = t4313 - t4316;
        float t4318 = t4315 * 32.0;
        float t4319 = t4318 + t4317;
        float t4320 = t4319 + 16.0;
        float t4321 = 6.283185 * t4317;
        float t4322 = (t4321 * 0.03125);
        float t4323 = metal::cos(t4322);
        float t4324 = metal::sin(t4322);
        int t4325 = (int)t4319;
        int t4326 = (int)t4320;
        int t4327 = t4050 + t4325;
        float t4328 = memory[50441716 + t4327];
        int t4329 = t4050 + t4325;
        int t4330 = t4329 + 512;
        float t4331 = memory[50441716 + t4330];
        int t4332 = t4050 + t4326;
        float t4333 = memory[50441716 + t4332];
        int t4334 = t4050 + t4326;
        int t4335 = t4334 + 512;
        float t4336 = memory[50441716 + t4335];
        float t4337 = t4323 * t4333;
        float t4338 = t4324 * t4336;
        float t4339 = t4337 - t4338;
        float t4340 = t4323 * t4336;
        float t4341 = t4324 * t4333;
        float t4342 = t4340 + t4341;
        int t4343 = t4050 + t4325;
        float t4344 = t4328 + t4339;
        memory[50441716 + t4343] = t4344;
        int t4346 = t4050 + t4325;
        int t4347 = t4346 + 512;
        float t4348 = t4331 + t4342;
        memory[50441716 + t4347] = t4348;
        int t4350 = t4050 + t4326;
        float t4351 = t4328 - t4339;
        memory[50441716 + t4350] = t4351;
        int t4353 = t4050 + t4326;
        int t4354 = t4353 + 512;
        float t4355 = t4331 - t4342;
        memory[50441716 + t4354] = t4355;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4358 = 0; _pr4358 < 256; _pr4358++) {
        float t4359 = (float)_pr4358;
        float t4360 = (t4359 * 0.03125);
        float t4361 = metal::floor(t4360);
        float t4362 = t4361 * 32.0;
        float t4363 = t4359 - t4362;
        float t4364 = t4361 * 64.0;
        float t4365 = t4364 + t4363;
        float t4366 = t4365 + 32.0;
        float t4367 = 6.283185 * t4363;
        float t4368 = (t4367 * 0.015625);
        float t4369 = metal::cos(t4368);
        float t4370 = metal::sin(t4368);
        int t4371 = (int)t4365;
        int t4372 = (int)t4366;
        int t4373 = t4050 + t4371;
        float t4374 = memory[50441716 + t4373];
        int t4375 = t4050 + t4371;
        int t4376 = t4375 + 512;
        float t4377 = memory[50441716 + t4376];
        int t4378 = t4050 + t4372;
        float t4379 = memory[50441716 + t4378];
        int t4380 = t4050 + t4372;
        int t4381 = t4380 + 512;
        float t4382 = memory[50441716 + t4381];
        float t4383 = t4369 * t4379;
        float t4384 = t4370 * t4382;
        float t4385 = t4383 - t4384;
        float t4386 = t4369 * t4382;
        float t4387 = t4370 * t4379;
        float t4388 = t4386 + t4387;
        int t4389 = t4050 + t4371;
        float t4390 = t4374 + t4385;
        memory[50441716 + t4389] = t4390;
        int t4392 = t4050 + t4371;
        int t4393 = t4392 + 512;
        float t4394 = t4377 + t4388;
        memory[50441716 + t4393] = t4394;
        int t4396 = t4050 + t4372;
        float t4397 = t4374 - t4385;
        memory[50441716 + t4396] = t4397;
        int t4399 = t4050 + t4372;
        int t4400 = t4399 + 512;
        float t4401 = t4377 - t4388;
        memory[50441716 + t4400] = t4401;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4404 = 0; _pr4404 < 256; _pr4404++) {
        float t4405 = (float)_pr4404;
        float t4406 = (t4405 * 0.015625);
        float t4407 = metal::floor(t4406);
        float t4408 = t4407 * 64.0;
        float t4409 = t4405 - t4408;
        float t4410 = t4407 * 128.0;
        float t4411 = t4410 + t4409;
        float t4412 = t4411 + 64.0;
        float t4413 = 6.283185 * t4409;
        float t4414 = (t4413 * 0.0078125);
        float t4415 = metal::cos(t4414);
        float t4416 = metal::sin(t4414);
        int t4417 = (int)t4411;
        int t4418 = (int)t4412;
        int t4419 = t4050 + t4417;
        float t4420 = memory[50441716 + t4419];
        int t4421 = t4050 + t4417;
        int t4422 = t4421 + 512;
        float t4423 = memory[50441716 + t4422];
        int t4424 = t4050 + t4418;
        float t4425 = memory[50441716 + t4424];
        int t4426 = t4050 + t4418;
        int t4427 = t4426 + 512;
        float t4428 = memory[50441716 + t4427];
        float t4429 = t4415 * t4425;
        float t4430 = t4416 * t4428;
        float t4431 = t4429 - t4430;
        float t4432 = t4415 * t4428;
        float t4433 = t4416 * t4425;
        float t4434 = t4432 + t4433;
        int t4435 = t4050 + t4417;
        float t4436 = t4420 + t4431;
        memory[50441716 + t4435] = t4436;
        int t4438 = t4050 + t4417;
        int t4439 = t4438 + 512;
        float t4440 = t4423 + t4434;
        memory[50441716 + t4439] = t4440;
        int t4442 = t4050 + t4418;
        float t4443 = t4420 - t4431;
        memory[50441716 + t4442] = t4443;
        int t4445 = t4050 + t4418;
        int t4446 = t4445 + 512;
        float t4447 = t4423 - t4434;
        memory[50441716 + t4446] = t4447;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4450 = 0; _pr4450 < 256; _pr4450++) {
        float t4451 = (float)_pr4450;
        float t4452 = (t4451 * 0.0078125);
        float t4453 = metal::floor(t4452);
        float t4454 = t4453 * 128.0;
        float t4455 = t4451 - t4454;
        float t4456 = t4453 * 256.0;
        float t4457 = t4456 + t4455;
        float t4458 = t4457 + 128.0;
        float t4459 = 6.283185 * t4455;
        float t4460 = (t4459 * 0.00390625);
        float t4461 = metal::cos(t4460);
        float t4462 = metal::sin(t4460);
        int t4463 = (int)t4457;
        int t4464 = (int)t4458;
        int t4465 = t4050 + t4463;
        float t4466 = memory[50441716 + t4465];
        int t4467 = t4050 + t4463;
        int t4468 = t4467 + 512;
        float t4469 = memory[50441716 + t4468];
        int t4470 = t4050 + t4464;
        float t4471 = memory[50441716 + t4470];
        int t4472 = t4050 + t4464;
        int t4473 = t4472 + 512;
        float t4474 = memory[50441716 + t4473];
        float t4475 = t4461 * t4471;
        float t4476 = t4462 * t4474;
        float t4477 = t4475 - t4476;
        float t4478 = t4461 * t4474;
        float t4479 = t4462 * t4471;
        float t4480 = t4478 + t4479;
        int t4481 = t4050 + t4463;
        float t4482 = t4466 + t4477;
        memory[50441716 + t4481] = t4482;
        int t4484 = t4050 + t4463;
        int t4485 = t4484 + 512;
        float t4486 = t4469 + t4480;
        memory[50441716 + t4485] = t4486;
        int t4488 = t4050 + t4464;
        float t4489 = t4466 - t4477;
        memory[50441716 + t4488] = t4489;
        int t4491 = t4050 + t4464;
        int t4492 = t4491 + 512;
        float t4493 = t4469 - t4480;
        memory[50441716 + t4492] = t4493;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4496 = 0; _pr4496 < 256; _pr4496++) {
        float t4497 = (float)_pr4496;
        float t4498 = (t4497 * 0.00390625);
        float t4499 = metal::floor(t4498);
        float t4500 = t4499 * 256.0;
        float t4501 = t4497 - t4500;
        float t4502 = t4499 * 512.0;
        float t4503 = t4502 + t4501;
        float t4504 = t4503 + 256.0;
        float t4505 = 6.283185 * t4501;
        float t4506 = (t4505 * 0.001953125);
        float t4507 = metal::cos(t4506);
        float t4508 = metal::sin(t4506);
        int t4509 = (int)t4503;
        int t4510 = (int)t4504;
        int t4511 = t4050 + t4509;
        float t4512 = memory[50441716 + t4511];
        int t4513 = t4050 + t4509;
        int t4514 = t4513 + 512;
        float t4515 = memory[50441716 + t4514];
        int t4516 = t4050 + t4510;
        float t4517 = memory[50441716 + t4516];
        int t4518 = t4050 + t4510;
        int t4519 = t4518 + 512;
        float t4520 = memory[50441716 + t4519];
        float t4521 = t4507 * t4517;
        float t4522 = t4508 * t4520;
        float t4523 = t4521 - t4522;
        float t4524 = t4507 * t4520;
        float t4525 = t4508 * t4517;
        float t4526 = t4524 + t4525;
        int t4527 = t4050 + t4509;
        float t4528 = t4512 + t4523;
        memory[50441716 + t4527] = t4528;
        int t4530 = t4050 + t4509;
        int t4531 = t4530 + 512;
        float t4532 = t4515 + t4526;
        memory[50441716 + t4531] = t4532;
        int t4534 = t4050 + t4510;
        float t4535 = t4512 - t4523;
        memory[50441716 + t4534] = t4535;
        int t4537 = t4050 + t4510;
        int t4538 = t4537 + 512;
        float t4539 = t4515 - t4526;
        memory[50441716 + t4538] = t4539;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4542 = 0; _pr4542 < 512; _pr4542++) {
        int t4543 = t4050 + _pr4542;
        float t4544 = memory[50441716 + t4543];
        float t4545 = t4544 * 7.599708e-06;
        float t4546 = memory[25460 + (int)_pr4542];
        int t4547 = t4051 + _pr4542;
        float t4548 = t4545 * t4546;
        memory[117550580 + t4547] = t4548;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t4551 = 0; t4551 < 512; t4551++) {
        float t4552 = (float)t4551;
        float t4553 = (t4552 - metal::floor(t4552 / 2.0) * 2.0);
        float t4554 = t4553;
        float t4555 = (t4552 * 0.5);
        float t4556 = metal::floor(t4555);
        float t4557 = t4554 * 2.0;
        float t4558 = (t4556 - metal::floor(t4556 / 2.0) * 2.0);
        float t4559 = t4557 + t4558;
        float t4560 = (t4556 * 0.5);
        float t4561 = metal::floor(t4560);
        float t4562 = t4559 * 2.0;
        float t4563 = (t4561 - metal::floor(t4561 / 2.0) * 2.0);
        float t4564 = t4562 + t4563;
        float t4565 = (t4561 * 0.5);
        float t4566 = metal::floor(t4565);
        float t4567 = t4564 * 2.0;
        float t4568 = (t4566 - metal::floor(t4566 / 2.0) * 2.0);
        float t4569 = t4567 + t4568;
        float t4570 = (t4566 * 0.5);
        float t4571 = metal::floor(t4570);
        float t4572 = t4569 * 2.0;
        float t4573 = (t4571 - metal::floor(t4571 / 2.0) * 2.0);
        float t4574 = t4572 + t4573;
        float t4575 = (t4571 * 0.5);
        float t4576 = metal::floor(t4575);
        float t4577 = t4574 * 2.0;
        float t4578 = (t4576 - metal::floor(t4576 / 2.0) * 2.0);
        float t4579 = t4577 + t4578;
        float t4580 = (t4576 * 0.5);
        float t4581 = metal::floor(t4580);
        float t4582 = t4579 * 2.0;
        float t4583 = (t4581 - metal::floor(t4581 / 2.0) * 2.0);
        float t4584 = t4582 + t4583;
        float t4585 = (t4581 * 0.5);
        float t4586 = metal::floor(t4585);
        float t4587 = t4584 * 2.0;
        float t4588 = (t4586 - metal::floor(t4586 / 2.0) * 2.0);
        float t4589 = t4587 + t4588;
        float t4590 = (t4586 * 0.5);
        float t4591 = metal::floor(t4590);
        float t4592 = t4589 * 2.0;
        float t4593 = (t4591 - metal::floor(t4591 / 2.0) * 2.0);
        float t4594 = t4592 + t4593;
        float t4595 = (t4591 * 0.5);
        float t4596 = metal::floor(t4595);
        float t4597 = (float)t4551;
        float t4598 = t4597 < t4594;
        int t4599 = (int)t4594;
        int t4600 = t4050 + t4551;
        float t4601 = memory[83996148 + t4600];
        int t4602 = t4050 + t4551;
        int t4603 = t4602 + 512;
        float t4604 = memory[83996148 + t4603];
        int t4605 = t4050 + t4599;
        float t4606 = memory[83996148 + t4605];
        int t4607 = t4050 + t4599;
        int t4608 = t4607 + 512;
        float t4609 = memory[83996148 + t4608];
        float t4610 = metal::select(t4601, t4606, t4598 > 0.0);
        float t4611 = metal::select(t4604, t4609, t4598 > 0.0);
        float t4612 = metal::select(t4606, t4601, t4598 > 0.0);
        float t4613 = metal::select(t4609, t4604, t4598 > 0.0);
        int t4614 = t4050 + t4551;
        memory[83996148 + t4614] = t4610;
        int t4616 = t4050 + t4551;
        int t4617 = t4616 + 512;
        memory[83996148 + t4617] = t4611;
        int t4619 = t4050 + t4599;
        memory[83996148 + t4619] = t4612;
        int t4621 = t4050 + t4599;
        int t4622 = t4621 + 512;
        memory[83996148 + t4622] = t4613;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4625 = 0; _pr4625 < 256; _pr4625++) {
        float t4626 = (float)_pr4625;
        float t4627 = t4626;
        float t4628 = metal::floor(t4627);
        float t4629 = t4628;
        float t4630 = t4626 - t4629;
        float t4631 = t4628 * 2.0;
        float t4632 = t4631 + t4630;
        float t4633 = t4632 + 1.0;
        float t4634 = 6.283185 * t4630;
        float t4635 = (t4634 * 0.5);
        float t4636 = metal::cos(t4635);
        float t4637 = metal::sin(t4635);
        int t4638 = (int)t4632;
        int t4639 = (int)t4633;
        int t4640 = t4050 + t4638;
        float t4641 = memory[83996148 + t4640];
        int t4642 = t4050 + t4638;
        int t4643 = t4642 + 512;
        float t4644 = memory[83996148 + t4643];
        int t4645 = t4050 + t4639;
        float t4646 = memory[83996148 + t4645];
        int t4647 = t4050 + t4639;
        int t4648 = t4647 + 512;
        float t4649 = memory[83996148 + t4648];
        float t4650 = t4636 * t4646;
        float t4651 = t4637 * t4649;
        float t4652 = t4650 - t4651;
        float t4653 = t4636 * t4649;
        float t4654 = t4637 * t4646;
        float t4655 = t4653 + t4654;
        int t4656 = t4050 + t4638;
        float t4657 = t4641 + t4652;
        memory[83996148 + t4656] = t4657;
        int t4659 = t4050 + t4638;
        int t4660 = t4659 + 512;
        float t4661 = t4644 + t4655;
        memory[83996148 + t4660] = t4661;
        int t4663 = t4050 + t4639;
        float t4664 = t4641 - t4652;
        memory[83996148 + t4663] = t4664;
        int t4666 = t4050 + t4639;
        int t4667 = t4666 + 512;
        float t4668 = t4644 - t4655;
        memory[83996148 + t4667] = t4668;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4671 = 0; _pr4671 < 256; _pr4671++) {
        float t4672 = (float)_pr4671;
        float t4673 = (t4672 * 0.5);
        float t4674 = metal::floor(t4673);
        float t4675 = t4674 * 2.0;
        float t4676 = t4672 - t4675;
        float t4677 = t4674 * 4.0;
        float t4678 = t4677 + t4676;
        float t4679 = t4678 + 2.0;
        float t4680 = 6.283185 * t4676;
        float t4681 = (t4680 * 0.25);
        float t4682 = metal::cos(t4681);
        float t4683 = metal::sin(t4681);
        int t4684 = (int)t4678;
        int t4685 = (int)t4679;
        int t4686 = t4050 + t4684;
        float t4687 = memory[83996148 + t4686];
        int t4688 = t4050 + t4684;
        int t4689 = t4688 + 512;
        float t4690 = memory[83996148 + t4689];
        int t4691 = t4050 + t4685;
        float t4692 = memory[83996148 + t4691];
        int t4693 = t4050 + t4685;
        int t4694 = t4693 + 512;
        float t4695 = memory[83996148 + t4694];
        float t4696 = t4682 * t4692;
        float t4697 = t4683 * t4695;
        float t4698 = t4696 - t4697;
        float t4699 = t4682 * t4695;
        float t4700 = t4683 * t4692;
        float t4701 = t4699 + t4700;
        int t4702 = t4050 + t4684;
        float t4703 = t4687 + t4698;
        memory[83996148 + t4702] = t4703;
        int t4705 = t4050 + t4684;
        int t4706 = t4705 + 512;
        float t4707 = t4690 + t4701;
        memory[83996148 + t4706] = t4707;
        int t4709 = t4050 + t4685;
        float t4710 = t4687 - t4698;
        memory[83996148 + t4709] = t4710;
        int t4712 = t4050 + t4685;
        int t4713 = t4712 + 512;
        float t4714 = t4690 - t4701;
        memory[83996148 + t4713] = t4714;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4717 = 0; _pr4717 < 256; _pr4717++) {
        float t4718 = (float)_pr4717;
        float t4719 = (t4718 * 0.25);
        float t4720 = metal::floor(t4719);
        float t4721 = t4720 * 4.0;
        float t4722 = t4718 - t4721;
        float t4723 = t4720 * 8.0;
        float t4724 = t4723 + t4722;
        float t4725 = t4724 + 4.0;
        float t4726 = 6.283185 * t4722;
        float t4727 = (t4726 * 0.125);
        float t4728 = metal::cos(t4727);
        float t4729 = metal::sin(t4727);
        int t4730 = (int)t4724;
        int t4731 = (int)t4725;
        int t4732 = t4050 + t4730;
        float t4733 = memory[83996148 + t4732];
        int t4734 = t4050 + t4730;
        int t4735 = t4734 + 512;
        float t4736 = memory[83996148 + t4735];
        int t4737 = t4050 + t4731;
        float t4738 = memory[83996148 + t4737];
        int t4739 = t4050 + t4731;
        int t4740 = t4739 + 512;
        float t4741 = memory[83996148 + t4740];
        float t4742 = t4728 * t4738;
        float t4743 = t4729 * t4741;
        float t4744 = t4742 - t4743;
        float t4745 = t4728 * t4741;
        float t4746 = t4729 * t4738;
        float t4747 = t4745 + t4746;
        int t4748 = t4050 + t4730;
        float t4749 = t4733 + t4744;
        memory[83996148 + t4748] = t4749;
        int t4751 = t4050 + t4730;
        int t4752 = t4751 + 512;
        float t4753 = t4736 + t4747;
        memory[83996148 + t4752] = t4753;
        int t4755 = t4050 + t4731;
        float t4756 = t4733 - t4744;
        memory[83996148 + t4755] = t4756;
        int t4758 = t4050 + t4731;
        int t4759 = t4758 + 512;
        float t4760 = t4736 - t4747;
        memory[83996148 + t4759] = t4760;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4763 = 0; _pr4763 < 256; _pr4763++) {
        float t4764 = (float)_pr4763;
        float t4765 = (t4764 * 0.125);
        float t4766 = metal::floor(t4765);
        float t4767 = t4766 * 8.0;
        float t4768 = t4764 - t4767;
        float t4769 = t4766 * 16.0;
        float t4770 = t4769 + t4768;
        float t4771 = t4770 + 8.0;
        float t4772 = 6.283185 * t4768;
        float t4773 = (t4772 * 0.0625);
        float t4774 = metal::cos(t4773);
        float t4775 = metal::sin(t4773);
        int t4776 = (int)t4770;
        int t4777 = (int)t4771;
        int t4778 = t4050 + t4776;
        float t4779 = memory[83996148 + t4778];
        int t4780 = t4050 + t4776;
        int t4781 = t4780 + 512;
        float t4782 = memory[83996148 + t4781];
        int t4783 = t4050 + t4777;
        float t4784 = memory[83996148 + t4783];
        int t4785 = t4050 + t4777;
        int t4786 = t4785 + 512;
        float t4787 = memory[83996148 + t4786];
        float t4788 = t4774 * t4784;
        float t4789 = t4775 * t4787;
        float t4790 = t4788 - t4789;
        float t4791 = t4774 * t4787;
        float t4792 = t4775 * t4784;
        float t4793 = t4791 + t4792;
        int t4794 = t4050 + t4776;
        float t4795 = t4779 + t4790;
        memory[83996148 + t4794] = t4795;
        int t4797 = t4050 + t4776;
        int t4798 = t4797 + 512;
        float t4799 = t4782 + t4793;
        memory[83996148 + t4798] = t4799;
        int t4801 = t4050 + t4777;
        float t4802 = t4779 - t4790;
        memory[83996148 + t4801] = t4802;
        int t4804 = t4050 + t4777;
        int t4805 = t4804 + 512;
        float t4806 = t4782 - t4793;
        memory[83996148 + t4805] = t4806;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4809 = 0; _pr4809 < 256; _pr4809++) {
        float t4810 = (float)_pr4809;
        float t4811 = (t4810 * 0.0625);
        float t4812 = metal::floor(t4811);
        float t4813 = t4812 * 16.0;
        float t4814 = t4810 - t4813;
        float t4815 = t4812 * 32.0;
        float t4816 = t4815 + t4814;
        float t4817 = t4816 + 16.0;
        float t4818 = 6.283185 * t4814;
        float t4819 = (t4818 * 0.03125);
        float t4820 = metal::cos(t4819);
        float t4821 = metal::sin(t4819);
        int t4822 = (int)t4816;
        int t4823 = (int)t4817;
        int t4824 = t4050 + t4822;
        float t4825 = memory[83996148 + t4824];
        int t4826 = t4050 + t4822;
        int t4827 = t4826 + 512;
        float t4828 = memory[83996148 + t4827];
        int t4829 = t4050 + t4823;
        float t4830 = memory[83996148 + t4829];
        int t4831 = t4050 + t4823;
        int t4832 = t4831 + 512;
        float t4833 = memory[83996148 + t4832];
        float t4834 = t4820 * t4830;
        float t4835 = t4821 * t4833;
        float t4836 = t4834 - t4835;
        float t4837 = t4820 * t4833;
        float t4838 = t4821 * t4830;
        float t4839 = t4837 + t4838;
        int t4840 = t4050 + t4822;
        float t4841 = t4825 + t4836;
        memory[83996148 + t4840] = t4841;
        int t4843 = t4050 + t4822;
        int t4844 = t4843 + 512;
        float t4845 = t4828 + t4839;
        memory[83996148 + t4844] = t4845;
        int t4847 = t4050 + t4823;
        float t4848 = t4825 - t4836;
        memory[83996148 + t4847] = t4848;
        int t4850 = t4050 + t4823;
        int t4851 = t4850 + 512;
        float t4852 = t4828 - t4839;
        memory[83996148 + t4851] = t4852;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4855 = 0; _pr4855 < 256; _pr4855++) {
        float t4856 = (float)_pr4855;
        float t4857 = (t4856 * 0.03125);
        float t4858 = metal::floor(t4857);
        float t4859 = t4858 * 32.0;
        float t4860 = t4856 - t4859;
        float t4861 = t4858 * 64.0;
        float t4862 = t4861 + t4860;
        float t4863 = t4862 + 32.0;
        float t4864 = 6.283185 * t4860;
        float t4865 = (t4864 * 0.015625);
        float t4866 = metal::cos(t4865);
        float t4867 = metal::sin(t4865);
        int t4868 = (int)t4862;
        int t4869 = (int)t4863;
        int t4870 = t4050 + t4868;
        float t4871 = memory[83996148 + t4870];
        int t4872 = t4050 + t4868;
        int t4873 = t4872 + 512;
        float t4874 = memory[83996148 + t4873];
        int t4875 = t4050 + t4869;
        float t4876 = memory[83996148 + t4875];
        int t4877 = t4050 + t4869;
        int t4878 = t4877 + 512;
        float t4879 = memory[83996148 + t4878];
        float t4880 = t4866 * t4876;
        float t4881 = t4867 * t4879;
        float t4882 = t4880 - t4881;
        float t4883 = t4866 * t4879;
        float t4884 = t4867 * t4876;
        float t4885 = t4883 + t4884;
        int t4886 = t4050 + t4868;
        float t4887 = t4871 + t4882;
        memory[83996148 + t4886] = t4887;
        int t4889 = t4050 + t4868;
        int t4890 = t4889 + 512;
        float t4891 = t4874 + t4885;
        memory[83996148 + t4890] = t4891;
        int t4893 = t4050 + t4869;
        float t4894 = t4871 - t4882;
        memory[83996148 + t4893] = t4894;
        int t4896 = t4050 + t4869;
        int t4897 = t4896 + 512;
        float t4898 = t4874 - t4885;
        memory[83996148 + t4897] = t4898;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4901 = 0; _pr4901 < 256; _pr4901++) {
        float t4902 = (float)_pr4901;
        float t4903 = (t4902 * 0.015625);
        float t4904 = metal::floor(t4903);
        float t4905 = t4904 * 64.0;
        float t4906 = t4902 - t4905;
        float t4907 = t4904 * 128.0;
        float t4908 = t4907 + t4906;
        float t4909 = t4908 + 64.0;
        float t4910 = 6.283185 * t4906;
        float t4911 = (t4910 * 0.0078125);
        float t4912 = metal::cos(t4911);
        float t4913 = metal::sin(t4911);
        int t4914 = (int)t4908;
        int t4915 = (int)t4909;
        int t4916 = t4050 + t4914;
        float t4917 = memory[83996148 + t4916];
        int t4918 = t4050 + t4914;
        int t4919 = t4918 + 512;
        float t4920 = memory[83996148 + t4919];
        int t4921 = t4050 + t4915;
        float t4922 = memory[83996148 + t4921];
        int t4923 = t4050 + t4915;
        int t4924 = t4923 + 512;
        float t4925 = memory[83996148 + t4924];
        float t4926 = t4912 * t4922;
        float t4927 = t4913 * t4925;
        float t4928 = t4926 - t4927;
        float t4929 = t4912 * t4925;
        float t4930 = t4913 * t4922;
        float t4931 = t4929 + t4930;
        int t4932 = t4050 + t4914;
        float t4933 = t4917 + t4928;
        memory[83996148 + t4932] = t4933;
        int t4935 = t4050 + t4914;
        int t4936 = t4935 + 512;
        float t4937 = t4920 + t4931;
        memory[83996148 + t4936] = t4937;
        int t4939 = t4050 + t4915;
        float t4940 = t4917 - t4928;
        memory[83996148 + t4939] = t4940;
        int t4942 = t4050 + t4915;
        int t4943 = t4942 + 512;
        float t4944 = t4920 - t4931;
        memory[83996148 + t4943] = t4944;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4947 = 0; _pr4947 < 256; _pr4947++) {
        float t4948 = (float)_pr4947;
        float t4949 = (t4948 * 0.0078125);
        float t4950 = metal::floor(t4949);
        float t4951 = t4950 * 128.0;
        float t4952 = t4948 - t4951;
        float t4953 = t4950 * 256.0;
        float t4954 = t4953 + t4952;
        float t4955 = t4954 + 128.0;
        float t4956 = 6.283185 * t4952;
        float t4957 = (t4956 * 0.00390625);
        float t4958 = metal::cos(t4957);
        float t4959 = metal::sin(t4957);
        int t4960 = (int)t4954;
        int t4961 = (int)t4955;
        int t4962 = t4050 + t4960;
        float t4963 = memory[83996148 + t4962];
        int t4964 = t4050 + t4960;
        int t4965 = t4964 + 512;
        float t4966 = memory[83996148 + t4965];
        int t4967 = t4050 + t4961;
        float t4968 = memory[83996148 + t4967];
        int t4969 = t4050 + t4961;
        int t4970 = t4969 + 512;
        float t4971 = memory[83996148 + t4970];
        float t4972 = t4958 * t4968;
        float t4973 = t4959 * t4971;
        float t4974 = t4972 - t4973;
        float t4975 = t4958 * t4971;
        float t4976 = t4959 * t4968;
        float t4977 = t4975 + t4976;
        int t4978 = t4050 + t4960;
        float t4979 = t4963 + t4974;
        memory[83996148 + t4978] = t4979;
        int t4981 = t4050 + t4960;
        int t4982 = t4981 + 512;
        float t4983 = t4966 + t4977;
        memory[83996148 + t4982] = t4983;
        int t4985 = t4050 + t4961;
        float t4986 = t4963 - t4974;
        memory[83996148 + t4985] = t4986;
        int t4988 = t4050 + t4961;
        int t4989 = t4988 + 512;
        float t4990 = t4966 - t4977;
        memory[83996148 + t4989] = t4990;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4993 = 0; _pr4993 < 256; _pr4993++) {
        float t4994 = (float)_pr4993;
        float t4995 = (t4994 * 0.00390625);
        float t4996 = metal::floor(t4995);
        float t4997 = t4996 * 256.0;
        float t4998 = t4994 - t4997;
        float t4999 = t4996 * 512.0;
        float t5000 = t4999 + t4998;
        float t5001 = t5000 + 256.0;
        float t5002 = 6.283185 * t4998;
        float t5003 = (t5002 * 0.001953125);
        float t5004 = metal::cos(t5003);
        float t5005 = metal::sin(t5003);
        int t5006 = (int)t5000;
        int t5007 = (int)t5001;
        int t5008 = t4050 + t5006;
        float t5009 = memory[83996148 + t5008];
        int t5010 = t4050 + t5006;
        int t5011 = t5010 + 512;
        float t5012 = memory[83996148 + t5011];
        int t5013 = t4050 + t5007;
        float t5014 = memory[83996148 + t5013];
        int t5015 = t4050 + t5007;
        int t5016 = t5015 + 512;
        float t5017 = memory[83996148 + t5016];
        float t5018 = t5004 * t5014;
        float t5019 = t5005 * t5017;
        float t5020 = t5018 - t5019;
        float t5021 = t5004 * t5017;
        float t5022 = t5005 * t5014;
        float t5023 = t5021 + t5022;
        int t5024 = t4050 + t5006;
        float t5025 = t5009 + t5020;
        memory[83996148 + t5024] = t5025;
        int t5027 = t4050 + t5006;
        int t5028 = t5027 + 512;
        float t5029 = t5012 + t5023;
        memory[83996148 + t5028] = t5029;
        int t5031 = t4050 + t5007;
        float t5032 = t5009 - t5020;
        memory[83996148 + t5031] = t5032;
        int t5034 = t4050 + t5007;
        int t5035 = t5034 + 512;
        float t5036 = t5012 - t5023;
        memory[83996148 + t5035] = t5036;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr5039 = 0; _pr5039 < 512; _pr5039++) {
        int t5040 = t4050 + _pr5039;
        float t5041 = memory[83996148 + t5040];
        float t5042 = t5041 * 7.599708e-06;
        float t5043 = memory[25460 + (int)_pr5039];
        int t5044 = t4051 + _pr5039;
        float t5045 = t5042 * t5043;
        memory[125955572 + t5044] = t5045;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t5049 = t[11*frameCount + id] > 0.0;
    if (t5049) {
      for (uint _pr5051 = 0; _pr5051 < 512; _pr5051++) {
        int t5052 = t4051 + _pr5051;
        memory[117550580 + t5052] = 0.0;
        int t5054 = t4051 + _pr5051;
        memory[125955572 + t5054] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5075), value: global(5075)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(537) - handled in variable access */
    int t5058 = id;
    float t5059 = 0.0;
    for (uint t5060 = 0; t5060 < 512; t5060++) {
      float t5061 = (float)t5060;
      float t5062 = (float)t5058;
      float t5063 = t5062 + t5061;
      int t5064 = 511 - t5060;
      float t5065 = frameCount - 1.0;
      float t5066 = metal::min(t5063, t5065);
      int t5067 = (int)t5066;
      int t5068 = t5067 * 512;
      int t5069 = t5068 + t5064;
      float t5070 = memory[117550580 + t5069];
      float t5071 = t5063 < frameCount;
      float t5072 = metal::select(0.0, t5070, t5071 > 0.0);
      float t5073 = t5059 + t5072;
      t5059 = t5073;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[19*frameCount + id] = (t5059 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5093), value: global(5093)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(537) - handled in variable access */
    int t5076 = id;
    float t5077 = 0.0;
    for (uint t5078 = 0; t5078 < 512; t5078++) {
      float t5079 = (float)t5078;
      float t5080 = (float)t5076;
      float t5081 = t5080 + t5079;
      int t5082 = 511 - t5078;
      float t5083 = frameCount - 1.0;
      float t5084 = metal::min(t5081, t5083);
      int t5085 = (int)t5084;
      int t5086 = t5085 * 512;
      int t5087 = t5086 + t5082;
      float t5088 = memory[125955572 + t5087];
      float t5089 = t5081 < frameCount;
      float t5090 = metal::select(0.0, t5088, t5089 > 0.0);
      float t5091 = t5077 + t5090;
      t5077 = t5091;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[20*frameCount + id] = (t5077 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5120), value: global(5120)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5119), value: global(5119)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5104), value: global(5104)) */
  float t5751 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5751)) {
    /* loadGlobal(5093) - handled in variable access */
    /* loadGlobal(5075) - handled in variable access */
    /* loadGlobal(3980) - handled in variable access */
    /* loadGlobal(3962) - handled in variable access */
    /* loadGlobal(482) - handled in variable access */
    /* loadGlobal(481) - handled in variable access */
    /* loadGlobal(463) - handled in variable access */
    /* loadGlobal(333) - handled in variable access */
    int t5094 = id;
    int t5095 = t5094 / 61;
    uint _frameIndex = (uint)(t5095);
    int t5096 = t5095 * 61;
    int t5097 = t5094 - t5096;
    int t5098 = t5097 == 0.0;
    if (t5098) {
      float t5100 = t[17*frameCount + _frameIndex] + t[19*frameCount + _frameIndex];
      float t5101 = t[18*frameCount + _frameIndex] + t[20*frameCount + _frameIndex];
      float t5102 = 0.015625 * t5100;
      float t5103 = t[7*frameCount + _frameIndex] * t5100;
      t[21*frameCount + _frameIndex] = t[6*frameCount + _frameIndex] * t5102;
      float t5105 = t[5*frameCount + _frameIndex] * t5102;
      float t5106 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
      float t5107 = t5106 < 0.0;
      float t5108 = t5106 + 61.0;
      float t5109 = metal::select(t5106, t5108, t5107 > 0.0);
      float t5110 = t5109;
      float t5111 = metal::floor(t5110);
      float t5112 = t5110 - t5111;
      float t5113 = t5111 + 1.0;
      float t5114 = t5113 >= 61.0;
      float t5115 = metal::select(t5113, 0.0, t5114 > 0.0);
      float t5116 = 1.0 - t5112;
      float t5117 = t5105 * t5116;
      float t5118 = t5105 * t5112;
      atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5111], t5117, metal::memory_order_relaxed);
      atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5115], t5118, metal::memory_order_relaxed);
    }
    float t5122 = memory[209874420 + t5097];
    float t5123 = memory[60596 + t5097];
    float t5124 = t5122 / t5123;
    float t5125 = memory[60596 + t5097];
    float t5126 = memory[60596 + t5097];
    float t5127 = t5125 * t5126;
    float t5128 = 1.0 / t5127;
    float t5129 = memory[209874420 + t5097];
    float t5130 = t5129 * -1.0;
    float t5131 = t5130 * t5128;
    float t5132 = t5124 + t5131;
    float t5133 = memory[60724 + t5097];
    float t5134 = metal::exp(t5133);
    float t5135 = t5134 * t5131;
    float t5136 = -1.0 * t5135;
    int t5137 = _frameIndex;
    int t5138 = t5137 * 61;
    int t5139 = t5138 + t5097;
    memory[60916 + t5139] = t5136;
    float t5141 = memory[60788 + t5097];
    float t5142 = t5141 * t5135;
    /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=201, axis=0, in=[61, 1], out=[1], inFA=true, outFA=true), value: empty) */
    float t5143 = 0.0;
    int t5144 = t5097;
    int t5145 = t5144;
    int t5146 = t5097 - t5145;
    int t5147 = t5144;
    int t5148 = t5147;
    for (uint t5149 = 0; t5149 < 61; t5149++) {
      int t5150 = t5149;
      int t5151 = t5148 + t5150;
      int t5152 = _frameIndex;
      int t5153 = t5152 * 61;
      int t5154 = t5153 + t5151;
      float t5155 = memory[60916 + t5154];
      float t5156 = t5143 + t5155;
      t5143 = t5156;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t5158 = _frameIndex;
    int t5159 = t5158;
    int t5160 = t5159 + t5097;
    memory[2158068 + t5160] = t5143;
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
    for (uint t5162 = 0; t5162 < 3904; t5162++) {
      /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=203, axis=2, in=[61, 1], out=[61, 1, 64], inFA=true, outFA=true), value: empty) */
      int t5163 = t5162 / 64;
      int t5164 = t5163 % 61;
      int t5165 = t5164 * 1;
      int t5166 = 0 + t5165;
      int t5167 = t5162 / 64;
      int t5168 = t5167 % 1;
      int t5169 = t5168 * 1;
      int t5170 = t5166 + t5169;
      float t5171 = (float)t5170;
      int t5172 = id;
      int t5173 = t5172 * 61;
      float t5174 = t5173 + t5171;
      int t5175 = (int)t5174;
      float t5176 = memory[60916 + t5175];
      float t5177 = (float)t5162;
      int t5178 = id;
      int t5179 = t5178 * 3904;
      float t5180 = t5179 + t5177;
      int t5181 = (int)t5180;
      memory[273837620 + t5181] = t5176;
      int t5183 = t5162 / 64;
      int t5184 = t5183 * 64;
      int t5185 = t5162 - t5184;
      int t5186 = t5185 / 64;
      int t5187 = t5186 * 64;
      int t5188 = t5185 - t5187;
      int t5189 = t5188 / 64;
      int t5190 = t5189 * 64;
      int t5191 = t5188 - t5190;
      float t5192 = memory[8576 + t5191];
      int t5193 = id;
      int t5194 = t5193 * 3904;
      int t5195 = t5194 + t5162;
      float t5196 = memory[273837620 + t5195];
      float t5197 = t5192 * t5196;
      int t5198 = id;
      int t5199 = t5198 * 3904;
      int t5200 = t5199 + t5162;
      memory[209874484 + t5200] = t5197;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5202 = 0; t5202 < 64; t5202++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=206, axis=0, in=[61, 1, 64], out=[1, 64], inFA=true, outFA=true), value: empty) */
      float t5203 = 0.0;
      int t5204 = t5202 / 64;
      int t5205 = t5204 * 64;
      int t5206 = t5202 - t5205;
      int t5207 = t5206;
      int t5208 = t5207;
      int t5209 = t5206 - t5208;
      int t5210 = t5204 * 64;
      int t5211 = t5210;
      int t5212 = t5207;
      int t5213 = t5211 + t5212;
      for (uint t5214 = 0; t5214 < 61; t5214++) {
        int t5215 = t5214 * 64;
        int t5216 = t5213 + t5215;
        int t5217 = t5214 * 64;
        int t5218 = t5217 + t5207;
        float t5219 = memory[37172 + t5218];
        float t5220 = t5214 + 0.0;
        int t5221 = id;
        int t5222 = t5221 * 61;
        float t5223 = t5222 + t5220;
        int t5224 = (int)t5223;
        float t5225 = memory[60916 + t5224];
        float t5226 = t5219 * t5225;
        float t5227 = t5203 + t5226;
        t5203 = t5227;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5229 = id;
      int t5230 = t5229 * 64;
      int t5231 = t5230 + t5202;
      memory[37809652 + t5231] = t5203;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5233), value: global(5233)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5104) - handled in variable access */
    /* loadGlobal(455) - handled in variable access */
    /* loadGlobal(373) - handled in variable access */
    t[24*frameCount + id] = t[3*frameCount + id] * t[21*frameCount + id];
    float t5234 = t[4*frameCount + id] * t[21*frameCount + id];
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
  float t5752 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5752)) {
    /* loadGlobal(5233) - handled in variable access */
    int t5235 = id;
    int t5236 = t5235 / 64;
    uint _frameIndex = (uint)(t5236);
    int t5237 = t5236 * 64;
    int t5238 = t5235 - t5237;
    int t5239 = t5236 * 64;
    int t5240 = t5239 + t5238;
    memory[60916 + t5240] = t[24*frameCount + _frameIndex];
    int t5242 = _frameIndex;
    int t5243 = t5242 * 64;
    int t5244 = t5243 + t5238;
    float t5245 = memory[1109492 + t5244];
    int t5246 = _frameIndex;
    int t5247 = t5246 * 64;
    int t5248 = t5247 + t5238;
    float t5249 = memory[60916 + t5248];
    float t5250 = t5245 * t5249;
    int t5251 = _frameIndex;
    int t5252 = t5251 * 64;
    int t5253 = t5252 + t5238;
    float t5254 = memory[3206644 + t5253];
    int t5255 = _frameIndex;
    int t5256 = t5255 * 64;
    int t5257 = t5256 + t5238;
    float t5258 = memory[60916 + t5257];
    float t5259 = t5254 * t5258;
    int t5260 = _frameIndex;
    int t5261 = t5260 * 64;
    int t5262 = t5261 + t5238;
    memory[42020340 + t5262] = t5259;
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
  float t5753 = frameCount * 3904.0;
  if (id >= 0 && id < (uint)(t5753)) {
    /* loadGlobal(333) - handled in variable access */
    int t5264 = id;
    int t5265 = t5264 / 3904;
    uint _frameIndex = (uint)(t5265);
    int t5266 = t5265 * 3904;
    int t5267 = t5264 - t5266;
    float t5268 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5269 = t5268 < 0.0;
    float t5270 = t5268 + 61.0;
    float t5271 = metal::select(t5268, t5270, t5269 > 0.0);
    float t5272 = metal::floor(t5271);
    float t5273 = t5272 + 1.0;
    float t5274 = t5273 >= 61.0;
    float t5275 = metal::select(t5273, 0.0, t5274 > 0.0);
    float t5276 = t5271 - t5272;
    int t5277 = _frameIndex;
    memory[3206644 + t5277] = t5272;
    memory[46231028 + t5277] = t5276;
    float t5280 = t5277 + 16384.0;
    int t5281 = (int)t5280;
    memory[3206644 + t5281] = t5275;
    float t5283 = 1.0 - t5276;
    float t5284 = t5277 * 64.0;
    for (uint _pr5285 = 0; _pr5285 < 64; _pr5285++) {
      float t5286 = (float)_pr5285;
      float t5287 = t5284 + t5286;
      int t5288 = (int)t5287;
      float t5289 = memory[42020340 + t5288];
      float t5290 = t5284 + t5286;
      float t5291 = t5289 * t5283;
      int t5292 = (int)t5290;
      memory[60916 + t5292] = t5291;
      float t5294 = t5289 * t5276;
      int t5295 = (int)t5290;
      memory[1109492 + t5295] = t5294;
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
  if (id < 3904) { uint _pr5298 = id;
    int t5299 = _pr5298 / 64;
    int t5300 = t5299 * 64;
    int t5301 = _pr5298 - t5300;
    float t5302 = (float)t5299;
    float t5303 = (float)t5301;
    float t5304 = 0.0;
    for (uint t5305 = 0; t5305 < 16384; t5305++) {
      float t5306 = (float)t5305;
      float t5307 = t5306 < frameCount;
      float t5308 = t5306 * 64.0;
      float t5309 = t5308 + t5303;
      float t5310 = memory[3206644 + (int)t5305];
      float t5311 = t5310 - t5302;
      float t5312 = metal::abs(t5311);
      float t5313 = t5312 < 0.5;
      int t5314 = (int)t5309;
      float t5315 = memory[60916 + t5314];
      float t5316 = t5307 * t5313;
      float t5317 = t5316 > 0.0;
      float t5318 = metal::select(0.0, t5315, t5317 > 0.0);
      float t5319 = t5304 + t5318;
      t5304 = t5319;
      float t5320 = t5306 + 16384.0;
      int t5321 = (int)t5320;
      float t5322 = memory[3206644 + t5321];
      float t5323 = t5322 - t5302;
      float t5324 = metal::abs(t5323);
      float t5325 = t5324 < 0.5;
      int t5326 = (int)t5309;
      float t5327 = memory[1109492 + t5326];
      float t5328 = t5307 * t5325;
      float t5329 = t5328 > 0.0;
      float t5330 = metal::select(0.0, t5327, t5329 > 0.0);
      float t5331 = t5304 + t5330;
      t5304 = t5331;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t5333 = t5302 * 64.0;
    float t5334 = t5333 + t5303;
    int t5335 = (int)t5334;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[337800756 + t5335], t5304, metal::memory_order_relaxed);
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
    for (uint t5338 = 0; t5338 < 3904; t5338++) {
      float t5339 = memory[337800756 + (int)t5338];
      float t5340 = memory[48884 + (int)t5338];
      float t5341 = t5339 / t5340;
      float t5342 = memory[48884 + (int)t5338];
      float t5343 = memory[48884 + (int)t5338];
      float t5344 = t5342 * t5343;
      float t5345 = 1.0 / t5344;
      float t5346 = memory[337800756 + (int)t5338];
      float t5347 = t5346 * -1.0;
      float t5348 = t5347 * t5345;
      float t5349 = t5341 + t5348;
      float t5350 = memory[56692 + (int)t5338];
      float t5351 = metal::exp(t5350);
      float t5352 = t5351 * t5348;
      float t5353 = -1.0 * t5352;
      int t5354 = id;
      int t5355 = t5354 * 3904;
      int t5356 = t5355 + t5338;
      memory[273837620 + t5356] = t5353;
      float t5358 = memory[44980 + (int)t5338];
      float t5359 = t5358 * t5352;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5360 = 0; t5360 < 64; t5360++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=232, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5361 = 0.0;
      int t5362 = t5360;
      int t5363 = t5362;
      int t5364 = t5360 - t5363;
      int t5365 = t5362;
      int t5366 = t5365;
      for (uint t5367 = 0; t5367 < 61; t5367++) {
        int t5368 = t5367 * 64;
        int t5369 = t5366 + t5368;
        int t5370 = id;
        int t5371 = t5370 * 3904;
        int t5372 = t5371 + t5369;
        float t5373 = memory[273837620 + t5372];
        float t5374 = t5361 + t5373;
        t5361 = t5374;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5376 = id;
      int t5377 = t5376 * 64;
      int t5378 = t5377 + t5360;
      memory[60916 + t5378] = t5361;
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
    for (uint t5380 = 0; t5380 < 3904; t5380++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=237, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5381 = 0.0;
      int t5382 = t5380 / 64;
      int t5383 = t5382 * 64;
      int t5384 = t5380 - t5383;
      int t5385 = t5384;
      int t5386 = t5385;
      int t5387 = t5384 - t5386;
      int t5388 = t5382 * 4096;
      int t5389 = t5388;
      int t5390 = t5385;
      int t5391 = t5389 + t5390;
      for (uint t5392 = 0; t5392 < 64; t5392++) {
        int t5393 = t5392 * 64;
        int t5394 = t5391 + t5393;
        int t5395 = t5392 * 64;
        int t5396 = t5395 + t5385;
        int t5397 = t5396 / 64;
        int t5398 = t5397 * 64;
        int t5399 = t5396 - t5398;
        int t5400 = t5399 * 64;
        int t5401 = t5397 + t5400;
        float t5402 = memory[4416 + t5401];
        int t5403 = t5382 * 64;
        int t5404 = t5403 + t5392;
        int t5405 = id;
        int t5406 = t5405 * 3904;
        int t5407 = t5406 + t5404;
        float t5408 = memory[273837620 + t5407];
        float t5409 = t5402 * t5408;
        float t5410 = t5381 + t5409;
        t5381 = t5410;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5412 = id;
      int t5413 = t5412 * 3904;
      int t5414 = t5413 + t5380;
      memory[404913524 + t5414] = t5381;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5416 = 0; t5416 < 4096; t5416++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=239, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5417 = 0.0;
      int t5418 = t5416 / 64;
      int t5419 = t5418 * 64;
      int t5420 = t5416 - t5419;
      int t5421 = t5420;
      int t5422 = t5421;
      int t5423 = t5420 - t5422;
      int t5424 = t5418 * 64;
      int t5425 = t5424;
      int t5426 = t5421;
      int t5427 = t5425 + t5426;
      for (uint t5428 = 0; t5428 < 61; t5428++) {
        int t5429 = t5428 * 4096;
        int t5430 = t5427 + t5429;
        int t5431 = t5428 * 64;
        int t5432 = t5431 + t5421;
        float t5433 = memory[37172 + t5432];
        int t5434 = t5428 * 64;
        int t5435 = t5434 + t5418;
        int t5436 = id;
        int t5437 = t5436 * 3904;
        int t5438 = t5437 + t5435;
        float t5439 = memory[273837620 + t5438];
        float t5440 = t5433 * t5439;
        float t5441 = t5417 + t5440;
        t5417 = t5441;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5443 = id;
      int t5444 = t5443 * 4096;
      int t5445 = t5444 + t5416;
      memory[337804660 + t5445] = t5417;
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
    for (uint t5447 = 0; t5447 < 3904; t5447++) {
      int t5448 = id;
      int t5449 = t5448 * 3904;
      int t5450 = t5449 + t5447;
      float t5451 = memory[209874484 + t5450];
      int t5452 = id;
      int t5453 = t5452 * 3904;
      int t5454 = t5453 + t5447;
      float t5455 = memory[404913524 + t5454];
      float t5456 = t5451 + t5455;
      float t5457 = memory[41076 + (int)t5447];
      float t5458 = metal::tanh(t5457);
      float t5459 = t5458 * t5458;
      float t5460 = 1.0 - t5459;
      float t5461 = t5460 * t5456;
      int t5462 = id;
      int t5463 = t5462 * 3904;
      int t5464 = t5463 + t5447;
      memory[468876660 + t5464] = t5461;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5466 = 0; t5466 < 64; t5466++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=250, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5467 = 0.0;
      int t5468 = t5466;
      int t5469 = t5468;
      int t5470 = t5466 - t5469;
      int t5471 = t5468;
      int t5472 = t5471;
      for (uint t5473 = 0; t5473 < 61; t5473++) {
        int t5474 = t5473 * 64;
        int t5475 = t5472 + t5474;
        int t5476 = t5473 * 64;
        int t5477 = t5476 + t5468;
        float t5478 = memory[25460 + t5477];
        int t5479 = t5473 * 64;
        int t5480 = t5479 + t5468;
        int t5481 = id;
        int t5482 = t5481 * 3904;
        int t5483 = t5482 + t5480;
        float t5484 = memory[273837620 + t5483];
        float t5485 = t5478 * t5484;
        float t5486 = t5467 + t5485;
        t5467 = t5486;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5488 = id;
      int t5489 = t5488 * 64;
      int t5490 = t5489 + t5466;
      memory[1109492 + t5490] = t5467;
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
    for (uint t5492 = 0; t5492 < 3904; t5492++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=255, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5493 = 0.0;
      int t5494 = t5492 / 64;
      int t5495 = t5494 * 64;
      int t5496 = t5492 - t5495;
      int t5497 = t5496;
      int t5498 = t5497;
      int t5499 = t5496 - t5498;
      int t5500 = t5494 * 4096;
      int t5501 = t5500;
      int t5502 = t5497;
      int t5503 = t5501 + t5502;
      for (uint t5504 = 0; t5504 < 64; t5504++) {
        int t5505 = t5504 * 64;
        int t5506 = t5503 + t5505;
        int t5507 = t5504 * 64;
        int t5508 = t5507 + t5497;
        int t5509 = t5508 / 64;
        int t5510 = t5509 * 64;
        int t5511 = t5508 - t5510;
        int t5512 = t5511 * 64;
        int t5513 = t5509 + t5512;
        float t5514 = memory[256 + t5513];
        int t5515 = t5494 * 64;
        int t5516 = t5515 + t5504;
        int t5517 = id;
        int t5518 = t5517 * 3904;
        int t5519 = t5518 + t5516;
        float t5520 = memory[468876660 + t5519];
        float t5521 = t5514 * t5520;
        float t5522 = t5493 + t5521;
        t5493 = t5522;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5524 = id;
      int t5525 = t5524 * 3904;
      int t5526 = t5525 + t5492;
      memory[209874484 + t5526] = t5493;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5528 = 0; t5528 < 4096; t5528++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=257, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5529 = 0.0;
      int t5530 = t5528 / 64;
      int t5531 = t5530 * 64;
      int t5532 = t5528 - t5531;
      int t5533 = t5532;
      int t5534 = t5533;
      int t5535 = t5532 - t5534;
      int t5536 = t5530 * 64;
      int t5537 = t5536;
      int t5538 = t5533;
      int t5539 = t5537 + t5538;
      for (uint t5540 = 0; t5540 < 61; t5540++) {
        int t5541 = t5540 * 4096;
        int t5542 = t5539 + t5541;
        int t5543 = t5540 * 64;
        int t5544 = t5543 + t5533;
        float t5545 = memory[29364 + t5544];
        int t5546 = t5540 * 64;
        int t5547 = t5546 + t5530;
        int t5548 = id;
        int t5549 = t5548 * 3904;
        int t5550 = t5549 + t5547;
        float t5551 = memory[468876660 + t5550];
        float t5552 = t5545 * t5551;
        float t5553 = t5529 + t5552;
        t5529 = t5553;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5555 = id;
      int t5556 = t5555 * 4096;
      int t5557 = t5556 + t5528;
      memory[532839796 + t5557] = t5529;
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
    for (uint t5559 = 0; t5559 < 3904; t5559++) {
      float t5560 = memory[33268 + (int)t5559];
      float t5561 = metal::tanh(t5560);
      float t5562 = t5561 * t5561;
      float t5563 = 1.0 - t5562;
      int t5564 = id;
      int t5565 = t5564 * 3904;
      int t5566 = t5565 + t5559;
      float t5567 = memory[209874484 + t5566];
      float t5568 = t5563 * t5567;
      int t5569 = id;
      int t5570 = t5569 * 3904;
      int t5571 = t5570 + t5559;
      memory[273837620 + t5571] = t5568;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5573 = 0; t5573 < 64; t5573++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=267, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5574 = 0.0;
      int t5575 = t5573;
      int t5576 = t5575;
      int t5577 = t5573 - t5576;
      int t5578 = t5575;
      int t5579 = t5578;
      for (uint t5580 = 0; t5580 < 61; t5580++) {
        int t5581 = t5580 * 64;
        int t5582 = t5579 + t5581;
        int t5583 = t5580 * 64;
        int t5584 = t5583 + t5575;
        float t5585 = memory[25460 + t5584];
        int t5586 = t5580 * 64;
        int t5587 = t5586 + t5575;
        int t5588 = id;
        int t5589 = t5588 * 3904;
        int t5590 = t5589 + t5587;
        float t5591 = memory[209874484 + t5590];
        float t5592 = t5585 * t5591;
        float t5593 = t5574 + t5592;
        t5574 = t5593;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5595 = id;
      int t5596 = t5595 * 64;
      int t5597 = t5596 + t5573;
      memory[3206644 + t5597] = t5574;
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
    for (uint t5599 = 0; t5599 < 183; t5599++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=272, axis=1, in=[61, 64, 3], out=[61, 3], inFA=true, outFA=true), value: empty) */
      float t5600 = 0.0;
      int t5601 = t5599 / 3;
      int t5602 = t5601 * 3;
      int t5603 = t5599 - t5602;
      int t5604 = t5603;
      int t5605 = t5604;
      int t5606 = t5603 - t5605;
      int t5607 = t5601 * 192;
      int t5608 = t5607;
      int t5609 = t5604;
      int t5610 = t5608 + t5609;
      for (uint t5611 = 0; t5611 < 64; t5611++) {
        int t5612 = t5611 * 3;
        int t5613 = t5610 + t5612;
        int t5614 = t5611 * 3;
        int t5615 = t5614 + t5604;
        int t5616 = t5615 / 3;
        int t5617 = t5616 * 3;
        int t5618 = t5615 - t5617;
        int t5619 = t5618 * 64;
        int t5620 = t5616 + t5619;
        float t5621 = memory[0 + t5620];
        int t5622 = t5601 * 64;
        int t5623 = t5622 + t5611;
        int t5624 = id;
        int t5625 = t5624 * 3904;
        int t5626 = t5625 + t5623;
        float t5627 = memory[273837620 + t5626];
        float t5628 = t5621 * t5627;
        float t5629 = t5600 + t5628;
        t5600 = t5629;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5631 = id;
      int t5632 = t5631 * 183;
      int t5633 = t5632 + t5599;
      memory[46231028 + t5633] = t5600;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 3]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5635 = 0; t5635 < 192; t5635++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=274, axis=0, in=[61, 64, 3], out=[64, 3], inFA=true, outFA=true), value: empty) */
      float t5636 = 0.0;
      int t5637 = t5635 / 3;
      int t5638 = t5637 * 3;
      int t5639 = t5635 - t5638;
      int t5640 = t5639;
      int t5641 = t5640;
      int t5642 = t5639 - t5641;
      int t5643 = t5637 * 3;
      int t5644 = t5643;
      int t5645 = t5640;
      int t5646 = t5644 + t5645;
      for (uint t5647 = 0; t5647 < 61; t5647++) {
        int t5648 = t5647 * 192;
        int t5649 = t5646 + t5648;
        int t5650 = t5647 * 3;
        int t5651 = t5650 + t5640;
        float t5652 = memory[8706 + t5651];
        int t5653 = t5647 * 64;
        int t5654 = t5653 + t5637;
        int t5655 = id;
        int t5656 = t5655 * 3904;
        int t5657 = t5656 + t5654;
        float t5658 = memory[273837620 + t5657];
        float t5659 = t5652 * t5658;
        float t5660 = t5636 + t5659;
        t5636 = t5660;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5662 = id;
      int t5663 = t5662 * 192;
      int t5664 = t5663 + t5635;
      memory[42020340 + t5664] = t5636;
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
  if (id < 192) { uint _pr5666 = id;
    float t5667 = 0.0;
    for (uint t5668 = 0; t5668 < 16384; t5668++) {
      int t5669 = t5668 * 192;
      int t5670 = t5669 + _pr5666;
      float t5671 = memory[42020340 + t5670];
      float t5672 = t5667 + t5671;
      t5667 = t5672;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5666] = t5667;
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
  if (id < 64) { uint _pr5676 = id;
    float t5677 = 0.0;
    for (uint t5678 = 0; t5678 < 16384; t5678++) {
      int t5679 = t5678 * 64;
      int t5680 = t5679 + _pr5676;
      float t5681 = memory[3206644 + t5680];
      float t5682 = t5677 + t5681;
      t5677 = t5682;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5676] = t5677;
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
  if (id < 4096) { uint _pr5686 = id;
    float t5687 = 0.0;
    for (uint t5688 = 0; t5688 < 16384; t5688++) {
      int t5689 = t5688 * 4096;
      int t5690 = t5689 + _pr5686;
      float t5691 = memory[532839796 + t5690];
      float t5692 = t5687 + t5691;
      t5687 = t5692;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[3206644 + (int)_pr5686] = t5687;
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
  if (id < 64) { uint _pr5696 = id;
    float t5697 = 0.0;
    for (uint t5698 = 0; t5698 < 16384; t5698++) {
      int t5699 = t5698 * 64;
      int t5700 = t5699 + _pr5696;
      float t5701 = memory[1109492 + t5700];
      float t5702 = t5697 + t5701;
      t5697 = t5702;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5696] = t5697;
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
  if (id < 4096) { uint _pr5706 = id;
    float t5707 = 0.0;
    for (uint t5708 = 0; t5708 < 16384; t5708++) {
      int t5709 = t5708 * 4096;
      int t5710 = t5709 + _pr5706;
      float t5711 = memory[337804660 + t5710];
      float t5712 = t5707 + t5711;
      t5707 = t5712;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[1109492 + (int)_pr5706] = t5707;
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
  if (id < 64) { uint _pr5716 = id;
    float t5717 = 0.0;
    for (uint t5718 = 0; t5718 < 16384; t5718++) {
      int t5719 = t5718 * 64;
      int t5720 = t5719 + _pr5716;
      float t5721 = memory[60916 + t5720];
      float t5722 = t5717 + t5721;
      t5717 = t5722;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5716] = t5717;
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
  if (id < 64) { uint _pr5726 = id;
    float t5727 = 0.0;
    for (uint t5728 = 0; t5728 < 16384; t5728++) {
      int t5729 = t5728 * 64;
      int t5730 = t5729 + _pr5726;
      float t5731 = memory[37809652 + t5730];
      float t5732 = t5727 + t5731;
      t5727 = t5732;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5726] = t5727;
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
  if (id < 1) { uint _pr5736 = id;
    float t5737 = 0.0;
    for (uint t5738 = 0; t5738 < 16384; t5738++) {
      int t5739 = t5738;
      int t5740 = t5739 + _pr5736;
      float t5741 = memory[2158068 + t5740];
      float t5742 = t5737 + t5741;
      t5737 = t5742;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[60596 + (int)_pr5736] = t5737;
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
    /* loadGlobal(5120) - handled in variable access */
    /* loadGlobal(5119) - handled in variable access */
    /* loadGlobal(2767) - handled in variable access */
    outputs[0 * frameCount + id] = t[16*frameCount + id];
  }
  #pragma clang diagnostic pop
}

