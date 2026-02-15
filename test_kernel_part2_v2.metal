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
    memory[37172 + t135] = t171;
    float t173 = metal::tanh(t171);
    memory[41076 + t135] = t173;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 2
// Kind: simd
// ThreadCountScale nil
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
  if (id >= 0 && id < (uint)(1)) {
    for (uint t175 = 0; t175 < 3904; t175++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=33, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t176 = 0.0;
      int t177 = t175 / 64;
      int t178 = t177 * 64;
      int t179 = t175 - t178;
      int t180 = t179;
      int t181 = t180;
      int t182 = t179 - t181;
      int t183 = t177 * 4096;
      int t184 = t183;
      int t185 = t180 * 64;
      int t186 = t184 + t185;
      for (uint t187 = 0; t187 < 64; t187++) {
        int t188 = t187;
        int t189 = t186 + t188;
        int t190 = t177 * 64;
        int t191 = t190 + t187;
        float t192 = memory[41076 + t191];
        int t193 = t180 * 64;
        int t194 = t193 + t187;
        int t195 = t194 / 64;
        int t196 = t195 * 64;
        int t197 = t194 - t196;
        int t198 = t197 * 64;
        int t199 = t195 + t198;
        float t200 = memory[4416 + t199];
        float t201 = t192 * t200;
        float t202 = t176 + t201;
        t176 = t202;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t175] = t176;
      float t205 = memory[25460 + (int)t175];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t206 = t175 / 64;
      int t207 = t206 * 64;
      int t208 = t175 - t207;
      int t209 = t208;
      float t210 = memory[8512 + t209];
      float t211 = t205 + t210;
      memory[52788 + (int)t175] = t211;
      float t213 = t211 * -1.0;
      memory[56692 + (int)t175] = t213;
      float t215 = metal::exp(t213);
      float t216 = 1.0 + t215;
      memory[48884 + (int)t175] = t216;
      float t218 = 1.0 / t216;
      memory[44980 + (int)t175] = t218;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 3
// Kind: simd
// ThreadCountScale nil
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
  if (id >= 0 && id < (uint)(1)) {
    for (uint t220 = 0; t220 < 61; t220++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=45, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t221 = 0.0;
      int t222 = t220;
      int t223 = t222;
      int t224 = t220 - t223;
      int t225 = t224;
      int t226 = t225;
      int t227 = t224 - t226;
      int t228 = t222 * 64;
      int t229 = t228;
      int t230 = t225 * 64;
      int t231 = t229 + t230;
      for (uint t232 = 0; t232 < 64; t232++) {
        int t233 = t232;
        int t234 = t231 + t233;
        int t235 = t222 * 64;
        int t236 = t235 + t232;
        float t237 = memory[41076 + t236];
        int t238 = t232 / 64;
        int t239 = t238 * 64;
        int t240 = t232 - t239;
        float t241 = memory[8576 + t240];
        float t242 = t237 * t241;
        float t243 = t221 + t242;
        t221 = t243;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t220] = t221;
      float t246 = memory[25460 + (int)t220];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t247 = t220;
      int t248 = t247;
      int t249 = t220 - t248;
      float t250 = memory[8640 + (int)0.0];
      float t251 = t246 + t250;
      memory[60660 + (int)t220] = t251;
      float t253 = t251 * -1.0;
      memory[60596 + (int)t220] = t253;
      float t255 = metal::exp(t253);
      float t256 = 1.0 + t255;
      memory[60724 + (int)t220] = t256;
      float t258 = 1.0 / t256;
      memory[60788 + (int)t220] = t258;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 4
// Kind: simd
// ThreadCountScale nil
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
  if (id >= 0 && id < (uint)(1)) {
    for (uint t260 = 0; t260 < 61; t260++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=57, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t261 = 0.0;
      int t262 = t260;
      int t263 = t262;
      int t264 = t260 - t263;
      int t265 = t264;
      int t266 = t265;
      int t267 = t264 - t266;
      int t268 = t262 * 64;
      int t269 = t268;
      int t270 = t265 * 64;
      int t271 = t269 + t270;
      for (uint t272 = 0; t272 < 64; t272++) {
        int t273 = t272;
        int t274 = t271 + t273;
        int t275 = t262 * 64;
        int t276 = t275 + t272;
        float t277 = memory[41076 + t276];
        int t278 = t272 / 64;
        int t279 = t278 * 64;
        int t280 = t272 - t279;
        float t281 = memory[8641 + t280];
        float t282 = t277 * t281;
        float t283 = t261 + t282;
        t261 = t283;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t260] = t261;
      float t286 = memory[25460 + (int)t260];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t287 = t260;
      int t288 = t287;
      int t289 = t260 - t288;
      float t290 = memory[8705 + (int)0.0];
      float t291 = t286 + t290;
      float t292 = t291 * -1.0;
      float t293 = metal::exp(t292);
      float t294 = 1.0 + t293;
      float t295 = 1.0 / t294;
      memory[60852 + (int)t260] = t295;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(297), value: global(297)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[599948660];
      float t298 = t[0*frameCount + i] + 0.003662333;
      float t299 = metal::select(t298, 0.0, 0.0 > 0.0);
      float t300 = t299;
      float t301 = (t300 * 0.016666668);
      float t302 = metal::floor(t301);
      float t303 = t302 * 60.0;
      float t304 = t299 - t303;
      memory[599948660] = t304;
      float t306 = t304 >= 60.0;
      if (t306) {
        float t308 = t304 - 60.0;
        memory[599948660] = t308;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(355), value: global(355)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(335), value: global(335)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(315), value: global(315)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(297) - handled in variable access */
    float t314 = metal::min(t[0*frameCount + id], 59.9999);
    t[1*frameCount + id] = metal::max(t314, 0.0);
    float t316 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t317 = t316 < 0.0;
    float t318 = t316 + 61.0;
    float t319 = metal::select(t316, t318, t317 > 0.0);
    float t320 = t319;
    float t321 = metal::floor(t320);
    float t322 = t320 - t321;
    float t323 = t321 + 1.0;
    float t324 = t323 >= 61.0;
    float t325 = metal::select(t323, 0.0, t324 > 0.0);
    int t326 = (int)t321;
    float t327 = memory[25273 + t326];
    int t328 = (int)t325;
    float t329 = memory[25273 + t328];
    float t330 = 1.0 - t322;
    float t331 = t327 * t330;
    float t332 = t329 * t322;
    float t333 = t331 + t332;
    float t334 = metal::max(t333, 20.0);
    t[2*frameCount + id] = metal::min(t334, 500.0);
    float t336 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t337 = t336 < 0.0;
    float t338 = t336 + 61.0;
    float t339 = metal::select(t336, t338, t337 > 0.0);
    float t340 = t339;
    float t341 = metal::floor(t340);
    float t342 = t340 - t341;
    float t343 = t341 + 1.0;
    float t344 = t343 >= 61.0;
    float t345 = metal::select(t343, 0.0, t344 > 0.0);
    int t346 = (int)t341;
    float t347 = memory[25334 + t346];
    int t348 = (int)t345;
    float t349 = memory[25334 + t348];
    float t350 = 1.0 - t342;
    float t351 = t347 * t350;
    float t352 = t349 * t342;
    float t353 = t351 + t352;
    float t354 = metal::min(t353, 1.0);
    t[3*frameCount + id] = metal::max(t354, 0.0);
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
  float t5724 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5724)) {
    /* loadGlobal(335) - handled in variable access */
    /* loadGlobal(315) - handled in variable access */
    int t356 = id;
    int t357 = t356 / 64;
    uint _frameIndex = (uint)(t357);
    int t358 = t357 * 64;
    int t359 = t356 - t358;
    float t360 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t361 = t360 < 0.0;
    float t362 = t360 + 61.0;
    float t363 = metal::select(t360, t362, t361 > 0.0);
    float t364 = metal::floor(t363);
    float t365 = t364 + 1.0;
    float t366 = t365 >= 61.0;
    float t367 = metal::select(t365, 0.0, t366 > 0.0);
    float t368 = t363 - t364;
    float t369 = 1.0 - t368;
    float t370 = t357 * 64.0;
    float t371 = (float)t359;
    float t372 = t364 * 64.0;
    float t373 = t372 + t371;
    int t374 = (int)t373;
    float t375 = memory[44980 + t374];
    float t376 = t367 * 64.0;
    float t377 = t376 + t371;
    int t378 = (int)t377;
    float t379 = memory[44980 + t378];
    float t380 = t369 * t375;
    float t381 = t368 * t379;
    float t382 = t380 + t381;
    float t383 = t370 + t371;
    int t384 = (int)t383;
    memory[60916 + t384] = t382;
    int t386 = (int)t383;
    memory[2158068 + t386] = t382;
    float t388 = memory[25395 + t359];
    float t389 = t388 * t[2*frameCount + _frameIndex];
    int t390 = _frameIndex;
    int t391 = t390 * 64;
    int t392 = t391 + t359;
    memory[1109492 + t392] = t389;
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
      int t394 = id;
      int t395 = i;
      int t396 = t395 * 64;
      int t397 = t396 + t394;
      float t398 = memory[1109492 + t397];
      float t399 = (t398 * 6.25e-05);
      float t400 = memory[25460 + t394];
      float t401 = t400 + t399;
      float t402 = metal::select(t401, 0.0, 0.0 > 0.0);
      float t403 = metal::floor(t402);
      float t404 = t402 - t403;
      float t405 = t404 >= 1.0;
      float t406 = t404 - 1.0;
      float t407 = metal::select(t404, t406, t405 > 0.0);
      float t408 = metal::select(t407, 0.0, 0.0 > 0.0);
      memory[25460 + t394] = t408;
      int t410 = i;
      int t411 = t410 * 64;
      int t412 = t411 + t394;
      memory[60916 + t412] = t400;
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
  float t5725 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5725)) {
    int t414 = id;
    int t415 = t414 / 64;
    uint _frameIndex = (uint)(t415);
    int t416 = t415 * 64;
    int t417 = t414 - t416;
    int t418 = _frameIndex;
    int t419 = t418 * 64;
    int t420 = t419 + t417;
    float t421 = memory[60916 + t420];
    float t422 = t421 * 6.283185;
    float t423 = metal::sin(t422);
    int t424 = _frameIndex;
    int t425 = t424 * 64;
    int t426 = t425 + t417;
    memory[3206644 + t426] = t423;
    int t428 = _frameIndex;
    int t429 = t428 * 64;
    int t430 = t429 + t417;
    float t431 = memory[2158068 + t430];
    float t432 = t423 * t431;
    int t433 = _frameIndex;
    int t434 = t433 * 64;
    int t435 = t434 + t417;
    memory[1109492 + t435] = t432;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(437), value: global(437)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    t[4*frameCount + id] = 0.0;
    for (uint t438 = 0; t438 < 64; t438++) {
      int t439 = id;
      int t440 = t439 * 64;
      int t441 = t440 + t438;
      float t442 = memory[1109492 + t441];
      float t443 = t[4*frameCount + id] + t442;
      t[4*frameCount + id] = t443;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(465), value: global(465)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(464), value: global(464)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(463), value: global(463)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(445), value: global(445)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(437) - handled in variable access */
    /* loadGlobal(355) - handled in variable access */
    /* loadGlobal(315) - handled in variable access */
    t[5*frameCount + id] = t[4*frameCount + id] * t[3*frameCount + id];
    float t446 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t447 = t446 < 0.0;
    float t448 = t446 + 61.0;
    float t449 = metal::select(t446, t448, t447 > 0.0);
    float t450 = t449;
    float t451 = metal::floor(t450);
    float t452 = t450 - t451;
    float t453 = t451 + 1.0;
    float t454 = t453 >= 61.0;
    float t455 = metal::select(t453, 0.0, t454 > 0.0);
    int t456 = (int)t451;
    float t457 = memory[60788 + t456];
    int t458 = (int)t455;
    float t459 = memory[60788 + t458];
    float t460 = 1.0 - t452;
    float t461 = t457 * t460;
    float t462 = t459 * t452;
    t[6*frameCount + id] = t461 + t462;
    t[7*frameCount + id] = t[5*frameCount + id] * t[6*frameCount + id];
    t[8*frameCount + id] = t[7*frameCount + id] * 0.015625;
    float t466 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t467 = t466 < 0.0;
    float t468 = t466 + 61.0;
    float t469 = metal::select(t466, t468, t467 > 0.0);
    float t470 = t469;
    float t471 = metal::floor(t470);
    float t472 = t470 - t471;
    float t473 = t471 + 1.0;
    float t474 = t473 >= 61.0;
    float t475 = metal::select(t473, 0.0, t474 > 0.0);
    int t476 = (int)t471;
    float t477 = memory[60852 + t476];
    int t478 = (int)t475;
    float t479 = memory[60852 + t478];
    float t480 = 1.0 - t472;
    float t481 = t477 * t480;
    float t482 = t479 * t472;
    float t483 = t481 + t482;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(484), value: global(484)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[9*frameCount + i] = memory[599948661];
      float t485 = t[9*frameCount + i] + 1.0;
      float t486 = metal::select(t485, 0.0, 0.0 > 0.0);
      float t487 = t486;
      float t488 = (t487 * 6.1035156e-05);
      float t489 = metal::floor(t488);
      float t490 = t489 * 16384.0;
      float t491 = t486 - t490;
      memory[599948661] = t491;
      float t493 = t491 >= 16384.0;
      if (t493) {
        float t495 = t491 - 16384.0;
        memory[599948661] = t495;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(518), value: global(518)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(484) - handled in variable access */
    float t501 = (t[9*frameCount + id] - metal::floor(t[9*frameCount + id] / 16384.0) * 16384.0);
    float t502 = t501 < 0.0;
    float t503 = t501 + 16384.0;
    float t504 = metal::select(t501, t503, t502 > 0.0);
    float t505 = t504;
    float t506 = metal::floor(t505);
    float t507 = t505 - t506;
    float t508 = t506 + 1.0;
    float t509 = t508 >= 16384.0;
    float t510 = metal::select(t508, 0.0, t509 > 0.0);
    int t511 = (int)t506;
    float t512 = memory[8889 + t511];
    int t513 = (int)t510;
    float t514 = memory[8889 + t513];
    float t515 = 1.0 - t507;
    float t516 = t512 * t515;
    float t517 = t514 * t507;
    t[10*frameCount + id] = t516 + t517;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(519), value: global(519)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[11*frameCount + i] = memory[599948662];
      float t520 = t[11*frameCount + i] + 1.0;
      float t521 = metal::select(t520, 0.0, 0.0 > 0.0);
      float t522 = t521;
      float t523 = (t522 * 0.0078125);
      float t524 = metal::floor(t523);
      float t525 = t524 * 128.0;
      float t526 = t521 - t525;
      memory[599948662] = t526;
      float t528 = t526 >= 128.0;
      if (t528) {
        float t530 = t526 - 128.0;
        memory[599948662] = t530;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(540), value: global(540)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(519) - handled in variable access */
    /* loadGlobal(518) - handled in variable access */
    /* loadGlobal(465) - handled in variable access */
    int t536 = id;
    int t537 = t536 * 1024;
    int t538 = t536 * 257;
    float t539 = t[11*frameCount + id] == 0.0;
    t[12*frameCount + id] = 0.0;
    if (t539) {
      for (uint _pr542 = 0; _pr542 < 512; _pr542++) {
        float t543 = (float)_pr542;
        float t544 = 6.283185 * t543;
        float t545 = (t544 * 0.0019569471);
        float t546 = metal::cos(t545);
        float t547 = 1.0 - t546;
        float t548 = 0.5 * t547;
        float t549 = (float)t536;
        float t550 = t549 - 511.0;
        float t551 = t550 + t543;
        float t552 = (t551 < 0 || t551 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t551];
        float t553 = (t551 < 0 || t551 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t551];
        int t554 = t537 + _pr542;
        float t555 = t552 * t548;
        memory[4255220 + t554] = t555;
        int t557 = t537 + _pr542;
        int t558 = t557 + 512;
        memory[4255220 + t558] = 0.0;
        int t560 = t537 + _pr542;
        float t561 = t553 * t548;
        memory[21032436 + t560] = t561;
        int t563 = t537 + _pr542;
        int t564 = t563 + 512;
        memory[21032436 + t564] = 0.0;
        memory[25460 + (int)_pr542] = t548;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t568 = 0; t568 < 512; t568++) {
        float t569 = (float)t568;
        float t570 = (t569 - metal::floor(t569 / 2.0) * 2.0);
        float t571 = t570;
        float t572 = (t569 * 0.5);
        float t573 = metal::floor(t572);
        float t574 = t571 * 2.0;
        float t575 = (t573 - metal::floor(t573 / 2.0) * 2.0);
        float t576 = t574 + t575;
        float t577 = (t573 * 0.5);
        float t578 = metal::floor(t577);
        float t579 = t576 * 2.0;
        float t580 = (t578 - metal::floor(t578 / 2.0) * 2.0);
        float t581 = t579 + t580;
        float t582 = (t578 * 0.5);
        float t583 = metal::floor(t582);
        float t584 = t581 * 2.0;
        float t585 = (t583 - metal::floor(t583 / 2.0) * 2.0);
        float t586 = t584 + t585;
        float t587 = (t583 * 0.5);
        float t588 = metal::floor(t587);
        float t589 = t586 * 2.0;
        float t590 = (t588 - metal::floor(t588 / 2.0) * 2.0);
        float t591 = t589 + t590;
        float t592 = (t588 * 0.5);
        float t593 = metal::floor(t592);
        float t594 = t591 * 2.0;
        float t595 = (t593 - metal::floor(t593 / 2.0) * 2.0);
        float t596 = t594 + t595;
        float t597 = (t593 * 0.5);
        float t598 = metal::floor(t597);
        float t599 = t596 * 2.0;
        float t600 = (t598 - metal::floor(t598 / 2.0) * 2.0);
        float t601 = t599 + t600;
        float t602 = (t598 * 0.5);
        float t603 = metal::floor(t602);
        float t604 = t601 * 2.0;
        float t605 = (t603 - metal::floor(t603 / 2.0) * 2.0);
        float t606 = t604 + t605;
        float t607 = (t603 * 0.5);
        float t608 = metal::floor(t607);
        float t609 = t606 * 2.0;
        float t610 = (t608 - metal::floor(t608 / 2.0) * 2.0);
        float t611 = t609 + t610;
        float t612 = (t608 * 0.5);
        float t613 = metal::floor(t612);
        float t614 = (float)t568;
        float t615 = t614 < t611;
        int t616 = (int)t611;
        int t617 = t537 + t568;
        float t618 = memory[4255220 + t617];
        int t619 = t537 + t568;
        int t620 = t619 + 512;
        float t621 = memory[4255220 + t620];
        int t622 = t537 + t616;
        float t623 = memory[4255220 + t622];
        int t624 = t537 + t616;
        int t625 = t624 + 512;
        float t626 = memory[4255220 + t625];
        float t627 = metal::select(t618, t623, t615 > 0.0);
        float t628 = metal::select(t621, t626, t615 > 0.0);
        float t629 = metal::select(t623, t618, t615 > 0.0);
        float t630 = metal::select(t626, t621, t615 > 0.0);
        int t631 = t537 + t568;
        memory[4255220 + t631] = t627;
        int t633 = t537 + t568;
        int t634 = t633 + 512;
        memory[4255220 + t634] = t628;
        int t636 = t537 + t616;
        memory[4255220 + t636] = t629;
        int t638 = t537 + t616;
        int t639 = t638 + 512;
        memory[4255220 + t639] = t630;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr642 = 0; _pr642 < 256; _pr642++) {
        float t643 = (float)_pr642;
        float t644 = t643;
        float t645 = metal::floor(t644);
        float t646 = t645;
        float t647 = t643 - t646;
        float t648 = t645 * 2.0;
        float t649 = t648 + t647;
        float t650 = t649 + 1.0;
        float t651 = -6.283185 * t647;
        float t652 = (t651 * 0.5);
        float t653 = metal::cos(t652);
        float t654 = metal::sin(t652);
        int t655 = (int)t649;
        int t656 = (int)t650;
        int t657 = t537 + t655;
        float t658 = memory[4255220 + t657];
        int t659 = t537 + t655;
        int t660 = t659 + 512;
        float t661 = memory[4255220 + t660];
        int t662 = t537 + t656;
        float t663 = memory[4255220 + t662];
        int t664 = t537 + t656;
        int t665 = t664 + 512;
        float t666 = memory[4255220 + t665];
        float t667 = t653 * t663;
        float t668 = t654 * t666;
        float t669 = t667 - t668;
        float t670 = t653 * t666;
        float t671 = t654 * t663;
        float t672 = t670 + t671;
        int t673 = t537 + t655;
        float t674 = t658 + t669;
        memory[4255220 + t673] = t674;
        int t676 = t537 + t655;
        int t677 = t676 + 512;
        float t678 = t661 + t672;
        memory[4255220 + t677] = t678;
        int t680 = t537 + t656;
        float t681 = t658 - t669;
        memory[4255220 + t680] = t681;
        int t683 = t537 + t656;
        int t684 = t683 + 512;
        float t685 = t661 - t672;
        memory[4255220 + t684] = t685;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr688 = 0; _pr688 < 256; _pr688++) {
        float t689 = (float)_pr688;
        float t690 = (t689 * 0.5);
        float t691 = metal::floor(t690);
        float t692 = t691 * 2.0;
        float t693 = t689 - t692;
        float t694 = t691 * 4.0;
        float t695 = t694 + t693;
        float t696 = t695 + 2.0;
        float t697 = -6.283185 * t693;
        float t698 = (t697 * 0.25);
        float t699 = metal::cos(t698);
        float t700 = metal::sin(t698);
        int t701 = (int)t695;
        int t702 = (int)t696;
        int t703 = t537 + t701;
        float t704 = memory[4255220 + t703];
        int t705 = t537 + t701;
        int t706 = t705 + 512;
        float t707 = memory[4255220 + t706];
        int t708 = t537 + t702;
        float t709 = memory[4255220 + t708];
        int t710 = t537 + t702;
        int t711 = t710 + 512;
        float t712 = memory[4255220 + t711];
        float t713 = t699 * t709;
        float t714 = t700 * t712;
        float t715 = t713 - t714;
        float t716 = t699 * t712;
        float t717 = t700 * t709;
        float t718 = t716 + t717;
        int t719 = t537 + t701;
        float t720 = t704 + t715;
        memory[4255220 + t719] = t720;
        int t722 = t537 + t701;
        int t723 = t722 + 512;
        float t724 = t707 + t718;
        memory[4255220 + t723] = t724;
        int t726 = t537 + t702;
        float t727 = t704 - t715;
        memory[4255220 + t726] = t727;
        int t729 = t537 + t702;
        int t730 = t729 + 512;
        float t731 = t707 - t718;
        memory[4255220 + t730] = t731;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr734 = 0; _pr734 < 256; _pr734++) {
        float t735 = (float)_pr734;
        float t736 = (t735 * 0.25);
        float t737 = metal::floor(t736);
        float t738 = t737 * 4.0;
        float t739 = t735 - t738;
        float t740 = t737 * 8.0;
        float t741 = t740 + t739;
        float t742 = t741 + 4.0;
        float t743 = -6.283185 * t739;
        float t744 = (t743 * 0.125);
        float t745 = metal::cos(t744);
        float t746 = metal::sin(t744);
        int t747 = (int)t741;
        int t748 = (int)t742;
        int t749 = t537 + t747;
        float t750 = memory[4255220 + t749];
        int t751 = t537 + t747;
        int t752 = t751 + 512;
        float t753 = memory[4255220 + t752];
        int t754 = t537 + t748;
        float t755 = memory[4255220 + t754];
        int t756 = t537 + t748;
        int t757 = t756 + 512;
        float t758 = memory[4255220 + t757];
        float t759 = t745 * t755;
        float t760 = t746 * t758;
        float t761 = t759 - t760;
        float t762 = t745 * t758;
        float t763 = t746 * t755;
        float t764 = t762 + t763;
        int t765 = t537 + t747;
        float t766 = t750 + t761;
        memory[4255220 + t765] = t766;
        int t768 = t537 + t747;
        int t769 = t768 + 512;
        float t770 = t753 + t764;
        memory[4255220 + t769] = t770;
        int t772 = t537 + t748;
        float t773 = t750 - t761;
        memory[4255220 + t772] = t773;
        int t775 = t537 + t748;
        int t776 = t775 + 512;
        float t777 = t753 - t764;
        memory[4255220 + t776] = t777;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr780 = 0; _pr780 < 256; _pr780++) {
        float t781 = (float)_pr780;
        float t782 = (t781 * 0.125);
        float t783 = metal::floor(t782);
        float t784 = t783 * 8.0;
        float t785 = t781 - t784;
        float t786 = t783 * 16.0;
        float t787 = t786 + t785;
        float t788 = t787 + 8.0;
        float t789 = -6.283185 * t785;
        float t790 = (t789 * 0.0625);
        float t791 = metal::cos(t790);
        float t792 = metal::sin(t790);
        int t793 = (int)t787;
        int t794 = (int)t788;
        int t795 = t537 + t793;
        float t796 = memory[4255220 + t795];
        int t797 = t537 + t793;
        int t798 = t797 + 512;
        float t799 = memory[4255220 + t798];
        int t800 = t537 + t794;
        float t801 = memory[4255220 + t800];
        int t802 = t537 + t794;
        int t803 = t802 + 512;
        float t804 = memory[4255220 + t803];
        float t805 = t791 * t801;
        float t806 = t792 * t804;
        float t807 = t805 - t806;
        float t808 = t791 * t804;
        float t809 = t792 * t801;
        float t810 = t808 + t809;
        int t811 = t537 + t793;
        float t812 = t796 + t807;
        memory[4255220 + t811] = t812;
        int t814 = t537 + t793;
        int t815 = t814 + 512;
        float t816 = t799 + t810;
        memory[4255220 + t815] = t816;
        int t818 = t537 + t794;
        float t819 = t796 - t807;
        memory[4255220 + t818] = t819;
        int t821 = t537 + t794;
        int t822 = t821 + 512;
        float t823 = t799 - t810;
        memory[4255220 + t822] = t823;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr826 = 0; _pr826 < 256; _pr826++) {
        float t827 = (float)_pr826;
        float t828 = (t827 * 0.0625);
        float t829 = metal::floor(t828);
        float t830 = t829 * 16.0;
        float t831 = t827 - t830;
        float t832 = t829 * 32.0;
        float t833 = t832 + t831;
        float t834 = t833 + 16.0;
        float t835 = -6.283185 * t831;
        float t836 = (t835 * 0.03125);
        float t837 = metal::cos(t836);
        float t838 = metal::sin(t836);
        int t839 = (int)t833;
        int t840 = (int)t834;
        int t841 = t537 + t839;
        float t842 = memory[4255220 + t841];
        int t843 = t537 + t839;
        int t844 = t843 + 512;
        float t845 = memory[4255220 + t844];
        int t846 = t537 + t840;
        float t847 = memory[4255220 + t846];
        int t848 = t537 + t840;
        int t849 = t848 + 512;
        float t850 = memory[4255220 + t849];
        float t851 = t837 * t847;
        float t852 = t838 * t850;
        float t853 = t851 - t852;
        float t854 = t837 * t850;
        float t855 = t838 * t847;
        float t856 = t854 + t855;
        int t857 = t537 + t839;
        float t858 = t842 + t853;
        memory[4255220 + t857] = t858;
        int t860 = t537 + t839;
        int t861 = t860 + 512;
        float t862 = t845 + t856;
        memory[4255220 + t861] = t862;
        int t864 = t537 + t840;
        float t865 = t842 - t853;
        memory[4255220 + t864] = t865;
        int t867 = t537 + t840;
        int t868 = t867 + 512;
        float t869 = t845 - t856;
        memory[4255220 + t868] = t869;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr872 = 0; _pr872 < 256; _pr872++) {
        float t873 = (float)_pr872;
        float t874 = (t873 * 0.03125);
        float t875 = metal::floor(t874);
        float t876 = t875 * 32.0;
        float t877 = t873 - t876;
        float t878 = t875 * 64.0;
        float t879 = t878 + t877;
        float t880 = t879 + 32.0;
        float t881 = -6.283185 * t877;
        float t882 = (t881 * 0.015625);
        float t883 = metal::cos(t882);
        float t884 = metal::sin(t882);
        int t885 = (int)t879;
        int t886 = (int)t880;
        int t887 = t537 + t885;
        float t888 = memory[4255220 + t887];
        int t889 = t537 + t885;
        int t890 = t889 + 512;
        float t891 = memory[4255220 + t890];
        int t892 = t537 + t886;
        float t893 = memory[4255220 + t892];
        int t894 = t537 + t886;
        int t895 = t894 + 512;
        float t896 = memory[4255220 + t895];
        float t897 = t883 * t893;
        float t898 = t884 * t896;
        float t899 = t897 - t898;
        float t900 = t883 * t896;
        float t901 = t884 * t893;
        float t902 = t900 + t901;
        int t903 = t537 + t885;
        float t904 = t888 + t899;
        memory[4255220 + t903] = t904;
        int t906 = t537 + t885;
        int t907 = t906 + 512;
        float t908 = t891 + t902;
        memory[4255220 + t907] = t908;
        int t910 = t537 + t886;
        float t911 = t888 - t899;
        memory[4255220 + t910] = t911;
        int t913 = t537 + t886;
        int t914 = t913 + 512;
        float t915 = t891 - t902;
        memory[4255220 + t914] = t915;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr918 = 0; _pr918 < 256; _pr918++) {
        float t919 = (float)_pr918;
        float t920 = (t919 * 0.015625);
        float t921 = metal::floor(t920);
        float t922 = t921 * 64.0;
        float t923 = t919 - t922;
        float t924 = t921 * 128.0;
        float t925 = t924 + t923;
        float t926 = t925 + 64.0;
        float t927 = -6.283185 * t923;
        float t928 = (t927 * 0.0078125);
        float t929 = metal::cos(t928);
        float t930 = metal::sin(t928);
        int t931 = (int)t925;
        int t932 = (int)t926;
        int t933 = t537 + t931;
        float t934 = memory[4255220 + t933];
        int t935 = t537 + t931;
        int t936 = t935 + 512;
        float t937 = memory[4255220 + t936];
        int t938 = t537 + t932;
        float t939 = memory[4255220 + t938];
        int t940 = t537 + t932;
        int t941 = t940 + 512;
        float t942 = memory[4255220 + t941];
        float t943 = t929 * t939;
        float t944 = t930 * t942;
        float t945 = t943 - t944;
        float t946 = t929 * t942;
        float t947 = t930 * t939;
        float t948 = t946 + t947;
        int t949 = t537 + t931;
        float t950 = t934 + t945;
        memory[4255220 + t949] = t950;
        int t952 = t537 + t931;
        int t953 = t952 + 512;
        float t954 = t937 + t948;
        memory[4255220 + t953] = t954;
        int t956 = t537 + t932;
        float t957 = t934 - t945;
        memory[4255220 + t956] = t957;
        int t959 = t537 + t932;
        int t960 = t959 + 512;
        float t961 = t937 - t948;
        memory[4255220 + t960] = t961;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr964 = 0; _pr964 < 256; _pr964++) {
        float t965 = (float)_pr964;
        float t966 = (t965 * 0.0078125);
        float t967 = metal::floor(t966);
        float t968 = t967 * 128.0;
        float t969 = t965 - t968;
        float t970 = t967 * 256.0;
        float t971 = t970 + t969;
        float t972 = t971 + 128.0;
        float t973 = -6.283185 * t969;
        float t974 = (t973 * 0.00390625);
        float t975 = metal::cos(t974);
        float t976 = metal::sin(t974);
        int t977 = (int)t971;
        int t978 = (int)t972;
        int t979 = t537 + t977;
        float t980 = memory[4255220 + t979];
        int t981 = t537 + t977;
        int t982 = t981 + 512;
        float t983 = memory[4255220 + t982];
        int t984 = t537 + t978;
        float t985 = memory[4255220 + t984];
        int t986 = t537 + t978;
        int t987 = t986 + 512;
        float t988 = memory[4255220 + t987];
        float t989 = t975 * t985;
        float t990 = t976 * t988;
        float t991 = t989 - t990;
        float t992 = t975 * t988;
        float t993 = t976 * t985;
        float t994 = t992 + t993;
        int t995 = t537 + t977;
        float t996 = t980 + t991;
        memory[4255220 + t995] = t996;
        int t998 = t537 + t977;
        int t999 = t998 + 512;
        float t1000 = t983 + t994;
        memory[4255220 + t999] = t1000;
        int t1002 = t537 + t978;
        float t1003 = t980 - t991;
        memory[4255220 + t1002] = t1003;
        int t1005 = t537 + t978;
        int t1006 = t1005 + 512;
        float t1007 = t983 - t994;
        memory[4255220 + t1006] = t1007;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1010 = 0; _pr1010 < 256; _pr1010++) {
        float t1011 = (float)_pr1010;
        float t1012 = (t1011 * 0.00390625);
        float t1013 = metal::floor(t1012);
        float t1014 = t1013 * 256.0;
        float t1015 = t1011 - t1014;
        float t1016 = t1013 * 512.0;
        float t1017 = t1016 + t1015;
        float t1018 = t1017 + 256.0;
        float t1019 = -6.283185 * t1015;
        float t1020 = (t1019 * 0.001953125);
        float t1021 = metal::cos(t1020);
        float t1022 = metal::sin(t1020);
        int t1023 = (int)t1017;
        int t1024 = (int)t1018;
        int t1025 = t537 + t1023;
        float t1026 = memory[4255220 + t1025];
        int t1027 = t537 + t1023;
        int t1028 = t1027 + 512;
        float t1029 = memory[4255220 + t1028];
        int t1030 = t537 + t1024;
        float t1031 = memory[4255220 + t1030];
        int t1032 = t537 + t1024;
        int t1033 = t1032 + 512;
        float t1034 = memory[4255220 + t1033];
        float t1035 = t1021 * t1031;
        float t1036 = t1022 * t1034;
        float t1037 = t1035 - t1036;
        float t1038 = t1021 * t1034;
        float t1039 = t1022 * t1031;
        float t1040 = t1038 + t1039;
        int t1041 = t537 + t1023;
        float t1042 = t1026 + t1037;
        memory[4255220 + t1041] = t1042;
        int t1044 = t537 + t1023;
        int t1045 = t1044 + 512;
        float t1046 = t1029 + t1040;
        memory[4255220 + t1045] = t1046;
        int t1048 = t537 + t1024;
        float t1049 = t1026 - t1037;
        memory[4255220 + t1048] = t1049;
        int t1051 = t537 + t1024;
        int t1052 = t1051 + 512;
        float t1053 = t1029 - t1040;
        memory[4255220 + t1052] = t1053;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1056 = 0; t1056 < 512; t1056++) {
        float t1057 = (float)t1056;
        float t1058 = (t1057 - metal::floor(t1057 / 2.0) * 2.0);
        float t1059 = t1058;
        float t1060 = (t1057 * 0.5);
        float t1061 = metal::floor(t1060);
        float t1062 = t1059 * 2.0;
        float t1063 = (t1061 - metal::floor(t1061 / 2.0) * 2.0);
        float t1064 = t1062 + t1063;
        float t1065 = (t1061 * 0.5);
        float t1066 = metal::floor(t1065);
        float t1067 = t1064 * 2.0;
        float t1068 = (t1066 - metal::floor(t1066 / 2.0) * 2.0);
        float t1069 = t1067 + t1068;
        float t1070 = (t1066 * 0.5);
        float t1071 = metal::floor(t1070);
        float t1072 = t1069 * 2.0;
        float t1073 = (t1071 - metal::floor(t1071 / 2.0) * 2.0);
        float t1074 = t1072 + t1073;
        float t1075 = (t1071 * 0.5);
        float t1076 = metal::floor(t1075);
        float t1077 = t1074 * 2.0;
        float t1078 = (t1076 - metal::floor(t1076 / 2.0) * 2.0);
        float t1079 = t1077 + t1078;
        float t1080 = (t1076 * 0.5);
        float t1081 = metal::floor(t1080);
        float t1082 = t1079 * 2.0;
        float t1083 = (t1081 - metal::floor(t1081 / 2.0) * 2.0);
        float t1084 = t1082 + t1083;
        float t1085 = (t1081 * 0.5);
        float t1086 = metal::floor(t1085);
        float t1087 = t1084 * 2.0;
        float t1088 = (t1086 - metal::floor(t1086 / 2.0) * 2.0);
        float t1089 = t1087 + t1088;
        float t1090 = (t1086 * 0.5);
        float t1091 = metal::floor(t1090);
        float t1092 = t1089 * 2.0;
        float t1093 = (t1091 - metal::floor(t1091 / 2.0) * 2.0);
        float t1094 = t1092 + t1093;
        float t1095 = (t1091 * 0.5);
        float t1096 = metal::floor(t1095);
        float t1097 = t1094 * 2.0;
        float t1098 = (t1096 - metal::floor(t1096 / 2.0) * 2.0);
        float t1099 = t1097 + t1098;
        float t1100 = (t1096 * 0.5);
        float t1101 = metal::floor(t1100);
        float t1102 = (float)t1056;
        float t1103 = t1102 < t1099;
        int t1104 = (int)t1099;
        int t1105 = t537 + t1056;
        float t1106 = memory[21032436 + t1105];
        int t1107 = t537 + t1056;
        int t1108 = t1107 + 512;
        float t1109 = memory[21032436 + t1108];
        int t1110 = t537 + t1104;
        float t1111 = memory[21032436 + t1110];
        int t1112 = t537 + t1104;
        int t1113 = t1112 + 512;
        float t1114 = memory[21032436 + t1113];
        float t1115 = metal::select(t1106, t1111, t1103 > 0.0);
        float t1116 = metal::select(t1109, t1114, t1103 > 0.0);
        float t1117 = metal::select(t1111, t1106, t1103 > 0.0);
        float t1118 = metal::select(t1114, t1109, t1103 > 0.0);
        int t1119 = t537 + t1056;
        memory[21032436 + t1119] = t1115;
        int t1121 = t537 + t1056;
        int t1122 = t1121 + 512;
        memory[21032436 + t1122] = t1116;
        int t1124 = t537 + t1104;
        memory[21032436 + t1124] = t1117;
        int t1126 = t537 + t1104;
        int t1127 = t1126 + 512;
        memory[21032436 + t1127] = t1118;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1130 = 0; _pr1130 < 256; _pr1130++) {
        float t1131 = (float)_pr1130;
        float t1132 = t1131;
        float t1133 = metal::floor(t1132);
        float t1134 = t1133;
        float t1135 = t1131 - t1134;
        float t1136 = t1133 * 2.0;
        float t1137 = t1136 + t1135;
        float t1138 = t1137 + 1.0;
        float t1139 = -6.283185 * t1135;
        float t1140 = (t1139 * 0.5);
        float t1141 = metal::cos(t1140);
        float t1142 = metal::sin(t1140);
        int t1143 = (int)t1137;
        int t1144 = (int)t1138;
        int t1145 = t537 + t1143;
        float t1146 = memory[21032436 + t1145];
        int t1147 = t537 + t1143;
        int t1148 = t1147 + 512;
        float t1149 = memory[21032436 + t1148];
        int t1150 = t537 + t1144;
        float t1151 = memory[21032436 + t1150];
        int t1152 = t537 + t1144;
        int t1153 = t1152 + 512;
        float t1154 = memory[21032436 + t1153];
        float t1155 = t1141 * t1151;
        float t1156 = t1142 * t1154;
        float t1157 = t1155 - t1156;
        float t1158 = t1141 * t1154;
        float t1159 = t1142 * t1151;
        float t1160 = t1158 + t1159;
        int t1161 = t537 + t1143;
        float t1162 = t1146 + t1157;
        memory[21032436 + t1161] = t1162;
        int t1164 = t537 + t1143;
        int t1165 = t1164 + 512;
        float t1166 = t1149 + t1160;
        memory[21032436 + t1165] = t1166;
        int t1168 = t537 + t1144;
        float t1169 = t1146 - t1157;
        memory[21032436 + t1168] = t1169;
        int t1171 = t537 + t1144;
        int t1172 = t1171 + 512;
        float t1173 = t1149 - t1160;
        memory[21032436 + t1172] = t1173;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1176 = 0; _pr1176 < 256; _pr1176++) {
        float t1177 = (float)_pr1176;
        float t1178 = (t1177 * 0.5);
        float t1179 = metal::floor(t1178);
        float t1180 = t1179 * 2.0;
        float t1181 = t1177 - t1180;
        float t1182 = t1179 * 4.0;
        float t1183 = t1182 + t1181;
        float t1184 = t1183 + 2.0;
        float t1185 = -6.283185 * t1181;
        float t1186 = (t1185 * 0.25);
        float t1187 = metal::cos(t1186);
        float t1188 = metal::sin(t1186);
        int t1189 = (int)t1183;
        int t1190 = (int)t1184;
        int t1191 = t537 + t1189;
        float t1192 = memory[21032436 + t1191];
        int t1193 = t537 + t1189;
        int t1194 = t1193 + 512;
        float t1195 = memory[21032436 + t1194];
        int t1196 = t537 + t1190;
        float t1197 = memory[21032436 + t1196];
        int t1198 = t537 + t1190;
        int t1199 = t1198 + 512;
        float t1200 = memory[21032436 + t1199];
        float t1201 = t1187 * t1197;
        float t1202 = t1188 * t1200;
        float t1203 = t1201 - t1202;
        float t1204 = t1187 * t1200;
        float t1205 = t1188 * t1197;
        float t1206 = t1204 + t1205;
        int t1207 = t537 + t1189;
        float t1208 = t1192 + t1203;
        memory[21032436 + t1207] = t1208;
        int t1210 = t537 + t1189;
        int t1211 = t1210 + 512;
        float t1212 = t1195 + t1206;
        memory[21032436 + t1211] = t1212;
        int t1214 = t537 + t1190;
        float t1215 = t1192 - t1203;
        memory[21032436 + t1214] = t1215;
        int t1217 = t537 + t1190;
        int t1218 = t1217 + 512;
        float t1219 = t1195 - t1206;
        memory[21032436 + t1218] = t1219;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1222 = 0; _pr1222 < 256; _pr1222++) {
        float t1223 = (float)_pr1222;
        float t1224 = (t1223 * 0.25);
        float t1225 = metal::floor(t1224);
        float t1226 = t1225 * 4.0;
        float t1227 = t1223 - t1226;
        float t1228 = t1225 * 8.0;
        float t1229 = t1228 + t1227;
        float t1230 = t1229 + 4.0;
        float t1231 = -6.283185 * t1227;
        float t1232 = (t1231 * 0.125);
        float t1233 = metal::cos(t1232);
        float t1234 = metal::sin(t1232);
        int t1235 = (int)t1229;
        int t1236 = (int)t1230;
        int t1237 = t537 + t1235;
        float t1238 = memory[21032436 + t1237];
        int t1239 = t537 + t1235;
        int t1240 = t1239 + 512;
        float t1241 = memory[21032436 + t1240];
        int t1242 = t537 + t1236;
        float t1243 = memory[21032436 + t1242];
        int t1244 = t537 + t1236;
        int t1245 = t1244 + 512;
        float t1246 = memory[21032436 + t1245];
        float t1247 = t1233 * t1243;
        float t1248 = t1234 * t1246;
        float t1249 = t1247 - t1248;
        float t1250 = t1233 * t1246;
        float t1251 = t1234 * t1243;
        float t1252 = t1250 + t1251;
        int t1253 = t537 + t1235;
        float t1254 = t1238 + t1249;
        memory[21032436 + t1253] = t1254;
        int t1256 = t537 + t1235;
        int t1257 = t1256 + 512;
        float t1258 = t1241 + t1252;
        memory[21032436 + t1257] = t1258;
        int t1260 = t537 + t1236;
        float t1261 = t1238 - t1249;
        memory[21032436 + t1260] = t1261;
        int t1263 = t537 + t1236;
        int t1264 = t1263 + 512;
        float t1265 = t1241 - t1252;
        memory[21032436 + t1264] = t1265;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1268 = 0; _pr1268 < 256; _pr1268++) {
        float t1269 = (float)_pr1268;
        float t1270 = (t1269 * 0.125);
        float t1271 = metal::floor(t1270);
        float t1272 = t1271 * 8.0;
        float t1273 = t1269 - t1272;
        float t1274 = t1271 * 16.0;
        float t1275 = t1274 + t1273;
        float t1276 = t1275 + 8.0;
        float t1277 = -6.283185 * t1273;
        float t1278 = (t1277 * 0.0625);
        float t1279 = metal::cos(t1278);
        float t1280 = metal::sin(t1278);
        int t1281 = (int)t1275;
        int t1282 = (int)t1276;
        int t1283 = t537 + t1281;
        float t1284 = memory[21032436 + t1283];
        int t1285 = t537 + t1281;
        int t1286 = t1285 + 512;
        float t1287 = memory[21032436 + t1286];
        int t1288 = t537 + t1282;
        float t1289 = memory[21032436 + t1288];
        int t1290 = t537 + t1282;
        int t1291 = t1290 + 512;
        float t1292 = memory[21032436 + t1291];
        float t1293 = t1279 * t1289;
        float t1294 = t1280 * t1292;
        float t1295 = t1293 - t1294;
        float t1296 = t1279 * t1292;
        float t1297 = t1280 * t1289;
        float t1298 = t1296 + t1297;
        int t1299 = t537 + t1281;
        float t1300 = t1284 + t1295;
        memory[21032436 + t1299] = t1300;
        int t1302 = t537 + t1281;
        int t1303 = t1302 + 512;
        float t1304 = t1287 + t1298;
        memory[21032436 + t1303] = t1304;
        int t1306 = t537 + t1282;
        float t1307 = t1284 - t1295;
        memory[21032436 + t1306] = t1307;
        int t1309 = t537 + t1282;
        int t1310 = t1309 + 512;
        float t1311 = t1287 - t1298;
        memory[21032436 + t1310] = t1311;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1314 = 0; _pr1314 < 256; _pr1314++) {
        float t1315 = (float)_pr1314;
        float t1316 = (t1315 * 0.0625);
        float t1317 = metal::floor(t1316);
        float t1318 = t1317 * 16.0;
        float t1319 = t1315 - t1318;
        float t1320 = t1317 * 32.0;
        float t1321 = t1320 + t1319;
        float t1322 = t1321 + 16.0;
        float t1323 = -6.283185 * t1319;
        float t1324 = (t1323 * 0.03125);
        float t1325 = metal::cos(t1324);
        float t1326 = metal::sin(t1324);
        int t1327 = (int)t1321;
        int t1328 = (int)t1322;
        int t1329 = t537 + t1327;
        float t1330 = memory[21032436 + t1329];
        int t1331 = t537 + t1327;
        int t1332 = t1331 + 512;
        float t1333 = memory[21032436 + t1332];
        int t1334 = t537 + t1328;
        float t1335 = memory[21032436 + t1334];
        int t1336 = t537 + t1328;
        int t1337 = t1336 + 512;
        float t1338 = memory[21032436 + t1337];
        float t1339 = t1325 * t1335;
        float t1340 = t1326 * t1338;
        float t1341 = t1339 - t1340;
        float t1342 = t1325 * t1338;
        float t1343 = t1326 * t1335;
        float t1344 = t1342 + t1343;
        int t1345 = t537 + t1327;
        float t1346 = t1330 + t1341;
        memory[21032436 + t1345] = t1346;
        int t1348 = t537 + t1327;
        int t1349 = t1348 + 512;
        float t1350 = t1333 + t1344;
        memory[21032436 + t1349] = t1350;
        int t1352 = t537 + t1328;
        float t1353 = t1330 - t1341;
        memory[21032436 + t1352] = t1353;
        int t1355 = t537 + t1328;
        int t1356 = t1355 + 512;
        float t1357 = t1333 - t1344;
        memory[21032436 + t1356] = t1357;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1360 = 0; _pr1360 < 256; _pr1360++) {
        float t1361 = (float)_pr1360;
        float t1362 = (t1361 * 0.03125);
        float t1363 = metal::floor(t1362);
        float t1364 = t1363 * 32.0;
        float t1365 = t1361 - t1364;
        float t1366 = t1363 * 64.0;
        float t1367 = t1366 + t1365;
        float t1368 = t1367 + 32.0;
        float t1369 = -6.283185 * t1365;
        float t1370 = (t1369 * 0.015625);
        float t1371 = metal::cos(t1370);
        float t1372 = metal::sin(t1370);
        int t1373 = (int)t1367;
        int t1374 = (int)t1368;
        int t1375 = t537 + t1373;
        float t1376 = memory[21032436 + t1375];
        int t1377 = t537 + t1373;
        int t1378 = t1377 + 512;
        float t1379 = memory[21032436 + t1378];
        int t1380 = t537 + t1374;
        float t1381 = memory[21032436 + t1380];
        int t1382 = t537 + t1374;
        int t1383 = t1382 + 512;
        float t1384 = memory[21032436 + t1383];
        float t1385 = t1371 * t1381;
        float t1386 = t1372 * t1384;
        float t1387 = t1385 - t1386;
        float t1388 = t1371 * t1384;
        float t1389 = t1372 * t1381;
        float t1390 = t1388 + t1389;
        int t1391 = t537 + t1373;
        float t1392 = t1376 + t1387;
        memory[21032436 + t1391] = t1392;
        int t1394 = t537 + t1373;
        int t1395 = t1394 + 512;
        float t1396 = t1379 + t1390;
        memory[21032436 + t1395] = t1396;
        int t1398 = t537 + t1374;
        float t1399 = t1376 - t1387;
        memory[21032436 + t1398] = t1399;
        int t1401 = t537 + t1374;
        int t1402 = t1401 + 512;
        float t1403 = t1379 - t1390;
        memory[21032436 + t1402] = t1403;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1406 = 0; _pr1406 < 256; _pr1406++) {
        float t1407 = (float)_pr1406;
        float t1408 = (t1407 * 0.015625);
        float t1409 = metal::floor(t1408);
        float t1410 = t1409 * 64.0;
        float t1411 = t1407 - t1410;
        float t1412 = t1409 * 128.0;
        float t1413 = t1412 + t1411;
        float t1414 = t1413 + 64.0;
        float t1415 = -6.283185 * t1411;
        float t1416 = (t1415 * 0.0078125);
        float t1417 = metal::cos(t1416);
        float t1418 = metal::sin(t1416);
        int t1419 = (int)t1413;
        int t1420 = (int)t1414;
        int t1421 = t537 + t1419;
        float t1422 = memory[21032436 + t1421];
        int t1423 = t537 + t1419;
        int t1424 = t1423 + 512;
        float t1425 = memory[21032436 + t1424];
        int t1426 = t537 + t1420;
        float t1427 = memory[21032436 + t1426];
        int t1428 = t537 + t1420;
        int t1429 = t1428 + 512;
        float t1430 = memory[21032436 + t1429];
        float t1431 = t1417 * t1427;
        float t1432 = t1418 * t1430;
        float t1433 = t1431 - t1432;
        float t1434 = t1417 * t1430;
        float t1435 = t1418 * t1427;
        float t1436 = t1434 + t1435;
        int t1437 = t537 + t1419;
        float t1438 = t1422 + t1433;
        memory[21032436 + t1437] = t1438;
        int t1440 = t537 + t1419;
        int t1441 = t1440 + 512;
        float t1442 = t1425 + t1436;
        memory[21032436 + t1441] = t1442;
        int t1444 = t537 + t1420;
        float t1445 = t1422 - t1433;
        memory[21032436 + t1444] = t1445;
        int t1447 = t537 + t1420;
        int t1448 = t1447 + 512;
        float t1449 = t1425 - t1436;
        memory[21032436 + t1448] = t1449;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1452 = 0; _pr1452 < 256; _pr1452++) {
        float t1453 = (float)_pr1452;
        float t1454 = (t1453 * 0.0078125);
        float t1455 = metal::floor(t1454);
        float t1456 = t1455 * 128.0;
        float t1457 = t1453 - t1456;
        float t1458 = t1455 * 256.0;
        float t1459 = t1458 + t1457;
        float t1460 = t1459 + 128.0;
        float t1461 = -6.283185 * t1457;
        float t1462 = (t1461 * 0.00390625);
        float t1463 = metal::cos(t1462);
        float t1464 = metal::sin(t1462);
        int t1465 = (int)t1459;
        int t1466 = (int)t1460;
        int t1467 = t537 + t1465;
        float t1468 = memory[21032436 + t1467];
        int t1469 = t537 + t1465;
        int t1470 = t1469 + 512;
        float t1471 = memory[21032436 + t1470];
        int t1472 = t537 + t1466;
        float t1473 = memory[21032436 + t1472];
        int t1474 = t537 + t1466;
        int t1475 = t1474 + 512;
        float t1476 = memory[21032436 + t1475];
        float t1477 = t1463 * t1473;
        float t1478 = t1464 * t1476;
        float t1479 = t1477 - t1478;
        float t1480 = t1463 * t1476;
        float t1481 = t1464 * t1473;
        float t1482 = t1480 + t1481;
        int t1483 = t537 + t1465;
        float t1484 = t1468 + t1479;
        memory[21032436 + t1483] = t1484;
        int t1486 = t537 + t1465;
        int t1487 = t1486 + 512;
        float t1488 = t1471 + t1482;
        memory[21032436 + t1487] = t1488;
        int t1490 = t537 + t1466;
        float t1491 = t1468 - t1479;
        memory[21032436 + t1490] = t1491;
        int t1493 = t537 + t1466;
        int t1494 = t1493 + 512;
        float t1495 = t1471 - t1482;
        memory[21032436 + t1494] = t1495;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1498 = 0; _pr1498 < 256; _pr1498++) {
        float t1499 = (float)_pr1498;
        float t1500 = (t1499 * 0.00390625);
        float t1501 = metal::floor(t1500);
        float t1502 = t1501 * 256.0;
        float t1503 = t1499 - t1502;
        float t1504 = t1501 * 512.0;
        float t1505 = t1504 + t1503;
        float t1506 = t1505 + 256.0;
        float t1507 = -6.283185 * t1503;
        float t1508 = (t1507 * 0.001953125);
        float t1509 = metal::cos(t1508);
        float t1510 = metal::sin(t1508);
        int t1511 = (int)t1505;
        int t1512 = (int)t1506;
        int t1513 = t537 + t1511;
        float t1514 = memory[21032436 + t1513];
        int t1515 = t537 + t1511;
        int t1516 = t1515 + 512;
        float t1517 = memory[21032436 + t1516];
        int t1518 = t537 + t1512;
        float t1519 = memory[21032436 + t1518];
        int t1520 = t537 + t1512;
        int t1521 = t1520 + 512;
        float t1522 = memory[21032436 + t1521];
        float t1523 = t1509 * t1519;
        float t1524 = t1510 * t1522;
        float t1525 = t1523 - t1524;
        float t1526 = t1509 * t1522;
        float t1527 = t1510 * t1519;
        float t1528 = t1526 + t1527;
        int t1529 = t537 + t1511;
        float t1530 = t1514 + t1525;
        memory[21032436 + t1529] = t1530;
        int t1532 = t537 + t1511;
        int t1533 = t1532 + 512;
        float t1534 = t1517 + t1528;
        memory[21032436 + t1533] = t1534;
        int t1536 = t537 + t1512;
        float t1537 = t1514 - t1525;
        memory[21032436 + t1536] = t1537;
        int t1539 = t537 + t1512;
        int t1540 = t1539 + 512;
        float t1541 = t1517 - t1528;
        memory[21032436 + t1540] = t1541;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1544 = 0; _pr1544 < 257; _pr1544++) {
        int t1545 = t537 + _pr1544;
        float t1546 = memory[4255220 + t1545];
        int t1547 = t537 + _pr1544;
        int t1548 = t1547 + 512;
        float t1549 = memory[4255220 + t1548];
        float t1550 = t1546 * t1546;
        float t1551 = t1549 * t1549;
        float t1552 = t1550 + t1551;
        float t1553 = metal::sqrt(t1552);
        int t1554 = t538 + _pr1544;
        memory[37809652 + t1554] = t1553;
        int t1556 = t537 + _pr1544;
        float t1557 = memory[21032436 + t1556];
        int t1558 = t537 + _pr1544;
        int t1559 = t1558 + 512;
        float t1560 = memory[21032436 + t1559];
        float t1561 = t1557 * t1557;
        float t1562 = t1560 * t1560;
        float t1563 = t1561 + t1562;
        float t1564 = metal::sqrt(t1563);
        int t1565 = t538 + _pr1544;
        memory[42020340 + t1565] = t1564;
        float t1567 = t1553 - t1564;
        int t1568 = t538 + _pr1544;
        float t1569 = t1567 * t1567;
        memory[46231028 + t1568] = t1569;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1572 = 0; t1572 < 257; t1572++) {
        int t1573 = t538 + t1572;
        float t1574 = memory[46231028 + t1573];
        float t1575 = t[12*frameCount + id] + t1574;
        t[12*frameCount + id] = t1575;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1579), value: global(1579)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(540) - handled in variable access */
    float t1578 = (t[12*frameCount + id] * 6.1035156e-05);
    t[13*frameCount + id] = t1578;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1580), value: global(1580)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[14*frameCount + i] = memory[599948663];
      float t1581 = t[14*frameCount + i] + 1.0;
      float t1582 = metal::select(t1581, 0.0, 0.0 > 0.0);
      float t1583 = t1582;
      float t1584 = (t1583 * 0.00390625);
      float t1585 = metal::floor(t1584);
      float t1586 = t1585 * 256.0;
      float t1587 = t1582 - t1586;
      memory[599948663] = t1587;
      float t1589 = t1587 >= 256.0;
      if (t1589) {
        float t1591 = t1587 - 256.0;
        memory[599948663] = t1591;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1601), value: global(1601)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1580) - handled in variable access */
    /* loadGlobal(518) - handled in variable access */
    /* loadGlobal(465) - handled in variable access */
    int t1597 = id;
    int t1598 = t1597 * 2048;
    int t1599 = t1597 * 513;
    float t1600 = t[14*frameCount + id] == 0.0;
    t[15*frameCount + id] = 0.0;
    if (t1600) {
      for (uint _pr1603 = 0; _pr1603 < 1024; _pr1603++) {
        float t1604 = (float)_pr1603;
        float t1605 = 6.283185 * t1604;
        float t1606 = (t1605 * 0.0009775171);
        float t1607 = metal::cos(t1606);
        float t1608 = 1.0 - t1607;
        float t1609 = 0.5 * t1608;
        float t1610 = (float)t1597;
        float t1611 = t1610 - 1023.0;
        float t1612 = t1611 + t1604;
        float t1613 = (t1612 < 0 || t1612 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t1612];
        float t1614 = (t1612 < 0 || t1612 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t1612];
        int t1615 = t1598 + _pr1603;
        float t1616 = t1613 * t1609;
        memory[50441716 + t1615] = t1616;
        int t1618 = t1598 + _pr1603;
        int t1619 = t1618 + 1024;
        memory[50441716 + t1619] = 0.0;
        int t1621 = t1598 + _pr1603;
        float t1622 = t1614 * t1609;
        memory[83996148 + t1621] = t1622;
        int t1624 = t1598 + _pr1603;
        int t1625 = t1624 + 1024;
        memory[83996148 + t1625] = 0.0;
        memory[44980 + (int)_pr1603] = t1609;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1629 = 0; t1629 < 1024; t1629++) {
        float t1630 = (float)t1629;
        float t1631 = (t1630 - metal::floor(t1630 / 2.0) * 2.0);
        float t1632 = t1631;
        float t1633 = (t1630 * 0.5);
        float t1634 = metal::floor(t1633);
        float t1635 = t1632 * 2.0;
        float t1636 = (t1634 - metal::floor(t1634 / 2.0) * 2.0);
        float t1637 = t1635 + t1636;
        float t1638 = (t1634 * 0.5);
        float t1639 = metal::floor(t1638);
        float t1640 = t1637 * 2.0;
        float t1641 = (t1639 - metal::floor(t1639 / 2.0) * 2.0);
        float t1642 = t1640 + t1641;
        float t1643 = (t1639 * 0.5);
        float t1644 = metal::floor(t1643);
        float t1645 = t1642 * 2.0;
        float t1646 = (t1644 - metal::floor(t1644 / 2.0) * 2.0);
        float t1647 = t1645 + t1646;
        float t1648 = (t1644 * 0.5);
        float t1649 = metal::floor(t1648);
        float t1650 = t1647 * 2.0;
        float t1651 = (t1649 - metal::floor(t1649 / 2.0) * 2.0);
        float t1652 = t1650 + t1651;
        float t1653 = (t1649 * 0.5);
        float t1654 = metal::floor(t1653);
        float t1655 = t1652 * 2.0;
        float t1656 = (t1654 - metal::floor(t1654 / 2.0) * 2.0);
        float t1657 = t1655 + t1656;
        float t1658 = (t1654 * 0.5);
        float t1659 = metal::floor(t1658);
        float t1660 = t1657 * 2.0;
        float t1661 = (t1659 - metal::floor(t1659 / 2.0) * 2.0);
        float t1662 = t1660 + t1661;
        float t1663 = (t1659 * 0.5);
        float t1664 = metal::floor(t1663);
        float t1665 = t1662 * 2.0;
        float t1666 = (t1664 - metal::floor(t1664 / 2.0) * 2.0);
        float t1667 = t1665 + t1666;
        float t1668 = (t1664 * 0.5);
        float t1669 = metal::floor(t1668);
        float t1670 = t1667 * 2.0;
        float t1671 = (t1669 - metal::floor(t1669 / 2.0) * 2.0);
        float t1672 = t1670 + t1671;
        float t1673 = (t1669 * 0.5);
        float t1674 = metal::floor(t1673);
        float t1675 = t1672 * 2.0;
        float t1676 = (t1674 - metal::floor(t1674 / 2.0) * 2.0);
        float t1677 = t1675 + t1676;
        float t1678 = (t1674 * 0.5);
        float t1679 = metal::floor(t1678);
        float t1680 = (float)t1629;
        float t1681 = t1680 < t1677;
        int t1682 = (int)t1677;
        int t1683 = t1598 + t1629;
        float t1684 = memory[50441716 + t1683];
        int t1685 = t1598 + t1629;
        int t1686 = t1685 + 1024;
        float t1687 = memory[50441716 + t1686];
        int t1688 = t1598 + t1682;
        float t1689 = memory[50441716 + t1688];
        int t1690 = t1598 + t1682;
        int t1691 = t1690 + 1024;
        float t1692 = memory[50441716 + t1691];
        float t1693 = metal::select(t1684, t1689, t1681 > 0.0);
        float t1694 = metal::select(t1687, t1692, t1681 > 0.0);
        float t1695 = metal::select(t1689, t1684, t1681 > 0.0);
        float t1696 = metal::select(t1692, t1687, t1681 > 0.0);
        int t1697 = t1598 + t1629;
        memory[50441716 + t1697] = t1693;
        int t1699 = t1598 + t1629;
        int t1700 = t1699 + 1024;
        memory[50441716 + t1700] = t1694;
        int t1702 = t1598 + t1682;
        memory[50441716 + t1702] = t1695;
        int t1704 = t1598 + t1682;
        int t1705 = t1704 + 1024;
        memory[50441716 + t1705] = t1696;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1708 = 0; _pr1708 < 512; _pr1708++) {
        float t1709 = (float)_pr1708;
        float t1710 = t1709;
        float t1711 = metal::floor(t1710);
        float t1712 = t1711;
        float t1713 = t1709 - t1712;
        float t1714 = t1711 * 2.0;
        float t1715 = t1714 + t1713;
        float t1716 = t1715 + 1.0;
        float t1717 = -6.283185 * t1713;
        float t1718 = (t1717 * 0.5);
        float t1719 = metal::cos(t1718);
        float t1720 = metal::sin(t1718);
        int t1721 = (int)t1715;
        int t1722 = (int)t1716;
        int t1723 = t1598 + t1721;
        float t1724 = memory[50441716 + t1723];
        int t1725 = t1598 + t1721;
        int t1726 = t1725 + 1024;
        float t1727 = memory[50441716 + t1726];
        int t1728 = t1598 + t1722;
        float t1729 = memory[50441716 + t1728];
        int t1730 = t1598 + t1722;
        int t1731 = t1730 + 1024;
        float t1732 = memory[50441716 + t1731];
        float t1733 = t1719 * t1729;
        float t1734 = t1720 * t1732;
        float t1735 = t1733 - t1734;
        float t1736 = t1719 * t1732;
        float t1737 = t1720 * t1729;
        float t1738 = t1736 + t1737;
        int t1739 = t1598 + t1721;
        float t1740 = t1724 + t1735;
        memory[50441716 + t1739] = t1740;
        int t1742 = t1598 + t1721;
        int t1743 = t1742 + 1024;
        float t1744 = t1727 + t1738;
        memory[50441716 + t1743] = t1744;
        int t1746 = t1598 + t1722;
        float t1747 = t1724 - t1735;
        memory[50441716 + t1746] = t1747;
        int t1749 = t1598 + t1722;
        int t1750 = t1749 + 1024;
        float t1751 = t1727 - t1738;
        memory[50441716 + t1750] = t1751;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1754 = 0; _pr1754 < 512; _pr1754++) {
        float t1755 = (float)_pr1754;
        float t1756 = (t1755 * 0.5);
        float t1757 = metal::floor(t1756);
        float t1758 = t1757 * 2.0;
        float t1759 = t1755 - t1758;
        float t1760 = t1757 * 4.0;
        float t1761 = t1760 + t1759;
        float t1762 = t1761 + 2.0;
        float t1763 = -6.283185 * t1759;
        float t1764 = (t1763 * 0.25);
        float t1765 = metal::cos(t1764);
        float t1766 = metal::sin(t1764);
        int t1767 = (int)t1761;
        int t1768 = (int)t1762;
        int t1769 = t1598 + t1767;
        float t1770 = memory[50441716 + t1769];
        int t1771 = t1598 + t1767;
        int t1772 = t1771 + 1024;
        float t1773 = memory[50441716 + t1772];
        int t1774 = t1598 + t1768;
        float t1775 = memory[50441716 + t1774];
        int t1776 = t1598 + t1768;
        int t1777 = t1776 + 1024;
        float t1778 = memory[50441716 + t1777];
        float t1779 = t1765 * t1775;
        float t1780 = t1766 * t1778;
        float t1781 = t1779 - t1780;
        float t1782 = t1765 * t1778;
        float t1783 = t1766 * t1775;
        float t1784 = t1782 + t1783;
        int t1785 = t1598 + t1767;
        float t1786 = t1770 + t1781;
        memory[50441716 + t1785] = t1786;
        int t1788 = t1598 + t1767;
        int t1789 = t1788 + 1024;
        float t1790 = t1773 + t1784;
        memory[50441716 + t1789] = t1790;
        int t1792 = t1598 + t1768;
        float t1793 = t1770 - t1781;
        memory[50441716 + t1792] = t1793;
        int t1795 = t1598 + t1768;
        int t1796 = t1795 + 1024;
        float t1797 = t1773 - t1784;
        memory[50441716 + t1796] = t1797;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1800 = 0; _pr1800 < 512; _pr1800++) {
        float t1801 = (float)_pr1800;
        float t1802 = (t1801 * 0.25);
        float t1803 = metal::floor(t1802);
        float t1804 = t1803 * 4.0;
        float t1805 = t1801 - t1804;
        float t1806 = t1803 * 8.0;
        float t1807 = t1806 + t1805;
        float t1808 = t1807 + 4.0;
        float t1809 = -6.283185 * t1805;
        float t1810 = (t1809 * 0.125);
        float t1811 = metal::cos(t1810);
        float t1812 = metal::sin(t1810);
        int t1813 = (int)t1807;
        int t1814 = (int)t1808;
        int t1815 = t1598 + t1813;
        float t1816 = memory[50441716 + t1815];
        int t1817 = t1598 + t1813;
        int t1818 = t1817 + 1024;
        float t1819 = memory[50441716 + t1818];
        int t1820 = t1598 + t1814;
        float t1821 = memory[50441716 + t1820];
        int t1822 = t1598 + t1814;
        int t1823 = t1822 + 1024;
        float t1824 = memory[50441716 + t1823];
        float t1825 = t1811 * t1821;
        float t1826 = t1812 * t1824;
        float t1827 = t1825 - t1826;
        float t1828 = t1811 * t1824;
        float t1829 = t1812 * t1821;
        float t1830 = t1828 + t1829;
        int t1831 = t1598 + t1813;
        float t1832 = t1816 + t1827;
        memory[50441716 + t1831] = t1832;
        int t1834 = t1598 + t1813;
        int t1835 = t1834 + 1024;
        float t1836 = t1819 + t1830;
        memory[50441716 + t1835] = t1836;
        int t1838 = t1598 + t1814;
        float t1839 = t1816 - t1827;
        memory[50441716 + t1838] = t1839;
        int t1841 = t1598 + t1814;
        int t1842 = t1841 + 1024;
        float t1843 = t1819 - t1830;
        memory[50441716 + t1842] = t1843;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1846 = 0; _pr1846 < 512; _pr1846++) {
        float t1847 = (float)_pr1846;
        float t1848 = (t1847 * 0.125);
        float t1849 = metal::floor(t1848);
        float t1850 = t1849 * 8.0;
        float t1851 = t1847 - t1850;
        float t1852 = t1849 * 16.0;
        float t1853 = t1852 + t1851;
        float t1854 = t1853 + 8.0;
        float t1855 = -6.283185 * t1851;
        float t1856 = (t1855 * 0.0625);
        float t1857 = metal::cos(t1856);
        float t1858 = metal::sin(t1856);
        int t1859 = (int)t1853;
        int t1860 = (int)t1854;
        int t1861 = t1598 + t1859;
        float t1862 = memory[50441716 + t1861];
        int t1863 = t1598 + t1859;
        int t1864 = t1863 + 1024;
        float t1865 = memory[50441716 + t1864];
        int t1866 = t1598 + t1860;
        float t1867 = memory[50441716 + t1866];
        int t1868 = t1598 + t1860;
        int t1869 = t1868 + 1024;
        float t1870 = memory[50441716 + t1869];
        float t1871 = t1857 * t1867;
        float t1872 = t1858 * t1870;
        float t1873 = t1871 - t1872;
        float t1874 = t1857 * t1870;
        float t1875 = t1858 * t1867;
        float t1876 = t1874 + t1875;
        int t1877 = t1598 + t1859;
        float t1878 = t1862 + t1873;
        memory[50441716 + t1877] = t1878;
        int t1880 = t1598 + t1859;
        int t1881 = t1880 + 1024;
        float t1882 = t1865 + t1876;
        memory[50441716 + t1881] = t1882;
        int t1884 = t1598 + t1860;
        float t1885 = t1862 - t1873;
        memory[50441716 + t1884] = t1885;
        int t1887 = t1598 + t1860;
        int t1888 = t1887 + 1024;
        float t1889 = t1865 - t1876;
        memory[50441716 + t1888] = t1889;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1892 = 0; _pr1892 < 512; _pr1892++) {
        float t1893 = (float)_pr1892;
        float t1894 = (t1893 * 0.0625);
        float t1895 = metal::floor(t1894);
        float t1896 = t1895 * 16.0;
        float t1897 = t1893 - t1896;
        float t1898 = t1895 * 32.0;
        float t1899 = t1898 + t1897;
        float t1900 = t1899 + 16.0;
        float t1901 = -6.283185 * t1897;
        float t1902 = (t1901 * 0.03125);
        float t1903 = metal::cos(t1902);
        float t1904 = metal::sin(t1902);
        int t1905 = (int)t1899;
        int t1906 = (int)t1900;
        int t1907 = t1598 + t1905;
        float t1908 = memory[50441716 + t1907];
        int t1909 = t1598 + t1905;
        int t1910 = t1909 + 1024;
        float t1911 = memory[50441716 + t1910];
        int t1912 = t1598 + t1906;
        float t1913 = memory[50441716 + t1912];
        int t1914 = t1598 + t1906;
        int t1915 = t1914 + 1024;
        float t1916 = memory[50441716 + t1915];
        float t1917 = t1903 * t1913;
        float t1918 = t1904 * t1916;
        float t1919 = t1917 - t1918;
        float t1920 = t1903 * t1916;
        float t1921 = t1904 * t1913;
        float t1922 = t1920 + t1921;
        int t1923 = t1598 + t1905;
        float t1924 = t1908 + t1919;
        memory[50441716 + t1923] = t1924;
        int t1926 = t1598 + t1905;
        int t1927 = t1926 + 1024;
        float t1928 = t1911 + t1922;
        memory[50441716 + t1927] = t1928;
        int t1930 = t1598 + t1906;
        float t1931 = t1908 - t1919;
        memory[50441716 + t1930] = t1931;
        int t1933 = t1598 + t1906;
        int t1934 = t1933 + 1024;
        float t1935 = t1911 - t1922;
        memory[50441716 + t1934] = t1935;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1938 = 0; _pr1938 < 512; _pr1938++) {
        float t1939 = (float)_pr1938;
        float t1940 = (t1939 * 0.03125);
        float t1941 = metal::floor(t1940);
        float t1942 = t1941 * 32.0;
        float t1943 = t1939 - t1942;
        float t1944 = t1941 * 64.0;
        float t1945 = t1944 + t1943;
        float t1946 = t1945 + 32.0;
        float t1947 = -6.283185 * t1943;
        float t1948 = (t1947 * 0.015625);
        float t1949 = metal::cos(t1948);
        float t1950 = metal::sin(t1948);
        int t1951 = (int)t1945;
        int t1952 = (int)t1946;
        int t1953 = t1598 + t1951;
        float t1954 = memory[50441716 + t1953];
        int t1955 = t1598 + t1951;
        int t1956 = t1955 + 1024;
        float t1957 = memory[50441716 + t1956];
        int t1958 = t1598 + t1952;
        float t1959 = memory[50441716 + t1958];
        int t1960 = t1598 + t1952;
        int t1961 = t1960 + 1024;
        float t1962 = memory[50441716 + t1961];
        float t1963 = t1949 * t1959;
        float t1964 = t1950 * t1962;
        float t1965 = t1963 - t1964;
        float t1966 = t1949 * t1962;
        float t1967 = t1950 * t1959;
        float t1968 = t1966 + t1967;
        int t1969 = t1598 + t1951;
        float t1970 = t1954 + t1965;
        memory[50441716 + t1969] = t1970;
        int t1972 = t1598 + t1951;
        int t1973 = t1972 + 1024;
        float t1974 = t1957 + t1968;
        memory[50441716 + t1973] = t1974;
        int t1976 = t1598 + t1952;
        float t1977 = t1954 - t1965;
        memory[50441716 + t1976] = t1977;
        int t1979 = t1598 + t1952;
        int t1980 = t1979 + 1024;
        float t1981 = t1957 - t1968;
        memory[50441716 + t1980] = t1981;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1984 = 0; _pr1984 < 512; _pr1984++) {
        float t1985 = (float)_pr1984;
        float t1986 = (t1985 * 0.015625);
        float t1987 = metal::floor(t1986);
        float t1988 = t1987 * 64.0;
        float t1989 = t1985 - t1988;
        float t1990 = t1987 * 128.0;
        float t1991 = t1990 + t1989;
        float t1992 = t1991 + 64.0;
        float t1993 = -6.283185 * t1989;
        float t1994 = (t1993 * 0.0078125);
        float t1995 = metal::cos(t1994);
        float t1996 = metal::sin(t1994);
        int t1997 = (int)t1991;
        int t1998 = (int)t1992;
        int t1999 = t1598 + t1997;
        float t2000 = memory[50441716 + t1999];
        int t2001 = t1598 + t1997;
        int t2002 = t2001 + 1024;
        float t2003 = memory[50441716 + t2002];
        int t2004 = t1598 + t1998;
        float t2005 = memory[50441716 + t2004];
        int t2006 = t1598 + t1998;
        int t2007 = t2006 + 1024;
        float t2008 = memory[50441716 + t2007];
        float t2009 = t1995 * t2005;
        float t2010 = t1996 * t2008;
        float t2011 = t2009 - t2010;
        float t2012 = t1995 * t2008;
        float t2013 = t1996 * t2005;
        float t2014 = t2012 + t2013;
        int t2015 = t1598 + t1997;
        float t2016 = t2000 + t2011;
        memory[50441716 + t2015] = t2016;
        int t2018 = t1598 + t1997;
        int t2019 = t2018 + 1024;
        float t2020 = t2003 + t2014;
        memory[50441716 + t2019] = t2020;
        int t2022 = t1598 + t1998;
        float t2023 = t2000 - t2011;
        memory[50441716 + t2022] = t2023;
        int t2025 = t1598 + t1998;
        int t2026 = t2025 + 1024;
        float t2027 = t2003 - t2014;
        memory[50441716 + t2026] = t2027;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2030 = 0; _pr2030 < 512; _pr2030++) {
        float t2031 = (float)_pr2030;
        float t2032 = (t2031 * 0.0078125);
        float t2033 = metal::floor(t2032);
        float t2034 = t2033 * 128.0;
        float t2035 = t2031 - t2034;
        float t2036 = t2033 * 256.0;
        float t2037 = t2036 + t2035;
        float t2038 = t2037 + 128.0;
        float t2039 = -6.283185 * t2035;
        float t2040 = (t2039 * 0.00390625);
        float t2041 = metal::cos(t2040);
        float t2042 = metal::sin(t2040);
        int t2043 = (int)t2037;
        int t2044 = (int)t2038;
        int t2045 = t1598 + t2043;
        float t2046 = memory[50441716 + t2045];
        int t2047 = t1598 + t2043;
        int t2048 = t2047 + 1024;
        float t2049 = memory[50441716 + t2048];
        int t2050 = t1598 + t2044;
        float t2051 = memory[50441716 + t2050];
        int t2052 = t1598 + t2044;
        int t2053 = t2052 + 1024;
        float t2054 = memory[50441716 + t2053];
        float t2055 = t2041 * t2051;
        float t2056 = t2042 * t2054;
        float t2057 = t2055 - t2056;
        float t2058 = t2041 * t2054;
        float t2059 = t2042 * t2051;
        float t2060 = t2058 + t2059;
        int t2061 = t1598 + t2043;
        float t2062 = t2046 + t2057;
        memory[50441716 + t2061] = t2062;
        int t2064 = t1598 + t2043;
        int t2065 = t2064 + 1024;
        float t2066 = t2049 + t2060;
        memory[50441716 + t2065] = t2066;
        int t2068 = t1598 + t2044;
        float t2069 = t2046 - t2057;
        memory[50441716 + t2068] = t2069;
        int t2071 = t1598 + t2044;
        int t2072 = t2071 + 1024;
        float t2073 = t2049 - t2060;
        memory[50441716 + t2072] = t2073;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2076 = 0; _pr2076 < 512; _pr2076++) {
        float t2077 = (float)_pr2076;
        float t2078 = (t2077 * 0.00390625);
        float t2079 = metal::floor(t2078);
        float t2080 = t2079 * 256.0;
        float t2081 = t2077 - t2080;
        float t2082 = t2079 * 512.0;
        float t2083 = t2082 + t2081;
        float t2084 = t2083 + 256.0;
        float t2085 = -6.283185 * t2081;
        float t2086 = (t2085 * 0.001953125);
        float t2087 = metal::cos(t2086);
        float t2088 = metal::sin(t2086);
        int t2089 = (int)t2083;
        int t2090 = (int)t2084;
        int t2091 = t1598 + t2089;
        float t2092 = memory[50441716 + t2091];
        int t2093 = t1598 + t2089;
        int t2094 = t2093 + 1024;
        float t2095 = memory[50441716 + t2094];
        int t2096 = t1598 + t2090;
        float t2097 = memory[50441716 + t2096];
        int t2098 = t1598 + t2090;
        int t2099 = t2098 + 1024;
        float t2100 = memory[50441716 + t2099];
        float t2101 = t2087 * t2097;
        float t2102 = t2088 * t2100;
        float t2103 = t2101 - t2102;
        float t2104 = t2087 * t2100;
        float t2105 = t2088 * t2097;
        float t2106 = t2104 + t2105;
        int t2107 = t1598 + t2089;
        float t2108 = t2092 + t2103;
        memory[50441716 + t2107] = t2108;
        int t2110 = t1598 + t2089;
        int t2111 = t2110 + 1024;
        float t2112 = t2095 + t2106;
        memory[50441716 + t2111] = t2112;
        int t2114 = t1598 + t2090;
        float t2115 = t2092 - t2103;
        memory[50441716 + t2114] = t2115;
        int t2117 = t1598 + t2090;
        int t2118 = t2117 + 1024;
        float t2119 = t2095 - t2106;
        memory[50441716 + t2118] = t2119;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2122 = 0; _pr2122 < 512; _pr2122++) {
        float t2123 = (float)_pr2122;
        float t2124 = (t2123 * 0.001953125);
        float t2125 = metal::floor(t2124);
        float t2126 = t2125 * 512.0;
        float t2127 = t2123 - t2126;
        float t2128 = t2125 * 1024.0;
        float t2129 = t2128 + t2127;
        float t2130 = t2129 + 512.0;
        float t2131 = -6.283185 * t2127;
        float t2132 = (t2131 * 0.0009765625);
        float t2133 = metal::cos(t2132);
        float t2134 = metal::sin(t2132);
        int t2135 = (int)t2129;
        int t2136 = (int)t2130;
        int t2137 = t1598 + t2135;
        float t2138 = memory[50441716 + t2137];
        int t2139 = t1598 + t2135;
        int t2140 = t2139 + 1024;
        float t2141 = memory[50441716 + t2140];
        int t2142 = t1598 + t2136;
        float t2143 = memory[50441716 + t2142];
        int t2144 = t1598 + t2136;
        int t2145 = t2144 + 1024;
        float t2146 = memory[50441716 + t2145];
        float t2147 = t2133 * t2143;
        float t2148 = t2134 * t2146;
        float t2149 = t2147 - t2148;
        float t2150 = t2133 * t2146;
        float t2151 = t2134 * t2143;
        float t2152 = t2150 + t2151;
        int t2153 = t1598 + t2135;
        float t2154 = t2138 + t2149;
        memory[50441716 + t2153] = t2154;
        int t2156 = t1598 + t2135;
        int t2157 = t2156 + 1024;
        float t2158 = t2141 + t2152;
        memory[50441716 + t2157] = t2158;
        int t2160 = t1598 + t2136;
        float t2161 = t2138 - t2149;
        memory[50441716 + t2160] = t2161;
        int t2163 = t1598 + t2136;
        int t2164 = t2163 + 1024;
        float t2165 = t2141 - t2152;
        memory[50441716 + t2164] = t2165;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2168 = 0; t2168 < 1024; t2168++) {
        float t2169 = (float)t2168;
        float t2170 = (t2169 - metal::floor(t2169 / 2.0) * 2.0);
        float t2171 = t2170;
        float t2172 = (t2169 * 0.5);
        float t2173 = metal::floor(t2172);
        float t2174 = t2171 * 2.0;
        float t2175 = (t2173 - metal::floor(t2173 / 2.0) * 2.0);
        float t2176 = t2174 + t2175;
        float t2177 = (t2173 * 0.5);
        float t2178 = metal::floor(t2177);
        float t2179 = t2176 * 2.0;
        float t2180 = (t2178 - metal::floor(t2178 / 2.0) * 2.0);
        float t2181 = t2179 + t2180;
        float t2182 = (t2178 * 0.5);
        float t2183 = metal::floor(t2182);
        float t2184 = t2181 * 2.0;
        float t2185 = (t2183 - metal::floor(t2183 / 2.0) * 2.0);
        float t2186 = t2184 + t2185;
        float t2187 = (t2183 * 0.5);
        float t2188 = metal::floor(t2187);
        float t2189 = t2186 * 2.0;
        float t2190 = (t2188 - metal::floor(t2188 / 2.0) * 2.0);
        float t2191 = t2189 + t2190;
        float t2192 = (t2188 * 0.5);
        float t2193 = metal::floor(t2192);
        float t2194 = t2191 * 2.0;
        float t2195 = (t2193 - metal::floor(t2193 / 2.0) * 2.0);
        float t2196 = t2194 + t2195;
        float t2197 = (t2193 * 0.5);
        float t2198 = metal::floor(t2197);
        float t2199 = t2196 * 2.0;
        float t2200 = (t2198 - metal::floor(t2198 / 2.0) * 2.0);
        float t2201 = t2199 + t2200;
        float t2202 = (t2198 * 0.5);
        float t2203 = metal::floor(t2202);
        float t2204 = t2201 * 2.0;
        float t2205 = (t2203 - metal::floor(t2203 / 2.0) * 2.0);
        float t2206 = t2204 + t2205;
        float t2207 = (t2203 * 0.5);
        float t2208 = metal::floor(t2207);
        float t2209 = t2206 * 2.0;
        float t2210 = (t2208 - metal::floor(t2208 / 2.0) * 2.0);
        float t2211 = t2209 + t2210;
        float t2212 = (t2208 * 0.5);
        float t2213 = metal::floor(t2212);
        float t2214 = t2211 * 2.0;
        float t2215 = (t2213 - metal::floor(t2213 / 2.0) * 2.0);
        float t2216 = t2214 + t2215;
        float t2217 = (t2213 * 0.5);
        float t2218 = metal::floor(t2217);
        float t2219 = (float)t2168;
        float t2220 = t2219 < t2216;
        int t2221 = (int)t2216;
        int t2222 = t1598 + t2168;
        float t2223 = memory[83996148 + t2222];
        int t2224 = t1598 + t2168;
        int t2225 = t2224 + 1024;
        float t2226 = memory[83996148 + t2225];
        int t2227 = t1598 + t2221;
        float t2228 = memory[83996148 + t2227];
        int t2229 = t1598 + t2221;
        int t2230 = t2229 + 1024;
        float t2231 = memory[83996148 + t2230];
        float t2232 = metal::select(t2223, t2228, t2220 > 0.0);
        float t2233 = metal::select(t2226, t2231, t2220 > 0.0);
        float t2234 = metal::select(t2228, t2223, t2220 > 0.0);
        float t2235 = metal::select(t2231, t2226, t2220 > 0.0);
        int t2236 = t1598 + t2168;
        memory[83996148 + t2236] = t2232;
        int t2238 = t1598 + t2168;
        int t2239 = t2238 + 1024;
        memory[83996148 + t2239] = t2233;
        int t2241 = t1598 + t2221;
        memory[83996148 + t2241] = t2234;
        int t2243 = t1598 + t2221;
        int t2244 = t2243 + 1024;
        memory[83996148 + t2244] = t2235;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2247 = 0; _pr2247 < 512; _pr2247++) {
        float t2248 = (float)_pr2247;
        float t2249 = t2248;
        float t2250 = metal::floor(t2249);
        float t2251 = t2250;
        float t2252 = t2248 - t2251;
        float t2253 = t2250 * 2.0;
        float t2254 = t2253 + t2252;
        float t2255 = t2254 + 1.0;
        float t2256 = -6.283185 * t2252;
        float t2257 = (t2256 * 0.5);
        float t2258 = metal::cos(t2257);
        float t2259 = metal::sin(t2257);
        int t2260 = (int)t2254;
        int t2261 = (int)t2255;
        int t2262 = t1598 + t2260;
        float t2263 = memory[83996148 + t2262];
        int t2264 = t1598 + t2260;
        int t2265 = t2264 + 1024;
        float t2266 = memory[83996148 + t2265];
        int t2267 = t1598 + t2261;
        float t2268 = memory[83996148 + t2267];
        int t2269 = t1598 + t2261;
        int t2270 = t2269 + 1024;
        float t2271 = memory[83996148 + t2270];
        float t2272 = t2258 * t2268;
        float t2273 = t2259 * t2271;
        float t2274 = t2272 - t2273;
        float t2275 = t2258 * t2271;
        float t2276 = t2259 * t2268;
        float t2277 = t2275 + t2276;
        int t2278 = t1598 + t2260;
        float t2279 = t2263 + t2274;
        memory[83996148 + t2278] = t2279;
        int t2281 = t1598 + t2260;
        int t2282 = t2281 + 1024;
        float t2283 = t2266 + t2277;
        memory[83996148 + t2282] = t2283;
        int t2285 = t1598 + t2261;
        float t2286 = t2263 - t2274;
        memory[83996148 + t2285] = t2286;
        int t2288 = t1598 + t2261;
        int t2289 = t2288 + 1024;
        float t2290 = t2266 - t2277;
        memory[83996148 + t2289] = t2290;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2293 = 0; _pr2293 < 512; _pr2293++) {
        float t2294 = (float)_pr2293;
        float t2295 = (t2294 * 0.5);
        float t2296 = metal::floor(t2295);
        float t2297 = t2296 * 2.0;
        float t2298 = t2294 - t2297;
        float t2299 = t2296 * 4.0;
        float t2300 = t2299 + t2298;
        float t2301 = t2300 + 2.0;
        float t2302 = -6.283185 * t2298;
        float t2303 = (t2302 * 0.25);
        float t2304 = metal::cos(t2303);
        float t2305 = metal::sin(t2303);
        int t2306 = (int)t2300;
        int t2307 = (int)t2301;
        int t2308 = t1598 + t2306;
        float t2309 = memory[83996148 + t2308];
        int t2310 = t1598 + t2306;
        int t2311 = t2310 + 1024;
        float t2312 = memory[83996148 + t2311];
        int t2313 = t1598 + t2307;
        float t2314 = memory[83996148 + t2313];
        int t2315 = t1598 + t2307;
        int t2316 = t2315 + 1024;
        float t2317 = memory[83996148 + t2316];
        float t2318 = t2304 * t2314;
        float t2319 = t2305 * t2317;
        float t2320 = t2318 - t2319;
        float t2321 = t2304 * t2317;
        float t2322 = t2305 * t2314;
        float t2323 = t2321 + t2322;
        int t2324 = t1598 + t2306;
        float t2325 = t2309 + t2320;
        memory[83996148 + t2324] = t2325;
        int t2327 = t1598 + t2306;
        int t2328 = t2327 + 1024;
        float t2329 = t2312 + t2323;
        memory[83996148 + t2328] = t2329;
        int t2331 = t1598 + t2307;
        float t2332 = t2309 - t2320;
        memory[83996148 + t2331] = t2332;
        int t2334 = t1598 + t2307;
        int t2335 = t2334 + 1024;
        float t2336 = t2312 - t2323;
        memory[83996148 + t2335] = t2336;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2339 = 0; _pr2339 < 512; _pr2339++) {
        float t2340 = (float)_pr2339;
        float t2341 = (t2340 * 0.25);
        float t2342 = metal::floor(t2341);
        float t2343 = t2342 * 4.0;
        float t2344 = t2340 - t2343;
        float t2345 = t2342 * 8.0;
        float t2346 = t2345 + t2344;
        float t2347 = t2346 + 4.0;
        float t2348 = -6.283185 * t2344;
        float t2349 = (t2348 * 0.125);
        float t2350 = metal::cos(t2349);
        float t2351 = metal::sin(t2349);
        int t2352 = (int)t2346;
        int t2353 = (int)t2347;
        int t2354 = t1598 + t2352;
        float t2355 = memory[83996148 + t2354];
        int t2356 = t1598 + t2352;
        int t2357 = t2356 + 1024;
        float t2358 = memory[83996148 + t2357];
        int t2359 = t1598 + t2353;
        float t2360 = memory[83996148 + t2359];
        int t2361 = t1598 + t2353;
        int t2362 = t2361 + 1024;
        float t2363 = memory[83996148 + t2362];
        float t2364 = t2350 * t2360;
        float t2365 = t2351 * t2363;
        float t2366 = t2364 - t2365;
        float t2367 = t2350 * t2363;
        float t2368 = t2351 * t2360;
        float t2369 = t2367 + t2368;
        int t2370 = t1598 + t2352;
        float t2371 = t2355 + t2366;
        memory[83996148 + t2370] = t2371;
        int t2373 = t1598 + t2352;
        int t2374 = t2373 + 1024;
        float t2375 = t2358 + t2369;
        memory[83996148 + t2374] = t2375;
        int t2377 = t1598 + t2353;
        float t2378 = t2355 - t2366;
        memory[83996148 + t2377] = t2378;
        int t2380 = t1598 + t2353;
        int t2381 = t2380 + 1024;
        float t2382 = t2358 - t2369;
        memory[83996148 + t2381] = t2382;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2385 = 0; _pr2385 < 512; _pr2385++) {
        float t2386 = (float)_pr2385;
        float t2387 = (t2386 * 0.125);
        float t2388 = metal::floor(t2387);
        float t2389 = t2388 * 8.0;
        float t2390 = t2386 - t2389;
        float t2391 = t2388 * 16.0;
        float t2392 = t2391 + t2390;
        float t2393 = t2392 + 8.0;
        float t2394 = -6.283185 * t2390;
        float t2395 = (t2394 * 0.0625);
        float t2396 = metal::cos(t2395);
        float t2397 = metal::sin(t2395);
        int t2398 = (int)t2392;
        int t2399 = (int)t2393;
        int t2400 = t1598 + t2398;
        float t2401 = memory[83996148 + t2400];
        int t2402 = t1598 + t2398;
        int t2403 = t2402 + 1024;
        float t2404 = memory[83996148 + t2403];
        int t2405 = t1598 + t2399;
        float t2406 = memory[83996148 + t2405];
        int t2407 = t1598 + t2399;
        int t2408 = t2407 + 1024;
        float t2409 = memory[83996148 + t2408];
        float t2410 = t2396 * t2406;
        float t2411 = t2397 * t2409;
        float t2412 = t2410 - t2411;
        float t2413 = t2396 * t2409;
        float t2414 = t2397 * t2406;
        float t2415 = t2413 + t2414;
        int t2416 = t1598 + t2398;
        float t2417 = t2401 + t2412;
        memory[83996148 + t2416] = t2417;
        int t2419 = t1598 + t2398;
        int t2420 = t2419 + 1024;
        float t2421 = t2404 + t2415;
        memory[83996148 + t2420] = t2421;
        int t2423 = t1598 + t2399;
        float t2424 = t2401 - t2412;
        memory[83996148 + t2423] = t2424;
        int t2426 = t1598 + t2399;
        int t2427 = t2426 + 1024;
        float t2428 = t2404 - t2415;
        memory[83996148 + t2427] = t2428;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2431 = 0; _pr2431 < 512; _pr2431++) {
        float t2432 = (float)_pr2431;
        float t2433 = (t2432 * 0.0625);
        float t2434 = metal::floor(t2433);
        float t2435 = t2434 * 16.0;
        float t2436 = t2432 - t2435;
        float t2437 = t2434 * 32.0;
        float t2438 = t2437 + t2436;
        float t2439 = t2438 + 16.0;
        float t2440 = -6.283185 * t2436;
        float t2441 = (t2440 * 0.03125);
        float t2442 = metal::cos(t2441);
        float t2443 = metal::sin(t2441);
        int t2444 = (int)t2438;
        int t2445 = (int)t2439;
        int t2446 = t1598 + t2444;
        float t2447 = memory[83996148 + t2446];
        int t2448 = t1598 + t2444;
        int t2449 = t2448 + 1024;
        float t2450 = memory[83996148 + t2449];
        int t2451 = t1598 + t2445;
        float t2452 = memory[83996148 + t2451];
        int t2453 = t1598 + t2445;
        int t2454 = t2453 + 1024;
        float t2455 = memory[83996148 + t2454];
        float t2456 = t2442 * t2452;
        float t2457 = t2443 * t2455;
        float t2458 = t2456 - t2457;
        float t2459 = t2442 * t2455;
        float t2460 = t2443 * t2452;
        float t2461 = t2459 + t2460;
        int t2462 = t1598 + t2444;
        float t2463 = t2447 + t2458;
        memory[83996148 + t2462] = t2463;
        int t2465 = t1598 + t2444;
        int t2466 = t2465 + 1024;
        float t2467 = t2450 + t2461;
        memory[83996148 + t2466] = t2467;
        int t2469 = t1598 + t2445;
        float t2470 = t2447 - t2458;
        memory[83996148 + t2469] = t2470;
        int t2472 = t1598 + t2445;
        int t2473 = t2472 + 1024;
        float t2474 = t2450 - t2461;
        memory[83996148 + t2473] = t2474;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2477 = 0; _pr2477 < 512; _pr2477++) {
        float t2478 = (float)_pr2477;
        float t2479 = (t2478 * 0.03125);
        float t2480 = metal::floor(t2479);
        float t2481 = t2480 * 32.0;
        float t2482 = t2478 - t2481;
        float t2483 = t2480 * 64.0;
        float t2484 = t2483 + t2482;
        float t2485 = t2484 + 32.0;
        float t2486 = -6.283185 * t2482;
        float t2487 = (t2486 * 0.015625);
        float t2488 = metal::cos(t2487);
        float t2489 = metal::sin(t2487);
        int t2490 = (int)t2484;
        int t2491 = (int)t2485;
        int t2492 = t1598 + t2490;
        float t2493 = memory[83996148 + t2492];
        int t2494 = t1598 + t2490;
        int t2495 = t2494 + 1024;
        float t2496 = memory[83996148 + t2495];
        int t2497 = t1598 + t2491;
        float t2498 = memory[83996148 + t2497];
        int t2499 = t1598 + t2491;
        int t2500 = t2499 + 1024;
        float t2501 = memory[83996148 + t2500];
        float t2502 = t2488 * t2498;
        float t2503 = t2489 * t2501;
        float t2504 = t2502 - t2503;
        float t2505 = t2488 * t2501;
        float t2506 = t2489 * t2498;
        float t2507 = t2505 + t2506;
        int t2508 = t1598 + t2490;
        float t2509 = t2493 + t2504;
        memory[83996148 + t2508] = t2509;
        int t2511 = t1598 + t2490;
        int t2512 = t2511 + 1024;
        float t2513 = t2496 + t2507;
        memory[83996148 + t2512] = t2513;
        int t2515 = t1598 + t2491;
        float t2516 = t2493 - t2504;
        memory[83996148 + t2515] = t2516;
        int t2518 = t1598 + t2491;
        int t2519 = t2518 + 1024;
        float t2520 = t2496 - t2507;
        memory[83996148 + t2519] = t2520;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2523 = 0; _pr2523 < 512; _pr2523++) {
        float t2524 = (float)_pr2523;
        float t2525 = (t2524 * 0.015625);
        float t2526 = metal::floor(t2525);
        float t2527 = t2526 * 64.0;
        float t2528 = t2524 - t2527;
        float t2529 = t2526 * 128.0;
        float t2530 = t2529 + t2528;
        float t2531 = t2530 + 64.0;
        float t2532 = -6.283185 * t2528;
        float t2533 = (t2532 * 0.0078125);
        float t2534 = metal::cos(t2533);
        float t2535 = metal::sin(t2533);
        int t2536 = (int)t2530;
        int t2537 = (int)t2531;
        int t2538 = t1598 + t2536;
        float t2539 = memory[83996148 + t2538];
        int t2540 = t1598 + t2536;
        int t2541 = t2540 + 1024;
        float t2542 = memory[83996148 + t2541];
        int t2543 = t1598 + t2537;
        float t2544 = memory[83996148 + t2543];
        int t2545 = t1598 + t2537;
        int t2546 = t2545 + 1024;
        float t2547 = memory[83996148 + t2546];
        float t2548 = t2534 * t2544;
        float t2549 = t2535 * t2547;
        float t2550 = t2548 - t2549;
        float t2551 = t2534 * t2547;
        float t2552 = t2535 * t2544;
        float t2553 = t2551 + t2552;
        int t2554 = t1598 + t2536;
        float t2555 = t2539 + t2550;
        memory[83996148 + t2554] = t2555;
        int t2557 = t1598 + t2536;
        int t2558 = t2557 + 1024;
        float t2559 = t2542 + t2553;
        memory[83996148 + t2558] = t2559;
        int t2561 = t1598 + t2537;
        float t2562 = t2539 - t2550;
        memory[83996148 + t2561] = t2562;
        int t2564 = t1598 + t2537;
        int t2565 = t2564 + 1024;
        float t2566 = t2542 - t2553;
        memory[83996148 + t2565] = t2566;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2569 = 0; _pr2569 < 512; _pr2569++) {
        float t2570 = (float)_pr2569;
        float t2571 = (t2570 * 0.0078125);
        float t2572 = metal::floor(t2571);
        float t2573 = t2572 * 128.0;
        float t2574 = t2570 - t2573;
        float t2575 = t2572 * 256.0;
        float t2576 = t2575 + t2574;
        float t2577 = t2576 + 128.0;
        float t2578 = -6.283185 * t2574;
        float t2579 = (t2578 * 0.00390625);
        float t2580 = metal::cos(t2579);
        float t2581 = metal::sin(t2579);
        int t2582 = (int)t2576;
        int t2583 = (int)t2577;
        int t2584 = t1598 + t2582;
        float t2585 = memory[83996148 + t2584];
        int t2586 = t1598 + t2582;
        int t2587 = t2586 + 1024;
        float t2588 = memory[83996148 + t2587];
        int t2589 = t1598 + t2583;
        float t2590 = memory[83996148 + t2589];
        int t2591 = t1598 + t2583;
        int t2592 = t2591 + 1024;
        float t2593 = memory[83996148 + t2592];
        float t2594 = t2580 * t2590;
        float t2595 = t2581 * t2593;
        float t2596 = t2594 - t2595;
        float t2597 = t2580 * t2593;
        float t2598 = t2581 * t2590;
        float t2599 = t2597 + t2598;
        int t2600 = t1598 + t2582;
        float t2601 = t2585 + t2596;
        memory[83996148 + t2600] = t2601;
        int t2603 = t1598 + t2582;
        int t2604 = t2603 + 1024;
        float t2605 = t2588 + t2599;
        memory[83996148 + t2604] = t2605;
        int t2607 = t1598 + t2583;
        float t2608 = t2585 - t2596;
        memory[83996148 + t2607] = t2608;
        int t2610 = t1598 + t2583;
        int t2611 = t2610 + 1024;
        float t2612 = t2588 - t2599;
        memory[83996148 + t2611] = t2612;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2615 = 0; _pr2615 < 512; _pr2615++) {
        float t2616 = (float)_pr2615;
        float t2617 = (t2616 * 0.00390625);
        float t2618 = metal::floor(t2617);
        float t2619 = t2618 * 256.0;
        float t2620 = t2616 - t2619;
        float t2621 = t2618 * 512.0;
        float t2622 = t2621 + t2620;
        float t2623 = t2622 + 256.0;
        float t2624 = -6.283185 * t2620;
        float t2625 = (t2624 * 0.001953125);
        float t2626 = metal::cos(t2625);
        float t2627 = metal::sin(t2625);
        int t2628 = (int)t2622;
        int t2629 = (int)t2623;
        int t2630 = t1598 + t2628;
        float t2631 = memory[83996148 + t2630];
        int t2632 = t1598 + t2628;
        int t2633 = t2632 + 1024;
        float t2634 = memory[83996148 + t2633];
        int t2635 = t1598 + t2629;
        float t2636 = memory[83996148 + t2635];
        int t2637 = t1598 + t2629;
        int t2638 = t2637 + 1024;
        float t2639 = memory[83996148 + t2638];
        float t2640 = t2626 * t2636;
        float t2641 = t2627 * t2639;
        float t2642 = t2640 - t2641;
        float t2643 = t2626 * t2639;
        float t2644 = t2627 * t2636;
        float t2645 = t2643 + t2644;
        int t2646 = t1598 + t2628;
        float t2647 = t2631 + t2642;
        memory[83996148 + t2646] = t2647;
        int t2649 = t1598 + t2628;
        int t2650 = t2649 + 1024;
        float t2651 = t2634 + t2645;
        memory[83996148 + t2650] = t2651;
        int t2653 = t1598 + t2629;
        float t2654 = t2631 - t2642;
        memory[83996148 + t2653] = t2654;
        int t2656 = t1598 + t2629;
        int t2657 = t2656 + 1024;
        float t2658 = t2634 - t2645;
        memory[83996148 + t2657] = t2658;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2661 = 0; _pr2661 < 512; _pr2661++) {
        float t2662 = (float)_pr2661;
        float t2663 = (t2662 * 0.001953125);
        float t2664 = metal::floor(t2663);
        float t2665 = t2664 * 512.0;
        float t2666 = t2662 - t2665;
        float t2667 = t2664 * 1024.0;
        float t2668 = t2667 + t2666;
        float t2669 = t2668 + 512.0;
        float t2670 = -6.283185 * t2666;
        float t2671 = (t2670 * 0.0009765625);
        float t2672 = metal::cos(t2671);
        float t2673 = metal::sin(t2671);
        int t2674 = (int)t2668;
        int t2675 = (int)t2669;
        int t2676 = t1598 + t2674;
        float t2677 = memory[83996148 + t2676];
        int t2678 = t1598 + t2674;
        int t2679 = t2678 + 1024;
        float t2680 = memory[83996148 + t2679];
        int t2681 = t1598 + t2675;
        float t2682 = memory[83996148 + t2681];
        int t2683 = t1598 + t2675;
        int t2684 = t2683 + 1024;
        float t2685 = memory[83996148 + t2684];
        float t2686 = t2672 * t2682;
        float t2687 = t2673 * t2685;
        float t2688 = t2686 - t2687;
        float t2689 = t2672 * t2685;
        float t2690 = t2673 * t2682;
        float t2691 = t2689 + t2690;
        int t2692 = t1598 + t2674;
        float t2693 = t2677 + t2688;
        memory[83996148 + t2692] = t2693;
        int t2695 = t1598 + t2674;
        int t2696 = t2695 + 1024;
        float t2697 = t2680 + t2691;
        memory[83996148 + t2696] = t2697;
        int t2699 = t1598 + t2675;
        float t2700 = t2677 - t2688;
        memory[83996148 + t2699] = t2700;
        int t2702 = t1598 + t2675;
        int t2703 = t2702 + 1024;
        float t2704 = t2680 - t2691;
        memory[83996148 + t2703] = t2704;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2707 = 0; _pr2707 < 513; _pr2707++) {
        int t2708 = t1598 + _pr2707;
        float t2709 = memory[50441716 + t2708];
        int t2710 = t1598 + _pr2707;
        int t2711 = t2710 + 1024;
        float t2712 = memory[50441716 + t2711];
        float t2713 = t2709 * t2709;
        float t2714 = t2712 * t2712;
        float t2715 = t2713 + t2714;
        float t2716 = metal::sqrt(t2715);
        int t2717 = t1599 + _pr2707;
        memory[117550580 + t2717] = t2716;
        int t2719 = t1598 + _pr2707;
        float t2720 = memory[83996148 + t2719];
        int t2721 = t1598 + _pr2707;
        int t2722 = t2721 + 1024;
        float t2723 = memory[83996148 + t2722];
        float t2724 = t2720 * t2720;
        float t2725 = t2723 * t2723;
        float t2726 = t2724 + t2725;
        float t2727 = metal::sqrt(t2726);
        int t2728 = t1599 + _pr2707;
        memory[125955572 + t2728] = t2727;
        float t2730 = t2716 - t2727;
        int t2731 = t1599 + _pr2707;
        float t2732 = t2730 * t2730;
        memory[134360564 + t2731] = t2732;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2735 = 0; t2735 < 513; t2735++) {
        int t2736 = t1599 + t2735;
        float t2737 = memory[134360564 + t2736];
        float t2738 = t[15*frameCount + id] + t2737;
        t[15*frameCount + id] = t2738;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2749), value: global(2749)) */
  float t5726 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5726)) {
    /* loadGlobal(1601) - handled in variable access */
    /* loadGlobal(1579) - handled in variable access */
    int t2741 = id;
    int t2742 = t2741 / 61;
    uint _frameIndex = (uint)(t2742);
    int t2743 = t2742 * 61;
    int t2744 = t2741 - t2743;
    float t2745 = (t[15*frameCount + _frameIndex] * 6.1035156e-05);
    float t2746 = t[13*frameCount + _frameIndex] + t2745;
    float t2747 = t2746 * 0.5;
    float t2748 = t2747;
    t[16*frameCount + _frameIndex] = t2748;
    float t2750 = t2747;
    float t2751 = t2746;
    float t2752 = (t[15*frameCount + _frameIndex] * 3.7252903e-09);
    float t2753 = -0.5 * t2752;
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
    /* loadGlobal(1580) - handled in variable access */
    /* loadGlobal(518) - handled in variable access */
    /* loadGlobal(465) - handled in variable access */
    int t2754 = id;
    int t2755 = t2754 * 2048;
    int t2756 = t2754 * 513;
    int t2757 = t2754 * 2048;
    float t2758 = t[14*frameCount + id] == 0.0;
    if (t2758) {
      for (uint _pr2760 = 0; _pr2760 < 513; _pr2760++) {
        int t2761 = t2756 + _pr2760;
        float t2762 = memory[117550580 + t2761];
        int t2763 = t2756 + _pr2760;
        float t2764 = memory[125955572 + t2763];
        int t2765 = t2755 + _pr2760;
        float t2766 = memory[50441716 + t2765];
        int t2767 = t2755 + _pr2760;
        int t2768 = t2767 + 1024;
        float t2769 = memory[50441716 + t2768];
        int t2770 = t2755 + _pr2760;
        float t2771 = memory[83996148 + t2770];
        int t2772 = t2755 + _pr2760;
        int t2773 = t2772 + 1024;
        float t2774 = memory[83996148 + t2773];
        float t2775 = t2762 - t2764;
        float t2776 = 2.0 * t2775;
        float t2777 = t2776 * 3.0517578e-05;
        float t2778 = t2762 - t2764;
        float t2779 = -2.0 * t2778;
        float t2780 = t2779 * 3.0517578e-05;
        float t2781 = metal::max(t2762, 1e-08);
        float t2782 = metal::max(t2764, 1e-08);
        float t2783 = t2777 * t2766;
        float t2784 = t2783 / t2781;
        float t2785 = t2777 * t2769;
        float t2786 = t2785 / t2781;
        float t2787 = t2780 * t2771;
        float t2788 = t2787 / t2782;
        float t2789 = t2780 * t2774;
        float t2790 = t2789 / t2782;
        int t2791 = t2757 + _pr2760;
        memory[142765556 + t2791] = t2784;
        int t2793 = t2757 + _pr2760;
        int t2794 = t2793 + 1024;
        memory[142765556 + t2794] = t2786;
        int t2796 = t2757 + _pr2760;
        memory[176319988 + t2796] = t2788;
        int t2798 = t2757 + _pr2760;
        int t2799 = t2798 + 1024;
        memory[176319988 + t2799] = t2790;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2802 = 0; _pr2802 < 511; _pr2802++) {
        int t2803 = _pr2802 + 513;
        int t2804 = t2757 + t2803;
        memory[142765556 + t2804] = 0.0;
        int t2806 = t2757 + t2803;
        int t2807 = t2806 + 1024;
        memory[142765556 + t2807] = 0.0;
        int t2809 = t2757 + t2803;
        memory[176319988 + t2809] = 0.0;
        int t2811 = t2757 + t2803;
        int t2812 = t2811 + 1024;
        memory[176319988 + t2812] = 0.0;
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
    /* loadGlobal(1580) - handled in variable access */
    int t2816 = id;
    int t2817 = t2816 * 2048;
    int t2818 = t2816 * 1024;
    float t2819 = t[14*frameCount + id] == 0.0;
    if (t2819) {
      for (uint t2821 = 0; t2821 < 1024; t2821++) {
        float t2822 = (float)t2821;
        float t2823 = (t2822 - metal::floor(t2822 / 2.0) * 2.0);
        float t2824 = t2823;
        float t2825 = (t2822 * 0.5);
        float t2826 = metal::floor(t2825);
        float t2827 = t2824 * 2.0;
        float t2828 = (t2826 - metal::floor(t2826 / 2.0) * 2.0);
        float t2829 = t2827 + t2828;
        float t2830 = (t2826 * 0.5);
        float t2831 = metal::floor(t2830);
        float t2832 = t2829 * 2.0;
        float t2833 = (t2831 - metal::floor(t2831 / 2.0) * 2.0);
        float t2834 = t2832 + t2833;
        float t2835 = (t2831 * 0.5);
        float t2836 = metal::floor(t2835);
        float t2837 = t2834 * 2.0;
        float t2838 = (t2836 - metal::floor(t2836 / 2.0) * 2.0);
        float t2839 = t2837 + t2838;
        float t2840 = (t2836 * 0.5);
        float t2841 = metal::floor(t2840);
        float t2842 = t2839 * 2.0;
        float t2843 = (t2841 - metal::floor(t2841 / 2.0) * 2.0);
        float t2844 = t2842 + t2843;
        float t2845 = (t2841 * 0.5);
        float t2846 = metal::floor(t2845);
        float t2847 = t2844 * 2.0;
        float t2848 = (t2846 - metal::floor(t2846 / 2.0) * 2.0);
        float t2849 = t2847 + t2848;
        float t2850 = (t2846 * 0.5);
        float t2851 = metal::floor(t2850);
        float t2852 = t2849 * 2.0;
        float t2853 = (t2851 - metal::floor(t2851 / 2.0) * 2.0);
        float t2854 = t2852 + t2853;
        float t2855 = (t2851 * 0.5);
        float t2856 = metal::floor(t2855);
        float t2857 = t2854 * 2.0;
        float t2858 = (t2856 - metal::floor(t2856 / 2.0) * 2.0);
        float t2859 = t2857 + t2858;
        float t2860 = (t2856 * 0.5);
        float t2861 = metal::floor(t2860);
        float t2862 = t2859 * 2.0;
        float t2863 = (t2861 - metal::floor(t2861 / 2.0) * 2.0);
        float t2864 = t2862 + t2863;
        float t2865 = (t2861 * 0.5);
        float t2866 = metal::floor(t2865);
        float t2867 = t2864 * 2.0;
        float t2868 = (t2866 - metal::floor(t2866 / 2.0) * 2.0);
        float t2869 = t2867 + t2868;
        float t2870 = (t2866 * 0.5);
        float t2871 = metal::floor(t2870);
        float t2872 = (float)t2821;
        float t2873 = t2872 < t2869;
        int t2874 = (int)t2869;
        int t2875 = t2817 + t2821;
        float t2876 = memory[142765556 + t2875];
        int t2877 = t2817 + t2821;
        int t2878 = t2877 + 1024;
        float t2879 = memory[142765556 + t2878];
        int t2880 = t2817 + t2874;
        float t2881 = memory[142765556 + t2880];
        int t2882 = t2817 + t2874;
        int t2883 = t2882 + 1024;
        float t2884 = memory[142765556 + t2883];
        float t2885 = metal::select(t2876, t2881, t2873 > 0.0);
        float t2886 = metal::select(t2879, t2884, t2873 > 0.0);
        float t2887 = metal::select(t2881, t2876, t2873 > 0.0);
        float t2888 = metal::select(t2884, t2879, t2873 > 0.0);
        int t2889 = t2817 + t2821;
        memory[142765556 + t2889] = t2885;
        int t2891 = t2817 + t2821;
        int t2892 = t2891 + 1024;
        memory[142765556 + t2892] = t2886;
        int t2894 = t2817 + t2874;
        memory[142765556 + t2894] = t2887;
        int t2896 = t2817 + t2874;
        int t2897 = t2896 + 1024;
        memory[142765556 + t2897] = t2888;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2900 = 0; _pr2900 < 512; _pr2900++) {
        float t2901 = (float)_pr2900;
        float t2902 = t2901;
        float t2903 = metal::floor(t2902);
        float t2904 = t2903;
        float t2905 = t2901 - t2904;
        float t2906 = t2903 * 2.0;
        float t2907 = t2906 + t2905;
        float t2908 = t2907 + 1.0;
        float t2909 = 6.283185 * t2905;
        float t2910 = (t2909 * 0.5);
        float t2911 = metal::cos(t2910);
        float t2912 = metal::sin(t2910);
        int t2913 = (int)t2907;
        int t2914 = (int)t2908;
        int t2915 = t2817 + t2913;
        float t2916 = memory[142765556 + t2915];
        int t2917 = t2817 + t2913;
        int t2918 = t2917 + 1024;
        float t2919 = memory[142765556 + t2918];
        int t2920 = t2817 + t2914;
        float t2921 = memory[142765556 + t2920];
        int t2922 = t2817 + t2914;
        int t2923 = t2922 + 1024;
        float t2924 = memory[142765556 + t2923];
        float t2925 = t2911 * t2921;
        float t2926 = t2912 * t2924;
        float t2927 = t2925 - t2926;
        float t2928 = t2911 * t2924;
        float t2929 = t2912 * t2921;
        float t2930 = t2928 + t2929;
        int t2931 = t2817 + t2913;
        float t2932 = t2916 + t2927;
        memory[142765556 + t2931] = t2932;
        int t2934 = t2817 + t2913;
        int t2935 = t2934 + 1024;
        float t2936 = t2919 + t2930;
        memory[142765556 + t2935] = t2936;
        int t2938 = t2817 + t2914;
        float t2939 = t2916 - t2927;
        memory[142765556 + t2938] = t2939;
        int t2941 = t2817 + t2914;
        int t2942 = t2941 + 1024;
        float t2943 = t2919 - t2930;
        memory[142765556 + t2942] = t2943;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2946 = 0; _pr2946 < 512; _pr2946++) {
        float t2947 = (float)_pr2946;
        float t2948 = (t2947 * 0.5);
        float t2949 = metal::floor(t2948);
        float t2950 = t2949 * 2.0;
        float t2951 = t2947 - t2950;
        float t2952 = t2949 * 4.0;
        float t2953 = t2952 + t2951;
        float t2954 = t2953 + 2.0;
        float t2955 = 6.283185 * t2951;
        float t2956 = (t2955 * 0.25);
        float t2957 = metal::cos(t2956);
        float t2958 = metal::sin(t2956);
        int t2959 = (int)t2953;
        int t2960 = (int)t2954;
        int t2961 = t2817 + t2959;
        float t2962 = memory[142765556 + t2961];
        int t2963 = t2817 + t2959;
        int t2964 = t2963 + 1024;
        float t2965 = memory[142765556 + t2964];
        int t2966 = t2817 + t2960;
        float t2967 = memory[142765556 + t2966];
        int t2968 = t2817 + t2960;
        int t2969 = t2968 + 1024;
        float t2970 = memory[142765556 + t2969];
        float t2971 = t2957 * t2967;
        float t2972 = t2958 * t2970;
        float t2973 = t2971 - t2972;
        float t2974 = t2957 * t2970;
        float t2975 = t2958 * t2967;
        float t2976 = t2974 + t2975;
        int t2977 = t2817 + t2959;
        float t2978 = t2962 + t2973;
        memory[142765556 + t2977] = t2978;
        int t2980 = t2817 + t2959;
        int t2981 = t2980 + 1024;
        float t2982 = t2965 + t2976;
        memory[142765556 + t2981] = t2982;
        int t2984 = t2817 + t2960;
        float t2985 = t2962 - t2973;
        memory[142765556 + t2984] = t2985;
        int t2987 = t2817 + t2960;
        int t2988 = t2987 + 1024;
        float t2989 = t2965 - t2976;
        memory[142765556 + t2988] = t2989;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2992 = 0; _pr2992 < 512; _pr2992++) {
        float t2993 = (float)_pr2992;
        float t2994 = (t2993 * 0.25);
        float t2995 = metal::floor(t2994);
        float t2996 = t2995 * 4.0;
        float t2997 = t2993 - t2996;
        float t2998 = t2995 * 8.0;
        float t2999 = t2998 + t2997;
        float t3000 = t2999 + 4.0;
        float t3001 = 6.283185 * t2997;
        float t3002 = (t3001 * 0.125);
        float t3003 = metal::cos(t3002);
        float t3004 = metal::sin(t3002);
        int t3005 = (int)t2999;
        int t3006 = (int)t3000;
        int t3007 = t2817 + t3005;
        float t3008 = memory[142765556 + t3007];
        int t3009 = t2817 + t3005;
        int t3010 = t3009 + 1024;
        float t3011 = memory[142765556 + t3010];
        int t3012 = t2817 + t3006;
        float t3013 = memory[142765556 + t3012];
        int t3014 = t2817 + t3006;
        int t3015 = t3014 + 1024;
        float t3016 = memory[142765556 + t3015];
        float t3017 = t3003 * t3013;
        float t3018 = t3004 * t3016;
        float t3019 = t3017 - t3018;
        float t3020 = t3003 * t3016;
        float t3021 = t3004 * t3013;
        float t3022 = t3020 + t3021;
        int t3023 = t2817 + t3005;
        float t3024 = t3008 + t3019;
        memory[142765556 + t3023] = t3024;
        int t3026 = t2817 + t3005;
        int t3027 = t3026 + 1024;
        float t3028 = t3011 + t3022;
        memory[142765556 + t3027] = t3028;
        int t3030 = t2817 + t3006;
        float t3031 = t3008 - t3019;
        memory[142765556 + t3030] = t3031;
        int t3033 = t2817 + t3006;
        int t3034 = t3033 + 1024;
        float t3035 = t3011 - t3022;
        memory[142765556 + t3034] = t3035;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3038 = 0; _pr3038 < 512; _pr3038++) {
        float t3039 = (float)_pr3038;
        float t3040 = (t3039 * 0.125);
        float t3041 = metal::floor(t3040);
        float t3042 = t3041 * 8.0;
        float t3043 = t3039 - t3042;
        float t3044 = t3041 * 16.0;
        float t3045 = t3044 + t3043;
        float t3046 = t3045 + 8.0;
        float t3047 = 6.283185 * t3043;
        float t3048 = (t3047 * 0.0625);
        float t3049 = metal::cos(t3048);
        float t3050 = metal::sin(t3048);
        int t3051 = (int)t3045;
        int t3052 = (int)t3046;
        int t3053 = t2817 + t3051;
        float t3054 = memory[142765556 + t3053];
        int t3055 = t2817 + t3051;
        int t3056 = t3055 + 1024;
        float t3057 = memory[142765556 + t3056];
        int t3058 = t2817 + t3052;
        float t3059 = memory[142765556 + t3058];
        int t3060 = t2817 + t3052;
        int t3061 = t3060 + 1024;
        float t3062 = memory[142765556 + t3061];
        float t3063 = t3049 * t3059;
        float t3064 = t3050 * t3062;
        float t3065 = t3063 - t3064;
        float t3066 = t3049 * t3062;
        float t3067 = t3050 * t3059;
        float t3068 = t3066 + t3067;
        int t3069 = t2817 + t3051;
        float t3070 = t3054 + t3065;
        memory[142765556 + t3069] = t3070;
        int t3072 = t2817 + t3051;
        int t3073 = t3072 + 1024;
        float t3074 = t3057 + t3068;
        memory[142765556 + t3073] = t3074;
        int t3076 = t2817 + t3052;
        float t3077 = t3054 - t3065;
        memory[142765556 + t3076] = t3077;
        int t3079 = t2817 + t3052;
        int t3080 = t3079 + 1024;
        float t3081 = t3057 - t3068;
        memory[142765556 + t3080] = t3081;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3084 = 0; _pr3084 < 512; _pr3084++) {
        float t3085 = (float)_pr3084;
        float t3086 = (t3085 * 0.0625);
        float t3087 = metal::floor(t3086);
        float t3088 = t3087 * 16.0;
        float t3089 = t3085 - t3088;
        float t3090 = t3087 * 32.0;
        float t3091 = t3090 + t3089;
        float t3092 = t3091 + 16.0;
        float t3093 = 6.283185 * t3089;
        float t3094 = (t3093 * 0.03125);
        float t3095 = metal::cos(t3094);
        float t3096 = metal::sin(t3094);
        int t3097 = (int)t3091;
        int t3098 = (int)t3092;
        int t3099 = t2817 + t3097;
        float t3100 = memory[142765556 + t3099];
        int t3101 = t2817 + t3097;
        int t3102 = t3101 + 1024;
        float t3103 = memory[142765556 + t3102];
        int t3104 = t2817 + t3098;
        float t3105 = memory[142765556 + t3104];
        int t3106 = t2817 + t3098;
        int t3107 = t3106 + 1024;
        float t3108 = memory[142765556 + t3107];
        float t3109 = t3095 * t3105;
        float t3110 = t3096 * t3108;
        float t3111 = t3109 - t3110;
        float t3112 = t3095 * t3108;
        float t3113 = t3096 * t3105;
        float t3114 = t3112 + t3113;
        int t3115 = t2817 + t3097;
        float t3116 = t3100 + t3111;
        memory[142765556 + t3115] = t3116;
        int t3118 = t2817 + t3097;
        int t3119 = t3118 + 1024;
        float t3120 = t3103 + t3114;
        memory[142765556 + t3119] = t3120;
        int t3122 = t2817 + t3098;
        float t3123 = t3100 - t3111;
        memory[142765556 + t3122] = t3123;
        int t3125 = t2817 + t3098;
        int t3126 = t3125 + 1024;
        float t3127 = t3103 - t3114;
        memory[142765556 + t3126] = t3127;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3130 = 0; _pr3130 < 512; _pr3130++) {
        float t3131 = (float)_pr3130;
        float t3132 = (t3131 * 0.03125);
        float t3133 = metal::floor(t3132);
        float t3134 = t3133 * 32.0;
        float t3135 = t3131 - t3134;
        float t3136 = t3133 * 64.0;
        float t3137 = t3136 + t3135;
        float t3138 = t3137 + 32.0;
        float t3139 = 6.283185 * t3135;
        float t3140 = (t3139 * 0.015625);
        float t3141 = metal::cos(t3140);
        float t3142 = metal::sin(t3140);
        int t3143 = (int)t3137;
        int t3144 = (int)t3138;
        int t3145 = t2817 + t3143;
        float t3146 = memory[142765556 + t3145];
        int t3147 = t2817 + t3143;
        int t3148 = t3147 + 1024;
        float t3149 = memory[142765556 + t3148];
        int t3150 = t2817 + t3144;
        float t3151 = memory[142765556 + t3150];
        int t3152 = t2817 + t3144;
        int t3153 = t3152 + 1024;
        float t3154 = memory[142765556 + t3153];
        float t3155 = t3141 * t3151;
        float t3156 = t3142 * t3154;
        float t3157 = t3155 - t3156;
        float t3158 = t3141 * t3154;
        float t3159 = t3142 * t3151;
        float t3160 = t3158 + t3159;
        int t3161 = t2817 + t3143;
        float t3162 = t3146 + t3157;
        memory[142765556 + t3161] = t3162;
        int t3164 = t2817 + t3143;
        int t3165 = t3164 + 1024;
        float t3166 = t3149 + t3160;
        memory[142765556 + t3165] = t3166;
        int t3168 = t2817 + t3144;
        float t3169 = t3146 - t3157;
        memory[142765556 + t3168] = t3169;
        int t3171 = t2817 + t3144;
        int t3172 = t3171 + 1024;
        float t3173 = t3149 - t3160;
        memory[142765556 + t3172] = t3173;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3176 = 0; _pr3176 < 512; _pr3176++) {
        float t3177 = (float)_pr3176;
        float t3178 = (t3177 * 0.015625);
        float t3179 = metal::floor(t3178);
        float t3180 = t3179 * 64.0;
        float t3181 = t3177 - t3180;
        float t3182 = t3179 * 128.0;
        float t3183 = t3182 + t3181;
        float t3184 = t3183 + 64.0;
        float t3185 = 6.283185 * t3181;
        float t3186 = (t3185 * 0.0078125);
        float t3187 = metal::cos(t3186);
        float t3188 = metal::sin(t3186);
        int t3189 = (int)t3183;
        int t3190 = (int)t3184;
        int t3191 = t2817 + t3189;
        float t3192 = memory[142765556 + t3191];
        int t3193 = t2817 + t3189;
        int t3194 = t3193 + 1024;
        float t3195 = memory[142765556 + t3194];
        int t3196 = t2817 + t3190;
        float t3197 = memory[142765556 + t3196];
        int t3198 = t2817 + t3190;
        int t3199 = t3198 + 1024;
        float t3200 = memory[142765556 + t3199];
        float t3201 = t3187 * t3197;
        float t3202 = t3188 * t3200;
        float t3203 = t3201 - t3202;
        float t3204 = t3187 * t3200;
        float t3205 = t3188 * t3197;
        float t3206 = t3204 + t3205;
        int t3207 = t2817 + t3189;
        float t3208 = t3192 + t3203;
        memory[142765556 + t3207] = t3208;
        int t3210 = t2817 + t3189;
        int t3211 = t3210 + 1024;
        float t3212 = t3195 + t3206;
        memory[142765556 + t3211] = t3212;
        int t3214 = t2817 + t3190;
        float t3215 = t3192 - t3203;
        memory[142765556 + t3214] = t3215;
        int t3217 = t2817 + t3190;
        int t3218 = t3217 + 1024;
        float t3219 = t3195 - t3206;
        memory[142765556 + t3218] = t3219;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3222 = 0; _pr3222 < 512; _pr3222++) {
        float t3223 = (float)_pr3222;
        float t3224 = (t3223 * 0.0078125);
        float t3225 = metal::floor(t3224);
        float t3226 = t3225 * 128.0;
        float t3227 = t3223 - t3226;
        float t3228 = t3225 * 256.0;
        float t3229 = t3228 + t3227;
        float t3230 = t3229 + 128.0;
        float t3231 = 6.283185 * t3227;
        float t3232 = (t3231 * 0.00390625);
        float t3233 = metal::cos(t3232);
        float t3234 = metal::sin(t3232);
        int t3235 = (int)t3229;
        int t3236 = (int)t3230;
        int t3237 = t2817 + t3235;
        float t3238 = memory[142765556 + t3237];
        int t3239 = t2817 + t3235;
        int t3240 = t3239 + 1024;
        float t3241 = memory[142765556 + t3240];
        int t3242 = t2817 + t3236;
        float t3243 = memory[142765556 + t3242];
        int t3244 = t2817 + t3236;
        int t3245 = t3244 + 1024;
        float t3246 = memory[142765556 + t3245];
        float t3247 = t3233 * t3243;
        float t3248 = t3234 * t3246;
        float t3249 = t3247 - t3248;
        float t3250 = t3233 * t3246;
        float t3251 = t3234 * t3243;
        float t3252 = t3250 + t3251;
        int t3253 = t2817 + t3235;
        float t3254 = t3238 + t3249;
        memory[142765556 + t3253] = t3254;
        int t3256 = t2817 + t3235;
        int t3257 = t3256 + 1024;
        float t3258 = t3241 + t3252;
        memory[142765556 + t3257] = t3258;
        int t3260 = t2817 + t3236;
        float t3261 = t3238 - t3249;
        memory[142765556 + t3260] = t3261;
        int t3263 = t2817 + t3236;
        int t3264 = t3263 + 1024;
        float t3265 = t3241 - t3252;
        memory[142765556 + t3264] = t3265;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3268 = 0; _pr3268 < 512; _pr3268++) {
        float t3269 = (float)_pr3268;
        float t3270 = (t3269 * 0.00390625);
        float t3271 = metal::floor(t3270);
        float t3272 = t3271 * 256.0;
        float t3273 = t3269 - t3272;
        float t3274 = t3271 * 512.0;
        float t3275 = t3274 + t3273;
        float t3276 = t3275 + 256.0;
        float t3277 = 6.283185 * t3273;
        float t3278 = (t3277 * 0.001953125);
        float t3279 = metal::cos(t3278);
        float t3280 = metal::sin(t3278);
        int t3281 = (int)t3275;
        int t3282 = (int)t3276;
        int t3283 = t2817 + t3281;
        float t3284 = memory[142765556 + t3283];
        int t3285 = t2817 + t3281;
        int t3286 = t3285 + 1024;
        float t3287 = memory[142765556 + t3286];
        int t3288 = t2817 + t3282;
        float t3289 = memory[142765556 + t3288];
        int t3290 = t2817 + t3282;
        int t3291 = t3290 + 1024;
        float t3292 = memory[142765556 + t3291];
        float t3293 = t3279 * t3289;
        float t3294 = t3280 * t3292;
        float t3295 = t3293 - t3294;
        float t3296 = t3279 * t3292;
        float t3297 = t3280 * t3289;
        float t3298 = t3296 + t3297;
        int t3299 = t2817 + t3281;
        float t3300 = t3284 + t3295;
        memory[142765556 + t3299] = t3300;
        int t3302 = t2817 + t3281;
        int t3303 = t3302 + 1024;
        float t3304 = t3287 + t3298;
        memory[142765556 + t3303] = t3304;
        int t3306 = t2817 + t3282;
        float t3307 = t3284 - t3295;
        memory[142765556 + t3306] = t3307;
        int t3309 = t2817 + t3282;
        int t3310 = t3309 + 1024;
        float t3311 = t3287 - t3298;
        memory[142765556 + t3310] = t3311;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3314 = 0; _pr3314 < 512; _pr3314++) {
        float t3315 = (float)_pr3314;
        float t3316 = (t3315 * 0.001953125);
        float t3317 = metal::floor(t3316);
        float t3318 = t3317 * 512.0;
        float t3319 = t3315 - t3318;
        float t3320 = t3317 * 1024.0;
        float t3321 = t3320 + t3319;
        float t3322 = t3321 + 512.0;
        float t3323 = 6.283185 * t3319;
        float t3324 = (t3323 * 0.0009765625);
        float t3325 = metal::cos(t3324);
        float t3326 = metal::sin(t3324);
        int t3327 = (int)t3321;
        int t3328 = (int)t3322;
        int t3329 = t2817 + t3327;
        float t3330 = memory[142765556 + t3329];
        int t3331 = t2817 + t3327;
        int t3332 = t3331 + 1024;
        float t3333 = memory[142765556 + t3332];
        int t3334 = t2817 + t3328;
        float t3335 = memory[142765556 + t3334];
        int t3336 = t2817 + t3328;
        int t3337 = t3336 + 1024;
        float t3338 = memory[142765556 + t3337];
        float t3339 = t3325 * t3335;
        float t3340 = t3326 * t3338;
        float t3341 = t3339 - t3340;
        float t3342 = t3325 * t3338;
        float t3343 = t3326 * t3335;
        float t3344 = t3342 + t3343;
        int t3345 = t2817 + t3327;
        float t3346 = t3330 + t3341;
        memory[142765556 + t3345] = t3346;
        int t3348 = t2817 + t3327;
        int t3349 = t3348 + 1024;
        float t3350 = t3333 + t3344;
        memory[142765556 + t3349] = t3350;
        int t3352 = t2817 + t3328;
        float t3353 = t3330 - t3341;
        memory[142765556 + t3352] = t3353;
        int t3355 = t2817 + t3328;
        int t3356 = t3355 + 1024;
        float t3357 = t3333 - t3344;
        memory[142765556 + t3356] = t3357;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3360 = 0; _pr3360 < 1024; _pr3360++) {
        int t3361 = t2817 + _pr3360;
        float t3362 = memory[142765556 + t3361];
        float t3363 = t3362 * 1.9036306e-06;
        float t3364 = memory[44980 + (int)_pr3360];
        int t3365 = t2818 + _pr3360;
        float t3366 = t3363 * t3364;
        memory[50441716 + t3365] = t3366;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t3369 = 0; t3369 < 1024; t3369++) {
        float t3370 = (float)t3369;
        float t3371 = (t3370 - metal::floor(t3370 / 2.0) * 2.0);
        float t3372 = t3371;
        float t3373 = (t3370 * 0.5);
        float t3374 = metal::floor(t3373);
        float t3375 = t3372 * 2.0;
        float t3376 = (t3374 - metal::floor(t3374 / 2.0) * 2.0);
        float t3377 = t3375 + t3376;
        float t3378 = (t3374 * 0.5);
        float t3379 = metal::floor(t3378);
        float t3380 = t3377 * 2.0;
        float t3381 = (t3379 - metal::floor(t3379 / 2.0) * 2.0);
        float t3382 = t3380 + t3381;
        float t3383 = (t3379 * 0.5);
        float t3384 = metal::floor(t3383);
        float t3385 = t3382 * 2.0;
        float t3386 = (t3384 - metal::floor(t3384 / 2.0) * 2.0);
        float t3387 = t3385 + t3386;
        float t3388 = (t3384 * 0.5);
        float t3389 = metal::floor(t3388);
        float t3390 = t3387 * 2.0;
        float t3391 = (t3389 - metal::floor(t3389 / 2.0) * 2.0);
        float t3392 = t3390 + t3391;
        float t3393 = (t3389 * 0.5);
        float t3394 = metal::floor(t3393);
        float t3395 = t3392 * 2.0;
        float t3396 = (t3394 - metal::floor(t3394 / 2.0) * 2.0);
        float t3397 = t3395 + t3396;
        float t3398 = (t3394 * 0.5);
        float t3399 = metal::floor(t3398);
        float t3400 = t3397 * 2.0;
        float t3401 = (t3399 - metal::floor(t3399 / 2.0) * 2.0);
        float t3402 = t3400 + t3401;
        float t3403 = (t3399 * 0.5);
        float t3404 = metal::floor(t3403);
        float t3405 = t3402 * 2.0;
        float t3406 = (t3404 - metal::floor(t3404 / 2.0) * 2.0);
        float t3407 = t3405 + t3406;
        float t3408 = (t3404 * 0.5);
        float t3409 = metal::floor(t3408);
        float t3410 = t3407 * 2.0;
        float t3411 = (t3409 - metal::floor(t3409 / 2.0) * 2.0);
        float t3412 = t3410 + t3411;
        float t3413 = (t3409 * 0.5);
        float t3414 = metal::floor(t3413);
        float t3415 = t3412 * 2.0;
        float t3416 = (t3414 - metal::floor(t3414 / 2.0) * 2.0);
        float t3417 = t3415 + t3416;
        float t3418 = (t3414 * 0.5);
        float t3419 = metal::floor(t3418);
        float t3420 = (float)t3369;
        float t3421 = t3420 < t3417;
        int t3422 = (int)t3417;
        int t3423 = t2817 + t3369;
        float t3424 = memory[176319988 + t3423];
        int t3425 = t2817 + t3369;
        int t3426 = t3425 + 1024;
        float t3427 = memory[176319988 + t3426];
        int t3428 = t2817 + t3422;
        float t3429 = memory[176319988 + t3428];
        int t3430 = t2817 + t3422;
        int t3431 = t3430 + 1024;
        float t3432 = memory[176319988 + t3431];
        float t3433 = metal::select(t3424, t3429, t3421 > 0.0);
        float t3434 = metal::select(t3427, t3432, t3421 > 0.0);
        float t3435 = metal::select(t3429, t3424, t3421 > 0.0);
        float t3436 = metal::select(t3432, t3427, t3421 > 0.0);
        int t3437 = t2817 + t3369;
        memory[176319988 + t3437] = t3433;
        int t3439 = t2817 + t3369;
        int t3440 = t3439 + 1024;
        memory[176319988 + t3440] = t3434;
        int t3442 = t2817 + t3422;
        memory[176319988 + t3442] = t3435;
        int t3444 = t2817 + t3422;
        int t3445 = t3444 + 1024;
        memory[176319988 + t3445] = t3436;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3448 = 0; _pr3448 < 512; _pr3448++) {
        float t3449 = (float)_pr3448;
        float t3450 = t3449;
        float t3451 = metal::floor(t3450);
        float t3452 = t3451;
        float t3453 = t3449 - t3452;
        float t3454 = t3451 * 2.0;
        float t3455 = t3454 + t3453;
        float t3456 = t3455 + 1.0;
        float t3457 = 6.283185 * t3453;
        float t3458 = (t3457 * 0.5);
        float t3459 = metal::cos(t3458);
        float t3460 = metal::sin(t3458);
        int t3461 = (int)t3455;
        int t3462 = (int)t3456;
        int t3463 = t2817 + t3461;
        float t3464 = memory[176319988 + t3463];
        int t3465 = t2817 + t3461;
        int t3466 = t3465 + 1024;
        float t3467 = memory[176319988 + t3466];
        int t3468 = t2817 + t3462;
        float t3469 = memory[176319988 + t3468];
        int t3470 = t2817 + t3462;
        int t3471 = t3470 + 1024;
        float t3472 = memory[176319988 + t3471];
        float t3473 = t3459 * t3469;
        float t3474 = t3460 * t3472;
        float t3475 = t3473 - t3474;
        float t3476 = t3459 * t3472;
        float t3477 = t3460 * t3469;
        float t3478 = t3476 + t3477;
        int t3479 = t2817 + t3461;
        float t3480 = t3464 + t3475;
        memory[176319988 + t3479] = t3480;
        int t3482 = t2817 + t3461;
        int t3483 = t3482 + 1024;
        float t3484 = t3467 + t3478;
        memory[176319988 + t3483] = t3484;
        int t3486 = t2817 + t3462;
        float t3487 = t3464 - t3475;
        memory[176319988 + t3486] = t3487;
        int t3489 = t2817 + t3462;
        int t3490 = t3489 + 1024;
        float t3491 = t3467 - t3478;
        memory[176319988 + t3490] = t3491;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3494 = 0; _pr3494 < 512; _pr3494++) {
        float t3495 = (float)_pr3494;
        float t3496 = (t3495 * 0.5);
        float t3497 = metal::floor(t3496);
        float t3498 = t3497 * 2.0;
        float t3499 = t3495 - t3498;
        float t3500 = t3497 * 4.0;
        float t3501 = t3500 + t3499;
        float t3502 = t3501 + 2.0;
        float t3503 = 6.283185 * t3499;
        float t3504 = (t3503 * 0.25);
        float t3505 = metal::cos(t3504);
        float t3506 = metal::sin(t3504);
        int t3507 = (int)t3501;
        int t3508 = (int)t3502;
        int t3509 = t2817 + t3507;
        float t3510 = memory[176319988 + t3509];
        int t3511 = t2817 + t3507;
        int t3512 = t3511 + 1024;
        float t3513 = memory[176319988 + t3512];
        int t3514 = t2817 + t3508;
        float t3515 = memory[176319988 + t3514];
        int t3516 = t2817 + t3508;
        int t3517 = t3516 + 1024;
        float t3518 = memory[176319988 + t3517];
        float t3519 = t3505 * t3515;
        float t3520 = t3506 * t3518;
        float t3521 = t3519 - t3520;
        float t3522 = t3505 * t3518;
        float t3523 = t3506 * t3515;
        float t3524 = t3522 + t3523;
        int t3525 = t2817 + t3507;
        float t3526 = t3510 + t3521;
        memory[176319988 + t3525] = t3526;
        int t3528 = t2817 + t3507;
        int t3529 = t3528 + 1024;
        float t3530 = t3513 + t3524;
        memory[176319988 + t3529] = t3530;
        int t3532 = t2817 + t3508;
        float t3533 = t3510 - t3521;
        memory[176319988 + t3532] = t3533;
        int t3535 = t2817 + t3508;
        int t3536 = t3535 + 1024;
        float t3537 = t3513 - t3524;
        memory[176319988 + t3536] = t3537;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3540 = 0; _pr3540 < 512; _pr3540++) {
        float t3541 = (float)_pr3540;
        float t3542 = (t3541 * 0.25);
        float t3543 = metal::floor(t3542);
        float t3544 = t3543 * 4.0;
        float t3545 = t3541 - t3544;
        float t3546 = t3543 * 8.0;
        float t3547 = t3546 + t3545;
        float t3548 = t3547 + 4.0;
        float t3549 = 6.283185 * t3545;
        float t3550 = (t3549 * 0.125);
        float t3551 = metal::cos(t3550);
        float t3552 = metal::sin(t3550);
        int t3553 = (int)t3547;
        int t3554 = (int)t3548;
        int t3555 = t2817 + t3553;
        float t3556 = memory[176319988 + t3555];
        int t3557 = t2817 + t3553;
        int t3558 = t3557 + 1024;
        float t3559 = memory[176319988 + t3558];
        int t3560 = t2817 + t3554;
        float t3561 = memory[176319988 + t3560];
        int t3562 = t2817 + t3554;
        int t3563 = t3562 + 1024;
        float t3564 = memory[176319988 + t3563];
        float t3565 = t3551 * t3561;
        float t3566 = t3552 * t3564;
        float t3567 = t3565 - t3566;
        float t3568 = t3551 * t3564;
        float t3569 = t3552 * t3561;
        float t3570 = t3568 + t3569;
        int t3571 = t2817 + t3553;
        float t3572 = t3556 + t3567;
        memory[176319988 + t3571] = t3572;
        int t3574 = t2817 + t3553;
        int t3575 = t3574 + 1024;
        float t3576 = t3559 + t3570;
        memory[176319988 + t3575] = t3576;
        int t3578 = t2817 + t3554;
        float t3579 = t3556 - t3567;
        memory[176319988 + t3578] = t3579;
        int t3581 = t2817 + t3554;
        int t3582 = t3581 + 1024;
        float t3583 = t3559 - t3570;
        memory[176319988 + t3582] = t3583;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3586 = 0; _pr3586 < 512; _pr3586++) {
        float t3587 = (float)_pr3586;
        float t3588 = (t3587 * 0.125);
        float t3589 = metal::floor(t3588);
        float t3590 = t3589 * 8.0;
        float t3591 = t3587 - t3590;
        float t3592 = t3589 * 16.0;
        float t3593 = t3592 + t3591;
        float t3594 = t3593 + 8.0;
        float t3595 = 6.283185 * t3591;
        float t3596 = (t3595 * 0.0625);
        float t3597 = metal::cos(t3596);
        float t3598 = metal::sin(t3596);
        int t3599 = (int)t3593;
        int t3600 = (int)t3594;
        int t3601 = t2817 + t3599;
        float t3602 = memory[176319988 + t3601];
        int t3603 = t2817 + t3599;
        int t3604 = t3603 + 1024;
        float t3605 = memory[176319988 + t3604];
        int t3606 = t2817 + t3600;
        float t3607 = memory[176319988 + t3606];
        int t3608 = t2817 + t3600;
        int t3609 = t3608 + 1024;
        float t3610 = memory[176319988 + t3609];
        float t3611 = t3597 * t3607;
        float t3612 = t3598 * t3610;
        float t3613 = t3611 - t3612;
        float t3614 = t3597 * t3610;
        float t3615 = t3598 * t3607;
        float t3616 = t3614 + t3615;
        int t3617 = t2817 + t3599;
        float t3618 = t3602 + t3613;
        memory[176319988 + t3617] = t3618;
        int t3620 = t2817 + t3599;
        int t3621 = t3620 + 1024;
        float t3622 = t3605 + t3616;
        memory[176319988 + t3621] = t3622;
        int t3624 = t2817 + t3600;
        float t3625 = t3602 - t3613;
        memory[176319988 + t3624] = t3625;
        int t3627 = t2817 + t3600;
        int t3628 = t3627 + 1024;
        float t3629 = t3605 - t3616;
        memory[176319988 + t3628] = t3629;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3632 = 0; _pr3632 < 512; _pr3632++) {
        float t3633 = (float)_pr3632;
        float t3634 = (t3633 * 0.0625);
        float t3635 = metal::floor(t3634);
        float t3636 = t3635 * 16.0;
        float t3637 = t3633 - t3636;
        float t3638 = t3635 * 32.0;
        float t3639 = t3638 + t3637;
        float t3640 = t3639 + 16.0;
        float t3641 = 6.283185 * t3637;
        float t3642 = (t3641 * 0.03125);
        float t3643 = metal::cos(t3642);
        float t3644 = metal::sin(t3642);
        int t3645 = (int)t3639;
        int t3646 = (int)t3640;
        int t3647 = t2817 + t3645;
        float t3648 = memory[176319988 + t3647];
        int t3649 = t2817 + t3645;
        int t3650 = t3649 + 1024;
        float t3651 = memory[176319988 + t3650];
        int t3652 = t2817 + t3646;
        float t3653 = memory[176319988 + t3652];
        int t3654 = t2817 + t3646;
        int t3655 = t3654 + 1024;
        float t3656 = memory[176319988 + t3655];
        float t3657 = t3643 * t3653;
        float t3658 = t3644 * t3656;
        float t3659 = t3657 - t3658;
        float t3660 = t3643 * t3656;
        float t3661 = t3644 * t3653;
        float t3662 = t3660 + t3661;
        int t3663 = t2817 + t3645;
        float t3664 = t3648 + t3659;
        memory[176319988 + t3663] = t3664;
        int t3666 = t2817 + t3645;
        int t3667 = t3666 + 1024;
        float t3668 = t3651 + t3662;
        memory[176319988 + t3667] = t3668;
        int t3670 = t2817 + t3646;
        float t3671 = t3648 - t3659;
        memory[176319988 + t3670] = t3671;
        int t3673 = t2817 + t3646;
        int t3674 = t3673 + 1024;
        float t3675 = t3651 - t3662;
        memory[176319988 + t3674] = t3675;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3678 = 0; _pr3678 < 512; _pr3678++) {
        float t3679 = (float)_pr3678;
        float t3680 = (t3679 * 0.03125);
        float t3681 = metal::floor(t3680);
        float t3682 = t3681 * 32.0;
        float t3683 = t3679 - t3682;
        float t3684 = t3681 * 64.0;
        float t3685 = t3684 + t3683;
        float t3686 = t3685 + 32.0;
        float t3687 = 6.283185 * t3683;
        float t3688 = (t3687 * 0.015625);
        float t3689 = metal::cos(t3688);
        float t3690 = metal::sin(t3688);
        int t3691 = (int)t3685;
        int t3692 = (int)t3686;
        int t3693 = t2817 + t3691;
        float t3694 = memory[176319988 + t3693];
        int t3695 = t2817 + t3691;
        int t3696 = t3695 + 1024;
        float t3697 = memory[176319988 + t3696];
        int t3698 = t2817 + t3692;
        float t3699 = memory[176319988 + t3698];
        int t3700 = t2817 + t3692;
        int t3701 = t3700 + 1024;
        float t3702 = memory[176319988 + t3701];
        float t3703 = t3689 * t3699;
        float t3704 = t3690 * t3702;
        float t3705 = t3703 - t3704;
        float t3706 = t3689 * t3702;
        float t3707 = t3690 * t3699;
        float t3708 = t3706 + t3707;
        int t3709 = t2817 + t3691;
        float t3710 = t3694 + t3705;
        memory[176319988 + t3709] = t3710;
        int t3712 = t2817 + t3691;
        int t3713 = t3712 + 1024;
        float t3714 = t3697 + t3708;
        memory[176319988 + t3713] = t3714;
        int t3716 = t2817 + t3692;
        float t3717 = t3694 - t3705;
        memory[176319988 + t3716] = t3717;
        int t3719 = t2817 + t3692;
        int t3720 = t3719 + 1024;
        float t3721 = t3697 - t3708;
        memory[176319988 + t3720] = t3721;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3724 = 0; _pr3724 < 512; _pr3724++) {
        float t3725 = (float)_pr3724;
        float t3726 = (t3725 * 0.015625);
        float t3727 = metal::floor(t3726);
        float t3728 = t3727 * 64.0;
        float t3729 = t3725 - t3728;
        float t3730 = t3727 * 128.0;
        float t3731 = t3730 + t3729;
        float t3732 = t3731 + 64.0;
        float t3733 = 6.283185 * t3729;
        float t3734 = (t3733 * 0.0078125);
        float t3735 = metal::cos(t3734);
        float t3736 = metal::sin(t3734);
        int t3737 = (int)t3731;
        int t3738 = (int)t3732;
        int t3739 = t2817 + t3737;
        float t3740 = memory[176319988 + t3739];
        int t3741 = t2817 + t3737;
        int t3742 = t3741 + 1024;
        float t3743 = memory[176319988 + t3742];
        int t3744 = t2817 + t3738;
        float t3745 = memory[176319988 + t3744];
        int t3746 = t2817 + t3738;
        int t3747 = t3746 + 1024;
        float t3748 = memory[176319988 + t3747];
        float t3749 = t3735 * t3745;
        float t3750 = t3736 * t3748;
        float t3751 = t3749 - t3750;
        float t3752 = t3735 * t3748;
        float t3753 = t3736 * t3745;
        float t3754 = t3752 + t3753;
        int t3755 = t2817 + t3737;
        float t3756 = t3740 + t3751;
        memory[176319988 + t3755] = t3756;
        int t3758 = t2817 + t3737;
        int t3759 = t3758 + 1024;
        float t3760 = t3743 + t3754;
        memory[176319988 + t3759] = t3760;
        int t3762 = t2817 + t3738;
        float t3763 = t3740 - t3751;
        memory[176319988 + t3762] = t3763;
        int t3765 = t2817 + t3738;
        int t3766 = t3765 + 1024;
        float t3767 = t3743 - t3754;
        memory[176319988 + t3766] = t3767;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3770 = 0; _pr3770 < 512; _pr3770++) {
        float t3771 = (float)_pr3770;
        float t3772 = (t3771 * 0.0078125);
        float t3773 = metal::floor(t3772);
        float t3774 = t3773 * 128.0;
        float t3775 = t3771 - t3774;
        float t3776 = t3773 * 256.0;
        float t3777 = t3776 + t3775;
        float t3778 = t3777 + 128.0;
        float t3779 = 6.283185 * t3775;
        float t3780 = (t3779 * 0.00390625);
        float t3781 = metal::cos(t3780);
        float t3782 = metal::sin(t3780);
        int t3783 = (int)t3777;
        int t3784 = (int)t3778;
        int t3785 = t2817 + t3783;
        float t3786 = memory[176319988 + t3785];
        int t3787 = t2817 + t3783;
        int t3788 = t3787 + 1024;
        float t3789 = memory[176319988 + t3788];
        int t3790 = t2817 + t3784;
        float t3791 = memory[176319988 + t3790];
        int t3792 = t2817 + t3784;
        int t3793 = t3792 + 1024;
        float t3794 = memory[176319988 + t3793];
        float t3795 = t3781 * t3791;
        float t3796 = t3782 * t3794;
        float t3797 = t3795 - t3796;
        float t3798 = t3781 * t3794;
        float t3799 = t3782 * t3791;
        float t3800 = t3798 + t3799;
        int t3801 = t2817 + t3783;
        float t3802 = t3786 + t3797;
        memory[176319988 + t3801] = t3802;
        int t3804 = t2817 + t3783;
        int t3805 = t3804 + 1024;
        float t3806 = t3789 + t3800;
        memory[176319988 + t3805] = t3806;
        int t3808 = t2817 + t3784;
        float t3809 = t3786 - t3797;
        memory[176319988 + t3808] = t3809;
        int t3811 = t2817 + t3784;
        int t3812 = t3811 + 1024;
        float t3813 = t3789 - t3800;
        memory[176319988 + t3812] = t3813;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3816 = 0; _pr3816 < 512; _pr3816++) {
        float t3817 = (float)_pr3816;
        float t3818 = (t3817 * 0.00390625);
        float t3819 = metal::floor(t3818);
        float t3820 = t3819 * 256.0;
        float t3821 = t3817 - t3820;
        float t3822 = t3819 * 512.0;
        float t3823 = t3822 + t3821;
        float t3824 = t3823 + 256.0;
        float t3825 = 6.283185 * t3821;
        float t3826 = (t3825 * 0.001953125);
        float t3827 = metal::cos(t3826);
        float t3828 = metal::sin(t3826);
        int t3829 = (int)t3823;
        int t3830 = (int)t3824;
        int t3831 = t2817 + t3829;
        float t3832 = memory[176319988 + t3831];
        int t3833 = t2817 + t3829;
        int t3834 = t3833 + 1024;
        float t3835 = memory[176319988 + t3834];
        int t3836 = t2817 + t3830;
        float t3837 = memory[176319988 + t3836];
        int t3838 = t2817 + t3830;
        int t3839 = t3838 + 1024;
        float t3840 = memory[176319988 + t3839];
        float t3841 = t3827 * t3837;
        float t3842 = t3828 * t3840;
        float t3843 = t3841 - t3842;
        float t3844 = t3827 * t3840;
        float t3845 = t3828 * t3837;
        float t3846 = t3844 + t3845;
        int t3847 = t2817 + t3829;
        float t3848 = t3832 + t3843;
        memory[176319988 + t3847] = t3848;
        int t3850 = t2817 + t3829;
        int t3851 = t3850 + 1024;
        float t3852 = t3835 + t3846;
        memory[176319988 + t3851] = t3852;
        int t3854 = t2817 + t3830;
        float t3855 = t3832 - t3843;
        memory[176319988 + t3854] = t3855;
        int t3857 = t2817 + t3830;
        int t3858 = t3857 + 1024;
        float t3859 = t3835 - t3846;
        memory[176319988 + t3858] = t3859;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3862 = 0; _pr3862 < 512; _pr3862++) {
        float t3863 = (float)_pr3862;
        float t3864 = (t3863 * 0.001953125);
        float t3865 = metal::floor(t3864);
        float t3866 = t3865 * 512.0;
        float t3867 = t3863 - t3866;
        float t3868 = t3865 * 1024.0;
        float t3869 = t3868 + t3867;
        float t3870 = t3869 + 512.0;
        float t3871 = 6.283185 * t3867;
        float t3872 = (t3871 * 0.0009765625);
        float t3873 = metal::cos(t3872);
        float t3874 = metal::sin(t3872);
        int t3875 = (int)t3869;
        int t3876 = (int)t3870;
        int t3877 = t2817 + t3875;
        float t3878 = memory[176319988 + t3877];
        int t3879 = t2817 + t3875;
        int t3880 = t3879 + 1024;
        float t3881 = memory[176319988 + t3880];
        int t3882 = t2817 + t3876;
        float t3883 = memory[176319988 + t3882];
        int t3884 = t2817 + t3876;
        int t3885 = t3884 + 1024;
        float t3886 = memory[176319988 + t3885];
        float t3887 = t3873 * t3883;
        float t3888 = t3874 * t3886;
        float t3889 = t3887 - t3888;
        float t3890 = t3873 * t3886;
        float t3891 = t3874 * t3883;
        float t3892 = t3890 + t3891;
        int t3893 = t2817 + t3875;
        float t3894 = t3878 + t3889;
        memory[176319988 + t3893] = t3894;
        int t3896 = t2817 + t3875;
        int t3897 = t3896 + 1024;
        float t3898 = t3881 + t3892;
        memory[176319988 + t3897] = t3898;
        int t3900 = t2817 + t3876;
        float t3901 = t3878 - t3889;
        memory[176319988 + t3900] = t3901;
        int t3903 = t2817 + t3876;
        int t3904 = t3903 + 1024;
        float t3905 = t3881 - t3892;
        memory[176319988 + t3904] = t3905;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3908 = 0; _pr3908 < 1024; _pr3908++) {
        int t3909 = t2817 + _pr3908;
        float t3910 = memory[176319988 + t3909];
        float t3911 = t3910 * 1.9036306e-06;
        float t3912 = memory[44980 + (int)_pr3908];
        int t3913 = t2818 + _pr3908;
        float t3914 = t3911 * t3912;
        memory[83996148 + t3913] = t3914;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t3918 = t[14*frameCount + id] > 0.0;
    if (t3918) {
      for (uint _pr3920 = 0; _pr3920 < 1024; _pr3920++) {
        int t3921 = t2818 + _pr3920;
        memory[50441716 + t3921] = 0.0;
        int t3923 = t2818 + _pr3920;
        memory[83996148 + t3923] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3944), value: global(3944)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1580) - handled in variable access */
    int t3927 = id;
    float t3928 = 0.0;
    for (uint t3929 = 0; t3929 < 1024; t3929++) {
      float t3930 = (float)t3929;
      float t3931 = (float)t3927;
      float t3932 = t3931 + t3930;
      int t3933 = 1023 - t3929;
      float t3934 = frameCount - 1.0;
      float t3935 = metal::min(t3932, t3934);
      int t3936 = (int)t3935;
      int t3937 = t3936 * 1024;
      int t3938 = t3937 + t3933;
      float t3939 = memory[50441716 + t3938];
      float t3940 = t3932 < frameCount;
      float t3941 = metal::select(0.0, t3939, t3940 > 0.0);
      float t3942 = t3928 + t3941;
      t3928 = t3942;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[17*frameCount + id] = (t3928 * 0.0013797212);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3962), value: global(3962)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1580) - handled in variable access */
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
      float t3957 = memory[83996148 + t3956];
      float t3958 = t3950 < frameCount;
      float t3959 = metal::select(0.0, t3957, t3958 > 0.0);
      float t3960 = t3946 + t3959;
      t3946 = t3960;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[18*frameCount + id] = (t3946 * 0.0013797212);
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
  float t5727 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5727)) {
    /* loadGlobal(540) - handled in variable access */
    int t3963 = id;
    int t3964 = t3963 / 61;
    uint _frameIndex = (uint)(t3964);
    int t3965 = t3964 * 61;
    int t3966 = t3963 - t3965;
    float t3967 = (t[12*frameCount + _frameIndex] * 3.7252903e-09);
    float t3968 = -0.5 * t3967;
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
    /* loadGlobal(519) - handled in variable access */
    /* loadGlobal(518) - handled in variable access */
    /* loadGlobal(465) - handled in variable access */
    int t3969 = id;
    int t3970 = t3969 * 1024;
    int t3971 = t3969 * 257;
    int t3972 = t3969 * 1024;
    float t3973 = t[11*frameCount + id] == 0.0;
    if (t3973) {
      for (uint _pr3975 = 0; _pr3975 < 257; _pr3975++) {
        int t3976 = t3971 + _pr3975;
        float t3977 = memory[37809652 + t3976];
        int t3978 = t3971 + _pr3975;
        float t3979 = memory[42020340 + t3978];
        int t3980 = t3970 + _pr3975;
        float t3981 = memory[4255220 + t3980];
        int t3982 = t3970 + _pr3975;
        int t3983 = t3982 + 512;
        float t3984 = memory[4255220 + t3983];
        int t3985 = t3970 + _pr3975;
        float t3986 = memory[21032436 + t3985];
        int t3987 = t3970 + _pr3975;
        int t3988 = t3987 + 512;
        float t3989 = memory[21032436 + t3988];
        float t3990 = t3977 - t3979;
        float t3991 = 2.0 * t3990;
        float t3992 = t3991 * 3.0517578e-05;
        float t3993 = t3977 - t3979;
        float t3994 = -2.0 * t3993;
        float t3995 = t3994 * 3.0517578e-05;
        float t3996 = metal::max(t3977, 1e-08);
        float t3997 = metal::max(t3979, 1e-08);
        float t3998 = t3992 * t3981;
        float t3999 = t3998 / t3996;
        float t4000 = t3992 * t3984;
        float t4001 = t4000 / t3996;
        float t4002 = t3995 * t3986;
        float t4003 = t4002 / t3997;
        float t4004 = t3995 * t3989;
        float t4005 = t4004 / t3997;
        int t4006 = t3972 + _pr3975;
        memory[50441716 + t4006] = t3999;
        int t4008 = t3972 + _pr3975;
        int t4009 = t4008 + 512;
        memory[50441716 + t4009] = t4001;
        int t4011 = t3972 + _pr3975;
        memory[83996148 + t4011] = t4003;
        int t4013 = t3972 + _pr3975;
        int t4014 = t4013 + 512;
        memory[83996148 + t4014] = t4005;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4017 = 0; _pr4017 < 255; _pr4017++) {
        int t4018 = _pr4017 + 257;
        int t4019 = t3972 + t4018;
        memory[50441716 + t4019] = 0.0;
        int t4021 = t3972 + t4018;
        int t4022 = t4021 + 512;
        memory[50441716 + t4022] = 0.0;
        int t4024 = t3972 + t4018;
        memory[83996148 + t4024] = 0.0;
        int t4026 = t3972 + t4018;
        int t4027 = t4026 + 512;
        memory[83996148 + t4027] = 0.0;
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
    /* loadGlobal(519) - handled in variable access */
    int t4031 = id;
    int t4032 = t4031 * 1024;
    int t4033 = t4031 * 512;
    float t4034 = t[11*frameCount + id] == 0.0;
    if (t4034) {
      for (uint t4036 = 0; t4036 < 512; t4036++) {
        float t4037 = (float)t4036;
        float t4038 = (t4037 - metal::floor(t4037 / 2.0) * 2.0);
        float t4039 = t4038;
        float t4040 = (t4037 * 0.5);
        float t4041 = metal::floor(t4040);
        float t4042 = t4039 * 2.0;
        float t4043 = (t4041 - metal::floor(t4041 / 2.0) * 2.0);
        float t4044 = t4042 + t4043;
        float t4045 = (t4041 * 0.5);
        float t4046 = metal::floor(t4045);
        float t4047 = t4044 * 2.0;
        float t4048 = (t4046 - metal::floor(t4046 / 2.0) * 2.0);
        float t4049 = t4047 + t4048;
        float t4050 = (t4046 * 0.5);
        float t4051 = metal::floor(t4050);
        float t4052 = t4049 * 2.0;
        float t4053 = (t4051 - metal::floor(t4051 / 2.0) * 2.0);
        float t4054 = t4052 + t4053;
        float t4055 = (t4051 * 0.5);
        float t4056 = metal::floor(t4055);
        float t4057 = t4054 * 2.0;
        float t4058 = (t4056 - metal::floor(t4056 / 2.0) * 2.0);
        float t4059 = t4057 + t4058;
        float t4060 = (t4056 * 0.5);
        float t4061 = metal::floor(t4060);
        float t4062 = t4059 * 2.0;
        float t4063 = (t4061 - metal::floor(t4061 / 2.0) * 2.0);
        float t4064 = t4062 + t4063;
        float t4065 = (t4061 * 0.5);
        float t4066 = metal::floor(t4065);
        float t4067 = t4064 * 2.0;
        float t4068 = (t4066 - metal::floor(t4066 / 2.0) * 2.0);
        float t4069 = t4067 + t4068;
        float t4070 = (t4066 * 0.5);
        float t4071 = metal::floor(t4070);
        float t4072 = t4069 * 2.0;
        float t4073 = (t4071 - metal::floor(t4071 / 2.0) * 2.0);
        float t4074 = t4072 + t4073;
        float t4075 = (t4071 * 0.5);
        float t4076 = metal::floor(t4075);
        float t4077 = t4074 * 2.0;
        float t4078 = (t4076 - metal::floor(t4076 / 2.0) * 2.0);
        float t4079 = t4077 + t4078;
        float t4080 = (t4076 * 0.5);
        float t4081 = metal::floor(t4080);
        float t4082 = (float)t4036;
        float t4083 = t4082 < t4079;
        int t4084 = (int)t4079;
        int t4085 = t4032 + t4036;
        float t4086 = memory[50441716 + t4085];
        int t4087 = t4032 + t4036;
        int t4088 = t4087 + 512;
        float t4089 = memory[50441716 + t4088];
        int t4090 = t4032 + t4084;
        float t4091 = memory[50441716 + t4090];
        int t4092 = t4032 + t4084;
        int t4093 = t4092 + 512;
        float t4094 = memory[50441716 + t4093];
        float t4095 = metal::select(t4086, t4091, t4083 > 0.0);
        float t4096 = metal::select(t4089, t4094, t4083 > 0.0);
        float t4097 = metal::select(t4091, t4086, t4083 > 0.0);
        float t4098 = metal::select(t4094, t4089, t4083 > 0.0);
        int t4099 = t4032 + t4036;
        memory[50441716 + t4099] = t4095;
        int t4101 = t4032 + t4036;
        int t4102 = t4101 + 512;
        memory[50441716 + t4102] = t4096;
        int t4104 = t4032 + t4084;
        memory[50441716 + t4104] = t4097;
        int t4106 = t4032 + t4084;
        int t4107 = t4106 + 512;
        memory[50441716 + t4107] = t4098;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4110 = 0; _pr4110 < 256; _pr4110++) {
        float t4111 = (float)_pr4110;
        float t4112 = t4111;
        float t4113 = metal::floor(t4112);
        float t4114 = t4113;
        float t4115 = t4111 - t4114;
        float t4116 = t4113 * 2.0;
        float t4117 = t4116 + t4115;
        float t4118 = t4117 + 1.0;
        float t4119 = 6.283185 * t4115;
        float t4120 = (t4119 * 0.5);
        float t4121 = metal::cos(t4120);
        float t4122 = metal::sin(t4120);
        int t4123 = (int)t4117;
        int t4124 = (int)t4118;
        int t4125 = t4032 + t4123;
        float t4126 = memory[50441716 + t4125];
        int t4127 = t4032 + t4123;
        int t4128 = t4127 + 512;
        float t4129 = memory[50441716 + t4128];
        int t4130 = t4032 + t4124;
        float t4131 = memory[50441716 + t4130];
        int t4132 = t4032 + t4124;
        int t4133 = t4132 + 512;
        float t4134 = memory[50441716 + t4133];
        float t4135 = t4121 * t4131;
        float t4136 = t4122 * t4134;
        float t4137 = t4135 - t4136;
        float t4138 = t4121 * t4134;
        float t4139 = t4122 * t4131;
        float t4140 = t4138 + t4139;
        int t4141 = t4032 + t4123;
        float t4142 = t4126 + t4137;
        memory[50441716 + t4141] = t4142;
        int t4144 = t4032 + t4123;
        int t4145 = t4144 + 512;
        float t4146 = t4129 + t4140;
        memory[50441716 + t4145] = t4146;
        int t4148 = t4032 + t4124;
        float t4149 = t4126 - t4137;
        memory[50441716 + t4148] = t4149;
        int t4151 = t4032 + t4124;
        int t4152 = t4151 + 512;
        float t4153 = t4129 - t4140;
        memory[50441716 + t4152] = t4153;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4156 = 0; _pr4156 < 256; _pr4156++) {
        float t4157 = (float)_pr4156;
        float t4158 = (t4157 * 0.5);
        float t4159 = metal::floor(t4158);
        float t4160 = t4159 * 2.0;
        float t4161 = t4157 - t4160;
        float t4162 = t4159 * 4.0;
        float t4163 = t4162 + t4161;
        float t4164 = t4163 + 2.0;
        float t4165 = 6.283185 * t4161;
        float t4166 = (t4165 * 0.25);
        float t4167 = metal::cos(t4166);
        float t4168 = metal::sin(t4166);
        int t4169 = (int)t4163;
        int t4170 = (int)t4164;
        int t4171 = t4032 + t4169;
        float t4172 = memory[50441716 + t4171];
        int t4173 = t4032 + t4169;
        int t4174 = t4173 + 512;
        float t4175 = memory[50441716 + t4174];
        int t4176 = t4032 + t4170;
        float t4177 = memory[50441716 + t4176];
        int t4178 = t4032 + t4170;
        int t4179 = t4178 + 512;
        float t4180 = memory[50441716 + t4179];
        float t4181 = t4167 * t4177;
        float t4182 = t4168 * t4180;
        float t4183 = t4181 - t4182;
        float t4184 = t4167 * t4180;
        float t4185 = t4168 * t4177;
        float t4186 = t4184 + t4185;
        int t4187 = t4032 + t4169;
        float t4188 = t4172 + t4183;
        memory[50441716 + t4187] = t4188;
        int t4190 = t4032 + t4169;
        int t4191 = t4190 + 512;
        float t4192 = t4175 + t4186;
        memory[50441716 + t4191] = t4192;
        int t4194 = t4032 + t4170;
        float t4195 = t4172 - t4183;
        memory[50441716 + t4194] = t4195;
        int t4197 = t4032 + t4170;
        int t4198 = t4197 + 512;
        float t4199 = t4175 - t4186;
        memory[50441716 + t4198] = t4199;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4202 = 0; _pr4202 < 256; _pr4202++) {
        float t4203 = (float)_pr4202;
        float t4204 = (t4203 * 0.25);
        float t4205 = metal::floor(t4204);
        float t4206 = t4205 * 4.0;
        float t4207 = t4203 - t4206;
        float t4208 = t4205 * 8.0;
        float t4209 = t4208 + t4207;
        float t4210 = t4209 + 4.0;
        float t4211 = 6.283185 * t4207;
        float t4212 = (t4211 * 0.125);
        float t4213 = metal::cos(t4212);
        float t4214 = metal::sin(t4212);
        int t4215 = (int)t4209;
        int t4216 = (int)t4210;
        int t4217 = t4032 + t4215;
        float t4218 = memory[50441716 + t4217];
        int t4219 = t4032 + t4215;
        int t4220 = t4219 + 512;
        float t4221 = memory[50441716 + t4220];
        int t4222 = t4032 + t4216;
        float t4223 = memory[50441716 + t4222];
        int t4224 = t4032 + t4216;
        int t4225 = t4224 + 512;
        float t4226 = memory[50441716 + t4225];
        float t4227 = t4213 * t4223;
        float t4228 = t4214 * t4226;
        float t4229 = t4227 - t4228;
        float t4230 = t4213 * t4226;
        float t4231 = t4214 * t4223;
        float t4232 = t4230 + t4231;
        int t4233 = t4032 + t4215;
        float t4234 = t4218 + t4229;
        memory[50441716 + t4233] = t4234;
        int t4236 = t4032 + t4215;
        int t4237 = t4236 + 512;
        float t4238 = t4221 + t4232;
        memory[50441716 + t4237] = t4238;
        int t4240 = t4032 + t4216;
        float t4241 = t4218 - t4229;
        memory[50441716 + t4240] = t4241;
        int t4243 = t4032 + t4216;
        int t4244 = t4243 + 512;
        float t4245 = t4221 - t4232;
        memory[50441716 + t4244] = t4245;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4248 = 0; _pr4248 < 256; _pr4248++) {
        float t4249 = (float)_pr4248;
        float t4250 = (t4249 * 0.125);
        float t4251 = metal::floor(t4250);
        float t4252 = t4251 * 8.0;
        float t4253 = t4249 - t4252;
        float t4254 = t4251 * 16.0;
        float t4255 = t4254 + t4253;
        float t4256 = t4255 + 8.0;
        float t4257 = 6.283185 * t4253;
        float t4258 = (t4257 * 0.0625);
        float t4259 = metal::cos(t4258);
        float t4260 = metal::sin(t4258);
        int t4261 = (int)t4255;
        int t4262 = (int)t4256;
        int t4263 = t4032 + t4261;
        float t4264 = memory[50441716 + t4263];
        int t4265 = t4032 + t4261;
        int t4266 = t4265 + 512;
        float t4267 = memory[50441716 + t4266];
        int t4268 = t4032 + t4262;
        float t4269 = memory[50441716 + t4268];
        int t4270 = t4032 + t4262;
        int t4271 = t4270 + 512;
        float t4272 = memory[50441716 + t4271];
        float t4273 = t4259 * t4269;
        float t4274 = t4260 * t4272;
        float t4275 = t4273 - t4274;
        float t4276 = t4259 * t4272;
        float t4277 = t4260 * t4269;
        float t4278 = t4276 + t4277;
        int t4279 = t4032 + t4261;
        float t4280 = t4264 + t4275;
        memory[50441716 + t4279] = t4280;
        int t4282 = t4032 + t4261;
        int t4283 = t4282 + 512;
        float t4284 = t4267 + t4278;
        memory[50441716 + t4283] = t4284;
        int t4286 = t4032 + t4262;
        float t4287 = t4264 - t4275;
        memory[50441716 + t4286] = t4287;
        int t4289 = t4032 + t4262;
        int t4290 = t4289 + 512;
        float t4291 = t4267 - t4278;
        memory[50441716 + t4290] = t4291;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4294 = 0; _pr4294 < 256; _pr4294++) {
        float t4295 = (float)_pr4294;
        float t4296 = (t4295 * 0.0625);
        float t4297 = metal::floor(t4296);
        float t4298 = t4297 * 16.0;
        float t4299 = t4295 - t4298;
        float t4300 = t4297 * 32.0;
        float t4301 = t4300 + t4299;
        float t4302 = t4301 + 16.0;
        float t4303 = 6.283185 * t4299;
        float t4304 = (t4303 * 0.03125);
        float t4305 = metal::cos(t4304);
        float t4306 = metal::sin(t4304);
        int t4307 = (int)t4301;
        int t4308 = (int)t4302;
        int t4309 = t4032 + t4307;
        float t4310 = memory[50441716 + t4309];
        int t4311 = t4032 + t4307;
        int t4312 = t4311 + 512;
        float t4313 = memory[50441716 + t4312];
        int t4314 = t4032 + t4308;
        float t4315 = memory[50441716 + t4314];
        int t4316 = t4032 + t4308;
        int t4317 = t4316 + 512;
        float t4318 = memory[50441716 + t4317];
        float t4319 = t4305 * t4315;
        float t4320 = t4306 * t4318;
        float t4321 = t4319 - t4320;
        float t4322 = t4305 * t4318;
        float t4323 = t4306 * t4315;
        float t4324 = t4322 + t4323;
        int t4325 = t4032 + t4307;
        float t4326 = t4310 + t4321;
        memory[50441716 + t4325] = t4326;
        int t4328 = t4032 + t4307;
        int t4329 = t4328 + 512;
        float t4330 = t4313 + t4324;
        memory[50441716 + t4329] = t4330;
        int t4332 = t4032 + t4308;
        float t4333 = t4310 - t4321;
        memory[50441716 + t4332] = t4333;
        int t4335 = t4032 + t4308;
        int t4336 = t4335 + 512;
        float t4337 = t4313 - t4324;
        memory[50441716 + t4336] = t4337;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4340 = 0; _pr4340 < 256; _pr4340++) {
        float t4341 = (float)_pr4340;
        float t4342 = (t4341 * 0.03125);
        float t4343 = metal::floor(t4342);
        float t4344 = t4343 * 32.0;
        float t4345 = t4341 - t4344;
        float t4346 = t4343 * 64.0;
        float t4347 = t4346 + t4345;
        float t4348 = t4347 + 32.0;
        float t4349 = 6.283185 * t4345;
        float t4350 = (t4349 * 0.015625);
        float t4351 = metal::cos(t4350);
        float t4352 = metal::sin(t4350);
        int t4353 = (int)t4347;
        int t4354 = (int)t4348;
        int t4355 = t4032 + t4353;
        float t4356 = memory[50441716 + t4355];
        int t4357 = t4032 + t4353;
        int t4358 = t4357 + 512;
        float t4359 = memory[50441716 + t4358];
        int t4360 = t4032 + t4354;
        float t4361 = memory[50441716 + t4360];
        int t4362 = t4032 + t4354;
        int t4363 = t4362 + 512;
        float t4364 = memory[50441716 + t4363];
        float t4365 = t4351 * t4361;
        float t4366 = t4352 * t4364;
        float t4367 = t4365 - t4366;
        float t4368 = t4351 * t4364;
        float t4369 = t4352 * t4361;
        float t4370 = t4368 + t4369;
        int t4371 = t4032 + t4353;
        float t4372 = t4356 + t4367;
        memory[50441716 + t4371] = t4372;
        int t4374 = t4032 + t4353;
        int t4375 = t4374 + 512;
        float t4376 = t4359 + t4370;
        memory[50441716 + t4375] = t4376;
        int t4378 = t4032 + t4354;
        float t4379 = t4356 - t4367;
        memory[50441716 + t4378] = t4379;
        int t4381 = t4032 + t4354;
        int t4382 = t4381 + 512;
        float t4383 = t4359 - t4370;
        memory[50441716 + t4382] = t4383;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4386 = 0; _pr4386 < 256; _pr4386++) {
        float t4387 = (float)_pr4386;
        float t4388 = (t4387 * 0.015625);
        float t4389 = metal::floor(t4388);
        float t4390 = t4389 * 64.0;
        float t4391 = t4387 - t4390;
        float t4392 = t4389 * 128.0;
        float t4393 = t4392 + t4391;
        float t4394 = t4393 + 64.0;
        float t4395 = 6.283185 * t4391;
        float t4396 = (t4395 * 0.0078125);
        float t4397 = metal::cos(t4396);
        float t4398 = metal::sin(t4396);
        int t4399 = (int)t4393;
        int t4400 = (int)t4394;
        int t4401 = t4032 + t4399;
        float t4402 = memory[50441716 + t4401];
        int t4403 = t4032 + t4399;
        int t4404 = t4403 + 512;
        float t4405 = memory[50441716 + t4404];
        int t4406 = t4032 + t4400;
        float t4407 = memory[50441716 + t4406];
        int t4408 = t4032 + t4400;
        int t4409 = t4408 + 512;
        float t4410 = memory[50441716 + t4409];
        float t4411 = t4397 * t4407;
        float t4412 = t4398 * t4410;
        float t4413 = t4411 - t4412;
        float t4414 = t4397 * t4410;
        float t4415 = t4398 * t4407;
        float t4416 = t4414 + t4415;
        int t4417 = t4032 + t4399;
        float t4418 = t4402 + t4413;
        memory[50441716 + t4417] = t4418;
        int t4420 = t4032 + t4399;
        int t4421 = t4420 + 512;
        float t4422 = t4405 + t4416;
        memory[50441716 + t4421] = t4422;
        int t4424 = t4032 + t4400;
        float t4425 = t4402 - t4413;
        memory[50441716 + t4424] = t4425;
        int t4427 = t4032 + t4400;
        int t4428 = t4427 + 512;
        float t4429 = t4405 - t4416;
        memory[50441716 + t4428] = t4429;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4432 = 0; _pr4432 < 256; _pr4432++) {
        float t4433 = (float)_pr4432;
        float t4434 = (t4433 * 0.0078125);
        float t4435 = metal::floor(t4434);
        float t4436 = t4435 * 128.0;
        float t4437 = t4433 - t4436;
        float t4438 = t4435 * 256.0;
        float t4439 = t4438 + t4437;
        float t4440 = t4439 + 128.0;
        float t4441 = 6.283185 * t4437;
        float t4442 = (t4441 * 0.00390625);
        float t4443 = metal::cos(t4442);
        float t4444 = metal::sin(t4442);
        int t4445 = (int)t4439;
        int t4446 = (int)t4440;
        int t4447 = t4032 + t4445;
        float t4448 = memory[50441716 + t4447];
        int t4449 = t4032 + t4445;
        int t4450 = t4449 + 512;
        float t4451 = memory[50441716 + t4450];
        int t4452 = t4032 + t4446;
        float t4453 = memory[50441716 + t4452];
        int t4454 = t4032 + t4446;
        int t4455 = t4454 + 512;
        float t4456 = memory[50441716 + t4455];
        float t4457 = t4443 * t4453;
        float t4458 = t4444 * t4456;
        float t4459 = t4457 - t4458;
        float t4460 = t4443 * t4456;
        float t4461 = t4444 * t4453;
        float t4462 = t4460 + t4461;
        int t4463 = t4032 + t4445;
        float t4464 = t4448 + t4459;
        memory[50441716 + t4463] = t4464;
        int t4466 = t4032 + t4445;
        int t4467 = t4466 + 512;
        float t4468 = t4451 + t4462;
        memory[50441716 + t4467] = t4468;
        int t4470 = t4032 + t4446;
        float t4471 = t4448 - t4459;
        memory[50441716 + t4470] = t4471;
        int t4473 = t4032 + t4446;
        int t4474 = t4473 + 512;
        float t4475 = t4451 - t4462;
        memory[50441716 + t4474] = t4475;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4478 = 0; _pr4478 < 256; _pr4478++) {
        float t4479 = (float)_pr4478;
        float t4480 = (t4479 * 0.00390625);
        float t4481 = metal::floor(t4480);
        float t4482 = t4481 * 256.0;
        float t4483 = t4479 - t4482;
        float t4484 = t4481 * 512.0;
        float t4485 = t4484 + t4483;
        float t4486 = t4485 + 256.0;
        float t4487 = 6.283185 * t4483;
        float t4488 = (t4487 * 0.001953125);
        float t4489 = metal::cos(t4488);
        float t4490 = metal::sin(t4488);
        int t4491 = (int)t4485;
        int t4492 = (int)t4486;
        int t4493 = t4032 + t4491;
        float t4494 = memory[50441716 + t4493];
        int t4495 = t4032 + t4491;
        int t4496 = t4495 + 512;
        float t4497 = memory[50441716 + t4496];
        int t4498 = t4032 + t4492;
        float t4499 = memory[50441716 + t4498];
        int t4500 = t4032 + t4492;
        int t4501 = t4500 + 512;
        float t4502 = memory[50441716 + t4501];
        float t4503 = t4489 * t4499;
        float t4504 = t4490 * t4502;
        float t4505 = t4503 - t4504;
        float t4506 = t4489 * t4502;
        float t4507 = t4490 * t4499;
        float t4508 = t4506 + t4507;
        int t4509 = t4032 + t4491;
        float t4510 = t4494 + t4505;
        memory[50441716 + t4509] = t4510;
        int t4512 = t4032 + t4491;
        int t4513 = t4512 + 512;
        float t4514 = t4497 + t4508;
        memory[50441716 + t4513] = t4514;
        int t4516 = t4032 + t4492;
        float t4517 = t4494 - t4505;
        memory[50441716 + t4516] = t4517;
        int t4519 = t4032 + t4492;
        int t4520 = t4519 + 512;
        float t4521 = t4497 - t4508;
        memory[50441716 + t4520] = t4521;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4524 = 0; _pr4524 < 512; _pr4524++) {
        int t4525 = t4032 + _pr4524;
        float t4526 = memory[50441716 + t4525];
        float t4527 = t4526 * 7.599708e-06;
        float t4528 = memory[25460 + (int)_pr4524];
        int t4529 = t4033 + _pr4524;
        float t4530 = t4527 * t4528;
        memory[117550580 + t4529] = t4530;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t4533 = 0; t4533 < 512; t4533++) {
        float t4534 = (float)t4533;
        float t4535 = (t4534 - metal::floor(t4534 / 2.0) * 2.0);
        float t4536 = t4535;
        float t4537 = (t4534 * 0.5);
        float t4538 = metal::floor(t4537);
        float t4539 = t4536 * 2.0;
        float t4540 = (t4538 - metal::floor(t4538 / 2.0) * 2.0);
        float t4541 = t4539 + t4540;
        float t4542 = (t4538 * 0.5);
        float t4543 = metal::floor(t4542);
        float t4544 = t4541 * 2.0;
        float t4545 = (t4543 - metal::floor(t4543 / 2.0) * 2.0);
        float t4546 = t4544 + t4545;
        float t4547 = (t4543 * 0.5);
        float t4548 = metal::floor(t4547);
        float t4549 = t4546 * 2.0;
        float t4550 = (t4548 - metal::floor(t4548 / 2.0) * 2.0);
        float t4551 = t4549 + t4550;
        float t4552 = (t4548 * 0.5);
        float t4553 = metal::floor(t4552);
        float t4554 = t4551 * 2.0;
        float t4555 = (t4553 - metal::floor(t4553 / 2.0) * 2.0);
        float t4556 = t4554 + t4555;
        float t4557 = (t4553 * 0.5);
        float t4558 = metal::floor(t4557);
        float t4559 = t4556 * 2.0;
        float t4560 = (t4558 - metal::floor(t4558 / 2.0) * 2.0);
        float t4561 = t4559 + t4560;
        float t4562 = (t4558 * 0.5);
        float t4563 = metal::floor(t4562);
        float t4564 = t4561 * 2.0;
        float t4565 = (t4563 - metal::floor(t4563 / 2.0) * 2.0);
        float t4566 = t4564 + t4565;
        float t4567 = (t4563 * 0.5);
        float t4568 = metal::floor(t4567);
        float t4569 = t4566 * 2.0;
        float t4570 = (t4568 - metal::floor(t4568 / 2.0) * 2.0);
        float t4571 = t4569 + t4570;
        float t4572 = (t4568 * 0.5);
        float t4573 = metal::floor(t4572);
        float t4574 = t4571 * 2.0;
        float t4575 = (t4573 - metal::floor(t4573 / 2.0) * 2.0);
        float t4576 = t4574 + t4575;
        float t4577 = (t4573 * 0.5);
        float t4578 = metal::floor(t4577);
        float t4579 = (float)t4533;
        float t4580 = t4579 < t4576;
        int t4581 = (int)t4576;
        int t4582 = t4032 + t4533;
        float t4583 = memory[83996148 + t4582];
        int t4584 = t4032 + t4533;
        int t4585 = t4584 + 512;
        float t4586 = memory[83996148 + t4585];
        int t4587 = t4032 + t4581;
        float t4588 = memory[83996148 + t4587];
        int t4589 = t4032 + t4581;
        int t4590 = t4589 + 512;
        float t4591 = memory[83996148 + t4590];
        float t4592 = metal::select(t4583, t4588, t4580 > 0.0);
        float t4593 = metal::select(t4586, t4591, t4580 > 0.0);
        float t4594 = metal::select(t4588, t4583, t4580 > 0.0);
        float t4595 = metal::select(t4591, t4586, t4580 > 0.0);
        int t4596 = t4032 + t4533;
        memory[83996148 + t4596] = t4592;
        int t4598 = t4032 + t4533;
        int t4599 = t4598 + 512;
        memory[83996148 + t4599] = t4593;
        int t4601 = t4032 + t4581;
        memory[83996148 + t4601] = t4594;
        int t4603 = t4032 + t4581;
        int t4604 = t4603 + 512;
        memory[83996148 + t4604] = t4595;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4607 = 0; _pr4607 < 256; _pr4607++) {
        float t4608 = (float)_pr4607;
        float t4609 = t4608;
        float t4610 = metal::floor(t4609);
        float t4611 = t4610;
        float t4612 = t4608 - t4611;
        float t4613 = t4610 * 2.0;
        float t4614 = t4613 + t4612;
        float t4615 = t4614 + 1.0;
        float t4616 = 6.283185 * t4612;
        float t4617 = (t4616 * 0.5);
        float t4618 = metal::cos(t4617);
        float t4619 = metal::sin(t4617);
        int t4620 = (int)t4614;
        int t4621 = (int)t4615;
        int t4622 = t4032 + t4620;
        float t4623 = memory[83996148 + t4622];
        int t4624 = t4032 + t4620;
        int t4625 = t4624 + 512;
        float t4626 = memory[83996148 + t4625];
        int t4627 = t4032 + t4621;
        float t4628 = memory[83996148 + t4627];
        int t4629 = t4032 + t4621;
        int t4630 = t4629 + 512;
        float t4631 = memory[83996148 + t4630];
        float t4632 = t4618 * t4628;
        float t4633 = t4619 * t4631;
        float t4634 = t4632 - t4633;
        float t4635 = t4618 * t4631;
        float t4636 = t4619 * t4628;
        float t4637 = t4635 + t4636;
        int t4638 = t4032 + t4620;
        float t4639 = t4623 + t4634;
        memory[83996148 + t4638] = t4639;
        int t4641 = t4032 + t4620;
        int t4642 = t4641 + 512;
        float t4643 = t4626 + t4637;
        memory[83996148 + t4642] = t4643;
        int t4645 = t4032 + t4621;
        float t4646 = t4623 - t4634;
        memory[83996148 + t4645] = t4646;
        int t4648 = t4032 + t4621;
        int t4649 = t4648 + 512;
        float t4650 = t4626 - t4637;
        memory[83996148 + t4649] = t4650;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4653 = 0; _pr4653 < 256; _pr4653++) {
        float t4654 = (float)_pr4653;
        float t4655 = (t4654 * 0.5);
        float t4656 = metal::floor(t4655);
        float t4657 = t4656 * 2.0;
        float t4658 = t4654 - t4657;
        float t4659 = t4656 * 4.0;
        float t4660 = t4659 + t4658;
        float t4661 = t4660 + 2.0;
        float t4662 = 6.283185 * t4658;
        float t4663 = (t4662 * 0.25);
        float t4664 = metal::cos(t4663);
        float t4665 = metal::sin(t4663);
        int t4666 = (int)t4660;
        int t4667 = (int)t4661;
        int t4668 = t4032 + t4666;
        float t4669 = memory[83996148 + t4668];
        int t4670 = t4032 + t4666;
        int t4671 = t4670 + 512;
        float t4672 = memory[83996148 + t4671];
        int t4673 = t4032 + t4667;
        float t4674 = memory[83996148 + t4673];
        int t4675 = t4032 + t4667;
        int t4676 = t4675 + 512;
        float t4677 = memory[83996148 + t4676];
        float t4678 = t4664 * t4674;
        float t4679 = t4665 * t4677;
        float t4680 = t4678 - t4679;
        float t4681 = t4664 * t4677;
        float t4682 = t4665 * t4674;
        float t4683 = t4681 + t4682;
        int t4684 = t4032 + t4666;
        float t4685 = t4669 + t4680;
        memory[83996148 + t4684] = t4685;
        int t4687 = t4032 + t4666;
        int t4688 = t4687 + 512;
        float t4689 = t4672 + t4683;
        memory[83996148 + t4688] = t4689;
        int t4691 = t4032 + t4667;
        float t4692 = t4669 - t4680;
        memory[83996148 + t4691] = t4692;
        int t4694 = t4032 + t4667;
        int t4695 = t4694 + 512;
        float t4696 = t4672 - t4683;
        memory[83996148 + t4695] = t4696;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4699 = 0; _pr4699 < 256; _pr4699++) {
        float t4700 = (float)_pr4699;
        float t4701 = (t4700 * 0.25);
        float t4702 = metal::floor(t4701);
        float t4703 = t4702 * 4.0;
        float t4704 = t4700 - t4703;
        float t4705 = t4702 * 8.0;
        float t4706 = t4705 + t4704;
        float t4707 = t4706 + 4.0;
        float t4708 = 6.283185 * t4704;
        float t4709 = (t4708 * 0.125);
        float t4710 = metal::cos(t4709);
        float t4711 = metal::sin(t4709);
        int t4712 = (int)t4706;
        int t4713 = (int)t4707;
        int t4714 = t4032 + t4712;
        float t4715 = memory[83996148 + t4714];
        int t4716 = t4032 + t4712;
        int t4717 = t4716 + 512;
        float t4718 = memory[83996148 + t4717];
        int t4719 = t4032 + t4713;
        float t4720 = memory[83996148 + t4719];
        int t4721 = t4032 + t4713;
        int t4722 = t4721 + 512;
        float t4723 = memory[83996148 + t4722];
        float t4724 = t4710 * t4720;
        float t4725 = t4711 * t4723;
        float t4726 = t4724 - t4725;
        float t4727 = t4710 * t4723;
        float t4728 = t4711 * t4720;
        float t4729 = t4727 + t4728;
        int t4730 = t4032 + t4712;
        float t4731 = t4715 + t4726;
        memory[83996148 + t4730] = t4731;
        int t4733 = t4032 + t4712;
        int t4734 = t4733 + 512;
        float t4735 = t4718 + t4729;
        memory[83996148 + t4734] = t4735;
        int t4737 = t4032 + t4713;
        float t4738 = t4715 - t4726;
        memory[83996148 + t4737] = t4738;
        int t4740 = t4032 + t4713;
        int t4741 = t4740 + 512;
        float t4742 = t4718 - t4729;
        memory[83996148 + t4741] = t4742;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4745 = 0; _pr4745 < 256; _pr4745++) {
        float t4746 = (float)_pr4745;
        float t4747 = (t4746 * 0.125);
        float t4748 = metal::floor(t4747);
        float t4749 = t4748 * 8.0;
        float t4750 = t4746 - t4749;
        float t4751 = t4748 * 16.0;
        float t4752 = t4751 + t4750;
        float t4753 = t4752 + 8.0;
        float t4754 = 6.283185 * t4750;
        float t4755 = (t4754 * 0.0625);
        float t4756 = metal::cos(t4755);
        float t4757 = metal::sin(t4755);
        int t4758 = (int)t4752;
        int t4759 = (int)t4753;
        int t4760 = t4032 + t4758;
        float t4761 = memory[83996148 + t4760];
        int t4762 = t4032 + t4758;
        int t4763 = t4762 + 512;
        float t4764 = memory[83996148 + t4763];
        int t4765 = t4032 + t4759;
        float t4766 = memory[83996148 + t4765];
        int t4767 = t4032 + t4759;
        int t4768 = t4767 + 512;
        float t4769 = memory[83996148 + t4768];
        float t4770 = t4756 * t4766;
        float t4771 = t4757 * t4769;
        float t4772 = t4770 - t4771;
        float t4773 = t4756 * t4769;
        float t4774 = t4757 * t4766;
        float t4775 = t4773 + t4774;
        int t4776 = t4032 + t4758;
        float t4777 = t4761 + t4772;
        memory[83996148 + t4776] = t4777;
        int t4779 = t4032 + t4758;
        int t4780 = t4779 + 512;
        float t4781 = t4764 + t4775;
        memory[83996148 + t4780] = t4781;
        int t4783 = t4032 + t4759;
        float t4784 = t4761 - t4772;
        memory[83996148 + t4783] = t4784;
        int t4786 = t4032 + t4759;
        int t4787 = t4786 + 512;
        float t4788 = t4764 - t4775;
        memory[83996148 + t4787] = t4788;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4791 = 0; _pr4791 < 256; _pr4791++) {
        float t4792 = (float)_pr4791;
        float t4793 = (t4792 * 0.0625);
        float t4794 = metal::floor(t4793);
        float t4795 = t4794 * 16.0;
        float t4796 = t4792 - t4795;
        float t4797 = t4794 * 32.0;
        float t4798 = t4797 + t4796;
        float t4799 = t4798 + 16.0;
        float t4800 = 6.283185 * t4796;
        float t4801 = (t4800 * 0.03125);
        float t4802 = metal::cos(t4801);
        float t4803 = metal::sin(t4801);
        int t4804 = (int)t4798;
        int t4805 = (int)t4799;
        int t4806 = t4032 + t4804;
        float t4807 = memory[83996148 + t4806];
        int t4808 = t4032 + t4804;
        int t4809 = t4808 + 512;
        float t4810 = memory[83996148 + t4809];
        int t4811 = t4032 + t4805;
        float t4812 = memory[83996148 + t4811];
        int t4813 = t4032 + t4805;
        int t4814 = t4813 + 512;
        float t4815 = memory[83996148 + t4814];
        float t4816 = t4802 * t4812;
        float t4817 = t4803 * t4815;
        float t4818 = t4816 - t4817;
        float t4819 = t4802 * t4815;
        float t4820 = t4803 * t4812;
        float t4821 = t4819 + t4820;
        int t4822 = t4032 + t4804;
        float t4823 = t4807 + t4818;
        memory[83996148 + t4822] = t4823;
        int t4825 = t4032 + t4804;
        int t4826 = t4825 + 512;
        float t4827 = t4810 + t4821;
        memory[83996148 + t4826] = t4827;
        int t4829 = t4032 + t4805;
        float t4830 = t4807 - t4818;
        memory[83996148 + t4829] = t4830;
        int t4832 = t4032 + t4805;
        int t4833 = t4832 + 512;
        float t4834 = t4810 - t4821;
        memory[83996148 + t4833] = t4834;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4837 = 0; _pr4837 < 256; _pr4837++) {
        float t4838 = (float)_pr4837;
        float t4839 = (t4838 * 0.03125);
        float t4840 = metal::floor(t4839);
        float t4841 = t4840 * 32.0;
        float t4842 = t4838 - t4841;
        float t4843 = t4840 * 64.0;
        float t4844 = t4843 + t4842;
        float t4845 = t4844 + 32.0;
        float t4846 = 6.283185 * t4842;
        float t4847 = (t4846 * 0.015625);
        float t4848 = metal::cos(t4847);
        float t4849 = metal::sin(t4847);
        int t4850 = (int)t4844;
        int t4851 = (int)t4845;
        int t4852 = t4032 + t4850;
        float t4853 = memory[83996148 + t4852];
        int t4854 = t4032 + t4850;
        int t4855 = t4854 + 512;
        float t4856 = memory[83996148 + t4855];
        int t4857 = t4032 + t4851;
        float t4858 = memory[83996148 + t4857];
        int t4859 = t4032 + t4851;
        int t4860 = t4859 + 512;
        float t4861 = memory[83996148 + t4860];
        float t4862 = t4848 * t4858;
        float t4863 = t4849 * t4861;
        float t4864 = t4862 - t4863;
        float t4865 = t4848 * t4861;
        float t4866 = t4849 * t4858;
        float t4867 = t4865 + t4866;
        int t4868 = t4032 + t4850;
        float t4869 = t4853 + t4864;
        memory[83996148 + t4868] = t4869;
        int t4871 = t4032 + t4850;
        int t4872 = t4871 + 512;
        float t4873 = t4856 + t4867;
        memory[83996148 + t4872] = t4873;
        int t4875 = t4032 + t4851;
        float t4876 = t4853 - t4864;
        memory[83996148 + t4875] = t4876;
        int t4878 = t4032 + t4851;
        int t4879 = t4878 + 512;
        float t4880 = t4856 - t4867;
        memory[83996148 + t4879] = t4880;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4883 = 0; _pr4883 < 256; _pr4883++) {
        float t4884 = (float)_pr4883;
        float t4885 = (t4884 * 0.015625);
        float t4886 = metal::floor(t4885);
        float t4887 = t4886 * 64.0;
        float t4888 = t4884 - t4887;
        float t4889 = t4886 * 128.0;
        float t4890 = t4889 + t4888;
        float t4891 = t4890 + 64.0;
        float t4892 = 6.283185 * t4888;
        float t4893 = (t4892 * 0.0078125);
        float t4894 = metal::cos(t4893);
        float t4895 = metal::sin(t4893);
        int t4896 = (int)t4890;
        int t4897 = (int)t4891;
        int t4898 = t4032 + t4896;
        float t4899 = memory[83996148 + t4898];
        int t4900 = t4032 + t4896;
        int t4901 = t4900 + 512;
        float t4902 = memory[83996148 + t4901];
        int t4903 = t4032 + t4897;
        float t4904 = memory[83996148 + t4903];
        int t4905 = t4032 + t4897;
        int t4906 = t4905 + 512;
        float t4907 = memory[83996148 + t4906];
        float t4908 = t4894 * t4904;
        float t4909 = t4895 * t4907;
        float t4910 = t4908 - t4909;
        float t4911 = t4894 * t4907;
        float t4912 = t4895 * t4904;
        float t4913 = t4911 + t4912;
        int t4914 = t4032 + t4896;
        float t4915 = t4899 + t4910;
        memory[83996148 + t4914] = t4915;
        int t4917 = t4032 + t4896;
        int t4918 = t4917 + 512;
        float t4919 = t4902 + t4913;
        memory[83996148 + t4918] = t4919;
        int t4921 = t4032 + t4897;
        float t4922 = t4899 - t4910;
        memory[83996148 + t4921] = t4922;
        int t4924 = t4032 + t4897;
        int t4925 = t4924 + 512;
        float t4926 = t4902 - t4913;
        memory[83996148 + t4925] = t4926;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4929 = 0; _pr4929 < 256; _pr4929++) {
        float t4930 = (float)_pr4929;
        float t4931 = (t4930 * 0.0078125);
        float t4932 = metal::floor(t4931);
        float t4933 = t4932 * 128.0;
        float t4934 = t4930 - t4933;
        float t4935 = t4932 * 256.0;
        float t4936 = t4935 + t4934;
        float t4937 = t4936 + 128.0;
        float t4938 = 6.283185 * t4934;
        float t4939 = (t4938 * 0.00390625);
        float t4940 = metal::cos(t4939);
        float t4941 = metal::sin(t4939);
        int t4942 = (int)t4936;
        int t4943 = (int)t4937;
        int t4944 = t4032 + t4942;
        float t4945 = memory[83996148 + t4944];
        int t4946 = t4032 + t4942;
        int t4947 = t4946 + 512;
        float t4948 = memory[83996148 + t4947];
        int t4949 = t4032 + t4943;
        float t4950 = memory[83996148 + t4949];
        int t4951 = t4032 + t4943;
        int t4952 = t4951 + 512;
        float t4953 = memory[83996148 + t4952];
        float t4954 = t4940 * t4950;
        float t4955 = t4941 * t4953;
        float t4956 = t4954 - t4955;
        float t4957 = t4940 * t4953;
        float t4958 = t4941 * t4950;
        float t4959 = t4957 + t4958;
        int t4960 = t4032 + t4942;
        float t4961 = t4945 + t4956;
        memory[83996148 + t4960] = t4961;
        int t4963 = t4032 + t4942;
        int t4964 = t4963 + 512;
        float t4965 = t4948 + t4959;
        memory[83996148 + t4964] = t4965;
        int t4967 = t4032 + t4943;
        float t4968 = t4945 - t4956;
        memory[83996148 + t4967] = t4968;
        int t4970 = t4032 + t4943;
        int t4971 = t4970 + 512;
        float t4972 = t4948 - t4959;
        memory[83996148 + t4971] = t4972;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4975 = 0; _pr4975 < 256; _pr4975++) {
        float t4976 = (float)_pr4975;
        float t4977 = (t4976 * 0.00390625);
        float t4978 = metal::floor(t4977);
        float t4979 = t4978 * 256.0;
        float t4980 = t4976 - t4979;
        float t4981 = t4978 * 512.0;
        float t4982 = t4981 + t4980;
        float t4983 = t4982 + 256.0;
        float t4984 = 6.283185 * t4980;
        float t4985 = (t4984 * 0.001953125);
        float t4986 = metal::cos(t4985);
        float t4987 = metal::sin(t4985);
        int t4988 = (int)t4982;
        int t4989 = (int)t4983;
        int t4990 = t4032 + t4988;
        float t4991 = memory[83996148 + t4990];
        int t4992 = t4032 + t4988;
        int t4993 = t4992 + 512;
        float t4994 = memory[83996148 + t4993];
        int t4995 = t4032 + t4989;
        float t4996 = memory[83996148 + t4995];
        int t4997 = t4032 + t4989;
        int t4998 = t4997 + 512;
        float t4999 = memory[83996148 + t4998];
        float t5000 = t4986 * t4996;
        float t5001 = t4987 * t4999;
        float t5002 = t5000 - t5001;
        float t5003 = t4986 * t4999;
        float t5004 = t4987 * t4996;
        float t5005 = t5003 + t5004;
        int t5006 = t4032 + t4988;
        float t5007 = t4991 + t5002;
        memory[83996148 + t5006] = t5007;
        int t5009 = t4032 + t4988;
        int t5010 = t5009 + 512;
        float t5011 = t4994 + t5005;
        memory[83996148 + t5010] = t5011;
        int t5013 = t4032 + t4989;
        float t5014 = t4991 - t5002;
        memory[83996148 + t5013] = t5014;
        int t5016 = t4032 + t4989;
        int t5017 = t5016 + 512;
        float t5018 = t4994 - t5005;
        memory[83996148 + t5017] = t5018;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr5021 = 0; _pr5021 < 512; _pr5021++) {
        int t5022 = t4032 + _pr5021;
        float t5023 = memory[83996148 + t5022];
        float t5024 = t5023 * 7.599708e-06;
        float t5025 = memory[25460 + (int)_pr5021];
        int t5026 = t4033 + _pr5021;
        float t5027 = t5024 * t5025;
        memory[125955572 + t5026] = t5027;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t5031 = t[11*frameCount + id] > 0.0;
    if (t5031) {
      for (uint _pr5033 = 0; _pr5033 < 512; _pr5033++) {
        int t5034 = t4033 + _pr5033;
        memory[117550580 + t5034] = 0.0;
        int t5036 = t4033 + _pr5033;
        memory[125955572 + t5036] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5057), value: global(5057)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(519) - handled in variable access */
    int t5040 = id;
    float t5041 = 0.0;
    for (uint t5042 = 0; t5042 < 512; t5042++) {
      float t5043 = (float)t5042;
      float t5044 = (float)t5040;
      float t5045 = t5044 + t5043;
      int t5046 = 511 - t5042;
      float t5047 = frameCount - 1.0;
      float t5048 = metal::min(t5045, t5047);
      int t5049 = (int)t5048;
      int t5050 = t5049 * 512;
      int t5051 = t5050 + t5046;
      float t5052 = memory[117550580 + t5051];
      float t5053 = t5045 < frameCount;
      float t5054 = metal::select(0.0, t5052, t5053 > 0.0);
      float t5055 = t5041 + t5054;
      t5041 = t5055;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[19*frameCount + id] = (t5041 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5075), value: global(5075)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(519) - handled in variable access */
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
      float t5070 = memory[125955572 + t5069];
      float t5071 = t5063 < frameCount;
      float t5072 = metal::select(0.0, t5070, t5071 > 0.0);
      float t5073 = t5059 + t5072;
      t5059 = t5073;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[20*frameCount + id] = (t5059 * 0.0027567567);
  }
  #pragma clang diagnostic pop
}



// KERNEL 29
// Kind: simd
// ThreadCountScale nil
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5096), value: global(5096)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5095), value: global(5095)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5080), value: global(5080)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5075) - handled in variable access */
    /* loadGlobal(5057) - handled in variable access */
    /* loadGlobal(3962) - handled in variable access */
    /* loadGlobal(3944) - handled in variable access */
    /* loadGlobal(464) - handled in variable access */
    /* loadGlobal(463) - handled in variable access */
    /* loadGlobal(445) - handled in variable access */
    /* loadGlobal(315) - handled in variable access */
    float t5076 = t[17*frameCount + id] + t[19*frameCount + id];
    float t5077 = t[18*frameCount + id] + t[20*frameCount + id];
    float t5078 = 0.015625 * t5076;
    float t5079 = t[7*frameCount + id] * t5076;
    t[21*frameCount + id] = t[6*frameCount + id] * t5078;
    float t5081 = t[5*frameCount + id] * t5078;
    float t5082 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t5083 = t5082 < 0.0;
    float t5084 = t5082 + 61.0;
    float t5085 = metal::select(t5082, t5084, t5083 > 0.0);
    float t5086 = t5085;
    float t5087 = metal::floor(t5086);
    float t5088 = t5086 - t5087;
    float t5089 = t5087 + 1.0;
    float t5090 = t5089 >= 61.0;
    float t5091 = metal::select(t5089, 0.0, t5090 > 0.0);
    float t5092 = 1.0 - t5088;
    float t5093 = t5081 * t5092;
    float t5094 = t5081 * t5088;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5087], t5093, metal::memory_order_relaxed);
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5091], t5094, metal::memory_order_relaxed);
    for (uint t5097 = 0; t5097 < 61; t5097++) {
      float t5098 = memory[209874420 + (int)t5097];
      float t5099 = memory[60724 + (int)t5097];
      float t5100 = t5098 / t5099;
      float t5101 = memory[60724 + (int)t5097];
      float t5102 = memory[60724 + (int)t5097];
      float t5103 = t5101 * t5102;
      float t5104 = 1.0 / t5103;
      float t5105 = memory[209874420 + (int)t5097];
      float t5106 = t5105 * -1.0;
      float t5107 = t5106 * t5104;
      float t5108 = t5100 + t5107;
      float t5109 = memory[60596 + (int)t5097];
      float t5110 = metal::exp(t5109);
      float t5111 = t5110 * t5107;
      float t5112 = -1.0 * t5111;
      int t5113 = id;
      int t5114 = t5113 * 61;
      int t5115 = t5114 + t5097;
      memory[1109492 + t5115] = t5112;
      float t5117 = memory[60660 + (int)t5097];
      float t5118 = t5117 * t5111;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5119 = 0; t5119 < 1; t5119++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=201, axis=0, in=[61, 1], out=[1], inFA=true, outFA=true), value: empty) */
      float t5120 = 0.0;
      int t5121 = t5119;
      int t5122 = t5121;
      int t5123 = t5119 - t5122;
      int t5124 = t5121;
      int t5125 = t5124;
      for (uint t5126 = 0; t5126 < 61; t5126++) {
        int t5127 = t5126;
        int t5128 = t5125 + t5127;
        int t5129 = id;
        int t5130 = t5129 * 61;
        int t5131 = t5130 + t5128;
        float t5132 = memory[1109492 + t5131];
        float t5133 = t5120 + t5132;
        t5120 = t5133;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5135 = id;
      int t5136 = t5135;
      int t5137 = t5136 + t5119;
      memory[60916 + t5137] = t5120;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
    for (uint t5139 = 0; t5139 < 3904; t5139++) {
      /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=203, axis=2, in=[61, 1], out=[61, 1, 64], inFA=true, outFA=true), value: empty) */
      int t5140 = t5139 / 64;
      int t5141 = t5140 % 61;
      int t5142 = t5141 * 1;
      int t5143 = 0 + t5142;
      int t5144 = t5139 / 64;
      int t5145 = t5144 % 1;
      int t5146 = t5145 * 1;
      int t5147 = t5143 + t5146;
      float t5148 = (float)t5147;
      int t5149 = id;
      int t5150 = t5149 * 61;
      float t5151 = t5150 + t5148;
      int t5152 = (int)t5151;
      float t5153 = memory[1109492 + t5152];
      float t5154 = (float)t5139;
      int t5155 = id;
      int t5156 = t5155 * 3904;
      float t5157 = t5156 + t5154;
      int t5158 = (int)t5157;
      memory[273837620 + t5158] = t5153;
      int t5160 = t5139 / 64;
      int t5161 = t5160 * 64;
      int t5162 = t5139 - t5161;
      int t5163 = t5162 / 64;
      int t5164 = t5163 * 64;
      int t5165 = t5162 - t5164;
      int t5166 = t5165 / 64;
      int t5167 = t5166 * 64;
      int t5168 = t5165 - t5167;
      float t5169 = memory[8576 + t5168];
      int t5170 = id;
      int t5171 = t5170 * 3904;
      int t5172 = t5171 + t5139;
      float t5173 = memory[273837620 + t5172];
      float t5174 = t5169 * t5173;
      int t5175 = id;
      int t5176 = t5175 * 3904;
      int t5177 = t5176 + t5139;
      memory[209874484 + t5177] = t5174;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5179 = 0; t5179 < 64; t5179++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=206, axis=0, in=[61, 1, 64], out=[1, 64], inFA=true, outFA=true), value: empty) */
      float t5180 = 0.0;
      int t5181 = t5179 / 64;
      int t5182 = t5181 * 64;
      int t5183 = t5179 - t5182;
      int t5184 = t5183;
      int t5185 = t5184;
      int t5186 = t5183 - t5185;
      int t5187 = t5181 * 64;
      int t5188 = t5187;
      int t5189 = t5184;
      int t5190 = t5188 + t5189;
      for (uint t5191 = 0; t5191 < 61; t5191++) {
        int t5192 = t5191 * 64;
        int t5193 = t5190 + t5192;
        int t5194 = t5191 * 64;
        int t5195 = t5194 + t5184;
        float t5196 = memory[41076 + t5195];
        float t5197 = t5191 + 0.0;
        int t5198 = id;
        int t5199 = t5198 * 61;
        float t5200 = t5199 + t5197;
        int t5201 = (int)t5200;
        float t5202 = memory[1109492 + t5201];
        float t5203 = t5196 * t5202;
        float t5204 = t5180 + t5203;
        t5180 = t5204;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5206 = id;
      int t5207 = t5206 * 64;
      int t5208 = t5207 + t5179;
      memory[37809652 + t5208] = t5180;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5210), value: global(5210)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5080) - handled in variable access */
    /* loadGlobal(437) - handled in variable access */
    /* loadGlobal(355) - handled in variable access */
    t[24*frameCount + id] = t[3*frameCount + id] * t[21*frameCount + id];
    float t5211 = t[4*frameCount + id] * t[21*frameCount + id];
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
  float t5728 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5728)) {
    /* loadGlobal(5210) - handled in variable access */
    int t5212 = id;
    int t5213 = t5212 / 64;
    uint _frameIndex = (uint)(t5213);
    int t5214 = t5213 * 64;
    int t5215 = t5212 - t5214;
    int t5216 = t5213 * 64;
    int t5217 = t5216 + t5215;
    memory[1109492 + t5217] = t[24*frameCount + _frameIndex];
    int t5219 = _frameIndex;
    int t5220 = t5219 * 64;
    int t5221 = t5220 + t5215;
    float t5222 = memory[2158068 + t5221];
    int t5223 = _frameIndex;
    int t5224 = t5223 * 64;
    int t5225 = t5224 + t5215;
    float t5226 = memory[1109492 + t5225];
    float t5227 = t5222 * t5226;
    int t5228 = _frameIndex;
    int t5229 = t5228 * 64;
    int t5230 = t5229 + t5215;
    float t5231 = memory[3206644 + t5230];
    int t5232 = _frameIndex;
    int t5233 = t5232 * 64;
    int t5234 = t5233 + t5215;
    float t5235 = memory[1109492 + t5234];
    float t5236 = t5231 * t5235;
    int t5237 = _frameIndex;
    int t5238 = t5237 * 64;
    int t5239 = t5238 + t5215;
    memory[42020340 + t5239] = t5236;
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
  float t5729 = frameCount * 3904.0;
  if (id >= 0 && id < (uint)(t5729)) {
    /* loadGlobal(315) - handled in variable access */
    int t5241 = id;
    int t5242 = t5241 / 3904;
    uint _frameIndex = (uint)(t5242);
    int t5243 = t5242 * 3904;
    int t5244 = t5241 - t5243;
    float t5245 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5246 = t5245 < 0.0;
    float t5247 = t5245 + 61.0;
    float t5248 = metal::select(t5245, t5247, t5246 > 0.0);
    float t5249 = metal::floor(t5248);
    float t5250 = t5249 + 1.0;
    float t5251 = t5250 >= 61.0;
    float t5252 = metal::select(t5250, 0.0, t5251 > 0.0);
    float t5253 = t5248 - t5249;
    int t5254 = _frameIndex;
    memory[3206644 + t5254] = t5249;
    memory[46231028 + t5254] = t5253;
    float t5257 = t5254 + 16384.0;
    int t5258 = (int)t5257;
    memory[3206644 + t5258] = t5252;
    float t5260 = 1.0 - t5253;
    float t5261 = t5254 * 64.0;
    for (uint _pr5262 = 0; _pr5262 < 64; _pr5262++) {
      float t5263 = (float)_pr5262;
      float t5264 = t5261 + t5263;
      int t5265 = (int)t5264;
      float t5266 = memory[42020340 + t5265];
      float t5267 = t5261 + t5263;
      float t5268 = t5266 * t5260;
      int t5269 = (int)t5267;
      memory[1109492 + t5269] = t5268;
      float t5271 = t5266 * t5253;
      int t5272 = (int)t5267;
      memory[2158068 + t5272] = t5271;
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
  if (id < 3904) { uint _pr5275 = id;
    int t5276 = _pr5275 / 64;
    int t5277 = t5276 * 64;
    int t5278 = _pr5275 - t5277;
    float t5279 = (float)t5276;
    float t5280 = (float)t5278;
    float t5281 = 0.0;
    for (uint t5282 = 0; t5282 < 16384; t5282++) {
      float t5283 = (float)t5282;
      float t5284 = t5283 < frameCount;
      float t5285 = t5283 * 64.0;
      float t5286 = t5285 + t5280;
      float t5287 = memory[3206644 + (int)t5282];
      float t5288 = t5287 - t5279;
      float t5289 = metal::abs(t5288);
      float t5290 = t5289 < 0.5;
      int t5291 = (int)t5286;
      float t5292 = memory[1109492 + t5291];
      float t5293 = t5284 * t5290;
      float t5294 = t5293 > 0.0;
      float t5295 = metal::select(0.0, t5292, t5294 > 0.0);
      float t5296 = t5281 + t5295;
      t5281 = t5296;
      float t5297 = t5283 + 16384.0;
      int t5298 = (int)t5297;
      float t5299 = memory[3206644 + t5298];
      float t5300 = t5299 - t5279;
      float t5301 = metal::abs(t5300);
      float t5302 = t5301 < 0.5;
      int t5303 = (int)t5286;
      float t5304 = memory[2158068 + t5303];
      float t5305 = t5284 * t5302;
      float t5306 = t5305 > 0.0;
      float t5307 = metal::select(0.0, t5304, t5306 > 0.0);
      float t5308 = t5281 + t5307;
      t5281 = t5308;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t5310 = t5279 * 64.0;
    float t5311 = t5310 + t5280;
    int t5312 = (int)t5311;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[337800756 + t5312], t5281, metal::memory_order_relaxed);
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
    for (uint t5315 = 0; t5315 < 3904; t5315++) {
      float t5316 = memory[337800756 + (int)t5315];
      float t5317 = memory[48884 + (int)t5315];
      float t5318 = t5316 / t5317;
      float t5319 = memory[48884 + (int)t5315];
      float t5320 = memory[48884 + (int)t5315];
      float t5321 = t5319 * t5320;
      float t5322 = 1.0 / t5321;
      float t5323 = memory[337800756 + (int)t5315];
      float t5324 = t5323 * -1.0;
      float t5325 = t5324 * t5322;
      float t5326 = t5318 + t5325;
      float t5327 = memory[56692 + (int)t5315];
      float t5328 = metal::exp(t5327);
      float t5329 = t5328 * t5325;
      float t5330 = -1.0 * t5329;
      int t5331 = id;
      int t5332 = t5331 * 3904;
      int t5333 = t5332 + t5315;
      memory[273837620 + t5333] = t5330;
      float t5335 = memory[52788 + (int)t5315];
      float t5336 = t5335 * t5329;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5337 = 0; t5337 < 64; t5337++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=232, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5338 = 0.0;
      int t5339 = t5337;
      int t5340 = t5339;
      int t5341 = t5337 - t5340;
      int t5342 = t5339;
      int t5343 = t5342;
      for (uint t5344 = 0; t5344 < 61; t5344++) {
        int t5345 = t5344 * 64;
        int t5346 = t5343 + t5345;
        int t5347 = id;
        int t5348 = t5347 * 3904;
        int t5349 = t5348 + t5346;
        float t5350 = memory[273837620 + t5349];
        float t5351 = t5338 + t5350;
        t5338 = t5351;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5353 = id;
      int t5354 = t5353 * 64;
      int t5355 = t5354 + t5337;
      memory[1109492 + t5355] = t5338;
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
    for (uint t5357 = 0; t5357 < 3904; t5357++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=237, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5358 = 0.0;
      int t5359 = t5357 / 64;
      int t5360 = t5359 * 64;
      int t5361 = t5357 - t5360;
      int t5362 = t5361;
      int t5363 = t5362;
      int t5364 = t5361 - t5363;
      int t5365 = t5359 * 4096;
      int t5366 = t5365;
      int t5367 = t5362;
      int t5368 = t5366 + t5367;
      for (uint t5369 = 0; t5369 < 64; t5369++) {
        int t5370 = t5369 * 64;
        int t5371 = t5368 + t5370;
        int t5372 = t5369 * 64;
        int t5373 = t5372 + t5362;
        int t5374 = t5373 / 64;
        int t5375 = t5374 * 64;
        int t5376 = t5373 - t5375;
        int t5377 = t5376 * 64;
        int t5378 = t5374 + t5377;
        float t5379 = memory[4416 + t5378];
        int t5380 = t5359 * 64;
        int t5381 = t5380 + t5369;
        int t5382 = id;
        int t5383 = t5382 * 3904;
        int t5384 = t5383 + t5381;
        float t5385 = memory[273837620 + t5384];
        float t5386 = t5379 * t5385;
        float t5387 = t5358 + t5386;
        t5358 = t5387;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5389 = id;
      int t5390 = t5389 * 3904;
      int t5391 = t5390 + t5357;
      memory[404913524 + t5391] = t5358;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5393 = 0; t5393 < 4096; t5393++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=239, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5394 = 0.0;
      int t5395 = t5393 / 64;
      int t5396 = t5395 * 64;
      int t5397 = t5393 - t5396;
      int t5398 = t5397;
      int t5399 = t5398;
      int t5400 = t5397 - t5399;
      int t5401 = t5395 * 64;
      int t5402 = t5401;
      int t5403 = t5398;
      int t5404 = t5402 + t5403;
      for (uint t5405 = 0; t5405 < 61; t5405++) {
        int t5406 = t5405 * 4096;
        int t5407 = t5404 + t5406;
        int t5408 = t5405 * 64;
        int t5409 = t5408 + t5398;
        float t5410 = memory[41076 + t5409];
        int t5411 = t5405 * 64;
        int t5412 = t5411 + t5395;
        int t5413 = id;
        int t5414 = t5413 * 3904;
        int t5415 = t5414 + t5412;
        float t5416 = memory[273837620 + t5415];
        float t5417 = t5410 * t5416;
        float t5418 = t5394 + t5417;
        t5394 = t5418;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5420 = id;
      int t5421 = t5420 * 4096;
      int t5422 = t5421 + t5393;
      memory[337804660 + t5422] = t5394;
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
    for (uint t5424 = 0; t5424 < 3904; t5424++) {
      int t5425 = id;
      int t5426 = t5425 * 3904;
      int t5427 = t5426 + t5424;
      float t5428 = memory[209874484 + t5427];
      int t5429 = id;
      int t5430 = t5429 * 3904;
      int t5431 = t5430 + t5424;
      float t5432 = memory[404913524 + t5431];
      float t5433 = t5428 + t5432;
      float t5434 = memory[37172 + (int)t5424];
      float t5435 = metal::tanh(t5434);
      float t5436 = t5435 * t5435;
      float t5437 = 1.0 - t5436;
      float t5438 = t5437 * t5433;
      int t5439 = id;
      int t5440 = t5439 * 3904;
      int t5441 = t5440 + t5424;
      memory[468876660 + t5441] = t5438;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5443 = 0; t5443 < 64; t5443++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=250, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5444 = 0.0;
      int t5445 = t5443;
      int t5446 = t5445;
      int t5447 = t5443 - t5446;
      int t5448 = t5445;
      int t5449 = t5448;
      for (uint t5450 = 0; t5450 < 61; t5450++) {
        int t5451 = t5450 * 64;
        int t5452 = t5449 + t5451;
        int t5453 = t5450 * 64;
        int t5454 = t5453 + t5445;
        float t5455 = memory[25460 + t5454];
        int t5456 = t5450 * 64;
        int t5457 = t5456 + t5445;
        int t5458 = id;
        int t5459 = t5458 * 3904;
        int t5460 = t5459 + t5457;
        float t5461 = memory[273837620 + t5460];
        float t5462 = t5455 * t5461;
        float t5463 = t5444 + t5462;
        t5444 = t5463;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5465 = id;
      int t5466 = t5465 * 64;
      int t5467 = t5466 + t5443;
      memory[2158068 + t5467] = t5444;
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
    for (uint t5469 = 0; t5469 < 3904; t5469++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=255, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5470 = 0.0;
      int t5471 = t5469 / 64;
      int t5472 = t5471 * 64;
      int t5473 = t5469 - t5472;
      int t5474 = t5473;
      int t5475 = t5474;
      int t5476 = t5473 - t5475;
      int t5477 = t5471 * 4096;
      int t5478 = t5477;
      int t5479 = t5474;
      int t5480 = t5478 + t5479;
      for (uint t5481 = 0; t5481 < 64; t5481++) {
        int t5482 = t5481 * 64;
        int t5483 = t5480 + t5482;
        int t5484 = t5481 * 64;
        int t5485 = t5484 + t5474;
        int t5486 = t5485 / 64;
        int t5487 = t5486 * 64;
        int t5488 = t5485 - t5487;
        int t5489 = t5488 * 64;
        int t5490 = t5486 + t5489;
        float t5491 = memory[256 + t5490];
        int t5492 = t5471 * 64;
        int t5493 = t5492 + t5481;
        int t5494 = id;
        int t5495 = t5494 * 3904;
        int t5496 = t5495 + t5493;
        float t5497 = memory[468876660 + t5496];
        float t5498 = t5491 * t5497;
        float t5499 = t5470 + t5498;
        t5470 = t5499;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5501 = id;
      int t5502 = t5501 * 3904;
      int t5503 = t5502 + t5469;
      memory[209874484 + t5503] = t5470;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5505 = 0; t5505 < 4096; t5505++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=257, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5506 = 0.0;
      int t5507 = t5505 / 64;
      int t5508 = t5507 * 64;
      int t5509 = t5505 - t5508;
      int t5510 = t5509;
      int t5511 = t5510;
      int t5512 = t5509 - t5511;
      int t5513 = t5507 * 64;
      int t5514 = t5513;
      int t5515 = t5510;
      int t5516 = t5514 + t5515;
      for (uint t5517 = 0; t5517 < 61; t5517++) {
        int t5518 = t5517 * 4096;
        int t5519 = t5516 + t5518;
        int t5520 = t5517 * 64;
        int t5521 = t5520 + t5510;
        float t5522 = memory[29364 + t5521];
        int t5523 = t5517 * 64;
        int t5524 = t5523 + t5507;
        int t5525 = id;
        int t5526 = t5525 * 3904;
        int t5527 = t5526 + t5524;
        float t5528 = memory[468876660 + t5527];
        float t5529 = t5522 * t5528;
        float t5530 = t5506 + t5529;
        t5506 = t5530;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5532 = id;
      int t5533 = t5532 * 4096;
      int t5534 = t5533 + t5505;
      memory[532839796 + t5534] = t5506;
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
    for (uint t5536 = 0; t5536 < 3904; t5536++) {
      float t5537 = memory[33268 + (int)t5536];
      float t5538 = metal::tanh(t5537);
      float t5539 = t5538 * t5538;
      float t5540 = 1.0 - t5539;
      int t5541 = id;
      int t5542 = t5541 * 3904;
      int t5543 = t5542 + t5536;
      float t5544 = memory[209874484 + t5543];
      float t5545 = t5540 * t5544;
      int t5546 = id;
      int t5547 = t5546 * 3904;
      int t5548 = t5547 + t5536;
      memory[273837620 + t5548] = t5545;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5550 = 0; t5550 < 64; t5550++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=267, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5551 = 0.0;
      int t5552 = t5550;
      int t5553 = t5552;
      int t5554 = t5550 - t5553;
      int t5555 = t5552;
      int t5556 = t5555;
      for (uint t5557 = 0; t5557 < 61; t5557++) {
        int t5558 = t5557 * 64;
        int t5559 = t5556 + t5558;
        int t5560 = t5557 * 64;
        int t5561 = t5560 + t5552;
        float t5562 = memory[25460 + t5561];
        int t5563 = t5557 * 64;
        int t5564 = t5563 + t5552;
        int t5565 = id;
        int t5566 = t5565 * 3904;
        int t5567 = t5566 + t5564;
        float t5568 = memory[209874484 + t5567];
        float t5569 = t5562 * t5568;
        float t5570 = t5551 + t5569;
        t5551 = t5570;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5572 = id;
      int t5573 = t5572 * 64;
      int t5574 = t5573 + t5550;
      memory[3206644 + t5574] = t5551;
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
    for (uint t5576 = 0; t5576 < 183; t5576++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=272, axis=1, in=[61, 64, 3], out=[61, 3], inFA=true, outFA=true), value: empty) */
      float t5577 = 0.0;
      int t5578 = t5576 / 3;
      int t5579 = t5578 * 3;
      int t5580 = t5576 - t5579;
      int t5581 = t5580;
      int t5582 = t5581;
      int t5583 = t5580 - t5582;
      int t5584 = t5578 * 192;
      int t5585 = t5584;
      int t5586 = t5581;
      int t5587 = t5585 + t5586;
      for (uint t5588 = 0; t5588 < 64; t5588++) {
        int t5589 = t5588 * 3;
        int t5590 = t5587 + t5589;
        int t5591 = t5588 * 3;
        int t5592 = t5591 + t5581;
        int t5593 = t5592 / 3;
        int t5594 = t5593 * 3;
        int t5595 = t5592 - t5594;
        int t5596 = t5595 * 64;
        int t5597 = t5593 + t5596;
        float t5598 = memory[0 + t5597];
        int t5599 = t5578 * 64;
        int t5600 = t5599 + t5588;
        int t5601 = id;
        int t5602 = t5601 * 3904;
        int t5603 = t5602 + t5600;
        float t5604 = memory[273837620 + t5603];
        float t5605 = t5598 * t5604;
        float t5606 = t5577 + t5605;
        t5577 = t5606;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5608 = id;
      int t5609 = t5608 * 183;
      int t5610 = t5609 + t5576;
      memory[42020340 + t5610] = t5577;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 3]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5612 = 0; t5612 < 192; t5612++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=274, axis=0, in=[61, 64, 3], out=[64, 3], inFA=true, outFA=true), value: empty) */
      float t5613 = 0.0;
      int t5614 = t5612 / 3;
      int t5615 = t5614 * 3;
      int t5616 = t5612 - t5615;
      int t5617 = t5616;
      int t5618 = t5617;
      int t5619 = t5616 - t5618;
      int t5620 = t5614 * 3;
      int t5621 = t5620;
      int t5622 = t5617;
      int t5623 = t5621 + t5622;
      for (uint t5624 = 0; t5624 < 61; t5624++) {
        int t5625 = t5624 * 192;
        int t5626 = t5623 + t5625;
        int t5627 = t5624 * 3;
        int t5628 = t5627 + t5617;
        float t5629 = memory[8706 + t5628];
        int t5630 = t5624 * 64;
        int t5631 = t5630 + t5614;
        int t5632 = id;
        int t5633 = t5632 * 3904;
        int t5634 = t5633 + t5631;
        float t5635 = memory[273837620 + t5634];
        float t5636 = t5629 * t5635;
        float t5637 = t5613 + t5636;
        t5613 = t5637;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5639 = id;
      int t5640 = t5639 * 192;
      int t5641 = t5640 + t5612;
      memory[46231028 + t5641] = t5613;
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
  if (id < 192) { uint _pr5643 = id;
    float t5644 = 0.0;
    for (uint t5645 = 0; t5645 < 16384; t5645++) {
      int t5646 = t5645 * 192;
      int t5647 = t5646 + _pr5643;
      float t5648 = memory[46231028 + t5647];
      float t5649 = t5644 + t5648;
      t5644 = t5649;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5643] = t5644;
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
  if (id < 64) { uint _pr5653 = id;
    float t5654 = 0.0;
    for (uint t5655 = 0; t5655 < 16384; t5655++) {
      int t5656 = t5655 * 64;
      int t5657 = t5656 + _pr5653;
      float t5658 = memory[3206644 + t5657];
      float t5659 = t5654 + t5658;
      t5654 = t5659;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5653] = t5654;
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
  if (id < 4096) { uint _pr5663 = id;
    float t5664 = 0.0;
    for (uint t5665 = 0; t5665 < 16384; t5665++) {
      int t5666 = t5665 * 4096;
      int t5667 = t5666 + _pr5663;
      float t5668 = memory[532839796 + t5667];
      float t5669 = t5664 + t5668;
      t5664 = t5669;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[3206644 + (int)_pr5663] = t5664;
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
  if (id < 64) { uint _pr5673 = id;
    float t5674 = 0.0;
    for (uint t5675 = 0; t5675 < 16384; t5675++) {
      int t5676 = t5675 * 64;
      int t5677 = t5676 + _pr5673;
      float t5678 = memory[2158068 + t5677];
      float t5679 = t5674 + t5678;
      t5674 = t5679;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5673] = t5674;
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
  if (id < 4096) { uint _pr5683 = id;
    float t5684 = 0.0;
    for (uint t5685 = 0; t5685 < 16384; t5685++) {
      int t5686 = t5685 * 4096;
      int t5687 = t5686 + _pr5683;
      float t5688 = memory[337804660 + t5687];
      float t5689 = t5684 + t5688;
      t5684 = t5689;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[2158068 + (int)_pr5683] = t5684;
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
  if (id < 64) { uint _pr5693 = id;
    float t5694 = 0.0;
    for (uint t5695 = 0; t5695 < 16384; t5695++) {
      int t5696 = t5695 * 64;
      int t5697 = t5696 + _pr5693;
      float t5698 = memory[1109492 + t5697];
      float t5699 = t5694 + t5698;
      t5694 = t5699;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5693] = t5694;
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
  if (id < 64) { uint _pr5703 = id;
    float t5704 = 0.0;
    for (uint t5705 = 0; t5705 < 16384; t5705++) {
      int t5706 = t5705 * 64;
      int t5707 = t5706 + _pr5703;
      float t5708 = memory[37809652 + t5707];
      float t5709 = t5704 + t5708;
      t5704 = t5709;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5703] = t5704;
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
  if (id < 1) { uint _pr5713 = id;
    float t5714 = 0.0;
    for (uint t5715 = 0; t5715 < 16384; t5715++) {
      int t5716 = t5715;
      int t5717 = t5716 + _pr5713;
      float t5718 = memory[60916 + t5717];
      float t5719 = t5714 + t5718;
      t5714 = t5719;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[60596 + (int)_pr5713] = t5714;
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
    /* loadGlobal(5096) - handled in variable access */
    /* loadGlobal(5095) - handled in variable access */
    /* loadGlobal(2749) - handled in variable access */
    outputs[0 * frameCount + id] = t[16*frameCount + id];
  }
  #pragma clang diagnostic pop
}

