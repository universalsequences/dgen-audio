// KERNEL 0
// FrameOrder: parallel
// DispatchMode: perFrameScaled(32768)
kernel void kernel_0(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4897 = frameCount * 32768.0;
  if (id >= 0 && id < (uint)(t4897)) {
    int t110 = id;
    int t111 = t110 / 32768;
    uint _frameIndex = (uint)(t111);
    int t112 = t111 * 32768;
    int t113 = t110 - t112;
    int t114 = t113 / 128;
    int t115 = t113 % 128;
    float t116 = 0.0;
    for (uint t117 = 0; t117 < 3; t117++) {
      int t118 = t114 * 3;
      int t119 = t118 + t117;
      int t120 = t117 * 128;
      int t121 = t120 + t115;
      float t122 = memory[25538 + t119];
      float t123 = memory[0 + t121];
      float t124 = t122 * t123;
      float t125 = t116 + t124;
      t116 = t125;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t127 = t114 * 128;
    int t128 = t127 + t115;
    memory[98752 + t128] = t116;
  }
  #pragma clang diagnostic pop
}



// KERNEL 1
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_1(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t130 = id;
    int t131 = t130 / 32768;
    uint _frameIndex = (uint)(t131);
    int t132 = t131 * 32768;
    int t133 = t130 - t132;
    float t134 = memory[98752 + t133];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t135 = t133 / 128;
    int t136 = t135 * 128;
    int t137 = t133 - t136;
    int t138 = t137;
    float t139 = memory[384 + t138];
    float t140 = t134 + t139;
    memory[164288 + t133] = t140;
    float t142 = metal::tanh(t140);
    memory[131520 + t133] = t142;
  }
  #pragma clang diagnostic pop
}



// KERNEL 2
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 32, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_2(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t144 = gid.y;
    int t145 = gid.x;
    int t146 = gid.z;
    metal::simdgroup_float8x8 t147 = metal::simdgroup_float8x8(0);
    for (uint t148 = 0; t148 < 16; t148++) {
      int t149 = t144 * 1024;
      int t150 = t149;
      int t151 = t148 * 8;
      int t152 = t150 + t151;
      metal::simdgroup_float8x8 t153 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t153, &memory[131520 + t152], 128);
      int t154 = t148 * 1024;
      int t155 = t154;
      int t156 = t145 * 8;
      int t157 = t155 + t156;
      metal::simdgroup_float8x8 t158 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t158, &memory[512 + t157], 128);
      metal::simdgroup_multiply_accumulate(t147, t153, t158, t147);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t160 = t144 * 1024;
    int t161 = t160;
    int t162 = t145 * 8;
    int t163 = t161 + t162;
    metal::simdgroup_store(t147, &memory[98752 + t163], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 3
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_3(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t165 = id;
    int t166 = t165 / 32768;
    uint _frameIndex = (uint)(t166);
    int t167 = t166 * 32768;
    int t168 = t165 - t167;
    float t169 = memory[98752 + t168];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t170 = t168 / 128;
    int t171 = t170 * 128;
    int t172 = t168 - t171;
    int t173 = t172;
    float t174 = memory[16896 + t173];
    float t175 = t169 + t174;
    memory[197056 + t168] = t175;
    float t177 = metal::tanh(t175);
    memory[229824 + t168] = t177;
  }
  #pragma clang diagnostic pop
}



// KERNEL 4
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 32, tilesN: 8, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_4(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t179 = gid.y;
    int t180 = gid.x;
    int t181 = gid.z;
    metal::simdgroup_float8x8 t182 = metal::simdgroup_float8x8(0);
    for (uint t183 = 0; t183 < 16; t183++) {
      int t184 = t179 * 1024;
      int t185 = t184;
      int t186 = t183 * 8;
      int t187 = t185 + t186;
      metal::simdgroup_float8x8 t188 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t188, &memory[229824 + t187], 128);
      int t189 = t183 * 512;
      int t190 = t189;
      int t191 = t180 * 8;
      int t192 = t190 + t191;
      metal::simdgroup_float8x8 t193 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t193, &memory[17024 + t192], 64);
      metal::simdgroup_multiply_accumulate(t182, t188, t193, t182);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t195 = t179 * 512;
    int t196 = t195;
    int t197 = t180 * 8;
    int t198 = t196 + t197;
    metal::simdgroup_store(t182, &memory[98752 + t198], 64);
  }
  #pragma clang diagnostic pop
}



// KERNEL 5
// FrameOrder: parallel
// DispatchMode: staticThreads(16384)
kernel void kernel_5(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(16384)) {
    int t200 = id;
    int t201 = t200 / 16384;
    uint _frameIndex = (uint)(t201);
    int t202 = t201 * 16384;
    int t203 = t200 - t202;
    float t204 = memory[98752 + t203];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t205 = t203 / 64;
    int t206 = t205 * 64;
    int t207 = t203 - t206;
    int t208 = t207;
    float t209 = memory[25216 + t208];
    float t210 = t204 + t209;
    memory[311744 + t203] = t210;
    float t212 = t210 * -1.0;
    memory[278976 + t203] = t212;
    float t214 = metal::exp(t212);
    float t215 = 1.0 + t214;
    memory[262592 + t203] = t215;
    float t217 = 1.0 / t215;
    memory[295360 + t203] = t217;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 128]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 6
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_6(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4898 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4898)) {
    int t219 = id;
    int t220 = t219 / 256;
    uint _frameIndex = (uint)(t220);
    int t221 = t220 * 256;
    int t222 = t219 - t221;
    int t223 = t222;
    int t224 = t222 % 1;
    float t225 = 0.0;
    for (uint t226 = 0; t226 < 128; t226++) {
      int t227 = t223 * 128;
      int t228 = t227 + t226;
      int t229 = t226;
      int t230 = t229 + t224;
      float t231 = memory[229824 + t228];
      float t232 = memory[25280 + t230];
      float t233 = t231 * t232;
      float t234 = t225 + t233;
      t225 = t234;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t236 = t223;
    int t237 = t236 + t224;
    memory[98752 + t237] = t225;
  }
  #pragma clang diagnostic pop
}



// KERNEL 7
// FrameOrder: parallel
// DispatchMode: staticThreads(256)
kernel void kernel_7(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(256)) {
    int t239 = id;
    int t240 = t239 / 256;
    uint _frameIndex = (uint)(t240);
    int t241 = t240 * 256;
    int t242 = t239 - t241;
    float t243 = memory[98752 + t242];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t244 = t242;
    int t245 = t244;
    int t246 = t242 - t245;
    float t247 = memory[25408 + (int)0.0];
    float t248 = t243 + t247;
    memory[328640 + t242] = t248;
    float t250 = t248 * -1.0;
    memory[328384 + t242] = t250;
    float t252 = metal::exp(t250);
    float t253 = 1.0 + t252;
    memory[328896 + t242] = t253;
    float t255 = 1.0 / t253;
    memory[328128 + t242] = t255;
  }
  #pragma clang diagnostic pop
}



// KERNEL 8
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_8(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4899 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4899)) {
    int t257 = id;
    int t258 = t257 / 256;
    uint _frameIndex = (uint)(t258);
    int t259 = t258 * 256;
    int t260 = t257 - t259;
    int t261 = t260;
    int t262 = t260 % 1;
    float t263 = 0.0;
    for (uint t264 = 0; t264 < 128; t264++) {
      int t265 = t261 * 128;
      int t266 = t265 + t264;
      int t267 = t264;
      int t268 = t267 + t262;
      float t269 = memory[229824 + t266];
      float t270 = memory[25409 + t268];
      float t271 = t269 * t270;
      float t272 = t263 + t271;
      t263 = t272;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t274 = t261;
    int t275 = t274 + t262;
    memory[98752 + t275] = t263;
  }
  #pragma clang diagnostic pop
}



// KERNEL 9
// FrameOrder: parallel
// DispatchMode: staticThreads(256)
kernel void kernel_9(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(256)) {
    int t277 = id;
    int t278 = t277 / 256;
    uint _frameIndex = (uint)(t278);
    int t279 = t278 * 256;
    int t280 = t277 - t279;
    float t281 = memory[98752 + t280];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t282 = t280;
    int t283 = t282;
    int t284 = t280 - t283;
    float t285 = memory[25537 + (int)0.0];
    float t286 = t281 + t285;
    float t287 = t286 * -1.0;
    float t288 = metal::exp(t287);
    float t289 = 1.0 + t288;
    float t290 = 1.0 / t289;
    memory[329152 + t280] = t290;
  }
  #pragma clang diagnostic pop
}



// KERNEL 10
// FrameOrder: sequential
// DispatchMode: singleThreaded
kernel void kernel_10(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(292), value: global(292)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[20664577];
      float t293 = t[0*frameCount + i] + 0.0038454495;
      float t294 = metal::select(t293, 0.0, 0.0 > 0.0);
      float t295 = t294;
      float t296 = (t295 * 0.015873017);
      float t297 = metal::floor(t296);
      float t298 = t297 * 63.0;
      float t299 = t294 - t298;
      memory[20664577] = t299;
      float t301 = t299 >= 63.0;
      if (t301) {
        float t303 = t299 - 63.0;
        memory[20664577] = t303;
      }
      if (0.0) {
        memory[20664577] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 11
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_11(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(310), value: global(310)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(292) - handled in variable access */
    float t309 = metal::min(t[0*frameCount + id], 62.9999);
    t[1*frameCount + id] = metal::max(t309, 0.0);
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([4, 64, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0, 2]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([4, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 12
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_12(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4900 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4900)) {
    /* loadGlobal(310) - handled in variable access */
    int t311 = id;
    int t312 = t311 / 256;
    uint _frameIndex = (uint)(t312);
    int t313 = t312 * 256;
    int t314 = t311 - t313;
    float t315 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t316 = t315 < 0.0;
    float t317 = t315 + 64.0;
    float t318 = metal::select(t315, t317, t316 > 0.0);
    float t319 = metal::floor(t318);
    float t320 = t319 + 1.0;
    float t321 = t320 >= 64.0;
    float t322 = metal::select(t320, 0.0, t321 > 0.0);
    float t323 = t318 - t319;
    float t324 = 1.0 - t323;
    float t325 = t312 * 256.0;
    int t326 = t314 / 64;
    int t327 = t326 * 64;
    int t328 = t314 - t327;
    float t329 = (float)t326;
    float t330 = (float)t328;
    float t331 = t319 * 64.0;
    float t332 = t329 * 4096.0;
    float t333 = t331 + t332;
    float t334 = t333 + t330;
    int t335 = (int)t334;
    float t336 = memory[295360 + t335];
    float t337 = t322 * 64.0;
    float t338 = t329 * 4096.0;
    float t339 = t337 + t338;
    float t340 = t339 + t330;
    int t341 = (int)t340;
    float t342 = memory[295360 + t341];
    float t343 = t324 * t336;
    float t344 = t323 * t342;
    float t345 = t343 + t344;
    float t346 = (float)t314;
    float t347 = t325 + t346;
    int t348 = (int)t347;
    memory[329408 + t348] = t345;
    int t350 = (int)t347;
    memory[4523712 + t350] = t345;
  }
  #pragma clang diagnostic pop
}



// KERNEL 13
// FrameOrder: parallel
// DispatchMode: perFrameScaled(4)
kernel void kernel_13(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4901 = frameCount * 4.0;
  if (id >= 0 && id < (uint)(t4901)) {
    /* loadGlobal(310) - handled in variable access */
    int t352 = id;
    int t353 = t352 / 4;
    uint _frameIndex = (uint)(t353);
    int t354 = t353 * 4;
    int t355 = t352 - t354;
    float t356 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t357 = t356 < 0.0;
    float t358 = t356 + 64.0;
    float t359 = metal::select(t356, t358, t357 > 0.0);
    float t360 = metal::floor(t359);
    float t361 = t360 + 1.0;
    float t362 = t361 >= 64.0;
    float t363 = metal::select(t361, 0.0, t362 > 0.0);
    float t364 = t359 - t360;
    float t365 = 1.0 - t364;
    float t366 = t353 * 4.0;
    float t367 = (float)t355;
    float t368 = t367 * 64.0;
    float t369 = t360 + t368;
    int t370 = (int)t369;
    float t371 = memory[328128 + t370];
    float t372 = t367 * 64.0;
    float t373 = t363 + t372;
    int t374 = (int)t373;
    float t375 = memory[328128 + t374];
    float t376 = t365 * t371;
    float t377 = t364 * t375;
    float t378 = t376 + t377;
    float t379 = (float)t355;
    float t380 = t366 + t379;
    int t381 = (int)t380;
    memory[329408 + t381] = t378;
    int t383 = (int)t380;
    memory[8914624 + t383] = t378;
    float t385 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t386 = t385 < 0.0;
    float t387 = t385 + 64.0;
    float t388 = metal::select(t385, t387, t386 > 0.0);
    float t389 = metal::floor(t388);
    float t390 = t389 + 1.0;
    float t391 = t390 >= 64.0;
    float t392 = metal::select(t390, 0.0, t391 > 0.0);
    float t393 = t388 - t389;
    float t394 = 1.0 - t393;
    float t395 = t353 * 4.0;
    float t396 = (float)t355;
    float t397 = t389 * 4.0;
    float t398 = t397 + t396;
    int t399 = (int)t398;
    float t400 = memory[26306 + t399];
    float t401 = t392 * 4.0;
    float t402 = t401 + t396;
    int t403 = (int)t402;
    float t404 = memory[26306 + t403];
    float t405 = t394 * t400;
    float t406 = t393 * t404;
    float t407 = t405 + t406;
    float t408 = (float)t355;
    float t409 = t395 + t408;
    int t410 = (int)t409;
    memory[8718016 + t410] = t407;
    int t412 = (int)t409;
    memory[8849088 + t412] = t407;
    float t414 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t415 = t414 < 0.0;
    float t416 = t414 + 64.0;
    float t417 = metal::select(t414, t416, t415 > 0.0);
    float t418 = metal::floor(t417);
    float t419 = t418 + 1.0;
    float t420 = t419 >= 64.0;
    float t421 = metal::select(t419, 0.0, t420 > 0.0);
    float t422 = t417 - t418;
    float t423 = 1.0 - t422;
    float t424 = t353 * 4.0;
    float t425 = (float)t355;
    float t426 = t418 * 4.0;
    float t427 = t426 + t425;
    int t428 = (int)t427;
    float t429 = memory[26562 + t428];
    float t430 = t421 * 4.0;
    float t431 = t430 + t425;
    int t432 = (int)t431;
    float t433 = memory[26562 + t432];
    float t434 = t423 * t429;
    float t435 = t422 * t433;
    float t436 = t434 + t435;
    float t437 = (float)t355;
    float t438 = t424 + t437;
    int t439 = (int)t438;
    memory[8783552 + t439] = t436;
    int t441 = (int)t438;
    memory[8980160 + t441] = t436;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([4, 1]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mexpandView[0m([4, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 14
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_14(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4902 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4902)) {
    int t443 = id;
    int t444 = t443 / 256;
    uint _frameIndex = (uint)(t444);
    int t445 = t444 * 256;
    int t446 = t443 - t445;
    int t447 = t446 / 64;
    int t448 = t447 * 64;
    int t449 = t446 - t448;
    int t450 = _frameIndex;
    int t451 = t450 * 4;
    int t452 = t451 + t447;
    float t453 = memory[8849088 + t452];
    float t454 = memory[26818 + t446];
    float t455 = t453 * t454;
    int t456 = _frameIndex;
    int t457 = t456 * 256;
    int t458 = t457 + t446;
    memory[329408 + t458] = t455;
  }
  #pragma clang diagnostic pop
}



// KERNEL 15
// FrameOrder: sequential
// DispatchMode: fixedWithFrameLoop(256)
kernel void kernel_15(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(256)) {
    for (uint i = 0; i < frameCount; i += 1) {
      int t460 = id;
      int t461 = i;
      int t462 = t461 * 256;
      int t463 = t462 + t460;
      float t464 = memory[329408 + t463];
      float t465 = (t464 * 6.25e-05);
      float t466 = memory[328128 + t460];
      float t467 = t466 + t465;
      float t468 = metal::select(t467, 0.0, 0.0 > 0.0);
      float t469 = metal::floor(t468);
      float t470 = t468 - t469;
      float t471 = t470 >= 1.0;
      float t472 = t470 - 1.0;
      float t473 = metal::select(t470, t472, t471 > 0.0);
      float t474 = metal::select(t473, 0.0, 0.0 > 0.0);
      memory[328128 + t460] = t474;
      int t476 = i;
      int t477 = t476 * 256;
      int t478 = t477 + t460;
      memory[9045696 + t478] = t466;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 16
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_16(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(310) - handled in variable access */
    for (uint t480 = 0; t480 < 256; t480++) {
      int t481 = id;
      int t482 = t481 * 256;
      int t483 = t482 + t480;
      float t484 = memory[9045696 + t483];
      float t485 = t484 * 6.283185;
      float t486 = metal::sin(t485);
      int t487 = id;
      int t488 = t487 * 256;
      int t489 = t488 + t480;
      memory[13305536 + t489] = t486;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t491 = 0; t491 < 4; t491++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=91, axis=1, in=[4, 64], out=[4], inFA=true, outFA=true), value: empty) */
      float t492 = 0.0;
      int t493 = t491;
      int t494 = t493;
      int t495 = t491 - t494;
      int t496 = t493 * 64;
      int t497 = t496;
      for (uint t498 = 0; t498 < 64; t498++) {
        int t499 = t498;
        int t500 = t497 + t499;
        int t501 = t493 * 64;
        int t502 = t501 + t498;
        int t503 = id;
        int t504 = t503 * 256;
        int t505 = t504 + t502;
        float t506 = memory[13305536 + t505];
        int t507 = t493 * 64;
        int t508 = t507 + t498;
        int t509 = id;
        int t510 = t509 * 256;
        int t511 = t510 + t508;
        float t512 = memory[4523712 + t511];
        float t513 = t506 * t512;
        float t514 = t492 + t513;
        t492 = t514;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t516 = id;
      int t517 = t516 * 4;
      int t518 = t517 + t491;
      memory[329408 + t518] = t492;
      int t520 = id;
      int t521 = t520 * 4;
      int t522 = t521 + t491;
      float t523 = memory[329408 + t522];
      int t524 = id;
      int t525 = t524 * 4;
      int t526 = t525 + t491;
      float t527 = memory[8980160 + t526];
      float t528 = t523 * t527;
      int t529 = id;
      int t530 = t529 * 4;
      int t531 = t530 + t491;
      memory[8849088 + t531] = t528;
      int t533 = id;
      int t534 = t533 * 4;
      int t535 = t534 + t491;
      float t536 = memory[8914624 + t535];
      float t537 = t528 * t536;
      int t538 = id;
      int t539 = t538 * 4;
      int t540 = t539 + t491;
      memory[13240000 + t540] = t537;
      float t542 = t537 * 0.015625;
      int t543 = id;
      int t544 = t543 * 4;
      int t545 = t544 + t491;
      memory[8783552 + t545] = t542;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([4, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      float t547 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 64.0) * 64.0);
      float t548 = t547 < 0.0;
      float t549 = t547 + 64.0;
      float t550 = metal::select(t547, t549, t548 > 0.0);
      float t551 = metal::floor(t550);
      float t552 = t551 + 1.0;
      float t553 = t552 >= 64.0;
      float t554 = metal::select(t552, 0.0, t553 > 0.0);
      float t555 = t550 - t551;
      float t556 = 1.0 - t555;
      int t557 = id;
      float t558 = t557 * 4.0;
      float t559 = (float)t491;
      float t560 = t559 * 64.0;
      float t561 = t551 + t560;
      int t562 = (int)t561;
      float t563 = memory[329152 + t562];
      float t564 = t559 * 64.0;
      float t565 = t554 + t564;
      int t566 = (int)t565;
      float t567 = memory[329152 + t566];
      float t568 = t556 * t563;
      float t569 = t555 * t567;
      float t570 = t568 + t569;
      float t571 = (float)t491;
      float t572 = t558 + t571;
      int t573 = (int)t572;
      memory[8718016 + t573] = t570;
      memory[328128 + (int)t491] = t570;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 17
// FrameOrder: sequential
// DispatchMode: singleThreaded
kernel void kernel_17(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(576), value: global(576)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[2*frameCount + i] = memory[20664578];
      float t577 = t[2*frameCount + i] + 1.0;
      float t578 = metal::select(t577, 0.0, 0.0 > 0.0);
      float t579 = t578;
      float t580 = (t579 * 6.1035156e-05);
      float t581 = metal::floor(t580);
      float t582 = t581 * 16384.0;
      float t583 = t578 - t582;
      memory[20664578] = t583;
      float t585 = t583 >= 16384.0;
      if (t585) {
        float t587 = t583 - 16384.0;
        memory[20664578] = t587;
      }
      if (0.0) {
        memory[20664578] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 18
// FrameOrder: parallel
// DispatchMode: perFrameScaled(4)
kernel void kernel_18(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4903 = frameCount * 4.0;
  if (id >= 0 && id < (uint)(t4903)) {
    /* loadGlobal(576) - handled in variable access */
    int t593 = id;
    int t594 = t593 / 4;
    uint _frameIndex = (uint)(t594);
    int t595 = t594 * 4;
    int t596 = t593 - t595;
    float t597 = (t[2*frameCount + _frameIndex] - metal::floor(t[2*frameCount + _frameIndex] / 16384.0) * 16384.0);
    float t598 = t597 < 0.0;
    float t599 = t597 + 16384.0;
    float t600 = metal::select(t597, t599, t598 > 0.0);
    float t601 = metal::floor(t600);
    float t602 = t601 + 1.0;
    float t603 = t602 >= 16384.0;
    float t604 = metal::select(t602, 0.0, t603 > 0.0);
    float t605 = t600 - t601;
    float t606 = 1.0 - t605;
    float t607 = t594 * 4.0;
    float t608 = (float)t596;
    float t609 = t601 * 4.0;
    float t610 = t609 + t608;
    int t611 = (int)t610;
    float t612 = memory[27074 + t611];
    float t613 = t604 * 4.0;
    float t614 = t613 + t608;
    int t615 = (int)t614;
    float t616 = memory[27074 + t615];
    float t617 = t606 * t612;
    float t618 = t605 * t616;
    float t619 = t617 + t618;
    float t620 = (float)t596;
    float t621 = t607 + t620;
    int t622 = (int)t621;
    memory[8718016 + t622] = t619;
    int t624 = (int)t621;
    memory[9045696 + t624] = t619;
  }
  #pragma clang diagnostic pop
}



// KERNEL 19
// FrameOrder: sequential
// DispatchMode: singleThreaded
kernel void kernel_19(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(638), value: global(638)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[3*frameCount + i] = memory[20664579];
      float t639 = t[3*frameCount + i] + 1.0;
      float t640 = metal::select(t639, 0.0, 0.0 > 0.0);
      float t641 = t640;
      float t642 = (t641 * 0.0078125);
      float t643 = metal::floor(t642);
      float t644 = t643 * 128.0;
      float t645 = t640 - t644;
      memory[20664579] = t645;
      float t647 = t645 >= 128.0;
      if (t647) {
        float t649 = t645 - 128.0;
        memory[20664579] = t649;
      }
      if (0.0) {
        memory[20664579] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 20
// FrameOrder: parallel
// DispatchMode: perFrameThreadgroup1
kernel void kernel_20(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1437), value: global(1437)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(638) - handled in variable access */
    int t655 = id;
    threadgroup float scratch_0[512];
    threadgroup float scratch_1[512];
    threadgroup float scratch_2[511];
    threadgroup float scratch_3[511];
    float t656 = t[3*frameCount + id] == 0.0;
    float t657 = 0.0;
    if (t656) {
      for (uint t659 = 0; t659 < 511; t659++) {
        float t660 = memory[93122 + (int)t659];
        scratch_2[(int)t659] = t660;
        float t662 = memory[93633 + (int)t659];
        scratch_3[(int)t659] = t662;
      }
      for (uint t665 = 0; t665 < 4; t665++) {
        int t666 = t655 / 128;
        int t667 = t666 * 4;
        int t668 = t667 + t665;
        int t669 = t668 * 1024;
        int t670 = t668 * 257;
        for (uint t671 = 0; t671 < 512; t671++) {
          float t672 = (float)t671;
          float t673 = memory[92610 + (int)t671];
          float t674 = (float)t655;
          float t675 = t674 - 511.0;
          float t676 = t675 + t672;
          float t677 = t676 >= 0.0;
          float t678 = t676 < frameCount;
          float t679 = t677 * t678;
          float t680 = frameCount - 1.0;
          float t681 = metal::min(t676, t680);
          float t682 = metal::max(0.0, t681);
          int t683 = (int)t682;
          int t684 = t683 * 4;
          int t685 = t684 + t665;
          float t686 = memory[8783552 + t685];
          float t687 = metal::select(0.0, t686, t679 > 0.0);
          float t688 = t687 * t673;
          scratch_0[(int)t671] = t688;
          scratch_1[(int)t671] = 0.0;
        }
        for (uint t692 = 0; t692 < 512; t692++) {
          float t693 = memory[94144 + (int)t692];
          float t694 = (float)t692;
          float t695 = t694 < t693;
          int t696 = (int)t693;
          float t697 = scratch_0[(int)t692];
          float t698 = scratch_1[(int)t692];
          float t699 = scratch_0[t696];
          float t700 = scratch_1[t696];
          float t701 = metal::select(t697, t699, t695 > 0.0);
          float t702 = metal::select(t698, t700, t695 > 0.0);
          float t703 = metal::select(t699, t697, t695 > 0.0);
          float t704 = metal::select(t700, t698, t695 > 0.0);
          scratch_0[(int)t692] = t701;
          scratch_1[(int)t692] = t702;
          scratch_0[t696] = t703;
          scratch_1[t696] = t704;
        }
        for (uint t710 = 0; t710 < 256; t710++) {
          float t711 = (float)t710;
          float t712 = t711;
          float t713 = metal::floor(t712);
          float t714 = t713;
          float t715 = t711 - t714;
          float t716 = t713 * 2.0;
          float t717 = t716 + t715;
          float t718 = t717 + 1.0;
          int t719 = (int)t715;
          int t720 = t719;
          float t721 = scratch_2[t720];
          float t722 = scratch_3[t720];
          float t723 = 0.0 - t722;
          int t724 = (int)t717;
          int t725 = (int)t718;
          float t726 = scratch_0[t724];
          float t727 = scratch_1[t724];
          float t728 = scratch_0[t725];
          float t729 = scratch_1[t725];
          float t730 = t721 * t728;
          float t731 = t723 * t729;
          float t732 = t730 - t731;
          float t733 = t721 * t729;
          float t734 = t723 * t728;
          float t735 = t733 + t734;
          float t736 = t726 + t732;
          scratch_0[t724] = t736;
          float t738 = t727 + t735;
          scratch_1[t724] = t738;
          float t740 = t726 - t732;
          scratch_0[t725] = t740;
          float t742 = t727 - t735;
          scratch_1[t725] = t742;
        }
        for (uint t745 = 0; t745 < 256; t745++) {
          float t746 = (float)t745;
          float t747 = (t746 * 0.5);
          float t748 = metal::floor(t747);
          float t749 = t748 * 2.0;
          float t750 = t746 - t749;
          float t751 = t748 * 4.0;
          float t752 = t751 + t750;
          float t753 = t752 + 2.0;
          int t754 = (int)t750;
          int t755 = 1 + t754;
          float t756 = scratch_2[t755];
          float t757 = scratch_3[t755];
          float t758 = 0.0 - t757;
          int t759 = (int)t752;
          int t760 = (int)t753;
          float t761 = scratch_0[t759];
          float t762 = scratch_1[t759];
          float t763 = scratch_0[t760];
          float t764 = scratch_1[t760];
          float t765 = t756 * t763;
          float t766 = t758 * t764;
          float t767 = t765 - t766;
          float t768 = t756 * t764;
          float t769 = t758 * t763;
          float t770 = t768 + t769;
          float t771 = t761 + t767;
          scratch_0[t759] = t771;
          float t773 = t762 + t770;
          scratch_1[t759] = t773;
          float t775 = t761 - t767;
          scratch_0[t760] = t775;
          float t777 = t762 - t770;
          scratch_1[t760] = t777;
        }
        for (uint t780 = 0; t780 < 256; t780++) {
          float t781 = (float)t780;
          float t782 = (t781 * 0.25);
          float t783 = metal::floor(t782);
          float t784 = t783 * 4.0;
          float t785 = t781 - t784;
          float t786 = t783 * 8.0;
          float t787 = t786 + t785;
          float t788 = t787 + 4.0;
          int t789 = (int)t785;
          int t790 = 3 + t789;
          float t791 = scratch_2[t790];
          float t792 = scratch_3[t790];
          float t793 = 0.0 - t792;
          int t794 = (int)t787;
          int t795 = (int)t788;
          float t796 = scratch_0[t794];
          float t797 = scratch_1[t794];
          float t798 = scratch_0[t795];
          float t799 = scratch_1[t795];
          float t800 = t791 * t798;
          float t801 = t793 * t799;
          float t802 = t800 - t801;
          float t803 = t791 * t799;
          float t804 = t793 * t798;
          float t805 = t803 + t804;
          float t806 = t796 + t802;
          scratch_0[t794] = t806;
          float t808 = t797 + t805;
          scratch_1[t794] = t808;
          float t810 = t796 - t802;
          scratch_0[t795] = t810;
          float t812 = t797 - t805;
          scratch_1[t795] = t812;
        }
        for (uint t815 = 0; t815 < 256; t815++) {
          float t816 = (float)t815;
          float t817 = (t816 * 0.125);
          float t818 = metal::floor(t817);
          float t819 = t818 * 8.0;
          float t820 = t816 - t819;
          float t821 = t818 * 16.0;
          float t822 = t821 + t820;
          float t823 = t822 + 8.0;
          int t824 = (int)t820;
          int t825 = 7 + t824;
          float t826 = scratch_2[t825];
          float t827 = scratch_3[t825];
          float t828 = 0.0 - t827;
          int t829 = (int)t822;
          int t830 = (int)t823;
          float t831 = scratch_0[t829];
          float t832 = scratch_1[t829];
          float t833 = scratch_0[t830];
          float t834 = scratch_1[t830];
          float t835 = t826 * t833;
          float t836 = t828 * t834;
          float t837 = t835 - t836;
          float t838 = t826 * t834;
          float t839 = t828 * t833;
          float t840 = t838 + t839;
          float t841 = t831 + t837;
          scratch_0[t829] = t841;
          float t843 = t832 + t840;
          scratch_1[t829] = t843;
          float t845 = t831 - t837;
          scratch_0[t830] = t845;
          float t847 = t832 - t840;
          scratch_1[t830] = t847;
        }
        for (uint t850 = 0; t850 < 256; t850++) {
          float t851 = (float)t850;
          float t852 = (t851 * 0.0625);
          float t853 = metal::floor(t852);
          float t854 = t853 * 16.0;
          float t855 = t851 - t854;
          float t856 = t853 * 32.0;
          float t857 = t856 + t855;
          float t858 = t857 + 16.0;
          int t859 = (int)t855;
          int t860 = 15 + t859;
          float t861 = scratch_2[t860];
          float t862 = scratch_3[t860];
          float t863 = 0.0 - t862;
          int t864 = (int)t857;
          int t865 = (int)t858;
          float t866 = scratch_0[t864];
          float t867 = scratch_1[t864];
          float t868 = scratch_0[t865];
          float t869 = scratch_1[t865];
          float t870 = t861 * t868;
          float t871 = t863 * t869;
          float t872 = t870 - t871;
          float t873 = t861 * t869;
          float t874 = t863 * t868;
          float t875 = t873 + t874;
          float t876 = t866 + t872;
          scratch_0[t864] = t876;
          float t878 = t867 + t875;
          scratch_1[t864] = t878;
          float t880 = t866 - t872;
          scratch_0[t865] = t880;
          float t882 = t867 - t875;
          scratch_1[t865] = t882;
        }
        for (uint t885 = 0; t885 < 256; t885++) {
          float t886 = (float)t885;
          float t887 = (t886 * 0.03125);
          float t888 = metal::floor(t887);
          float t889 = t888 * 32.0;
          float t890 = t886 - t889;
          float t891 = t888 * 64.0;
          float t892 = t891 + t890;
          float t893 = t892 + 32.0;
          int t894 = (int)t890;
          int t895 = 31 + t894;
          float t896 = scratch_2[t895];
          float t897 = scratch_3[t895];
          float t898 = 0.0 - t897;
          int t899 = (int)t892;
          int t900 = (int)t893;
          float t901 = scratch_0[t899];
          float t902 = scratch_1[t899];
          float t903 = scratch_0[t900];
          float t904 = scratch_1[t900];
          float t905 = t896 * t903;
          float t906 = t898 * t904;
          float t907 = t905 - t906;
          float t908 = t896 * t904;
          float t909 = t898 * t903;
          float t910 = t908 + t909;
          float t911 = t901 + t907;
          scratch_0[t899] = t911;
          float t913 = t902 + t910;
          scratch_1[t899] = t913;
          float t915 = t901 - t907;
          scratch_0[t900] = t915;
          float t917 = t902 - t910;
          scratch_1[t900] = t917;
        }
        for (uint t920 = 0; t920 < 256; t920++) {
          float t921 = (float)t920;
          float t922 = (t921 * 0.015625);
          float t923 = metal::floor(t922);
          float t924 = t923 * 64.0;
          float t925 = t921 - t924;
          float t926 = t923 * 128.0;
          float t927 = t926 + t925;
          float t928 = t927 + 64.0;
          int t929 = (int)t925;
          int t930 = 63 + t929;
          float t931 = scratch_2[t930];
          float t932 = scratch_3[t930];
          float t933 = 0.0 - t932;
          int t934 = (int)t927;
          int t935 = (int)t928;
          float t936 = scratch_0[t934];
          float t937 = scratch_1[t934];
          float t938 = scratch_0[t935];
          float t939 = scratch_1[t935];
          float t940 = t931 * t938;
          float t941 = t933 * t939;
          float t942 = t940 - t941;
          float t943 = t931 * t939;
          float t944 = t933 * t938;
          float t945 = t943 + t944;
          float t946 = t936 + t942;
          scratch_0[t934] = t946;
          float t948 = t937 + t945;
          scratch_1[t934] = t948;
          float t950 = t936 - t942;
          scratch_0[t935] = t950;
          float t952 = t937 - t945;
          scratch_1[t935] = t952;
        }
        for (uint t955 = 0; t955 < 256; t955++) {
          float t956 = (float)t955;
          float t957 = (t956 * 0.0078125);
          float t958 = metal::floor(t957);
          float t959 = t958 * 128.0;
          float t960 = t956 - t959;
          float t961 = t958 * 256.0;
          float t962 = t961 + t960;
          float t963 = t962 + 128.0;
          int t964 = (int)t960;
          int t965 = 127 + t964;
          float t966 = scratch_2[t965];
          float t967 = scratch_3[t965];
          float t968 = 0.0 - t967;
          int t969 = (int)t962;
          int t970 = (int)t963;
          float t971 = scratch_0[t969];
          float t972 = scratch_1[t969];
          float t973 = scratch_0[t970];
          float t974 = scratch_1[t970];
          float t975 = t966 * t973;
          float t976 = t968 * t974;
          float t977 = t975 - t976;
          float t978 = t966 * t974;
          float t979 = t968 * t973;
          float t980 = t978 + t979;
          float t981 = t971 + t977;
          scratch_0[t969] = t981;
          float t983 = t972 + t980;
          scratch_1[t969] = t983;
          float t985 = t971 - t977;
          scratch_0[t970] = t985;
          float t987 = t972 - t980;
          scratch_1[t970] = t987;
        }
        for (uint t990 = 0; t990 < 256; t990++) {
          float t991 = (float)t990;
          float t992 = (t991 * 0.00390625);
          float t993 = metal::floor(t992);
          float t994 = t993 * 256.0;
          float t995 = t991 - t994;
          float t996 = t993 * 512.0;
          float t997 = t996 + t995;
          float t998 = t997 + 256.0;
          int t999 = (int)t995;
          int t1000 = 255 + t999;
          float t1001 = scratch_2[t1000];
          float t1002 = scratch_3[t1000];
          float t1003 = 0.0 - t1002;
          int t1004 = (int)t997;
          int t1005 = (int)t998;
          float t1006 = scratch_0[t1004];
          float t1007 = scratch_1[t1004];
          float t1008 = scratch_0[t1005];
          float t1009 = scratch_1[t1005];
          float t1010 = t1001 * t1008;
          float t1011 = t1003 * t1009;
          float t1012 = t1010 - t1011;
          float t1013 = t1001 * t1009;
          float t1014 = t1003 * t1008;
          float t1015 = t1013 + t1014;
          float t1016 = t1006 + t1012;
          scratch_0[t1004] = t1016;
          float t1018 = t1007 + t1015;
          scratch_1[t1004] = t1018;
          float t1020 = t1006 - t1012;
          scratch_0[t1005] = t1020;
          float t1022 = t1007 - t1015;
          scratch_1[t1005] = t1022;
        }
        for (uint t1025 = 0; t1025 < 512; t1025++) {
          float t1026 = scratch_0[(int)t1025];
          float t1027 = scratch_1[(int)t1025];
          int t1028 = t669 + t1025;
          memory[17499840 + t1028] = t1026;
          int t1030 = t669 + t1025;
          int t1031 = t1030 + 512;
          memory[17499840 + t1031] = t1027;
        }
        for (uint t1034 = 0; t1034 < 257; t1034++) {
          float t1035 = scratch_0[(int)t1034];
          float t1036 = scratch_1[(int)t1034];
          float t1037 = t1035 * t1035;
          float t1038 = t1036 * t1036;
          float t1039 = t1037 + t1038;
          float t1040 = metal::sqrt(t1039);
          int t1041 = t670 + t1034;
          memory[18548416 + t1041] = t1040;
        }
        int t1044 = t655 / 128;
        int t1045 = t1044 * 4;
        int t1046 = t1045 + t665;
        int t1047 = t1046 * 1024;
        int t1048 = t1046 * 257;
        for (uint t1049 = 0; t1049 < 512; t1049++) {
          float t1050 = (float)t1049;
          float t1051 = memory[92610 + (int)t1049];
          float t1052 = (float)t655;
          float t1053 = t1052 - 511.0;
          float t1054 = t1053 + t1050;
          float t1055 = t1054 >= 0.0;
          float t1056 = t1054 < frameCount;
          float t1057 = t1055 * t1056;
          float t1058 = frameCount - 1.0;
          float t1059 = metal::min(t1054, t1058);
          float t1060 = metal::max(0.0, t1059);
          int t1061 = (int)t1060;
          int t1062 = t1061 * 4;
          int t1063 = t1062 + t665;
          float t1064 = memory[9045696 + t1063];
          float t1065 = metal::select(0.0, t1064, t1057 > 0.0);
          float t1066 = t1065 * t1051;
          scratch_0[(int)t1049] = t1066;
          scratch_1[(int)t1049] = 0.0;
        }
        for (uint t1070 = 0; t1070 < 512; t1070++) {
          float t1071 = memory[94144 + (int)t1070];
          float t1072 = (float)t1070;
          float t1073 = t1072 < t1071;
          int t1074 = (int)t1071;
          float t1075 = scratch_0[(int)t1070];
          float t1076 = scratch_1[(int)t1070];
          float t1077 = scratch_0[t1074];
          float t1078 = scratch_1[t1074];
          float t1079 = metal::select(t1075, t1077, t1073 > 0.0);
          float t1080 = metal::select(t1076, t1078, t1073 > 0.0);
          float t1081 = metal::select(t1077, t1075, t1073 > 0.0);
          float t1082 = metal::select(t1078, t1076, t1073 > 0.0);
          scratch_0[(int)t1070] = t1079;
          scratch_1[(int)t1070] = t1080;
          scratch_0[t1074] = t1081;
          scratch_1[t1074] = t1082;
        }
        for (uint t1088 = 0; t1088 < 256; t1088++) {
          float t1089 = (float)t1088;
          float t1090 = t1089;
          float t1091 = metal::floor(t1090);
          float t1092 = t1091;
          float t1093 = t1089 - t1092;
          float t1094 = t1091 * 2.0;
          float t1095 = t1094 + t1093;
          float t1096 = t1095 + 1.0;
          int t1097 = (int)t1093;
          int t1098 = t1097;
          float t1099 = scratch_2[t1098];
          float t1100 = scratch_3[t1098];
          float t1101 = 0.0 - t1100;
          int t1102 = (int)t1095;
          int t1103 = (int)t1096;
          float t1104 = scratch_0[t1102];
          float t1105 = scratch_1[t1102];
          float t1106 = scratch_0[t1103];
          float t1107 = scratch_1[t1103];
          float t1108 = t1099 * t1106;
          float t1109 = t1101 * t1107;
          float t1110 = t1108 - t1109;
          float t1111 = t1099 * t1107;
          float t1112 = t1101 * t1106;
          float t1113 = t1111 + t1112;
          float t1114 = t1104 + t1110;
          scratch_0[t1102] = t1114;
          float t1116 = t1105 + t1113;
          scratch_1[t1102] = t1116;
          float t1118 = t1104 - t1110;
          scratch_0[t1103] = t1118;
          float t1120 = t1105 - t1113;
          scratch_1[t1103] = t1120;
        }
        for (uint t1123 = 0; t1123 < 256; t1123++) {
          float t1124 = (float)t1123;
          float t1125 = (t1124 * 0.5);
          float t1126 = metal::floor(t1125);
          float t1127 = t1126 * 2.0;
          float t1128 = t1124 - t1127;
          float t1129 = t1126 * 4.0;
          float t1130 = t1129 + t1128;
          float t1131 = t1130 + 2.0;
          int t1132 = (int)t1128;
          int t1133 = 1 + t1132;
          float t1134 = scratch_2[t1133];
          float t1135 = scratch_3[t1133];
          float t1136 = 0.0 - t1135;
          int t1137 = (int)t1130;
          int t1138 = (int)t1131;
          float t1139 = scratch_0[t1137];
          float t1140 = scratch_1[t1137];
          float t1141 = scratch_0[t1138];
          float t1142 = scratch_1[t1138];
          float t1143 = t1134 * t1141;
          float t1144 = t1136 * t1142;
          float t1145 = t1143 - t1144;
          float t1146 = t1134 * t1142;
          float t1147 = t1136 * t1141;
          float t1148 = t1146 + t1147;
          float t1149 = t1139 + t1145;
          scratch_0[t1137] = t1149;
          float t1151 = t1140 + t1148;
          scratch_1[t1137] = t1151;
          float t1153 = t1139 - t1145;
          scratch_0[t1138] = t1153;
          float t1155 = t1140 - t1148;
          scratch_1[t1138] = t1155;
        }
        for (uint t1158 = 0; t1158 < 256; t1158++) {
          float t1159 = (float)t1158;
          float t1160 = (t1159 * 0.25);
          float t1161 = metal::floor(t1160);
          float t1162 = t1161 * 4.0;
          float t1163 = t1159 - t1162;
          float t1164 = t1161 * 8.0;
          float t1165 = t1164 + t1163;
          float t1166 = t1165 + 4.0;
          int t1167 = (int)t1163;
          int t1168 = 3 + t1167;
          float t1169 = scratch_2[t1168];
          float t1170 = scratch_3[t1168];
          float t1171 = 0.0 - t1170;
          int t1172 = (int)t1165;
          int t1173 = (int)t1166;
          float t1174 = scratch_0[t1172];
          float t1175 = scratch_1[t1172];
          float t1176 = scratch_0[t1173];
          float t1177 = scratch_1[t1173];
          float t1178 = t1169 * t1176;
          float t1179 = t1171 * t1177;
          float t1180 = t1178 - t1179;
          float t1181 = t1169 * t1177;
          float t1182 = t1171 * t1176;
          float t1183 = t1181 + t1182;
          float t1184 = t1174 + t1180;
          scratch_0[t1172] = t1184;
          float t1186 = t1175 + t1183;
          scratch_1[t1172] = t1186;
          float t1188 = t1174 - t1180;
          scratch_0[t1173] = t1188;
          float t1190 = t1175 - t1183;
          scratch_1[t1173] = t1190;
        }
        for (uint t1193 = 0; t1193 < 256; t1193++) {
          float t1194 = (float)t1193;
          float t1195 = (t1194 * 0.125);
          float t1196 = metal::floor(t1195);
          float t1197 = t1196 * 8.0;
          float t1198 = t1194 - t1197;
          float t1199 = t1196 * 16.0;
          float t1200 = t1199 + t1198;
          float t1201 = t1200 + 8.0;
          int t1202 = (int)t1198;
          int t1203 = 7 + t1202;
          float t1204 = scratch_2[t1203];
          float t1205 = scratch_3[t1203];
          float t1206 = 0.0 - t1205;
          int t1207 = (int)t1200;
          int t1208 = (int)t1201;
          float t1209 = scratch_0[t1207];
          float t1210 = scratch_1[t1207];
          float t1211 = scratch_0[t1208];
          float t1212 = scratch_1[t1208];
          float t1213 = t1204 * t1211;
          float t1214 = t1206 * t1212;
          float t1215 = t1213 - t1214;
          float t1216 = t1204 * t1212;
          float t1217 = t1206 * t1211;
          float t1218 = t1216 + t1217;
          float t1219 = t1209 + t1215;
          scratch_0[t1207] = t1219;
          float t1221 = t1210 + t1218;
          scratch_1[t1207] = t1221;
          float t1223 = t1209 - t1215;
          scratch_0[t1208] = t1223;
          float t1225 = t1210 - t1218;
          scratch_1[t1208] = t1225;
        }
        for (uint t1228 = 0; t1228 < 256; t1228++) {
          float t1229 = (float)t1228;
          float t1230 = (t1229 * 0.0625);
          float t1231 = metal::floor(t1230);
          float t1232 = t1231 * 16.0;
          float t1233 = t1229 - t1232;
          float t1234 = t1231 * 32.0;
          float t1235 = t1234 + t1233;
          float t1236 = t1235 + 16.0;
          int t1237 = (int)t1233;
          int t1238 = 15 + t1237;
          float t1239 = scratch_2[t1238];
          float t1240 = scratch_3[t1238];
          float t1241 = 0.0 - t1240;
          int t1242 = (int)t1235;
          int t1243 = (int)t1236;
          float t1244 = scratch_0[t1242];
          float t1245 = scratch_1[t1242];
          float t1246 = scratch_0[t1243];
          float t1247 = scratch_1[t1243];
          float t1248 = t1239 * t1246;
          float t1249 = t1241 * t1247;
          float t1250 = t1248 - t1249;
          float t1251 = t1239 * t1247;
          float t1252 = t1241 * t1246;
          float t1253 = t1251 + t1252;
          float t1254 = t1244 + t1250;
          scratch_0[t1242] = t1254;
          float t1256 = t1245 + t1253;
          scratch_1[t1242] = t1256;
          float t1258 = t1244 - t1250;
          scratch_0[t1243] = t1258;
          float t1260 = t1245 - t1253;
          scratch_1[t1243] = t1260;
        }
        for (uint t1263 = 0; t1263 < 256; t1263++) {
          float t1264 = (float)t1263;
          float t1265 = (t1264 * 0.03125);
          float t1266 = metal::floor(t1265);
          float t1267 = t1266 * 32.0;
          float t1268 = t1264 - t1267;
          float t1269 = t1266 * 64.0;
          float t1270 = t1269 + t1268;
          float t1271 = t1270 + 32.0;
          int t1272 = (int)t1268;
          int t1273 = 31 + t1272;
          float t1274 = scratch_2[t1273];
          float t1275 = scratch_3[t1273];
          float t1276 = 0.0 - t1275;
          int t1277 = (int)t1270;
          int t1278 = (int)t1271;
          float t1279 = scratch_0[t1277];
          float t1280 = scratch_1[t1277];
          float t1281 = scratch_0[t1278];
          float t1282 = scratch_1[t1278];
          float t1283 = t1274 * t1281;
          float t1284 = t1276 * t1282;
          float t1285 = t1283 - t1284;
          float t1286 = t1274 * t1282;
          float t1287 = t1276 * t1281;
          float t1288 = t1286 + t1287;
          float t1289 = t1279 + t1285;
          scratch_0[t1277] = t1289;
          float t1291 = t1280 + t1288;
          scratch_1[t1277] = t1291;
          float t1293 = t1279 - t1285;
          scratch_0[t1278] = t1293;
          float t1295 = t1280 - t1288;
          scratch_1[t1278] = t1295;
        }
        for (uint t1298 = 0; t1298 < 256; t1298++) {
          float t1299 = (float)t1298;
          float t1300 = (t1299 * 0.015625);
          float t1301 = metal::floor(t1300);
          float t1302 = t1301 * 64.0;
          float t1303 = t1299 - t1302;
          float t1304 = t1301 * 128.0;
          float t1305 = t1304 + t1303;
          float t1306 = t1305 + 64.0;
          int t1307 = (int)t1303;
          int t1308 = 63 + t1307;
          float t1309 = scratch_2[t1308];
          float t1310 = scratch_3[t1308];
          float t1311 = 0.0 - t1310;
          int t1312 = (int)t1305;
          int t1313 = (int)t1306;
          float t1314 = scratch_0[t1312];
          float t1315 = scratch_1[t1312];
          float t1316 = scratch_0[t1313];
          float t1317 = scratch_1[t1313];
          float t1318 = t1309 * t1316;
          float t1319 = t1311 * t1317;
          float t1320 = t1318 - t1319;
          float t1321 = t1309 * t1317;
          float t1322 = t1311 * t1316;
          float t1323 = t1321 + t1322;
          float t1324 = t1314 + t1320;
          scratch_0[t1312] = t1324;
          float t1326 = t1315 + t1323;
          scratch_1[t1312] = t1326;
          float t1328 = t1314 - t1320;
          scratch_0[t1313] = t1328;
          float t1330 = t1315 - t1323;
          scratch_1[t1313] = t1330;
        }
        for (uint t1333 = 0; t1333 < 256; t1333++) {
          float t1334 = (float)t1333;
          float t1335 = (t1334 * 0.0078125);
          float t1336 = metal::floor(t1335);
          float t1337 = t1336 * 128.0;
          float t1338 = t1334 - t1337;
          float t1339 = t1336 * 256.0;
          float t1340 = t1339 + t1338;
          float t1341 = t1340 + 128.0;
          int t1342 = (int)t1338;
          int t1343 = 127 + t1342;
          float t1344 = scratch_2[t1343];
          float t1345 = scratch_3[t1343];
          float t1346 = 0.0 - t1345;
          int t1347 = (int)t1340;
          int t1348 = (int)t1341;
          float t1349 = scratch_0[t1347];
          float t1350 = scratch_1[t1347];
          float t1351 = scratch_0[t1348];
          float t1352 = scratch_1[t1348];
          float t1353 = t1344 * t1351;
          float t1354 = t1346 * t1352;
          float t1355 = t1353 - t1354;
          float t1356 = t1344 * t1352;
          float t1357 = t1346 * t1351;
          float t1358 = t1356 + t1357;
          float t1359 = t1349 + t1355;
          scratch_0[t1347] = t1359;
          float t1361 = t1350 + t1358;
          scratch_1[t1347] = t1361;
          float t1363 = t1349 - t1355;
          scratch_0[t1348] = t1363;
          float t1365 = t1350 - t1358;
          scratch_1[t1348] = t1365;
        }
        for (uint t1368 = 0; t1368 < 256; t1368++) {
          float t1369 = (float)t1368;
          float t1370 = (t1369 * 0.00390625);
          float t1371 = metal::floor(t1370);
          float t1372 = t1371 * 256.0;
          float t1373 = t1369 - t1372;
          float t1374 = t1371 * 512.0;
          float t1375 = t1374 + t1373;
          float t1376 = t1375 + 256.0;
          int t1377 = (int)t1373;
          int t1378 = 255 + t1377;
          float t1379 = scratch_2[t1378];
          float t1380 = scratch_3[t1378];
          float t1381 = 0.0 - t1380;
          int t1382 = (int)t1375;
          int t1383 = (int)t1376;
          float t1384 = scratch_0[t1382];
          float t1385 = scratch_1[t1382];
          float t1386 = scratch_0[t1383];
          float t1387 = scratch_1[t1383];
          float t1388 = t1379 * t1386;
          float t1389 = t1381 * t1387;
          float t1390 = t1388 - t1389;
          float t1391 = t1379 * t1387;
          float t1392 = t1381 * t1386;
          float t1393 = t1391 + t1392;
          float t1394 = t1384 + t1390;
          scratch_0[t1382] = t1394;
          float t1396 = t1385 + t1393;
          scratch_1[t1382] = t1396;
          float t1398 = t1384 - t1390;
          scratch_0[t1383] = t1398;
          float t1400 = t1385 - t1393;
          scratch_1[t1383] = t1400;
        }
        for (uint t1403 = 0; t1403 < 512; t1403++) {
          float t1404 = scratch_0[(int)t1403];
          float t1405 = scratch_1[(int)t1403];
          int t1406 = t1047 + t1403;
          memory[18024128 + t1406] = t1404;
          int t1408 = t1047 + t1403;
          int t1409 = t1408 + 512;
          memory[18024128 + t1409] = t1405;
        }
        for (uint t1412 = 0; t1412 < 257; t1412++) {
          float t1413 = scratch_0[(int)t1412];
          float t1414 = scratch_1[(int)t1412];
          float t1415 = t1413 * t1413;
          float t1416 = t1414 * t1414;
          float t1417 = t1415 + t1416;
          float t1418 = metal::sqrt(t1417);
          int t1419 = t1048 + t1412;
          memory[18680000 + t1419] = t1418;
        }
        int t1422 = t655 / 128;
        int t1423 = t1422 * 4;
        int t1424 = t1423 + t665;
        int t1425 = t1424 * 257;
        for (uint t1426 = 0; t1426 < 257; t1426++) {
          int t1427 = t1425 + t1426;
          float t1428 = memory[18548416 + t1427];
          int t1429 = t1425 + t1426;
          float t1430 = memory[18680000 + t1429];
          float t1431 = t1428 - t1430;
          float t1432 = t1431 * t1431;
          float t1433 = t657 + t1432;
          t657 = t1433;
        }
      }
    }
    t[4*frameCount + id] = (t657 * 0.25);
  }
  #pragma clang diagnostic pop
}



// KERNEL 21
// FrameOrder: parallel
// DispatchMode: perFrameScaled(1024)
kernel void kernel_21(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1443), value: global(1443)) */
  float t4904 = frameCount * 1024.0;
  if (id >= 0 && id < (uint)(t4904)) {
    /* loadGlobal(1437) - handled in variable access */
    int t1438 = id;
    int t1439 = t1438 / 1024;
    uint _frameIndex = (uint)(t1439);
    int t1440 = t1439 * 1024;
    int t1441 = t1438 - t1440;
    float t1442 = (t[4*frameCount + _frameIndex] * 6.1035156e-05);
    t[5*frameCount + _frameIndex] = t1442;
  }
  #pragma clang diagnostic pop
}



// KERNEL 22
// FrameOrder: sequential
// DispatchMode: singleThreaded
kernel void kernel_22(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1452), value: global(1452)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[6*frameCount + i] = memory[20664580];
      float t1453 = t[6*frameCount + i] + 1.0;
      float t1454 = metal::select(t1453, 0.0, 0.0 > 0.0);
      float t1455 = t1454;
      float t1456 = (t1455 * 0.00390625);
      float t1457 = metal::floor(t1456);
      float t1458 = t1457 * 256.0;
      float t1459 = t1454 - t1458;
      memory[20664580] = t1459;
      float t1461 = t1459 >= 256.0;
      if (t1461) {
        float t1463 = t1459 - 256.0;
        memory[20664580] = t1463;
      }
      if (0.0) {
        memory[20664580] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 23
// FrameOrder: parallel
// DispatchMode: perFrameThreadgroup1
kernel void kernel_23(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2321), value: global(2321)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1452) - handled in variable access */
    int t1469 = id;
    threadgroup float scratch_0[1024];
    threadgroup float scratch_1[1024];
    threadgroup float scratch_2[1023];
    threadgroup float scratch_3[1023];
    float t1470 = t[6*frameCount + id] == 0.0;
    float t1471 = 0.0;
    if (t1470) {
      for (uint t1473 = 0; t1473 < 1023; t1473++) {
        float t1474 = memory[95680 + (int)t1473];
        scratch_2[(int)t1473] = t1474;
        float t1476 = memory[96703 + (int)t1473];
        scratch_3[(int)t1473] = t1476;
      }
      for (uint t1479 = 0; t1479 < 4; t1479++) {
        int t1480 = t1469 / 256;
        int t1481 = t1480 * 4;
        int t1482 = t1481 + t1479;
        int t1483 = t1482 * 2048;
        int t1484 = t1482 * 513;
        for (uint t1485 = 0; t1485 < 1024; t1485++) {
          float t1486 = (float)t1485;
          float t1487 = memory[94656 + (int)t1485];
          float t1488 = (float)t1469;
          float t1489 = t1488 - 1023.0;
          float t1490 = t1489 + t1486;
          float t1491 = t1490 >= 0.0;
          float t1492 = t1490 < frameCount;
          float t1493 = t1491 * t1492;
          float t1494 = frameCount - 1.0;
          float t1495 = metal::min(t1490, t1494);
          float t1496 = metal::max(0.0, t1495);
          int t1497 = (int)t1496;
          int t1498 = t1497 * 4;
          int t1499 = t1498 + t1479;
          float t1500 = memory[8783552 + t1499];
          float t1501 = metal::select(0.0, t1500, t1493 > 0.0);
          float t1502 = t1501 * t1487;
          scratch_0[(int)t1485] = t1502;
          scratch_1[(int)t1485] = 0.0;
        }
        for (uint t1506 = 0; t1506 < 1024; t1506++) {
          float t1507 = memory[97726 + (int)t1506];
          float t1508 = (float)t1506;
          float t1509 = t1508 < t1507;
          int t1510 = (int)t1507;
          float t1511 = scratch_0[(int)t1506];
          float t1512 = scratch_1[(int)t1506];
          float t1513 = scratch_0[t1510];
          float t1514 = scratch_1[t1510];
          float t1515 = metal::select(t1511, t1513, t1509 > 0.0);
          float t1516 = metal::select(t1512, t1514, t1509 > 0.0);
          float t1517 = metal::select(t1513, t1511, t1509 > 0.0);
          float t1518 = metal::select(t1514, t1512, t1509 > 0.0);
          scratch_0[(int)t1506] = t1515;
          scratch_1[(int)t1506] = t1516;
          scratch_0[t1510] = t1517;
          scratch_1[t1510] = t1518;
        }
        for (uint t1524 = 0; t1524 < 512; t1524++) {
          float t1525 = (float)t1524;
          float t1526 = t1525;
          float t1527 = metal::floor(t1526);
          float t1528 = t1527;
          float t1529 = t1525 - t1528;
          float t1530 = t1527 * 2.0;
          float t1531 = t1530 + t1529;
          float t1532 = t1531 + 1.0;
          int t1533 = (int)t1529;
          int t1534 = t1533;
          float t1535 = scratch_2[t1534];
          float t1536 = scratch_3[t1534];
          float t1537 = 0.0 - t1536;
          int t1538 = (int)t1531;
          int t1539 = (int)t1532;
          float t1540 = scratch_0[t1538];
          float t1541 = scratch_1[t1538];
          float t1542 = scratch_0[t1539];
          float t1543 = scratch_1[t1539];
          float t1544 = t1535 * t1542;
          float t1545 = t1537 * t1543;
          float t1546 = t1544 - t1545;
          float t1547 = t1535 * t1543;
          float t1548 = t1537 * t1542;
          float t1549 = t1547 + t1548;
          float t1550 = t1540 + t1546;
          scratch_0[t1538] = t1550;
          float t1552 = t1541 + t1549;
          scratch_1[t1538] = t1552;
          float t1554 = t1540 - t1546;
          scratch_0[t1539] = t1554;
          float t1556 = t1541 - t1549;
          scratch_1[t1539] = t1556;
        }
        for (uint t1559 = 0; t1559 < 512; t1559++) {
          float t1560 = (float)t1559;
          float t1561 = (t1560 * 0.5);
          float t1562 = metal::floor(t1561);
          float t1563 = t1562 * 2.0;
          float t1564 = t1560 - t1563;
          float t1565 = t1562 * 4.0;
          float t1566 = t1565 + t1564;
          float t1567 = t1566 + 2.0;
          int t1568 = (int)t1564;
          int t1569 = 1 + t1568;
          float t1570 = scratch_2[t1569];
          float t1571 = scratch_3[t1569];
          float t1572 = 0.0 - t1571;
          int t1573 = (int)t1566;
          int t1574 = (int)t1567;
          float t1575 = scratch_0[t1573];
          float t1576 = scratch_1[t1573];
          float t1577 = scratch_0[t1574];
          float t1578 = scratch_1[t1574];
          float t1579 = t1570 * t1577;
          float t1580 = t1572 * t1578;
          float t1581 = t1579 - t1580;
          float t1582 = t1570 * t1578;
          float t1583 = t1572 * t1577;
          float t1584 = t1582 + t1583;
          float t1585 = t1575 + t1581;
          scratch_0[t1573] = t1585;
          float t1587 = t1576 + t1584;
          scratch_1[t1573] = t1587;
          float t1589 = t1575 - t1581;
          scratch_0[t1574] = t1589;
          float t1591 = t1576 - t1584;
          scratch_1[t1574] = t1591;
        }
        for (uint t1594 = 0; t1594 < 512; t1594++) {
          float t1595 = (float)t1594;
          float t1596 = (t1595 * 0.25);
          float t1597 = metal::floor(t1596);
          float t1598 = t1597 * 4.0;
          float t1599 = t1595 - t1598;
          float t1600 = t1597 * 8.0;
          float t1601 = t1600 + t1599;
          float t1602 = t1601 + 4.0;
          int t1603 = (int)t1599;
          int t1604 = 3 + t1603;
          float t1605 = scratch_2[t1604];
          float t1606 = scratch_3[t1604];
          float t1607 = 0.0 - t1606;
          int t1608 = (int)t1601;
          int t1609 = (int)t1602;
          float t1610 = scratch_0[t1608];
          float t1611 = scratch_1[t1608];
          float t1612 = scratch_0[t1609];
          float t1613 = scratch_1[t1609];
          float t1614 = t1605 * t1612;
          float t1615 = t1607 * t1613;
          float t1616 = t1614 - t1615;
          float t1617 = t1605 * t1613;
          float t1618 = t1607 * t1612;
          float t1619 = t1617 + t1618;
          float t1620 = t1610 + t1616;
          scratch_0[t1608] = t1620;
          float t1622 = t1611 + t1619;
          scratch_1[t1608] = t1622;
          float t1624 = t1610 - t1616;
          scratch_0[t1609] = t1624;
          float t1626 = t1611 - t1619;
          scratch_1[t1609] = t1626;
        }
        for (uint t1629 = 0; t1629 < 512; t1629++) {
          float t1630 = (float)t1629;
          float t1631 = (t1630 * 0.125);
          float t1632 = metal::floor(t1631);
          float t1633 = t1632 * 8.0;
          float t1634 = t1630 - t1633;
          float t1635 = t1632 * 16.0;
          float t1636 = t1635 + t1634;
          float t1637 = t1636 + 8.0;
          int t1638 = (int)t1634;
          int t1639 = 7 + t1638;
          float t1640 = scratch_2[t1639];
          float t1641 = scratch_3[t1639];
          float t1642 = 0.0 - t1641;
          int t1643 = (int)t1636;
          int t1644 = (int)t1637;
          float t1645 = scratch_0[t1643];
          float t1646 = scratch_1[t1643];
          float t1647 = scratch_0[t1644];
          float t1648 = scratch_1[t1644];
          float t1649 = t1640 * t1647;
          float t1650 = t1642 * t1648;
          float t1651 = t1649 - t1650;
          float t1652 = t1640 * t1648;
          float t1653 = t1642 * t1647;
          float t1654 = t1652 + t1653;
          float t1655 = t1645 + t1651;
          scratch_0[t1643] = t1655;
          float t1657 = t1646 + t1654;
          scratch_1[t1643] = t1657;
          float t1659 = t1645 - t1651;
          scratch_0[t1644] = t1659;
          float t1661 = t1646 - t1654;
          scratch_1[t1644] = t1661;
        }
        for (uint t1664 = 0; t1664 < 512; t1664++) {
          float t1665 = (float)t1664;
          float t1666 = (t1665 * 0.0625);
          float t1667 = metal::floor(t1666);
          float t1668 = t1667 * 16.0;
          float t1669 = t1665 - t1668;
          float t1670 = t1667 * 32.0;
          float t1671 = t1670 + t1669;
          float t1672 = t1671 + 16.0;
          int t1673 = (int)t1669;
          int t1674 = 15 + t1673;
          float t1675 = scratch_2[t1674];
          float t1676 = scratch_3[t1674];
          float t1677 = 0.0 - t1676;
          int t1678 = (int)t1671;
          int t1679 = (int)t1672;
          float t1680 = scratch_0[t1678];
          float t1681 = scratch_1[t1678];
          float t1682 = scratch_0[t1679];
          float t1683 = scratch_1[t1679];
          float t1684 = t1675 * t1682;
          float t1685 = t1677 * t1683;
          float t1686 = t1684 - t1685;
          float t1687 = t1675 * t1683;
          float t1688 = t1677 * t1682;
          float t1689 = t1687 + t1688;
          float t1690 = t1680 + t1686;
          scratch_0[t1678] = t1690;
          float t1692 = t1681 + t1689;
          scratch_1[t1678] = t1692;
          float t1694 = t1680 - t1686;
          scratch_0[t1679] = t1694;
          float t1696 = t1681 - t1689;
          scratch_1[t1679] = t1696;
        }
        for (uint t1699 = 0; t1699 < 512; t1699++) {
          float t1700 = (float)t1699;
          float t1701 = (t1700 * 0.03125);
          float t1702 = metal::floor(t1701);
          float t1703 = t1702 * 32.0;
          float t1704 = t1700 - t1703;
          float t1705 = t1702 * 64.0;
          float t1706 = t1705 + t1704;
          float t1707 = t1706 + 32.0;
          int t1708 = (int)t1704;
          int t1709 = 31 + t1708;
          float t1710 = scratch_2[t1709];
          float t1711 = scratch_3[t1709];
          float t1712 = 0.0 - t1711;
          int t1713 = (int)t1706;
          int t1714 = (int)t1707;
          float t1715 = scratch_0[t1713];
          float t1716 = scratch_1[t1713];
          float t1717 = scratch_0[t1714];
          float t1718 = scratch_1[t1714];
          float t1719 = t1710 * t1717;
          float t1720 = t1712 * t1718;
          float t1721 = t1719 - t1720;
          float t1722 = t1710 * t1718;
          float t1723 = t1712 * t1717;
          float t1724 = t1722 + t1723;
          float t1725 = t1715 + t1721;
          scratch_0[t1713] = t1725;
          float t1727 = t1716 + t1724;
          scratch_1[t1713] = t1727;
          float t1729 = t1715 - t1721;
          scratch_0[t1714] = t1729;
          float t1731 = t1716 - t1724;
          scratch_1[t1714] = t1731;
        }
        for (uint t1734 = 0; t1734 < 512; t1734++) {
          float t1735 = (float)t1734;
          float t1736 = (t1735 * 0.015625);
          float t1737 = metal::floor(t1736);
          float t1738 = t1737 * 64.0;
          float t1739 = t1735 - t1738;
          float t1740 = t1737 * 128.0;
          float t1741 = t1740 + t1739;
          float t1742 = t1741 + 64.0;
          int t1743 = (int)t1739;
          int t1744 = 63 + t1743;
          float t1745 = scratch_2[t1744];
          float t1746 = scratch_3[t1744];
          float t1747 = 0.0 - t1746;
          int t1748 = (int)t1741;
          int t1749 = (int)t1742;
          float t1750 = scratch_0[t1748];
          float t1751 = scratch_1[t1748];
          float t1752 = scratch_0[t1749];
          float t1753 = scratch_1[t1749];
          float t1754 = t1745 * t1752;
          float t1755 = t1747 * t1753;
          float t1756 = t1754 - t1755;
          float t1757 = t1745 * t1753;
          float t1758 = t1747 * t1752;
          float t1759 = t1757 + t1758;
          float t1760 = t1750 + t1756;
          scratch_0[t1748] = t1760;
          float t1762 = t1751 + t1759;
          scratch_1[t1748] = t1762;
          float t1764 = t1750 - t1756;
          scratch_0[t1749] = t1764;
          float t1766 = t1751 - t1759;
          scratch_1[t1749] = t1766;
        }
        for (uint t1769 = 0; t1769 < 512; t1769++) {
          float t1770 = (float)t1769;
          float t1771 = (t1770 * 0.0078125);
          float t1772 = metal::floor(t1771);
          float t1773 = t1772 * 128.0;
          float t1774 = t1770 - t1773;
          float t1775 = t1772 * 256.0;
          float t1776 = t1775 + t1774;
          float t1777 = t1776 + 128.0;
          int t1778 = (int)t1774;
          int t1779 = 127 + t1778;
          float t1780 = scratch_2[t1779];
          float t1781 = scratch_3[t1779];
          float t1782 = 0.0 - t1781;
          int t1783 = (int)t1776;
          int t1784 = (int)t1777;
          float t1785 = scratch_0[t1783];
          float t1786 = scratch_1[t1783];
          float t1787 = scratch_0[t1784];
          float t1788 = scratch_1[t1784];
          float t1789 = t1780 * t1787;
          float t1790 = t1782 * t1788;
          float t1791 = t1789 - t1790;
          float t1792 = t1780 * t1788;
          float t1793 = t1782 * t1787;
          float t1794 = t1792 + t1793;
          float t1795 = t1785 + t1791;
          scratch_0[t1783] = t1795;
          float t1797 = t1786 + t1794;
          scratch_1[t1783] = t1797;
          float t1799 = t1785 - t1791;
          scratch_0[t1784] = t1799;
          float t1801 = t1786 - t1794;
          scratch_1[t1784] = t1801;
        }
        for (uint t1804 = 0; t1804 < 512; t1804++) {
          float t1805 = (float)t1804;
          float t1806 = (t1805 * 0.00390625);
          float t1807 = metal::floor(t1806);
          float t1808 = t1807 * 256.0;
          float t1809 = t1805 - t1808;
          float t1810 = t1807 * 512.0;
          float t1811 = t1810 + t1809;
          float t1812 = t1811 + 256.0;
          int t1813 = (int)t1809;
          int t1814 = 255 + t1813;
          float t1815 = scratch_2[t1814];
          float t1816 = scratch_3[t1814];
          float t1817 = 0.0 - t1816;
          int t1818 = (int)t1811;
          int t1819 = (int)t1812;
          float t1820 = scratch_0[t1818];
          float t1821 = scratch_1[t1818];
          float t1822 = scratch_0[t1819];
          float t1823 = scratch_1[t1819];
          float t1824 = t1815 * t1822;
          float t1825 = t1817 * t1823;
          float t1826 = t1824 - t1825;
          float t1827 = t1815 * t1823;
          float t1828 = t1817 * t1822;
          float t1829 = t1827 + t1828;
          float t1830 = t1820 + t1826;
          scratch_0[t1818] = t1830;
          float t1832 = t1821 + t1829;
          scratch_1[t1818] = t1832;
          float t1834 = t1820 - t1826;
          scratch_0[t1819] = t1834;
          float t1836 = t1821 - t1829;
          scratch_1[t1819] = t1836;
        }
        for (uint t1839 = 0; t1839 < 512; t1839++) {
          float t1840 = (float)t1839;
          float t1841 = (t1840 * 0.001953125);
          float t1842 = metal::floor(t1841);
          float t1843 = t1842 * 512.0;
          float t1844 = t1840 - t1843;
          float t1845 = t1842 * 1024.0;
          float t1846 = t1845 + t1844;
          float t1847 = t1846 + 512.0;
          int t1848 = (int)t1844;
          int t1849 = 511 + t1848;
          float t1850 = scratch_2[t1849];
          float t1851 = scratch_3[t1849];
          float t1852 = 0.0 - t1851;
          int t1853 = (int)t1846;
          int t1854 = (int)t1847;
          float t1855 = scratch_0[t1853];
          float t1856 = scratch_1[t1853];
          float t1857 = scratch_0[t1854];
          float t1858 = scratch_1[t1854];
          float t1859 = t1850 * t1857;
          float t1860 = t1852 * t1858;
          float t1861 = t1859 - t1860;
          float t1862 = t1850 * t1858;
          float t1863 = t1852 * t1857;
          float t1864 = t1862 + t1863;
          float t1865 = t1855 + t1861;
          scratch_0[t1853] = t1865;
          float t1867 = t1856 + t1864;
          scratch_1[t1853] = t1867;
          float t1869 = t1855 - t1861;
          scratch_0[t1854] = t1869;
          float t1871 = t1856 - t1864;
          scratch_1[t1854] = t1871;
        }
        for (uint t1874 = 0; t1874 < 1024; t1874++) {
          float t1875 = scratch_0[(int)t1874];
          float t1876 = scratch_1[(int)t1874];
          int t1877 = t1483 + t1874;
          memory[18811584 + t1877] = t1875;
          int t1879 = t1483 + t1874;
          int t1880 = t1879 + 1024;
          memory[18811584 + t1880] = t1876;
        }
        for (uint t1883 = 0; t1883 < 513; t1883++) {
          float t1884 = scratch_0[(int)t1883];
          float t1885 = scratch_1[(int)t1883];
          float t1886 = t1884 * t1884;
          float t1887 = t1885 * t1885;
          float t1888 = t1886 + t1887;
          float t1889 = metal::sqrt(t1888);
          int t1890 = t1484 + t1883;
          memory[19860160 + t1890] = t1889;
        }
        int t1893 = t1469 / 256;
        int t1894 = t1893 * 4;
        int t1895 = t1894 + t1479;
        int t1896 = t1895 * 2048;
        int t1897 = t1895 * 513;
        for (uint t1898 = 0; t1898 < 1024; t1898++) {
          float t1899 = (float)t1898;
          float t1900 = memory[94656 + (int)t1898];
          float t1901 = (float)t1469;
          float t1902 = t1901 - 1023.0;
          float t1903 = t1902 + t1899;
          float t1904 = t1903 >= 0.0;
          float t1905 = t1903 < frameCount;
          float t1906 = t1904 * t1905;
          float t1907 = frameCount - 1.0;
          float t1908 = metal::min(t1903, t1907);
          float t1909 = metal::max(0.0, t1908);
          int t1910 = (int)t1909;
          int t1911 = t1910 * 4;
          int t1912 = t1911 + t1479;
          float t1913 = memory[9045696 + t1912];
          float t1914 = metal::select(0.0, t1913, t1906 > 0.0);
          float t1915 = t1914 * t1900;
          scratch_0[(int)t1898] = t1915;
          scratch_1[(int)t1898] = 0.0;
        }
        for (uint t1919 = 0; t1919 < 1024; t1919++) {
          float t1920 = memory[97726 + (int)t1919];
          float t1921 = (float)t1919;
          float t1922 = t1921 < t1920;
          int t1923 = (int)t1920;
          float t1924 = scratch_0[(int)t1919];
          float t1925 = scratch_1[(int)t1919];
          float t1926 = scratch_0[t1923];
          float t1927 = scratch_1[t1923];
          float t1928 = metal::select(t1924, t1926, t1922 > 0.0);
          float t1929 = metal::select(t1925, t1927, t1922 > 0.0);
          float t1930 = metal::select(t1926, t1924, t1922 > 0.0);
          float t1931 = metal::select(t1927, t1925, t1922 > 0.0);
          scratch_0[(int)t1919] = t1928;
          scratch_1[(int)t1919] = t1929;
          scratch_0[t1923] = t1930;
          scratch_1[t1923] = t1931;
        }
        for (uint t1937 = 0; t1937 < 512; t1937++) {
          float t1938 = (float)t1937;
          float t1939 = t1938;
          float t1940 = metal::floor(t1939);
          float t1941 = t1940;
          float t1942 = t1938 - t1941;
          float t1943 = t1940 * 2.0;
          float t1944 = t1943 + t1942;
          float t1945 = t1944 + 1.0;
          int t1946 = (int)t1942;
          int t1947 = t1946;
          float t1948 = scratch_2[t1947];
          float t1949 = scratch_3[t1947];
          float t1950 = 0.0 - t1949;
          int t1951 = (int)t1944;
          int t1952 = (int)t1945;
          float t1953 = scratch_0[t1951];
          float t1954 = scratch_1[t1951];
          float t1955 = scratch_0[t1952];
          float t1956 = scratch_1[t1952];
          float t1957 = t1948 * t1955;
          float t1958 = t1950 * t1956;
          float t1959 = t1957 - t1958;
          float t1960 = t1948 * t1956;
          float t1961 = t1950 * t1955;
          float t1962 = t1960 + t1961;
          float t1963 = t1953 + t1959;
          scratch_0[t1951] = t1963;
          float t1965 = t1954 + t1962;
          scratch_1[t1951] = t1965;
          float t1967 = t1953 - t1959;
          scratch_0[t1952] = t1967;
          float t1969 = t1954 - t1962;
          scratch_1[t1952] = t1969;
        }
        for (uint t1972 = 0; t1972 < 512; t1972++) {
          float t1973 = (float)t1972;
          float t1974 = (t1973 * 0.5);
          float t1975 = metal::floor(t1974);
          float t1976 = t1975 * 2.0;
          float t1977 = t1973 - t1976;
          float t1978 = t1975 * 4.0;
          float t1979 = t1978 + t1977;
          float t1980 = t1979 + 2.0;
          int t1981 = (int)t1977;
          int t1982 = 1 + t1981;
          float t1983 = scratch_2[t1982];
          float t1984 = scratch_3[t1982];
          float t1985 = 0.0 - t1984;
          int t1986 = (int)t1979;
          int t1987 = (int)t1980;
          float t1988 = scratch_0[t1986];
          float t1989 = scratch_1[t1986];
          float t1990 = scratch_0[t1987];
          float t1991 = scratch_1[t1987];
          float t1992 = t1983 * t1990;
          float t1993 = t1985 * t1991;
          float t1994 = t1992 - t1993;
          float t1995 = t1983 * t1991;
          float t1996 = t1985 * t1990;
          float t1997 = t1995 + t1996;
          float t1998 = t1988 + t1994;
          scratch_0[t1986] = t1998;
          float t2000 = t1989 + t1997;
          scratch_1[t1986] = t2000;
          float t2002 = t1988 - t1994;
          scratch_0[t1987] = t2002;
          float t2004 = t1989 - t1997;
          scratch_1[t1987] = t2004;
        }
        for (uint t2007 = 0; t2007 < 512; t2007++) {
          float t2008 = (float)t2007;
          float t2009 = (t2008 * 0.25);
          float t2010 = metal::floor(t2009);
          float t2011 = t2010 * 4.0;
          float t2012 = t2008 - t2011;
          float t2013 = t2010 * 8.0;
          float t2014 = t2013 + t2012;
          float t2015 = t2014 + 4.0;
          int t2016 = (int)t2012;
          int t2017 = 3 + t2016;
          float t2018 = scratch_2[t2017];
          float t2019 = scratch_3[t2017];
          float t2020 = 0.0 - t2019;
          int t2021 = (int)t2014;
          int t2022 = (int)t2015;
          float t2023 = scratch_0[t2021];
          float t2024 = scratch_1[t2021];
          float t2025 = scratch_0[t2022];
          float t2026 = scratch_1[t2022];
          float t2027 = t2018 * t2025;
          float t2028 = t2020 * t2026;
          float t2029 = t2027 - t2028;
          float t2030 = t2018 * t2026;
          float t2031 = t2020 * t2025;
          float t2032 = t2030 + t2031;
          float t2033 = t2023 + t2029;
          scratch_0[t2021] = t2033;
          float t2035 = t2024 + t2032;
          scratch_1[t2021] = t2035;
          float t2037 = t2023 - t2029;
          scratch_0[t2022] = t2037;
          float t2039 = t2024 - t2032;
          scratch_1[t2022] = t2039;
        }
        for (uint t2042 = 0; t2042 < 512; t2042++) {
          float t2043 = (float)t2042;
          float t2044 = (t2043 * 0.125);
          float t2045 = metal::floor(t2044);
          float t2046 = t2045 * 8.0;
          float t2047 = t2043 - t2046;
          float t2048 = t2045 * 16.0;
          float t2049 = t2048 + t2047;
          float t2050 = t2049 + 8.0;
          int t2051 = (int)t2047;
          int t2052 = 7 + t2051;
          float t2053 = scratch_2[t2052];
          float t2054 = scratch_3[t2052];
          float t2055 = 0.0 - t2054;
          int t2056 = (int)t2049;
          int t2057 = (int)t2050;
          float t2058 = scratch_0[t2056];
          float t2059 = scratch_1[t2056];
          float t2060 = scratch_0[t2057];
          float t2061 = scratch_1[t2057];
          float t2062 = t2053 * t2060;
          float t2063 = t2055 * t2061;
          float t2064 = t2062 - t2063;
          float t2065 = t2053 * t2061;
          float t2066 = t2055 * t2060;
          float t2067 = t2065 + t2066;
          float t2068 = t2058 + t2064;
          scratch_0[t2056] = t2068;
          float t2070 = t2059 + t2067;
          scratch_1[t2056] = t2070;
          float t2072 = t2058 - t2064;
          scratch_0[t2057] = t2072;
          float t2074 = t2059 - t2067;
          scratch_1[t2057] = t2074;
        }
        for (uint t2077 = 0; t2077 < 512; t2077++) {
          float t2078 = (float)t2077;
          float t2079 = (t2078 * 0.0625);
          float t2080 = metal::floor(t2079);
          float t2081 = t2080 * 16.0;
          float t2082 = t2078 - t2081;
          float t2083 = t2080 * 32.0;
          float t2084 = t2083 + t2082;
          float t2085 = t2084 + 16.0;
          int t2086 = (int)t2082;
          int t2087 = 15 + t2086;
          float t2088 = scratch_2[t2087];
          float t2089 = scratch_3[t2087];
          float t2090 = 0.0 - t2089;
          int t2091 = (int)t2084;
          int t2092 = (int)t2085;
          float t2093 = scratch_0[t2091];
          float t2094 = scratch_1[t2091];
          float t2095 = scratch_0[t2092];
          float t2096 = scratch_1[t2092];
          float t2097 = t2088 * t2095;
          float t2098 = t2090 * t2096;
          float t2099 = t2097 - t2098;
          float t2100 = t2088 * t2096;
          float t2101 = t2090 * t2095;
          float t2102 = t2100 + t2101;
          float t2103 = t2093 + t2099;
          scratch_0[t2091] = t2103;
          float t2105 = t2094 + t2102;
          scratch_1[t2091] = t2105;
          float t2107 = t2093 - t2099;
          scratch_0[t2092] = t2107;
          float t2109 = t2094 - t2102;
          scratch_1[t2092] = t2109;
        }
        for (uint t2112 = 0; t2112 < 512; t2112++) {
          float t2113 = (float)t2112;
          float t2114 = (t2113 * 0.03125);
          float t2115 = metal::floor(t2114);
          float t2116 = t2115 * 32.0;
          float t2117 = t2113 - t2116;
          float t2118 = t2115 * 64.0;
          float t2119 = t2118 + t2117;
          float t2120 = t2119 + 32.0;
          int t2121 = (int)t2117;
          int t2122 = 31 + t2121;
          float t2123 = scratch_2[t2122];
          float t2124 = scratch_3[t2122];
          float t2125 = 0.0 - t2124;
          int t2126 = (int)t2119;
          int t2127 = (int)t2120;
          float t2128 = scratch_0[t2126];
          float t2129 = scratch_1[t2126];
          float t2130 = scratch_0[t2127];
          float t2131 = scratch_1[t2127];
          float t2132 = t2123 * t2130;
          float t2133 = t2125 * t2131;
          float t2134 = t2132 - t2133;
          float t2135 = t2123 * t2131;
          float t2136 = t2125 * t2130;
          float t2137 = t2135 + t2136;
          float t2138 = t2128 + t2134;
          scratch_0[t2126] = t2138;
          float t2140 = t2129 + t2137;
          scratch_1[t2126] = t2140;
          float t2142 = t2128 - t2134;
          scratch_0[t2127] = t2142;
          float t2144 = t2129 - t2137;
          scratch_1[t2127] = t2144;
        }
        for (uint t2147 = 0; t2147 < 512; t2147++) {
          float t2148 = (float)t2147;
          float t2149 = (t2148 * 0.015625);
          float t2150 = metal::floor(t2149);
          float t2151 = t2150 * 64.0;
          float t2152 = t2148 - t2151;
          float t2153 = t2150 * 128.0;
          float t2154 = t2153 + t2152;
          float t2155 = t2154 + 64.0;
          int t2156 = (int)t2152;
          int t2157 = 63 + t2156;
          float t2158 = scratch_2[t2157];
          float t2159 = scratch_3[t2157];
          float t2160 = 0.0 - t2159;
          int t2161 = (int)t2154;
          int t2162 = (int)t2155;
          float t2163 = scratch_0[t2161];
          float t2164 = scratch_1[t2161];
          float t2165 = scratch_0[t2162];
          float t2166 = scratch_1[t2162];
          float t2167 = t2158 * t2165;
          float t2168 = t2160 * t2166;
          float t2169 = t2167 - t2168;
          float t2170 = t2158 * t2166;
          float t2171 = t2160 * t2165;
          float t2172 = t2170 + t2171;
          float t2173 = t2163 + t2169;
          scratch_0[t2161] = t2173;
          float t2175 = t2164 + t2172;
          scratch_1[t2161] = t2175;
          float t2177 = t2163 - t2169;
          scratch_0[t2162] = t2177;
          float t2179 = t2164 - t2172;
          scratch_1[t2162] = t2179;
        }
        for (uint t2182 = 0; t2182 < 512; t2182++) {
          float t2183 = (float)t2182;
          float t2184 = (t2183 * 0.0078125);
          float t2185 = metal::floor(t2184);
          float t2186 = t2185 * 128.0;
          float t2187 = t2183 - t2186;
          float t2188 = t2185 * 256.0;
          float t2189 = t2188 + t2187;
          float t2190 = t2189 + 128.0;
          int t2191 = (int)t2187;
          int t2192 = 127 + t2191;
          float t2193 = scratch_2[t2192];
          float t2194 = scratch_3[t2192];
          float t2195 = 0.0 - t2194;
          int t2196 = (int)t2189;
          int t2197 = (int)t2190;
          float t2198 = scratch_0[t2196];
          float t2199 = scratch_1[t2196];
          float t2200 = scratch_0[t2197];
          float t2201 = scratch_1[t2197];
          float t2202 = t2193 * t2200;
          float t2203 = t2195 * t2201;
          float t2204 = t2202 - t2203;
          float t2205 = t2193 * t2201;
          float t2206 = t2195 * t2200;
          float t2207 = t2205 + t2206;
          float t2208 = t2198 + t2204;
          scratch_0[t2196] = t2208;
          float t2210 = t2199 + t2207;
          scratch_1[t2196] = t2210;
          float t2212 = t2198 - t2204;
          scratch_0[t2197] = t2212;
          float t2214 = t2199 - t2207;
          scratch_1[t2197] = t2214;
        }
        for (uint t2217 = 0; t2217 < 512; t2217++) {
          float t2218 = (float)t2217;
          float t2219 = (t2218 * 0.00390625);
          float t2220 = metal::floor(t2219);
          float t2221 = t2220 * 256.0;
          float t2222 = t2218 - t2221;
          float t2223 = t2220 * 512.0;
          float t2224 = t2223 + t2222;
          float t2225 = t2224 + 256.0;
          int t2226 = (int)t2222;
          int t2227 = 255 + t2226;
          float t2228 = scratch_2[t2227];
          float t2229 = scratch_3[t2227];
          float t2230 = 0.0 - t2229;
          int t2231 = (int)t2224;
          int t2232 = (int)t2225;
          float t2233 = scratch_0[t2231];
          float t2234 = scratch_1[t2231];
          float t2235 = scratch_0[t2232];
          float t2236 = scratch_1[t2232];
          float t2237 = t2228 * t2235;
          float t2238 = t2230 * t2236;
          float t2239 = t2237 - t2238;
          float t2240 = t2228 * t2236;
          float t2241 = t2230 * t2235;
          float t2242 = t2240 + t2241;
          float t2243 = t2233 + t2239;
          scratch_0[t2231] = t2243;
          float t2245 = t2234 + t2242;
          scratch_1[t2231] = t2245;
          float t2247 = t2233 - t2239;
          scratch_0[t2232] = t2247;
          float t2249 = t2234 - t2242;
          scratch_1[t2232] = t2249;
        }
        for (uint t2252 = 0; t2252 < 512; t2252++) {
          float t2253 = (float)t2252;
          float t2254 = (t2253 * 0.001953125);
          float t2255 = metal::floor(t2254);
          float t2256 = t2255 * 512.0;
          float t2257 = t2253 - t2256;
          float t2258 = t2255 * 1024.0;
          float t2259 = t2258 + t2257;
          float t2260 = t2259 + 512.0;
          int t2261 = (int)t2257;
          int t2262 = 511 + t2261;
          float t2263 = scratch_2[t2262];
          float t2264 = scratch_3[t2262];
          float t2265 = 0.0 - t2264;
          int t2266 = (int)t2259;
          int t2267 = (int)t2260;
          float t2268 = scratch_0[t2266];
          float t2269 = scratch_1[t2266];
          float t2270 = scratch_0[t2267];
          float t2271 = scratch_1[t2267];
          float t2272 = t2263 * t2270;
          float t2273 = t2265 * t2271;
          float t2274 = t2272 - t2273;
          float t2275 = t2263 * t2271;
          float t2276 = t2265 * t2270;
          float t2277 = t2275 + t2276;
          float t2278 = t2268 + t2274;
          scratch_0[t2266] = t2278;
          float t2280 = t2269 + t2277;
          scratch_1[t2266] = t2280;
          float t2282 = t2268 - t2274;
          scratch_0[t2267] = t2282;
          float t2284 = t2269 - t2277;
          scratch_1[t2267] = t2284;
        }
        for (uint t2287 = 0; t2287 < 1024; t2287++) {
          float t2288 = scratch_0[(int)t2287];
          float t2289 = scratch_1[(int)t2287];
          int t2290 = t1896 + t2287;
          memory[19335872 + t2290] = t2288;
          int t2292 = t1896 + t2287;
          int t2293 = t2292 + 1024;
          memory[19335872 + t2293] = t2289;
        }
        for (uint t2296 = 0; t2296 < 513; t2296++) {
          float t2297 = scratch_0[(int)t2296];
          float t2298 = scratch_1[(int)t2296];
          float t2299 = t2297 * t2297;
          float t2300 = t2298 * t2298;
          float t2301 = t2299 + t2300;
          float t2302 = metal::sqrt(t2301);
          int t2303 = t1897 + t2296;
          memory[19991488 + t2303] = t2302;
        }
        int t2306 = t1469 / 256;
        int t2307 = t2306 * 4;
        int t2308 = t2307 + t1479;
        int t2309 = t2308 * 513;
        for (uint t2310 = 0; t2310 < 513; t2310++) {
          int t2311 = t2309 + t2310;
          float t2312 = memory[19860160 + t2311];
          int t2313 = t2309 + t2310;
          float t2314 = memory[19991488 + t2313];
          float t2315 = t2312 - t2314;
          float t2316 = t2315 * t2315;
          float t2317 = t1471 + t2316;
          t1471 = t2317;
        }
      }
    }
    t[7*frameCount + id] = (t1471 * 0.25);
  }
  #pragma clang diagnostic pop
}



// KERNEL 24
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_24(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2326), value: global(2326)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2321) - handled in variable access */
    /* loadGlobal(1443) - handled in variable access */
    float t2322 = (t[7*frameCount + id] * 6.1035156e-05);
    float t2323 = t[5*frameCount + id] + t2322;
    float t2324 = t2323 * 0.5;
    float t2325 = t2324;
    t[8*frameCount + id] = t2325;
    float t2327 = t2324;
    float t2328 = t2323;
    float t2329 = (t[7*frameCount + id] * 3.7252903e-09);
    float t2330 = -0.5 * t2329;
  }
  #pragma clang diagnostic pop
}



// KERNEL 25
// FrameOrder: parallel
// DispatchMode: perFrame
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
    /* loadGlobal(1452) - handled in variable access */
    int t2331 = id;
    float t2332 = t[6*frameCount + id] == 0.0;
    if (t2332) {
      int t2334 = t2331 / 256;
      for (uint t2335 = 0; t2335 < 4; t2335++) {
        int t2336 = t2334 * 4;
        int t2337 = t2336 + t2335;
        int t2338 = t2337 * 2048;
        int t2339 = t2337 * 513;
        int t2340 = t2337 * 2048;
        for (uint _pr2341 = 0; _pr2341 < 513; _pr2341++) {
          int t2342 = t2339 + _pr2341;
          float t2343 = memory[19860160 + t2342];
          int t2344 = t2339 + _pr2341;
          float t2345 = memory[19991488 + t2344];
          int t2346 = t2338 + _pr2341;
          float t2347 = memory[18811584 + t2346];
          int t2348 = t2338 + _pr2341;
          int t2349 = t2348 + 1024;
          float t2350 = memory[18811584 + t2349];
          int t2351 = t2338 + _pr2341;
          float t2352 = memory[19335872 + t2351];
          int t2353 = t2338 + _pr2341;
          int t2354 = t2353 + 1024;
          float t2355 = memory[19335872 + t2354];
          float t2356 = t2343 - t2345;
          float t2357 = 2.0 * t2356;
          float t2358 = t2357 * 7.6293945e-06;
          float t2359 = t2343 - t2345;
          float t2360 = -2.0 * t2359;
          float t2361 = t2360 * 7.6293945e-06;
          float t2362 = metal::max(t2343, 1e-08);
          float t2363 = metal::max(t2345, 1e-08);
          float t2364 = t2358 * t2347;
          float t2365 = t2364 / t2362;
          float t2366 = t2358 * t2350;
          float t2367 = t2366 / t2362;
          float t2368 = t2361 * t2352;
          float t2369 = t2368 / t2363;
          float t2370 = t2361 * t2355;
          float t2371 = t2370 / t2363;
          int t2372 = t2340 + _pr2341;
          memory[9045696 + t2372] = t2365;
          int t2374 = t2340 + _pr2341;
          int t2375 = t2374 + 1024;
          memory[9045696 + t2375] = t2367;
          int t2377 = t2340 + _pr2341;
          memory[20122816 + t2377] = t2369;
          int t2379 = t2340 + _pr2341;
          int t2380 = t2379 + 1024;
          memory[20122816 + t2380] = t2371;
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
        for (uint _pr2383 = 0; _pr2383 < 511; _pr2383++) {
          int t2384 = _pr2383 + 513;
          int t2385 = t2340 + t2384;
          memory[9045696 + t2385] = 0.0;
          int t2387 = t2340 + t2384;
          int t2388 = t2387 + 1024;
          memory[9045696 + t2388] = 0.0;
          int t2390 = t2340 + t2384;
          memory[20122816 + t2390] = 0.0;
          int t2392 = t2340 + t2384;
          int t2393 = t2392 + 1024;
          memory[20122816 + t2393] = 0.0;
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 26
// FrameOrder: parallel
// DispatchMode: perFrameThreadgroup1
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
    /* loadGlobal(1452) - handled in variable access */
    int t2398 = id;
    threadgroup float scratch_0[1024];
    threadgroup float scratch_1[1024];
    threadgroup float scratch_2[1023];
    threadgroup float scratch_3[1023];
    float t2399 = t[6*frameCount + id] == 0.0;
    if (t2399) {
      for (uint t2401 = 0; t2401 < 1023; t2401++) {
        float t2402 = memory[95680 + (int)t2401];
        scratch_2[(int)t2401] = t2402;
        float t2404 = memory[96703 + (int)t2401];
        scratch_3[(int)t2401] = t2404;
      }
      for (uint t2407 = 0; t2407 < 4; t2407++) {
        int t2408 = t2398 / 256;
        int t2409 = t2408 * 4;
        int t2410 = t2409 + t2407;
        int t2411 = t2410 * 2048;
        int t2412 = t2410 * 1024;
        for (uint t2413 = 0; t2413 < 1024; t2413++) {
          int t2414 = t2411 + t2413;
          float t2415 = memory[9045696 + t2414];
          int t2416 = t2411 + t2413;
          int t2417 = t2416 + 1024;
          float t2418 = memory[9045696 + t2417];
          scratch_0[(int)t2413] = t2415;
          scratch_1[(int)t2413] = t2418;
        }
        for (uint t2422 = 0; t2422 < 1024; t2422++) {
          float t2423 = memory[97726 + (int)t2422];
          float t2424 = (float)t2422;
          float t2425 = t2424 < t2423;
          int t2426 = (int)t2423;
          float t2427 = scratch_0[(int)t2422];
          float t2428 = scratch_1[(int)t2422];
          float t2429 = scratch_0[t2426];
          float t2430 = scratch_1[t2426];
          float t2431 = metal::select(t2427, t2429, t2425 > 0.0);
          float t2432 = metal::select(t2428, t2430, t2425 > 0.0);
          float t2433 = metal::select(t2429, t2427, t2425 > 0.0);
          float t2434 = metal::select(t2430, t2428, t2425 > 0.0);
          scratch_0[(int)t2422] = t2431;
          scratch_1[(int)t2422] = t2432;
          scratch_0[t2426] = t2433;
          scratch_1[t2426] = t2434;
        }
        for (uint t2440 = 0; t2440 < 512; t2440++) {
          float t2441 = (float)t2440;
          float t2442 = t2441;
          float t2443 = metal::floor(t2442);
          float t2444 = t2443;
          float t2445 = t2441 - t2444;
          float t2446 = t2443 * 2.0;
          float t2447 = t2446 + t2445;
          float t2448 = t2447 + 1.0;
          int t2449 = (int)t2445;
          int t2450 = t2449;
          float t2451 = scratch_2[t2450];
          float t2452 = scratch_3[t2450];
          int t2453 = (int)t2447;
          int t2454 = (int)t2448;
          float t2455 = scratch_0[t2453];
          float t2456 = scratch_1[t2453];
          float t2457 = scratch_0[t2454];
          float t2458 = scratch_1[t2454];
          float t2459 = t2451 * t2457;
          float t2460 = t2452 * t2458;
          float t2461 = t2459 - t2460;
          float t2462 = t2451 * t2458;
          float t2463 = t2452 * t2457;
          float t2464 = t2462 + t2463;
          float t2465 = t2455 + t2461;
          scratch_0[t2453] = t2465;
          float t2467 = t2456 + t2464;
          scratch_1[t2453] = t2467;
          float t2469 = t2455 - t2461;
          scratch_0[t2454] = t2469;
          float t2471 = t2456 - t2464;
          scratch_1[t2454] = t2471;
        }
        for (uint t2474 = 0; t2474 < 512; t2474++) {
          float t2475 = (float)t2474;
          float t2476 = (t2475 * 0.5);
          float t2477 = metal::floor(t2476);
          float t2478 = t2477 * 2.0;
          float t2479 = t2475 - t2478;
          float t2480 = t2477 * 4.0;
          float t2481 = t2480 + t2479;
          float t2482 = t2481 + 2.0;
          int t2483 = (int)t2479;
          int t2484 = 1 + t2483;
          float t2485 = scratch_2[t2484];
          float t2486 = scratch_3[t2484];
          int t2487 = (int)t2481;
          int t2488 = (int)t2482;
          float t2489 = scratch_0[t2487];
          float t2490 = scratch_1[t2487];
          float t2491 = scratch_0[t2488];
          float t2492 = scratch_1[t2488];
          float t2493 = t2485 * t2491;
          float t2494 = t2486 * t2492;
          float t2495 = t2493 - t2494;
          float t2496 = t2485 * t2492;
          float t2497 = t2486 * t2491;
          float t2498 = t2496 + t2497;
          float t2499 = t2489 + t2495;
          scratch_0[t2487] = t2499;
          float t2501 = t2490 + t2498;
          scratch_1[t2487] = t2501;
          float t2503 = t2489 - t2495;
          scratch_0[t2488] = t2503;
          float t2505 = t2490 - t2498;
          scratch_1[t2488] = t2505;
        }
        for (uint t2508 = 0; t2508 < 512; t2508++) {
          float t2509 = (float)t2508;
          float t2510 = (t2509 * 0.25);
          float t2511 = metal::floor(t2510);
          float t2512 = t2511 * 4.0;
          float t2513 = t2509 - t2512;
          float t2514 = t2511 * 8.0;
          float t2515 = t2514 + t2513;
          float t2516 = t2515 + 4.0;
          int t2517 = (int)t2513;
          int t2518 = 3 + t2517;
          float t2519 = scratch_2[t2518];
          float t2520 = scratch_3[t2518];
          int t2521 = (int)t2515;
          int t2522 = (int)t2516;
          float t2523 = scratch_0[t2521];
          float t2524 = scratch_1[t2521];
          float t2525 = scratch_0[t2522];
          float t2526 = scratch_1[t2522];
          float t2527 = t2519 * t2525;
          float t2528 = t2520 * t2526;
          float t2529 = t2527 - t2528;
          float t2530 = t2519 * t2526;
          float t2531 = t2520 * t2525;
          float t2532 = t2530 + t2531;
          float t2533 = t2523 + t2529;
          scratch_0[t2521] = t2533;
          float t2535 = t2524 + t2532;
          scratch_1[t2521] = t2535;
          float t2537 = t2523 - t2529;
          scratch_0[t2522] = t2537;
          float t2539 = t2524 - t2532;
          scratch_1[t2522] = t2539;
        }
        for (uint t2542 = 0; t2542 < 512; t2542++) {
          float t2543 = (float)t2542;
          float t2544 = (t2543 * 0.125);
          float t2545 = metal::floor(t2544);
          float t2546 = t2545 * 8.0;
          float t2547 = t2543 - t2546;
          float t2548 = t2545 * 16.0;
          float t2549 = t2548 + t2547;
          float t2550 = t2549 + 8.0;
          int t2551 = (int)t2547;
          int t2552 = 7 + t2551;
          float t2553 = scratch_2[t2552];
          float t2554 = scratch_3[t2552];
          int t2555 = (int)t2549;
          int t2556 = (int)t2550;
          float t2557 = scratch_0[t2555];
          float t2558 = scratch_1[t2555];
          float t2559 = scratch_0[t2556];
          float t2560 = scratch_1[t2556];
          float t2561 = t2553 * t2559;
          float t2562 = t2554 * t2560;
          float t2563 = t2561 - t2562;
          float t2564 = t2553 * t2560;
          float t2565 = t2554 * t2559;
          float t2566 = t2564 + t2565;
          float t2567 = t2557 + t2563;
          scratch_0[t2555] = t2567;
          float t2569 = t2558 + t2566;
          scratch_1[t2555] = t2569;
          float t2571 = t2557 - t2563;
          scratch_0[t2556] = t2571;
          float t2573 = t2558 - t2566;
          scratch_1[t2556] = t2573;
        }
        for (uint t2576 = 0; t2576 < 512; t2576++) {
          float t2577 = (float)t2576;
          float t2578 = (t2577 * 0.0625);
          float t2579 = metal::floor(t2578);
          float t2580 = t2579 * 16.0;
          float t2581 = t2577 - t2580;
          float t2582 = t2579 * 32.0;
          float t2583 = t2582 + t2581;
          float t2584 = t2583 + 16.0;
          int t2585 = (int)t2581;
          int t2586 = 15 + t2585;
          float t2587 = scratch_2[t2586];
          float t2588 = scratch_3[t2586];
          int t2589 = (int)t2583;
          int t2590 = (int)t2584;
          float t2591 = scratch_0[t2589];
          float t2592 = scratch_1[t2589];
          float t2593 = scratch_0[t2590];
          float t2594 = scratch_1[t2590];
          float t2595 = t2587 * t2593;
          float t2596 = t2588 * t2594;
          float t2597 = t2595 - t2596;
          float t2598 = t2587 * t2594;
          float t2599 = t2588 * t2593;
          float t2600 = t2598 + t2599;
          float t2601 = t2591 + t2597;
          scratch_0[t2589] = t2601;
          float t2603 = t2592 + t2600;
          scratch_1[t2589] = t2603;
          float t2605 = t2591 - t2597;
          scratch_0[t2590] = t2605;
          float t2607 = t2592 - t2600;
          scratch_1[t2590] = t2607;
        }
        for (uint t2610 = 0; t2610 < 512; t2610++) {
          float t2611 = (float)t2610;
          float t2612 = (t2611 * 0.03125);
          float t2613 = metal::floor(t2612);
          float t2614 = t2613 * 32.0;
          float t2615 = t2611 - t2614;
          float t2616 = t2613 * 64.0;
          float t2617 = t2616 + t2615;
          float t2618 = t2617 + 32.0;
          int t2619 = (int)t2615;
          int t2620 = 31 + t2619;
          float t2621 = scratch_2[t2620];
          float t2622 = scratch_3[t2620];
          int t2623 = (int)t2617;
          int t2624 = (int)t2618;
          float t2625 = scratch_0[t2623];
          float t2626 = scratch_1[t2623];
          float t2627 = scratch_0[t2624];
          float t2628 = scratch_1[t2624];
          float t2629 = t2621 * t2627;
          float t2630 = t2622 * t2628;
          float t2631 = t2629 - t2630;
          float t2632 = t2621 * t2628;
          float t2633 = t2622 * t2627;
          float t2634 = t2632 + t2633;
          float t2635 = t2625 + t2631;
          scratch_0[t2623] = t2635;
          float t2637 = t2626 + t2634;
          scratch_1[t2623] = t2637;
          float t2639 = t2625 - t2631;
          scratch_0[t2624] = t2639;
          float t2641 = t2626 - t2634;
          scratch_1[t2624] = t2641;
        }
        for (uint t2644 = 0; t2644 < 512; t2644++) {
          float t2645 = (float)t2644;
          float t2646 = (t2645 * 0.015625);
          float t2647 = metal::floor(t2646);
          float t2648 = t2647 * 64.0;
          float t2649 = t2645 - t2648;
          float t2650 = t2647 * 128.0;
          float t2651 = t2650 + t2649;
          float t2652 = t2651 + 64.0;
          int t2653 = (int)t2649;
          int t2654 = 63 + t2653;
          float t2655 = scratch_2[t2654];
          float t2656 = scratch_3[t2654];
          int t2657 = (int)t2651;
          int t2658 = (int)t2652;
          float t2659 = scratch_0[t2657];
          float t2660 = scratch_1[t2657];
          float t2661 = scratch_0[t2658];
          float t2662 = scratch_1[t2658];
          float t2663 = t2655 * t2661;
          float t2664 = t2656 * t2662;
          float t2665 = t2663 - t2664;
          float t2666 = t2655 * t2662;
          float t2667 = t2656 * t2661;
          float t2668 = t2666 + t2667;
          float t2669 = t2659 + t2665;
          scratch_0[t2657] = t2669;
          float t2671 = t2660 + t2668;
          scratch_1[t2657] = t2671;
          float t2673 = t2659 - t2665;
          scratch_0[t2658] = t2673;
          float t2675 = t2660 - t2668;
          scratch_1[t2658] = t2675;
        }
        for (uint t2678 = 0; t2678 < 512; t2678++) {
          float t2679 = (float)t2678;
          float t2680 = (t2679 * 0.0078125);
          float t2681 = metal::floor(t2680);
          float t2682 = t2681 * 128.0;
          float t2683 = t2679 - t2682;
          float t2684 = t2681 * 256.0;
          float t2685 = t2684 + t2683;
          float t2686 = t2685 + 128.0;
          int t2687 = (int)t2683;
          int t2688 = 127 + t2687;
          float t2689 = scratch_2[t2688];
          float t2690 = scratch_3[t2688];
          int t2691 = (int)t2685;
          int t2692 = (int)t2686;
          float t2693 = scratch_0[t2691];
          float t2694 = scratch_1[t2691];
          float t2695 = scratch_0[t2692];
          float t2696 = scratch_1[t2692];
          float t2697 = t2689 * t2695;
          float t2698 = t2690 * t2696;
          float t2699 = t2697 - t2698;
          float t2700 = t2689 * t2696;
          float t2701 = t2690 * t2695;
          float t2702 = t2700 + t2701;
          float t2703 = t2693 + t2699;
          scratch_0[t2691] = t2703;
          float t2705 = t2694 + t2702;
          scratch_1[t2691] = t2705;
          float t2707 = t2693 - t2699;
          scratch_0[t2692] = t2707;
          float t2709 = t2694 - t2702;
          scratch_1[t2692] = t2709;
        }
        for (uint t2712 = 0; t2712 < 512; t2712++) {
          float t2713 = (float)t2712;
          float t2714 = (t2713 * 0.00390625);
          float t2715 = metal::floor(t2714);
          float t2716 = t2715 * 256.0;
          float t2717 = t2713 - t2716;
          float t2718 = t2715 * 512.0;
          float t2719 = t2718 + t2717;
          float t2720 = t2719 + 256.0;
          int t2721 = (int)t2717;
          int t2722 = 255 + t2721;
          float t2723 = scratch_2[t2722];
          float t2724 = scratch_3[t2722];
          int t2725 = (int)t2719;
          int t2726 = (int)t2720;
          float t2727 = scratch_0[t2725];
          float t2728 = scratch_1[t2725];
          float t2729 = scratch_0[t2726];
          float t2730 = scratch_1[t2726];
          float t2731 = t2723 * t2729;
          float t2732 = t2724 * t2730;
          float t2733 = t2731 - t2732;
          float t2734 = t2723 * t2730;
          float t2735 = t2724 * t2729;
          float t2736 = t2734 + t2735;
          float t2737 = t2727 + t2733;
          scratch_0[t2725] = t2737;
          float t2739 = t2728 + t2736;
          scratch_1[t2725] = t2739;
          float t2741 = t2727 - t2733;
          scratch_0[t2726] = t2741;
          float t2743 = t2728 - t2736;
          scratch_1[t2726] = t2743;
        }
        for (uint t2746 = 0; t2746 < 512; t2746++) {
          float t2747 = (float)t2746;
          float t2748 = (t2747 * 0.001953125);
          float t2749 = metal::floor(t2748);
          float t2750 = t2749 * 512.0;
          float t2751 = t2747 - t2750;
          float t2752 = t2749 * 1024.0;
          float t2753 = t2752 + t2751;
          float t2754 = t2753 + 512.0;
          int t2755 = (int)t2751;
          int t2756 = 511 + t2755;
          float t2757 = scratch_2[t2756];
          float t2758 = scratch_3[t2756];
          int t2759 = (int)t2753;
          int t2760 = (int)t2754;
          float t2761 = scratch_0[t2759];
          float t2762 = scratch_1[t2759];
          float t2763 = scratch_0[t2760];
          float t2764 = scratch_1[t2760];
          float t2765 = t2757 * t2763;
          float t2766 = t2758 * t2764;
          float t2767 = t2765 - t2766;
          float t2768 = t2757 * t2764;
          float t2769 = t2758 * t2763;
          float t2770 = t2768 + t2769;
          float t2771 = t2761 + t2767;
          scratch_0[t2759] = t2771;
          float t2773 = t2762 + t2770;
          scratch_1[t2759] = t2773;
          float t2775 = t2761 - t2767;
          scratch_0[t2760] = t2775;
          float t2777 = t2762 - t2770;
          scratch_1[t2760] = t2777;
        }
        for (uint t2780 = 0; t2780 < 1024; t2780++) {
          float t2781 = scratch_0[(int)t2780];
          float t2782 = t2781 * 1.9036306e-06;
          float t2783 = memory[94656 + (int)t2780];
          int t2784 = t2412 + t2780;
          float t2785 = t2782 * t2783;
          memory[18811584 + t2784] = t2785;
        }
        int t2788 = t2398 / 256;
        int t2789 = t2788 * 4;
        int t2790 = t2789 + t2407;
        int t2791 = t2790 * 2048;
        int t2792 = t2790 * 1024;
        for (uint t2793 = 0; t2793 < 1024; t2793++) {
          int t2794 = t2791 + t2793;
          float t2795 = memory[20122816 + t2794];
          int t2796 = t2791 + t2793;
          int t2797 = t2796 + 1024;
          float t2798 = memory[20122816 + t2797];
          scratch_0[(int)t2793] = t2795;
          scratch_1[(int)t2793] = t2798;
        }
        for (uint t2802 = 0; t2802 < 1024; t2802++) {
          float t2803 = memory[97726 + (int)t2802];
          float t2804 = (float)t2802;
          float t2805 = t2804 < t2803;
          int t2806 = (int)t2803;
          float t2807 = scratch_0[(int)t2802];
          float t2808 = scratch_1[(int)t2802];
          float t2809 = scratch_0[t2806];
          float t2810 = scratch_1[t2806];
          float t2811 = metal::select(t2807, t2809, t2805 > 0.0);
          float t2812 = metal::select(t2808, t2810, t2805 > 0.0);
          float t2813 = metal::select(t2809, t2807, t2805 > 0.0);
          float t2814 = metal::select(t2810, t2808, t2805 > 0.0);
          scratch_0[(int)t2802] = t2811;
          scratch_1[(int)t2802] = t2812;
          scratch_0[t2806] = t2813;
          scratch_1[t2806] = t2814;
        }
        for (uint t2820 = 0; t2820 < 512; t2820++) {
          float t2821 = (float)t2820;
          float t2822 = t2821;
          float t2823 = metal::floor(t2822);
          float t2824 = t2823;
          float t2825 = t2821 - t2824;
          float t2826 = t2823 * 2.0;
          float t2827 = t2826 + t2825;
          float t2828 = t2827 + 1.0;
          int t2829 = (int)t2825;
          int t2830 = t2829;
          float t2831 = scratch_2[t2830];
          float t2832 = scratch_3[t2830];
          int t2833 = (int)t2827;
          int t2834 = (int)t2828;
          float t2835 = scratch_0[t2833];
          float t2836 = scratch_1[t2833];
          float t2837 = scratch_0[t2834];
          float t2838 = scratch_1[t2834];
          float t2839 = t2831 * t2837;
          float t2840 = t2832 * t2838;
          float t2841 = t2839 - t2840;
          float t2842 = t2831 * t2838;
          float t2843 = t2832 * t2837;
          float t2844 = t2842 + t2843;
          float t2845 = t2835 + t2841;
          scratch_0[t2833] = t2845;
          float t2847 = t2836 + t2844;
          scratch_1[t2833] = t2847;
          float t2849 = t2835 - t2841;
          scratch_0[t2834] = t2849;
          float t2851 = t2836 - t2844;
          scratch_1[t2834] = t2851;
        }
        for (uint t2854 = 0; t2854 < 512; t2854++) {
          float t2855 = (float)t2854;
          float t2856 = (t2855 * 0.5);
          float t2857 = metal::floor(t2856);
          float t2858 = t2857 * 2.0;
          float t2859 = t2855 - t2858;
          float t2860 = t2857 * 4.0;
          float t2861 = t2860 + t2859;
          float t2862 = t2861 + 2.0;
          int t2863 = (int)t2859;
          int t2864 = 1 + t2863;
          float t2865 = scratch_2[t2864];
          float t2866 = scratch_3[t2864];
          int t2867 = (int)t2861;
          int t2868 = (int)t2862;
          float t2869 = scratch_0[t2867];
          float t2870 = scratch_1[t2867];
          float t2871 = scratch_0[t2868];
          float t2872 = scratch_1[t2868];
          float t2873 = t2865 * t2871;
          float t2874 = t2866 * t2872;
          float t2875 = t2873 - t2874;
          float t2876 = t2865 * t2872;
          float t2877 = t2866 * t2871;
          float t2878 = t2876 + t2877;
          float t2879 = t2869 + t2875;
          scratch_0[t2867] = t2879;
          float t2881 = t2870 + t2878;
          scratch_1[t2867] = t2881;
          float t2883 = t2869 - t2875;
          scratch_0[t2868] = t2883;
          float t2885 = t2870 - t2878;
          scratch_1[t2868] = t2885;
        }
        for (uint t2888 = 0; t2888 < 512; t2888++) {
          float t2889 = (float)t2888;
          float t2890 = (t2889 * 0.25);
          float t2891 = metal::floor(t2890);
          float t2892 = t2891 * 4.0;
          float t2893 = t2889 - t2892;
          float t2894 = t2891 * 8.0;
          float t2895 = t2894 + t2893;
          float t2896 = t2895 + 4.0;
          int t2897 = (int)t2893;
          int t2898 = 3 + t2897;
          float t2899 = scratch_2[t2898];
          float t2900 = scratch_3[t2898];
          int t2901 = (int)t2895;
          int t2902 = (int)t2896;
          float t2903 = scratch_0[t2901];
          float t2904 = scratch_1[t2901];
          float t2905 = scratch_0[t2902];
          float t2906 = scratch_1[t2902];
          float t2907 = t2899 * t2905;
          float t2908 = t2900 * t2906;
          float t2909 = t2907 - t2908;
          float t2910 = t2899 * t2906;
          float t2911 = t2900 * t2905;
          float t2912 = t2910 + t2911;
          float t2913 = t2903 + t2909;
          scratch_0[t2901] = t2913;
          float t2915 = t2904 + t2912;
          scratch_1[t2901] = t2915;
          float t2917 = t2903 - t2909;
          scratch_0[t2902] = t2917;
          float t2919 = t2904 - t2912;
          scratch_1[t2902] = t2919;
        }
        for (uint t2922 = 0; t2922 < 512; t2922++) {
          float t2923 = (float)t2922;
          float t2924 = (t2923 * 0.125);
          float t2925 = metal::floor(t2924);
          float t2926 = t2925 * 8.0;
          float t2927 = t2923 - t2926;
          float t2928 = t2925 * 16.0;
          float t2929 = t2928 + t2927;
          float t2930 = t2929 + 8.0;
          int t2931 = (int)t2927;
          int t2932 = 7 + t2931;
          float t2933 = scratch_2[t2932];
          float t2934 = scratch_3[t2932];
          int t2935 = (int)t2929;
          int t2936 = (int)t2930;
          float t2937 = scratch_0[t2935];
          float t2938 = scratch_1[t2935];
          float t2939 = scratch_0[t2936];
          float t2940 = scratch_1[t2936];
          float t2941 = t2933 * t2939;
          float t2942 = t2934 * t2940;
          float t2943 = t2941 - t2942;
          float t2944 = t2933 * t2940;
          float t2945 = t2934 * t2939;
          float t2946 = t2944 + t2945;
          float t2947 = t2937 + t2943;
          scratch_0[t2935] = t2947;
          float t2949 = t2938 + t2946;
          scratch_1[t2935] = t2949;
          float t2951 = t2937 - t2943;
          scratch_0[t2936] = t2951;
          float t2953 = t2938 - t2946;
          scratch_1[t2936] = t2953;
        }
        for (uint t2956 = 0; t2956 < 512; t2956++) {
          float t2957 = (float)t2956;
          float t2958 = (t2957 * 0.0625);
          float t2959 = metal::floor(t2958);
          float t2960 = t2959 * 16.0;
          float t2961 = t2957 - t2960;
          float t2962 = t2959 * 32.0;
          float t2963 = t2962 + t2961;
          float t2964 = t2963 + 16.0;
          int t2965 = (int)t2961;
          int t2966 = 15 + t2965;
          float t2967 = scratch_2[t2966];
          float t2968 = scratch_3[t2966];
          int t2969 = (int)t2963;
          int t2970 = (int)t2964;
          float t2971 = scratch_0[t2969];
          float t2972 = scratch_1[t2969];
          float t2973 = scratch_0[t2970];
          float t2974 = scratch_1[t2970];
          float t2975 = t2967 * t2973;
          float t2976 = t2968 * t2974;
          float t2977 = t2975 - t2976;
          float t2978 = t2967 * t2974;
          float t2979 = t2968 * t2973;
          float t2980 = t2978 + t2979;
          float t2981 = t2971 + t2977;
          scratch_0[t2969] = t2981;
          float t2983 = t2972 + t2980;
          scratch_1[t2969] = t2983;
          float t2985 = t2971 - t2977;
          scratch_0[t2970] = t2985;
          float t2987 = t2972 - t2980;
          scratch_1[t2970] = t2987;
        }
        for (uint t2990 = 0; t2990 < 512; t2990++) {
          float t2991 = (float)t2990;
          float t2992 = (t2991 * 0.03125);
          float t2993 = metal::floor(t2992);
          float t2994 = t2993 * 32.0;
          float t2995 = t2991 - t2994;
          float t2996 = t2993 * 64.0;
          float t2997 = t2996 + t2995;
          float t2998 = t2997 + 32.0;
          int t2999 = (int)t2995;
          int t3000 = 31 + t2999;
          float t3001 = scratch_2[t3000];
          float t3002 = scratch_3[t3000];
          int t3003 = (int)t2997;
          int t3004 = (int)t2998;
          float t3005 = scratch_0[t3003];
          float t3006 = scratch_1[t3003];
          float t3007 = scratch_0[t3004];
          float t3008 = scratch_1[t3004];
          float t3009 = t3001 * t3007;
          float t3010 = t3002 * t3008;
          float t3011 = t3009 - t3010;
          float t3012 = t3001 * t3008;
          float t3013 = t3002 * t3007;
          float t3014 = t3012 + t3013;
          float t3015 = t3005 + t3011;
          scratch_0[t3003] = t3015;
          float t3017 = t3006 + t3014;
          scratch_1[t3003] = t3017;
          float t3019 = t3005 - t3011;
          scratch_0[t3004] = t3019;
          float t3021 = t3006 - t3014;
          scratch_1[t3004] = t3021;
        }
        for (uint t3024 = 0; t3024 < 512; t3024++) {
          float t3025 = (float)t3024;
          float t3026 = (t3025 * 0.015625);
          float t3027 = metal::floor(t3026);
          float t3028 = t3027 * 64.0;
          float t3029 = t3025 - t3028;
          float t3030 = t3027 * 128.0;
          float t3031 = t3030 + t3029;
          float t3032 = t3031 + 64.0;
          int t3033 = (int)t3029;
          int t3034 = 63 + t3033;
          float t3035 = scratch_2[t3034];
          float t3036 = scratch_3[t3034];
          int t3037 = (int)t3031;
          int t3038 = (int)t3032;
          float t3039 = scratch_0[t3037];
          float t3040 = scratch_1[t3037];
          float t3041 = scratch_0[t3038];
          float t3042 = scratch_1[t3038];
          float t3043 = t3035 * t3041;
          float t3044 = t3036 * t3042;
          float t3045 = t3043 - t3044;
          float t3046 = t3035 * t3042;
          float t3047 = t3036 * t3041;
          float t3048 = t3046 + t3047;
          float t3049 = t3039 + t3045;
          scratch_0[t3037] = t3049;
          float t3051 = t3040 + t3048;
          scratch_1[t3037] = t3051;
          float t3053 = t3039 - t3045;
          scratch_0[t3038] = t3053;
          float t3055 = t3040 - t3048;
          scratch_1[t3038] = t3055;
        }
        for (uint t3058 = 0; t3058 < 512; t3058++) {
          float t3059 = (float)t3058;
          float t3060 = (t3059 * 0.0078125);
          float t3061 = metal::floor(t3060);
          float t3062 = t3061 * 128.0;
          float t3063 = t3059 - t3062;
          float t3064 = t3061 * 256.0;
          float t3065 = t3064 + t3063;
          float t3066 = t3065 + 128.0;
          int t3067 = (int)t3063;
          int t3068 = 127 + t3067;
          float t3069 = scratch_2[t3068];
          float t3070 = scratch_3[t3068];
          int t3071 = (int)t3065;
          int t3072 = (int)t3066;
          float t3073 = scratch_0[t3071];
          float t3074 = scratch_1[t3071];
          float t3075 = scratch_0[t3072];
          float t3076 = scratch_1[t3072];
          float t3077 = t3069 * t3075;
          float t3078 = t3070 * t3076;
          float t3079 = t3077 - t3078;
          float t3080 = t3069 * t3076;
          float t3081 = t3070 * t3075;
          float t3082 = t3080 + t3081;
          float t3083 = t3073 + t3079;
          scratch_0[t3071] = t3083;
          float t3085 = t3074 + t3082;
          scratch_1[t3071] = t3085;
          float t3087 = t3073 - t3079;
          scratch_0[t3072] = t3087;
          float t3089 = t3074 - t3082;
          scratch_1[t3072] = t3089;
        }
        for (uint t3092 = 0; t3092 < 512; t3092++) {
          float t3093 = (float)t3092;
          float t3094 = (t3093 * 0.00390625);
          float t3095 = metal::floor(t3094);
          float t3096 = t3095 * 256.0;
          float t3097 = t3093 - t3096;
          float t3098 = t3095 * 512.0;
          float t3099 = t3098 + t3097;
          float t3100 = t3099 + 256.0;
          int t3101 = (int)t3097;
          int t3102 = 255 + t3101;
          float t3103 = scratch_2[t3102];
          float t3104 = scratch_3[t3102];
          int t3105 = (int)t3099;
          int t3106 = (int)t3100;
          float t3107 = scratch_0[t3105];
          float t3108 = scratch_1[t3105];
          float t3109 = scratch_0[t3106];
          float t3110 = scratch_1[t3106];
          float t3111 = t3103 * t3109;
          float t3112 = t3104 * t3110;
          float t3113 = t3111 - t3112;
          float t3114 = t3103 * t3110;
          float t3115 = t3104 * t3109;
          float t3116 = t3114 + t3115;
          float t3117 = t3107 + t3113;
          scratch_0[t3105] = t3117;
          float t3119 = t3108 + t3116;
          scratch_1[t3105] = t3119;
          float t3121 = t3107 - t3113;
          scratch_0[t3106] = t3121;
          float t3123 = t3108 - t3116;
          scratch_1[t3106] = t3123;
        }
        for (uint t3126 = 0; t3126 < 512; t3126++) {
          float t3127 = (float)t3126;
          float t3128 = (t3127 * 0.001953125);
          float t3129 = metal::floor(t3128);
          float t3130 = t3129 * 512.0;
          float t3131 = t3127 - t3130;
          float t3132 = t3129 * 1024.0;
          float t3133 = t3132 + t3131;
          float t3134 = t3133 + 512.0;
          int t3135 = (int)t3131;
          int t3136 = 511 + t3135;
          float t3137 = scratch_2[t3136];
          float t3138 = scratch_3[t3136];
          int t3139 = (int)t3133;
          int t3140 = (int)t3134;
          float t3141 = scratch_0[t3139];
          float t3142 = scratch_1[t3139];
          float t3143 = scratch_0[t3140];
          float t3144 = scratch_1[t3140];
          float t3145 = t3137 * t3143;
          float t3146 = t3138 * t3144;
          float t3147 = t3145 - t3146;
          float t3148 = t3137 * t3144;
          float t3149 = t3138 * t3143;
          float t3150 = t3148 + t3149;
          float t3151 = t3141 + t3147;
          scratch_0[t3139] = t3151;
          float t3153 = t3142 + t3150;
          scratch_1[t3139] = t3153;
          float t3155 = t3141 - t3147;
          scratch_0[t3140] = t3155;
          float t3157 = t3142 - t3150;
          scratch_1[t3140] = t3157;
        }
        for (uint t3160 = 0; t3160 < 1024; t3160++) {
          float t3161 = scratch_0[(int)t3160];
          float t3162 = t3161 * 1.9036306e-06;
          float t3163 = memory[94656 + (int)t3160];
          int t3164 = t2792 + t3160;
          float t3165 = t3162 * t3163;
          memory[19335872 + t3164] = t3165;
        }
      }
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 27
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_27(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1452) - handled in variable access */
    int t3170 = id;
    for (uint t3171 = 0; t3171 < 4; t3171++) {
      float t3172 = 0.0;
      int t3173 = (int)frameCount;
      int t3174 = t3173 + 256;
      int t3175 = t3174 - 1;
      int t3176 = t3175 / 256;
      int t3177 = t3176 - 1;
      int t3178 = t3170 + 1024;
      int t3179 = t3178 - 1;
      int t3180 = t3179 / 256;
      float t3181 = metal::min(t3180, t3177);
      int t3182 = t3176 * 4;
      int t3183 = t3182 * 1024;
      int t3184 = t3183 - 1;
      for (uint t3185 = 0; t3185 < 5; t3185++) {
        float t3186 = t3181 - t3185;
        float t3187 = t3186 * 256.0;
        float t3188 = t3186 >= 0.0;
        float t3189 = (float)t3170;
        float t3190 = t3187 >= t3189;
        float t3191 = (float)t3176;
        float t3192 = t3186 < t3191;
        float t3193 = t3188 * t3190;
        float t3194 = t3193 * t3192;
        float t3195 = t3170 - t3187;
        float t3196 = t3195 + 1024.0;
        float t3197 = t3196 - 1.0;
        float t3198 = t3186 * 4.0;
        float t3199 = t3198 + t3171;
        float t3200 = t3199 * 1024.0;
        float t3201 = t3200 + t3197;
        float t3202 = (float)t3184;
        float t3203 = metal::min(t3201, t3202);
        float t3204 = metal::max(0.0, t3203);
        int t3205 = (int)t3204;
        float t3206 = memory[18811584 + t3205];
        float t3207 = metal::select(0.0, t3206, t3194 > 0.0);
        float t3208 = t3172 + t3207;
        t3172 = t3208;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      float t3210 = (t3172 * 0.0013797212);
      int t3211 = t3170 * 4;
      int t3212 = t3211 + t3171;
      memory[8718016 + t3212] = t3210;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 28
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_28(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1452) - handled in variable access */
    int t3215 = id;
    for (uint t3216 = 0; t3216 < 4; t3216++) {
      float t3217 = 0.0;
      int t3218 = (int)frameCount;
      int t3219 = t3218 + 256;
      int t3220 = t3219 - 1;
      int t3221 = t3220 / 256;
      int t3222 = t3221 - 1;
      int t3223 = t3215 + 1024;
      int t3224 = t3223 - 1;
      int t3225 = t3224 / 256;
      float t3226 = metal::min(t3225, t3222);
      int t3227 = t3221 * 4;
      int t3228 = t3227 * 1024;
      int t3229 = t3228 - 1;
      for (uint t3230 = 0; t3230 < 5; t3230++) {
        float t3231 = t3226 - t3230;
        float t3232 = t3231 * 256.0;
        float t3233 = t3231 >= 0.0;
        float t3234 = (float)t3215;
        float t3235 = t3232 >= t3234;
        float t3236 = (float)t3221;
        float t3237 = t3231 < t3236;
        float t3238 = t3233 * t3235;
        float t3239 = t3238 * t3237;
        float t3240 = t3215 - t3232;
        float t3241 = t3240 + 1024.0;
        float t3242 = t3241 - 1.0;
        float t3243 = t3231 * 4.0;
        float t3244 = t3243 + t3216;
        float t3245 = t3244 * 1024.0;
        float t3246 = t3245 + t3242;
        float t3247 = (float)t3229;
        float t3248 = metal::min(t3246, t3247);
        float t3249 = metal::max(0.0, t3248);
        int t3250 = (int)t3249;
        float t3251 = memory[19335872 + t3250];
        float t3252 = metal::select(0.0, t3251, t3239 > 0.0);
        float t3253 = t3217 + t3252;
        t3217 = t3253;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      float t3255 = (t3217 * 0.0013797212);
      int t3256 = t3215 * 4;
      int t3257 = t3256 + t3216;
      memory[8783552 + t3257] = t3255;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 29
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_29(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1437) - handled in variable access */
    float t3264 = (t[4*frameCount + id] * 3.7252903e-09);
    float t3265 = -0.5 * t3264;
  }
  #pragma clang diagnostic pop
}



// KERNEL 30
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_30(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(638) - handled in variable access */
    int t3266 = id;
    float t3267 = t[3*frameCount + id] == 0.0;
    if (t3267) {
      int t3269 = t3266 / 128;
      for (uint t3270 = 0; t3270 < 4; t3270++) {
        int t3271 = t3269 * 4;
        int t3272 = t3271 + t3270;
        int t3273 = t3272 * 1024;
        int t3274 = t3272 * 257;
        int t3275 = t3272 * 1024;
        for (uint _pr3276 = 0; _pr3276 < 257; _pr3276++) {
          int t3277 = t3274 + _pr3276;
          float t3278 = memory[18548416 + t3277];
          int t3279 = t3274 + _pr3276;
          float t3280 = memory[18680000 + t3279];
          int t3281 = t3273 + _pr3276;
          float t3282 = memory[17499840 + t3281];
          int t3283 = t3273 + _pr3276;
          int t3284 = t3283 + 512;
          float t3285 = memory[17499840 + t3284];
          int t3286 = t3273 + _pr3276;
          float t3287 = memory[18024128 + t3286];
          int t3288 = t3273 + _pr3276;
          int t3289 = t3288 + 512;
          float t3290 = memory[18024128 + t3289];
          float t3291 = t3278 - t3280;
          float t3292 = 2.0 * t3291;
          float t3293 = t3292 * 7.6293945e-06;
          float t3294 = t3278 - t3280;
          float t3295 = -2.0 * t3294;
          float t3296 = t3295 * 7.6293945e-06;
          float t3297 = metal::max(t3278, 1e-08);
          float t3298 = metal::max(t3280, 1e-08);
          float t3299 = t3293 * t3282;
          float t3300 = t3299 / t3297;
          float t3301 = t3293 * t3285;
          float t3302 = t3301 / t3297;
          float t3303 = t3296 * t3287;
          float t3304 = t3303 / t3298;
          float t3305 = t3296 * t3290;
          float t3306 = t3305 / t3298;
          int t3307 = t3275 + _pr3276;
          memory[18811584 + t3307] = t3300;
          int t3309 = t3275 + _pr3276;
          int t3310 = t3309 + 512;
          memory[18811584 + t3310] = t3302;
          int t3312 = t3275 + _pr3276;
          memory[19335872 + t3312] = t3304;
          int t3314 = t3275 + _pr3276;
          int t3315 = t3314 + 512;
          memory[19335872 + t3315] = t3306;
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
        for (uint _pr3318 = 0; _pr3318 < 255; _pr3318++) {
          int t3319 = _pr3318 + 257;
          int t3320 = t3275 + t3319;
          memory[18811584 + t3320] = 0.0;
          int t3322 = t3275 + t3319;
          int t3323 = t3322 + 512;
          memory[18811584 + t3323] = 0.0;
          int t3325 = t3275 + t3319;
          memory[19335872 + t3325] = 0.0;
          int t3327 = t3275 + t3319;
          int t3328 = t3327 + 512;
          memory[19335872 + t3328] = 0.0;
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 31
// FrameOrder: parallel
// DispatchMode: perFrameThreadgroup1
kernel void kernel_31(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(638) - handled in variable access */
    int t3333 = id;
    threadgroup float scratch_0[512];
    threadgroup float scratch_1[512];
    threadgroup float scratch_2[511];
    threadgroup float scratch_3[511];
    float t3334 = t[3*frameCount + id] == 0.0;
    if (t3334) {
      for (uint t3336 = 0; t3336 < 511; t3336++) {
        float t3337 = memory[93122 + (int)t3336];
        scratch_2[(int)t3336] = t3337;
        float t3339 = memory[93633 + (int)t3336];
        scratch_3[(int)t3336] = t3339;
      }
      for (uint t3342 = 0; t3342 < 4; t3342++) {
        int t3343 = t3333 / 128;
        int t3344 = t3343 * 4;
        int t3345 = t3344 + t3342;
        int t3346 = t3345 * 1024;
        int t3347 = t3345 * 512;
        for (uint t3348 = 0; t3348 < 512; t3348++) {
          int t3349 = t3346 + t3348;
          float t3350 = memory[18811584 + t3349];
          int t3351 = t3346 + t3348;
          int t3352 = t3351 + 512;
          float t3353 = memory[18811584 + t3352];
          scratch_0[(int)t3348] = t3350;
          scratch_1[(int)t3348] = t3353;
        }
        for (uint t3357 = 0; t3357 < 512; t3357++) {
          float t3358 = memory[94144 + (int)t3357];
          float t3359 = (float)t3357;
          float t3360 = t3359 < t3358;
          int t3361 = (int)t3358;
          float t3362 = scratch_0[(int)t3357];
          float t3363 = scratch_1[(int)t3357];
          float t3364 = scratch_0[t3361];
          float t3365 = scratch_1[t3361];
          float t3366 = metal::select(t3362, t3364, t3360 > 0.0);
          float t3367 = metal::select(t3363, t3365, t3360 > 0.0);
          float t3368 = metal::select(t3364, t3362, t3360 > 0.0);
          float t3369 = metal::select(t3365, t3363, t3360 > 0.0);
          scratch_0[(int)t3357] = t3366;
          scratch_1[(int)t3357] = t3367;
          scratch_0[t3361] = t3368;
          scratch_1[t3361] = t3369;
        }
        for (uint t3375 = 0; t3375 < 256; t3375++) {
          float t3376 = (float)t3375;
          float t3377 = t3376;
          float t3378 = metal::floor(t3377);
          float t3379 = t3378;
          float t3380 = t3376 - t3379;
          float t3381 = t3378 * 2.0;
          float t3382 = t3381 + t3380;
          float t3383 = t3382 + 1.0;
          int t3384 = (int)t3380;
          int t3385 = t3384;
          float t3386 = scratch_2[t3385];
          float t3387 = scratch_3[t3385];
          int t3388 = (int)t3382;
          int t3389 = (int)t3383;
          float t3390 = scratch_0[t3388];
          float t3391 = scratch_1[t3388];
          float t3392 = scratch_0[t3389];
          float t3393 = scratch_1[t3389];
          float t3394 = t3386 * t3392;
          float t3395 = t3387 * t3393;
          float t3396 = t3394 - t3395;
          float t3397 = t3386 * t3393;
          float t3398 = t3387 * t3392;
          float t3399 = t3397 + t3398;
          float t3400 = t3390 + t3396;
          scratch_0[t3388] = t3400;
          float t3402 = t3391 + t3399;
          scratch_1[t3388] = t3402;
          float t3404 = t3390 - t3396;
          scratch_0[t3389] = t3404;
          float t3406 = t3391 - t3399;
          scratch_1[t3389] = t3406;
        }
        for (uint t3409 = 0; t3409 < 256; t3409++) {
          float t3410 = (float)t3409;
          float t3411 = (t3410 * 0.5);
          float t3412 = metal::floor(t3411);
          float t3413 = t3412 * 2.0;
          float t3414 = t3410 - t3413;
          float t3415 = t3412 * 4.0;
          float t3416 = t3415 + t3414;
          float t3417 = t3416 + 2.0;
          int t3418 = (int)t3414;
          int t3419 = 1 + t3418;
          float t3420 = scratch_2[t3419];
          float t3421 = scratch_3[t3419];
          int t3422 = (int)t3416;
          int t3423 = (int)t3417;
          float t3424 = scratch_0[t3422];
          float t3425 = scratch_1[t3422];
          float t3426 = scratch_0[t3423];
          float t3427 = scratch_1[t3423];
          float t3428 = t3420 * t3426;
          float t3429 = t3421 * t3427;
          float t3430 = t3428 - t3429;
          float t3431 = t3420 * t3427;
          float t3432 = t3421 * t3426;
          float t3433 = t3431 + t3432;
          float t3434 = t3424 + t3430;
          scratch_0[t3422] = t3434;
          float t3436 = t3425 + t3433;
          scratch_1[t3422] = t3436;
          float t3438 = t3424 - t3430;
          scratch_0[t3423] = t3438;
          float t3440 = t3425 - t3433;
          scratch_1[t3423] = t3440;
        }
        for (uint t3443 = 0; t3443 < 256; t3443++) {
          float t3444 = (float)t3443;
          float t3445 = (t3444 * 0.25);
          float t3446 = metal::floor(t3445);
          float t3447 = t3446 * 4.0;
          float t3448 = t3444 - t3447;
          float t3449 = t3446 * 8.0;
          float t3450 = t3449 + t3448;
          float t3451 = t3450 + 4.0;
          int t3452 = (int)t3448;
          int t3453 = 3 + t3452;
          float t3454 = scratch_2[t3453];
          float t3455 = scratch_3[t3453];
          int t3456 = (int)t3450;
          int t3457 = (int)t3451;
          float t3458 = scratch_0[t3456];
          float t3459 = scratch_1[t3456];
          float t3460 = scratch_0[t3457];
          float t3461 = scratch_1[t3457];
          float t3462 = t3454 * t3460;
          float t3463 = t3455 * t3461;
          float t3464 = t3462 - t3463;
          float t3465 = t3454 * t3461;
          float t3466 = t3455 * t3460;
          float t3467 = t3465 + t3466;
          float t3468 = t3458 + t3464;
          scratch_0[t3456] = t3468;
          float t3470 = t3459 + t3467;
          scratch_1[t3456] = t3470;
          float t3472 = t3458 - t3464;
          scratch_0[t3457] = t3472;
          float t3474 = t3459 - t3467;
          scratch_1[t3457] = t3474;
        }
        for (uint t3477 = 0; t3477 < 256; t3477++) {
          float t3478 = (float)t3477;
          float t3479 = (t3478 * 0.125);
          float t3480 = metal::floor(t3479);
          float t3481 = t3480 * 8.0;
          float t3482 = t3478 - t3481;
          float t3483 = t3480 * 16.0;
          float t3484 = t3483 + t3482;
          float t3485 = t3484 + 8.0;
          int t3486 = (int)t3482;
          int t3487 = 7 + t3486;
          float t3488 = scratch_2[t3487];
          float t3489 = scratch_3[t3487];
          int t3490 = (int)t3484;
          int t3491 = (int)t3485;
          float t3492 = scratch_0[t3490];
          float t3493 = scratch_1[t3490];
          float t3494 = scratch_0[t3491];
          float t3495 = scratch_1[t3491];
          float t3496 = t3488 * t3494;
          float t3497 = t3489 * t3495;
          float t3498 = t3496 - t3497;
          float t3499 = t3488 * t3495;
          float t3500 = t3489 * t3494;
          float t3501 = t3499 + t3500;
          float t3502 = t3492 + t3498;
          scratch_0[t3490] = t3502;
          float t3504 = t3493 + t3501;
          scratch_1[t3490] = t3504;
          float t3506 = t3492 - t3498;
          scratch_0[t3491] = t3506;
          float t3508 = t3493 - t3501;
          scratch_1[t3491] = t3508;
        }
        for (uint t3511 = 0; t3511 < 256; t3511++) {
          float t3512 = (float)t3511;
          float t3513 = (t3512 * 0.0625);
          float t3514 = metal::floor(t3513);
          float t3515 = t3514 * 16.0;
          float t3516 = t3512 - t3515;
          float t3517 = t3514 * 32.0;
          float t3518 = t3517 + t3516;
          float t3519 = t3518 + 16.0;
          int t3520 = (int)t3516;
          int t3521 = 15 + t3520;
          float t3522 = scratch_2[t3521];
          float t3523 = scratch_3[t3521];
          int t3524 = (int)t3518;
          int t3525 = (int)t3519;
          float t3526 = scratch_0[t3524];
          float t3527 = scratch_1[t3524];
          float t3528 = scratch_0[t3525];
          float t3529 = scratch_1[t3525];
          float t3530 = t3522 * t3528;
          float t3531 = t3523 * t3529;
          float t3532 = t3530 - t3531;
          float t3533 = t3522 * t3529;
          float t3534 = t3523 * t3528;
          float t3535 = t3533 + t3534;
          float t3536 = t3526 + t3532;
          scratch_0[t3524] = t3536;
          float t3538 = t3527 + t3535;
          scratch_1[t3524] = t3538;
          float t3540 = t3526 - t3532;
          scratch_0[t3525] = t3540;
          float t3542 = t3527 - t3535;
          scratch_1[t3525] = t3542;
        }
        for (uint t3545 = 0; t3545 < 256; t3545++) {
          float t3546 = (float)t3545;
          float t3547 = (t3546 * 0.03125);
          float t3548 = metal::floor(t3547);
          float t3549 = t3548 * 32.0;
          float t3550 = t3546 - t3549;
          float t3551 = t3548 * 64.0;
          float t3552 = t3551 + t3550;
          float t3553 = t3552 + 32.0;
          int t3554 = (int)t3550;
          int t3555 = 31 + t3554;
          float t3556 = scratch_2[t3555];
          float t3557 = scratch_3[t3555];
          int t3558 = (int)t3552;
          int t3559 = (int)t3553;
          float t3560 = scratch_0[t3558];
          float t3561 = scratch_1[t3558];
          float t3562 = scratch_0[t3559];
          float t3563 = scratch_1[t3559];
          float t3564 = t3556 * t3562;
          float t3565 = t3557 * t3563;
          float t3566 = t3564 - t3565;
          float t3567 = t3556 * t3563;
          float t3568 = t3557 * t3562;
          float t3569 = t3567 + t3568;
          float t3570 = t3560 + t3566;
          scratch_0[t3558] = t3570;
          float t3572 = t3561 + t3569;
          scratch_1[t3558] = t3572;
          float t3574 = t3560 - t3566;
          scratch_0[t3559] = t3574;
          float t3576 = t3561 - t3569;
          scratch_1[t3559] = t3576;
        }
        for (uint t3579 = 0; t3579 < 256; t3579++) {
          float t3580 = (float)t3579;
          float t3581 = (t3580 * 0.015625);
          float t3582 = metal::floor(t3581);
          float t3583 = t3582 * 64.0;
          float t3584 = t3580 - t3583;
          float t3585 = t3582 * 128.0;
          float t3586 = t3585 + t3584;
          float t3587 = t3586 + 64.0;
          int t3588 = (int)t3584;
          int t3589 = 63 + t3588;
          float t3590 = scratch_2[t3589];
          float t3591 = scratch_3[t3589];
          int t3592 = (int)t3586;
          int t3593 = (int)t3587;
          float t3594 = scratch_0[t3592];
          float t3595 = scratch_1[t3592];
          float t3596 = scratch_0[t3593];
          float t3597 = scratch_1[t3593];
          float t3598 = t3590 * t3596;
          float t3599 = t3591 * t3597;
          float t3600 = t3598 - t3599;
          float t3601 = t3590 * t3597;
          float t3602 = t3591 * t3596;
          float t3603 = t3601 + t3602;
          float t3604 = t3594 + t3600;
          scratch_0[t3592] = t3604;
          float t3606 = t3595 + t3603;
          scratch_1[t3592] = t3606;
          float t3608 = t3594 - t3600;
          scratch_0[t3593] = t3608;
          float t3610 = t3595 - t3603;
          scratch_1[t3593] = t3610;
        }
        for (uint t3613 = 0; t3613 < 256; t3613++) {
          float t3614 = (float)t3613;
          float t3615 = (t3614 * 0.0078125);
          float t3616 = metal::floor(t3615);
          float t3617 = t3616 * 128.0;
          float t3618 = t3614 - t3617;
          float t3619 = t3616 * 256.0;
          float t3620 = t3619 + t3618;
          float t3621 = t3620 + 128.0;
          int t3622 = (int)t3618;
          int t3623 = 127 + t3622;
          float t3624 = scratch_2[t3623];
          float t3625 = scratch_3[t3623];
          int t3626 = (int)t3620;
          int t3627 = (int)t3621;
          float t3628 = scratch_0[t3626];
          float t3629 = scratch_1[t3626];
          float t3630 = scratch_0[t3627];
          float t3631 = scratch_1[t3627];
          float t3632 = t3624 * t3630;
          float t3633 = t3625 * t3631;
          float t3634 = t3632 - t3633;
          float t3635 = t3624 * t3631;
          float t3636 = t3625 * t3630;
          float t3637 = t3635 + t3636;
          float t3638 = t3628 + t3634;
          scratch_0[t3626] = t3638;
          float t3640 = t3629 + t3637;
          scratch_1[t3626] = t3640;
          float t3642 = t3628 - t3634;
          scratch_0[t3627] = t3642;
          float t3644 = t3629 - t3637;
          scratch_1[t3627] = t3644;
        }
        for (uint t3647 = 0; t3647 < 256; t3647++) {
          float t3648 = (float)t3647;
          float t3649 = (t3648 * 0.00390625);
          float t3650 = metal::floor(t3649);
          float t3651 = t3650 * 256.0;
          float t3652 = t3648 - t3651;
          float t3653 = t3650 * 512.0;
          float t3654 = t3653 + t3652;
          float t3655 = t3654 + 256.0;
          int t3656 = (int)t3652;
          int t3657 = 255 + t3656;
          float t3658 = scratch_2[t3657];
          float t3659 = scratch_3[t3657];
          int t3660 = (int)t3654;
          int t3661 = (int)t3655;
          float t3662 = scratch_0[t3660];
          float t3663 = scratch_1[t3660];
          float t3664 = scratch_0[t3661];
          float t3665 = scratch_1[t3661];
          float t3666 = t3658 * t3664;
          float t3667 = t3659 * t3665;
          float t3668 = t3666 - t3667;
          float t3669 = t3658 * t3665;
          float t3670 = t3659 * t3664;
          float t3671 = t3669 + t3670;
          float t3672 = t3662 + t3668;
          scratch_0[t3660] = t3672;
          float t3674 = t3663 + t3671;
          scratch_1[t3660] = t3674;
          float t3676 = t3662 - t3668;
          scratch_0[t3661] = t3676;
          float t3678 = t3663 - t3671;
          scratch_1[t3661] = t3678;
        }
        for (uint t3681 = 0; t3681 < 512; t3681++) {
          float t3682 = scratch_0[(int)t3681];
          float t3683 = t3682 * 7.599708e-06;
          float t3684 = memory[92610 + (int)t3681];
          int t3685 = t3347 + t3681;
          float t3686 = t3683 * t3684;
          memory[17499840 + t3685] = t3686;
        }
        int t3689 = t3333 / 128;
        int t3690 = t3689 * 4;
        int t3691 = t3690 + t3342;
        int t3692 = t3691 * 1024;
        int t3693 = t3691 * 512;
        for (uint t3694 = 0; t3694 < 512; t3694++) {
          int t3695 = t3692 + t3694;
          float t3696 = memory[19335872 + t3695];
          int t3697 = t3692 + t3694;
          int t3698 = t3697 + 512;
          float t3699 = memory[19335872 + t3698];
          scratch_0[(int)t3694] = t3696;
          scratch_1[(int)t3694] = t3699;
        }
        for (uint t3703 = 0; t3703 < 512; t3703++) {
          float t3704 = memory[94144 + (int)t3703];
          float t3705 = (float)t3703;
          float t3706 = t3705 < t3704;
          int t3707 = (int)t3704;
          float t3708 = scratch_0[(int)t3703];
          float t3709 = scratch_1[(int)t3703];
          float t3710 = scratch_0[t3707];
          float t3711 = scratch_1[t3707];
          float t3712 = metal::select(t3708, t3710, t3706 > 0.0);
          float t3713 = metal::select(t3709, t3711, t3706 > 0.0);
          float t3714 = metal::select(t3710, t3708, t3706 > 0.0);
          float t3715 = metal::select(t3711, t3709, t3706 > 0.0);
          scratch_0[(int)t3703] = t3712;
          scratch_1[(int)t3703] = t3713;
          scratch_0[t3707] = t3714;
          scratch_1[t3707] = t3715;
        }
        for (uint t3721 = 0; t3721 < 256; t3721++) {
          float t3722 = (float)t3721;
          float t3723 = t3722;
          float t3724 = metal::floor(t3723);
          float t3725 = t3724;
          float t3726 = t3722 - t3725;
          float t3727 = t3724 * 2.0;
          float t3728 = t3727 + t3726;
          float t3729 = t3728 + 1.0;
          int t3730 = (int)t3726;
          int t3731 = t3730;
          float t3732 = scratch_2[t3731];
          float t3733 = scratch_3[t3731];
          int t3734 = (int)t3728;
          int t3735 = (int)t3729;
          float t3736 = scratch_0[t3734];
          float t3737 = scratch_1[t3734];
          float t3738 = scratch_0[t3735];
          float t3739 = scratch_1[t3735];
          float t3740 = t3732 * t3738;
          float t3741 = t3733 * t3739;
          float t3742 = t3740 - t3741;
          float t3743 = t3732 * t3739;
          float t3744 = t3733 * t3738;
          float t3745 = t3743 + t3744;
          float t3746 = t3736 + t3742;
          scratch_0[t3734] = t3746;
          float t3748 = t3737 + t3745;
          scratch_1[t3734] = t3748;
          float t3750 = t3736 - t3742;
          scratch_0[t3735] = t3750;
          float t3752 = t3737 - t3745;
          scratch_1[t3735] = t3752;
        }
        for (uint t3755 = 0; t3755 < 256; t3755++) {
          float t3756 = (float)t3755;
          float t3757 = (t3756 * 0.5);
          float t3758 = metal::floor(t3757);
          float t3759 = t3758 * 2.0;
          float t3760 = t3756 - t3759;
          float t3761 = t3758 * 4.0;
          float t3762 = t3761 + t3760;
          float t3763 = t3762 + 2.0;
          int t3764 = (int)t3760;
          int t3765 = 1 + t3764;
          float t3766 = scratch_2[t3765];
          float t3767 = scratch_3[t3765];
          int t3768 = (int)t3762;
          int t3769 = (int)t3763;
          float t3770 = scratch_0[t3768];
          float t3771 = scratch_1[t3768];
          float t3772 = scratch_0[t3769];
          float t3773 = scratch_1[t3769];
          float t3774 = t3766 * t3772;
          float t3775 = t3767 * t3773;
          float t3776 = t3774 - t3775;
          float t3777 = t3766 * t3773;
          float t3778 = t3767 * t3772;
          float t3779 = t3777 + t3778;
          float t3780 = t3770 + t3776;
          scratch_0[t3768] = t3780;
          float t3782 = t3771 + t3779;
          scratch_1[t3768] = t3782;
          float t3784 = t3770 - t3776;
          scratch_0[t3769] = t3784;
          float t3786 = t3771 - t3779;
          scratch_1[t3769] = t3786;
        }
        for (uint t3789 = 0; t3789 < 256; t3789++) {
          float t3790 = (float)t3789;
          float t3791 = (t3790 * 0.25);
          float t3792 = metal::floor(t3791);
          float t3793 = t3792 * 4.0;
          float t3794 = t3790 - t3793;
          float t3795 = t3792 * 8.0;
          float t3796 = t3795 + t3794;
          float t3797 = t3796 + 4.0;
          int t3798 = (int)t3794;
          int t3799 = 3 + t3798;
          float t3800 = scratch_2[t3799];
          float t3801 = scratch_3[t3799];
          int t3802 = (int)t3796;
          int t3803 = (int)t3797;
          float t3804 = scratch_0[t3802];
          float t3805 = scratch_1[t3802];
          float t3806 = scratch_0[t3803];
          float t3807 = scratch_1[t3803];
          float t3808 = t3800 * t3806;
          float t3809 = t3801 * t3807;
          float t3810 = t3808 - t3809;
          float t3811 = t3800 * t3807;
          float t3812 = t3801 * t3806;
          float t3813 = t3811 + t3812;
          float t3814 = t3804 + t3810;
          scratch_0[t3802] = t3814;
          float t3816 = t3805 + t3813;
          scratch_1[t3802] = t3816;
          float t3818 = t3804 - t3810;
          scratch_0[t3803] = t3818;
          float t3820 = t3805 - t3813;
          scratch_1[t3803] = t3820;
        }
        for (uint t3823 = 0; t3823 < 256; t3823++) {
          float t3824 = (float)t3823;
          float t3825 = (t3824 * 0.125);
          float t3826 = metal::floor(t3825);
          float t3827 = t3826 * 8.0;
          float t3828 = t3824 - t3827;
          float t3829 = t3826 * 16.0;
          float t3830 = t3829 + t3828;
          float t3831 = t3830 + 8.0;
          int t3832 = (int)t3828;
          int t3833 = 7 + t3832;
          float t3834 = scratch_2[t3833];
          float t3835 = scratch_3[t3833];
          int t3836 = (int)t3830;
          int t3837 = (int)t3831;
          float t3838 = scratch_0[t3836];
          float t3839 = scratch_1[t3836];
          float t3840 = scratch_0[t3837];
          float t3841 = scratch_1[t3837];
          float t3842 = t3834 * t3840;
          float t3843 = t3835 * t3841;
          float t3844 = t3842 - t3843;
          float t3845 = t3834 * t3841;
          float t3846 = t3835 * t3840;
          float t3847 = t3845 + t3846;
          float t3848 = t3838 + t3844;
          scratch_0[t3836] = t3848;
          float t3850 = t3839 + t3847;
          scratch_1[t3836] = t3850;
          float t3852 = t3838 - t3844;
          scratch_0[t3837] = t3852;
          float t3854 = t3839 - t3847;
          scratch_1[t3837] = t3854;
        }
        for (uint t3857 = 0; t3857 < 256; t3857++) {
          float t3858 = (float)t3857;
          float t3859 = (t3858 * 0.0625);
          float t3860 = metal::floor(t3859);
          float t3861 = t3860 * 16.0;
          float t3862 = t3858 - t3861;
          float t3863 = t3860 * 32.0;
          float t3864 = t3863 + t3862;
          float t3865 = t3864 + 16.0;
          int t3866 = (int)t3862;
          int t3867 = 15 + t3866;
          float t3868 = scratch_2[t3867];
          float t3869 = scratch_3[t3867];
          int t3870 = (int)t3864;
          int t3871 = (int)t3865;
          float t3872 = scratch_0[t3870];
          float t3873 = scratch_1[t3870];
          float t3874 = scratch_0[t3871];
          float t3875 = scratch_1[t3871];
          float t3876 = t3868 * t3874;
          float t3877 = t3869 * t3875;
          float t3878 = t3876 - t3877;
          float t3879 = t3868 * t3875;
          float t3880 = t3869 * t3874;
          float t3881 = t3879 + t3880;
          float t3882 = t3872 + t3878;
          scratch_0[t3870] = t3882;
          float t3884 = t3873 + t3881;
          scratch_1[t3870] = t3884;
          float t3886 = t3872 - t3878;
          scratch_0[t3871] = t3886;
          float t3888 = t3873 - t3881;
          scratch_1[t3871] = t3888;
        }
        for (uint t3891 = 0; t3891 < 256; t3891++) {
          float t3892 = (float)t3891;
          float t3893 = (t3892 * 0.03125);
          float t3894 = metal::floor(t3893);
          float t3895 = t3894 * 32.0;
          float t3896 = t3892 - t3895;
          float t3897 = t3894 * 64.0;
          float t3898 = t3897 + t3896;
          float t3899 = t3898 + 32.0;
          int t3900 = (int)t3896;
          int t3901 = 31 + t3900;
          float t3902 = scratch_2[t3901];
          float t3903 = scratch_3[t3901];
          int t3904 = (int)t3898;
          int t3905 = (int)t3899;
          float t3906 = scratch_0[t3904];
          float t3907 = scratch_1[t3904];
          float t3908 = scratch_0[t3905];
          float t3909 = scratch_1[t3905];
          float t3910 = t3902 * t3908;
          float t3911 = t3903 * t3909;
          float t3912 = t3910 - t3911;
          float t3913 = t3902 * t3909;
          float t3914 = t3903 * t3908;
          float t3915 = t3913 + t3914;
          float t3916 = t3906 + t3912;
          scratch_0[t3904] = t3916;
          float t3918 = t3907 + t3915;
          scratch_1[t3904] = t3918;
          float t3920 = t3906 - t3912;
          scratch_0[t3905] = t3920;
          float t3922 = t3907 - t3915;
          scratch_1[t3905] = t3922;
        }
        for (uint t3925 = 0; t3925 < 256; t3925++) {
          float t3926 = (float)t3925;
          float t3927 = (t3926 * 0.015625);
          float t3928 = metal::floor(t3927);
          float t3929 = t3928 * 64.0;
          float t3930 = t3926 - t3929;
          float t3931 = t3928 * 128.0;
          float t3932 = t3931 + t3930;
          float t3933 = t3932 + 64.0;
          int t3934 = (int)t3930;
          int t3935 = 63 + t3934;
          float t3936 = scratch_2[t3935];
          float t3937 = scratch_3[t3935];
          int t3938 = (int)t3932;
          int t3939 = (int)t3933;
          float t3940 = scratch_0[t3938];
          float t3941 = scratch_1[t3938];
          float t3942 = scratch_0[t3939];
          float t3943 = scratch_1[t3939];
          float t3944 = t3936 * t3942;
          float t3945 = t3937 * t3943;
          float t3946 = t3944 - t3945;
          float t3947 = t3936 * t3943;
          float t3948 = t3937 * t3942;
          float t3949 = t3947 + t3948;
          float t3950 = t3940 + t3946;
          scratch_0[t3938] = t3950;
          float t3952 = t3941 + t3949;
          scratch_1[t3938] = t3952;
          float t3954 = t3940 - t3946;
          scratch_0[t3939] = t3954;
          float t3956 = t3941 - t3949;
          scratch_1[t3939] = t3956;
        }
        for (uint t3959 = 0; t3959 < 256; t3959++) {
          float t3960 = (float)t3959;
          float t3961 = (t3960 * 0.0078125);
          float t3962 = metal::floor(t3961);
          float t3963 = t3962 * 128.0;
          float t3964 = t3960 - t3963;
          float t3965 = t3962 * 256.0;
          float t3966 = t3965 + t3964;
          float t3967 = t3966 + 128.0;
          int t3968 = (int)t3964;
          int t3969 = 127 + t3968;
          float t3970 = scratch_2[t3969];
          float t3971 = scratch_3[t3969];
          int t3972 = (int)t3966;
          int t3973 = (int)t3967;
          float t3974 = scratch_0[t3972];
          float t3975 = scratch_1[t3972];
          float t3976 = scratch_0[t3973];
          float t3977 = scratch_1[t3973];
          float t3978 = t3970 * t3976;
          float t3979 = t3971 * t3977;
          float t3980 = t3978 - t3979;
          float t3981 = t3970 * t3977;
          float t3982 = t3971 * t3976;
          float t3983 = t3981 + t3982;
          float t3984 = t3974 + t3980;
          scratch_0[t3972] = t3984;
          float t3986 = t3975 + t3983;
          scratch_1[t3972] = t3986;
          float t3988 = t3974 - t3980;
          scratch_0[t3973] = t3988;
          float t3990 = t3975 - t3983;
          scratch_1[t3973] = t3990;
        }
        for (uint t3993 = 0; t3993 < 256; t3993++) {
          float t3994 = (float)t3993;
          float t3995 = (t3994 * 0.00390625);
          float t3996 = metal::floor(t3995);
          float t3997 = t3996 * 256.0;
          float t3998 = t3994 - t3997;
          float t3999 = t3996 * 512.0;
          float t4000 = t3999 + t3998;
          float t4001 = t4000 + 256.0;
          int t4002 = (int)t3998;
          int t4003 = 255 + t4002;
          float t4004 = scratch_2[t4003];
          float t4005 = scratch_3[t4003];
          int t4006 = (int)t4000;
          int t4007 = (int)t4001;
          float t4008 = scratch_0[t4006];
          float t4009 = scratch_1[t4006];
          float t4010 = scratch_0[t4007];
          float t4011 = scratch_1[t4007];
          float t4012 = t4004 * t4010;
          float t4013 = t4005 * t4011;
          float t4014 = t4012 - t4013;
          float t4015 = t4004 * t4011;
          float t4016 = t4005 * t4010;
          float t4017 = t4015 + t4016;
          float t4018 = t4008 + t4014;
          scratch_0[t4006] = t4018;
          float t4020 = t4009 + t4017;
          scratch_1[t4006] = t4020;
          float t4022 = t4008 - t4014;
          scratch_0[t4007] = t4022;
          float t4024 = t4009 - t4017;
          scratch_1[t4007] = t4024;
        }
        for (uint t4027 = 0; t4027 < 512; t4027++) {
          float t4028 = scratch_0[(int)t4027];
          float t4029 = t4028 * 7.599708e-06;
          float t4030 = memory[92610 + (int)t4027];
          int t4031 = t3693 + t4027;
          float t4032 = t4029 * t4030;
          memory[18024128 + t4031] = t4032;
        }
      }
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 32
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_32(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(638) - handled in variable access */
    int t4037 = id;
    for (uint t4038 = 0; t4038 < 4; t4038++) {
      float t4039 = 0.0;
      int t4040 = (int)frameCount;
      int t4041 = t4040 + 128;
      int t4042 = t4041 - 1;
      int t4043 = t4042 / 128;
      int t4044 = t4043 - 1;
      int t4045 = t4037 + 512;
      int t4046 = t4045 - 1;
      int t4047 = t4046 / 128;
      float t4048 = metal::min(t4047, t4044);
      int t4049 = t4043 * 4;
      int t4050 = t4049 * 512;
      int t4051 = t4050 - 1;
      for (uint t4052 = 0; t4052 < 5; t4052++) {
        float t4053 = t4048 - t4052;
        float t4054 = t4053 * 128.0;
        float t4055 = t4053 >= 0.0;
        float t4056 = (float)t4037;
        float t4057 = t4054 >= t4056;
        float t4058 = (float)t4043;
        float t4059 = t4053 < t4058;
        float t4060 = t4055 * t4057;
        float t4061 = t4060 * t4059;
        float t4062 = t4037 - t4054;
        float t4063 = t4062 + 512.0;
        float t4064 = t4063 - 1.0;
        float t4065 = t4053 * 4.0;
        float t4066 = t4065 + t4038;
        float t4067 = t4066 * 512.0;
        float t4068 = t4067 + t4064;
        float t4069 = (float)t4051;
        float t4070 = metal::min(t4068, t4069);
        float t4071 = metal::max(0.0, t4070);
        int t4072 = (int)t4071;
        float t4073 = memory[17499840 + t4072];
        float t4074 = metal::select(0.0, t4073, t4061 > 0.0);
        float t4075 = t4039 + t4074;
        t4039 = t4075;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      float t4077 = (t4039 * 0.0027567567);
      int t4078 = t4037 * 4;
      int t4079 = t4078 + t4038;
      memory[19860160 + t4079] = t4077;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 33
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_33(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(638) - handled in variable access */
    int t4082 = id;
    for (uint t4083 = 0; t4083 < 4; t4083++) {
      float t4084 = 0.0;
      int t4085 = (int)frameCount;
      int t4086 = t4085 + 128;
      int t4087 = t4086 - 1;
      int t4088 = t4087 / 128;
      int t4089 = t4088 - 1;
      int t4090 = t4082 + 512;
      int t4091 = t4090 - 1;
      int t4092 = t4091 / 128;
      float t4093 = metal::min(t4092, t4089);
      int t4094 = t4088 * 4;
      int t4095 = t4094 * 512;
      int t4096 = t4095 - 1;
      for (uint t4097 = 0; t4097 < 5; t4097++) {
        float t4098 = t4093 - t4097;
        float t4099 = t4098 * 128.0;
        float t4100 = t4098 >= 0.0;
        float t4101 = (float)t4082;
        float t4102 = t4099 >= t4101;
        float t4103 = (float)t4088;
        float t4104 = t4098 < t4103;
        float t4105 = t4100 * t4102;
        float t4106 = t4105 * t4104;
        float t4107 = t4082 - t4099;
        float t4108 = t4107 + 512.0;
        float t4109 = t4108 - 1.0;
        float t4110 = t4098 * 4.0;
        float t4111 = t4110 + t4083;
        float t4112 = t4111 * 512.0;
        float t4113 = t4112 + t4109;
        float t4114 = (float)t4096;
        float t4115 = metal::min(t4113, t4114);
        float t4116 = metal::max(0.0, t4115);
        int t4117 = (int)t4116;
        float t4118 = memory[18024128 + t4117];
        float t4119 = metal::select(0.0, t4118, t4106 > 0.0);
        float t4120 = t4084 + t4119;
        t4084 = t4120;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      float t4122 = (t4084 * 0.0027567567);
      int t4123 = t4082 * 4;
      int t4124 = t4123 + t4083;
      memory[19991488 + t4124] = t4122;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 34
// FrameOrder: parallel
// DispatchMode: perFrameScaled(4)
kernel void kernel_34(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4905 = frameCount * 4.0;
  if (id >= 0 && id < (uint)(t4905)) {
    int t4127 = id;
    int t4128 = t4127 / 4;
    uint _frameIndex = (uint)(t4128);
    int t4129 = t4128 * 4;
    int t4130 = t4127 - t4129;
    int t4131 = _frameIndex;
    int t4132 = t4131 * 4;
    int t4133 = t4132 + t4130;
    float t4134 = memory[8718016 + t4133];
    int t4135 = _frameIndex;
    int t4136 = t4135 * 4;
    int t4137 = t4136 + t4130;
    float t4138 = memory[19860160 + t4137];
    float t4139 = t4134 + t4138;
    int t4140 = _frameIndex;
    int t4141 = t4140 * 4;
    int t4142 = t4141 + t4130;
    float t4143 = memory[8783552 + t4142];
    int t4144 = _frameIndex;
    int t4145 = t4144 * 4;
    int t4146 = t4145 + t4130;
    float t4147 = memory[19991488 + t4146];
    float t4148 = t4143 + t4147;
    float t4149 = 0.015625 * t4139;
    int t4150 = _frameIndex;
    int t4151 = t4150 * 4;
    int t4152 = t4151 + t4130;
    float t4153 = memory[13240000 + t4152];
    float t4154 = t4153 * t4139;
    int t4155 = _frameIndex;
    int t4156 = t4155 * 4;
    int t4157 = t4156 + t4130;
    float t4158 = memory[8914624 + t4157];
    float t4159 = t4158 * t4149;
    int t4160 = _frameIndex;
    int t4161 = t4160 * 4;
    int t4162 = t4161 + t4130;
    memory[18548416 + t4162] = t4159;
    int t4164 = _frameIndex;
    int t4165 = t4164 * 4;
    int t4166 = t4165 + t4130;
    float t4167 = memory[8849088 + t4166];
    float t4168 = t4167 * t4149;
    int t4169 = _frameIndex;
    int t4170 = t4169 * 4;
    int t4171 = t4170 + t4130;
    memory[18680000 + t4171] = t4168;
  }
  #pragma clang diagnostic pop
}



// KERNEL 35
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_35(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4906 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4906)) {
    /* loadGlobal(310) - handled in variable access */
    int t4173 = id;
    int t4174 = t4173 / 256;
    uint _frameIndex = (uint)(t4174);
    int t4175 = t4174 * 256;
    int t4176 = t4173 - t4175;
    float t4177 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t4178 = t4177 < 0.0;
    float t4179 = t4177 + 64.0;
    float t4180 = metal::select(t4177, t4179, t4178 > 0.0);
    float t4181 = metal::floor(t4180);
    float t4182 = t4181 + 1.0;
    float t4183 = t4182 >= 64.0;
    float t4184 = metal::select(t4182, 0.0, t4183 > 0.0);
    float t4185 = t4180 - t4181;
    int t4186 = _frameIndex;
    memory[98752 + t4186] = t4181;
    memory[295360 + t4186] = t4185;
    float t4189 = t4186 + 16384.0;
    int t4190 = (int)t4189;
    memory[98752 + t4190] = t4184;
    float t4192 = 1.0 - t4185;
    float t4193 = t4186 * 4.0;
    for (uint _pr4194 = 0; _pr4194 < 4; _pr4194++) {
      float t4195 = (float)_pr4194;
      float t4196 = t4193 + t4195;
      int t4197 = (int)t4196;
      float t4198 = memory[18680000 + t4197];
      float t4199 = t4193 + t4195;
      float t4200 = t4198 * t4192;
      int t4201 = (int)t4199;
      memory[8718016 + t4201] = t4200;
      float t4203 = t4198 * t4185;
      int t4204 = (int)t4199;
      memory[8783552 + t4204] = t4203;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 36
// FrameOrder: sequential
// DispatchMode: staticThreads(256)
kernel void kernel_36(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 256) { uint _pr4207 = id;
    int t4208 = _pr4207 / 4;
    int t4209 = t4208 * 4;
    int t4210 = _pr4207 - t4209;
    float t4211 = (float)t4208;
    float t4212 = (float)t4210;
    float t4213 = 0.0;
    for (uint t4214 = 0; t4214 < 16384; t4214++) {
      float t4215 = (float)t4214;
      float t4216 = t4215 < frameCount;
      float t4217 = t4215 * 4.0;
      float t4218 = t4217 + t4212;
      float t4219 = memory[98752 + (int)t4214];
      float t4220 = t4219 - t4211;
      float t4221 = metal::abs(t4220);
      float t4222 = t4221 < 0.5;
      int t4223 = (int)t4218;
      float t4224 = memory[8718016 + t4223];
      float t4225 = t4216 * t4222;
      float t4226 = t4225 > 0.0;
      float t4227 = metal::select(0.0, t4224, t4226 > 0.0);
      float t4228 = t4213 + t4227;
      t4213 = t4228;
      float t4229 = t4215 + 16384.0;
      int t4230 = (int)t4229;
      float t4231 = memory[98752 + t4230];
      float t4232 = t4231 - t4211;
      float t4233 = metal::abs(t4232);
      float t4234 = t4233 < 0.5;
      int t4235 = (int)t4218;
      float t4236 = memory[8783552 + t4235];
      float t4237 = t4216 * t4234;
      float t4238 = t4237 > 0.0;
      float t4239 = metal::select(0.0, t4236, t4238 > 0.0);
      float t4240 = t4213 + t4239;
      t4213 = t4240;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4242 = t4211 * 4.0;
    float t4243 = t4242 + t4212;
    int t4244 = (int)t4243;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20647104 + t4244], t4213, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 37
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_37(
    constant uint &frameCount [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4907 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4907)) {
    int t4247 = id;
    int t4248 = t4247 / 256;
    uint _frameIndex = (uint)(t4248);
    int t4249 = t4248 * 256;
    int t4250 = t4247 - t4249;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([256, 1]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 38
// FrameOrder: parallel
// DispatchMode: staticThreads(1)
kernel void kernel_38(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint t4251 = 0; t4251 < 256; t4251++) {
      int t4252 = t4251;
      int t4253 = t4252;
      int t4254 = t4251 - t4253;
      int t4255 = t4252 / 64;
      int t4256 = t4255 * 64;
      int t4257 = t4252 - t4256;
      int t4258 = t4257 * 4;
      int t4259 = t4255 + t4258;
      float t4260 = memory[20647104 + t4259];
      float t4261 = memory[328896 + (int)t4251];
      float t4262 = t4260 / t4261;
      float t4263 = memory[328896 + (int)t4251];
      float t4264 = memory[328896 + (int)t4251];
      float t4265 = t4263 * t4264;
      float t4266 = 1.0 / t4265;
      int t4267 = t4251;
      int t4268 = t4267;
      int t4269 = t4251 - t4268;
      int t4270 = t4267 / 64;
      int t4271 = t4270 * 64;
      int t4272 = t4267 - t4271;
      int t4273 = t4272 * 4;
      int t4274 = t4270 + t4273;
      float t4275 = memory[20647104 + t4274];
      float t4276 = t4275 * -1.0;
      float t4277 = t4276 * t4266;
      float t4278 = t4262 + t4277;
      float t4279 = memory[328384 + (int)t4251];
      float t4280 = metal::exp(t4279);
      float t4281 = t4280 * t4277;
      float t4282 = -1.0 * t4281;
      memory[328128 + (int)t4251] = t4282;
      float t4284 = memory[328640 + (int)t4251];
      float t4285 = t4284 * t4281;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t4286 = 0; t4286 < 1; t4286++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=184, axis=0, in=[256, 1], out=[1], inFA=false, outFA=false), value: empty) */
      float t4287 = 0.0;
      int t4288 = t4286;
      int t4289 = t4288;
      int t4290 = t4286 - t4289;
      int t4291 = t4288;
      int t4292 = t4291;
      for (uint t4293 = 0; t4293 < 256; t4293++) {
        int t4294 = t4293;
        int t4295 = t4292 + t4294;
        float t4296 = memory[328128 + t4295];
        float t4297 = t4287 + t4296;
        t4287 = t4297;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[329152 + (int)t4286] = t4287;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 39
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_39(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t4300 = id;
    int t4301 = t4300 / 32768;
    uint _frameIndex = (uint)(t4301);
    int t4302 = t4301 * 32768;
    int t4303 = t4300 - t4302;
    /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=186, axis=2, in=[256, 1], out=[256, 1, 128], inFA=false, outFA=false), value: empty) */
    int t4304 = t4303 / 128;
    int t4305 = t4304 % 256;
    int t4306 = t4305 * 1;
    int t4307 = 0 + t4306;
    int t4308 = t4303 / 128;
    int t4309 = t4308 % 1;
    int t4310 = t4309 * 1;
    int t4311 = t4307 + t4310;
    float t4312 = memory[328128 + t4311];
    memory[98752 + t4303] = t4312;
    int t4314 = t4303 / 128;
    int t4315 = t4314 * 128;
    int t4316 = t4303 - t4315;
    int t4317 = t4316 / 128;
    int t4318 = t4317 * 128;
    int t4319 = t4316 - t4318;
    int t4320 = t4319 / 128;
    int t4321 = t4320 * 128;
    int t4322 = t4319 - t4321;
    float t4323 = memory[25280 + t4322];
    float t4324 = memory[98752 + t4303];
    float t4325 = t4323 * t4324;
    memory[8718016 + t4303] = t4325;
  }
  #pragma clang diagnostic pop
}



// KERNEL 40
// FrameOrder: parallel
// DispatchMode: perFrameScaled(128)
kernel void kernel_40(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4908 = frameCount * 128.0;
  if (id >= 0 && id < (uint)(t4908)) {
    int t4327 = id;
    int t4328 = t4327 / 128;
    uint _frameIndex = (uint)(t4328);
    int t4329 = t4328 * 128;
    int t4330 = t4327 - t4329;
    int t4331 = t4330 / 128;
    int t4332 = t4330 % 128;
    float t4333 = 0.0;
    for (uint t4334 = 0; t4334 < 256; t4334++) {
      int t4335 = t4334;
      int t4336 = t4335 + t4331;
      int t4337 = t4334 * 128;
      int t4338 = t4337 + t4332;
      float t4339 = memory[328128 + t4336];
      float t4340 = memory[229824 + t4338];
      float t4341 = t4339 * t4340;
      float t4342 = t4333 + t4341;
      t4333 = t4342;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4344 = t4331 * 128;
    int t4345 = t4344 + t4332;
    memory[328384 + t4345] = t4333;
  }
  #pragma clang diagnostic pop
}



// KERNEL 41
// FrameOrder: parallel
// DispatchMode: perFrameScaled(4)
kernel void kernel_41(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4909 = frameCount * 4.0;
  if (id >= 0 && id < (uint)(t4909)) {
    int t4347 = id;
    int t4348 = t4347 / 4;
    uint _frameIndex = (uint)(t4348);
    int t4349 = t4348 * 4;
    int t4350 = t4347 - t4349;
    int t4351 = _frameIndex;
    int t4352 = t4351 * 4;
    int t4353 = t4352 + t4350;
    float t4354 = memory[8980160 + t4353];
    int t4355 = _frameIndex;
    int t4356 = t4355 * 4;
    int t4357 = t4356 + t4350;
    float t4358 = memory[18548416 + t4357];
    float t4359 = t4354 * t4358;
    int t4360 = _frameIndex;
    int t4361 = t4360 * 4;
    int t4362 = t4361 + t4350;
    memory[8783552 + t4362] = t4359;
    int t4364 = _frameIndex;
    int t4365 = t4364 * 4;
    int t4366 = t4365 + t4350;
    float t4367 = memory[329408 + t4366];
    int t4368 = _frameIndex;
    int t4369 = t4368 * 4;
    int t4370 = t4369 + t4350;
    float t4371 = memory[18548416 + t4370];
    float t4372 = t4367 * t4371;
  }
  #pragma clang diagnostic pop
}



// KERNEL 42
// FrameOrder: parallel
// DispatchMode: perFrameScaled(256)
kernel void kernel_42(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4910 = frameCount * 256.0;
  if (id >= 0 && id < (uint)(t4910)) {
    int t4373 = id;
    int t4374 = t4373 / 256;
    uint _frameIndex = (uint)(t4374);
    int t4375 = t4374 * 256;
    int t4376 = t4373 - t4375;
    /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=196, axis=1, in=[4], out=[4, 64], inFA=true, outFA=true), value: empty) */
    int t4377 = t4376 / 64;
    int t4378 = t4377 % 4;
    int t4379 = t4378 * 1;
    int t4380 = 0 + t4379;
    float t4381 = (float)t4380;
    int t4382 = _frameIndex;
    int t4383 = t4382 * 4;
    float t4384 = t4383 + t4381;
    int t4385 = (int)t4384;
    float t4386 = memory[8783552 + t4385];
    float t4387 = (float)t4376;
    int t4388 = _frameIndex;
    int t4389 = t4388 * 256;
    float t4390 = t4389 + t4387;
    int t4391 = (int)t4390;
    memory[329408 + t4391] = t4386;
    int t4393 = _frameIndex;
    int t4394 = t4393 * 256;
    int t4395 = t4394 + t4376;
    float t4396 = memory[4523712 + t4395];
    int t4397 = _frameIndex;
    int t4398 = t4397 * 256;
    int t4399 = t4398 + t4376;
    float t4400 = memory[329408 + t4399];
    float t4401 = t4396 * t4400;
    int t4402 = _frameIndex;
    int t4403 = t4402 * 256;
    int t4404 = t4403 + t4376;
    float t4405 = memory[13305536 + t4404];
    int t4406 = _frameIndex;
    int t4407 = t4406 * 256;
    int t4408 = t4407 + t4376;
    float t4409 = memory[329408 + t4408];
    float t4410 = t4405 * t4409;
    int t4411 = _frameIndex;
    int t4412 = t4411 * 256;
    int t4413 = t4412 + t4376;
    memory[9045696 + t4413] = t4410;
  }
  #pragma clang diagnostic pop
}



// KERNEL 43
// FrameOrder: parallel
// DispatchMode: perFrameScaled(16384)
kernel void kernel_43(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4911 = frameCount * 16384.0;
  if (id >= 0 && id < (uint)(t4911)) {
    /* loadGlobal(310) - handled in variable access */
    int t4415 = id;
    int t4416 = t4415 / 16384;
    uint _frameIndex = (uint)(t4416);
    int t4417 = t4416 * 16384;
    int t4418 = t4415 - t4417;
    float t4419 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 64.0) * 64.0);
    float t4420 = t4419 < 0.0;
    float t4421 = t4419 + 64.0;
    float t4422 = metal::select(t4419, t4421, t4420 > 0.0);
    float t4423 = metal::floor(t4422);
    float t4424 = t4423 + 1.0;
    float t4425 = t4424 >= 64.0;
    float t4426 = metal::select(t4424, 0.0, t4425 > 0.0);
    float t4427 = t4422 - t4423;
    int t4428 = _frameIndex;
    memory[98752 + t4428] = t4423;
    memory[295360 + t4428] = t4427;
    float t4431 = t4428 + 16384.0;
    int t4432 = (int)t4431;
    memory[98752 + t4432] = t4426;
    float t4434 = 1.0 - t4427;
    float t4435 = t4428 * 256.0;
    for (uint _pr4436 = 0; _pr4436 < 256; _pr4436++) {
      float t4437 = (float)_pr4436;
      float t4438 = t4435 + t4437;
      int t4439 = (int)t4438;
      float t4440 = memory[9045696 + t4439];
      float t4441 = t4435 + t4437;
      float t4442 = t4440 * t4434;
      int t4443 = (int)t4441;
      memory[329408 + t4443] = t4442;
      float t4445 = t4440 * t4427;
      int t4446 = (int)t4441;
      memory[4523712 + t4446] = t4445;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 44
// FrameOrder: sequential
// DispatchMode: staticThreads(16384)
kernel void kernel_44(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 16384) { uint _pr4449 = id;
    int t4450 = _pr4449 / 256;
    int t4451 = t4450 * 256;
    int t4452 = _pr4449 - t4451;
    float t4453 = (float)t4450;
    float t4454 = (float)t4452;
    float t4455 = 0.0;
    for (uint t4456 = 0; t4456 < 16384; t4456++) {
      float t4457 = (float)t4456;
      float t4458 = t4457 < frameCount;
      float t4459 = t4457 * 256.0;
      float t4460 = t4459 + t4454;
      float t4461 = memory[98752 + (int)t4456];
      float t4462 = t4461 - t4453;
      float t4463 = metal::abs(t4462);
      float t4464 = t4463 < 0.5;
      int t4465 = (int)t4460;
      float t4466 = memory[329408 + t4465];
      float t4467 = t4458 * t4464;
      float t4468 = t4467 > 0.0;
      float t4469 = metal::select(0.0, t4466, t4468 > 0.0);
      float t4470 = t4455 + t4469;
      t4455 = t4470;
      float t4471 = t4457 + 16384.0;
      int t4472 = (int)t4471;
      float t4473 = memory[98752 + t4472];
      float t4474 = t4473 - t4453;
      float t4475 = metal::abs(t4474);
      float t4476 = t4475 < 0.5;
      int t4477 = (int)t4460;
      float t4478 = memory[4523712 + t4477];
      float t4479 = t4458 * t4476;
      float t4480 = t4479 > 0.0;
      float t4481 = metal::select(0.0, t4478, t4480 > 0.0);
      float t4482 = t4455 + t4481;
      t4455 = t4482;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4484 = t4453 * 256.0;
    float t4485 = t4484 + t4454;
    int t4486 = (int)t4485;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20647360 + t4486], t4455, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 45
// FrameOrder: parallel
// DispatchMode: perFrameScaled(16384)
kernel void kernel_45(
    constant uint &frameCount [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4912 = frameCount * 16384.0;
  if (id >= 0 && id < (uint)(t4912)) {
    int t4489 = id;
    int t4490 = t4489 / 16384;
    uint _frameIndex = (uint)(t4490);
    int t4491 = t4490 * 16384;
    int t4492 = t4489 - t4491;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0, 2]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([256, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 46
// FrameOrder: parallel
// DispatchMode: staticThreads(1)
kernel void kernel_46(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint t4493 = 0; t4493 < 16384; t4493++) {
      int t4494 = t4493 / 64;
      int t4495 = t4494 * 64;
      int t4496 = t4493 - t4495;
      int t4497 = t4494 * 64;
      int t4498 = t4497 + t4496;
      int t4499 = t4498 / 4096;
      int t4500 = t4499 * 4096;
      int t4501 = t4498 - t4500;
      int t4502 = t4501 / 64;
      int t4503 = t4502 * 64;
      int t4504 = t4501 - t4503;
      int t4505 = t4499 * 64;
      int t4506 = t4502 * 256;
      int t4507 = t4505 + t4506;
      int t4508 = t4507 + t4504;
      float t4509 = memory[20647360 + t4508];
      float t4510 = memory[262592 + (int)t4493];
      float t4511 = t4509 / t4510;
      float t4512 = memory[262592 + (int)t4493];
      float t4513 = memory[262592 + (int)t4493];
      float t4514 = t4512 * t4513;
      float t4515 = 1.0 / t4514;
      int t4516 = t4493 / 64;
      int t4517 = t4516 * 64;
      int t4518 = t4493 - t4517;
      int t4519 = t4516 * 64;
      int t4520 = t4519 + t4518;
      int t4521 = t4520 / 4096;
      int t4522 = t4521 * 4096;
      int t4523 = t4520 - t4522;
      int t4524 = t4523 / 64;
      int t4525 = t4524 * 64;
      int t4526 = t4523 - t4525;
      int t4527 = t4521 * 64;
      int t4528 = t4524 * 256;
      int t4529 = t4527 + t4528;
      int t4530 = t4529 + t4526;
      float t4531 = memory[20647360 + t4530];
      float t4532 = t4531 * -1.0;
      float t4533 = t4532 * t4515;
      float t4534 = t4511 + t4533;
      float t4535 = memory[278976 + (int)t4493];
      float t4536 = metal::exp(t4535);
      float t4537 = t4536 * t4533;
      float t4538 = -1.0 * t4537;
      memory[295360 + (int)t4493] = t4538;
      float t4540 = memory[311744 + (int)t4493];
      float t4541 = t4540 * t4537;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t4542 = 0; t4542 < 64; t4542++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=217, axis=0, in=[256, 64], out=[64], inFA=false, outFA=false), value: empty) */
      float t4543 = 0.0;
      int t4544 = t4542;
      int t4545 = t4544;
      int t4546 = t4542 - t4545;
      int t4547 = t4544;
      int t4548 = t4547;
      for (uint t4549 = 0; t4549 < 256; t4549++) {
        int t4550 = t4549 * 64;
        int t4551 = t4548 + t4550;
        float t4552 = memory[295360 + t4551];
        float t4553 = t4543 + t4552;
        t4543 = t4553;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[328128 + (int)t4542] = t4543;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 47
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 32, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_47(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4556 = gid.y;
    int t4557 = gid.x;
    int t4558 = gid.z;
    metal::simdgroup_float8x8 t4559 = metal::simdgroup_float8x8(0);
    for (uint t4560 = 0; t4560 < 8; t4560++) {
      int t4561 = t4556 * 512;
      int t4562 = t4561;
      int t4563 = t4560 * 8;
      int t4564 = t4562 + t4563;
      metal::simdgroup_float8x8 t4565 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4565, &memory[295360 + t4564], 64);
      int t4566 = t4557 * 512;
      int t4567 = t4566;
      int t4568 = t4560 * 8;
      int t4569 = t4567 + t4568;
      metal::simdgroup_float8x8 t4570 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4570, &memory[17024 + t4569], 64, ulong2(0, 0), true);
      metal::simdgroup_multiply_accumulate(t4559, t4565, t4570, t4559);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4572 = t4556 * 1024;
    int t4573 = t4572;
    int t4574 = t4557 * 8;
    int t4575 = t4573 + t4574;
    metal::simdgroup_store(t4559, &memory[98752 + t4575], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 48
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 8, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_48(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4577 = gid.y;
    int t4578 = gid.x;
    int t4579 = gid.z;
    metal::simdgroup_float8x8 t4580 = metal::simdgroup_float8x8(0);
    for (uint t4581 = 0; t4581 < 32; t4581++) {
      int t4582 = t4581 * 512;
      int t4583 = t4582;
      int t4584 = t4577 * 8;
      int t4585 = t4583 + t4584;
      metal::simdgroup_float8x8 t4586 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4586, &memory[295360 + t4585], 64, ulong2(0, 0), true);
      int t4587 = t4581 * 1024;
      int t4588 = t4587;
      int t4589 = t4578 * 8;
      int t4590 = t4588 + t4589;
      metal::simdgroup_float8x8 t4591 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4591, &memory[229824 + t4590], 128);
      metal::simdgroup_multiply_accumulate(t4580, t4586, t4591, t4580);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4593 = t4577 * 1024;
    int t4594 = t4593;
    int t4595 = t4578 * 8;
    int t4596 = t4594 + t4595;
    metal::simdgroup_store(t4580, &memory[262592 + t4596], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 49
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_49(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t4598 = id;
    int t4599 = t4598 / 32768;
    uint _frameIndex = (uint)(t4599);
    int t4600 = t4599 * 32768;
    int t4601 = t4598 - t4600;
    float t4602 = memory[8718016 + t4601];
    float t4603 = memory[98752 + t4601];
    float t4604 = t4602 + t4603;
    memory[8849088 + t4601] = t4604;
    float t4606 = memory[197056 + t4601];
    float t4607 = metal::tanh(t4606);
    float t4608 = t4607 * t4607;
    float t4609 = 1.0 - t4608;
    memory[8783552 + t4601] = t4609;
    float t4611 = t4609 * t4604;
    memory[8914624 + t4601] = t4611;
  }
  #pragma clang diagnostic pop
}



// KERNEL 50
// FrameOrder: parallel
// DispatchMode: staticThreads(128)
kernel void kernel_50(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(128)) {
    int t4613 = id;
    int t4614 = t4613 / 128;
    uint _frameIndex = (uint)(t4614);
    int t4615 = t4614 * 128;
    int t4616 = t4613 - t4615;
    float t4617 = 0.0;
    for (uint t4618 = 0; t4618 < 256; t4618++) {
      int t4619 = t4618 * 128;
      int t4620 = t4619 + t4616;
      float t4621 = memory[8783552 + t4620];
      int t4622 = t4618 * 128;
      int t4623 = t4622 + t4616;
      float t4624 = memory[8849088 + t4623];
      float t4625 = t4621 * t4624;
      float t4626 = t4617 + t4625;
      t4617 = t4626;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[328640 + t4616] = t4617;
  }
  #pragma clang diagnostic pop
}



// KERNEL 51
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 32, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_51(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4633 = gid.y;
    int t4634 = gid.x;
    int t4635 = gid.z;
    metal::simdgroup_float8x8 t4636 = metal::simdgroup_float8x8(0);
    for (uint t4637 = 0; t4637 < 16; t4637++) {
      int t4638 = t4633 * 1024;
      int t4639 = t4638;
      int t4640 = t4637 * 8;
      int t4641 = t4639 + t4640;
      metal::simdgroup_float8x8 t4642 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4642, &memory[8914624 + t4641], 128);
      int t4643 = t4634 * 1024;
      int t4644 = t4643;
      int t4645 = t4637 * 8;
      int t4646 = t4644 + t4645;
      metal::simdgroup_float8x8 t4647 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4647, &memory[512 + t4646], 128, ulong2(0, 0), true);
      metal::simdgroup_multiply_accumulate(t4636, t4642, t4647, t4636);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4649 = t4633 * 1024;
    int t4650 = t4649;
    int t4651 = t4634 * 8;
    int t4652 = t4650 + t4651;
    metal::simdgroup_store(t4636, &memory[98752 + t4652], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 52
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 16, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_52(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4654 = gid.y;
    int t4655 = gid.x;
    int t4656 = gid.z;
    metal::simdgroup_float8x8 t4657 = metal::simdgroup_float8x8(0);
    for (uint t4658 = 0; t4658 < 32; t4658++) {
      int t4659 = t4658 * 1024;
      int t4660 = t4659;
      int t4661 = t4654 * 8;
      int t4662 = t4660 + t4661;
      metal::simdgroup_float8x8 t4663 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4663, &memory[8914624 + t4662], 128, ulong2(0, 0), true);
      int t4664 = t4658 * 1024;
      int t4665 = t4664;
      int t4666 = t4655 * 8;
      int t4667 = t4665 + t4666;
      metal::simdgroup_float8x8 t4668 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4668, &memory[131520 + t4667], 128);
      metal::simdgroup_multiply_accumulate(t4657, t4663, t4668, t4657);
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4670 = t4654 * 1024;
    int t4671 = t4670;
    int t4672 = t4655 * 8;
    int t4673 = t4671 + t4672;
    metal::simdgroup_store(t4657, &memory[262592 + t4673], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 53
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_53(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t4675 = id;
    int t4676 = t4675 / 32768;
    uint _frameIndex = (uint)(t4676);
    int t4677 = t4676 * 32768;
    int t4678 = t4675 - t4677;
    float t4679 = memory[164288 + t4678];
    float t4680 = metal::tanh(t4679);
    float t4681 = t4680 * t4680;
    float t4682 = 1.0 - t4681;
    memory[8718016 + t4678] = t4682;
    float t4684 = memory[98752 + t4678];
    float t4685 = t4682 * t4684;
    memory[197056 + t4678] = t4685;
  }
  #pragma clang diagnostic pop
}



// KERNEL 54
// FrameOrder: parallel
// DispatchMode: staticThreads(128)
kernel void kernel_54(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(128)) {
    int t4687 = id;
    int t4688 = t4687 / 128;
    uint _frameIndex = (uint)(t4688);
    int t4689 = t4688 * 128;
    int t4690 = t4687 - t4689;
    float t4691 = 0.0;
    for (uint t4692 = 0; t4692 < 256; t4692++) {
      int t4693 = t4692 * 128;
      int t4694 = t4693 + t4690;
      float t4695 = memory[8718016 + t4694];
      int t4696 = t4692 * 128;
      int t4697 = t4696 + t4690;
      float t4698 = memory[98752 + t4697];
      float t4699 = t4695 * t4698;
      float t4700 = t4691 + t4699;
      t4691 = t4700;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[328896 + t4690] = t4691;
  }
  #pragma clang diagnostic pop
}



// KERNEL 55
// FrameOrder: parallel
// DispatchMode: perFrameScaled(768)
kernel void kernel_55(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4913 = frameCount * 768.0;
  if (id >= 0 && id < (uint)(t4913)) {
    int t4707 = id;
    int t4708 = t4707 / 768;
    uint _frameIndex = (uint)(t4708);
    int t4709 = t4708 * 768;
    int t4710 = t4707 - t4709;
    int t4711 = t4710 / 3;
    int t4712 = t4710 % 3;
    float t4713 = 0.0;
    for (uint t4714 = 0; t4714 < 128; t4714++) {
      int t4715 = t4711 * 128;
      int t4716 = t4715 + t4714;
      int t4717 = t4712 * 128;
      int t4718 = t4717 + t4714;
      float t4719 = memory[197056 + t4716];
      float t4720 = memory[0 + t4718];
      float t4721 = t4719 * t4720;
      float t4722 = t4713 + t4721;
      t4713 = t4722;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4724 = t4711 * 3;
    int t4725 = t4724 + t4712;
    memory[262592 + t4725] = t4713;
  }
  #pragma clang diagnostic pop
}



// KERNEL 56
// FrameOrder: parallel
// DispatchMode: perFrameScaled(384)
kernel void kernel_56(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4914 = frameCount * 384.0;
  if (id >= 0 && id < (uint)(t4914)) {
    int t4727 = id;
    int t4728 = t4727 / 384;
    uint _frameIndex = (uint)(t4728);
    int t4729 = t4728 * 384;
    int t4730 = t4727 - t4729;
    int t4731 = t4730 / 3;
    int t4732 = t4730 % 3;
    float t4733 = 0.0;
    for (uint t4734 = 0; t4734 < 256; t4734++) {
      int t4735 = t4734 * 128;
      int t4736 = t4735 + t4731;
      int t4737 = t4734 * 3;
      int t4738 = t4737 + t4732;
      float t4739 = memory[197056 + t4736];
      float t4740 = memory[25538 + t4738];
      float t4741 = t4739 * t4740;
      float t4742 = t4733 + t4741;
      t4733 = t4742;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4744 = t4731 * 3;
    int t4745 = t4744 + t4732;
    memory[262592 + t4745] = t4733;
  }
  #pragma clang diagnostic pop
}



// KERNEL 57
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_57(
    constant uint &frameCount [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 128, 3]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([128, 3]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 58
// FrameOrder: sequential
// DispatchMode: staticThreads(384)
kernel void kernel_58(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 384) { uint _pr4747 = id;
    float t4748 = (float)_pr4747;
    float t4749 = (t4748 * 0.0078125);
    float t4750 = metal::floor(t4749);
    float t4751 = t4750 * 128.0;
    float t4752 = t4748 - t4751;
    float t4753 = t4752 * 3.0;
    float t4754 = t4750 + t4753;
    int t4755 = (int)t4754;
    float t4756 = memory[262592 + t4755];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20663744 + (int)_pr4747], t4756, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 59
// FrameOrder: sequential
// DispatchMode: staticThreads(128)
kernel void kernel_59(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 128) { uint _pr4759 = id;
    float t4760 = (float)_pr4759;
    float t4761 = (t4760 * 0.0078125);
    float t4762 = metal::floor(t4761);
    float t4763 = t4762 * 128.0;
    float t4764 = t4760 - t4763;
    int t4765 = (int)t4764;
    float t4766 = memory[328896 + t4765];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20664128 + (int)_pr4759], t4766, metal::memory_order_relaxed);
  }
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 128) { uint _pr4769 = id;
    float t4770 = (float)_pr4769;
    float t4771 = (t4770 * 0.0078125);
    float t4772 = metal::floor(t4771);
    float t4773 = t4772 * 128.0;
    float t4774 = t4770 - t4773;
    int t4775 = (int)t4774;
    float t4776 = memory[328640 + t4775];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20664256 + (int)_pr4769], t4776, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 60
// FrameOrder: sequential
// DispatchMode: staticThreads(64)
kernel void kernel_60(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr4779 = id;
    float t4780 = (float)_pr4779;
    float t4781 = (t4780 * 0.015625);
    float t4782 = metal::floor(t4781);
    float t4783 = t4782 * 64.0;
    float t4784 = t4780 - t4783;
    int t4785 = (int)t4784;
    float t4786 = memory[328128 + t4785];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20664384 + (int)_pr4779], t4786, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 61
// FrameOrder: sequential
// DispatchMode: staticThreads(128)
kernel void kernel_61(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 128) { uint _pr4789 = id;
    float t4790 = (float)_pr4789;
    float t4791 = t4790;
    float t4792 = metal::floor(t4791);
    float t4793 = t4792;
    float t4794 = t4790 - t4793;
    int t4795 = (int)t4792;
    float t4796 = memory[328384 + t4795];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20664448 + (int)_pr4789], t4796, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 62
// FrameOrder: sequential
// DispatchMode: staticThreads(1)
kernel void kernel_62(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 1) { uint _pr4799 = id;
    float t4800 = (float)_pr4799;
    float t4801 = t4800;
    float t4802 = metal::floor(t4801);
    float t4803 = t4802;
    float t4804 = t4800 - t4803;
    float t4805 = memory[329152 + (int)0.0];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[20664576 + (int)_pr4799], t4805, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 63
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_63(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2326) - handled in variable access */
  }
  #pragma clang diagnostic pop
}



// KERNEL 64
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 16, tilesN: 16, depth: Optional(256))
#include <metal_simdgroup_matrix>
kernel void kernel_64(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4808 = gid.y;
    int t4809 = gid.x;
    int t4810 = gid.z;
    metal::simdgroup_float8x8 t4811 = metal::simdgroup_float8x8(0);
    int t4812 = (int)frameCount;
    int t4813 = t4810 * 64;
    for (uint t4814 = 0; t4814 < 64; t4814++) {
      int t4815 = t4813 + t4814;
      int t4816 = t4815 < t4812;
      if (t4816) {
        for (uint t4818 = 0; t4818 < 32; t4818++) {
          int t4819 = t4818 * 1024;
          int t4820 = t4819;
          int t4821 = t4808 * 8;
          int t4822 = t4820 + t4821;
          metal::simdgroup_float8x8 t4823 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4823, &memory[8914624 + t4822], 128, ulong2(0, 0), true);
          int t4824 = t4818 * 1024;
          int t4825 = t4824;
          int t4826 = t4809 * 8;
          int t4827 = t4825 + t4826;
          metal::simdgroup_float8x8 t4828 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4828, &memory[131520 + t4827], 128);
          metal::simdgroup_multiply_accumulate(t4811, t4823, t4828, t4811);
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4832 = t4810 * 16384;
    int t4833 = t4808 * 1024;
    int t4834 = t4832 + t4833;
    int t4835 = t4809 * 8;
    int t4836 = t4834 + t4835;
    metal::simdgroup_store(t4811, &memory[329408 + t4836], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 65
// FrameOrder: sequential
// DispatchMode: staticThreads(16384)
kernel void kernel_65(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 16384) { uint _pr4838 = id;
    float t4839 = 0.0;
    for (uint t4840 = 0; t4840 < 256; t4840++) {
      int t4841 = t4840 * 16384;
      int t4842 = t4841 + _pr4838;
      float t4843 = memory[329408 + t4842];
      float t4844 = t4839 + t4843;
      t4839 = t4844;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4846 = _pr4838 / 128;
    int t4847 = _pr4838 % 128;
    int t4848 = t4847 * 128;
    int t4849 = t4848 + t4846;
    memory[262592 + t4849] = t4839;
  }
  #pragma clang diagnostic pop
}



// KERNEL 66
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_66(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2326) - handled in variable access */
  }
  #pragma clang diagnostic pop
}



// KERNEL 67
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 8, tilesN: 16, depth: Optional(256))
#include <metal_simdgroup_matrix>
kernel void kernel_67(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4852 = gid.y;
    int t4853 = gid.x;
    int t4854 = gid.z;
    metal::simdgroup_float8x8 t4855 = metal::simdgroup_float8x8(0);
    int t4856 = (int)frameCount;
    int t4857 = t4854 * 64;
    for (uint t4858 = 0; t4858 < 64; t4858++) {
      int t4859 = t4857 + t4858;
      int t4860 = t4859 < t4856;
      if (t4860) {
        for (uint t4862 = 0; t4862 < 32; t4862++) {
          int t4863 = t4862 * 512;
          int t4864 = t4863;
          int t4865 = t4852 * 8;
          int t4866 = t4864 + t4865;
          metal::simdgroup_float8x8 t4867 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4867, &memory[295360 + t4866], 64, ulong2(0, 0), true);
          int t4868 = t4862 * 1024;
          int t4869 = t4868;
          int t4870 = t4853 * 8;
          int t4871 = t4869 + t4870;
          metal::simdgroup_float8x8 t4872 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4872, &memory[229824 + t4871], 128);
          metal::simdgroup_multiply_accumulate(t4855, t4867, t4872, t4855);
        } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4876 = t4854 * 8192;
    int t4877 = t4852 * 1024;
    int t4878 = t4876 + t4877;
    int t4879 = t4853 * 8;
    int t4880 = t4878 + t4879;
    metal::simdgroup_store(t4855, &memory[329408 + t4880], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 68
// FrameOrder: sequential
// DispatchMode: staticThreads(8192)
kernel void kernel_68(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 8192) { uint _pr4882 = id;
    float t4883 = 0.0;
    for (uint t4884 = 0; t4884 < 256; t4884++) {
      int t4885 = t4884 * 8192;
      int t4886 = t4885 + _pr4882;
      float t4887 = memory[329408 + t4886];
      float t4888 = t4883 + t4887;
      t4883 = t4888;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4890 = _pr4882 / 128;
    int t4891 = _pr4882 % 128;
    int t4892 = t4891 * 64;
    int t4893 = t4892 + t4890;
    memory[262592 + t4893] = t4883;
  }
  #pragma clang diagnostic pop
}



// KERNEL 69
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_69(
    device float *outputs [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2326) - handled in variable access */
    outputs[0 * frameCount + id] = t[8*frameCount + id];
  }
  #pragma clang diagnostic pop
}

