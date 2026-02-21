// KERNEL 0
// FrameOrder: parallel
// DispatchMode: perFrameScaled(65536)
kernel void kernel_0(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4923 = frameCount * 65536.0;
  if (id >= 0 && id < (uint)(t4923)) {
    int t110 = id;
    int t111 = t110 / 65536;
    uint _frameIndex = (uint)(t111);
    int t112 = t111 * 65536;
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
    memory[165824 + t128] = t116;
  }
  #pragma clang diagnostic pop
}



// KERNEL 1
// FrameOrder: parallel
// DispatchMode: staticThreads(65536)
kernel void kernel_1(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(65536)) {
    int t130 = id;
    int t131 = t130 / 65536;
    uint _frameIndex = (uint)(t131);
    int t132 = t131 * 65536;
    int t133 = t130 - t132;
    float t134 = memory[165824 + t133];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t135 = t133 / 128;
    int t136 = t135 * 128;
    int t137 = t133 - t136;
    int t138 = t137;
    float t139 = memory[384 + t138];
    float t140 = t134 + t139;
    memory[296896 + t133] = t140;
    float t142 = metal::tanh(t140);
    memory[231360 + t133] = t142;
  }
  #pragma clang diagnostic pop
}



// KERNEL 2
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 64, tilesN: 16, depth: nil)
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
      metal::simdgroup_float8x8 t153 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t153, &memory[231360 + t152], 128);
      int t154 = t148 * 1024;
      int t155 = t154;
      int t156 = t145 * 8;
      int t157 = t155 + t156;
      metal::simdgroup_float8x8 t158 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t158, &memory[512 + t157], 128);
      metal::simdgroup_multiply_accumulate(t147, t153, t158, t147);
    }
    int t160 = t144 * 1024;
    int t161 = t160;
    int t162 = t145 * 8;
    int t163 = t161 + t162;
    metal::simdgroup_store(t147, &memory[165824 + t163], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 3
// FrameOrder: parallel
// DispatchMode: staticThreads(65536)
kernel void kernel_3(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(65536)) {
    int t165 = id;
    int t166 = t165 / 65536;
    uint _frameIndex = (uint)(t166);
    int t167 = t166 * 65536;
    int t168 = t165 - t167;
    float t169 = memory[165824 + t168];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t170 = t168 / 128;
    int t171 = t170 * 128;
    int t172 = t168 - t171;
    int t173 = t172;
    float t174 = memory[16896 + t173];
    float t175 = t169 + t174;
    memory[362432 + t168] = t175;
    float t177 = metal::tanh(t175);
    memory[427968 + t168] = t177;
  }
  #pragma clang diagnostic pop
}



// KERNEL 4
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 64, tilesN: 8, depth: nil)
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
      metal::simdgroup_float8x8 t188 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t188, &memory[427968 + t187], 128);
      int t189 = t183 * 512;
      int t190 = t189;
      int t191 = t180 * 8;
      int t192 = t190 + t191;
      metal::simdgroup_float8x8 t193 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t193, &memory[17024 + t192], 64);
      metal::simdgroup_multiply_accumulate(t182, t188, t193, t182);
    }
    int t195 = t179 * 512;
    int t196 = t195;
    int t197 = t180 * 8;
    int t198 = t196 + t197;
    metal::simdgroup_store(t182, &memory[165824 + t198], 64);
  }
  #pragma clang diagnostic pop
}



// KERNEL 5
// FrameOrder: parallel
// DispatchMode: staticThreads(32768)
kernel void kernel_5(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(32768)) {
    int t200 = id;
    int t201 = t200 / 32768;
    uint _frameIndex = (uint)(t201);
    int t202 = t201 * 32768;
    int t203 = t200 - t202;
    float t204 = memory[165824 + t203];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t205 = t203 / 64;
    int t206 = t205 * 64;
    int t207 = t203 - t206;
    int t208 = t207;
    float t209 = memory[25216 + t208];
    float t210 = t204 + t209;
    memory[493504 + t203] = t210;
    float t212 = t210 * -1.0;
    memory[591808 + t203] = t212;
    float t214 = metal::exp(t212);
    float t215 = 1.0 + t214;
    memory[559040 + t203] = t215;
    float t217 = 1.0 / t215;
    memory[526272 + t203] = t217;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 128]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 6
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_6(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4924 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4924)) {
    int t219 = id;
    int t220 = t219 / 512;
    uint _frameIndex = (uint)(t220);
    int t221 = t220 * 512;
    int t222 = t219 - t221;
    int t223 = t222;
    int t224 = t222 % 1;
    float t225 = 0.0;
    for (uint t226 = 0; t226 < 128; t226++) {
      int t227 = t223 * 128;
      int t228 = t227 + t226;
      int t229 = t226;
      int t230 = t229 + t224;
      float t231 = memory[427968 + t228];
      float t232 = memory[25280 + t230];
      float t233 = t231 * t232;
      float t234 = t225 + t233;
      t225 = t234;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t236 = t223;
    int t237 = t236 + t224;
    memory[165824 + t237] = t225;
  }
  #pragma clang diagnostic pop
}



// KERNEL 7
// FrameOrder: parallel
// DispatchMode: staticThreads(512)
kernel void kernel_7(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(512)) {
    int t239 = id;
    int t240 = t239 / 512;
    uint _frameIndex = (uint)(t240);
    int t241 = t240 * 512;
    int t242 = t239 - t241;
    float t243 = memory[165824 + t242];
    /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
    int t244 = t242;
    int t245 = t244;
    int t246 = t242 - t245;
    float t247 = memory[25408 + (int)0.0];
    float t248 = t243 + t247;
    memory[624576 + t242] = t248;
    float t250 = t248 * -1.0;
    memory[625600 + t242] = t250;
    float t252 = metal::exp(t250);
    float t253 = 1.0 + t252;
    memory[626112 + t242] = t253;
    float t255 = 1.0 / t253;
    memory[625088 + t242] = t255;
  }
  #pragma clang diagnostic pop
}



// KERNEL 8
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_8(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4925 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4925)) {
    int t257 = id;
    int t258 = t257 / 512;
    uint _frameIndex = (uint)(t258);
    int t259 = t258 * 512;
    int t260 = t257 - t259;
    int t261 = t260;
    int t262 = t260 % 1;
    float t263 = 0.0;
    for (uint t264 = 0; t264 < 128; t264++) {
      int t265 = t261 * 128;
      int t266 = t265 + t264;
      int t267 = t264;
      int t268 = t267 + t262;
      float t269 = memory[427968 + t266];
      float t270 = memory[25409 + t268];
      float t271 = t269 * t270;
      float t272 = t263 + t271;
      t263 = t272;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t274 = t261;
    int t275 = t274 + t262;
    memory[165824 + t275] = t263;
  }
  #pragma clang diagnostic pop
}



// KERNEL 9
// FrameOrder: parallel
// DispatchMode: staticThreads(512)
kernel void kernel_9(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(512)) {
    int t277 = id;
    int t278 = t277 / 512;
    uint _frameIndex = (uint)(t278);
    int t279 = t278 * 512;
    int t280 = t277 - t279;
    float t281 = memory[165824 + t280];
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
    memory[626624 + t280] = t290;
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
      t[0*frameCount + i] = memory[41427713];
      float t293 = t[0*frameCount + i] + 0.0038454495;
      float t294 = metal::select(t293, 0.0, 0.0 > 0.0);
      float t295 = t294;
      float t296 = (t295 * 0.015873017);
      float t297 = metal::floor(t296);
      float t298 = t297 * 63.0;
      float t299 = t294 - t298;
      memory[41427713] = t299;
      float t301 = t299 >= 63.0;
      if (t301) {
        float t303 = t299 - 63.0;
        memory[41427713] = t303;
      }
      if (0.0) {
        memory[41427713] = 0.0;
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
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([8, 64, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0, 2]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([8, 64]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 12
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_12(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4926 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4926)) {
    /* loadGlobal(310) - handled in variable access */
    int t311 = id;
    int t312 = t311 / 512;
    uint _frameIndex = (uint)(t312);
    int t313 = t312 * 512;
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
    float t325 = t312 * 512.0;
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
    float t336 = memory[526272 + t335];
    float t337 = t322 * 64.0;
    float t338 = t329 * 4096.0;
    float t339 = t337 + t338;
    float t340 = t339 + t330;
    int t341 = (int)t340;
    float t342 = memory[526272 + t341];
    float t343 = t324 * t336;
    float t344 = t323 * t342;
    float t345 = t343 + t344;
    float t346 = (float)t314;
    float t347 = t325 + t346;
    int t348 = (int)t347;
    memory[627136 + t348] = t345;
    int t350 = (int)t347;
    memory[9015744 + t350] = t345;
  }
  #pragma clang diagnostic pop
}



// KERNEL 13
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_13(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4927 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4927)) {
    /* loadGlobal(310) - handled in variable access */
    int t352 = id;
    int t353 = t352 / 8;
    uint _frameIndex = (uint)(t353);
    int t354 = t353 * 8;
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
    float t366 = t353 * 8.0;
    float t367 = (float)t355;
    float t368 = t367 * 64.0;
    float t369 = t360 + t368;
    int t370 = (int)t369;
    float t371 = memory[625088 + t370];
    float t372 = t367 * 64.0;
    float t373 = t363 + t372;
    int t374 = (int)t373;
    float t375 = memory[625088 + t374];
    float t376 = t365 * t371;
    float t377 = t364 * t375;
    float t378 = t376 + t377;
    float t379 = (float)t355;
    float t380 = t366 + t379;
    int t381 = (int)t380;
    memory[627136 + t381] = t378;
    int t383 = (int)t380;
    memory[17666496 + t383] = t378;
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
    float t395 = t353 * 8.0;
    float t396 = (float)t355;
    float t397 = t389 * 8.0;
    float t398 = t397 + t396;
    int t399 = (int)t398;
    float t400 = memory[27074 + t399];
    float t401 = t392 * 8.0;
    float t402 = t401 + t396;
    int t403 = (int)t402;
    float t404 = memory[27074 + t403];
    float t405 = t394 * t400;
    float t406 = t393 * t404;
    float t407 = t405 + t406;
    float t408 = (float)t355;
    float t409 = t395 + t408;
    int t410 = (int)t409;
    memory[17404352 + t410] = t407;
    int t412 = (int)t409;
    memory[17797568 + t412] = t407;
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
    float t424 = t353 * 8.0;
    float t425 = (float)t355;
    float t426 = t418 * 8.0;
    float t427 = t426 + t425;
    int t428 = (int)t427;
    float t429 = memory[27586 + t428];
    float t430 = t421 * 8.0;
    float t431 = t430 + t425;
    int t432 = (int)t431;
    float t433 = memory[27586 + t432];
    float t434 = t423 * t429;
    float t435 = t422 * t433;
    float t436 = t434 + t435;
    float t437 = (float)t355;
    float t438 = t424 + t437;
    int t439 = (int)t438;
    memory[17535424 + t439] = t436;
    int t441 = (int)t438;
    memory[17928640 + t441] = t436;
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([8, 1]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mexpandView[0m([8, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 14
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_14(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4928 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4928)) {
    int t443 = id;
    int t444 = t443 / 512;
    uint _frameIndex = (uint)(t444);
    int t445 = t444 * 512;
    int t446 = t443 - t445;
    int t447 = t446 / 64;
    int t448 = t447 * 64;
    int t449 = t446 - t448;
    int t450 = _frameIndex;
    int t451 = t450 * 8;
    int t452 = t451 + t447;
    float t453 = memory[17797568 + t452];
    float t454 = memory[28098 + t446];
    float t455 = t453 * t454;
    int t456 = _frameIndex;
    int t457 = t456 * 512;
    int t458 = t457 + t446;
    memory[627136 + t458] = t455;
  }
  #pragma clang diagnostic pop
}



// KERNEL 15
// FrameOrder: sequential
// DispatchMode: fixedWithFrameLoop(512)
kernel void kernel_15(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(512)) {
    for (uint i = 0; i < frameCount; i += 1) {
      int t460 = id;
      int t461 = i;
      int t462 = t461 * 512;
      int t463 = t462 + t460;
      float t464 = memory[627136 + t463];
      float t465 = (t464 * 6.25e-05);
      float t466 = memory[625088 + t460];
      float t467 = t466 + t465;
      float t468 = metal::select(t467, 0.0, 0.0 > 0.0);
      float t469 = metal::floor(t468);
      float t470 = t468 - t469;
      float t471 = t470 >= 1.0;
      float t472 = t470 - 1.0;
      float t473 = metal::select(t470, t472, t471 > 0.0);
      float t474 = metal::select(t473, 0.0, 0.0 > 0.0);
      memory[625088 + t460] = t474;
      int t476 = i;
      int t477 = t476 * 512;
      int t478 = t477 + t460;
      memory[18059712 + t478] = t466;
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
    for (uint t480 = 0; t480 < 512; t480++) {
      int t481 = id;
      int t482 = t481 * 512;
      int t483 = t482 + t480;
      float t484 = memory[18059712 + t483];
      float t485 = t484 * 6.283185;
      float t486 = metal::sin(t485);
      int t487 = id;
      int t488 = t487 * 512;
      int t489 = t488 + t480;
      memory[627136 + t489] = t486;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t491 = 0; t491 < 8; t491++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=91, axis=1, in=[8, 64], out=[8], inFA=true, outFA=true), value: empty) */
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
        int t504 = t503 * 512;
        int t505 = t504 + t502;
        float t506 = memory[627136 + t505];
        int t507 = t493 * 64;
        int t508 = t507 + t498;
        int t509 = id;
        int t510 = t509 * 512;
        int t511 = t510 + t508;
        float t512 = memory[9015744 + t511];
        float t513 = t506 * t512;
        float t514 = t492 + t513;
        t492 = t514;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t516 = id;
      int t517 = t516 * 8;
      int t518 = t517 + t491;
      memory[17797568 + t518] = t492;
      int t520 = id;
      int t521 = t520 * 8;
      int t522 = t521 + t491;
      float t523 = memory[17797568 + t522];
      int t524 = id;
      int t525 = t524 * 8;
      int t526 = t525 + t491;
      float t527 = memory[17928640 + t526];
      float t528 = t523 * t527;
      int t529 = id;
      int t530 = t529 * 8;
      int t531 = t530 + t491;
      memory[26448320 + t531] = t528;
      int t533 = id;
      int t534 = t533 * 8;
      int t535 = t534 + t491;
      float t536 = memory[17666496 + t535];
      float t537 = t528 * t536;
      int t538 = id;
      int t539 = t538 * 8;
      int t540 = t539 + t491;
      memory[17535424 + t540] = t537;
      float t542 = t537 * 0.015625;
      int t543 = id;
      int t544 = t543 * 8;
      int t545 = t544 + t491;
      memory[26579392 + t545] = t542;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([8, 64]), value: empty) */
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
      float t558 = t557 * 8.0;
      float t559 = (float)t491;
      float t560 = t559 * 64.0;
      float t561 = t551 + t560;
      int t562 = (int)t561;
      float t563 = memory[626624 + t562];
      float t564 = t559 * 64.0;
      float t565 = t554 + t564;
      int t566 = (int)t565;
      float t567 = memory[626624 + t566];
      float t568 = t556 * t563;
      float t569 = t555 * t567;
      float t570 = t568 + t569;
      float t571 = (float)t491;
      float t572 = t558 + t571;
      int t573 = (int)t572;
      memory[17404352 + t573] = t570;
      memory[625088 + (int)t491] = t570;
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
      t[2*frameCount + i] = memory[41427714];
      float t577 = t[2*frameCount + i] + 1.0;
      float t578 = metal::select(t577, 0.0, 0.0 > 0.0);
      float t579 = t578;
      float t580 = (t579 * 6.1035156e-05);
      float t581 = metal::floor(t580);
      float t582 = t581 * 16384.0;
      float t583 = t578 - t582;
      memory[41427714] = t583;
      float t585 = t583 >= 16384.0;
      if (t585) {
        float t587 = t583 - 16384.0;
        memory[41427714] = t587;
      }
      if (0.0) {
        memory[41427714] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 18
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_18(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4929 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4929)) {
    /* loadGlobal(576) - handled in variable access */
    int t593 = id;
    int t594 = t593 / 8;
    uint _frameIndex = (uint)(t594);
    int t595 = t594 * 8;
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
    float t607 = t594 * 8.0;
    float t608 = (float)t596;
    float t609 = t601 * 8.0;
    float t610 = t609 + t608;
    int t611 = (int)t610;
    float t612 = memory[28610 + t611];
    float t613 = t604 * 8.0;
    float t614 = t613 + t608;
    int t615 = (int)t614;
    float t616 = memory[28610 + t615];
    float t617 = t606 * t612;
    float t618 = t605 * t616;
    float t619 = t617 + t618;
    float t620 = (float)t596;
    float t621 = t607 + t620;
    int t622 = (int)t621;
    memory[17404352 + t622] = t619;
    int t624 = (int)t621;
    memory[18059712 + t624] = t619;
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
      t[3*frameCount + i] = memory[41427715];
      float t639 = t[3*frameCount + i] + 1.0;
      float t640 = metal::select(t639, 0.0, 0.0 > 0.0);
      float t641 = t640;
      float t642 = (t641 * 0.0078125);
      float t643 = metal::floor(t642);
      float t644 = t643 * 128.0;
      float t645 = t640 - t644;
      memory[41427715] = t645;
      float t647 = t645 >= 128.0;
      if (t647) {
        float t649 = t645 - 128.0;
        memory[41427715] = t649;
      }
      if (0.0) {
        memory[41427715] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 20
// FrameOrder: parallel
// DispatchMode: perFrameScaledThreadgroup1(8)
kernel void kernel_20(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4930 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4930)) {
    /* loadGlobal(638) - handled in variable access */
    int t655 = id;
    int t656 = t655 / 8;
    uint _frameIndex = (uint)(t656);
    int t657 = t656 * 8;
    int t658 = t655 - t657;
    threadgroup float scratch_0[512];
    threadgroup float scratch_1[512];
    threadgroup float scratch_2[511];
    threadgroup float scratch_3[511];
    float t659 = t[3*frameCount + _frameIndex] == 0.0;
    if (t659) {
      for (uint t661 = 0; t661 < 511; t661++) {
        float t662 = memory[160194 + (int)t661];
        scratch_2[(int)t661] = t662;
        float t664 = memory[160705 + (int)t661];
        scratch_3[(int)t661] = t664;
      }
      int t667 = t656 / 128;
      int t668 = t667 * 8;
      int t669 = t668 + t658;
      int t670 = t669 * 1024;
      int t671 = t669 * 257;
      for (uint t672 = 0; t672 < 512; t672++) {
        float t673 = (float)t672;
        float t674 = memory[159682 + (int)t672];
        float t675 = (float)t656;
        float t676 = t675 - 511.0;
        float t677 = t676 + t673;
        float t678 = t677 >= 0.0;
        float t679 = t677 < frameCount;
        float t680 = t678 * t679;
        float t681 = frameCount - 1.0;
        float t682 = metal::min(t677, t681);
        float t683 = metal::max(0.0, t682);
        int t684 = (int)t683;
        int t685 = t684 * 8;
        int t686 = t685 + t658;
        float t687 = memory[26579392 + t686];
        float t688 = metal::select(0.0, t687, t680 > 0.0);
        float t689 = t688 * t674;
        scratch_0[(int)t672] = t689;
        scratch_1[(int)t672] = 0.0;
      }
      for (uint t693 = 0; t693 < 512; t693++) {
        float t694 = memory[161216 + (int)t693];
        float t695 = (float)t693;
        float t696 = t695 < t694;
        int t697 = (int)t694;
        float t698 = scratch_0[(int)t693];
        float t699 = scratch_1[(int)t693];
        float t700 = scratch_0[t697];
        float t701 = scratch_1[t697];
        float t702 = metal::select(t698, t700, t696 > 0.0);
        float t703 = metal::select(t699, t701, t696 > 0.0);
        float t704 = metal::select(t700, t698, t696 > 0.0);
        float t705 = metal::select(t701, t699, t696 > 0.0);
        scratch_0[(int)t693] = t702;
        scratch_1[(int)t693] = t703;
        scratch_0[t697] = t704;
        scratch_1[t697] = t705;
      }
      for (uint t711 = 0; t711 < 256; t711++) {
        float t712 = (float)t711;
        float t713 = t712;
        float t714 = metal::floor(t713);
        float t715 = t714;
        float t716 = t712 - t715;
        float t717 = t714 * 2.0;
        float t718 = t717 + t716;
        float t719 = t718 + 1.0;
        int t720 = (int)t716;
        int t721 = t720;
        float t722 = scratch_2[t721];
        float t723 = scratch_3[t721];
        float t724 = 0.0 - t723;
        int t725 = (int)t718;
        int t726 = (int)t719;
        float t727 = scratch_0[t725];
        float t728 = scratch_1[t725];
        float t729 = scratch_0[t726];
        float t730 = scratch_1[t726];
        float t731 = t722 * t729;
        float t732 = t724 * t730;
        float t733 = t731 - t732;
        float t734 = t722 * t730;
        float t735 = t724 * t729;
        float t736 = t734 + t735;
        float t737 = t727 + t733;
        scratch_0[t725] = t737;
        float t739 = t728 + t736;
        scratch_1[t725] = t739;
        float t741 = t727 - t733;
        scratch_0[t726] = t741;
        float t743 = t728 - t736;
        scratch_1[t726] = t743;
      }
      for (uint t746 = 0; t746 < 256; t746++) {
        float t747 = (float)t746;
        float t748 = (t747 * 0.5);
        float t749 = metal::floor(t748);
        float t750 = t749 * 2.0;
        float t751 = t747 - t750;
        float t752 = t749 * 4.0;
        float t753 = t752 + t751;
        float t754 = t753 + 2.0;
        int t755 = (int)t751;
        int t756 = 1 + t755;
        float t757 = scratch_2[t756];
        float t758 = scratch_3[t756];
        float t759 = 0.0 - t758;
        int t760 = (int)t753;
        int t761 = (int)t754;
        float t762 = scratch_0[t760];
        float t763 = scratch_1[t760];
        float t764 = scratch_0[t761];
        float t765 = scratch_1[t761];
        float t766 = t757 * t764;
        float t767 = t759 * t765;
        float t768 = t766 - t767;
        float t769 = t757 * t765;
        float t770 = t759 * t764;
        float t771 = t769 + t770;
        float t772 = t762 + t768;
        scratch_0[t760] = t772;
        float t774 = t763 + t771;
        scratch_1[t760] = t774;
        float t776 = t762 - t768;
        scratch_0[t761] = t776;
        float t778 = t763 - t771;
        scratch_1[t761] = t778;
      }
      for (uint t781 = 0; t781 < 256; t781++) {
        float t782 = (float)t781;
        float t783 = (t782 * 0.25);
        float t784 = metal::floor(t783);
        float t785 = t784 * 4.0;
        float t786 = t782 - t785;
        float t787 = t784 * 8.0;
        float t788 = t787 + t786;
        float t789 = t788 + 4.0;
        int t790 = (int)t786;
        int t791 = 3 + t790;
        float t792 = scratch_2[t791];
        float t793 = scratch_3[t791];
        float t794 = 0.0 - t793;
        int t795 = (int)t788;
        int t796 = (int)t789;
        float t797 = scratch_0[t795];
        float t798 = scratch_1[t795];
        float t799 = scratch_0[t796];
        float t800 = scratch_1[t796];
        float t801 = t792 * t799;
        float t802 = t794 * t800;
        float t803 = t801 - t802;
        float t804 = t792 * t800;
        float t805 = t794 * t799;
        float t806 = t804 + t805;
        float t807 = t797 + t803;
        scratch_0[t795] = t807;
        float t809 = t798 + t806;
        scratch_1[t795] = t809;
        float t811 = t797 - t803;
        scratch_0[t796] = t811;
        float t813 = t798 - t806;
        scratch_1[t796] = t813;
      }
      for (uint t816 = 0; t816 < 256; t816++) {
        float t817 = (float)t816;
        float t818 = (t817 * 0.125);
        float t819 = metal::floor(t818);
        float t820 = t819 * 8.0;
        float t821 = t817 - t820;
        float t822 = t819 * 16.0;
        float t823 = t822 + t821;
        float t824 = t823 + 8.0;
        int t825 = (int)t821;
        int t826 = 7 + t825;
        float t827 = scratch_2[t826];
        float t828 = scratch_3[t826];
        float t829 = 0.0 - t828;
        int t830 = (int)t823;
        int t831 = (int)t824;
        float t832 = scratch_0[t830];
        float t833 = scratch_1[t830];
        float t834 = scratch_0[t831];
        float t835 = scratch_1[t831];
        float t836 = t827 * t834;
        float t837 = t829 * t835;
        float t838 = t836 - t837;
        float t839 = t827 * t835;
        float t840 = t829 * t834;
        float t841 = t839 + t840;
        float t842 = t832 + t838;
        scratch_0[t830] = t842;
        float t844 = t833 + t841;
        scratch_1[t830] = t844;
        float t846 = t832 - t838;
        scratch_0[t831] = t846;
        float t848 = t833 - t841;
        scratch_1[t831] = t848;
      }
      for (uint t851 = 0; t851 < 256; t851++) {
        float t852 = (float)t851;
        float t853 = (t852 * 0.0625);
        float t854 = metal::floor(t853);
        float t855 = t854 * 16.0;
        float t856 = t852 - t855;
        float t857 = t854 * 32.0;
        float t858 = t857 + t856;
        float t859 = t858 + 16.0;
        int t860 = (int)t856;
        int t861 = 15 + t860;
        float t862 = scratch_2[t861];
        float t863 = scratch_3[t861];
        float t864 = 0.0 - t863;
        int t865 = (int)t858;
        int t866 = (int)t859;
        float t867 = scratch_0[t865];
        float t868 = scratch_1[t865];
        float t869 = scratch_0[t866];
        float t870 = scratch_1[t866];
        float t871 = t862 * t869;
        float t872 = t864 * t870;
        float t873 = t871 - t872;
        float t874 = t862 * t870;
        float t875 = t864 * t869;
        float t876 = t874 + t875;
        float t877 = t867 + t873;
        scratch_0[t865] = t877;
        float t879 = t868 + t876;
        scratch_1[t865] = t879;
        float t881 = t867 - t873;
        scratch_0[t866] = t881;
        float t883 = t868 - t876;
        scratch_1[t866] = t883;
      }
      for (uint t886 = 0; t886 < 256; t886++) {
        float t887 = (float)t886;
        float t888 = (t887 * 0.03125);
        float t889 = metal::floor(t888);
        float t890 = t889 * 32.0;
        float t891 = t887 - t890;
        float t892 = t889 * 64.0;
        float t893 = t892 + t891;
        float t894 = t893 + 32.0;
        int t895 = (int)t891;
        int t896 = 31 + t895;
        float t897 = scratch_2[t896];
        float t898 = scratch_3[t896];
        float t899 = 0.0 - t898;
        int t900 = (int)t893;
        int t901 = (int)t894;
        float t902 = scratch_0[t900];
        float t903 = scratch_1[t900];
        float t904 = scratch_0[t901];
        float t905 = scratch_1[t901];
        float t906 = t897 * t904;
        float t907 = t899 * t905;
        float t908 = t906 - t907;
        float t909 = t897 * t905;
        float t910 = t899 * t904;
        float t911 = t909 + t910;
        float t912 = t902 + t908;
        scratch_0[t900] = t912;
        float t914 = t903 + t911;
        scratch_1[t900] = t914;
        float t916 = t902 - t908;
        scratch_0[t901] = t916;
        float t918 = t903 - t911;
        scratch_1[t901] = t918;
      }
      for (uint t921 = 0; t921 < 256; t921++) {
        float t922 = (float)t921;
        float t923 = (t922 * 0.015625);
        float t924 = metal::floor(t923);
        float t925 = t924 * 64.0;
        float t926 = t922 - t925;
        float t927 = t924 * 128.0;
        float t928 = t927 + t926;
        float t929 = t928 + 64.0;
        int t930 = (int)t926;
        int t931 = 63 + t930;
        float t932 = scratch_2[t931];
        float t933 = scratch_3[t931];
        float t934 = 0.0 - t933;
        int t935 = (int)t928;
        int t936 = (int)t929;
        float t937 = scratch_0[t935];
        float t938 = scratch_1[t935];
        float t939 = scratch_0[t936];
        float t940 = scratch_1[t936];
        float t941 = t932 * t939;
        float t942 = t934 * t940;
        float t943 = t941 - t942;
        float t944 = t932 * t940;
        float t945 = t934 * t939;
        float t946 = t944 + t945;
        float t947 = t937 + t943;
        scratch_0[t935] = t947;
        float t949 = t938 + t946;
        scratch_1[t935] = t949;
        float t951 = t937 - t943;
        scratch_0[t936] = t951;
        float t953 = t938 - t946;
        scratch_1[t936] = t953;
      }
      for (uint t956 = 0; t956 < 256; t956++) {
        float t957 = (float)t956;
        float t958 = (t957 * 0.0078125);
        float t959 = metal::floor(t958);
        float t960 = t959 * 128.0;
        float t961 = t957 - t960;
        float t962 = t959 * 256.0;
        float t963 = t962 + t961;
        float t964 = t963 + 128.0;
        int t965 = (int)t961;
        int t966 = 127 + t965;
        float t967 = scratch_2[t966];
        float t968 = scratch_3[t966];
        float t969 = 0.0 - t968;
        int t970 = (int)t963;
        int t971 = (int)t964;
        float t972 = scratch_0[t970];
        float t973 = scratch_1[t970];
        float t974 = scratch_0[t971];
        float t975 = scratch_1[t971];
        float t976 = t967 * t974;
        float t977 = t969 * t975;
        float t978 = t976 - t977;
        float t979 = t967 * t975;
        float t980 = t969 * t974;
        float t981 = t979 + t980;
        float t982 = t972 + t978;
        scratch_0[t970] = t982;
        float t984 = t973 + t981;
        scratch_1[t970] = t984;
        float t986 = t972 - t978;
        scratch_0[t971] = t986;
        float t988 = t973 - t981;
        scratch_1[t971] = t988;
      }
      for (uint t991 = 0; t991 < 256; t991++) {
        float t992 = (float)t991;
        float t993 = (t992 * 0.00390625);
        float t994 = metal::floor(t993);
        float t995 = t994 * 256.0;
        float t996 = t992 - t995;
        float t997 = t994 * 512.0;
        float t998 = t997 + t996;
        float t999 = t998 + 256.0;
        int t1000 = (int)t996;
        int t1001 = 255 + t1000;
        float t1002 = scratch_2[t1001];
        float t1003 = scratch_3[t1001];
        float t1004 = 0.0 - t1003;
        int t1005 = (int)t998;
        int t1006 = (int)t999;
        float t1007 = scratch_0[t1005];
        float t1008 = scratch_1[t1005];
        float t1009 = scratch_0[t1006];
        float t1010 = scratch_1[t1006];
        float t1011 = t1002 * t1009;
        float t1012 = t1004 * t1010;
        float t1013 = t1011 - t1012;
        float t1014 = t1002 * t1010;
        float t1015 = t1004 * t1009;
        float t1016 = t1014 + t1015;
        float t1017 = t1007 + t1013;
        scratch_0[t1005] = t1017;
        float t1019 = t1008 + t1016;
        scratch_1[t1005] = t1019;
        float t1021 = t1007 - t1013;
        scratch_0[t1006] = t1021;
        float t1023 = t1008 - t1016;
        scratch_1[t1006] = t1023;
      }
      for (uint t1026 = 0; t1026 < 512; t1026++) {
        float t1027 = scratch_0[(int)t1026];
        float t1028 = scratch_1[(int)t1026];
        int t1029 = t670 + t1026;
        memory[26710464 + t1029] = t1027;
        int t1031 = t670 + t1026;
        int t1032 = t1031 + 512;
        memory[26710464 + t1032] = t1028;
      }
      for (uint t1035 = 0; t1035 < 257; t1035++) {
        float t1036 = scratch_0[(int)t1035];
        float t1037 = scratch_1[(int)t1035];
        float t1038 = t1036 * t1036;
        float t1039 = t1037 * t1037;
        float t1040 = t1038 + t1039;
        float t1041 = metal::sqrt(t1040);
        int t1042 = t671 + t1035;
        memory[28807616 + t1042] = t1041;
      }
      int t1045 = t656 / 128;
      int t1046 = t1045 * 8;
      int t1047 = t1046 + t658;
      int t1048 = t1047 * 1024;
      int t1049 = t1047 * 257;
      for (uint t1050 = 0; t1050 < 512; t1050++) {
        float t1051 = (float)t1050;
        float t1052 = memory[159682 + (int)t1050];
        float t1053 = (float)t656;
        float t1054 = t1053 - 511.0;
        float t1055 = t1054 + t1051;
        float t1056 = t1055 >= 0.0;
        float t1057 = t1055 < frameCount;
        float t1058 = t1056 * t1057;
        float t1059 = frameCount - 1.0;
        float t1060 = metal::min(t1055, t1059);
        float t1061 = metal::max(0.0, t1060);
        int t1062 = (int)t1061;
        int t1063 = t1062 * 8;
        int t1064 = t1063 + t658;
        float t1065 = memory[18059712 + t1064];
        float t1066 = metal::select(0.0, t1065, t1058 > 0.0);
        float t1067 = t1066 * t1052;
        scratch_0[(int)t1050] = t1067;
        scratch_1[(int)t1050] = 0.0;
      }
      for (uint t1071 = 0; t1071 < 512; t1071++) {
        float t1072 = memory[161216 + (int)t1071];
        float t1073 = (float)t1071;
        float t1074 = t1073 < t1072;
        int t1075 = (int)t1072;
        float t1076 = scratch_0[(int)t1071];
        float t1077 = scratch_1[(int)t1071];
        float t1078 = scratch_0[t1075];
        float t1079 = scratch_1[t1075];
        float t1080 = metal::select(t1076, t1078, t1074 > 0.0);
        float t1081 = metal::select(t1077, t1079, t1074 > 0.0);
        float t1082 = metal::select(t1078, t1076, t1074 > 0.0);
        float t1083 = metal::select(t1079, t1077, t1074 > 0.0);
        scratch_0[(int)t1071] = t1080;
        scratch_1[(int)t1071] = t1081;
        scratch_0[t1075] = t1082;
        scratch_1[t1075] = t1083;
      }
      for (uint t1089 = 0; t1089 < 256; t1089++) {
        float t1090 = (float)t1089;
        float t1091 = t1090;
        float t1092 = metal::floor(t1091);
        float t1093 = t1092;
        float t1094 = t1090 - t1093;
        float t1095 = t1092 * 2.0;
        float t1096 = t1095 + t1094;
        float t1097 = t1096 + 1.0;
        int t1098 = (int)t1094;
        int t1099 = t1098;
        float t1100 = scratch_2[t1099];
        float t1101 = scratch_3[t1099];
        float t1102 = 0.0 - t1101;
        int t1103 = (int)t1096;
        int t1104 = (int)t1097;
        float t1105 = scratch_0[t1103];
        float t1106 = scratch_1[t1103];
        float t1107 = scratch_0[t1104];
        float t1108 = scratch_1[t1104];
        float t1109 = t1100 * t1107;
        float t1110 = t1102 * t1108;
        float t1111 = t1109 - t1110;
        float t1112 = t1100 * t1108;
        float t1113 = t1102 * t1107;
        float t1114 = t1112 + t1113;
        float t1115 = t1105 + t1111;
        scratch_0[t1103] = t1115;
        float t1117 = t1106 + t1114;
        scratch_1[t1103] = t1117;
        float t1119 = t1105 - t1111;
        scratch_0[t1104] = t1119;
        float t1121 = t1106 - t1114;
        scratch_1[t1104] = t1121;
      }
      for (uint t1124 = 0; t1124 < 256; t1124++) {
        float t1125 = (float)t1124;
        float t1126 = (t1125 * 0.5);
        float t1127 = metal::floor(t1126);
        float t1128 = t1127 * 2.0;
        float t1129 = t1125 - t1128;
        float t1130 = t1127 * 4.0;
        float t1131 = t1130 + t1129;
        float t1132 = t1131 + 2.0;
        int t1133 = (int)t1129;
        int t1134 = 1 + t1133;
        float t1135 = scratch_2[t1134];
        float t1136 = scratch_3[t1134];
        float t1137 = 0.0 - t1136;
        int t1138 = (int)t1131;
        int t1139 = (int)t1132;
        float t1140 = scratch_0[t1138];
        float t1141 = scratch_1[t1138];
        float t1142 = scratch_0[t1139];
        float t1143 = scratch_1[t1139];
        float t1144 = t1135 * t1142;
        float t1145 = t1137 * t1143;
        float t1146 = t1144 - t1145;
        float t1147 = t1135 * t1143;
        float t1148 = t1137 * t1142;
        float t1149 = t1147 + t1148;
        float t1150 = t1140 + t1146;
        scratch_0[t1138] = t1150;
        float t1152 = t1141 + t1149;
        scratch_1[t1138] = t1152;
        float t1154 = t1140 - t1146;
        scratch_0[t1139] = t1154;
        float t1156 = t1141 - t1149;
        scratch_1[t1139] = t1156;
      }
      for (uint t1159 = 0; t1159 < 256; t1159++) {
        float t1160 = (float)t1159;
        float t1161 = (t1160 * 0.25);
        float t1162 = metal::floor(t1161);
        float t1163 = t1162 * 4.0;
        float t1164 = t1160 - t1163;
        float t1165 = t1162 * 8.0;
        float t1166 = t1165 + t1164;
        float t1167 = t1166 + 4.0;
        int t1168 = (int)t1164;
        int t1169 = 3 + t1168;
        float t1170 = scratch_2[t1169];
        float t1171 = scratch_3[t1169];
        float t1172 = 0.0 - t1171;
        int t1173 = (int)t1166;
        int t1174 = (int)t1167;
        float t1175 = scratch_0[t1173];
        float t1176 = scratch_1[t1173];
        float t1177 = scratch_0[t1174];
        float t1178 = scratch_1[t1174];
        float t1179 = t1170 * t1177;
        float t1180 = t1172 * t1178;
        float t1181 = t1179 - t1180;
        float t1182 = t1170 * t1178;
        float t1183 = t1172 * t1177;
        float t1184 = t1182 + t1183;
        float t1185 = t1175 + t1181;
        scratch_0[t1173] = t1185;
        float t1187 = t1176 + t1184;
        scratch_1[t1173] = t1187;
        float t1189 = t1175 - t1181;
        scratch_0[t1174] = t1189;
        float t1191 = t1176 - t1184;
        scratch_1[t1174] = t1191;
      }
      for (uint t1194 = 0; t1194 < 256; t1194++) {
        float t1195 = (float)t1194;
        float t1196 = (t1195 * 0.125);
        float t1197 = metal::floor(t1196);
        float t1198 = t1197 * 8.0;
        float t1199 = t1195 - t1198;
        float t1200 = t1197 * 16.0;
        float t1201 = t1200 + t1199;
        float t1202 = t1201 + 8.0;
        int t1203 = (int)t1199;
        int t1204 = 7 + t1203;
        float t1205 = scratch_2[t1204];
        float t1206 = scratch_3[t1204];
        float t1207 = 0.0 - t1206;
        int t1208 = (int)t1201;
        int t1209 = (int)t1202;
        float t1210 = scratch_0[t1208];
        float t1211 = scratch_1[t1208];
        float t1212 = scratch_0[t1209];
        float t1213 = scratch_1[t1209];
        float t1214 = t1205 * t1212;
        float t1215 = t1207 * t1213;
        float t1216 = t1214 - t1215;
        float t1217 = t1205 * t1213;
        float t1218 = t1207 * t1212;
        float t1219 = t1217 + t1218;
        float t1220 = t1210 + t1216;
        scratch_0[t1208] = t1220;
        float t1222 = t1211 + t1219;
        scratch_1[t1208] = t1222;
        float t1224 = t1210 - t1216;
        scratch_0[t1209] = t1224;
        float t1226 = t1211 - t1219;
        scratch_1[t1209] = t1226;
      }
      for (uint t1229 = 0; t1229 < 256; t1229++) {
        float t1230 = (float)t1229;
        float t1231 = (t1230 * 0.0625);
        float t1232 = metal::floor(t1231);
        float t1233 = t1232 * 16.0;
        float t1234 = t1230 - t1233;
        float t1235 = t1232 * 32.0;
        float t1236 = t1235 + t1234;
        float t1237 = t1236 + 16.0;
        int t1238 = (int)t1234;
        int t1239 = 15 + t1238;
        float t1240 = scratch_2[t1239];
        float t1241 = scratch_3[t1239];
        float t1242 = 0.0 - t1241;
        int t1243 = (int)t1236;
        int t1244 = (int)t1237;
        float t1245 = scratch_0[t1243];
        float t1246 = scratch_1[t1243];
        float t1247 = scratch_0[t1244];
        float t1248 = scratch_1[t1244];
        float t1249 = t1240 * t1247;
        float t1250 = t1242 * t1248;
        float t1251 = t1249 - t1250;
        float t1252 = t1240 * t1248;
        float t1253 = t1242 * t1247;
        float t1254 = t1252 + t1253;
        float t1255 = t1245 + t1251;
        scratch_0[t1243] = t1255;
        float t1257 = t1246 + t1254;
        scratch_1[t1243] = t1257;
        float t1259 = t1245 - t1251;
        scratch_0[t1244] = t1259;
        float t1261 = t1246 - t1254;
        scratch_1[t1244] = t1261;
      }
      for (uint t1264 = 0; t1264 < 256; t1264++) {
        float t1265 = (float)t1264;
        float t1266 = (t1265 * 0.03125);
        float t1267 = metal::floor(t1266);
        float t1268 = t1267 * 32.0;
        float t1269 = t1265 - t1268;
        float t1270 = t1267 * 64.0;
        float t1271 = t1270 + t1269;
        float t1272 = t1271 + 32.0;
        int t1273 = (int)t1269;
        int t1274 = 31 + t1273;
        float t1275 = scratch_2[t1274];
        float t1276 = scratch_3[t1274];
        float t1277 = 0.0 - t1276;
        int t1278 = (int)t1271;
        int t1279 = (int)t1272;
        float t1280 = scratch_0[t1278];
        float t1281 = scratch_1[t1278];
        float t1282 = scratch_0[t1279];
        float t1283 = scratch_1[t1279];
        float t1284 = t1275 * t1282;
        float t1285 = t1277 * t1283;
        float t1286 = t1284 - t1285;
        float t1287 = t1275 * t1283;
        float t1288 = t1277 * t1282;
        float t1289 = t1287 + t1288;
        float t1290 = t1280 + t1286;
        scratch_0[t1278] = t1290;
        float t1292 = t1281 + t1289;
        scratch_1[t1278] = t1292;
        float t1294 = t1280 - t1286;
        scratch_0[t1279] = t1294;
        float t1296 = t1281 - t1289;
        scratch_1[t1279] = t1296;
      }
      for (uint t1299 = 0; t1299 < 256; t1299++) {
        float t1300 = (float)t1299;
        float t1301 = (t1300 * 0.015625);
        float t1302 = metal::floor(t1301);
        float t1303 = t1302 * 64.0;
        float t1304 = t1300 - t1303;
        float t1305 = t1302 * 128.0;
        float t1306 = t1305 + t1304;
        float t1307 = t1306 + 64.0;
        int t1308 = (int)t1304;
        int t1309 = 63 + t1308;
        float t1310 = scratch_2[t1309];
        float t1311 = scratch_3[t1309];
        float t1312 = 0.0 - t1311;
        int t1313 = (int)t1306;
        int t1314 = (int)t1307;
        float t1315 = scratch_0[t1313];
        float t1316 = scratch_1[t1313];
        float t1317 = scratch_0[t1314];
        float t1318 = scratch_1[t1314];
        float t1319 = t1310 * t1317;
        float t1320 = t1312 * t1318;
        float t1321 = t1319 - t1320;
        float t1322 = t1310 * t1318;
        float t1323 = t1312 * t1317;
        float t1324 = t1322 + t1323;
        float t1325 = t1315 + t1321;
        scratch_0[t1313] = t1325;
        float t1327 = t1316 + t1324;
        scratch_1[t1313] = t1327;
        float t1329 = t1315 - t1321;
        scratch_0[t1314] = t1329;
        float t1331 = t1316 - t1324;
        scratch_1[t1314] = t1331;
      }
      for (uint t1334 = 0; t1334 < 256; t1334++) {
        float t1335 = (float)t1334;
        float t1336 = (t1335 * 0.0078125);
        float t1337 = metal::floor(t1336);
        float t1338 = t1337 * 128.0;
        float t1339 = t1335 - t1338;
        float t1340 = t1337 * 256.0;
        float t1341 = t1340 + t1339;
        float t1342 = t1341 + 128.0;
        int t1343 = (int)t1339;
        int t1344 = 127 + t1343;
        float t1345 = scratch_2[t1344];
        float t1346 = scratch_3[t1344];
        float t1347 = 0.0 - t1346;
        int t1348 = (int)t1341;
        int t1349 = (int)t1342;
        float t1350 = scratch_0[t1348];
        float t1351 = scratch_1[t1348];
        float t1352 = scratch_0[t1349];
        float t1353 = scratch_1[t1349];
        float t1354 = t1345 * t1352;
        float t1355 = t1347 * t1353;
        float t1356 = t1354 - t1355;
        float t1357 = t1345 * t1353;
        float t1358 = t1347 * t1352;
        float t1359 = t1357 + t1358;
        float t1360 = t1350 + t1356;
        scratch_0[t1348] = t1360;
        float t1362 = t1351 + t1359;
        scratch_1[t1348] = t1362;
        float t1364 = t1350 - t1356;
        scratch_0[t1349] = t1364;
        float t1366 = t1351 - t1359;
        scratch_1[t1349] = t1366;
      }
      for (uint t1369 = 0; t1369 < 256; t1369++) {
        float t1370 = (float)t1369;
        float t1371 = (t1370 * 0.00390625);
        float t1372 = metal::floor(t1371);
        float t1373 = t1372 * 256.0;
        float t1374 = t1370 - t1373;
        float t1375 = t1372 * 512.0;
        float t1376 = t1375 + t1374;
        float t1377 = t1376 + 256.0;
        int t1378 = (int)t1374;
        int t1379 = 255 + t1378;
        float t1380 = scratch_2[t1379];
        float t1381 = scratch_3[t1379];
        float t1382 = 0.0 - t1381;
        int t1383 = (int)t1376;
        int t1384 = (int)t1377;
        float t1385 = scratch_0[t1383];
        float t1386 = scratch_1[t1383];
        float t1387 = scratch_0[t1384];
        float t1388 = scratch_1[t1384];
        float t1389 = t1380 * t1387;
        float t1390 = t1382 * t1388;
        float t1391 = t1389 - t1390;
        float t1392 = t1380 * t1388;
        float t1393 = t1382 * t1387;
        float t1394 = t1392 + t1393;
        float t1395 = t1385 + t1391;
        scratch_0[t1383] = t1395;
        float t1397 = t1386 + t1394;
        scratch_1[t1383] = t1397;
        float t1399 = t1385 - t1391;
        scratch_0[t1384] = t1399;
        float t1401 = t1386 - t1394;
        scratch_1[t1384] = t1401;
      }
      for (uint t1404 = 0; t1404 < 512; t1404++) {
        float t1405 = scratch_0[(int)t1404];
        float t1406 = scratch_1[(int)t1404];
        int t1407 = t1048 + t1404;
        memory[27759040 + t1407] = t1405;
        int t1409 = t1048 + t1404;
        int t1410 = t1409 + 512;
        memory[27759040 + t1410] = t1406;
      }
      for (uint t1413 = 0; t1413 < 257; t1413++) {
        float t1414 = scratch_0[(int)t1413];
        float t1415 = scratch_1[(int)t1413];
        float t1416 = t1414 * t1414;
        float t1417 = t1415 * t1415;
        float t1418 = t1416 + t1417;
        float t1419 = metal::sqrt(t1418);
        int t1420 = t1049 + t1413;
        memory[29070784 + t1420] = t1419;
      }
      int t1423 = t656 / 128;
      int t1424 = t1423 * 8;
      int t1425 = t1424 + t658;
      int t1426 = t1425 * 257;
      float t1427 = 0.0;
      for (uint t1428 = 0; t1428 < 257; t1428++) {
        int t1429 = t1426 + t1428;
        float t1430 = memory[28807616 + t1429];
        int t1431 = t1426 + t1428;
        float t1432 = memory[29070784 + t1431];
        float t1433 = t1430 - t1432;
        float t1434 = t1433 * t1433;
        float t1435 = t1427 + t1434;
        t1427 = t1435;
      }
      memory[526272 + t1425] = t1427;
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 21
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_21(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1452), value: global(1452)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    int t1439 = id;
    int t1440 = t1439 / 128;
    int t1441 = t1439 % 128;
    int t1442 = t1441 == 0.0;
    float t1443 = 0.0;
    if (t1442) {
      for (uint t1445 = 0; t1445 < 8; t1445++) {
        int t1446 = t1440 * 8;
        int t1447 = t1446 + t1445;
        float t1448 = memory[526272 + t1447];
        float t1449 = t1443 + t1448;
        t1443 = t1449;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    t[4*frameCount + id] = (t1443 * 0.125);
  }
  #pragma clang diagnostic pop
}



// KERNEL 22
// FrameOrder: parallel
// DispatchMode: perFrameScaled(1024)
kernel void kernel_22(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1458), value: global(1458)) */
  float t4931 = frameCount * 1024.0;
  if (id >= 0 && id < (uint)(t4931)) {
    /* loadGlobal(1452) - handled in variable access */
    int t1453 = id;
    int t1454 = t1453 / 1024;
    uint _frameIndex = (uint)(t1454);
    int t1455 = t1454 * 1024;
    int t1456 = t1453 - t1455;
    float t1457 = (t[4*frameCount + _frameIndex] * 6.1035156e-05);
    t[5*frameCount + _frameIndex] = t1457;
  }
  #pragma clang diagnostic pop
}



// KERNEL 23
// FrameOrder: sequential
// DispatchMode: singleThreaded
kernel void kernel_23(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1467), value: global(1467)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[6*frameCount + i] = memory[41427716];
      float t1468 = t[6*frameCount + i] + 1.0;
      float t1469 = metal::select(t1468, 0.0, 0.0 > 0.0);
      float t1470 = t1469;
      float t1471 = (t1470 * 0.00390625);
      float t1472 = metal::floor(t1471);
      float t1473 = t1472 * 256.0;
      float t1474 = t1469 - t1473;
      memory[41427716] = t1474;
      float t1476 = t1474 >= 256.0;
      if (t1476) {
        float t1478 = t1474 - 256.0;
        memory[41427716] = t1478;
      }
      if (0.0) {
        memory[41427716] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 24
// FrameOrder: parallel
// DispatchMode: perFrameScaledThreadgroup1(8)
kernel void kernel_24(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4932 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4932)) {
    /* loadGlobal(1467) - handled in variable access */
    int t1484 = id;
    int t1485 = t1484 / 8;
    uint _frameIndex = (uint)(t1485);
    int t1486 = t1485 * 8;
    int t1487 = t1484 - t1486;
    threadgroup float scratch_0[1024];
    threadgroup float scratch_1[1024];
    threadgroup float scratch_2[1023];
    threadgroup float scratch_3[1023];
    float t1488 = t[6*frameCount + _frameIndex] == 0.0;
    if (t1488) {
      for (uint t1490 = 0; t1490 < 1023; t1490++) {
        float t1491 = memory[162752 + (int)t1490];
        scratch_2[(int)t1490] = t1491;
        float t1493 = memory[163775 + (int)t1490];
        scratch_3[(int)t1490] = t1493;
      }
      int t1496 = t1485 / 256;
      int t1497 = t1496 * 8;
      int t1498 = t1497 + t1487;
      int t1499 = t1498 * 2048;
      int t1500 = t1498 * 513;
      for (uint t1501 = 0; t1501 < 1024; t1501++) {
        float t1502 = (float)t1501;
        float t1503 = memory[161728 + (int)t1501];
        float t1504 = (float)t1485;
        float t1505 = t1504 - 1023.0;
        float t1506 = t1505 + t1502;
        float t1507 = t1506 >= 0.0;
        float t1508 = t1506 < frameCount;
        float t1509 = t1507 * t1508;
        float t1510 = frameCount - 1.0;
        float t1511 = metal::min(t1506, t1510);
        float t1512 = metal::max(0.0, t1511);
        int t1513 = (int)t1512;
        int t1514 = t1513 * 8;
        int t1515 = t1514 + t1487;
        float t1516 = memory[26579392 + t1515];
        float t1517 = metal::select(0.0, t1516, t1509 > 0.0);
        float t1518 = t1517 * t1503;
        scratch_0[(int)t1501] = t1518;
        scratch_1[(int)t1501] = 0.0;
      }
      for (uint t1522 = 0; t1522 < 1024; t1522++) {
        float t1523 = memory[164798 + (int)t1522];
        float t1524 = (float)t1522;
        float t1525 = t1524 < t1523;
        int t1526 = (int)t1523;
        float t1527 = scratch_0[(int)t1522];
        float t1528 = scratch_1[(int)t1522];
        float t1529 = scratch_0[t1526];
        float t1530 = scratch_1[t1526];
        float t1531 = metal::select(t1527, t1529, t1525 > 0.0);
        float t1532 = metal::select(t1528, t1530, t1525 > 0.0);
        float t1533 = metal::select(t1529, t1527, t1525 > 0.0);
        float t1534 = metal::select(t1530, t1528, t1525 > 0.0);
        scratch_0[(int)t1522] = t1531;
        scratch_1[(int)t1522] = t1532;
        scratch_0[t1526] = t1533;
        scratch_1[t1526] = t1534;
      }
      for (uint t1540 = 0; t1540 < 512; t1540++) {
        float t1541 = (float)t1540;
        float t1542 = t1541;
        float t1543 = metal::floor(t1542);
        float t1544 = t1543;
        float t1545 = t1541 - t1544;
        float t1546 = t1543 * 2.0;
        float t1547 = t1546 + t1545;
        float t1548 = t1547 + 1.0;
        int t1549 = (int)t1545;
        int t1550 = t1549;
        float t1551 = scratch_2[t1550];
        float t1552 = scratch_3[t1550];
        float t1553 = 0.0 - t1552;
        int t1554 = (int)t1547;
        int t1555 = (int)t1548;
        float t1556 = scratch_0[t1554];
        float t1557 = scratch_1[t1554];
        float t1558 = scratch_0[t1555];
        float t1559 = scratch_1[t1555];
        float t1560 = t1551 * t1558;
        float t1561 = t1553 * t1559;
        float t1562 = t1560 - t1561;
        float t1563 = t1551 * t1559;
        float t1564 = t1553 * t1558;
        float t1565 = t1563 + t1564;
        float t1566 = t1556 + t1562;
        scratch_0[t1554] = t1566;
        float t1568 = t1557 + t1565;
        scratch_1[t1554] = t1568;
        float t1570 = t1556 - t1562;
        scratch_0[t1555] = t1570;
        float t1572 = t1557 - t1565;
        scratch_1[t1555] = t1572;
      }
      for (uint t1575 = 0; t1575 < 512; t1575++) {
        float t1576 = (float)t1575;
        float t1577 = (t1576 * 0.5);
        float t1578 = metal::floor(t1577);
        float t1579 = t1578 * 2.0;
        float t1580 = t1576 - t1579;
        float t1581 = t1578 * 4.0;
        float t1582 = t1581 + t1580;
        float t1583 = t1582 + 2.0;
        int t1584 = (int)t1580;
        int t1585 = 1 + t1584;
        float t1586 = scratch_2[t1585];
        float t1587 = scratch_3[t1585];
        float t1588 = 0.0 - t1587;
        int t1589 = (int)t1582;
        int t1590 = (int)t1583;
        float t1591 = scratch_0[t1589];
        float t1592 = scratch_1[t1589];
        float t1593 = scratch_0[t1590];
        float t1594 = scratch_1[t1590];
        float t1595 = t1586 * t1593;
        float t1596 = t1588 * t1594;
        float t1597 = t1595 - t1596;
        float t1598 = t1586 * t1594;
        float t1599 = t1588 * t1593;
        float t1600 = t1598 + t1599;
        float t1601 = t1591 + t1597;
        scratch_0[t1589] = t1601;
        float t1603 = t1592 + t1600;
        scratch_1[t1589] = t1603;
        float t1605 = t1591 - t1597;
        scratch_0[t1590] = t1605;
        float t1607 = t1592 - t1600;
        scratch_1[t1590] = t1607;
      }
      for (uint t1610 = 0; t1610 < 512; t1610++) {
        float t1611 = (float)t1610;
        float t1612 = (t1611 * 0.25);
        float t1613 = metal::floor(t1612);
        float t1614 = t1613 * 4.0;
        float t1615 = t1611 - t1614;
        float t1616 = t1613 * 8.0;
        float t1617 = t1616 + t1615;
        float t1618 = t1617 + 4.0;
        int t1619 = (int)t1615;
        int t1620 = 3 + t1619;
        float t1621 = scratch_2[t1620];
        float t1622 = scratch_3[t1620];
        float t1623 = 0.0 - t1622;
        int t1624 = (int)t1617;
        int t1625 = (int)t1618;
        float t1626 = scratch_0[t1624];
        float t1627 = scratch_1[t1624];
        float t1628 = scratch_0[t1625];
        float t1629 = scratch_1[t1625];
        float t1630 = t1621 * t1628;
        float t1631 = t1623 * t1629;
        float t1632 = t1630 - t1631;
        float t1633 = t1621 * t1629;
        float t1634 = t1623 * t1628;
        float t1635 = t1633 + t1634;
        float t1636 = t1626 + t1632;
        scratch_0[t1624] = t1636;
        float t1638 = t1627 + t1635;
        scratch_1[t1624] = t1638;
        float t1640 = t1626 - t1632;
        scratch_0[t1625] = t1640;
        float t1642 = t1627 - t1635;
        scratch_1[t1625] = t1642;
      }
      for (uint t1645 = 0; t1645 < 512; t1645++) {
        float t1646 = (float)t1645;
        float t1647 = (t1646 * 0.125);
        float t1648 = metal::floor(t1647);
        float t1649 = t1648 * 8.0;
        float t1650 = t1646 - t1649;
        float t1651 = t1648 * 16.0;
        float t1652 = t1651 + t1650;
        float t1653 = t1652 + 8.0;
        int t1654 = (int)t1650;
        int t1655 = 7 + t1654;
        float t1656 = scratch_2[t1655];
        float t1657 = scratch_3[t1655];
        float t1658 = 0.0 - t1657;
        int t1659 = (int)t1652;
        int t1660 = (int)t1653;
        float t1661 = scratch_0[t1659];
        float t1662 = scratch_1[t1659];
        float t1663 = scratch_0[t1660];
        float t1664 = scratch_1[t1660];
        float t1665 = t1656 * t1663;
        float t1666 = t1658 * t1664;
        float t1667 = t1665 - t1666;
        float t1668 = t1656 * t1664;
        float t1669 = t1658 * t1663;
        float t1670 = t1668 + t1669;
        float t1671 = t1661 + t1667;
        scratch_0[t1659] = t1671;
        float t1673 = t1662 + t1670;
        scratch_1[t1659] = t1673;
        float t1675 = t1661 - t1667;
        scratch_0[t1660] = t1675;
        float t1677 = t1662 - t1670;
        scratch_1[t1660] = t1677;
      }
      for (uint t1680 = 0; t1680 < 512; t1680++) {
        float t1681 = (float)t1680;
        float t1682 = (t1681 * 0.0625);
        float t1683 = metal::floor(t1682);
        float t1684 = t1683 * 16.0;
        float t1685 = t1681 - t1684;
        float t1686 = t1683 * 32.0;
        float t1687 = t1686 + t1685;
        float t1688 = t1687 + 16.0;
        int t1689 = (int)t1685;
        int t1690 = 15 + t1689;
        float t1691 = scratch_2[t1690];
        float t1692 = scratch_3[t1690];
        float t1693 = 0.0 - t1692;
        int t1694 = (int)t1687;
        int t1695 = (int)t1688;
        float t1696 = scratch_0[t1694];
        float t1697 = scratch_1[t1694];
        float t1698 = scratch_0[t1695];
        float t1699 = scratch_1[t1695];
        float t1700 = t1691 * t1698;
        float t1701 = t1693 * t1699;
        float t1702 = t1700 - t1701;
        float t1703 = t1691 * t1699;
        float t1704 = t1693 * t1698;
        float t1705 = t1703 + t1704;
        float t1706 = t1696 + t1702;
        scratch_0[t1694] = t1706;
        float t1708 = t1697 + t1705;
        scratch_1[t1694] = t1708;
        float t1710 = t1696 - t1702;
        scratch_0[t1695] = t1710;
        float t1712 = t1697 - t1705;
        scratch_1[t1695] = t1712;
      }
      for (uint t1715 = 0; t1715 < 512; t1715++) {
        float t1716 = (float)t1715;
        float t1717 = (t1716 * 0.03125);
        float t1718 = metal::floor(t1717);
        float t1719 = t1718 * 32.0;
        float t1720 = t1716 - t1719;
        float t1721 = t1718 * 64.0;
        float t1722 = t1721 + t1720;
        float t1723 = t1722 + 32.0;
        int t1724 = (int)t1720;
        int t1725 = 31 + t1724;
        float t1726 = scratch_2[t1725];
        float t1727 = scratch_3[t1725];
        float t1728 = 0.0 - t1727;
        int t1729 = (int)t1722;
        int t1730 = (int)t1723;
        float t1731 = scratch_0[t1729];
        float t1732 = scratch_1[t1729];
        float t1733 = scratch_0[t1730];
        float t1734 = scratch_1[t1730];
        float t1735 = t1726 * t1733;
        float t1736 = t1728 * t1734;
        float t1737 = t1735 - t1736;
        float t1738 = t1726 * t1734;
        float t1739 = t1728 * t1733;
        float t1740 = t1738 + t1739;
        float t1741 = t1731 + t1737;
        scratch_0[t1729] = t1741;
        float t1743 = t1732 + t1740;
        scratch_1[t1729] = t1743;
        float t1745 = t1731 - t1737;
        scratch_0[t1730] = t1745;
        float t1747 = t1732 - t1740;
        scratch_1[t1730] = t1747;
      }
      for (uint t1750 = 0; t1750 < 512; t1750++) {
        float t1751 = (float)t1750;
        float t1752 = (t1751 * 0.015625);
        float t1753 = metal::floor(t1752);
        float t1754 = t1753 * 64.0;
        float t1755 = t1751 - t1754;
        float t1756 = t1753 * 128.0;
        float t1757 = t1756 + t1755;
        float t1758 = t1757 + 64.0;
        int t1759 = (int)t1755;
        int t1760 = 63 + t1759;
        float t1761 = scratch_2[t1760];
        float t1762 = scratch_3[t1760];
        float t1763 = 0.0 - t1762;
        int t1764 = (int)t1757;
        int t1765 = (int)t1758;
        float t1766 = scratch_0[t1764];
        float t1767 = scratch_1[t1764];
        float t1768 = scratch_0[t1765];
        float t1769 = scratch_1[t1765];
        float t1770 = t1761 * t1768;
        float t1771 = t1763 * t1769;
        float t1772 = t1770 - t1771;
        float t1773 = t1761 * t1769;
        float t1774 = t1763 * t1768;
        float t1775 = t1773 + t1774;
        float t1776 = t1766 + t1772;
        scratch_0[t1764] = t1776;
        float t1778 = t1767 + t1775;
        scratch_1[t1764] = t1778;
        float t1780 = t1766 - t1772;
        scratch_0[t1765] = t1780;
        float t1782 = t1767 - t1775;
        scratch_1[t1765] = t1782;
      }
      for (uint t1785 = 0; t1785 < 512; t1785++) {
        float t1786 = (float)t1785;
        float t1787 = (t1786 * 0.0078125);
        float t1788 = metal::floor(t1787);
        float t1789 = t1788 * 128.0;
        float t1790 = t1786 - t1789;
        float t1791 = t1788 * 256.0;
        float t1792 = t1791 + t1790;
        float t1793 = t1792 + 128.0;
        int t1794 = (int)t1790;
        int t1795 = 127 + t1794;
        float t1796 = scratch_2[t1795];
        float t1797 = scratch_3[t1795];
        float t1798 = 0.0 - t1797;
        int t1799 = (int)t1792;
        int t1800 = (int)t1793;
        float t1801 = scratch_0[t1799];
        float t1802 = scratch_1[t1799];
        float t1803 = scratch_0[t1800];
        float t1804 = scratch_1[t1800];
        float t1805 = t1796 * t1803;
        float t1806 = t1798 * t1804;
        float t1807 = t1805 - t1806;
        float t1808 = t1796 * t1804;
        float t1809 = t1798 * t1803;
        float t1810 = t1808 + t1809;
        float t1811 = t1801 + t1807;
        scratch_0[t1799] = t1811;
        float t1813 = t1802 + t1810;
        scratch_1[t1799] = t1813;
        float t1815 = t1801 - t1807;
        scratch_0[t1800] = t1815;
        float t1817 = t1802 - t1810;
        scratch_1[t1800] = t1817;
      }
      for (uint t1820 = 0; t1820 < 512; t1820++) {
        float t1821 = (float)t1820;
        float t1822 = (t1821 * 0.00390625);
        float t1823 = metal::floor(t1822);
        float t1824 = t1823 * 256.0;
        float t1825 = t1821 - t1824;
        float t1826 = t1823 * 512.0;
        float t1827 = t1826 + t1825;
        float t1828 = t1827 + 256.0;
        int t1829 = (int)t1825;
        int t1830 = 255 + t1829;
        float t1831 = scratch_2[t1830];
        float t1832 = scratch_3[t1830];
        float t1833 = 0.0 - t1832;
        int t1834 = (int)t1827;
        int t1835 = (int)t1828;
        float t1836 = scratch_0[t1834];
        float t1837 = scratch_1[t1834];
        float t1838 = scratch_0[t1835];
        float t1839 = scratch_1[t1835];
        float t1840 = t1831 * t1838;
        float t1841 = t1833 * t1839;
        float t1842 = t1840 - t1841;
        float t1843 = t1831 * t1839;
        float t1844 = t1833 * t1838;
        float t1845 = t1843 + t1844;
        float t1846 = t1836 + t1842;
        scratch_0[t1834] = t1846;
        float t1848 = t1837 + t1845;
        scratch_1[t1834] = t1848;
        float t1850 = t1836 - t1842;
        scratch_0[t1835] = t1850;
        float t1852 = t1837 - t1845;
        scratch_1[t1835] = t1852;
      }
      for (uint t1855 = 0; t1855 < 512; t1855++) {
        float t1856 = (float)t1855;
        float t1857 = (t1856 * 0.001953125);
        float t1858 = metal::floor(t1857);
        float t1859 = t1858 * 512.0;
        float t1860 = t1856 - t1859;
        float t1861 = t1858 * 1024.0;
        float t1862 = t1861 + t1860;
        float t1863 = t1862 + 512.0;
        int t1864 = (int)t1860;
        int t1865 = 511 + t1864;
        float t1866 = scratch_2[t1865];
        float t1867 = scratch_3[t1865];
        float t1868 = 0.0 - t1867;
        int t1869 = (int)t1862;
        int t1870 = (int)t1863;
        float t1871 = scratch_0[t1869];
        float t1872 = scratch_1[t1869];
        float t1873 = scratch_0[t1870];
        float t1874 = scratch_1[t1870];
        float t1875 = t1866 * t1873;
        float t1876 = t1868 * t1874;
        float t1877 = t1875 - t1876;
        float t1878 = t1866 * t1874;
        float t1879 = t1868 * t1873;
        float t1880 = t1878 + t1879;
        float t1881 = t1871 + t1877;
        scratch_0[t1869] = t1881;
        float t1883 = t1872 + t1880;
        scratch_1[t1869] = t1883;
        float t1885 = t1871 - t1877;
        scratch_0[t1870] = t1885;
        float t1887 = t1872 - t1880;
        scratch_1[t1870] = t1887;
      }
      for (uint t1890 = 0; t1890 < 1024; t1890++) {
        float t1891 = scratch_0[(int)t1890];
        float t1892 = scratch_1[(int)t1890];
        int t1893 = t1499 + t1890;
        memory[29333952 + t1893] = t1891;
        int t1895 = t1499 + t1890;
        int t1896 = t1895 + 1024;
        memory[29333952 + t1896] = t1892;
      }
      for (uint t1899 = 0; t1899 < 513; t1899++) {
        float t1900 = scratch_0[(int)t1899];
        float t1901 = scratch_1[(int)t1899];
        float t1902 = t1900 * t1900;
        float t1903 = t1901 * t1901;
        float t1904 = t1902 + t1903;
        float t1905 = metal::sqrt(t1904);
        int t1906 = t1500 + t1899;
        memory[31431104 + t1906] = t1905;
      }
      int t1909 = t1485 / 256;
      int t1910 = t1909 * 8;
      int t1911 = t1910 + t1487;
      int t1912 = t1911 * 2048;
      int t1913 = t1911 * 513;
      for (uint t1914 = 0; t1914 < 1024; t1914++) {
        float t1915 = (float)t1914;
        float t1916 = memory[161728 + (int)t1914];
        float t1917 = (float)t1485;
        float t1918 = t1917 - 1023.0;
        float t1919 = t1918 + t1915;
        float t1920 = t1919 >= 0.0;
        float t1921 = t1919 < frameCount;
        float t1922 = t1920 * t1921;
        float t1923 = frameCount - 1.0;
        float t1924 = metal::min(t1919, t1923);
        float t1925 = metal::max(0.0, t1924);
        int t1926 = (int)t1925;
        int t1927 = t1926 * 8;
        int t1928 = t1927 + t1487;
        float t1929 = memory[18059712 + t1928];
        float t1930 = metal::select(0.0, t1929, t1922 > 0.0);
        float t1931 = t1930 * t1916;
        scratch_0[(int)t1914] = t1931;
        scratch_1[(int)t1914] = 0.0;
      }
      for (uint t1935 = 0; t1935 < 1024; t1935++) {
        float t1936 = memory[164798 + (int)t1935];
        float t1937 = (float)t1935;
        float t1938 = t1937 < t1936;
        int t1939 = (int)t1936;
        float t1940 = scratch_0[(int)t1935];
        float t1941 = scratch_1[(int)t1935];
        float t1942 = scratch_0[t1939];
        float t1943 = scratch_1[t1939];
        float t1944 = metal::select(t1940, t1942, t1938 > 0.0);
        float t1945 = metal::select(t1941, t1943, t1938 > 0.0);
        float t1946 = metal::select(t1942, t1940, t1938 > 0.0);
        float t1947 = metal::select(t1943, t1941, t1938 > 0.0);
        scratch_0[(int)t1935] = t1944;
        scratch_1[(int)t1935] = t1945;
        scratch_0[t1939] = t1946;
        scratch_1[t1939] = t1947;
      }
      for (uint t1953 = 0; t1953 < 512; t1953++) {
        float t1954 = (float)t1953;
        float t1955 = t1954;
        float t1956 = metal::floor(t1955);
        float t1957 = t1956;
        float t1958 = t1954 - t1957;
        float t1959 = t1956 * 2.0;
        float t1960 = t1959 + t1958;
        float t1961 = t1960 + 1.0;
        int t1962 = (int)t1958;
        int t1963 = t1962;
        float t1964 = scratch_2[t1963];
        float t1965 = scratch_3[t1963];
        float t1966 = 0.0 - t1965;
        int t1967 = (int)t1960;
        int t1968 = (int)t1961;
        float t1969 = scratch_0[t1967];
        float t1970 = scratch_1[t1967];
        float t1971 = scratch_0[t1968];
        float t1972 = scratch_1[t1968];
        float t1973 = t1964 * t1971;
        float t1974 = t1966 * t1972;
        float t1975 = t1973 - t1974;
        float t1976 = t1964 * t1972;
        float t1977 = t1966 * t1971;
        float t1978 = t1976 + t1977;
        float t1979 = t1969 + t1975;
        scratch_0[t1967] = t1979;
        float t1981 = t1970 + t1978;
        scratch_1[t1967] = t1981;
        float t1983 = t1969 - t1975;
        scratch_0[t1968] = t1983;
        float t1985 = t1970 - t1978;
        scratch_1[t1968] = t1985;
      }
      for (uint t1988 = 0; t1988 < 512; t1988++) {
        float t1989 = (float)t1988;
        float t1990 = (t1989 * 0.5);
        float t1991 = metal::floor(t1990);
        float t1992 = t1991 * 2.0;
        float t1993 = t1989 - t1992;
        float t1994 = t1991 * 4.0;
        float t1995 = t1994 + t1993;
        float t1996 = t1995 + 2.0;
        int t1997 = (int)t1993;
        int t1998 = 1 + t1997;
        float t1999 = scratch_2[t1998];
        float t2000 = scratch_3[t1998];
        float t2001 = 0.0 - t2000;
        int t2002 = (int)t1995;
        int t2003 = (int)t1996;
        float t2004 = scratch_0[t2002];
        float t2005 = scratch_1[t2002];
        float t2006 = scratch_0[t2003];
        float t2007 = scratch_1[t2003];
        float t2008 = t1999 * t2006;
        float t2009 = t2001 * t2007;
        float t2010 = t2008 - t2009;
        float t2011 = t1999 * t2007;
        float t2012 = t2001 * t2006;
        float t2013 = t2011 + t2012;
        float t2014 = t2004 + t2010;
        scratch_0[t2002] = t2014;
        float t2016 = t2005 + t2013;
        scratch_1[t2002] = t2016;
        float t2018 = t2004 - t2010;
        scratch_0[t2003] = t2018;
        float t2020 = t2005 - t2013;
        scratch_1[t2003] = t2020;
      }
      for (uint t2023 = 0; t2023 < 512; t2023++) {
        float t2024 = (float)t2023;
        float t2025 = (t2024 * 0.25);
        float t2026 = metal::floor(t2025);
        float t2027 = t2026 * 4.0;
        float t2028 = t2024 - t2027;
        float t2029 = t2026 * 8.0;
        float t2030 = t2029 + t2028;
        float t2031 = t2030 + 4.0;
        int t2032 = (int)t2028;
        int t2033 = 3 + t2032;
        float t2034 = scratch_2[t2033];
        float t2035 = scratch_3[t2033];
        float t2036 = 0.0 - t2035;
        int t2037 = (int)t2030;
        int t2038 = (int)t2031;
        float t2039 = scratch_0[t2037];
        float t2040 = scratch_1[t2037];
        float t2041 = scratch_0[t2038];
        float t2042 = scratch_1[t2038];
        float t2043 = t2034 * t2041;
        float t2044 = t2036 * t2042;
        float t2045 = t2043 - t2044;
        float t2046 = t2034 * t2042;
        float t2047 = t2036 * t2041;
        float t2048 = t2046 + t2047;
        float t2049 = t2039 + t2045;
        scratch_0[t2037] = t2049;
        float t2051 = t2040 + t2048;
        scratch_1[t2037] = t2051;
        float t2053 = t2039 - t2045;
        scratch_0[t2038] = t2053;
        float t2055 = t2040 - t2048;
        scratch_1[t2038] = t2055;
      }
      for (uint t2058 = 0; t2058 < 512; t2058++) {
        float t2059 = (float)t2058;
        float t2060 = (t2059 * 0.125);
        float t2061 = metal::floor(t2060);
        float t2062 = t2061 * 8.0;
        float t2063 = t2059 - t2062;
        float t2064 = t2061 * 16.0;
        float t2065 = t2064 + t2063;
        float t2066 = t2065 + 8.0;
        int t2067 = (int)t2063;
        int t2068 = 7 + t2067;
        float t2069 = scratch_2[t2068];
        float t2070 = scratch_3[t2068];
        float t2071 = 0.0 - t2070;
        int t2072 = (int)t2065;
        int t2073 = (int)t2066;
        float t2074 = scratch_0[t2072];
        float t2075 = scratch_1[t2072];
        float t2076 = scratch_0[t2073];
        float t2077 = scratch_1[t2073];
        float t2078 = t2069 * t2076;
        float t2079 = t2071 * t2077;
        float t2080 = t2078 - t2079;
        float t2081 = t2069 * t2077;
        float t2082 = t2071 * t2076;
        float t2083 = t2081 + t2082;
        float t2084 = t2074 + t2080;
        scratch_0[t2072] = t2084;
        float t2086 = t2075 + t2083;
        scratch_1[t2072] = t2086;
        float t2088 = t2074 - t2080;
        scratch_0[t2073] = t2088;
        float t2090 = t2075 - t2083;
        scratch_1[t2073] = t2090;
      }
      for (uint t2093 = 0; t2093 < 512; t2093++) {
        float t2094 = (float)t2093;
        float t2095 = (t2094 * 0.0625);
        float t2096 = metal::floor(t2095);
        float t2097 = t2096 * 16.0;
        float t2098 = t2094 - t2097;
        float t2099 = t2096 * 32.0;
        float t2100 = t2099 + t2098;
        float t2101 = t2100 + 16.0;
        int t2102 = (int)t2098;
        int t2103 = 15 + t2102;
        float t2104 = scratch_2[t2103];
        float t2105 = scratch_3[t2103];
        float t2106 = 0.0 - t2105;
        int t2107 = (int)t2100;
        int t2108 = (int)t2101;
        float t2109 = scratch_0[t2107];
        float t2110 = scratch_1[t2107];
        float t2111 = scratch_0[t2108];
        float t2112 = scratch_1[t2108];
        float t2113 = t2104 * t2111;
        float t2114 = t2106 * t2112;
        float t2115 = t2113 - t2114;
        float t2116 = t2104 * t2112;
        float t2117 = t2106 * t2111;
        float t2118 = t2116 + t2117;
        float t2119 = t2109 + t2115;
        scratch_0[t2107] = t2119;
        float t2121 = t2110 + t2118;
        scratch_1[t2107] = t2121;
        float t2123 = t2109 - t2115;
        scratch_0[t2108] = t2123;
        float t2125 = t2110 - t2118;
        scratch_1[t2108] = t2125;
      }
      for (uint t2128 = 0; t2128 < 512; t2128++) {
        float t2129 = (float)t2128;
        float t2130 = (t2129 * 0.03125);
        float t2131 = metal::floor(t2130);
        float t2132 = t2131 * 32.0;
        float t2133 = t2129 - t2132;
        float t2134 = t2131 * 64.0;
        float t2135 = t2134 + t2133;
        float t2136 = t2135 + 32.0;
        int t2137 = (int)t2133;
        int t2138 = 31 + t2137;
        float t2139 = scratch_2[t2138];
        float t2140 = scratch_3[t2138];
        float t2141 = 0.0 - t2140;
        int t2142 = (int)t2135;
        int t2143 = (int)t2136;
        float t2144 = scratch_0[t2142];
        float t2145 = scratch_1[t2142];
        float t2146 = scratch_0[t2143];
        float t2147 = scratch_1[t2143];
        float t2148 = t2139 * t2146;
        float t2149 = t2141 * t2147;
        float t2150 = t2148 - t2149;
        float t2151 = t2139 * t2147;
        float t2152 = t2141 * t2146;
        float t2153 = t2151 + t2152;
        float t2154 = t2144 + t2150;
        scratch_0[t2142] = t2154;
        float t2156 = t2145 + t2153;
        scratch_1[t2142] = t2156;
        float t2158 = t2144 - t2150;
        scratch_0[t2143] = t2158;
        float t2160 = t2145 - t2153;
        scratch_1[t2143] = t2160;
      }
      for (uint t2163 = 0; t2163 < 512; t2163++) {
        float t2164 = (float)t2163;
        float t2165 = (t2164 * 0.015625);
        float t2166 = metal::floor(t2165);
        float t2167 = t2166 * 64.0;
        float t2168 = t2164 - t2167;
        float t2169 = t2166 * 128.0;
        float t2170 = t2169 + t2168;
        float t2171 = t2170 + 64.0;
        int t2172 = (int)t2168;
        int t2173 = 63 + t2172;
        float t2174 = scratch_2[t2173];
        float t2175 = scratch_3[t2173];
        float t2176 = 0.0 - t2175;
        int t2177 = (int)t2170;
        int t2178 = (int)t2171;
        float t2179 = scratch_0[t2177];
        float t2180 = scratch_1[t2177];
        float t2181 = scratch_0[t2178];
        float t2182 = scratch_1[t2178];
        float t2183 = t2174 * t2181;
        float t2184 = t2176 * t2182;
        float t2185 = t2183 - t2184;
        float t2186 = t2174 * t2182;
        float t2187 = t2176 * t2181;
        float t2188 = t2186 + t2187;
        float t2189 = t2179 + t2185;
        scratch_0[t2177] = t2189;
        float t2191 = t2180 + t2188;
        scratch_1[t2177] = t2191;
        float t2193 = t2179 - t2185;
        scratch_0[t2178] = t2193;
        float t2195 = t2180 - t2188;
        scratch_1[t2178] = t2195;
      }
      for (uint t2198 = 0; t2198 < 512; t2198++) {
        float t2199 = (float)t2198;
        float t2200 = (t2199 * 0.0078125);
        float t2201 = metal::floor(t2200);
        float t2202 = t2201 * 128.0;
        float t2203 = t2199 - t2202;
        float t2204 = t2201 * 256.0;
        float t2205 = t2204 + t2203;
        float t2206 = t2205 + 128.0;
        int t2207 = (int)t2203;
        int t2208 = 127 + t2207;
        float t2209 = scratch_2[t2208];
        float t2210 = scratch_3[t2208];
        float t2211 = 0.0 - t2210;
        int t2212 = (int)t2205;
        int t2213 = (int)t2206;
        float t2214 = scratch_0[t2212];
        float t2215 = scratch_1[t2212];
        float t2216 = scratch_0[t2213];
        float t2217 = scratch_1[t2213];
        float t2218 = t2209 * t2216;
        float t2219 = t2211 * t2217;
        float t2220 = t2218 - t2219;
        float t2221 = t2209 * t2217;
        float t2222 = t2211 * t2216;
        float t2223 = t2221 + t2222;
        float t2224 = t2214 + t2220;
        scratch_0[t2212] = t2224;
        float t2226 = t2215 + t2223;
        scratch_1[t2212] = t2226;
        float t2228 = t2214 - t2220;
        scratch_0[t2213] = t2228;
        float t2230 = t2215 - t2223;
        scratch_1[t2213] = t2230;
      }
      for (uint t2233 = 0; t2233 < 512; t2233++) {
        float t2234 = (float)t2233;
        float t2235 = (t2234 * 0.00390625);
        float t2236 = metal::floor(t2235);
        float t2237 = t2236 * 256.0;
        float t2238 = t2234 - t2237;
        float t2239 = t2236 * 512.0;
        float t2240 = t2239 + t2238;
        float t2241 = t2240 + 256.0;
        int t2242 = (int)t2238;
        int t2243 = 255 + t2242;
        float t2244 = scratch_2[t2243];
        float t2245 = scratch_3[t2243];
        float t2246 = 0.0 - t2245;
        int t2247 = (int)t2240;
        int t2248 = (int)t2241;
        float t2249 = scratch_0[t2247];
        float t2250 = scratch_1[t2247];
        float t2251 = scratch_0[t2248];
        float t2252 = scratch_1[t2248];
        float t2253 = t2244 * t2251;
        float t2254 = t2246 * t2252;
        float t2255 = t2253 - t2254;
        float t2256 = t2244 * t2252;
        float t2257 = t2246 * t2251;
        float t2258 = t2256 + t2257;
        float t2259 = t2249 + t2255;
        scratch_0[t2247] = t2259;
        float t2261 = t2250 + t2258;
        scratch_1[t2247] = t2261;
        float t2263 = t2249 - t2255;
        scratch_0[t2248] = t2263;
        float t2265 = t2250 - t2258;
        scratch_1[t2248] = t2265;
      }
      for (uint t2268 = 0; t2268 < 512; t2268++) {
        float t2269 = (float)t2268;
        float t2270 = (t2269 * 0.001953125);
        float t2271 = metal::floor(t2270);
        float t2272 = t2271 * 512.0;
        float t2273 = t2269 - t2272;
        float t2274 = t2271 * 1024.0;
        float t2275 = t2274 + t2273;
        float t2276 = t2275 + 512.0;
        int t2277 = (int)t2273;
        int t2278 = 511 + t2277;
        float t2279 = scratch_2[t2278];
        float t2280 = scratch_3[t2278];
        float t2281 = 0.0 - t2280;
        int t2282 = (int)t2275;
        int t2283 = (int)t2276;
        float t2284 = scratch_0[t2282];
        float t2285 = scratch_1[t2282];
        float t2286 = scratch_0[t2283];
        float t2287 = scratch_1[t2283];
        float t2288 = t2279 * t2286;
        float t2289 = t2281 * t2287;
        float t2290 = t2288 - t2289;
        float t2291 = t2279 * t2287;
        float t2292 = t2281 * t2286;
        float t2293 = t2291 + t2292;
        float t2294 = t2284 + t2290;
        scratch_0[t2282] = t2294;
        float t2296 = t2285 + t2293;
        scratch_1[t2282] = t2296;
        float t2298 = t2284 - t2290;
        scratch_0[t2283] = t2298;
        float t2300 = t2285 - t2293;
        scratch_1[t2283] = t2300;
      }
      for (uint t2303 = 0; t2303 < 1024; t2303++) {
        float t2304 = scratch_0[(int)t2303];
        float t2305 = scratch_1[(int)t2303];
        int t2306 = t1912 + t2303;
        memory[30382528 + t2306] = t2304;
        int t2308 = t1912 + t2303;
        int t2309 = t2308 + 1024;
        memory[30382528 + t2309] = t2305;
      }
      for (uint t2312 = 0; t2312 < 513; t2312++) {
        float t2313 = scratch_0[(int)t2312];
        float t2314 = scratch_1[(int)t2312];
        float t2315 = t2313 * t2313;
        float t2316 = t2314 * t2314;
        float t2317 = t2315 + t2316;
        float t2318 = metal::sqrt(t2317);
        int t2319 = t1913 + t2312;
        memory[31693760 + t2319] = t2318;
      }
      int t2322 = t1485 / 256;
      int t2323 = t2322 * 8;
      int t2324 = t2323 + t1487;
      int t2325 = t2324 * 513;
      float t2326 = 0.0;
      for (uint t2327 = 0; t2327 < 513; t2327++) {
        int t2328 = t2325 + t2327;
        float t2329 = memory[31431104 + t2328];
        int t2330 = t2325 + t2327;
        float t2331 = memory[31693760 + t2330];
        float t2332 = t2329 - t2331;
        float t2333 = t2332 * t2332;
        float t2334 = t2326 + t2333;
        t2326 = t2334;
      }
      memory[625088 + t2324] = t2326;
    }
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2351), value: global(2351)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    int t2338 = id;
    int t2339 = t2338 / 256;
    int t2340 = t2338 % 256;
    int t2341 = t2340 == 0.0;
    float t2342 = 0.0;
    if (t2341) {
      for (uint t2344 = 0; t2344 < 8; t2344++) {
        int t2345 = t2339 * 8;
        int t2346 = t2345 + t2344;
        float t2347 = memory[625088 + t2346];
        float t2348 = t2342 + t2347;
        t2342 = t2348;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    t[7*frameCount + id] = (t2342 * 0.125);
  }
  #pragma clang diagnostic pop
}



// KERNEL 26
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_26(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2356), value: global(2356)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2351) - handled in variable access */
    /* loadGlobal(1458) - handled in variable access */
    float t2352 = (t[7*frameCount + id] * 6.1035156e-05);
    float t2353 = t[5*frameCount + id] + t2352;
    float t2354 = t2353 * 0.5;
    float t2355 = t2354;
    t[8*frameCount + id] = t2355;
    float t2357 = t2354;
    float t2358 = t2353;
    float t2359 = (t[7*frameCount + id] * 3.7252903e-09);
    float t2360 = -0.5 * t2359;
  }
  #pragma clang diagnostic pop
}



// KERNEL 27
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_27(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4933 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4933)) {
    /* loadGlobal(1467) - handled in variable access */
    int t2361 = id;
    int t2362 = t2361 / 8;
    uint _frameIndex = (uint)(t2362);
    int t2363 = t2362 * 8;
    int t2364 = t2361 - t2363;
    float t2365 = t[6*frameCount + _frameIndex] == 0.0;
    if (t2365) {
      int t2367 = t2362 / 256;
      int t2368 = t2367 * 8;
      int t2369 = t2368 + t2364;
      int t2370 = t2369 * 2048;
      int t2371 = t2369 * 513;
      int t2372 = t2369 * 2048;
      for (uint _pr2373 = 0; _pr2373 < 513; _pr2373++) {
        int t2374 = t2371 + _pr2373;
        float t2375 = memory[31431104 + t2374];
        int t2376 = t2371 + _pr2373;
        float t2377 = memory[31693760 + t2376];
        int t2378 = t2370 + _pr2373;
        float t2379 = memory[29333952 + t2378];
        int t2380 = t2370 + _pr2373;
        int t2381 = t2380 + 1024;
        float t2382 = memory[29333952 + t2381];
        int t2383 = t2370 + _pr2373;
        float t2384 = memory[30382528 + t2383];
        int t2385 = t2370 + _pr2373;
        int t2386 = t2385 + 1024;
        float t2387 = memory[30382528 + t2386];
        float t2388 = t2375 - t2377;
        float t2389 = 2.0 * t2388;
        float t2390 = t2389 * 3.8146973e-06;
        float t2391 = t2375 - t2377;
        float t2392 = -2.0 * t2391;
        float t2393 = t2392 * 3.8146973e-06;
        float t2394 = metal::max(t2375, 1e-08);
        float t2395 = metal::max(t2377, 1e-08);
        float t2396 = t2390 * t2379;
        float t2397 = t2396 / t2394;
        float t2398 = t2390 * t2382;
        float t2399 = t2398 / t2394;
        float t2400 = t2393 * t2384;
        float t2401 = t2400 / t2395;
        float t2402 = t2393 * t2387;
        float t2403 = t2402 / t2395;
        int t2404 = t2372 + _pr2373;
        memory[18059712 + t2404] = t2397;
        int t2406 = t2372 + _pr2373;
        int t2407 = t2406 + 1024;
        memory[18059712 + t2407] = t2399;
        int t2409 = t2372 + _pr2373;
        memory[31956416 + t2409] = t2401;
        int t2411 = t2372 + _pr2373;
        int t2412 = t2411 + 1024;
        memory[31956416 + t2412] = t2403;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2415 = 0; _pr2415 < 511; _pr2415++) {
        int t2416 = _pr2415 + 513;
        int t2417 = t2372 + t2416;
        memory[18059712 + t2417] = 0.0;
        int t2419 = t2372 + t2416;
        int t2420 = t2419 + 1024;
        memory[18059712 + t2420] = 0.0;
        int t2422 = t2372 + t2416;
        memory[31956416 + t2422] = 0.0;
        int t2424 = t2372 + t2416;
        int t2425 = t2424 + 1024;
        memory[31956416 + t2425] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 28
// FrameOrder: parallel
// DispatchMode: perFrameScaledThreadgroup1(8)
kernel void kernel_28(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4934 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4934)) {
    /* loadGlobal(1467) - handled in variable access */
    int t2429 = id;
    int t2430 = t2429 / 8;
    uint _frameIndex = (uint)(t2430);
    int t2431 = t2430 * 8;
    int t2432 = t2429 - t2431;
    threadgroup float scratch_0[1024];
    threadgroup float scratch_1[1024];
    threadgroup float scratch_2[1023];
    threadgroup float scratch_3[1023];
    float t2433 = t[6*frameCount + _frameIndex] == 0.0;
    if (t2433) {
      for (uint t2435 = 0; t2435 < 1023; t2435++) {
        float t2436 = memory[162752 + (int)t2435];
        scratch_2[(int)t2435] = t2436;
        float t2438 = memory[163775 + (int)t2435];
        scratch_3[(int)t2435] = t2438;
      }
      int t2441 = t2430 / 256;
      int t2442 = t2441 * 8;
      int t2443 = t2442 + t2432;
      int t2444 = t2443 * 2048;
      int t2445 = t2443 * 1024;
      for (uint t2446 = 0; t2446 < 1024; t2446++) {
        int t2447 = t2444 + t2446;
        float t2448 = memory[18059712 + t2447];
        int t2449 = t2444 + t2446;
        int t2450 = t2449 + 1024;
        float t2451 = memory[18059712 + t2450];
        scratch_0[(int)t2446] = t2448;
        scratch_1[(int)t2446] = t2451;
      }
      for (uint t2455 = 0; t2455 < 1024; t2455++) {
        float t2456 = memory[164798 + (int)t2455];
        float t2457 = (float)t2455;
        float t2458 = t2457 < t2456;
        int t2459 = (int)t2456;
        float t2460 = scratch_0[(int)t2455];
        float t2461 = scratch_1[(int)t2455];
        float t2462 = scratch_0[t2459];
        float t2463 = scratch_1[t2459];
        float t2464 = metal::select(t2460, t2462, t2458 > 0.0);
        float t2465 = metal::select(t2461, t2463, t2458 > 0.0);
        float t2466 = metal::select(t2462, t2460, t2458 > 0.0);
        float t2467 = metal::select(t2463, t2461, t2458 > 0.0);
        scratch_0[(int)t2455] = t2464;
        scratch_1[(int)t2455] = t2465;
        scratch_0[t2459] = t2466;
        scratch_1[t2459] = t2467;
      }
      for (uint t2473 = 0; t2473 < 512; t2473++) {
        float t2474 = (float)t2473;
        float t2475 = t2474;
        float t2476 = metal::floor(t2475);
        float t2477 = t2476;
        float t2478 = t2474 - t2477;
        float t2479 = t2476 * 2.0;
        float t2480 = t2479 + t2478;
        float t2481 = t2480 + 1.0;
        int t2482 = (int)t2478;
        int t2483 = t2482;
        float t2484 = scratch_2[t2483];
        float t2485 = scratch_3[t2483];
        int t2486 = (int)t2480;
        int t2487 = (int)t2481;
        float t2488 = scratch_0[t2486];
        float t2489 = scratch_1[t2486];
        float t2490 = scratch_0[t2487];
        float t2491 = scratch_1[t2487];
        float t2492 = t2484 * t2490;
        float t2493 = t2485 * t2491;
        float t2494 = t2492 - t2493;
        float t2495 = t2484 * t2491;
        float t2496 = t2485 * t2490;
        float t2497 = t2495 + t2496;
        float t2498 = t2488 + t2494;
        scratch_0[t2486] = t2498;
        float t2500 = t2489 + t2497;
        scratch_1[t2486] = t2500;
        float t2502 = t2488 - t2494;
        scratch_0[t2487] = t2502;
        float t2504 = t2489 - t2497;
        scratch_1[t2487] = t2504;
      }
      for (uint t2507 = 0; t2507 < 512; t2507++) {
        float t2508 = (float)t2507;
        float t2509 = (t2508 * 0.5);
        float t2510 = metal::floor(t2509);
        float t2511 = t2510 * 2.0;
        float t2512 = t2508 - t2511;
        float t2513 = t2510 * 4.0;
        float t2514 = t2513 + t2512;
        float t2515 = t2514 + 2.0;
        int t2516 = (int)t2512;
        int t2517 = 1 + t2516;
        float t2518 = scratch_2[t2517];
        float t2519 = scratch_3[t2517];
        int t2520 = (int)t2514;
        int t2521 = (int)t2515;
        float t2522 = scratch_0[t2520];
        float t2523 = scratch_1[t2520];
        float t2524 = scratch_0[t2521];
        float t2525 = scratch_1[t2521];
        float t2526 = t2518 * t2524;
        float t2527 = t2519 * t2525;
        float t2528 = t2526 - t2527;
        float t2529 = t2518 * t2525;
        float t2530 = t2519 * t2524;
        float t2531 = t2529 + t2530;
        float t2532 = t2522 + t2528;
        scratch_0[t2520] = t2532;
        float t2534 = t2523 + t2531;
        scratch_1[t2520] = t2534;
        float t2536 = t2522 - t2528;
        scratch_0[t2521] = t2536;
        float t2538 = t2523 - t2531;
        scratch_1[t2521] = t2538;
      }
      for (uint t2541 = 0; t2541 < 512; t2541++) {
        float t2542 = (float)t2541;
        float t2543 = (t2542 * 0.25);
        float t2544 = metal::floor(t2543);
        float t2545 = t2544 * 4.0;
        float t2546 = t2542 - t2545;
        float t2547 = t2544 * 8.0;
        float t2548 = t2547 + t2546;
        float t2549 = t2548 + 4.0;
        int t2550 = (int)t2546;
        int t2551 = 3 + t2550;
        float t2552 = scratch_2[t2551];
        float t2553 = scratch_3[t2551];
        int t2554 = (int)t2548;
        int t2555 = (int)t2549;
        float t2556 = scratch_0[t2554];
        float t2557 = scratch_1[t2554];
        float t2558 = scratch_0[t2555];
        float t2559 = scratch_1[t2555];
        float t2560 = t2552 * t2558;
        float t2561 = t2553 * t2559;
        float t2562 = t2560 - t2561;
        float t2563 = t2552 * t2559;
        float t2564 = t2553 * t2558;
        float t2565 = t2563 + t2564;
        float t2566 = t2556 + t2562;
        scratch_0[t2554] = t2566;
        float t2568 = t2557 + t2565;
        scratch_1[t2554] = t2568;
        float t2570 = t2556 - t2562;
        scratch_0[t2555] = t2570;
        float t2572 = t2557 - t2565;
        scratch_1[t2555] = t2572;
      }
      for (uint t2575 = 0; t2575 < 512; t2575++) {
        float t2576 = (float)t2575;
        float t2577 = (t2576 * 0.125);
        float t2578 = metal::floor(t2577);
        float t2579 = t2578 * 8.0;
        float t2580 = t2576 - t2579;
        float t2581 = t2578 * 16.0;
        float t2582 = t2581 + t2580;
        float t2583 = t2582 + 8.0;
        int t2584 = (int)t2580;
        int t2585 = 7 + t2584;
        float t2586 = scratch_2[t2585];
        float t2587 = scratch_3[t2585];
        int t2588 = (int)t2582;
        int t2589 = (int)t2583;
        float t2590 = scratch_0[t2588];
        float t2591 = scratch_1[t2588];
        float t2592 = scratch_0[t2589];
        float t2593 = scratch_1[t2589];
        float t2594 = t2586 * t2592;
        float t2595 = t2587 * t2593;
        float t2596 = t2594 - t2595;
        float t2597 = t2586 * t2593;
        float t2598 = t2587 * t2592;
        float t2599 = t2597 + t2598;
        float t2600 = t2590 + t2596;
        scratch_0[t2588] = t2600;
        float t2602 = t2591 + t2599;
        scratch_1[t2588] = t2602;
        float t2604 = t2590 - t2596;
        scratch_0[t2589] = t2604;
        float t2606 = t2591 - t2599;
        scratch_1[t2589] = t2606;
      }
      for (uint t2609 = 0; t2609 < 512; t2609++) {
        float t2610 = (float)t2609;
        float t2611 = (t2610 * 0.0625);
        float t2612 = metal::floor(t2611);
        float t2613 = t2612 * 16.0;
        float t2614 = t2610 - t2613;
        float t2615 = t2612 * 32.0;
        float t2616 = t2615 + t2614;
        float t2617 = t2616 + 16.0;
        int t2618 = (int)t2614;
        int t2619 = 15 + t2618;
        float t2620 = scratch_2[t2619];
        float t2621 = scratch_3[t2619];
        int t2622 = (int)t2616;
        int t2623 = (int)t2617;
        float t2624 = scratch_0[t2622];
        float t2625 = scratch_1[t2622];
        float t2626 = scratch_0[t2623];
        float t2627 = scratch_1[t2623];
        float t2628 = t2620 * t2626;
        float t2629 = t2621 * t2627;
        float t2630 = t2628 - t2629;
        float t2631 = t2620 * t2627;
        float t2632 = t2621 * t2626;
        float t2633 = t2631 + t2632;
        float t2634 = t2624 + t2630;
        scratch_0[t2622] = t2634;
        float t2636 = t2625 + t2633;
        scratch_1[t2622] = t2636;
        float t2638 = t2624 - t2630;
        scratch_0[t2623] = t2638;
        float t2640 = t2625 - t2633;
        scratch_1[t2623] = t2640;
      }
      for (uint t2643 = 0; t2643 < 512; t2643++) {
        float t2644 = (float)t2643;
        float t2645 = (t2644 * 0.03125);
        float t2646 = metal::floor(t2645);
        float t2647 = t2646 * 32.0;
        float t2648 = t2644 - t2647;
        float t2649 = t2646 * 64.0;
        float t2650 = t2649 + t2648;
        float t2651 = t2650 + 32.0;
        int t2652 = (int)t2648;
        int t2653 = 31 + t2652;
        float t2654 = scratch_2[t2653];
        float t2655 = scratch_3[t2653];
        int t2656 = (int)t2650;
        int t2657 = (int)t2651;
        float t2658 = scratch_0[t2656];
        float t2659 = scratch_1[t2656];
        float t2660 = scratch_0[t2657];
        float t2661 = scratch_1[t2657];
        float t2662 = t2654 * t2660;
        float t2663 = t2655 * t2661;
        float t2664 = t2662 - t2663;
        float t2665 = t2654 * t2661;
        float t2666 = t2655 * t2660;
        float t2667 = t2665 + t2666;
        float t2668 = t2658 + t2664;
        scratch_0[t2656] = t2668;
        float t2670 = t2659 + t2667;
        scratch_1[t2656] = t2670;
        float t2672 = t2658 - t2664;
        scratch_0[t2657] = t2672;
        float t2674 = t2659 - t2667;
        scratch_1[t2657] = t2674;
      }
      for (uint t2677 = 0; t2677 < 512; t2677++) {
        float t2678 = (float)t2677;
        float t2679 = (t2678 * 0.015625);
        float t2680 = metal::floor(t2679);
        float t2681 = t2680 * 64.0;
        float t2682 = t2678 - t2681;
        float t2683 = t2680 * 128.0;
        float t2684 = t2683 + t2682;
        float t2685 = t2684 + 64.0;
        int t2686 = (int)t2682;
        int t2687 = 63 + t2686;
        float t2688 = scratch_2[t2687];
        float t2689 = scratch_3[t2687];
        int t2690 = (int)t2684;
        int t2691 = (int)t2685;
        float t2692 = scratch_0[t2690];
        float t2693 = scratch_1[t2690];
        float t2694 = scratch_0[t2691];
        float t2695 = scratch_1[t2691];
        float t2696 = t2688 * t2694;
        float t2697 = t2689 * t2695;
        float t2698 = t2696 - t2697;
        float t2699 = t2688 * t2695;
        float t2700 = t2689 * t2694;
        float t2701 = t2699 + t2700;
        float t2702 = t2692 + t2698;
        scratch_0[t2690] = t2702;
        float t2704 = t2693 + t2701;
        scratch_1[t2690] = t2704;
        float t2706 = t2692 - t2698;
        scratch_0[t2691] = t2706;
        float t2708 = t2693 - t2701;
        scratch_1[t2691] = t2708;
      }
      for (uint t2711 = 0; t2711 < 512; t2711++) {
        float t2712 = (float)t2711;
        float t2713 = (t2712 * 0.0078125);
        float t2714 = metal::floor(t2713);
        float t2715 = t2714 * 128.0;
        float t2716 = t2712 - t2715;
        float t2717 = t2714 * 256.0;
        float t2718 = t2717 + t2716;
        float t2719 = t2718 + 128.0;
        int t2720 = (int)t2716;
        int t2721 = 127 + t2720;
        float t2722 = scratch_2[t2721];
        float t2723 = scratch_3[t2721];
        int t2724 = (int)t2718;
        int t2725 = (int)t2719;
        float t2726 = scratch_0[t2724];
        float t2727 = scratch_1[t2724];
        float t2728 = scratch_0[t2725];
        float t2729 = scratch_1[t2725];
        float t2730 = t2722 * t2728;
        float t2731 = t2723 * t2729;
        float t2732 = t2730 - t2731;
        float t2733 = t2722 * t2729;
        float t2734 = t2723 * t2728;
        float t2735 = t2733 + t2734;
        float t2736 = t2726 + t2732;
        scratch_0[t2724] = t2736;
        float t2738 = t2727 + t2735;
        scratch_1[t2724] = t2738;
        float t2740 = t2726 - t2732;
        scratch_0[t2725] = t2740;
        float t2742 = t2727 - t2735;
        scratch_1[t2725] = t2742;
      }
      for (uint t2745 = 0; t2745 < 512; t2745++) {
        float t2746 = (float)t2745;
        float t2747 = (t2746 * 0.00390625);
        float t2748 = metal::floor(t2747);
        float t2749 = t2748 * 256.0;
        float t2750 = t2746 - t2749;
        float t2751 = t2748 * 512.0;
        float t2752 = t2751 + t2750;
        float t2753 = t2752 + 256.0;
        int t2754 = (int)t2750;
        int t2755 = 255 + t2754;
        float t2756 = scratch_2[t2755];
        float t2757 = scratch_3[t2755];
        int t2758 = (int)t2752;
        int t2759 = (int)t2753;
        float t2760 = scratch_0[t2758];
        float t2761 = scratch_1[t2758];
        float t2762 = scratch_0[t2759];
        float t2763 = scratch_1[t2759];
        float t2764 = t2756 * t2762;
        float t2765 = t2757 * t2763;
        float t2766 = t2764 - t2765;
        float t2767 = t2756 * t2763;
        float t2768 = t2757 * t2762;
        float t2769 = t2767 + t2768;
        float t2770 = t2760 + t2766;
        scratch_0[t2758] = t2770;
        float t2772 = t2761 + t2769;
        scratch_1[t2758] = t2772;
        float t2774 = t2760 - t2766;
        scratch_0[t2759] = t2774;
        float t2776 = t2761 - t2769;
        scratch_1[t2759] = t2776;
      }
      for (uint t2779 = 0; t2779 < 512; t2779++) {
        float t2780 = (float)t2779;
        float t2781 = (t2780 * 0.001953125);
        float t2782 = metal::floor(t2781);
        float t2783 = t2782 * 512.0;
        float t2784 = t2780 - t2783;
        float t2785 = t2782 * 1024.0;
        float t2786 = t2785 + t2784;
        float t2787 = t2786 + 512.0;
        int t2788 = (int)t2784;
        int t2789 = 511 + t2788;
        float t2790 = scratch_2[t2789];
        float t2791 = scratch_3[t2789];
        int t2792 = (int)t2786;
        int t2793 = (int)t2787;
        float t2794 = scratch_0[t2792];
        float t2795 = scratch_1[t2792];
        float t2796 = scratch_0[t2793];
        float t2797 = scratch_1[t2793];
        float t2798 = t2790 * t2796;
        float t2799 = t2791 * t2797;
        float t2800 = t2798 - t2799;
        float t2801 = t2790 * t2797;
        float t2802 = t2791 * t2796;
        float t2803 = t2801 + t2802;
        float t2804 = t2794 + t2800;
        scratch_0[t2792] = t2804;
        float t2806 = t2795 + t2803;
        scratch_1[t2792] = t2806;
        float t2808 = t2794 - t2800;
        scratch_0[t2793] = t2808;
        float t2810 = t2795 - t2803;
        scratch_1[t2793] = t2810;
      }
      for (uint t2813 = 0; t2813 < 1024; t2813++) {
        float t2814 = scratch_0[(int)t2813];
        float t2815 = t2814 * 1.9036306e-06;
        float t2816 = memory[161728 + (int)t2813];
        int t2817 = t2445 + t2813;
        float t2818 = t2815 * t2816;
        memory[29333952 + t2817] = t2818;
      }
      int t2821 = t2430 / 256;
      int t2822 = t2821 * 8;
      int t2823 = t2822 + t2432;
      int t2824 = t2823 * 2048;
      int t2825 = t2823 * 1024;
      for (uint t2826 = 0; t2826 < 1024; t2826++) {
        int t2827 = t2824 + t2826;
        float t2828 = memory[31956416 + t2827];
        int t2829 = t2824 + t2826;
        int t2830 = t2829 + 1024;
        float t2831 = memory[31956416 + t2830];
        scratch_0[(int)t2826] = t2828;
        scratch_1[(int)t2826] = t2831;
      }
      for (uint t2835 = 0; t2835 < 1024; t2835++) {
        float t2836 = memory[164798 + (int)t2835];
        float t2837 = (float)t2835;
        float t2838 = t2837 < t2836;
        int t2839 = (int)t2836;
        float t2840 = scratch_0[(int)t2835];
        float t2841 = scratch_1[(int)t2835];
        float t2842 = scratch_0[t2839];
        float t2843 = scratch_1[t2839];
        float t2844 = metal::select(t2840, t2842, t2838 > 0.0);
        float t2845 = metal::select(t2841, t2843, t2838 > 0.0);
        float t2846 = metal::select(t2842, t2840, t2838 > 0.0);
        float t2847 = metal::select(t2843, t2841, t2838 > 0.0);
        scratch_0[(int)t2835] = t2844;
        scratch_1[(int)t2835] = t2845;
        scratch_0[t2839] = t2846;
        scratch_1[t2839] = t2847;
      }
      for (uint t2853 = 0; t2853 < 512; t2853++) {
        float t2854 = (float)t2853;
        float t2855 = t2854;
        float t2856 = metal::floor(t2855);
        float t2857 = t2856;
        float t2858 = t2854 - t2857;
        float t2859 = t2856 * 2.0;
        float t2860 = t2859 + t2858;
        float t2861 = t2860 + 1.0;
        int t2862 = (int)t2858;
        int t2863 = t2862;
        float t2864 = scratch_2[t2863];
        float t2865 = scratch_3[t2863];
        int t2866 = (int)t2860;
        int t2867 = (int)t2861;
        float t2868 = scratch_0[t2866];
        float t2869 = scratch_1[t2866];
        float t2870 = scratch_0[t2867];
        float t2871 = scratch_1[t2867];
        float t2872 = t2864 * t2870;
        float t2873 = t2865 * t2871;
        float t2874 = t2872 - t2873;
        float t2875 = t2864 * t2871;
        float t2876 = t2865 * t2870;
        float t2877 = t2875 + t2876;
        float t2878 = t2868 + t2874;
        scratch_0[t2866] = t2878;
        float t2880 = t2869 + t2877;
        scratch_1[t2866] = t2880;
        float t2882 = t2868 - t2874;
        scratch_0[t2867] = t2882;
        float t2884 = t2869 - t2877;
        scratch_1[t2867] = t2884;
      }
      for (uint t2887 = 0; t2887 < 512; t2887++) {
        float t2888 = (float)t2887;
        float t2889 = (t2888 * 0.5);
        float t2890 = metal::floor(t2889);
        float t2891 = t2890 * 2.0;
        float t2892 = t2888 - t2891;
        float t2893 = t2890 * 4.0;
        float t2894 = t2893 + t2892;
        float t2895 = t2894 + 2.0;
        int t2896 = (int)t2892;
        int t2897 = 1 + t2896;
        float t2898 = scratch_2[t2897];
        float t2899 = scratch_3[t2897];
        int t2900 = (int)t2894;
        int t2901 = (int)t2895;
        float t2902 = scratch_0[t2900];
        float t2903 = scratch_1[t2900];
        float t2904 = scratch_0[t2901];
        float t2905 = scratch_1[t2901];
        float t2906 = t2898 * t2904;
        float t2907 = t2899 * t2905;
        float t2908 = t2906 - t2907;
        float t2909 = t2898 * t2905;
        float t2910 = t2899 * t2904;
        float t2911 = t2909 + t2910;
        float t2912 = t2902 + t2908;
        scratch_0[t2900] = t2912;
        float t2914 = t2903 + t2911;
        scratch_1[t2900] = t2914;
        float t2916 = t2902 - t2908;
        scratch_0[t2901] = t2916;
        float t2918 = t2903 - t2911;
        scratch_1[t2901] = t2918;
      }
      for (uint t2921 = 0; t2921 < 512; t2921++) {
        float t2922 = (float)t2921;
        float t2923 = (t2922 * 0.25);
        float t2924 = metal::floor(t2923);
        float t2925 = t2924 * 4.0;
        float t2926 = t2922 - t2925;
        float t2927 = t2924 * 8.0;
        float t2928 = t2927 + t2926;
        float t2929 = t2928 + 4.0;
        int t2930 = (int)t2926;
        int t2931 = 3 + t2930;
        float t2932 = scratch_2[t2931];
        float t2933 = scratch_3[t2931];
        int t2934 = (int)t2928;
        int t2935 = (int)t2929;
        float t2936 = scratch_0[t2934];
        float t2937 = scratch_1[t2934];
        float t2938 = scratch_0[t2935];
        float t2939 = scratch_1[t2935];
        float t2940 = t2932 * t2938;
        float t2941 = t2933 * t2939;
        float t2942 = t2940 - t2941;
        float t2943 = t2932 * t2939;
        float t2944 = t2933 * t2938;
        float t2945 = t2943 + t2944;
        float t2946 = t2936 + t2942;
        scratch_0[t2934] = t2946;
        float t2948 = t2937 + t2945;
        scratch_1[t2934] = t2948;
        float t2950 = t2936 - t2942;
        scratch_0[t2935] = t2950;
        float t2952 = t2937 - t2945;
        scratch_1[t2935] = t2952;
      }
      for (uint t2955 = 0; t2955 < 512; t2955++) {
        float t2956 = (float)t2955;
        float t2957 = (t2956 * 0.125);
        float t2958 = metal::floor(t2957);
        float t2959 = t2958 * 8.0;
        float t2960 = t2956 - t2959;
        float t2961 = t2958 * 16.0;
        float t2962 = t2961 + t2960;
        float t2963 = t2962 + 8.0;
        int t2964 = (int)t2960;
        int t2965 = 7 + t2964;
        float t2966 = scratch_2[t2965];
        float t2967 = scratch_3[t2965];
        int t2968 = (int)t2962;
        int t2969 = (int)t2963;
        float t2970 = scratch_0[t2968];
        float t2971 = scratch_1[t2968];
        float t2972 = scratch_0[t2969];
        float t2973 = scratch_1[t2969];
        float t2974 = t2966 * t2972;
        float t2975 = t2967 * t2973;
        float t2976 = t2974 - t2975;
        float t2977 = t2966 * t2973;
        float t2978 = t2967 * t2972;
        float t2979 = t2977 + t2978;
        float t2980 = t2970 + t2976;
        scratch_0[t2968] = t2980;
        float t2982 = t2971 + t2979;
        scratch_1[t2968] = t2982;
        float t2984 = t2970 - t2976;
        scratch_0[t2969] = t2984;
        float t2986 = t2971 - t2979;
        scratch_1[t2969] = t2986;
      }
      for (uint t2989 = 0; t2989 < 512; t2989++) {
        float t2990 = (float)t2989;
        float t2991 = (t2990 * 0.0625);
        float t2992 = metal::floor(t2991);
        float t2993 = t2992 * 16.0;
        float t2994 = t2990 - t2993;
        float t2995 = t2992 * 32.0;
        float t2996 = t2995 + t2994;
        float t2997 = t2996 + 16.0;
        int t2998 = (int)t2994;
        int t2999 = 15 + t2998;
        float t3000 = scratch_2[t2999];
        float t3001 = scratch_3[t2999];
        int t3002 = (int)t2996;
        int t3003 = (int)t2997;
        float t3004 = scratch_0[t3002];
        float t3005 = scratch_1[t3002];
        float t3006 = scratch_0[t3003];
        float t3007 = scratch_1[t3003];
        float t3008 = t3000 * t3006;
        float t3009 = t3001 * t3007;
        float t3010 = t3008 - t3009;
        float t3011 = t3000 * t3007;
        float t3012 = t3001 * t3006;
        float t3013 = t3011 + t3012;
        float t3014 = t3004 + t3010;
        scratch_0[t3002] = t3014;
        float t3016 = t3005 + t3013;
        scratch_1[t3002] = t3016;
        float t3018 = t3004 - t3010;
        scratch_0[t3003] = t3018;
        float t3020 = t3005 - t3013;
        scratch_1[t3003] = t3020;
      }
      for (uint t3023 = 0; t3023 < 512; t3023++) {
        float t3024 = (float)t3023;
        float t3025 = (t3024 * 0.03125);
        float t3026 = metal::floor(t3025);
        float t3027 = t3026 * 32.0;
        float t3028 = t3024 - t3027;
        float t3029 = t3026 * 64.0;
        float t3030 = t3029 + t3028;
        float t3031 = t3030 + 32.0;
        int t3032 = (int)t3028;
        int t3033 = 31 + t3032;
        float t3034 = scratch_2[t3033];
        float t3035 = scratch_3[t3033];
        int t3036 = (int)t3030;
        int t3037 = (int)t3031;
        float t3038 = scratch_0[t3036];
        float t3039 = scratch_1[t3036];
        float t3040 = scratch_0[t3037];
        float t3041 = scratch_1[t3037];
        float t3042 = t3034 * t3040;
        float t3043 = t3035 * t3041;
        float t3044 = t3042 - t3043;
        float t3045 = t3034 * t3041;
        float t3046 = t3035 * t3040;
        float t3047 = t3045 + t3046;
        float t3048 = t3038 + t3044;
        scratch_0[t3036] = t3048;
        float t3050 = t3039 + t3047;
        scratch_1[t3036] = t3050;
        float t3052 = t3038 - t3044;
        scratch_0[t3037] = t3052;
        float t3054 = t3039 - t3047;
        scratch_1[t3037] = t3054;
      }
      for (uint t3057 = 0; t3057 < 512; t3057++) {
        float t3058 = (float)t3057;
        float t3059 = (t3058 * 0.015625);
        float t3060 = metal::floor(t3059);
        float t3061 = t3060 * 64.0;
        float t3062 = t3058 - t3061;
        float t3063 = t3060 * 128.0;
        float t3064 = t3063 + t3062;
        float t3065 = t3064 + 64.0;
        int t3066 = (int)t3062;
        int t3067 = 63 + t3066;
        float t3068 = scratch_2[t3067];
        float t3069 = scratch_3[t3067];
        int t3070 = (int)t3064;
        int t3071 = (int)t3065;
        float t3072 = scratch_0[t3070];
        float t3073 = scratch_1[t3070];
        float t3074 = scratch_0[t3071];
        float t3075 = scratch_1[t3071];
        float t3076 = t3068 * t3074;
        float t3077 = t3069 * t3075;
        float t3078 = t3076 - t3077;
        float t3079 = t3068 * t3075;
        float t3080 = t3069 * t3074;
        float t3081 = t3079 + t3080;
        float t3082 = t3072 + t3078;
        scratch_0[t3070] = t3082;
        float t3084 = t3073 + t3081;
        scratch_1[t3070] = t3084;
        float t3086 = t3072 - t3078;
        scratch_0[t3071] = t3086;
        float t3088 = t3073 - t3081;
        scratch_1[t3071] = t3088;
      }
      for (uint t3091 = 0; t3091 < 512; t3091++) {
        float t3092 = (float)t3091;
        float t3093 = (t3092 * 0.0078125);
        float t3094 = metal::floor(t3093);
        float t3095 = t3094 * 128.0;
        float t3096 = t3092 - t3095;
        float t3097 = t3094 * 256.0;
        float t3098 = t3097 + t3096;
        float t3099 = t3098 + 128.0;
        int t3100 = (int)t3096;
        int t3101 = 127 + t3100;
        float t3102 = scratch_2[t3101];
        float t3103 = scratch_3[t3101];
        int t3104 = (int)t3098;
        int t3105 = (int)t3099;
        float t3106 = scratch_0[t3104];
        float t3107 = scratch_1[t3104];
        float t3108 = scratch_0[t3105];
        float t3109 = scratch_1[t3105];
        float t3110 = t3102 * t3108;
        float t3111 = t3103 * t3109;
        float t3112 = t3110 - t3111;
        float t3113 = t3102 * t3109;
        float t3114 = t3103 * t3108;
        float t3115 = t3113 + t3114;
        float t3116 = t3106 + t3112;
        scratch_0[t3104] = t3116;
        float t3118 = t3107 + t3115;
        scratch_1[t3104] = t3118;
        float t3120 = t3106 - t3112;
        scratch_0[t3105] = t3120;
        float t3122 = t3107 - t3115;
        scratch_1[t3105] = t3122;
      }
      for (uint t3125 = 0; t3125 < 512; t3125++) {
        float t3126 = (float)t3125;
        float t3127 = (t3126 * 0.00390625);
        float t3128 = metal::floor(t3127);
        float t3129 = t3128 * 256.0;
        float t3130 = t3126 - t3129;
        float t3131 = t3128 * 512.0;
        float t3132 = t3131 + t3130;
        float t3133 = t3132 + 256.0;
        int t3134 = (int)t3130;
        int t3135 = 255 + t3134;
        float t3136 = scratch_2[t3135];
        float t3137 = scratch_3[t3135];
        int t3138 = (int)t3132;
        int t3139 = (int)t3133;
        float t3140 = scratch_0[t3138];
        float t3141 = scratch_1[t3138];
        float t3142 = scratch_0[t3139];
        float t3143 = scratch_1[t3139];
        float t3144 = t3136 * t3142;
        float t3145 = t3137 * t3143;
        float t3146 = t3144 - t3145;
        float t3147 = t3136 * t3143;
        float t3148 = t3137 * t3142;
        float t3149 = t3147 + t3148;
        float t3150 = t3140 + t3146;
        scratch_0[t3138] = t3150;
        float t3152 = t3141 + t3149;
        scratch_1[t3138] = t3152;
        float t3154 = t3140 - t3146;
        scratch_0[t3139] = t3154;
        float t3156 = t3141 - t3149;
        scratch_1[t3139] = t3156;
      }
      for (uint t3159 = 0; t3159 < 512; t3159++) {
        float t3160 = (float)t3159;
        float t3161 = (t3160 * 0.001953125);
        float t3162 = metal::floor(t3161);
        float t3163 = t3162 * 512.0;
        float t3164 = t3160 - t3163;
        float t3165 = t3162 * 1024.0;
        float t3166 = t3165 + t3164;
        float t3167 = t3166 + 512.0;
        int t3168 = (int)t3164;
        int t3169 = 511 + t3168;
        float t3170 = scratch_2[t3169];
        float t3171 = scratch_3[t3169];
        int t3172 = (int)t3166;
        int t3173 = (int)t3167;
        float t3174 = scratch_0[t3172];
        float t3175 = scratch_1[t3172];
        float t3176 = scratch_0[t3173];
        float t3177 = scratch_1[t3173];
        float t3178 = t3170 * t3176;
        float t3179 = t3171 * t3177;
        float t3180 = t3178 - t3179;
        float t3181 = t3170 * t3177;
        float t3182 = t3171 * t3176;
        float t3183 = t3181 + t3182;
        float t3184 = t3174 + t3180;
        scratch_0[t3172] = t3184;
        float t3186 = t3175 + t3183;
        scratch_1[t3172] = t3186;
        float t3188 = t3174 - t3180;
        scratch_0[t3173] = t3188;
        float t3190 = t3175 - t3183;
        scratch_1[t3173] = t3190;
      }
      for (uint t3193 = 0; t3193 < 1024; t3193++) {
        float t3194 = scratch_0[(int)t3193];
        float t3195 = t3194 * 1.9036306e-06;
        float t3196 = memory[161728 + (int)t3193];
        int t3197 = t2825 + t3193;
        float t3198 = t3195 * t3196;
        memory[30382528 + t3197] = t3198;
      }
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 29
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_29(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4935 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4935)) {
    /* loadGlobal(1467) - handled in variable access */
    int t3202 = id;
    int t3203 = t3202 / 8;
    uint _frameIndex = (uint)(t3203);
    int t3204 = t3203 * 8;
    int t3205 = t3202 - t3204;
    float t3206 = 0.0;
    int t3207 = (int)frameCount;
    int t3208 = t3207 + 256;
    int t3209 = t3208 - 1;
    int t3210 = t3209 / 256;
    int t3211 = t3210 - 1;
    int t3212 = t3203 + 1024;
    int t3213 = t3212 - 1;
    int t3214 = t3213 / 256;
    float t3215 = metal::min(t3214, t3211);
    int t3216 = t3210 * 8;
    int t3217 = t3216 * 1024;
    int t3218 = t3217 - 1;
    for (uint t3219 = 0; t3219 < 5; t3219++) {
      float t3220 = t3215 - t3219;
      float t3221 = t3220 * 256.0;
      float t3222 = t3220 >= 0.0;
      float t3223 = (float)t3203;
      float t3224 = t3221 >= t3223;
      float t3225 = (float)t3210;
      float t3226 = t3220 < t3225;
      float t3227 = t3222 * t3224;
      float t3228 = t3227 * t3226;
      float t3229 = t3203 - t3221;
      float t3230 = t3229 + 1024.0;
      float t3231 = t3230 - 1.0;
      float t3232 = t3220 * 8.0;
      float t3233 = t3232 + t3205;
      float t3234 = t3233 * 1024.0;
      float t3235 = t3234 + t3231;
      float t3236 = (float)t3218;
      float t3237 = metal::min(t3235, t3236);
      float t3238 = metal::max(0.0, t3237);
      int t3239 = (int)t3238;
      float t3240 = memory[29333952 + t3239];
      float t3241 = metal::select(0.0, t3240, t3228 > 0.0);
      float t3242 = t3206 + t3241;
      t3206 = t3242;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t3244 = (t3206 * 0.0013797212);
    int t3245 = t3203 * 8;
    int t3246 = t3245 + t3205;
    memory[17404352 + t3246] = t3244;
  }
  #pragma clang diagnostic pop
}



// KERNEL 30
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_30(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4936 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4936)) {
    /* loadGlobal(1467) - handled in variable access */
    int t3248 = id;
    int t3249 = t3248 / 8;
    uint _frameIndex = (uint)(t3249);
    int t3250 = t3249 * 8;
    int t3251 = t3248 - t3250;
    float t3252 = 0.0;
    int t3253 = (int)frameCount;
    int t3254 = t3253 + 256;
    int t3255 = t3254 - 1;
    int t3256 = t3255 / 256;
    int t3257 = t3256 - 1;
    int t3258 = t3249 + 1024;
    int t3259 = t3258 - 1;
    int t3260 = t3259 / 256;
    float t3261 = metal::min(t3260, t3257);
    int t3262 = t3256 * 8;
    int t3263 = t3262 * 1024;
    int t3264 = t3263 - 1;
    for (uint t3265 = 0; t3265 < 5; t3265++) {
      float t3266 = t3261 - t3265;
      float t3267 = t3266 * 256.0;
      float t3268 = t3266 >= 0.0;
      float t3269 = (float)t3249;
      float t3270 = t3267 >= t3269;
      float t3271 = (float)t3256;
      float t3272 = t3266 < t3271;
      float t3273 = t3268 * t3270;
      float t3274 = t3273 * t3272;
      float t3275 = t3249 - t3267;
      float t3276 = t3275 + 1024.0;
      float t3277 = t3276 - 1.0;
      float t3278 = t3266 * 8.0;
      float t3279 = t3278 + t3251;
      float t3280 = t3279 * 1024.0;
      float t3281 = t3280 + t3277;
      float t3282 = (float)t3264;
      float t3283 = metal::min(t3281, t3282);
      float t3284 = metal::max(0.0, t3283);
      int t3285 = (int)t3284;
      float t3286 = memory[30382528 + t3285];
      float t3287 = metal::select(0.0, t3286, t3274 > 0.0);
      float t3288 = t3252 + t3287;
      t3252 = t3288;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t3290 = (t3252 * 0.0013797212);
    int t3291 = t3249 * 8;
    int t3292 = t3291 + t3251;
    memory[26579392 + t3292] = t3290;
  }
  #pragma clang diagnostic pop
}



// KERNEL 31
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_31(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1452) - handled in variable access */
    float t3298 = (t[4*frameCount + id] * 3.7252903e-09);
    float t3299 = -0.5 * t3298;
  }
  #pragma clang diagnostic pop
}



// KERNEL 32
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_32(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4937 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4937)) {
    /* loadGlobal(638) - handled in variable access */
    int t3300 = id;
    int t3301 = t3300 / 8;
    uint _frameIndex = (uint)(t3301);
    int t3302 = t3301 * 8;
    int t3303 = t3300 - t3302;
    float t3304 = t[3*frameCount + _frameIndex] == 0.0;
    if (t3304) {
      int t3306 = t3301 / 128;
      int t3307 = t3306 * 8;
      int t3308 = t3307 + t3303;
      int t3309 = t3308 * 1024;
      int t3310 = t3308 * 257;
      int t3311 = t3308 * 1024;
      for (uint _pr3312 = 0; _pr3312 < 257; _pr3312++) {
        int t3313 = t3310 + _pr3312;
        float t3314 = memory[28807616 + t3313];
        int t3315 = t3310 + _pr3312;
        float t3316 = memory[29070784 + t3315];
        int t3317 = t3309 + _pr3312;
        float t3318 = memory[26710464 + t3317];
        int t3319 = t3309 + _pr3312;
        int t3320 = t3319 + 512;
        float t3321 = memory[26710464 + t3320];
        int t3322 = t3309 + _pr3312;
        float t3323 = memory[27759040 + t3322];
        int t3324 = t3309 + _pr3312;
        int t3325 = t3324 + 512;
        float t3326 = memory[27759040 + t3325];
        float t3327 = t3314 - t3316;
        float t3328 = 2.0 * t3327;
        float t3329 = t3328 * 3.8146973e-06;
        float t3330 = t3314 - t3316;
        float t3331 = -2.0 * t3330;
        float t3332 = t3331 * 3.8146973e-06;
        float t3333 = metal::max(t3314, 1e-08);
        float t3334 = metal::max(t3316, 1e-08);
        float t3335 = t3329 * t3318;
        float t3336 = t3335 / t3333;
        float t3337 = t3329 * t3321;
        float t3338 = t3337 / t3333;
        float t3339 = t3332 * t3323;
        float t3340 = t3339 / t3334;
        float t3341 = t3332 * t3326;
        float t3342 = t3341 / t3334;
        int t3343 = t3311 + _pr3312;
        memory[29333952 + t3343] = t3336;
        int t3345 = t3311 + _pr3312;
        int t3346 = t3345 + 512;
        memory[29333952 + t3346] = t3338;
        int t3348 = t3311 + _pr3312;
        memory[30382528 + t3348] = t3340;
        int t3350 = t3311 + _pr3312;
        int t3351 = t3350 + 512;
        memory[30382528 + t3351] = t3342;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3354 = 0; _pr3354 < 255; _pr3354++) {
        int t3355 = _pr3354 + 257;
        int t3356 = t3311 + t3355;
        memory[29333952 + t3356] = 0.0;
        int t3358 = t3311 + t3355;
        int t3359 = t3358 + 512;
        memory[29333952 + t3359] = 0.0;
        int t3361 = t3311 + t3355;
        memory[30382528 + t3361] = 0.0;
        int t3363 = t3311 + t3355;
        int t3364 = t3363 + 512;
        memory[30382528 + t3364] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 33
// FrameOrder: parallel
// DispatchMode: perFrameScaledThreadgroup1(8)
kernel void kernel_33(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4938 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4938)) {
    /* loadGlobal(638) - handled in variable access */
    int t3368 = id;
    int t3369 = t3368 / 8;
    uint _frameIndex = (uint)(t3369);
    int t3370 = t3369 * 8;
    int t3371 = t3368 - t3370;
    threadgroup float scratch_0[512];
    threadgroup float scratch_1[512];
    threadgroup float scratch_2[511];
    threadgroup float scratch_3[511];
    float t3372 = t[3*frameCount + _frameIndex] == 0.0;
    if (t3372) {
      for (uint t3374 = 0; t3374 < 511; t3374++) {
        float t3375 = memory[160194 + (int)t3374];
        scratch_2[(int)t3374] = t3375;
        float t3377 = memory[160705 + (int)t3374];
        scratch_3[(int)t3374] = t3377;
      }
      int t3380 = t3369 / 128;
      int t3381 = t3380 * 8;
      int t3382 = t3381 + t3371;
      int t3383 = t3382 * 1024;
      int t3384 = t3382 * 512;
      for (uint t3385 = 0; t3385 < 512; t3385++) {
        int t3386 = t3383 + t3385;
        float t3387 = memory[29333952 + t3386];
        int t3388 = t3383 + t3385;
        int t3389 = t3388 + 512;
        float t3390 = memory[29333952 + t3389];
        scratch_0[(int)t3385] = t3387;
        scratch_1[(int)t3385] = t3390;
      }
      for (uint t3394 = 0; t3394 < 512; t3394++) {
        float t3395 = memory[161216 + (int)t3394];
        float t3396 = (float)t3394;
        float t3397 = t3396 < t3395;
        int t3398 = (int)t3395;
        float t3399 = scratch_0[(int)t3394];
        float t3400 = scratch_1[(int)t3394];
        float t3401 = scratch_0[t3398];
        float t3402 = scratch_1[t3398];
        float t3403 = metal::select(t3399, t3401, t3397 > 0.0);
        float t3404 = metal::select(t3400, t3402, t3397 > 0.0);
        float t3405 = metal::select(t3401, t3399, t3397 > 0.0);
        float t3406 = metal::select(t3402, t3400, t3397 > 0.0);
        scratch_0[(int)t3394] = t3403;
        scratch_1[(int)t3394] = t3404;
        scratch_0[t3398] = t3405;
        scratch_1[t3398] = t3406;
      }
      for (uint t3412 = 0; t3412 < 256; t3412++) {
        float t3413 = (float)t3412;
        float t3414 = t3413;
        float t3415 = metal::floor(t3414);
        float t3416 = t3415;
        float t3417 = t3413 - t3416;
        float t3418 = t3415 * 2.0;
        float t3419 = t3418 + t3417;
        float t3420 = t3419 + 1.0;
        int t3421 = (int)t3417;
        int t3422 = t3421;
        float t3423 = scratch_2[t3422];
        float t3424 = scratch_3[t3422];
        int t3425 = (int)t3419;
        int t3426 = (int)t3420;
        float t3427 = scratch_0[t3425];
        float t3428 = scratch_1[t3425];
        float t3429 = scratch_0[t3426];
        float t3430 = scratch_1[t3426];
        float t3431 = t3423 * t3429;
        float t3432 = t3424 * t3430;
        float t3433 = t3431 - t3432;
        float t3434 = t3423 * t3430;
        float t3435 = t3424 * t3429;
        float t3436 = t3434 + t3435;
        float t3437 = t3427 + t3433;
        scratch_0[t3425] = t3437;
        float t3439 = t3428 + t3436;
        scratch_1[t3425] = t3439;
        float t3441 = t3427 - t3433;
        scratch_0[t3426] = t3441;
        float t3443 = t3428 - t3436;
        scratch_1[t3426] = t3443;
      }
      for (uint t3446 = 0; t3446 < 256; t3446++) {
        float t3447 = (float)t3446;
        float t3448 = (t3447 * 0.5);
        float t3449 = metal::floor(t3448);
        float t3450 = t3449 * 2.0;
        float t3451 = t3447 - t3450;
        float t3452 = t3449 * 4.0;
        float t3453 = t3452 + t3451;
        float t3454 = t3453 + 2.0;
        int t3455 = (int)t3451;
        int t3456 = 1 + t3455;
        float t3457 = scratch_2[t3456];
        float t3458 = scratch_3[t3456];
        int t3459 = (int)t3453;
        int t3460 = (int)t3454;
        float t3461 = scratch_0[t3459];
        float t3462 = scratch_1[t3459];
        float t3463 = scratch_0[t3460];
        float t3464 = scratch_1[t3460];
        float t3465 = t3457 * t3463;
        float t3466 = t3458 * t3464;
        float t3467 = t3465 - t3466;
        float t3468 = t3457 * t3464;
        float t3469 = t3458 * t3463;
        float t3470 = t3468 + t3469;
        float t3471 = t3461 + t3467;
        scratch_0[t3459] = t3471;
        float t3473 = t3462 + t3470;
        scratch_1[t3459] = t3473;
        float t3475 = t3461 - t3467;
        scratch_0[t3460] = t3475;
        float t3477 = t3462 - t3470;
        scratch_1[t3460] = t3477;
      }
      for (uint t3480 = 0; t3480 < 256; t3480++) {
        float t3481 = (float)t3480;
        float t3482 = (t3481 * 0.25);
        float t3483 = metal::floor(t3482);
        float t3484 = t3483 * 4.0;
        float t3485 = t3481 - t3484;
        float t3486 = t3483 * 8.0;
        float t3487 = t3486 + t3485;
        float t3488 = t3487 + 4.0;
        int t3489 = (int)t3485;
        int t3490 = 3 + t3489;
        float t3491 = scratch_2[t3490];
        float t3492 = scratch_3[t3490];
        int t3493 = (int)t3487;
        int t3494 = (int)t3488;
        float t3495 = scratch_0[t3493];
        float t3496 = scratch_1[t3493];
        float t3497 = scratch_0[t3494];
        float t3498 = scratch_1[t3494];
        float t3499 = t3491 * t3497;
        float t3500 = t3492 * t3498;
        float t3501 = t3499 - t3500;
        float t3502 = t3491 * t3498;
        float t3503 = t3492 * t3497;
        float t3504 = t3502 + t3503;
        float t3505 = t3495 + t3501;
        scratch_0[t3493] = t3505;
        float t3507 = t3496 + t3504;
        scratch_1[t3493] = t3507;
        float t3509 = t3495 - t3501;
        scratch_0[t3494] = t3509;
        float t3511 = t3496 - t3504;
        scratch_1[t3494] = t3511;
      }
      for (uint t3514 = 0; t3514 < 256; t3514++) {
        float t3515 = (float)t3514;
        float t3516 = (t3515 * 0.125);
        float t3517 = metal::floor(t3516);
        float t3518 = t3517 * 8.0;
        float t3519 = t3515 - t3518;
        float t3520 = t3517 * 16.0;
        float t3521 = t3520 + t3519;
        float t3522 = t3521 + 8.0;
        int t3523 = (int)t3519;
        int t3524 = 7 + t3523;
        float t3525 = scratch_2[t3524];
        float t3526 = scratch_3[t3524];
        int t3527 = (int)t3521;
        int t3528 = (int)t3522;
        float t3529 = scratch_0[t3527];
        float t3530 = scratch_1[t3527];
        float t3531 = scratch_0[t3528];
        float t3532 = scratch_1[t3528];
        float t3533 = t3525 * t3531;
        float t3534 = t3526 * t3532;
        float t3535 = t3533 - t3534;
        float t3536 = t3525 * t3532;
        float t3537 = t3526 * t3531;
        float t3538 = t3536 + t3537;
        float t3539 = t3529 + t3535;
        scratch_0[t3527] = t3539;
        float t3541 = t3530 + t3538;
        scratch_1[t3527] = t3541;
        float t3543 = t3529 - t3535;
        scratch_0[t3528] = t3543;
        float t3545 = t3530 - t3538;
        scratch_1[t3528] = t3545;
      }
      for (uint t3548 = 0; t3548 < 256; t3548++) {
        float t3549 = (float)t3548;
        float t3550 = (t3549 * 0.0625);
        float t3551 = metal::floor(t3550);
        float t3552 = t3551 * 16.0;
        float t3553 = t3549 - t3552;
        float t3554 = t3551 * 32.0;
        float t3555 = t3554 + t3553;
        float t3556 = t3555 + 16.0;
        int t3557 = (int)t3553;
        int t3558 = 15 + t3557;
        float t3559 = scratch_2[t3558];
        float t3560 = scratch_3[t3558];
        int t3561 = (int)t3555;
        int t3562 = (int)t3556;
        float t3563 = scratch_0[t3561];
        float t3564 = scratch_1[t3561];
        float t3565 = scratch_0[t3562];
        float t3566 = scratch_1[t3562];
        float t3567 = t3559 * t3565;
        float t3568 = t3560 * t3566;
        float t3569 = t3567 - t3568;
        float t3570 = t3559 * t3566;
        float t3571 = t3560 * t3565;
        float t3572 = t3570 + t3571;
        float t3573 = t3563 + t3569;
        scratch_0[t3561] = t3573;
        float t3575 = t3564 + t3572;
        scratch_1[t3561] = t3575;
        float t3577 = t3563 - t3569;
        scratch_0[t3562] = t3577;
        float t3579 = t3564 - t3572;
        scratch_1[t3562] = t3579;
      }
      for (uint t3582 = 0; t3582 < 256; t3582++) {
        float t3583 = (float)t3582;
        float t3584 = (t3583 * 0.03125);
        float t3585 = metal::floor(t3584);
        float t3586 = t3585 * 32.0;
        float t3587 = t3583 - t3586;
        float t3588 = t3585 * 64.0;
        float t3589 = t3588 + t3587;
        float t3590 = t3589 + 32.0;
        int t3591 = (int)t3587;
        int t3592 = 31 + t3591;
        float t3593 = scratch_2[t3592];
        float t3594 = scratch_3[t3592];
        int t3595 = (int)t3589;
        int t3596 = (int)t3590;
        float t3597 = scratch_0[t3595];
        float t3598 = scratch_1[t3595];
        float t3599 = scratch_0[t3596];
        float t3600 = scratch_1[t3596];
        float t3601 = t3593 * t3599;
        float t3602 = t3594 * t3600;
        float t3603 = t3601 - t3602;
        float t3604 = t3593 * t3600;
        float t3605 = t3594 * t3599;
        float t3606 = t3604 + t3605;
        float t3607 = t3597 + t3603;
        scratch_0[t3595] = t3607;
        float t3609 = t3598 + t3606;
        scratch_1[t3595] = t3609;
        float t3611 = t3597 - t3603;
        scratch_0[t3596] = t3611;
        float t3613 = t3598 - t3606;
        scratch_1[t3596] = t3613;
      }
      for (uint t3616 = 0; t3616 < 256; t3616++) {
        float t3617 = (float)t3616;
        float t3618 = (t3617 * 0.015625);
        float t3619 = metal::floor(t3618);
        float t3620 = t3619 * 64.0;
        float t3621 = t3617 - t3620;
        float t3622 = t3619 * 128.0;
        float t3623 = t3622 + t3621;
        float t3624 = t3623 + 64.0;
        int t3625 = (int)t3621;
        int t3626 = 63 + t3625;
        float t3627 = scratch_2[t3626];
        float t3628 = scratch_3[t3626];
        int t3629 = (int)t3623;
        int t3630 = (int)t3624;
        float t3631 = scratch_0[t3629];
        float t3632 = scratch_1[t3629];
        float t3633 = scratch_0[t3630];
        float t3634 = scratch_1[t3630];
        float t3635 = t3627 * t3633;
        float t3636 = t3628 * t3634;
        float t3637 = t3635 - t3636;
        float t3638 = t3627 * t3634;
        float t3639 = t3628 * t3633;
        float t3640 = t3638 + t3639;
        float t3641 = t3631 + t3637;
        scratch_0[t3629] = t3641;
        float t3643 = t3632 + t3640;
        scratch_1[t3629] = t3643;
        float t3645 = t3631 - t3637;
        scratch_0[t3630] = t3645;
        float t3647 = t3632 - t3640;
        scratch_1[t3630] = t3647;
      }
      for (uint t3650 = 0; t3650 < 256; t3650++) {
        float t3651 = (float)t3650;
        float t3652 = (t3651 * 0.0078125);
        float t3653 = metal::floor(t3652);
        float t3654 = t3653 * 128.0;
        float t3655 = t3651 - t3654;
        float t3656 = t3653 * 256.0;
        float t3657 = t3656 + t3655;
        float t3658 = t3657 + 128.0;
        int t3659 = (int)t3655;
        int t3660 = 127 + t3659;
        float t3661 = scratch_2[t3660];
        float t3662 = scratch_3[t3660];
        int t3663 = (int)t3657;
        int t3664 = (int)t3658;
        float t3665 = scratch_0[t3663];
        float t3666 = scratch_1[t3663];
        float t3667 = scratch_0[t3664];
        float t3668 = scratch_1[t3664];
        float t3669 = t3661 * t3667;
        float t3670 = t3662 * t3668;
        float t3671 = t3669 - t3670;
        float t3672 = t3661 * t3668;
        float t3673 = t3662 * t3667;
        float t3674 = t3672 + t3673;
        float t3675 = t3665 + t3671;
        scratch_0[t3663] = t3675;
        float t3677 = t3666 + t3674;
        scratch_1[t3663] = t3677;
        float t3679 = t3665 - t3671;
        scratch_0[t3664] = t3679;
        float t3681 = t3666 - t3674;
        scratch_1[t3664] = t3681;
      }
      for (uint t3684 = 0; t3684 < 256; t3684++) {
        float t3685 = (float)t3684;
        float t3686 = (t3685 * 0.00390625);
        float t3687 = metal::floor(t3686);
        float t3688 = t3687 * 256.0;
        float t3689 = t3685 - t3688;
        float t3690 = t3687 * 512.0;
        float t3691 = t3690 + t3689;
        float t3692 = t3691 + 256.0;
        int t3693 = (int)t3689;
        int t3694 = 255 + t3693;
        float t3695 = scratch_2[t3694];
        float t3696 = scratch_3[t3694];
        int t3697 = (int)t3691;
        int t3698 = (int)t3692;
        float t3699 = scratch_0[t3697];
        float t3700 = scratch_1[t3697];
        float t3701 = scratch_0[t3698];
        float t3702 = scratch_1[t3698];
        float t3703 = t3695 * t3701;
        float t3704 = t3696 * t3702;
        float t3705 = t3703 - t3704;
        float t3706 = t3695 * t3702;
        float t3707 = t3696 * t3701;
        float t3708 = t3706 + t3707;
        float t3709 = t3699 + t3705;
        scratch_0[t3697] = t3709;
        float t3711 = t3700 + t3708;
        scratch_1[t3697] = t3711;
        float t3713 = t3699 - t3705;
        scratch_0[t3698] = t3713;
        float t3715 = t3700 - t3708;
        scratch_1[t3698] = t3715;
      }
      for (uint t3718 = 0; t3718 < 512; t3718++) {
        float t3719 = scratch_0[(int)t3718];
        float t3720 = t3719 * 7.599708e-06;
        float t3721 = memory[159682 + (int)t3718];
        int t3722 = t3384 + t3718;
        float t3723 = t3720 * t3721;
        memory[26710464 + t3722] = t3723;
      }
      int t3726 = t3369 / 128;
      int t3727 = t3726 * 8;
      int t3728 = t3727 + t3371;
      int t3729 = t3728 * 1024;
      int t3730 = t3728 * 512;
      for (uint t3731 = 0; t3731 < 512; t3731++) {
        int t3732 = t3729 + t3731;
        float t3733 = memory[30382528 + t3732];
        int t3734 = t3729 + t3731;
        int t3735 = t3734 + 512;
        float t3736 = memory[30382528 + t3735];
        scratch_0[(int)t3731] = t3733;
        scratch_1[(int)t3731] = t3736;
      }
      for (uint t3740 = 0; t3740 < 512; t3740++) {
        float t3741 = memory[161216 + (int)t3740];
        float t3742 = (float)t3740;
        float t3743 = t3742 < t3741;
        int t3744 = (int)t3741;
        float t3745 = scratch_0[(int)t3740];
        float t3746 = scratch_1[(int)t3740];
        float t3747 = scratch_0[t3744];
        float t3748 = scratch_1[t3744];
        float t3749 = metal::select(t3745, t3747, t3743 > 0.0);
        float t3750 = metal::select(t3746, t3748, t3743 > 0.0);
        float t3751 = metal::select(t3747, t3745, t3743 > 0.0);
        float t3752 = metal::select(t3748, t3746, t3743 > 0.0);
        scratch_0[(int)t3740] = t3749;
        scratch_1[(int)t3740] = t3750;
        scratch_0[t3744] = t3751;
        scratch_1[t3744] = t3752;
      }
      for (uint t3758 = 0; t3758 < 256; t3758++) {
        float t3759 = (float)t3758;
        float t3760 = t3759;
        float t3761 = metal::floor(t3760);
        float t3762 = t3761;
        float t3763 = t3759 - t3762;
        float t3764 = t3761 * 2.0;
        float t3765 = t3764 + t3763;
        float t3766 = t3765 + 1.0;
        int t3767 = (int)t3763;
        int t3768 = t3767;
        float t3769 = scratch_2[t3768];
        float t3770 = scratch_3[t3768];
        int t3771 = (int)t3765;
        int t3772 = (int)t3766;
        float t3773 = scratch_0[t3771];
        float t3774 = scratch_1[t3771];
        float t3775 = scratch_0[t3772];
        float t3776 = scratch_1[t3772];
        float t3777 = t3769 * t3775;
        float t3778 = t3770 * t3776;
        float t3779 = t3777 - t3778;
        float t3780 = t3769 * t3776;
        float t3781 = t3770 * t3775;
        float t3782 = t3780 + t3781;
        float t3783 = t3773 + t3779;
        scratch_0[t3771] = t3783;
        float t3785 = t3774 + t3782;
        scratch_1[t3771] = t3785;
        float t3787 = t3773 - t3779;
        scratch_0[t3772] = t3787;
        float t3789 = t3774 - t3782;
        scratch_1[t3772] = t3789;
      }
      for (uint t3792 = 0; t3792 < 256; t3792++) {
        float t3793 = (float)t3792;
        float t3794 = (t3793 * 0.5);
        float t3795 = metal::floor(t3794);
        float t3796 = t3795 * 2.0;
        float t3797 = t3793 - t3796;
        float t3798 = t3795 * 4.0;
        float t3799 = t3798 + t3797;
        float t3800 = t3799 + 2.0;
        int t3801 = (int)t3797;
        int t3802 = 1 + t3801;
        float t3803 = scratch_2[t3802];
        float t3804 = scratch_3[t3802];
        int t3805 = (int)t3799;
        int t3806 = (int)t3800;
        float t3807 = scratch_0[t3805];
        float t3808 = scratch_1[t3805];
        float t3809 = scratch_0[t3806];
        float t3810 = scratch_1[t3806];
        float t3811 = t3803 * t3809;
        float t3812 = t3804 * t3810;
        float t3813 = t3811 - t3812;
        float t3814 = t3803 * t3810;
        float t3815 = t3804 * t3809;
        float t3816 = t3814 + t3815;
        float t3817 = t3807 + t3813;
        scratch_0[t3805] = t3817;
        float t3819 = t3808 + t3816;
        scratch_1[t3805] = t3819;
        float t3821 = t3807 - t3813;
        scratch_0[t3806] = t3821;
        float t3823 = t3808 - t3816;
        scratch_1[t3806] = t3823;
      }
      for (uint t3826 = 0; t3826 < 256; t3826++) {
        float t3827 = (float)t3826;
        float t3828 = (t3827 * 0.25);
        float t3829 = metal::floor(t3828);
        float t3830 = t3829 * 4.0;
        float t3831 = t3827 - t3830;
        float t3832 = t3829 * 8.0;
        float t3833 = t3832 + t3831;
        float t3834 = t3833 + 4.0;
        int t3835 = (int)t3831;
        int t3836 = 3 + t3835;
        float t3837 = scratch_2[t3836];
        float t3838 = scratch_3[t3836];
        int t3839 = (int)t3833;
        int t3840 = (int)t3834;
        float t3841 = scratch_0[t3839];
        float t3842 = scratch_1[t3839];
        float t3843 = scratch_0[t3840];
        float t3844 = scratch_1[t3840];
        float t3845 = t3837 * t3843;
        float t3846 = t3838 * t3844;
        float t3847 = t3845 - t3846;
        float t3848 = t3837 * t3844;
        float t3849 = t3838 * t3843;
        float t3850 = t3848 + t3849;
        float t3851 = t3841 + t3847;
        scratch_0[t3839] = t3851;
        float t3853 = t3842 + t3850;
        scratch_1[t3839] = t3853;
        float t3855 = t3841 - t3847;
        scratch_0[t3840] = t3855;
        float t3857 = t3842 - t3850;
        scratch_1[t3840] = t3857;
      }
      for (uint t3860 = 0; t3860 < 256; t3860++) {
        float t3861 = (float)t3860;
        float t3862 = (t3861 * 0.125);
        float t3863 = metal::floor(t3862);
        float t3864 = t3863 * 8.0;
        float t3865 = t3861 - t3864;
        float t3866 = t3863 * 16.0;
        float t3867 = t3866 + t3865;
        float t3868 = t3867 + 8.0;
        int t3869 = (int)t3865;
        int t3870 = 7 + t3869;
        float t3871 = scratch_2[t3870];
        float t3872 = scratch_3[t3870];
        int t3873 = (int)t3867;
        int t3874 = (int)t3868;
        float t3875 = scratch_0[t3873];
        float t3876 = scratch_1[t3873];
        float t3877 = scratch_0[t3874];
        float t3878 = scratch_1[t3874];
        float t3879 = t3871 * t3877;
        float t3880 = t3872 * t3878;
        float t3881 = t3879 - t3880;
        float t3882 = t3871 * t3878;
        float t3883 = t3872 * t3877;
        float t3884 = t3882 + t3883;
        float t3885 = t3875 + t3881;
        scratch_0[t3873] = t3885;
        float t3887 = t3876 + t3884;
        scratch_1[t3873] = t3887;
        float t3889 = t3875 - t3881;
        scratch_0[t3874] = t3889;
        float t3891 = t3876 - t3884;
        scratch_1[t3874] = t3891;
      }
      for (uint t3894 = 0; t3894 < 256; t3894++) {
        float t3895 = (float)t3894;
        float t3896 = (t3895 * 0.0625);
        float t3897 = metal::floor(t3896);
        float t3898 = t3897 * 16.0;
        float t3899 = t3895 - t3898;
        float t3900 = t3897 * 32.0;
        float t3901 = t3900 + t3899;
        float t3902 = t3901 + 16.0;
        int t3903 = (int)t3899;
        int t3904 = 15 + t3903;
        float t3905 = scratch_2[t3904];
        float t3906 = scratch_3[t3904];
        int t3907 = (int)t3901;
        int t3908 = (int)t3902;
        float t3909 = scratch_0[t3907];
        float t3910 = scratch_1[t3907];
        float t3911 = scratch_0[t3908];
        float t3912 = scratch_1[t3908];
        float t3913 = t3905 * t3911;
        float t3914 = t3906 * t3912;
        float t3915 = t3913 - t3914;
        float t3916 = t3905 * t3912;
        float t3917 = t3906 * t3911;
        float t3918 = t3916 + t3917;
        float t3919 = t3909 + t3915;
        scratch_0[t3907] = t3919;
        float t3921 = t3910 + t3918;
        scratch_1[t3907] = t3921;
        float t3923 = t3909 - t3915;
        scratch_0[t3908] = t3923;
        float t3925 = t3910 - t3918;
        scratch_1[t3908] = t3925;
      }
      for (uint t3928 = 0; t3928 < 256; t3928++) {
        float t3929 = (float)t3928;
        float t3930 = (t3929 * 0.03125);
        float t3931 = metal::floor(t3930);
        float t3932 = t3931 * 32.0;
        float t3933 = t3929 - t3932;
        float t3934 = t3931 * 64.0;
        float t3935 = t3934 + t3933;
        float t3936 = t3935 + 32.0;
        int t3937 = (int)t3933;
        int t3938 = 31 + t3937;
        float t3939 = scratch_2[t3938];
        float t3940 = scratch_3[t3938];
        int t3941 = (int)t3935;
        int t3942 = (int)t3936;
        float t3943 = scratch_0[t3941];
        float t3944 = scratch_1[t3941];
        float t3945 = scratch_0[t3942];
        float t3946 = scratch_1[t3942];
        float t3947 = t3939 * t3945;
        float t3948 = t3940 * t3946;
        float t3949 = t3947 - t3948;
        float t3950 = t3939 * t3946;
        float t3951 = t3940 * t3945;
        float t3952 = t3950 + t3951;
        float t3953 = t3943 + t3949;
        scratch_0[t3941] = t3953;
        float t3955 = t3944 + t3952;
        scratch_1[t3941] = t3955;
        float t3957 = t3943 - t3949;
        scratch_0[t3942] = t3957;
        float t3959 = t3944 - t3952;
        scratch_1[t3942] = t3959;
      }
      for (uint t3962 = 0; t3962 < 256; t3962++) {
        float t3963 = (float)t3962;
        float t3964 = (t3963 * 0.015625);
        float t3965 = metal::floor(t3964);
        float t3966 = t3965 * 64.0;
        float t3967 = t3963 - t3966;
        float t3968 = t3965 * 128.0;
        float t3969 = t3968 + t3967;
        float t3970 = t3969 + 64.0;
        int t3971 = (int)t3967;
        int t3972 = 63 + t3971;
        float t3973 = scratch_2[t3972];
        float t3974 = scratch_3[t3972];
        int t3975 = (int)t3969;
        int t3976 = (int)t3970;
        float t3977 = scratch_0[t3975];
        float t3978 = scratch_1[t3975];
        float t3979 = scratch_0[t3976];
        float t3980 = scratch_1[t3976];
        float t3981 = t3973 * t3979;
        float t3982 = t3974 * t3980;
        float t3983 = t3981 - t3982;
        float t3984 = t3973 * t3980;
        float t3985 = t3974 * t3979;
        float t3986 = t3984 + t3985;
        float t3987 = t3977 + t3983;
        scratch_0[t3975] = t3987;
        float t3989 = t3978 + t3986;
        scratch_1[t3975] = t3989;
        float t3991 = t3977 - t3983;
        scratch_0[t3976] = t3991;
        float t3993 = t3978 - t3986;
        scratch_1[t3976] = t3993;
      }
      for (uint t3996 = 0; t3996 < 256; t3996++) {
        float t3997 = (float)t3996;
        float t3998 = (t3997 * 0.0078125);
        float t3999 = metal::floor(t3998);
        float t4000 = t3999 * 128.0;
        float t4001 = t3997 - t4000;
        float t4002 = t3999 * 256.0;
        float t4003 = t4002 + t4001;
        float t4004 = t4003 + 128.0;
        int t4005 = (int)t4001;
        int t4006 = 127 + t4005;
        float t4007 = scratch_2[t4006];
        float t4008 = scratch_3[t4006];
        int t4009 = (int)t4003;
        int t4010 = (int)t4004;
        float t4011 = scratch_0[t4009];
        float t4012 = scratch_1[t4009];
        float t4013 = scratch_0[t4010];
        float t4014 = scratch_1[t4010];
        float t4015 = t4007 * t4013;
        float t4016 = t4008 * t4014;
        float t4017 = t4015 - t4016;
        float t4018 = t4007 * t4014;
        float t4019 = t4008 * t4013;
        float t4020 = t4018 + t4019;
        float t4021 = t4011 + t4017;
        scratch_0[t4009] = t4021;
        float t4023 = t4012 + t4020;
        scratch_1[t4009] = t4023;
        float t4025 = t4011 - t4017;
        scratch_0[t4010] = t4025;
        float t4027 = t4012 - t4020;
        scratch_1[t4010] = t4027;
      }
      for (uint t4030 = 0; t4030 < 256; t4030++) {
        float t4031 = (float)t4030;
        float t4032 = (t4031 * 0.00390625);
        float t4033 = metal::floor(t4032);
        float t4034 = t4033 * 256.0;
        float t4035 = t4031 - t4034;
        float t4036 = t4033 * 512.0;
        float t4037 = t4036 + t4035;
        float t4038 = t4037 + 256.0;
        int t4039 = (int)t4035;
        int t4040 = 255 + t4039;
        float t4041 = scratch_2[t4040];
        float t4042 = scratch_3[t4040];
        int t4043 = (int)t4037;
        int t4044 = (int)t4038;
        float t4045 = scratch_0[t4043];
        float t4046 = scratch_1[t4043];
        float t4047 = scratch_0[t4044];
        float t4048 = scratch_1[t4044];
        float t4049 = t4041 * t4047;
        float t4050 = t4042 * t4048;
        float t4051 = t4049 - t4050;
        float t4052 = t4041 * t4048;
        float t4053 = t4042 * t4047;
        float t4054 = t4052 + t4053;
        float t4055 = t4045 + t4051;
        scratch_0[t4043] = t4055;
        float t4057 = t4046 + t4054;
        scratch_1[t4043] = t4057;
        float t4059 = t4045 - t4051;
        scratch_0[t4044] = t4059;
        float t4061 = t4046 - t4054;
        scratch_1[t4044] = t4061;
      }
      for (uint t4064 = 0; t4064 < 512; t4064++) {
        float t4065 = scratch_0[(int)t4064];
        float t4066 = t4065 * 7.599708e-06;
        float t4067 = memory[159682 + (int)t4064];
        int t4068 = t3730 + t4064;
        float t4069 = t4066 * t4067;
        memory[27759040 + t4068] = t4069;
      }
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 34
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_34(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4939 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4939)) {
    /* loadGlobal(638) - handled in variable access */
    int t4073 = id;
    int t4074 = t4073 / 8;
    uint _frameIndex = (uint)(t4074);
    int t4075 = t4074 * 8;
    int t4076 = t4073 - t4075;
    float t4077 = 0.0;
    int t4078 = (int)frameCount;
    int t4079 = t4078 + 128;
    int t4080 = t4079 - 1;
    int t4081 = t4080 / 128;
    int t4082 = t4081 - 1;
    int t4083 = t4074 + 512;
    int t4084 = t4083 - 1;
    int t4085 = t4084 / 128;
    float t4086 = metal::min(t4085, t4082);
    int t4087 = t4081 * 8;
    int t4088 = t4087 * 512;
    int t4089 = t4088 - 1;
    for (uint t4090 = 0; t4090 < 5; t4090++) {
      float t4091 = t4086 - t4090;
      float t4092 = t4091 * 128.0;
      float t4093 = t4091 >= 0.0;
      float t4094 = (float)t4074;
      float t4095 = t4092 >= t4094;
      float t4096 = (float)t4081;
      float t4097 = t4091 < t4096;
      float t4098 = t4093 * t4095;
      float t4099 = t4098 * t4097;
      float t4100 = t4074 - t4092;
      float t4101 = t4100 + 512.0;
      float t4102 = t4101 - 1.0;
      float t4103 = t4091 * 8.0;
      float t4104 = t4103 + t4076;
      float t4105 = t4104 * 512.0;
      float t4106 = t4105 + t4102;
      float t4107 = (float)t4089;
      float t4108 = metal::min(t4106, t4107);
      float t4109 = metal::max(0.0, t4108);
      int t4110 = (int)t4109;
      float t4111 = memory[26710464 + t4110];
      float t4112 = metal::select(0.0, t4111, t4099 > 0.0);
      float t4113 = t4077 + t4112;
      t4077 = t4113;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4115 = (t4077 * 0.0027567567);
    int t4116 = t4074 * 8;
    int t4117 = t4116 + t4076;
    memory[31431104 + t4117] = t4115;
  }
  #pragma clang diagnostic pop
}



// KERNEL 35
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_35(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4940 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4940)) {
    /* loadGlobal(638) - handled in variable access */
    int t4119 = id;
    int t4120 = t4119 / 8;
    uint _frameIndex = (uint)(t4120);
    int t4121 = t4120 * 8;
    int t4122 = t4119 - t4121;
    float t4123 = 0.0;
    int t4124 = (int)frameCount;
    int t4125 = t4124 + 128;
    int t4126 = t4125 - 1;
    int t4127 = t4126 / 128;
    int t4128 = t4127 - 1;
    int t4129 = t4120 + 512;
    int t4130 = t4129 - 1;
    int t4131 = t4130 / 128;
    float t4132 = metal::min(t4131, t4128);
    int t4133 = t4127 * 8;
    int t4134 = t4133 * 512;
    int t4135 = t4134 - 1;
    for (uint t4136 = 0; t4136 < 5; t4136++) {
      float t4137 = t4132 - t4136;
      float t4138 = t4137 * 128.0;
      float t4139 = t4137 >= 0.0;
      float t4140 = (float)t4120;
      float t4141 = t4138 >= t4140;
      float t4142 = (float)t4127;
      float t4143 = t4137 < t4142;
      float t4144 = t4139 * t4141;
      float t4145 = t4144 * t4143;
      float t4146 = t4120 - t4138;
      float t4147 = t4146 + 512.0;
      float t4148 = t4147 - 1.0;
      float t4149 = t4137 * 8.0;
      float t4150 = t4149 + t4122;
      float t4151 = t4150 * 512.0;
      float t4152 = t4151 + t4148;
      float t4153 = (float)t4135;
      float t4154 = metal::min(t4152, t4153);
      float t4155 = metal::max(0.0, t4154);
      int t4156 = (int)t4155;
      float t4157 = memory[27759040 + t4156];
      float t4158 = metal::select(0.0, t4157, t4145 > 0.0);
      float t4159 = t4123 + t4158;
      t4123 = t4159;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4161 = (t4123 * 0.0027567567);
    int t4162 = t4120 * 8;
    int t4163 = t4162 + t4122;
    memory[31693760 + t4163] = t4161;
  }
  #pragma clang diagnostic pop
}



// KERNEL 36
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_36(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4941 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4941)) {
    int t4165 = id;
    int t4166 = t4165 / 8;
    uint _frameIndex = (uint)(t4166);
    int t4167 = t4166 * 8;
    int t4168 = t4165 - t4167;
    int t4169 = _frameIndex;
    int t4170 = t4169 * 8;
    int t4171 = t4170 + t4168;
    float t4172 = memory[17404352 + t4171];
    int t4173 = _frameIndex;
    int t4174 = t4173 * 8;
    int t4175 = t4174 + t4168;
    float t4176 = memory[31431104 + t4175];
    float t4177 = t4172 + t4176;
    int t4178 = _frameIndex;
    int t4179 = t4178 * 8;
    int t4180 = t4179 + t4168;
    float t4181 = memory[26579392 + t4180];
    int t4182 = _frameIndex;
    int t4183 = t4182 * 8;
    int t4184 = t4183 + t4168;
    float t4185 = memory[31693760 + t4184];
    float t4186 = t4181 + t4185;
    float t4187 = 0.015625 * t4177;
    int t4188 = _frameIndex;
    int t4189 = t4188 * 8;
    int t4190 = t4189 + t4168;
    float t4191 = memory[17535424 + t4190];
    float t4192 = t4191 * t4177;
    int t4193 = _frameIndex;
    int t4194 = t4193 * 8;
    int t4195 = t4194 + t4168;
    float t4196 = memory[17666496 + t4195];
    float t4197 = t4196 * t4187;
    int t4198 = _frameIndex;
    int t4199 = t4198 * 8;
    int t4200 = t4199 + t4168;
    memory[28807616 + t4200] = t4197;
    int t4202 = _frameIndex;
    int t4203 = t4202 * 8;
    int t4204 = t4203 + t4168;
    float t4205 = memory[26448320 + t4204];
    float t4206 = t4205 * t4187;
    int t4207 = _frameIndex;
    int t4208 = t4207 * 8;
    int t4209 = t4208 + t4168;
    memory[29070784 + t4209] = t4206;
  }
  #pragma clang diagnostic pop
}



// KERNEL 37
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_37(
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
    float t4211 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 64.0) * 64.0);
    float t4212 = t4211 < 0.0;
    float t4213 = t4211 + 64.0;
    float t4214 = metal::select(t4211, t4213, t4212 > 0.0);
    float t4215 = metal::floor(t4214);
    float t4216 = t4215 + 1.0;
    float t4217 = t4216 >= 64.0;
    float t4218 = metal::select(t4216, 0.0, t4217 > 0.0);
    float t4219 = t4214 - t4215;
    int t4220 = id;
    memory[526272 + t4220] = t4215;
    memory[165824 + t4220] = t4219;
    float t4223 = t4220 + 16384.0;
    int t4224 = (int)t4223;
    memory[526272 + t4224] = t4218;
    float t4226 = 1.0 - t4219;
    float t4227 = t4220 * 8.0;
    for (uint _pr4228 = 0; _pr4228 < 8; _pr4228++) {
      float t4229 = (float)_pr4228;
      float t4230 = t4227 + t4229;
      int t4231 = (int)t4230;
      float t4232 = memory[29070784 + t4231];
      float t4233 = t4227 + t4229;
      float t4234 = t4232 * t4226;
      int t4235 = (int)t4233;
      memory[17404352 + t4235] = t4234;
      float t4237 = t4232 * t4219;
      int t4238 = (int)t4233;
      memory[17535424 + t4238] = t4237;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 38
// FrameOrder: sequential
// DispatchMode: staticThreads(512)
kernel void kernel_38(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 512) { uint _pr4241 = id;
    int t4242 = _pr4241 / 8;
    int t4243 = t4242 * 8;
    int t4244 = _pr4241 - t4243;
    float t4245 = (float)t4242;
    float t4246 = (float)t4244;
    float t4247 = 0.0;
    for (uint t4248 = 0; t4248 < 16384; t4248++) {
      float t4249 = (float)t4248;
      float t4250 = t4249 < frameCount;
      float t4251 = t4249 * 8.0;
      float t4252 = t4251 + t4246;
      float t4253 = memory[526272 + (int)t4248];
      float t4254 = t4253 - t4245;
      float t4255 = metal::abs(t4254);
      float t4256 = t4255 < 0.5;
      int t4257 = (int)t4252;
      float t4258 = memory[17404352 + t4257];
      float t4259 = t4250 * t4256;
      float t4260 = t4259 > 0.0;
      float t4261 = metal::select(0.0, t4258, t4260 > 0.0);
      float t4262 = t4247 + t4261;
      t4247 = t4262;
      float t4263 = t4249 + 16384.0;
      int t4264 = (int)t4263;
      float t4265 = memory[526272 + t4264];
      float t4266 = t4265 - t4245;
      float t4267 = metal::abs(t4266);
      float t4268 = t4267 < 0.5;
      int t4269 = (int)t4252;
      float t4270 = memory[17535424 + t4269];
      float t4271 = t4250 * t4268;
      float t4272 = t4271 > 0.0;
      float t4273 = metal::select(0.0, t4270, t4272 > 0.0);
      float t4274 = t4247 + t4273;
      t4247 = t4274;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4276 = t4245 * 8.0;
    float t4277 = t4276 + t4246;
    int t4278 = (int)t4277;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[33004992 + t4278], t4247, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 39
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_39(
    constant uint &frameCount [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4942 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4942)) {
    int t4281 = id;
    int t4282 = t4281 / 512;
    uint _frameIndex = (uint)(t4282);
    int t4283 = t4282 * 512;
    int t4284 = t4281 - t4283;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([512, 1]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 40
// FrameOrder: parallel
// DispatchMode: staticThreads(1)
kernel void kernel_40(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint t4285 = 0; t4285 < 512; t4285++) {
      int t4286 = t4285;
      int t4287 = t4286;
      int t4288 = t4285 - t4287;
      int t4289 = t4286 / 64;
      int t4290 = t4289 * 64;
      int t4291 = t4286 - t4290;
      int t4292 = t4291 * 8;
      int t4293 = t4289 + t4292;
      float t4294 = memory[33004992 + t4293];
      float t4295 = memory[626112 + (int)t4285];
      float t4296 = t4294 / t4295;
      float t4297 = memory[626112 + (int)t4285];
      float t4298 = memory[626112 + (int)t4285];
      float t4299 = t4297 * t4298;
      float t4300 = 1.0 / t4299;
      int t4301 = t4285;
      int t4302 = t4301;
      int t4303 = t4285 - t4302;
      int t4304 = t4301 / 64;
      int t4305 = t4304 * 64;
      int t4306 = t4301 - t4305;
      int t4307 = t4306 * 8;
      int t4308 = t4304 + t4307;
      float t4309 = memory[33004992 + t4308];
      float t4310 = t4309 * -1.0;
      float t4311 = t4310 * t4300;
      float t4312 = t4296 + t4311;
      float t4313 = memory[625600 + (int)t4285];
      float t4314 = metal::exp(t4313);
      float t4315 = t4314 * t4311;
      float t4316 = -1.0 * t4315;
      memory[625088 + (int)t4285] = t4316;
      float t4318 = memory[624576 + (int)t4285];
      float t4319 = t4318 * t4315;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t4320 = 0; t4320 < 1; t4320++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=186, axis=0, in=[512, 1], out=[1], inFA=false, outFA=false), value: empty) */
      float t4321 = 0.0;
      int t4322 = t4320;
      int t4323 = t4322;
      int t4324 = t4320 - t4323;
      int t4325 = t4322;
      int t4326 = t4325;
      for (uint t4327 = 0; t4327 < 512; t4327++) {
        int t4328 = t4327;
        int t4329 = t4326 + t4328;
        float t4330 = memory[625088 + t4329];
        float t4331 = t4321 + t4330;
        t4321 = t4331;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[626624 + (int)t4320] = t4321;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 41
// FrameOrder: parallel
// DispatchMode: staticThreads(65536)
kernel void kernel_41(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(65536)) {
    int t4334 = id;
    int t4335 = t4334 / 65536;
    uint _frameIndex = (uint)(t4335);
    int t4336 = t4335 * 65536;
    int t4337 = t4334 - t4336;
    /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=188, axis=2, in=[512, 1], out=[512, 1, 128], inFA=false, outFA=false), value: empty) */
    int t4338 = t4337 / 128;
    int t4339 = t4338 % 512;
    int t4340 = t4339 * 1;
    int t4341 = 0 + t4340;
    int t4342 = t4337 / 128;
    int t4343 = t4342 % 1;
    int t4344 = t4343 * 1;
    int t4345 = t4341 + t4344;
    float t4346 = memory[625088 + t4345];
    memory[165824 + t4337] = t4346;
    int t4348 = t4337 / 128;
    int t4349 = t4348 * 128;
    int t4350 = t4337 - t4349;
    int t4351 = t4350 / 128;
    int t4352 = t4351 * 128;
    int t4353 = t4350 - t4352;
    int t4354 = t4353 / 128;
    int t4355 = t4354 * 128;
    int t4356 = t4353 - t4355;
    float t4357 = memory[25280 + t4356];
    float t4358 = memory[165824 + t4337];
    float t4359 = t4357 * t4358;
    memory[17404352 + t4337] = t4359;
  }
  #pragma clang diagnostic pop
}



// KERNEL 42
// FrameOrder: parallel
// DispatchMode: perFrameScaled(128)
kernel void kernel_42(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4943 = frameCount * 128.0;
  if (id >= 0 && id < (uint)(t4943)) {
    int t4361 = id;
    int t4362 = t4361 / 128;
    uint _frameIndex = (uint)(t4362);
    int t4363 = t4362 * 128;
    int t4364 = t4361 - t4363;
    int t4365 = t4364 / 128;
    int t4366 = t4364 % 128;
    float t4367 = 0.0;
    for (uint t4368 = 0; t4368 < 512; t4368++) {
      int t4369 = t4368;
      int t4370 = t4369 + t4365;
      int t4371 = t4368 * 128;
      int t4372 = t4371 + t4366;
      float t4373 = memory[625088 + t4370];
      float t4374 = memory[427968 + t4372];
      float t4375 = t4373 * t4374;
      float t4376 = t4367 + t4375;
      t4367 = t4376;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4378 = t4365 * 128;
    int t4379 = t4378 + t4366;
    memory[624576 + t4379] = t4367;
  }
  #pragma clang diagnostic pop
}



// KERNEL 43
// FrameOrder: parallel
// DispatchMode: perFrameScaled(8)
kernel void kernel_43(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4944 = frameCount * 8.0;
  if (id >= 0 && id < (uint)(t4944)) {
    int t4381 = id;
    int t4382 = t4381 / 8;
    uint _frameIndex = (uint)(t4382);
    int t4383 = t4382 * 8;
    int t4384 = t4381 - t4383;
    int t4385 = _frameIndex;
    int t4386 = t4385 * 8;
    int t4387 = t4386 + t4384;
    float t4388 = memory[17928640 + t4387];
    int t4389 = _frameIndex;
    int t4390 = t4389 * 8;
    int t4391 = t4390 + t4384;
    float t4392 = memory[28807616 + t4391];
    float t4393 = t4388 * t4392;
    int t4394 = _frameIndex;
    int t4395 = t4394 * 8;
    int t4396 = t4395 + t4384;
    memory[17535424 + t4396] = t4393;
    int t4398 = _frameIndex;
    int t4399 = t4398 * 8;
    int t4400 = t4399 + t4384;
    float t4401 = memory[17797568 + t4400];
    int t4402 = _frameIndex;
    int t4403 = t4402 * 8;
    int t4404 = t4403 + t4384;
    float t4405 = memory[28807616 + t4404];
    float t4406 = t4401 * t4405;
  }
  #pragma clang diagnostic pop
}



// KERNEL 44
// FrameOrder: parallel
// DispatchMode: perFrameScaled(512)
kernel void kernel_44(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4945 = frameCount * 512.0;
  if (id >= 0 && id < (uint)(t4945)) {
    int t4407 = id;
    int t4408 = t4407 / 512;
    uint _frameIndex = (uint)(t4408);
    int t4409 = t4408 * 512;
    int t4410 = t4407 - t4409;
    /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=198, axis=1, in=[8], out=[8, 64], inFA=true, outFA=true), value: empty) */
    int t4411 = t4410 / 64;
    int t4412 = t4411 % 8;
    int t4413 = t4412 * 1;
    int t4414 = 0 + t4413;
    float t4415 = (float)t4414;
    int t4416 = _frameIndex;
    int t4417 = t4416 * 8;
    float t4418 = t4417 + t4415;
    int t4419 = (int)t4418;
    float t4420 = memory[17535424 + t4419];
    float t4421 = (float)t4410;
    int t4422 = _frameIndex;
    int t4423 = t4422 * 512;
    float t4424 = t4423 + t4421;
    int t4425 = (int)t4424;
    memory[33005504 + t4425] = t4420;
    int t4427 = _frameIndex;
    int t4428 = t4427 * 512;
    int t4429 = t4428 + t4410;
    float t4430 = memory[9015744 + t4429];
    int t4431 = _frameIndex;
    int t4432 = t4431 * 512;
    int t4433 = t4432 + t4410;
    float t4434 = memory[33005504 + t4433];
    float t4435 = t4430 * t4434;
    int t4436 = _frameIndex;
    int t4437 = t4436 * 512;
    int t4438 = t4437 + t4410;
    float t4439 = memory[627136 + t4438];
    int t4440 = _frameIndex;
    int t4441 = t4440 * 512;
    int t4442 = t4441 + t4410;
    float t4443 = memory[33005504 + t4442];
    float t4444 = t4439 * t4443;
    int t4445 = _frameIndex;
    int t4446 = t4445 * 512;
    int t4447 = t4446 + t4410;
    memory[18059712 + t4447] = t4444;
  }
  #pragma clang diagnostic pop
}



// KERNEL 45
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_45(
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
    float t4449 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 64.0) * 64.0);
    float t4450 = t4449 < 0.0;
    float t4451 = t4449 + 64.0;
    float t4452 = metal::select(t4449, t4451, t4450 > 0.0);
    float t4453 = metal::floor(t4452);
    float t4454 = t4453 + 1.0;
    float t4455 = t4454 >= 64.0;
    float t4456 = metal::select(t4454, 0.0, t4455 > 0.0);
    float t4457 = t4452 - t4453;
    int t4458 = id;
    memory[526272 + t4458] = t4453;
    memory[165824 + t4458] = t4457;
    float t4461 = t4458 + 16384.0;
    int t4462 = (int)t4461;
    memory[526272 + t4462] = t4456;
    float t4464 = 1.0 - t4457;
    float t4465 = t4458 * 512.0;
    for (uint _pr4466 = 0; _pr4466 < 512; _pr4466++) {
      float t4467 = (float)_pr4466;
      float t4468 = t4465 + t4467;
      int t4469 = (int)t4468;
      float t4470 = memory[18059712 + t4469];
      float t4471 = t4465 + t4467;
      float t4472 = t4470 * t4464;
      int t4473 = (int)t4471;
      memory[627136 + t4473] = t4472;
      float t4475 = t4470 * t4457;
      int t4476 = (int)t4471;
      memory[9015744 + t4476] = t4475;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 46
// FrameOrder: sequential
// DispatchMode: staticThreads(32768)
kernel void kernel_46(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 32768) { uint _pr4479 = id;
    int t4480 = _pr4479 / 512;
    int t4481 = t4480 * 512;
    int t4482 = _pr4479 - t4481;
    float t4483 = (float)t4480;
    float t4484 = (float)t4482;
    float t4485 = 0.0;
    for (uint t4486 = 0; t4486 < 16384; t4486++) {
      float t4487 = (float)t4486;
      float t4488 = t4487 < frameCount;
      float t4489 = t4487 * 512.0;
      float t4490 = t4489 + t4484;
      float t4491 = memory[526272 + (int)t4486];
      float t4492 = t4491 - t4483;
      float t4493 = metal::abs(t4492);
      float t4494 = t4493 < 0.5;
      int t4495 = (int)t4490;
      float t4496 = memory[627136 + t4495];
      float t4497 = t4488 * t4494;
      float t4498 = t4497 > 0.0;
      float t4499 = metal::select(0.0, t4496, t4498 > 0.0);
      float t4500 = t4485 + t4499;
      t4485 = t4500;
      float t4501 = t4487 + 16384.0;
      int t4502 = (int)t4501;
      float t4503 = memory[526272 + t4502];
      float t4504 = t4503 - t4483;
      float t4505 = metal::abs(t4504);
      float t4506 = t4505 < 0.5;
      int t4507 = (int)t4490;
      float t4508 = memory[9015744 + t4507];
      float t4509 = t4488 * t4506;
      float t4510 = t4509 > 0.0;
      float t4511 = metal::select(0.0, t4508, t4510 > 0.0);
      float t4512 = t4485 + t4511;
      t4485 = t4512;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t4514 = t4483 * 512.0;
    float t4515 = t4514 + t4484;
    int t4516 = (int)t4515;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41394112 + t4516], t4485, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 47
// FrameOrder: parallel
// DispatchMode: perFrameScaled(32768)
kernel void kernel_47(
    constant uint &frameCount [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4946 = frameCount * 32768.0;
  if (id >= 0 && id < (uint)(t4946)) {
    int t4519 = id;
    int t4520 = t4519 / 32768;
    uint _frameIndex = (uint)(t4520);
    int t4521 = t4520 * 32768;
    int t4522 = t4519 - t4521;
    /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0, 2]), value: empty) */
    /* [1mUOp[0m(op: [38;5;51mreshape[0m([512, 64]), value: empty) */
  }
  #pragma clang diagnostic pop
}



// KERNEL 48
// FrameOrder: parallel
// DispatchMode: staticThreads(1)
kernel void kernel_48(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint t4523 = 0; t4523 < 32768; t4523++) {
      int t4524 = t4523 / 64;
      int t4525 = t4524 * 64;
      int t4526 = t4523 - t4525;
      int t4527 = t4524 * 64;
      int t4528 = t4527 + t4526;
      int t4529 = t4528 / 4096;
      int t4530 = t4529 * 4096;
      int t4531 = t4528 - t4530;
      int t4532 = t4531 / 64;
      int t4533 = t4532 * 64;
      int t4534 = t4531 - t4533;
      int t4535 = t4529 * 64;
      int t4536 = t4532 * 512;
      int t4537 = t4535 + t4536;
      int t4538 = t4537 + t4534;
      float t4539 = memory[41394112 + t4538];
      float t4540 = memory[559040 + (int)t4523];
      float t4541 = t4539 / t4540;
      float t4542 = memory[559040 + (int)t4523];
      float t4543 = memory[559040 + (int)t4523];
      float t4544 = t4542 * t4543;
      float t4545 = 1.0 / t4544;
      int t4546 = t4523 / 64;
      int t4547 = t4546 * 64;
      int t4548 = t4523 - t4547;
      int t4549 = t4546 * 64;
      int t4550 = t4549 + t4548;
      int t4551 = t4550 / 4096;
      int t4552 = t4551 * 4096;
      int t4553 = t4550 - t4552;
      int t4554 = t4553 / 64;
      int t4555 = t4554 * 64;
      int t4556 = t4553 - t4555;
      int t4557 = t4551 * 64;
      int t4558 = t4554 * 512;
      int t4559 = t4557 + t4558;
      int t4560 = t4559 + t4556;
      float t4561 = memory[41394112 + t4560];
      float t4562 = t4561 * -1.0;
      float t4563 = t4562 * t4545;
      float t4564 = t4541 + t4563;
      float t4565 = memory[591808 + (int)t4523];
      float t4566 = metal::exp(t4565);
      float t4567 = t4566 * t4563;
      float t4568 = -1.0 * t4567;
      memory[526272 + (int)t4523] = t4568;
      float t4570 = memory[493504 + (int)t4523];
      float t4571 = t4570 * t4567;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t4572 = 0; t4572 < 64; t4572++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=219, axis=0, in=[512, 64], out=[64], inFA=false, outFA=false), value: empty) */
      float t4573 = 0.0;
      int t4574 = t4572;
      int t4575 = t4574;
      int t4576 = t4572 - t4575;
      int t4577 = t4574;
      int t4578 = t4577;
      for (uint t4579 = 0; t4579 < 512; t4579++) {
        int t4580 = t4579 * 64;
        int t4581 = t4578 + t4580;
        float t4582 = memory[526272 + t4581];
        float t4583 = t4573 + t4582;
        t4573 = t4583;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[625088 + (int)t4572] = t4573;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 49
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 64, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_49(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4586 = gid.y;
    int t4587 = gid.x;
    int t4588 = gid.z;
    metal::simdgroup_float8x8 t4589 = metal::simdgroup_float8x8(0);
    for (uint t4590 = 0; t4590 < 8; t4590++) {
      int t4591 = t4586 * 512;
      int t4592 = t4591;
      int t4593 = t4590 * 8;
      int t4594 = t4592 + t4593;
      metal::simdgroup_float8x8 t4595 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4595, &memory[526272 + t4594], 64);
      int t4596 = t4587 * 512;
      int t4597 = t4596;
      int t4598 = t4590 * 8;
      int t4599 = t4597 + t4598;
      metal::simdgroup_float8x8 t4600 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4600, &memory[17024 + t4599], 64, ulong2(0, 0), true);
      metal::simdgroup_multiply_accumulate(t4589, t4595, t4600, t4589);
    }
    int t4602 = t4586 * 1024;
    int t4603 = t4602;
    int t4604 = t4587 * 8;
    int t4605 = t4603 + t4604;
    metal::simdgroup_store(t4589, &memory[165824 + t4605], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 50
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 8, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_50(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4607 = gid.y;
    int t4608 = gid.x;
    int t4609 = gid.z;
    metal::simdgroup_float8x8 t4610 = metal::simdgroup_float8x8(0);
    for (uint t4611 = 0; t4611 < 64; t4611++) {
      int t4612 = t4611 * 512;
      int t4613 = t4612;
      int t4614 = t4607 * 8;
      int t4615 = t4613 + t4614;
      metal::simdgroup_float8x8 t4616 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4616, &memory[526272 + t4615], 64, ulong2(0, 0), true);
      int t4617 = t4611 * 1024;
      int t4618 = t4617;
      int t4619 = t4608 * 8;
      int t4620 = t4618 + t4619;
      metal::simdgroup_float8x8 t4621 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4621, &memory[427968 + t4620], 128);
      metal::simdgroup_multiply_accumulate(t4610, t4616, t4621, t4610);
    }
    int t4623 = t4607 * 1024;
    int t4624 = t4623;
    int t4625 = t4608 * 8;
    int t4626 = t4624 + t4625;
    metal::simdgroup_store(t4610, &memory[493504 + t4626], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 51
// FrameOrder: parallel
// DispatchMode: staticThreads(65536)
kernel void kernel_51(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(65536)) {
    int t4628 = id;
    int t4629 = t4628 / 65536;
    uint _frameIndex = (uint)(t4629);
    int t4630 = t4629 * 65536;
    int t4631 = t4628 - t4630;
    float t4632 = memory[17404352 + t4631];
    float t4633 = memory[165824 + t4631];
    float t4634 = t4632 + t4633;
    memory[17535424 + t4631] = t4634;
    float t4636 = memory[362432 + t4631];
    float t4637 = metal::tanh(t4636);
    float t4638 = t4637 * t4637;
    float t4639 = 1.0 - t4638;
    memory[17666496 + t4631] = t4639;
    float t4641 = t4639 * t4634;
    memory[17797568 + t4631] = t4641;
  }
  #pragma clang diagnostic pop
}



// KERNEL 52
// FrameOrder: parallel
// DispatchMode: staticThreads(128)
kernel void kernel_52(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(128)) {
    int t4643 = id;
    int t4644 = t4643 / 128;
    uint _frameIndex = (uint)(t4644);
    int t4645 = t4644 * 128;
    int t4646 = t4643 - t4645;
    float t4647 = 0.0;
    for (uint t4648 = 0; t4648 < 512; t4648++) {
      int t4649 = t4648 * 128;
      int t4650 = t4649 + t4646;
      float t4651 = memory[17666496 + t4650];
      int t4652 = t4648 * 128;
      int t4653 = t4652 + t4646;
      float t4654 = memory[17535424 + t4653];
      float t4655 = t4651 * t4654;
      float t4656 = t4647 + t4655;
      t4647 = t4656;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[625600 + t4646] = t4647;
  }
  #pragma clang diagnostic pop
}



// KERNEL 53
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 64, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_53(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4663 = gid.y;
    int t4664 = gid.x;
    int t4665 = gid.z;
    metal::simdgroup_float8x8 t4666 = metal::simdgroup_float8x8(0);
    for (uint t4667 = 0; t4667 < 16; t4667++) {
      int t4668 = t4663 * 1024;
      int t4669 = t4668;
      int t4670 = t4667 * 8;
      int t4671 = t4669 + t4670;
      metal::simdgroup_float8x8 t4672 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4672, &memory[17797568 + t4671], 128);
      int t4673 = t4664 * 1024;
      int t4674 = t4673;
      int t4675 = t4667 * 8;
      int t4676 = t4674 + t4675;
      metal::simdgroup_float8x8 t4677 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4677, &memory[512 + t4676], 128, ulong2(0, 0), true);
      metal::simdgroup_multiply_accumulate(t4666, t4672, t4677, t4666);
    }
    int t4679 = t4663 * 1024;
    int t4680 = t4679;
    int t4681 = t4664 * 8;
    int t4682 = t4680 + t4681;
    metal::simdgroup_store(t4666, &memory[165824 + t4682], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 54
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 16, tilesN: 16, depth: nil)
#include <metal_simdgroup_matrix>
kernel void kernel_54(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4684 = gid.y;
    int t4685 = gid.x;
    int t4686 = gid.z;
    metal::simdgroup_float8x8 t4687 = metal::simdgroup_float8x8(0);
    for (uint t4688 = 0; t4688 < 64; t4688++) {
      int t4689 = t4688 * 1024;
      int t4690 = t4689;
      int t4691 = t4684 * 8;
      int t4692 = t4690 + t4691;
      metal::simdgroup_float8x8 t4693 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4693, &memory[17797568 + t4692], 128, ulong2(0, 0), true);
      int t4694 = t4688 * 1024;
      int t4695 = t4694;
      int t4696 = t4685 * 8;
      int t4697 = t4695 + t4696;
      metal::simdgroup_float8x8 t4698 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4698, &memory[231360 + t4697], 128);
      metal::simdgroup_multiply_accumulate(t4687, t4693, t4698, t4687);
    }
    int t4700 = t4684 * 1024;
    int t4701 = t4700;
    int t4702 = t4685 * 8;
    int t4703 = t4701 + t4702;
    metal::simdgroup_store(t4687, &memory[493504 + t4703], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 55
// FrameOrder: parallel
// DispatchMode: staticThreads(65536)
kernel void kernel_55(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(65536)) {
    int t4705 = id;
    int t4706 = t4705 / 65536;
    uint _frameIndex = (uint)(t4706);
    int t4707 = t4706 * 65536;
    int t4708 = t4705 - t4707;
    float t4709 = memory[296896 + t4708];
    float t4710 = metal::tanh(t4709);
    float t4711 = t4710 * t4710;
    float t4712 = 1.0 - t4711;
    memory[17404352 + t4708] = t4712;
    float t4714 = memory[165824 + t4708];
    float t4715 = t4712 * t4714;
    memory[362432 + t4708] = t4715;
  }
  #pragma clang diagnostic pop
}



// KERNEL 56
// FrameOrder: parallel
// DispatchMode: staticThreads(128)
kernel void kernel_56(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(128)) {
    int t4717 = id;
    int t4718 = t4717 / 128;
    uint _frameIndex = (uint)(t4718);
    int t4719 = t4718 * 128;
    int t4720 = t4717 - t4719;
    float t4721 = 0.0;
    for (uint t4722 = 0; t4722 < 512; t4722++) {
      int t4723 = t4722 * 128;
      int t4724 = t4723 + t4720;
      float t4725 = memory[17404352 + t4724];
      int t4726 = t4722 * 128;
      int t4727 = t4726 + t4720;
      float t4728 = memory[165824 + t4727];
      float t4729 = t4725 * t4728;
      float t4730 = t4721 + t4729;
      t4721 = t4730;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[626112 + t4720] = t4721;
  }
  #pragma clang diagnostic pop
}



// KERNEL 57
// FrameOrder: parallel
// DispatchMode: perFrameScaled(1536)
kernel void kernel_57(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4947 = frameCount * 1536.0;
  if (id >= 0 && id < (uint)(t4947)) {
    int t4737 = id;
    int t4738 = t4737 / 1536;
    uint _frameIndex = (uint)(t4738);
    int t4739 = t4738 * 1536;
    int t4740 = t4737 - t4739;
    int t4741 = t4740 / 3;
    int t4742 = t4740 % 3;
    float t4743 = 0.0;
    for (uint t4744 = 0; t4744 < 128; t4744++) {
      int t4745 = t4741 * 128;
      int t4746 = t4745 + t4744;
      int t4747 = t4742 * 128;
      int t4748 = t4747 + t4744;
      float t4749 = memory[362432 + t4746];
      float t4750 = memory[0 + t4748];
      float t4751 = t4749 * t4750;
      float t4752 = t4743 + t4751;
      t4743 = t4752;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4754 = t4741 * 3;
    int t4755 = t4754 + t4742;
    memory[493504 + t4755] = t4743;
  }
  #pragma clang diagnostic pop
}



// KERNEL 58
// FrameOrder: parallel
// DispatchMode: perFrameScaled(384)
kernel void kernel_58(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t4948 = frameCount * 384.0;
  if (id >= 0 && id < (uint)(t4948)) {
    int t4757 = id;
    int t4758 = t4757 / 384;
    uint _frameIndex = (uint)(t4758);
    int t4759 = t4758 * 384;
    int t4760 = t4757 - t4759;
    int t4761 = t4760 / 3;
    int t4762 = t4760 % 3;
    float t4763 = 0.0;
    for (uint t4764 = 0; t4764 < 512; t4764++) {
      int t4765 = t4764 * 128;
      int t4766 = t4765 + t4761;
      int t4767 = t4764 * 3;
      int t4768 = t4767 + t4762;
      float t4769 = memory[362432 + t4766];
      float t4770 = memory[25538 + t4768];
      float t4771 = t4769 * t4770;
      float t4772 = t4763 + t4771;
      t4763 = t4772;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4774 = t4761 * 3;
    int t4775 = t4774 + t4762;
    memory[33004992 + t4775] = t4763;
  }
  #pragma clang diagnostic pop
}



// KERNEL 59
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_59(
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



// KERNEL 60
// FrameOrder: sequential
// DispatchMode: staticThreads(384)
kernel void kernel_60(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 384) { uint _pr4777 = id;
    float t4778 = (float)_pr4777;
    float t4779 = (t4778 * 0.0078125);
    float t4780 = metal::floor(t4779);
    float t4781 = t4780 * 128.0;
    float t4782 = t4778 - t4781;
    float t4783 = t4782 * 3.0;
    float t4784 = t4780 + t4783;
    int t4785 = (int)t4784;
    float t4786 = memory[33004992 + t4785];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41426880 + (int)_pr4777], t4786, metal::memory_order_relaxed);
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
    float t4791 = (t4790 * 0.0078125);
    float t4792 = metal::floor(t4791);
    float t4793 = t4792 * 128.0;
    float t4794 = t4790 - t4793;
    int t4795 = (int)t4794;
    float t4796 = memory[626112 + t4795];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41427264 + (int)_pr4789], t4796, metal::memory_order_relaxed);
  }
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 128) { uint _pr4799 = id;
    float t4800 = (float)_pr4799;
    float t4801 = (t4800 * 0.0078125);
    float t4802 = metal::floor(t4801);
    float t4803 = t4802 * 128.0;
    float t4804 = t4800 - t4803;
    int t4805 = (int)t4804;
    float t4806 = memory[625600 + t4805];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41427392 + (int)_pr4799], t4806, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 62
// FrameOrder: sequential
// DispatchMode: staticThreads(64)
kernel void kernel_62(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr4809 = id;
    float t4810 = (float)_pr4809;
    float t4811 = (t4810 * 0.015625);
    float t4812 = metal::floor(t4811);
    float t4813 = t4812 * 64.0;
    float t4814 = t4810 - t4813;
    int t4815 = (int)t4814;
    float t4816 = memory[625088 + t4815];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41427520 + (int)_pr4809], t4816, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 63
// FrameOrder: sequential
// DispatchMode: staticThreads(128)
kernel void kernel_63(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 128) { uint _pr4819 = id;
    float t4820 = (float)_pr4819;
    float t4821 = t4820;
    float t4822 = metal::floor(t4821);
    float t4823 = t4822;
    float t4824 = t4820 - t4823;
    int t4825 = (int)t4822;
    float t4826 = memory[624576 + t4825];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41427584 + (int)_pr4819], t4826, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 64
// FrameOrder: sequential
// DispatchMode: staticThreads(1)
kernel void kernel_64(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 1) { uint _pr4829 = id;
    float t4830 = (float)_pr4829;
    float t4831 = t4830;
    float t4832 = metal::floor(t4831);
    float t4833 = t4832;
    float t4834 = t4830 - t4833;
    float t4835 = memory[626624 + (int)0.0];
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[41427712 + (int)_pr4829], t4835, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 65
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_65(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2356) - handled in variable access */
  }
  #pragma clang diagnostic pop
}



// KERNEL 66
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 16, tilesN: 16, depth: Optional(256))
#include <metal_simdgroup_matrix>
kernel void kernel_66(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4838 = gid.y;
    int t4839 = gid.x;
    int t4840 = gid.z;
    metal::simdgroup_float8x8 t4841 = metal::simdgroup_float8x8(0);
    int t4842 = (int)frameCount;
    int t4843 = t4840 * 64;
    int t4844 = t4843 + 64;
    int t4845 = t4840 == 0.0;
    if (t4845) {
      for (uint t4847 = 0; t4847 < 64; t4847++) {
        int t4848 = t4847 * 1024;
        int t4849 = t4848;
        int t4850 = t4838 * 8;
        int t4851 = t4849 + t4850;
        metal::simdgroup_float8x8 t4852 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4852, &memory[17797568 + t4851], 128, ulong2(0, 0), true);
        int t4853 = t4847 * 1024;
        int t4854 = t4853;
        int t4855 = t4839 * 8;
        int t4856 = t4854 + t4855;
        metal::simdgroup_float8x8 t4857 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4857, &memory[231360 + t4856], 128);
        metal::simdgroup_multiply_accumulate(t4841, t4852, t4857, t4841);
      }
    }
    int t4860 = t4840 * 16384;
    int t4861 = t4838 * 1024;
    int t4862 = t4860 + t4861;
    int t4863 = t4839 * 8;
    int t4864 = t4862 + t4863;
    metal::simdgroup_store(t4841, &memory[627136 + t4864], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 67
// FrameOrder: sequential
// DispatchMode: staticThreads(16384)
kernel void kernel_67(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 16384) { uint _pr4866 = id;
    float t4867 = 0.0;
    for (uint t4868 = 0; t4868 < 256; t4868++) {
      int t4869 = t4868 * 16384;
      int t4870 = t4869 + _pr4866;
      float t4871 = memory[627136 + t4870];
      float t4872 = t4867 + t4871;
      t4867 = t4872;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4874 = _pr4866 / 128;
    int t4875 = _pr4866 % 128;
    int t4876 = t4875 * 128;
    int t4877 = t4876 + t4874;
    memory[493504 + t4877] = t4867;
  }
  #pragma clang diagnostic pop
}



// KERNEL 68
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_68(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2356) - handled in variable access */
  }
  #pragma clang diagnostic pop
}



// KERNEL 69
// FrameOrder: parallel
// DispatchMode: gemm(tilesM: 8, tilesN: 16, depth: Optional(256))
#include <metal_simdgroup_matrix>
kernel void kernel_69(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  {
    int t4880 = gid.y;
    int t4881 = gid.x;
    int t4882 = gid.z;
    metal::simdgroup_float8x8 t4883 = metal::simdgroup_float8x8(0);
    int t4884 = (int)frameCount;
    int t4885 = t4882 * 64;
    int t4886 = t4885 + 64;
    int t4887 = t4882 == 0.0;
    if (t4887) {
      for (uint t4889 = 0; t4889 < 64; t4889++) {
        int t4890 = t4889 * 512;
        int t4891 = t4890;
        int t4892 = t4880 * 8;
        int t4893 = t4891 + t4892;
        metal::simdgroup_float8x8 t4894 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4894, &memory[526272 + t4893], 64, ulong2(0, 0), true);
        int t4895 = t4889 * 1024;
        int t4896 = t4895;
        int t4897 = t4881 * 8;
        int t4898 = t4896 + t4897;
        metal::simdgroup_float8x8 t4899 = metal::simdgroup_float8x8(0); metal::simdgroup_load(t4899, &memory[427968 + t4898], 128);
        metal::simdgroup_multiply_accumulate(t4883, t4894, t4899, t4883);
      }
    }
    int t4902 = t4882 * 8192;
    int t4903 = t4880 * 1024;
    int t4904 = t4902 + t4903;
    int t4905 = t4881 * 8;
    int t4906 = t4904 + t4905;
    metal::simdgroup_store(t4883, &memory[627136 + t4906], 128);
  }
  #pragma clang diagnostic pop
}



// KERNEL 70
// FrameOrder: sequential
// DispatchMode: staticThreads(8192)
kernel void kernel_70(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 8192) { uint _pr4908 = id;
    float t4909 = 0.0;
    for (uint t4910 = 0; t4910 < 256; t4910++) {
      int t4911 = t4910 * 8192;
      int t4912 = t4911 + _pr4908;
      float t4913 = memory[627136 + t4912];
      float t4914 = t4909 + t4913;
      t4909 = t4914;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    int t4916 = _pr4908 / 128;
    int t4917 = _pr4908 % 128;
    int t4918 = t4917 * 64;
    int t4919 = t4918 + t4916;
    memory[493504 + t4919] = t4909;
  }
  #pragma clang diagnostic pop
}



// KERNEL 71
// FrameOrder: parallel
// DispatchMode: perFrame
kernel void kernel_71(
    device float *outputs [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(2356) - handled in variable access */
    outputs[0 * frameCount + id] = t[8*frameCount + id];
  }
  #pragma clang diagnostic pop
}

