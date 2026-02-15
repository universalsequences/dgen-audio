// KERNEL 0
// Kind: simd
// ThreadCountScale nil
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
  if (id >= 0 && id < (uint)(1)) {
    for (uint t89 = 0; t89 < 3904; t89++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=19, axis=2, in=[61, 64, 3], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t90 = 0.0;
      int t91 = t89 / 64;
      int t92 = t91 * 64;
      int t93 = t89 - t92;
      int t94 = t93;
      int t95 = t94;
      int t96 = t93 - t95;
      int t97 = t91 * 192;
      int t98 = t97;
      int t99 = t94 * 3;
      int t100 = t98 + t99;
      for (uint t101 = 0; t101 < 3; t101++) {
        int t102 = t101;
        int t103 = t100 + t102;
        int t104 = t91 * 3;
        int t105 = t104 + t101;
        float t106 = memory[8706 + t105];
        int t107 = t94 * 3;
        int t108 = t107 + t101;
        int t109 = t108 / 3;
        int t110 = t109 * 3;
        int t111 = t108 - t110;
        int t112 = t111 * 64;
        int t113 = t109 + t112;
        float t114 = memory[0 + t113];
        float t115 = t106 * t114;
        float t116 = t90 + t115;
        t90 = t116;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t89] = t90;
      float t119 = memory[25460 + (int)t89];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t120 = t89 / 64;
      int t121 = t120 * 64;
      int t122 = t89 - t121;
      int t123 = t122;
      float t124 = memory[192 + t123];
      float t125 = t119 + t124;
      memory[33268 + (int)t89] = t125;
      float t127 = metal::tanh(t125);
      memory[29364 + (int)t89] = t127;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 1
// Kind: simd
// ThreadCountScale nil
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
  if (id >= 0 && id < (uint)(1)) {
    for (uint t129 = 0; t129 < 3904; t129++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=26, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t130 = 0.0;
      int t131 = t129 / 64;
      int t132 = t131 * 64;
      int t133 = t129 - t132;
      int t134 = t133;
      int t135 = t134;
      int t136 = t133 - t135;
      int t137 = t131 * 4096;
      int t138 = t137;
      int t139 = t134 * 64;
      int t140 = t138 + t139;
      for (uint t141 = 0; t141 < 64; t141++) {
        int t142 = t141;
        int t143 = t140 + t142;
        int t144 = t131 * 64;
        int t145 = t144 + t141;
        float t146 = memory[29364 + t145];
        int t147 = t134 * 64;
        int t148 = t147 + t141;
        int t149 = t148 / 64;
        int t150 = t149 * 64;
        int t151 = t148 - t150;
        int t152 = t151 * 64;
        int t153 = t149 + t152;
        float t154 = memory[256 + t153];
        float t155 = t146 * t154;
        float t156 = t130 + t155;
        t130 = t156;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t129] = t130;
      float t159 = memory[25460 + (int)t129];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t160 = t129 / 64;
      int t161 = t160 * 64;
      int t162 = t129 - t161;
      int t163 = t162;
      float t164 = memory[4352 + t163];
      float t165 = t159 + t164;
      memory[37172 + (int)t129] = t165;
      float t167 = metal::tanh(t165);
      memory[41076 + (int)t129] = t167;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
    for (uint t169 = 0; t169 < 3904; t169++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=33, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t170 = 0.0;
      int t171 = t169 / 64;
      int t172 = t171 * 64;
      int t173 = t169 - t172;
      int t174 = t173;
      int t175 = t174;
      int t176 = t173 - t175;
      int t177 = t171 * 4096;
      int t178 = t177;
      int t179 = t174 * 64;
      int t180 = t178 + t179;
      for (uint t181 = 0; t181 < 64; t181++) {
        int t182 = t181;
        int t183 = t180 + t182;
        int t184 = t171 * 64;
        int t185 = t184 + t181;
        float t186 = memory[41076 + t185];
        int t187 = t174 * 64;
        int t188 = t187 + t181;
        int t189 = t188 / 64;
        int t190 = t189 * 64;
        int t191 = t188 - t190;
        int t192 = t191 * 64;
        int t193 = t189 + t192;
        float t194 = memory[4416 + t193];
        float t195 = t186 * t194;
        float t196 = t170 + t195;
        t170 = t196;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t169] = t170;
      float t199 = memory[25460 + (int)t169];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t200 = t169 / 64;
      int t201 = t200 * 64;
      int t202 = t169 - t201;
      int t203 = t202;
      float t204 = memory[8512 + t203];
      float t205 = t199 + t204;
      memory[56692 + (int)t169] = t205;
      float t207 = t205 * -1.0;
      memory[44980 + (int)t169] = t207;
      float t209 = metal::exp(t207);
      float t210 = 1.0 + t209;
      memory[52788 + (int)t169] = t210;
      float t212 = 1.0 / t210;
      memory[48884 + (int)t169] = t212;
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
    for (uint t214 = 0; t214 < 61; t214++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=45, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t215 = 0.0;
      int t216 = t214;
      int t217 = t216;
      int t218 = t214 - t217;
      int t219 = t218;
      int t220 = t219;
      int t221 = t218 - t220;
      int t222 = t216 * 64;
      int t223 = t222;
      int t224 = t219 * 64;
      int t225 = t223 + t224;
      for (uint t226 = 0; t226 < 64; t226++) {
        int t227 = t226;
        int t228 = t225 + t227;
        int t229 = t216 * 64;
        int t230 = t229 + t226;
        float t231 = memory[41076 + t230];
        int t232 = t226 / 64;
        int t233 = t232 * 64;
        int t234 = t226 - t233;
        float t235 = memory[8576 + t234];
        float t236 = t231 * t235;
        float t237 = t215 + t236;
        t215 = t237;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t214] = t215;
      float t240 = memory[25460 + (int)t214];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t241 = t214;
      int t242 = t241;
      int t243 = t214 - t242;
      float t244 = memory[8640 + (int)0.0];
      float t245 = t240 + t244;
      memory[60724 + (int)t214] = t245;
      float t247 = t245 * -1.0;
      memory[60596 + (int)t214] = t247;
      float t249 = metal::exp(t247);
      float t250 = 1.0 + t249;
      memory[60788 + (int)t214] = t250;
      float t252 = 1.0 / t250;
      memory[60660 + (int)t214] = t252;
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
    for (uint t254 = 0; t254 < 61; t254++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=57, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t255 = 0.0;
      int t256 = t254;
      int t257 = t256;
      int t258 = t254 - t257;
      int t259 = t258;
      int t260 = t259;
      int t261 = t258 - t260;
      int t262 = t256 * 64;
      int t263 = t262;
      int t264 = t259 * 64;
      int t265 = t263 + t264;
      for (uint t266 = 0; t266 < 64; t266++) {
        int t267 = t266;
        int t268 = t265 + t267;
        int t269 = t256 * 64;
        int t270 = t269 + t266;
        float t271 = memory[41076 + t270];
        int t272 = t266 / 64;
        int t273 = t272 * 64;
        int t274 = t266 - t273;
        float t275 = memory[8641 + t274];
        float t276 = t271 * t275;
        float t277 = t255 + t276;
        t255 = t277;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[25460 + (int)t254] = t255;
      float t280 = memory[25460 + (int)t254];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t281 = t254;
      int t282 = t281;
      int t283 = t254 - t282;
      float t284 = memory[8705 + (int)0.0];
      float t285 = t280 + t284;
      float t286 = t285 * -1.0;
      float t287 = metal::exp(t286);
      float t288 = 1.0 + t287;
      float t289 = 1.0 / t288;
      memory[60852 + (int)t254] = t289;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(291), value: global(291)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[599948660];
      float t292 = t[0*frameCount + i] + 0.003662333;
      float t293 = metal::select(t292, 0.0, 0.0 > 0.0);
      float t294 = t293;
      float t295 = (t294 * 0.016666668);
      float t296 = metal::floor(t295);
      float t297 = t296 * 60.0;
      float t298 = t293 - t297;
      memory[599948660] = t298;
      float t300 = t298 >= 60.0;
      if (t300) {
        float t302 = t298 - 60.0;
        memory[599948660] = t302;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(349), value: global(349)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(329), value: global(329)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(309), value: global(309)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(291) - handled in variable access */
    float t308 = metal::min(t[0*frameCount + id], 59.9999);
    t[1*frameCount + id] = metal::max(t308, 0.0);
    float t310 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t311 = t310 < 0.0;
    float t312 = t310 + 61.0;
    float t313 = metal::select(t310, t312, t311 > 0.0);
    float t314 = t313;
    float t315 = metal::floor(t314);
    float t316 = t314 - t315;
    float t317 = t315 + 1.0;
    float t318 = t317 >= 61.0;
    float t319 = metal::select(t317, 0.0, t318 > 0.0);
    int t320 = (int)t315;
    float t321 = memory[25273 + t320];
    int t322 = (int)t319;
    float t323 = memory[25273 + t322];
    float t324 = 1.0 - t316;
    float t325 = t321 * t324;
    float t326 = t323 * t316;
    float t327 = t325 + t326;
    float t328 = metal::max(t327, 20.0);
    t[2*frameCount + id] = metal::min(t328, 500.0);
    float t330 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t331 = t330 < 0.0;
    float t332 = t330 + 61.0;
    float t333 = metal::select(t330, t332, t331 > 0.0);
    float t334 = t333;
    float t335 = metal::floor(t334);
    float t336 = t334 - t335;
    float t337 = t335 + 1.0;
    float t338 = t337 >= 61.0;
    float t339 = metal::select(t337, 0.0, t338 > 0.0);
    int t340 = (int)t335;
    float t341 = memory[25334 + t340];
    int t342 = (int)t339;
    float t343 = memory[25334 + t342];
    float t344 = 1.0 - t336;
    float t345 = t341 * t344;
    float t346 = t343 * t336;
    float t347 = t345 + t346;
    float t348 = metal::min(t347, 1.0);
    t[3*frameCount + id] = metal::max(t348, 0.0);
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
  float t5718 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5718)) {
    /* loadGlobal(329) - handled in variable access */
    /* loadGlobal(309) - handled in variable access */
    int t350 = id;
    int t351 = t350 / 64;
    uint _frameIndex = (uint)(t351);
    int t352 = t351 * 64;
    int t353 = t350 - t352;
    float t354 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t355 = t354 < 0.0;
    float t356 = t354 + 61.0;
    float t357 = metal::select(t354, t356, t355 > 0.0);
    float t358 = metal::floor(t357);
    float t359 = t358 + 1.0;
    float t360 = t359 >= 61.0;
    float t361 = metal::select(t359, 0.0, t360 > 0.0);
    float t362 = t357 - t358;
    float t363 = 1.0 - t362;
    float t364 = t351 * 64.0;
    float t365 = (float)t353;
    float t366 = t358 * 64.0;
    float t367 = t366 + t365;
    int t368 = (int)t367;
    float t369 = memory[48884 + t368];
    float t370 = t361 * 64.0;
    float t371 = t370 + t365;
    int t372 = (int)t371;
    float t373 = memory[48884 + t372];
    float t374 = t363 * t369;
    float t375 = t362 * t373;
    float t376 = t374 + t375;
    float t377 = t364 + t365;
    int t378 = (int)t377;
    memory[60916 + t378] = t376;
    int t380 = (int)t377;
    memory[1109492 + t380] = t376;
    float t382 = memory[25395 + t353];
    float t383 = t382 * t[2*frameCount + _frameIndex];
    int t384 = _frameIndex;
    int t385 = t384 * 64;
    int t386 = t385 + t353;
    memory[2158068 + t386] = t383;
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
      int t388 = id;
      int t389 = i;
      int t390 = t389 * 64;
      int t391 = t390 + t388;
      float t392 = memory[2158068 + t391];
      float t393 = (t392 * 6.25e-05);
      float t394 = memory[25460 + t388];
      float t395 = t394 + t393;
      float t396 = metal::select(t395, 0.0, 0.0 > 0.0);
      float t397 = metal::floor(t396);
      float t398 = t396 - t397;
      float t399 = t398 >= 1.0;
      float t400 = t398 - 1.0;
      float t401 = metal::select(t398, t400, t399 > 0.0);
      float t402 = metal::select(t401, 0.0, 0.0 > 0.0);
      memory[25460 + t388] = t402;
      int t404 = i;
      int t405 = t404 * 64;
      int t406 = t405 + t388;
      memory[60916 + t406] = t394;
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
  float t5719 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5719)) {
    int t408 = id;
    int t409 = t408 / 64;
    uint _frameIndex = (uint)(t409);
    int t410 = t409 * 64;
    int t411 = t408 - t410;
    int t412 = _frameIndex;
    int t413 = t412 * 64;
    int t414 = t413 + t411;
    float t415 = memory[60916 + t414];
    float t416 = t415 * 6.283185;
    float t417 = metal::sin(t416);
    int t418 = _frameIndex;
    int t419 = t418 * 64;
    int t420 = t419 + t411;
    memory[2158068 + t420] = t417;
    int t422 = _frameIndex;
    int t423 = t422 * 64;
    int t424 = t423 + t411;
    float t425 = memory[1109492 + t424];
    float t426 = t417 * t425;
    int t427 = _frameIndex;
    int t428 = t427 * 64;
    int t429 = t428 + t411;
    memory[3206644 + t429] = t426;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(431), value: global(431)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    t[4*frameCount + id] = 0.0;
    for (uint t432 = 0; t432 < 64; t432++) {
      int t433 = id;
      int t434 = t433 * 64;
      int t435 = t434 + t432;
      float t436 = memory[3206644 + t435];
      float t437 = t[4*frameCount + id] + t436;
      t[4*frameCount + id] = t437;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(459), value: global(459)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(458), value: global(458)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(457), value: global(457)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(439), value: global(439)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(431) - handled in variable access */
    /* loadGlobal(349) - handled in variable access */
    /* loadGlobal(309) - handled in variable access */
    t[5*frameCount + id] = t[4*frameCount + id] * t[3*frameCount + id];
    float t440 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t441 = t440 < 0.0;
    float t442 = t440 + 61.0;
    float t443 = metal::select(t440, t442, t441 > 0.0);
    float t444 = t443;
    float t445 = metal::floor(t444);
    float t446 = t444 - t445;
    float t447 = t445 + 1.0;
    float t448 = t447 >= 61.0;
    float t449 = metal::select(t447, 0.0, t448 > 0.0);
    int t450 = (int)t445;
    float t451 = memory[60660 + t450];
    int t452 = (int)t449;
    float t453 = memory[60660 + t452];
    float t454 = 1.0 - t446;
    float t455 = t451 * t454;
    float t456 = t453 * t446;
    t[6*frameCount + id] = t455 + t456;
    t[7*frameCount + id] = t[5*frameCount + id] * t[6*frameCount + id];
    t[8*frameCount + id] = t[7*frameCount + id] * 0.015625;
    float t460 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t461 = t460 < 0.0;
    float t462 = t460 + 61.0;
    float t463 = metal::select(t460, t462, t461 > 0.0);
    float t464 = t463;
    float t465 = metal::floor(t464);
    float t466 = t464 - t465;
    float t467 = t465 + 1.0;
    float t468 = t467 >= 61.0;
    float t469 = metal::select(t467, 0.0, t468 > 0.0);
    int t470 = (int)t465;
    float t471 = memory[60852 + t470];
    int t472 = (int)t469;
    float t473 = memory[60852 + t472];
    float t474 = 1.0 - t466;
    float t475 = t471 * t474;
    float t476 = t473 * t466;
    float t477 = t475 + t476;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(478), value: global(478)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[9*frameCount + i] = memory[599948661];
      float t479 = t[9*frameCount + i] + 1.0;
      float t480 = metal::select(t479, 0.0, 0.0 > 0.0);
      float t481 = t480;
      float t482 = (t481 * 6.1035156e-05);
      float t483 = metal::floor(t482);
      float t484 = t483 * 16384.0;
      float t485 = t480 - t484;
      memory[599948661] = t485;
      float t487 = t485 >= 16384.0;
      if (t487) {
        float t489 = t485 - 16384.0;
        memory[599948661] = t489;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(512), value: global(512)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(478) - handled in variable access */
    float t495 = (t[9*frameCount + id] - metal::floor(t[9*frameCount + id] / 16384.0) * 16384.0);
    float t496 = t495 < 0.0;
    float t497 = t495 + 16384.0;
    float t498 = metal::select(t495, t497, t496 > 0.0);
    float t499 = t498;
    float t500 = metal::floor(t499);
    float t501 = t499 - t500;
    float t502 = t500 + 1.0;
    float t503 = t502 >= 16384.0;
    float t504 = metal::select(t502, 0.0, t503 > 0.0);
    int t505 = (int)t500;
    float t506 = memory[8889 + t505];
    int t507 = (int)t504;
    float t508 = memory[8889 + t507];
    float t509 = 1.0 - t501;
    float t510 = t506 * t509;
    float t511 = t508 * t501;
    t[10*frameCount + id] = t510 + t511;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(513), value: global(513)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[11*frameCount + i] = memory[599948662];
      float t514 = t[11*frameCount + i] + 1.0;
      float t515 = metal::select(t514, 0.0, 0.0 > 0.0);
      float t516 = t515;
      float t517 = (t516 * 0.0078125);
      float t518 = metal::floor(t517);
      float t519 = t518 * 128.0;
      float t520 = t515 - t519;
      memory[599948662] = t520;
      float t522 = t520 >= 128.0;
      if (t522) {
        float t524 = t520 - 128.0;
        memory[599948662] = t524;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(534), value: global(534)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(513) - handled in variable access */
    /* loadGlobal(512) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    int t530 = id;
    int t531 = t530 * 1024;
    int t532 = t530 * 257;
    float t533 = t[11*frameCount + id] == 0.0;
    t[12*frameCount + id] = 0.0;
    if (t533) {
      for (uint _pr536 = 0; _pr536 < 512; _pr536++) {
        float t537 = (float)_pr536;
        float t538 = 6.283185 * t537;
        float t539 = (t538 * 0.0019569471);
        float t540 = metal::cos(t539);
        float t541 = 1.0 - t540;
        float t542 = 0.5 * t541;
        float t543 = (float)t530;
        float t544 = t543 - 511.0;
        float t545 = t544 + t537;
        float t546 = (t545 < 0 || t545 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t545];
        float t547 = (t545 < 0 || t545 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t545];
        int t548 = t531 + _pr536;
        float t549 = t546 * t542;
        memory[4255220 + t548] = t549;
        int t551 = t531 + _pr536;
        int t552 = t551 + 512;
        memory[4255220 + t552] = 0.0;
        int t554 = t531 + _pr536;
        float t555 = t547 * t542;
        memory[21032436 + t554] = t555;
        int t557 = t531 + _pr536;
        int t558 = t557 + 512;
        memory[21032436 + t558] = 0.0;
        memory[25460 + (int)_pr536] = t542;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t562 = 0; t562 < 512; t562++) {
        float t563 = (float)t562;
        float t564 = (t563 - metal::floor(t563 / 2.0) * 2.0);
        float t565 = t564;
        float t566 = (t563 * 0.5);
        float t567 = metal::floor(t566);
        float t568 = t565 * 2.0;
        float t569 = (t567 - metal::floor(t567 / 2.0) * 2.0);
        float t570 = t568 + t569;
        float t571 = (t567 * 0.5);
        float t572 = metal::floor(t571);
        float t573 = t570 * 2.0;
        float t574 = (t572 - metal::floor(t572 / 2.0) * 2.0);
        float t575 = t573 + t574;
        float t576 = (t572 * 0.5);
        float t577 = metal::floor(t576);
        float t578 = t575 * 2.0;
        float t579 = (t577 - metal::floor(t577 / 2.0) * 2.0);
        float t580 = t578 + t579;
        float t581 = (t577 * 0.5);
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
        float t608 = (float)t562;
        float t609 = t608 < t605;
        int t610 = (int)t605;
        int t611 = t531 + t562;
        float t612 = memory[4255220 + t611];
        int t613 = t531 + t562;
        int t614 = t613 + 512;
        float t615 = memory[4255220 + t614];
        int t616 = t531 + t610;
        float t617 = memory[4255220 + t616];
        int t618 = t531 + t610;
        int t619 = t618 + 512;
        float t620 = memory[4255220 + t619];
        float t621 = metal::select(t612, t617, t609 > 0.0);
        float t622 = metal::select(t615, t620, t609 > 0.0);
        float t623 = metal::select(t617, t612, t609 > 0.0);
        float t624 = metal::select(t620, t615, t609 > 0.0);
        int t625 = t531 + t562;
        memory[4255220 + t625] = t621;
        int t627 = t531 + t562;
        int t628 = t627 + 512;
        memory[4255220 + t628] = t622;
        int t630 = t531 + t610;
        memory[4255220 + t630] = t623;
        int t632 = t531 + t610;
        int t633 = t632 + 512;
        memory[4255220 + t633] = t624;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr636 = 0; _pr636 < 256; _pr636++) {
        float t637 = (float)_pr636;
        float t638 = t637;
        float t639 = metal::floor(t638);
        float t640 = t639;
        float t641 = t637 - t640;
        float t642 = t639 * 2.0;
        float t643 = t642 + t641;
        float t644 = t643 + 1.0;
        float t645 = -6.283185 * t641;
        float t646 = (t645 * 0.5);
        float t647 = metal::cos(t646);
        float t648 = metal::sin(t646);
        int t649 = (int)t643;
        int t650 = (int)t644;
        int t651 = t531 + t649;
        float t652 = memory[4255220 + t651];
        int t653 = t531 + t649;
        int t654 = t653 + 512;
        float t655 = memory[4255220 + t654];
        int t656 = t531 + t650;
        float t657 = memory[4255220 + t656];
        int t658 = t531 + t650;
        int t659 = t658 + 512;
        float t660 = memory[4255220 + t659];
        float t661 = t647 * t657;
        float t662 = t648 * t660;
        float t663 = t661 - t662;
        float t664 = t647 * t660;
        float t665 = t648 * t657;
        float t666 = t664 + t665;
        int t667 = t531 + t649;
        float t668 = t652 + t663;
        memory[4255220 + t667] = t668;
        int t670 = t531 + t649;
        int t671 = t670 + 512;
        float t672 = t655 + t666;
        memory[4255220 + t671] = t672;
        int t674 = t531 + t650;
        float t675 = t652 - t663;
        memory[4255220 + t674] = t675;
        int t677 = t531 + t650;
        int t678 = t677 + 512;
        float t679 = t655 - t666;
        memory[4255220 + t678] = t679;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr682 = 0; _pr682 < 256; _pr682++) {
        float t683 = (float)_pr682;
        float t684 = (t683 * 0.5);
        float t685 = metal::floor(t684);
        float t686 = t685 * 2.0;
        float t687 = t683 - t686;
        float t688 = t685 * 4.0;
        float t689 = t688 + t687;
        float t690 = t689 + 2.0;
        float t691 = -6.283185 * t687;
        float t692 = (t691 * 0.25);
        float t693 = metal::cos(t692);
        float t694 = metal::sin(t692);
        int t695 = (int)t689;
        int t696 = (int)t690;
        int t697 = t531 + t695;
        float t698 = memory[4255220 + t697];
        int t699 = t531 + t695;
        int t700 = t699 + 512;
        float t701 = memory[4255220 + t700];
        int t702 = t531 + t696;
        float t703 = memory[4255220 + t702];
        int t704 = t531 + t696;
        int t705 = t704 + 512;
        float t706 = memory[4255220 + t705];
        float t707 = t693 * t703;
        float t708 = t694 * t706;
        float t709 = t707 - t708;
        float t710 = t693 * t706;
        float t711 = t694 * t703;
        float t712 = t710 + t711;
        int t713 = t531 + t695;
        float t714 = t698 + t709;
        memory[4255220 + t713] = t714;
        int t716 = t531 + t695;
        int t717 = t716 + 512;
        float t718 = t701 + t712;
        memory[4255220 + t717] = t718;
        int t720 = t531 + t696;
        float t721 = t698 - t709;
        memory[4255220 + t720] = t721;
        int t723 = t531 + t696;
        int t724 = t723 + 512;
        float t725 = t701 - t712;
        memory[4255220 + t724] = t725;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr728 = 0; _pr728 < 256; _pr728++) {
        float t729 = (float)_pr728;
        float t730 = (t729 * 0.25);
        float t731 = metal::floor(t730);
        float t732 = t731 * 4.0;
        float t733 = t729 - t732;
        float t734 = t731 * 8.0;
        float t735 = t734 + t733;
        float t736 = t735 + 4.0;
        float t737 = -6.283185 * t733;
        float t738 = (t737 * 0.125);
        float t739 = metal::cos(t738);
        float t740 = metal::sin(t738);
        int t741 = (int)t735;
        int t742 = (int)t736;
        int t743 = t531 + t741;
        float t744 = memory[4255220 + t743];
        int t745 = t531 + t741;
        int t746 = t745 + 512;
        float t747 = memory[4255220 + t746];
        int t748 = t531 + t742;
        float t749 = memory[4255220 + t748];
        int t750 = t531 + t742;
        int t751 = t750 + 512;
        float t752 = memory[4255220 + t751];
        float t753 = t739 * t749;
        float t754 = t740 * t752;
        float t755 = t753 - t754;
        float t756 = t739 * t752;
        float t757 = t740 * t749;
        float t758 = t756 + t757;
        int t759 = t531 + t741;
        float t760 = t744 + t755;
        memory[4255220 + t759] = t760;
        int t762 = t531 + t741;
        int t763 = t762 + 512;
        float t764 = t747 + t758;
        memory[4255220 + t763] = t764;
        int t766 = t531 + t742;
        float t767 = t744 - t755;
        memory[4255220 + t766] = t767;
        int t769 = t531 + t742;
        int t770 = t769 + 512;
        float t771 = t747 - t758;
        memory[4255220 + t770] = t771;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr774 = 0; _pr774 < 256; _pr774++) {
        float t775 = (float)_pr774;
        float t776 = (t775 * 0.125);
        float t777 = metal::floor(t776);
        float t778 = t777 * 8.0;
        float t779 = t775 - t778;
        float t780 = t777 * 16.0;
        float t781 = t780 + t779;
        float t782 = t781 + 8.0;
        float t783 = -6.283185 * t779;
        float t784 = (t783 * 0.0625);
        float t785 = metal::cos(t784);
        float t786 = metal::sin(t784);
        int t787 = (int)t781;
        int t788 = (int)t782;
        int t789 = t531 + t787;
        float t790 = memory[4255220 + t789];
        int t791 = t531 + t787;
        int t792 = t791 + 512;
        float t793 = memory[4255220 + t792];
        int t794 = t531 + t788;
        float t795 = memory[4255220 + t794];
        int t796 = t531 + t788;
        int t797 = t796 + 512;
        float t798 = memory[4255220 + t797];
        float t799 = t785 * t795;
        float t800 = t786 * t798;
        float t801 = t799 - t800;
        float t802 = t785 * t798;
        float t803 = t786 * t795;
        float t804 = t802 + t803;
        int t805 = t531 + t787;
        float t806 = t790 + t801;
        memory[4255220 + t805] = t806;
        int t808 = t531 + t787;
        int t809 = t808 + 512;
        float t810 = t793 + t804;
        memory[4255220 + t809] = t810;
        int t812 = t531 + t788;
        float t813 = t790 - t801;
        memory[4255220 + t812] = t813;
        int t815 = t531 + t788;
        int t816 = t815 + 512;
        float t817 = t793 - t804;
        memory[4255220 + t816] = t817;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr820 = 0; _pr820 < 256; _pr820++) {
        float t821 = (float)_pr820;
        float t822 = (t821 * 0.0625);
        float t823 = metal::floor(t822);
        float t824 = t823 * 16.0;
        float t825 = t821 - t824;
        float t826 = t823 * 32.0;
        float t827 = t826 + t825;
        float t828 = t827 + 16.0;
        float t829 = -6.283185 * t825;
        float t830 = (t829 * 0.03125);
        float t831 = metal::cos(t830);
        float t832 = metal::sin(t830);
        int t833 = (int)t827;
        int t834 = (int)t828;
        int t835 = t531 + t833;
        float t836 = memory[4255220 + t835];
        int t837 = t531 + t833;
        int t838 = t837 + 512;
        float t839 = memory[4255220 + t838];
        int t840 = t531 + t834;
        float t841 = memory[4255220 + t840];
        int t842 = t531 + t834;
        int t843 = t842 + 512;
        float t844 = memory[4255220 + t843];
        float t845 = t831 * t841;
        float t846 = t832 * t844;
        float t847 = t845 - t846;
        float t848 = t831 * t844;
        float t849 = t832 * t841;
        float t850 = t848 + t849;
        int t851 = t531 + t833;
        float t852 = t836 + t847;
        memory[4255220 + t851] = t852;
        int t854 = t531 + t833;
        int t855 = t854 + 512;
        float t856 = t839 + t850;
        memory[4255220 + t855] = t856;
        int t858 = t531 + t834;
        float t859 = t836 - t847;
        memory[4255220 + t858] = t859;
        int t861 = t531 + t834;
        int t862 = t861 + 512;
        float t863 = t839 - t850;
        memory[4255220 + t862] = t863;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr866 = 0; _pr866 < 256; _pr866++) {
        float t867 = (float)_pr866;
        float t868 = (t867 * 0.03125);
        float t869 = metal::floor(t868);
        float t870 = t869 * 32.0;
        float t871 = t867 - t870;
        float t872 = t869 * 64.0;
        float t873 = t872 + t871;
        float t874 = t873 + 32.0;
        float t875 = -6.283185 * t871;
        float t876 = (t875 * 0.015625);
        float t877 = metal::cos(t876);
        float t878 = metal::sin(t876);
        int t879 = (int)t873;
        int t880 = (int)t874;
        int t881 = t531 + t879;
        float t882 = memory[4255220 + t881];
        int t883 = t531 + t879;
        int t884 = t883 + 512;
        float t885 = memory[4255220 + t884];
        int t886 = t531 + t880;
        float t887 = memory[4255220 + t886];
        int t888 = t531 + t880;
        int t889 = t888 + 512;
        float t890 = memory[4255220 + t889];
        float t891 = t877 * t887;
        float t892 = t878 * t890;
        float t893 = t891 - t892;
        float t894 = t877 * t890;
        float t895 = t878 * t887;
        float t896 = t894 + t895;
        int t897 = t531 + t879;
        float t898 = t882 + t893;
        memory[4255220 + t897] = t898;
        int t900 = t531 + t879;
        int t901 = t900 + 512;
        float t902 = t885 + t896;
        memory[4255220 + t901] = t902;
        int t904 = t531 + t880;
        float t905 = t882 - t893;
        memory[4255220 + t904] = t905;
        int t907 = t531 + t880;
        int t908 = t907 + 512;
        float t909 = t885 - t896;
        memory[4255220 + t908] = t909;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr912 = 0; _pr912 < 256; _pr912++) {
        float t913 = (float)_pr912;
        float t914 = (t913 * 0.015625);
        float t915 = metal::floor(t914);
        float t916 = t915 * 64.0;
        float t917 = t913 - t916;
        float t918 = t915 * 128.0;
        float t919 = t918 + t917;
        float t920 = t919 + 64.0;
        float t921 = -6.283185 * t917;
        float t922 = (t921 * 0.0078125);
        float t923 = metal::cos(t922);
        float t924 = metal::sin(t922);
        int t925 = (int)t919;
        int t926 = (int)t920;
        int t927 = t531 + t925;
        float t928 = memory[4255220 + t927];
        int t929 = t531 + t925;
        int t930 = t929 + 512;
        float t931 = memory[4255220 + t930];
        int t932 = t531 + t926;
        float t933 = memory[4255220 + t932];
        int t934 = t531 + t926;
        int t935 = t934 + 512;
        float t936 = memory[4255220 + t935];
        float t937 = t923 * t933;
        float t938 = t924 * t936;
        float t939 = t937 - t938;
        float t940 = t923 * t936;
        float t941 = t924 * t933;
        float t942 = t940 + t941;
        int t943 = t531 + t925;
        float t944 = t928 + t939;
        memory[4255220 + t943] = t944;
        int t946 = t531 + t925;
        int t947 = t946 + 512;
        float t948 = t931 + t942;
        memory[4255220 + t947] = t948;
        int t950 = t531 + t926;
        float t951 = t928 - t939;
        memory[4255220 + t950] = t951;
        int t953 = t531 + t926;
        int t954 = t953 + 512;
        float t955 = t931 - t942;
        memory[4255220 + t954] = t955;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr958 = 0; _pr958 < 256; _pr958++) {
        float t959 = (float)_pr958;
        float t960 = (t959 * 0.0078125);
        float t961 = metal::floor(t960);
        float t962 = t961 * 128.0;
        float t963 = t959 - t962;
        float t964 = t961 * 256.0;
        float t965 = t964 + t963;
        float t966 = t965 + 128.0;
        float t967 = -6.283185 * t963;
        float t968 = (t967 * 0.00390625);
        float t969 = metal::cos(t968);
        float t970 = metal::sin(t968);
        int t971 = (int)t965;
        int t972 = (int)t966;
        int t973 = t531 + t971;
        float t974 = memory[4255220 + t973];
        int t975 = t531 + t971;
        int t976 = t975 + 512;
        float t977 = memory[4255220 + t976];
        int t978 = t531 + t972;
        float t979 = memory[4255220 + t978];
        int t980 = t531 + t972;
        int t981 = t980 + 512;
        float t982 = memory[4255220 + t981];
        float t983 = t969 * t979;
        float t984 = t970 * t982;
        float t985 = t983 - t984;
        float t986 = t969 * t982;
        float t987 = t970 * t979;
        float t988 = t986 + t987;
        int t989 = t531 + t971;
        float t990 = t974 + t985;
        memory[4255220 + t989] = t990;
        int t992 = t531 + t971;
        int t993 = t992 + 512;
        float t994 = t977 + t988;
        memory[4255220 + t993] = t994;
        int t996 = t531 + t972;
        float t997 = t974 - t985;
        memory[4255220 + t996] = t997;
        int t999 = t531 + t972;
        int t1000 = t999 + 512;
        float t1001 = t977 - t988;
        memory[4255220 + t1000] = t1001;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1004 = 0; _pr1004 < 256; _pr1004++) {
        float t1005 = (float)_pr1004;
        float t1006 = (t1005 * 0.00390625);
        float t1007 = metal::floor(t1006);
        float t1008 = t1007 * 256.0;
        float t1009 = t1005 - t1008;
        float t1010 = t1007 * 512.0;
        float t1011 = t1010 + t1009;
        float t1012 = t1011 + 256.0;
        float t1013 = -6.283185 * t1009;
        float t1014 = (t1013 * 0.001953125);
        float t1015 = metal::cos(t1014);
        float t1016 = metal::sin(t1014);
        int t1017 = (int)t1011;
        int t1018 = (int)t1012;
        int t1019 = t531 + t1017;
        float t1020 = memory[4255220 + t1019];
        int t1021 = t531 + t1017;
        int t1022 = t1021 + 512;
        float t1023 = memory[4255220 + t1022];
        int t1024 = t531 + t1018;
        float t1025 = memory[4255220 + t1024];
        int t1026 = t531 + t1018;
        int t1027 = t1026 + 512;
        float t1028 = memory[4255220 + t1027];
        float t1029 = t1015 * t1025;
        float t1030 = t1016 * t1028;
        float t1031 = t1029 - t1030;
        float t1032 = t1015 * t1028;
        float t1033 = t1016 * t1025;
        float t1034 = t1032 + t1033;
        int t1035 = t531 + t1017;
        float t1036 = t1020 + t1031;
        memory[4255220 + t1035] = t1036;
        int t1038 = t531 + t1017;
        int t1039 = t1038 + 512;
        float t1040 = t1023 + t1034;
        memory[4255220 + t1039] = t1040;
        int t1042 = t531 + t1018;
        float t1043 = t1020 - t1031;
        memory[4255220 + t1042] = t1043;
        int t1045 = t531 + t1018;
        int t1046 = t1045 + 512;
        float t1047 = t1023 - t1034;
        memory[4255220 + t1046] = t1047;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1050 = 0; t1050 < 512; t1050++) {
        float t1051 = (float)t1050;
        float t1052 = (t1051 - metal::floor(t1051 / 2.0) * 2.0);
        float t1053 = t1052;
        float t1054 = (t1051 * 0.5);
        float t1055 = metal::floor(t1054);
        float t1056 = t1053 * 2.0;
        float t1057 = (t1055 - metal::floor(t1055 / 2.0) * 2.0);
        float t1058 = t1056 + t1057;
        float t1059 = (t1055 * 0.5);
        float t1060 = metal::floor(t1059);
        float t1061 = t1058 * 2.0;
        float t1062 = (t1060 - metal::floor(t1060 / 2.0) * 2.0);
        float t1063 = t1061 + t1062;
        float t1064 = (t1060 * 0.5);
        float t1065 = metal::floor(t1064);
        float t1066 = t1063 * 2.0;
        float t1067 = (t1065 - metal::floor(t1065 / 2.0) * 2.0);
        float t1068 = t1066 + t1067;
        float t1069 = (t1065 * 0.5);
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
        float t1096 = (float)t1050;
        float t1097 = t1096 < t1093;
        int t1098 = (int)t1093;
        int t1099 = t531 + t1050;
        float t1100 = memory[21032436 + t1099];
        int t1101 = t531 + t1050;
        int t1102 = t1101 + 512;
        float t1103 = memory[21032436 + t1102];
        int t1104 = t531 + t1098;
        float t1105 = memory[21032436 + t1104];
        int t1106 = t531 + t1098;
        int t1107 = t1106 + 512;
        float t1108 = memory[21032436 + t1107];
        float t1109 = metal::select(t1100, t1105, t1097 > 0.0);
        float t1110 = metal::select(t1103, t1108, t1097 > 0.0);
        float t1111 = metal::select(t1105, t1100, t1097 > 0.0);
        float t1112 = metal::select(t1108, t1103, t1097 > 0.0);
        int t1113 = t531 + t1050;
        memory[21032436 + t1113] = t1109;
        int t1115 = t531 + t1050;
        int t1116 = t1115 + 512;
        memory[21032436 + t1116] = t1110;
        int t1118 = t531 + t1098;
        memory[21032436 + t1118] = t1111;
        int t1120 = t531 + t1098;
        int t1121 = t1120 + 512;
        memory[21032436 + t1121] = t1112;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1124 = 0; _pr1124 < 256; _pr1124++) {
        float t1125 = (float)_pr1124;
        float t1126 = t1125;
        float t1127 = metal::floor(t1126);
        float t1128 = t1127;
        float t1129 = t1125 - t1128;
        float t1130 = t1127 * 2.0;
        float t1131 = t1130 + t1129;
        float t1132 = t1131 + 1.0;
        float t1133 = -6.283185 * t1129;
        float t1134 = (t1133 * 0.5);
        float t1135 = metal::cos(t1134);
        float t1136 = metal::sin(t1134);
        int t1137 = (int)t1131;
        int t1138 = (int)t1132;
        int t1139 = t531 + t1137;
        float t1140 = memory[21032436 + t1139];
        int t1141 = t531 + t1137;
        int t1142 = t1141 + 512;
        float t1143 = memory[21032436 + t1142];
        int t1144 = t531 + t1138;
        float t1145 = memory[21032436 + t1144];
        int t1146 = t531 + t1138;
        int t1147 = t1146 + 512;
        float t1148 = memory[21032436 + t1147];
        float t1149 = t1135 * t1145;
        float t1150 = t1136 * t1148;
        float t1151 = t1149 - t1150;
        float t1152 = t1135 * t1148;
        float t1153 = t1136 * t1145;
        float t1154 = t1152 + t1153;
        int t1155 = t531 + t1137;
        float t1156 = t1140 + t1151;
        memory[21032436 + t1155] = t1156;
        int t1158 = t531 + t1137;
        int t1159 = t1158 + 512;
        float t1160 = t1143 + t1154;
        memory[21032436 + t1159] = t1160;
        int t1162 = t531 + t1138;
        float t1163 = t1140 - t1151;
        memory[21032436 + t1162] = t1163;
        int t1165 = t531 + t1138;
        int t1166 = t1165 + 512;
        float t1167 = t1143 - t1154;
        memory[21032436 + t1166] = t1167;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1170 = 0; _pr1170 < 256; _pr1170++) {
        float t1171 = (float)_pr1170;
        float t1172 = (t1171 * 0.5);
        float t1173 = metal::floor(t1172);
        float t1174 = t1173 * 2.0;
        float t1175 = t1171 - t1174;
        float t1176 = t1173 * 4.0;
        float t1177 = t1176 + t1175;
        float t1178 = t1177 + 2.0;
        float t1179 = -6.283185 * t1175;
        float t1180 = (t1179 * 0.25);
        float t1181 = metal::cos(t1180);
        float t1182 = metal::sin(t1180);
        int t1183 = (int)t1177;
        int t1184 = (int)t1178;
        int t1185 = t531 + t1183;
        float t1186 = memory[21032436 + t1185];
        int t1187 = t531 + t1183;
        int t1188 = t1187 + 512;
        float t1189 = memory[21032436 + t1188];
        int t1190 = t531 + t1184;
        float t1191 = memory[21032436 + t1190];
        int t1192 = t531 + t1184;
        int t1193 = t1192 + 512;
        float t1194 = memory[21032436 + t1193];
        float t1195 = t1181 * t1191;
        float t1196 = t1182 * t1194;
        float t1197 = t1195 - t1196;
        float t1198 = t1181 * t1194;
        float t1199 = t1182 * t1191;
        float t1200 = t1198 + t1199;
        int t1201 = t531 + t1183;
        float t1202 = t1186 + t1197;
        memory[21032436 + t1201] = t1202;
        int t1204 = t531 + t1183;
        int t1205 = t1204 + 512;
        float t1206 = t1189 + t1200;
        memory[21032436 + t1205] = t1206;
        int t1208 = t531 + t1184;
        float t1209 = t1186 - t1197;
        memory[21032436 + t1208] = t1209;
        int t1211 = t531 + t1184;
        int t1212 = t1211 + 512;
        float t1213 = t1189 - t1200;
        memory[21032436 + t1212] = t1213;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1216 = 0; _pr1216 < 256; _pr1216++) {
        float t1217 = (float)_pr1216;
        float t1218 = (t1217 * 0.25);
        float t1219 = metal::floor(t1218);
        float t1220 = t1219 * 4.0;
        float t1221 = t1217 - t1220;
        float t1222 = t1219 * 8.0;
        float t1223 = t1222 + t1221;
        float t1224 = t1223 + 4.0;
        float t1225 = -6.283185 * t1221;
        float t1226 = (t1225 * 0.125);
        float t1227 = metal::cos(t1226);
        float t1228 = metal::sin(t1226);
        int t1229 = (int)t1223;
        int t1230 = (int)t1224;
        int t1231 = t531 + t1229;
        float t1232 = memory[21032436 + t1231];
        int t1233 = t531 + t1229;
        int t1234 = t1233 + 512;
        float t1235 = memory[21032436 + t1234];
        int t1236 = t531 + t1230;
        float t1237 = memory[21032436 + t1236];
        int t1238 = t531 + t1230;
        int t1239 = t1238 + 512;
        float t1240 = memory[21032436 + t1239];
        float t1241 = t1227 * t1237;
        float t1242 = t1228 * t1240;
        float t1243 = t1241 - t1242;
        float t1244 = t1227 * t1240;
        float t1245 = t1228 * t1237;
        float t1246 = t1244 + t1245;
        int t1247 = t531 + t1229;
        float t1248 = t1232 + t1243;
        memory[21032436 + t1247] = t1248;
        int t1250 = t531 + t1229;
        int t1251 = t1250 + 512;
        float t1252 = t1235 + t1246;
        memory[21032436 + t1251] = t1252;
        int t1254 = t531 + t1230;
        float t1255 = t1232 - t1243;
        memory[21032436 + t1254] = t1255;
        int t1257 = t531 + t1230;
        int t1258 = t1257 + 512;
        float t1259 = t1235 - t1246;
        memory[21032436 + t1258] = t1259;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1262 = 0; _pr1262 < 256; _pr1262++) {
        float t1263 = (float)_pr1262;
        float t1264 = (t1263 * 0.125);
        float t1265 = metal::floor(t1264);
        float t1266 = t1265 * 8.0;
        float t1267 = t1263 - t1266;
        float t1268 = t1265 * 16.0;
        float t1269 = t1268 + t1267;
        float t1270 = t1269 + 8.0;
        float t1271 = -6.283185 * t1267;
        float t1272 = (t1271 * 0.0625);
        float t1273 = metal::cos(t1272);
        float t1274 = metal::sin(t1272);
        int t1275 = (int)t1269;
        int t1276 = (int)t1270;
        int t1277 = t531 + t1275;
        float t1278 = memory[21032436 + t1277];
        int t1279 = t531 + t1275;
        int t1280 = t1279 + 512;
        float t1281 = memory[21032436 + t1280];
        int t1282 = t531 + t1276;
        float t1283 = memory[21032436 + t1282];
        int t1284 = t531 + t1276;
        int t1285 = t1284 + 512;
        float t1286 = memory[21032436 + t1285];
        float t1287 = t1273 * t1283;
        float t1288 = t1274 * t1286;
        float t1289 = t1287 - t1288;
        float t1290 = t1273 * t1286;
        float t1291 = t1274 * t1283;
        float t1292 = t1290 + t1291;
        int t1293 = t531 + t1275;
        float t1294 = t1278 + t1289;
        memory[21032436 + t1293] = t1294;
        int t1296 = t531 + t1275;
        int t1297 = t1296 + 512;
        float t1298 = t1281 + t1292;
        memory[21032436 + t1297] = t1298;
        int t1300 = t531 + t1276;
        float t1301 = t1278 - t1289;
        memory[21032436 + t1300] = t1301;
        int t1303 = t531 + t1276;
        int t1304 = t1303 + 512;
        float t1305 = t1281 - t1292;
        memory[21032436 + t1304] = t1305;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1308 = 0; _pr1308 < 256; _pr1308++) {
        float t1309 = (float)_pr1308;
        float t1310 = (t1309 * 0.0625);
        float t1311 = metal::floor(t1310);
        float t1312 = t1311 * 16.0;
        float t1313 = t1309 - t1312;
        float t1314 = t1311 * 32.0;
        float t1315 = t1314 + t1313;
        float t1316 = t1315 + 16.0;
        float t1317 = -6.283185 * t1313;
        float t1318 = (t1317 * 0.03125);
        float t1319 = metal::cos(t1318);
        float t1320 = metal::sin(t1318);
        int t1321 = (int)t1315;
        int t1322 = (int)t1316;
        int t1323 = t531 + t1321;
        float t1324 = memory[21032436 + t1323];
        int t1325 = t531 + t1321;
        int t1326 = t1325 + 512;
        float t1327 = memory[21032436 + t1326];
        int t1328 = t531 + t1322;
        float t1329 = memory[21032436 + t1328];
        int t1330 = t531 + t1322;
        int t1331 = t1330 + 512;
        float t1332 = memory[21032436 + t1331];
        float t1333 = t1319 * t1329;
        float t1334 = t1320 * t1332;
        float t1335 = t1333 - t1334;
        float t1336 = t1319 * t1332;
        float t1337 = t1320 * t1329;
        float t1338 = t1336 + t1337;
        int t1339 = t531 + t1321;
        float t1340 = t1324 + t1335;
        memory[21032436 + t1339] = t1340;
        int t1342 = t531 + t1321;
        int t1343 = t1342 + 512;
        float t1344 = t1327 + t1338;
        memory[21032436 + t1343] = t1344;
        int t1346 = t531 + t1322;
        float t1347 = t1324 - t1335;
        memory[21032436 + t1346] = t1347;
        int t1349 = t531 + t1322;
        int t1350 = t1349 + 512;
        float t1351 = t1327 - t1338;
        memory[21032436 + t1350] = t1351;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1354 = 0; _pr1354 < 256; _pr1354++) {
        float t1355 = (float)_pr1354;
        float t1356 = (t1355 * 0.03125);
        float t1357 = metal::floor(t1356);
        float t1358 = t1357 * 32.0;
        float t1359 = t1355 - t1358;
        float t1360 = t1357 * 64.0;
        float t1361 = t1360 + t1359;
        float t1362 = t1361 + 32.0;
        float t1363 = -6.283185 * t1359;
        float t1364 = (t1363 * 0.015625);
        float t1365 = metal::cos(t1364);
        float t1366 = metal::sin(t1364);
        int t1367 = (int)t1361;
        int t1368 = (int)t1362;
        int t1369 = t531 + t1367;
        float t1370 = memory[21032436 + t1369];
        int t1371 = t531 + t1367;
        int t1372 = t1371 + 512;
        float t1373 = memory[21032436 + t1372];
        int t1374 = t531 + t1368;
        float t1375 = memory[21032436 + t1374];
        int t1376 = t531 + t1368;
        int t1377 = t1376 + 512;
        float t1378 = memory[21032436 + t1377];
        float t1379 = t1365 * t1375;
        float t1380 = t1366 * t1378;
        float t1381 = t1379 - t1380;
        float t1382 = t1365 * t1378;
        float t1383 = t1366 * t1375;
        float t1384 = t1382 + t1383;
        int t1385 = t531 + t1367;
        float t1386 = t1370 + t1381;
        memory[21032436 + t1385] = t1386;
        int t1388 = t531 + t1367;
        int t1389 = t1388 + 512;
        float t1390 = t1373 + t1384;
        memory[21032436 + t1389] = t1390;
        int t1392 = t531 + t1368;
        float t1393 = t1370 - t1381;
        memory[21032436 + t1392] = t1393;
        int t1395 = t531 + t1368;
        int t1396 = t1395 + 512;
        float t1397 = t1373 - t1384;
        memory[21032436 + t1396] = t1397;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1400 = 0; _pr1400 < 256; _pr1400++) {
        float t1401 = (float)_pr1400;
        float t1402 = (t1401 * 0.015625);
        float t1403 = metal::floor(t1402);
        float t1404 = t1403 * 64.0;
        float t1405 = t1401 - t1404;
        float t1406 = t1403 * 128.0;
        float t1407 = t1406 + t1405;
        float t1408 = t1407 + 64.0;
        float t1409 = -6.283185 * t1405;
        float t1410 = (t1409 * 0.0078125);
        float t1411 = metal::cos(t1410);
        float t1412 = metal::sin(t1410);
        int t1413 = (int)t1407;
        int t1414 = (int)t1408;
        int t1415 = t531 + t1413;
        float t1416 = memory[21032436 + t1415];
        int t1417 = t531 + t1413;
        int t1418 = t1417 + 512;
        float t1419 = memory[21032436 + t1418];
        int t1420 = t531 + t1414;
        float t1421 = memory[21032436 + t1420];
        int t1422 = t531 + t1414;
        int t1423 = t1422 + 512;
        float t1424 = memory[21032436 + t1423];
        float t1425 = t1411 * t1421;
        float t1426 = t1412 * t1424;
        float t1427 = t1425 - t1426;
        float t1428 = t1411 * t1424;
        float t1429 = t1412 * t1421;
        float t1430 = t1428 + t1429;
        int t1431 = t531 + t1413;
        float t1432 = t1416 + t1427;
        memory[21032436 + t1431] = t1432;
        int t1434 = t531 + t1413;
        int t1435 = t1434 + 512;
        float t1436 = t1419 + t1430;
        memory[21032436 + t1435] = t1436;
        int t1438 = t531 + t1414;
        float t1439 = t1416 - t1427;
        memory[21032436 + t1438] = t1439;
        int t1441 = t531 + t1414;
        int t1442 = t1441 + 512;
        float t1443 = t1419 - t1430;
        memory[21032436 + t1442] = t1443;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1446 = 0; _pr1446 < 256; _pr1446++) {
        float t1447 = (float)_pr1446;
        float t1448 = (t1447 * 0.0078125);
        float t1449 = metal::floor(t1448);
        float t1450 = t1449 * 128.0;
        float t1451 = t1447 - t1450;
        float t1452 = t1449 * 256.0;
        float t1453 = t1452 + t1451;
        float t1454 = t1453 + 128.0;
        float t1455 = -6.283185 * t1451;
        float t1456 = (t1455 * 0.00390625);
        float t1457 = metal::cos(t1456);
        float t1458 = metal::sin(t1456);
        int t1459 = (int)t1453;
        int t1460 = (int)t1454;
        int t1461 = t531 + t1459;
        float t1462 = memory[21032436 + t1461];
        int t1463 = t531 + t1459;
        int t1464 = t1463 + 512;
        float t1465 = memory[21032436 + t1464];
        int t1466 = t531 + t1460;
        float t1467 = memory[21032436 + t1466];
        int t1468 = t531 + t1460;
        int t1469 = t1468 + 512;
        float t1470 = memory[21032436 + t1469];
        float t1471 = t1457 * t1467;
        float t1472 = t1458 * t1470;
        float t1473 = t1471 - t1472;
        float t1474 = t1457 * t1470;
        float t1475 = t1458 * t1467;
        float t1476 = t1474 + t1475;
        int t1477 = t531 + t1459;
        float t1478 = t1462 + t1473;
        memory[21032436 + t1477] = t1478;
        int t1480 = t531 + t1459;
        int t1481 = t1480 + 512;
        float t1482 = t1465 + t1476;
        memory[21032436 + t1481] = t1482;
        int t1484 = t531 + t1460;
        float t1485 = t1462 - t1473;
        memory[21032436 + t1484] = t1485;
        int t1487 = t531 + t1460;
        int t1488 = t1487 + 512;
        float t1489 = t1465 - t1476;
        memory[21032436 + t1488] = t1489;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1492 = 0; _pr1492 < 256; _pr1492++) {
        float t1493 = (float)_pr1492;
        float t1494 = (t1493 * 0.00390625);
        float t1495 = metal::floor(t1494);
        float t1496 = t1495 * 256.0;
        float t1497 = t1493 - t1496;
        float t1498 = t1495 * 512.0;
        float t1499 = t1498 + t1497;
        float t1500 = t1499 + 256.0;
        float t1501 = -6.283185 * t1497;
        float t1502 = (t1501 * 0.001953125);
        float t1503 = metal::cos(t1502);
        float t1504 = metal::sin(t1502);
        int t1505 = (int)t1499;
        int t1506 = (int)t1500;
        int t1507 = t531 + t1505;
        float t1508 = memory[21032436 + t1507];
        int t1509 = t531 + t1505;
        int t1510 = t1509 + 512;
        float t1511 = memory[21032436 + t1510];
        int t1512 = t531 + t1506;
        float t1513 = memory[21032436 + t1512];
        int t1514 = t531 + t1506;
        int t1515 = t1514 + 512;
        float t1516 = memory[21032436 + t1515];
        float t1517 = t1503 * t1513;
        float t1518 = t1504 * t1516;
        float t1519 = t1517 - t1518;
        float t1520 = t1503 * t1516;
        float t1521 = t1504 * t1513;
        float t1522 = t1520 + t1521;
        int t1523 = t531 + t1505;
        float t1524 = t1508 + t1519;
        memory[21032436 + t1523] = t1524;
        int t1526 = t531 + t1505;
        int t1527 = t1526 + 512;
        float t1528 = t1511 + t1522;
        memory[21032436 + t1527] = t1528;
        int t1530 = t531 + t1506;
        float t1531 = t1508 - t1519;
        memory[21032436 + t1530] = t1531;
        int t1533 = t531 + t1506;
        int t1534 = t1533 + 512;
        float t1535 = t1511 - t1522;
        memory[21032436 + t1534] = t1535;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1538 = 0; _pr1538 < 257; _pr1538++) {
        int t1539 = t531 + _pr1538;
        float t1540 = memory[4255220 + t1539];
        int t1541 = t531 + _pr1538;
        int t1542 = t1541 + 512;
        float t1543 = memory[4255220 + t1542];
        float t1544 = t1540 * t1540;
        float t1545 = t1543 * t1543;
        float t1546 = t1544 + t1545;
        float t1547 = metal::sqrt(t1546);
        int t1548 = t532 + _pr1538;
        memory[37809652 + t1548] = t1547;
        int t1550 = t531 + _pr1538;
        float t1551 = memory[21032436 + t1550];
        int t1552 = t531 + _pr1538;
        int t1553 = t1552 + 512;
        float t1554 = memory[21032436 + t1553];
        float t1555 = t1551 * t1551;
        float t1556 = t1554 * t1554;
        float t1557 = t1555 + t1556;
        float t1558 = metal::sqrt(t1557);
        int t1559 = t532 + _pr1538;
        memory[42020340 + t1559] = t1558;
        float t1561 = t1547 - t1558;
        int t1562 = t532 + _pr1538;
        float t1563 = t1561 * t1561;
        memory[46231028 + t1562] = t1563;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1566 = 0; t1566 < 257; t1566++) {
        int t1567 = t532 + t1566;
        float t1568 = memory[46231028 + t1567];
        float t1569 = t[12*frameCount + id] + t1568;
        t[12*frameCount + id] = t1569;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1573), value: global(1573)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(534) - handled in variable access */
    float t1572 = (t[12*frameCount + id] * 6.1035156e-05);
    t[13*frameCount + id] = t1572;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1574), value: global(1574)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[14*frameCount + i] = memory[599948663];
      float t1575 = t[14*frameCount + i] + 1.0;
      float t1576 = metal::select(t1575, 0.0, 0.0 > 0.0);
      float t1577 = t1576;
      float t1578 = (t1577 * 0.00390625);
      float t1579 = metal::floor(t1578);
      float t1580 = t1579 * 256.0;
      float t1581 = t1576 - t1580;
      memory[599948663] = t1581;
      float t1583 = t1581 >= 256.0;
      if (t1583) {
        float t1585 = t1581 - 256.0;
        memory[599948663] = t1585;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1595), value: global(1595)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1574) - handled in variable access */
    /* loadGlobal(512) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    int t1591 = id;
    int t1592 = t1591 * 2048;
    int t1593 = t1591 * 513;
    float t1594 = t[14*frameCount + id] == 0.0;
    t[15*frameCount + id] = 0.0;
    if (t1594) {
      for (uint _pr1597 = 0; _pr1597 < 1024; _pr1597++) {
        float t1598 = (float)_pr1597;
        float t1599 = 6.283185 * t1598;
        float t1600 = (t1599 * 0.0009775171);
        float t1601 = metal::cos(t1600);
        float t1602 = 1.0 - t1601;
        float t1603 = 0.5 * t1602;
        float t1604 = (float)t1591;
        float t1605 = t1604 - 1023.0;
        float t1606 = t1605 + t1598;
        float t1607 = (t1606 < 0 || t1606 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t1606];
        float t1608 = (t1606 < 0 || t1606 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t1606];
        int t1609 = t1592 + _pr1597;
        float t1610 = t1607 * t1603;
        memory[50441716 + t1609] = t1610;
        int t1612 = t1592 + _pr1597;
        int t1613 = t1612 + 1024;
        memory[50441716 + t1613] = 0.0;
        int t1615 = t1592 + _pr1597;
        float t1616 = t1608 * t1603;
        memory[83996148 + t1615] = t1616;
        int t1618 = t1592 + _pr1597;
        int t1619 = t1618 + 1024;
        memory[83996148 + t1619] = 0.0;
        memory[48884 + (int)_pr1597] = t1603;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1623 = 0; t1623 < 1024; t1623++) {
        float t1624 = (float)t1623;
        float t1625 = (t1624 - metal::floor(t1624 / 2.0) * 2.0);
        float t1626 = t1625;
        float t1627 = (t1624 * 0.5);
        float t1628 = metal::floor(t1627);
        float t1629 = t1626 * 2.0;
        float t1630 = (t1628 - metal::floor(t1628 / 2.0) * 2.0);
        float t1631 = t1629 + t1630;
        float t1632 = (t1628 * 0.5);
        float t1633 = metal::floor(t1632);
        float t1634 = t1631 * 2.0;
        float t1635 = (t1633 - metal::floor(t1633 / 2.0) * 2.0);
        float t1636 = t1634 + t1635;
        float t1637 = (t1633 * 0.5);
        float t1638 = metal::floor(t1637);
        float t1639 = t1636 * 2.0;
        float t1640 = (t1638 - metal::floor(t1638 / 2.0) * 2.0);
        float t1641 = t1639 + t1640;
        float t1642 = (t1638 * 0.5);
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
        float t1674 = (float)t1623;
        float t1675 = t1674 < t1671;
        int t1676 = (int)t1671;
        int t1677 = t1592 + t1623;
        float t1678 = memory[50441716 + t1677];
        int t1679 = t1592 + t1623;
        int t1680 = t1679 + 1024;
        float t1681 = memory[50441716 + t1680];
        int t1682 = t1592 + t1676;
        float t1683 = memory[50441716 + t1682];
        int t1684 = t1592 + t1676;
        int t1685 = t1684 + 1024;
        float t1686 = memory[50441716 + t1685];
        float t1687 = metal::select(t1678, t1683, t1675 > 0.0);
        float t1688 = metal::select(t1681, t1686, t1675 > 0.0);
        float t1689 = metal::select(t1683, t1678, t1675 > 0.0);
        float t1690 = metal::select(t1686, t1681, t1675 > 0.0);
        int t1691 = t1592 + t1623;
        memory[50441716 + t1691] = t1687;
        int t1693 = t1592 + t1623;
        int t1694 = t1693 + 1024;
        memory[50441716 + t1694] = t1688;
        int t1696 = t1592 + t1676;
        memory[50441716 + t1696] = t1689;
        int t1698 = t1592 + t1676;
        int t1699 = t1698 + 1024;
        memory[50441716 + t1699] = t1690;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1702 = 0; _pr1702 < 512; _pr1702++) {
        float t1703 = (float)_pr1702;
        float t1704 = t1703;
        float t1705 = metal::floor(t1704);
        float t1706 = t1705;
        float t1707 = t1703 - t1706;
        float t1708 = t1705 * 2.0;
        float t1709 = t1708 + t1707;
        float t1710 = t1709 + 1.0;
        float t1711 = -6.283185 * t1707;
        float t1712 = (t1711 * 0.5);
        float t1713 = metal::cos(t1712);
        float t1714 = metal::sin(t1712);
        int t1715 = (int)t1709;
        int t1716 = (int)t1710;
        int t1717 = t1592 + t1715;
        float t1718 = memory[50441716 + t1717];
        int t1719 = t1592 + t1715;
        int t1720 = t1719 + 1024;
        float t1721 = memory[50441716 + t1720];
        int t1722 = t1592 + t1716;
        float t1723 = memory[50441716 + t1722];
        int t1724 = t1592 + t1716;
        int t1725 = t1724 + 1024;
        float t1726 = memory[50441716 + t1725];
        float t1727 = t1713 * t1723;
        float t1728 = t1714 * t1726;
        float t1729 = t1727 - t1728;
        float t1730 = t1713 * t1726;
        float t1731 = t1714 * t1723;
        float t1732 = t1730 + t1731;
        int t1733 = t1592 + t1715;
        float t1734 = t1718 + t1729;
        memory[50441716 + t1733] = t1734;
        int t1736 = t1592 + t1715;
        int t1737 = t1736 + 1024;
        float t1738 = t1721 + t1732;
        memory[50441716 + t1737] = t1738;
        int t1740 = t1592 + t1716;
        float t1741 = t1718 - t1729;
        memory[50441716 + t1740] = t1741;
        int t1743 = t1592 + t1716;
        int t1744 = t1743 + 1024;
        float t1745 = t1721 - t1732;
        memory[50441716 + t1744] = t1745;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1748 = 0; _pr1748 < 512; _pr1748++) {
        float t1749 = (float)_pr1748;
        float t1750 = (t1749 * 0.5);
        float t1751 = metal::floor(t1750);
        float t1752 = t1751 * 2.0;
        float t1753 = t1749 - t1752;
        float t1754 = t1751 * 4.0;
        float t1755 = t1754 + t1753;
        float t1756 = t1755 + 2.0;
        float t1757 = -6.283185 * t1753;
        float t1758 = (t1757 * 0.25);
        float t1759 = metal::cos(t1758);
        float t1760 = metal::sin(t1758);
        int t1761 = (int)t1755;
        int t1762 = (int)t1756;
        int t1763 = t1592 + t1761;
        float t1764 = memory[50441716 + t1763];
        int t1765 = t1592 + t1761;
        int t1766 = t1765 + 1024;
        float t1767 = memory[50441716 + t1766];
        int t1768 = t1592 + t1762;
        float t1769 = memory[50441716 + t1768];
        int t1770 = t1592 + t1762;
        int t1771 = t1770 + 1024;
        float t1772 = memory[50441716 + t1771];
        float t1773 = t1759 * t1769;
        float t1774 = t1760 * t1772;
        float t1775 = t1773 - t1774;
        float t1776 = t1759 * t1772;
        float t1777 = t1760 * t1769;
        float t1778 = t1776 + t1777;
        int t1779 = t1592 + t1761;
        float t1780 = t1764 + t1775;
        memory[50441716 + t1779] = t1780;
        int t1782 = t1592 + t1761;
        int t1783 = t1782 + 1024;
        float t1784 = t1767 + t1778;
        memory[50441716 + t1783] = t1784;
        int t1786 = t1592 + t1762;
        float t1787 = t1764 - t1775;
        memory[50441716 + t1786] = t1787;
        int t1789 = t1592 + t1762;
        int t1790 = t1789 + 1024;
        float t1791 = t1767 - t1778;
        memory[50441716 + t1790] = t1791;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1794 = 0; _pr1794 < 512; _pr1794++) {
        float t1795 = (float)_pr1794;
        float t1796 = (t1795 * 0.25);
        float t1797 = metal::floor(t1796);
        float t1798 = t1797 * 4.0;
        float t1799 = t1795 - t1798;
        float t1800 = t1797 * 8.0;
        float t1801 = t1800 + t1799;
        float t1802 = t1801 + 4.0;
        float t1803 = -6.283185 * t1799;
        float t1804 = (t1803 * 0.125);
        float t1805 = metal::cos(t1804);
        float t1806 = metal::sin(t1804);
        int t1807 = (int)t1801;
        int t1808 = (int)t1802;
        int t1809 = t1592 + t1807;
        float t1810 = memory[50441716 + t1809];
        int t1811 = t1592 + t1807;
        int t1812 = t1811 + 1024;
        float t1813 = memory[50441716 + t1812];
        int t1814 = t1592 + t1808;
        float t1815 = memory[50441716 + t1814];
        int t1816 = t1592 + t1808;
        int t1817 = t1816 + 1024;
        float t1818 = memory[50441716 + t1817];
        float t1819 = t1805 * t1815;
        float t1820 = t1806 * t1818;
        float t1821 = t1819 - t1820;
        float t1822 = t1805 * t1818;
        float t1823 = t1806 * t1815;
        float t1824 = t1822 + t1823;
        int t1825 = t1592 + t1807;
        float t1826 = t1810 + t1821;
        memory[50441716 + t1825] = t1826;
        int t1828 = t1592 + t1807;
        int t1829 = t1828 + 1024;
        float t1830 = t1813 + t1824;
        memory[50441716 + t1829] = t1830;
        int t1832 = t1592 + t1808;
        float t1833 = t1810 - t1821;
        memory[50441716 + t1832] = t1833;
        int t1835 = t1592 + t1808;
        int t1836 = t1835 + 1024;
        float t1837 = t1813 - t1824;
        memory[50441716 + t1836] = t1837;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1840 = 0; _pr1840 < 512; _pr1840++) {
        float t1841 = (float)_pr1840;
        float t1842 = (t1841 * 0.125);
        float t1843 = metal::floor(t1842);
        float t1844 = t1843 * 8.0;
        float t1845 = t1841 - t1844;
        float t1846 = t1843 * 16.0;
        float t1847 = t1846 + t1845;
        float t1848 = t1847 + 8.0;
        float t1849 = -6.283185 * t1845;
        float t1850 = (t1849 * 0.0625);
        float t1851 = metal::cos(t1850);
        float t1852 = metal::sin(t1850);
        int t1853 = (int)t1847;
        int t1854 = (int)t1848;
        int t1855 = t1592 + t1853;
        float t1856 = memory[50441716 + t1855];
        int t1857 = t1592 + t1853;
        int t1858 = t1857 + 1024;
        float t1859 = memory[50441716 + t1858];
        int t1860 = t1592 + t1854;
        float t1861 = memory[50441716 + t1860];
        int t1862 = t1592 + t1854;
        int t1863 = t1862 + 1024;
        float t1864 = memory[50441716 + t1863];
        float t1865 = t1851 * t1861;
        float t1866 = t1852 * t1864;
        float t1867 = t1865 - t1866;
        float t1868 = t1851 * t1864;
        float t1869 = t1852 * t1861;
        float t1870 = t1868 + t1869;
        int t1871 = t1592 + t1853;
        float t1872 = t1856 + t1867;
        memory[50441716 + t1871] = t1872;
        int t1874 = t1592 + t1853;
        int t1875 = t1874 + 1024;
        float t1876 = t1859 + t1870;
        memory[50441716 + t1875] = t1876;
        int t1878 = t1592 + t1854;
        float t1879 = t1856 - t1867;
        memory[50441716 + t1878] = t1879;
        int t1881 = t1592 + t1854;
        int t1882 = t1881 + 1024;
        float t1883 = t1859 - t1870;
        memory[50441716 + t1882] = t1883;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1886 = 0; _pr1886 < 512; _pr1886++) {
        float t1887 = (float)_pr1886;
        float t1888 = (t1887 * 0.0625);
        float t1889 = metal::floor(t1888);
        float t1890 = t1889 * 16.0;
        float t1891 = t1887 - t1890;
        float t1892 = t1889 * 32.0;
        float t1893 = t1892 + t1891;
        float t1894 = t1893 + 16.0;
        float t1895 = -6.283185 * t1891;
        float t1896 = (t1895 * 0.03125);
        float t1897 = metal::cos(t1896);
        float t1898 = metal::sin(t1896);
        int t1899 = (int)t1893;
        int t1900 = (int)t1894;
        int t1901 = t1592 + t1899;
        float t1902 = memory[50441716 + t1901];
        int t1903 = t1592 + t1899;
        int t1904 = t1903 + 1024;
        float t1905 = memory[50441716 + t1904];
        int t1906 = t1592 + t1900;
        float t1907 = memory[50441716 + t1906];
        int t1908 = t1592 + t1900;
        int t1909 = t1908 + 1024;
        float t1910 = memory[50441716 + t1909];
        float t1911 = t1897 * t1907;
        float t1912 = t1898 * t1910;
        float t1913 = t1911 - t1912;
        float t1914 = t1897 * t1910;
        float t1915 = t1898 * t1907;
        float t1916 = t1914 + t1915;
        int t1917 = t1592 + t1899;
        float t1918 = t1902 + t1913;
        memory[50441716 + t1917] = t1918;
        int t1920 = t1592 + t1899;
        int t1921 = t1920 + 1024;
        float t1922 = t1905 + t1916;
        memory[50441716 + t1921] = t1922;
        int t1924 = t1592 + t1900;
        float t1925 = t1902 - t1913;
        memory[50441716 + t1924] = t1925;
        int t1927 = t1592 + t1900;
        int t1928 = t1927 + 1024;
        float t1929 = t1905 - t1916;
        memory[50441716 + t1928] = t1929;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1932 = 0; _pr1932 < 512; _pr1932++) {
        float t1933 = (float)_pr1932;
        float t1934 = (t1933 * 0.03125);
        float t1935 = metal::floor(t1934);
        float t1936 = t1935 * 32.0;
        float t1937 = t1933 - t1936;
        float t1938 = t1935 * 64.0;
        float t1939 = t1938 + t1937;
        float t1940 = t1939 + 32.0;
        float t1941 = -6.283185 * t1937;
        float t1942 = (t1941 * 0.015625);
        float t1943 = metal::cos(t1942);
        float t1944 = metal::sin(t1942);
        int t1945 = (int)t1939;
        int t1946 = (int)t1940;
        int t1947 = t1592 + t1945;
        float t1948 = memory[50441716 + t1947];
        int t1949 = t1592 + t1945;
        int t1950 = t1949 + 1024;
        float t1951 = memory[50441716 + t1950];
        int t1952 = t1592 + t1946;
        float t1953 = memory[50441716 + t1952];
        int t1954 = t1592 + t1946;
        int t1955 = t1954 + 1024;
        float t1956 = memory[50441716 + t1955];
        float t1957 = t1943 * t1953;
        float t1958 = t1944 * t1956;
        float t1959 = t1957 - t1958;
        float t1960 = t1943 * t1956;
        float t1961 = t1944 * t1953;
        float t1962 = t1960 + t1961;
        int t1963 = t1592 + t1945;
        float t1964 = t1948 + t1959;
        memory[50441716 + t1963] = t1964;
        int t1966 = t1592 + t1945;
        int t1967 = t1966 + 1024;
        float t1968 = t1951 + t1962;
        memory[50441716 + t1967] = t1968;
        int t1970 = t1592 + t1946;
        float t1971 = t1948 - t1959;
        memory[50441716 + t1970] = t1971;
        int t1973 = t1592 + t1946;
        int t1974 = t1973 + 1024;
        float t1975 = t1951 - t1962;
        memory[50441716 + t1974] = t1975;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1978 = 0; _pr1978 < 512; _pr1978++) {
        float t1979 = (float)_pr1978;
        float t1980 = (t1979 * 0.015625);
        float t1981 = metal::floor(t1980);
        float t1982 = t1981 * 64.0;
        float t1983 = t1979 - t1982;
        float t1984 = t1981 * 128.0;
        float t1985 = t1984 + t1983;
        float t1986 = t1985 + 64.0;
        float t1987 = -6.283185 * t1983;
        float t1988 = (t1987 * 0.0078125);
        float t1989 = metal::cos(t1988);
        float t1990 = metal::sin(t1988);
        int t1991 = (int)t1985;
        int t1992 = (int)t1986;
        int t1993 = t1592 + t1991;
        float t1994 = memory[50441716 + t1993];
        int t1995 = t1592 + t1991;
        int t1996 = t1995 + 1024;
        float t1997 = memory[50441716 + t1996];
        int t1998 = t1592 + t1992;
        float t1999 = memory[50441716 + t1998];
        int t2000 = t1592 + t1992;
        int t2001 = t2000 + 1024;
        float t2002 = memory[50441716 + t2001];
        float t2003 = t1989 * t1999;
        float t2004 = t1990 * t2002;
        float t2005 = t2003 - t2004;
        float t2006 = t1989 * t2002;
        float t2007 = t1990 * t1999;
        float t2008 = t2006 + t2007;
        int t2009 = t1592 + t1991;
        float t2010 = t1994 + t2005;
        memory[50441716 + t2009] = t2010;
        int t2012 = t1592 + t1991;
        int t2013 = t2012 + 1024;
        float t2014 = t1997 + t2008;
        memory[50441716 + t2013] = t2014;
        int t2016 = t1592 + t1992;
        float t2017 = t1994 - t2005;
        memory[50441716 + t2016] = t2017;
        int t2019 = t1592 + t1992;
        int t2020 = t2019 + 1024;
        float t2021 = t1997 - t2008;
        memory[50441716 + t2020] = t2021;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2024 = 0; _pr2024 < 512; _pr2024++) {
        float t2025 = (float)_pr2024;
        float t2026 = (t2025 * 0.0078125);
        float t2027 = metal::floor(t2026);
        float t2028 = t2027 * 128.0;
        float t2029 = t2025 - t2028;
        float t2030 = t2027 * 256.0;
        float t2031 = t2030 + t2029;
        float t2032 = t2031 + 128.0;
        float t2033 = -6.283185 * t2029;
        float t2034 = (t2033 * 0.00390625);
        float t2035 = metal::cos(t2034);
        float t2036 = metal::sin(t2034);
        int t2037 = (int)t2031;
        int t2038 = (int)t2032;
        int t2039 = t1592 + t2037;
        float t2040 = memory[50441716 + t2039];
        int t2041 = t1592 + t2037;
        int t2042 = t2041 + 1024;
        float t2043 = memory[50441716 + t2042];
        int t2044 = t1592 + t2038;
        float t2045 = memory[50441716 + t2044];
        int t2046 = t1592 + t2038;
        int t2047 = t2046 + 1024;
        float t2048 = memory[50441716 + t2047];
        float t2049 = t2035 * t2045;
        float t2050 = t2036 * t2048;
        float t2051 = t2049 - t2050;
        float t2052 = t2035 * t2048;
        float t2053 = t2036 * t2045;
        float t2054 = t2052 + t2053;
        int t2055 = t1592 + t2037;
        float t2056 = t2040 + t2051;
        memory[50441716 + t2055] = t2056;
        int t2058 = t1592 + t2037;
        int t2059 = t2058 + 1024;
        float t2060 = t2043 + t2054;
        memory[50441716 + t2059] = t2060;
        int t2062 = t1592 + t2038;
        float t2063 = t2040 - t2051;
        memory[50441716 + t2062] = t2063;
        int t2065 = t1592 + t2038;
        int t2066 = t2065 + 1024;
        float t2067 = t2043 - t2054;
        memory[50441716 + t2066] = t2067;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2070 = 0; _pr2070 < 512; _pr2070++) {
        float t2071 = (float)_pr2070;
        float t2072 = (t2071 * 0.00390625);
        float t2073 = metal::floor(t2072);
        float t2074 = t2073 * 256.0;
        float t2075 = t2071 - t2074;
        float t2076 = t2073 * 512.0;
        float t2077 = t2076 + t2075;
        float t2078 = t2077 + 256.0;
        float t2079 = -6.283185 * t2075;
        float t2080 = (t2079 * 0.001953125);
        float t2081 = metal::cos(t2080);
        float t2082 = metal::sin(t2080);
        int t2083 = (int)t2077;
        int t2084 = (int)t2078;
        int t2085 = t1592 + t2083;
        float t2086 = memory[50441716 + t2085];
        int t2087 = t1592 + t2083;
        int t2088 = t2087 + 1024;
        float t2089 = memory[50441716 + t2088];
        int t2090 = t1592 + t2084;
        float t2091 = memory[50441716 + t2090];
        int t2092 = t1592 + t2084;
        int t2093 = t2092 + 1024;
        float t2094 = memory[50441716 + t2093];
        float t2095 = t2081 * t2091;
        float t2096 = t2082 * t2094;
        float t2097 = t2095 - t2096;
        float t2098 = t2081 * t2094;
        float t2099 = t2082 * t2091;
        float t2100 = t2098 + t2099;
        int t2101 = t1592 + t2083;
        float t2102 = t2086 + t2097;
        memory[50441716 + t2101] = t2102;
        int t2104 = t1592 + t2083;
        int t2105 = t2104 + 1024;
        float t2106 = t2089 + t2100;
        memory[50441716 + t2105] = t2106;
        int t2108 = t1592 + t2084;
        float t2109 = t2086 - t2097;
        memory[50441716 + t2108] = t2109;
        int t2111 = t1592 + t2084;
        int t2112 = t2111 + 1024;
        float t2113 = t2089 - t2100;
        memory[50441716 + t2112] = t2113;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2116 = 0; _pr2116 < 512; _pr2116++) {
        float t2117 = (float)_pr2116;
        float t2118 = (t2117 * 0.001953125);
        float t2119 = metal::floor(t2118);
        float t2120 = t2119 * 512.0;
        float t2121 = t2117 - t2120;
        float t2122 = t2119 * 1024.0;
        float t2123 = t2122 + t2121;
        float t2124 = t2123 + 512.0;
        float t2125 = -6.283185 * t2121;
        float t2126 = (t2125 * 0.0009765625);
        float t2127 = metal::cos(t2126);
        float t2128 = metal::sin(t2126);
        int t2129 = (int)t2123;
        int t2130 = (int)t2124;
        int t2131 = t1592 + t2129;
        float t2132 = memory[50441716 + t2131];
        int t2133 = t1592 + t2129;
        int t2134 = t2133 + 1024;
        float t2135 = memory[50441716 + t2134];
        int t2136 = t1592 + t2130;
        float t2137 = memory[50441716 + t2136];
        int t2138 = t1592 + t2130;
        int t2139 = t2138 + 1024;
        float t2140 = memory[50441716 + t2139];
        float t2141 = t2127 * t2137;
        float t2142 = t2128 * t2140;
        float t2143 = t2141 - t2142;
        float t2144 = t2127 * t2140;
        float t2145 = t2128 * t2137;
        float t2146 = t2144 + t2145;
        int t2147 = t1592 + t2129;
        float t2148 = t2132 + t2143;
        memory[50441716 + t2147] = t2148;
        int t2150 = t1592 + t2129;
        int t2151 = t2150 + 1024;
        float t2152 = t2135 + t2146;
        memory[50441716 + t2151] = t2152;
        int t2154 = t1592 + t2130;
        float t2155 = t2132 - t2143;
        memory[50441716 + t2154] = t2155;
        int t2157 = t1592 + t2130;
        int t2158 = t2157 + 1024;
        float t2159 = t2135 - t2146;
        memory[50441716 + t2158] = t2159;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2162 = 0; t2162 < 1024; t2162++) {
        float t2163 = (float)t2162;
        float t2164 = (t2163 - metal::floor(t2163 / 2.0) * 2.0);
        float t2165 = t2164;
        float t2166 = (t2163 * 0.5);
        float t2167 = metal::floor(t2166);
        float t2168 = t2165 * 2.0;
        float t2169 = (t2167 - metal::floor(t2167 / 2.0) * 2.0);
        float t2170 = t2168 + t2169;
        float t2171 = (t2167 * 0.5);
        float t2172 = metal::floor(t2171);
        float t2173 = t2170 * 2.0;
        float t2174 = (t2172 - metal::floor(t2172 / 2.0) * 2.0);
        float t2175 = t2173 + t2174;
        float t2176 = (t2172 * 0.5);
        float t2177 = metal::floor(t2176);
        float t2178 = t2175 * 2.0;
        float t2179 = (t2177 - metal::floor(t2177 / 2.0) * 2.0);
        float t2180 = t2178 + t2179;
        float t2181 = (t2177 * 0.5);
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
        float t2213 = (float)t2162;
        float t2214 = t2213 < t2210;
        int t2215 = (int)t2210;
        int t2216 = t1592 + t2162;
        float t2217 = memory[83996148 + t2216];
        int t2218 = t1592 + t2162;
        int t2219 = t2218 + 1024;
        float t2220 = memory[83996148 + t2219];
        int t2221 = t1592 + t2215;
        float t2222 = memory[83996148 + t2221];
        int t2223 = t1592 + t2215;
        int t2224 = t2223 + 1024;
        float t2225 = memory[83996148 + t2224];
        float t2226 = metal::select(t2217, t2222, t2214 > 0.0);
        float t2227 = metal::select(t2220, t2225, t2214 > 0.0);
        float t2228 = metal::select(t2222, t2217, t2214 > 0.0);
        float t2229 = metal::select(t2225, t2220, t2214 > 0.0);
        int t2230 = t1592 + t2162;
        memory[83996148 + t2230] = t2226;
        int t2232 = t1592 + t2162;
        int t2233 = t2232 + 1024;
        memory[83996148 + t2233] = t2227;
        int t2235 = t1592 + t2215;
        memory[83996148 + t2235] = t2228;
        int t2237 = t1592 + t2215;
        int t2238 = t2237 + 1024;
        memory[83996148 + t2238] = t2229;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2241 = 0; _pr2241 < 512; _pr2241++) {
        float t2242 = (float)_pr2241;
        float t2243 = t2242;
        float t2244 = metal::floor(t2243);
        float t2245 = t2244;
        float t2246 = t2242 - t2245;
        float t2247 = t2244 * 2.0;
        float t2248 = t2247 + t2246;
        float t2249 = t2248 + 1.0;
        float t2250 = -6.283185 * t2246;
        float t2251 = (t2250 * 0.5);
        float t2252 = metal::cos(t2251);
        float t2253 = metal::sin(t2251);
        int t2254 = (int)t2248;
        int t2255 = (int)t2249;
        int t2256 = t1592 + t2254;
        float t2257 = memory[83996148 + t2256];
        int t2258 = t1592 + t2254;
        int t2259 = t2258 + 1024;
        float t2260 = memory[83996148 + t2259];
        int t2261 = t1592 + t2255;
        float t2262 = memory[83996148 + t2261];
        int t2263 = t1592 + t2255;
        int t2264 = t2263 + 1024;
        float t2265 = memory[83996148 + t2264];
        float t2266 = t2252 * t2262;
        float t2267 = t2253 * t2265;
        float t2268 = t2266 - t2267;
        float t2269 = t2252 * t2265;
        float t2270 = t2253 * t2262;
        float t2271 = t2269 + t2270;
        int t2272 = t1592 + t2254;
        float t2273 = t2257 + t2268;
        memory[83996148 + t2272] = t2273;
        int t2275 = t1592 + t2254;
        int t2276 = t2275 + 1024;
        float t2277 = t2260 + t2271;
        memory[83996148 + t2276] = t2277;
        int t2279 = t1592 + t2255;
        float t2280 = t2257 - t2268;
        memory[83996148 + t2279] = t2280;
        int t2282 = t1592 + t2255;
        int t2283 = t2282 + 1024;
        float t2284 = t2260 - t2271;
        memory[83996148 + t2283] = t2284;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2287 = 0; _pr2287 < 512; _pr2287++) {
        float t2288 = (float)_pr2287;
        float t2289 = (t2288 * 0.5);
        float t2290 = metal::floor(t2289);
        float t2291 = t2290 * 2.0;
        float t2292 = t2288 - t2291;
        float t2293 = t2290 * 4.0;
        float t2294 = t2293 + t2292;
        float t2295 = t2294 + 2.0;
        float t2296 = -6.283185 * t2292;
        float t2297 = (t2296 * 0.25);
        float t2298 = metal::cos(t2297);
        float t2299 = metal::sin(t2297);
        int t2300 = (int)t2294;
        int t2301 = (int)t2295;
        int t2302 = t1592 + t2300;
        float t2303 = memory[83996148 + t2302];
        int t2304 = t1592 + t2300;
        int t2305 = t2304 + 1024;
        float t2306 = memory[83996148 + t2305];
        int t2307 = t1592 + t2301;
        float t2308 = memory[83996148 + t2307];
        int t2309 = t1592 + t2301;
        int t2310 = t2309 + 1024;
        float t2311 = memory[83996148 + t2310];
        float t2312 = t2298 * t2308;
        float t2313 = t2299 * t2311;
        float t2314 = t2312 - t2313;
        float t2315 = t2298 * t2311;
        float t2316 = t2299 * t2308;
        float t2317 = t2315 + t2316;
        int t2318 = t1592 + t2300;
        float t2319 = t2303 + t2314;
        memory[83996148 + t2318] = t2319;
        int t2321 = t1592 + t2300;
        int t2322 = t2321 + 1024;
        float t2323 = t2306 + t2317;
        memory[83996148 + t2322] = t2323;
        int t2325 = t1592 + t2301;
        float t2326 = t2303 - t2314;
        memory[83996148 + t2325] = t2326;
        int t2328 = t1592 + t2301;
        int t2329 = t2328 + 1024;
        float t2330 = t2306 - t2317;
        memory[83996148 + t2329] = t2330;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2333 = 0; _pr2333 < 512; _pr2333++) {
        float t2334 = (float)_pr2333;
        float t2335 = (t2334 * 0.25);
        float t2336 = metal::floor(t2335);
        float t2337 = t2336 * 4.0;
        float t2338 = t2334 - t2337;
        float t2339 = t2336 * 8.0;
        float t2340 = t2339 + t2338;
        float t2341 = t2340 + 4.0;
        float t2342 = -6.283185 * t2338;
        float t2343 = (t2342 * 0.125);
        float t2344 = metal::cos(t2343);
        float t2345 = metal::sin(t2343);
        int t2346 = (int)t2340;
        int t2347 = (int)t2341;
        int t2348 = t1592 + t2346;
        float t2349 = memory[83996148 + t2348];
        int t2350 = t1592 + t2346;
        int t2351 = t2350 + 1024;
        float t2352 = memory[83996148 + t2351];
        int t2353 = t1592 + t2347;
        float t2354 = memory[83996148 + t2353];
        int t2355 = t1592 + t2347;
        int t2356 = t2355 + 1024;
        float t2357 = memory[83996148 + t2356];
        float t2358 = t2344 * t2354;
        float t2359 = t2345 * t2357;
        float t2360 = t2358 - t2359;
        float t2361 = t2344 * t2357;
        float t2362 = t2345 * t2354;
        float t2363 = t2361 + t2362;
        int t2364 = t1592 + t2346;
        float t2365 = t2349 + t2360;
        memory[83996148 + t2364] = t2365;
        int t2367 = t1592 + t2346;
        int t2368 = t2367 + 1024;
        float t2369 = t2352 + t2363;
        memory[83996148 + t2368] = t2369;
        int t2371 = t1592 + t2347;
        float t2372 = t2349 - t2360;
        memory[83996148 + t2371] = t2372;
        int t2374 = t1592 + t2347;
        int t2375 = t2374 + 1024;
        float t2376 = t2352 - t2363;
        memory[83996148 + t2375] = t2376;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2379 = 0; _pr2379 < 512; _pr2379++) {
        float t2380 = (float)_pr2379;
        float t2381 = (t2380 * 0.125);
        float t2382 = metal::floor(t2381);
        float t2383 = t2382 * 8.0;
        float t2384 = t2380 - t2383;
        float t2385 = t2382 * 16.0;
        float t2386 = t2385 + t2384;
        float t2387 = t2386 + 8.0;
        float t2388 = -6.283185 * t2384;
        float t2389 = (t2388 * 0.0625);
        float t2390 = metal::cos(t2389);
        float t2391 = metal::sin(t2389);
        int t2392 = (int)t2386;
        int t2393 = (int)t2387;
        int t2394 = t1592 + t2392;
        float t2395 = memory[83996148 + t2394];
        int t2396 = t1592 + t2392;
        int t2397 = t2396 + 1024;
        float t2398 = memory[83996148 + t2397];
        int t2399 = t1592 + t2393;
        float t2400 = memory[83996148 + t2399];
        int t2401 = t1592 + t2393;
        int t2402 = t2401 + 1024;
        float t2403 = memory[83996148 + t2402];
        float t2404 = t2390 * t2400;
        float t2405 = t2391 * t2403;
        float t2406 = t2404 - t2405;
        float t2407 = t2390 * t2403;
        float t2408 = t2391 * t2400;
        float t2409 = t2407 + t2408;
        int t2410 = t1592 + t2392;
        float t2411 = t2395 + t2406;
        memory[83996148 + t2410] = t2411;
        int t2413 = t1592 + t2392;
        int t2414 = t2413 + 1024;
        float t2415 = t2398 + t2409;
        memory[83996148 + t2414] = t2415;
        int t2417 = t1592 + t2393;
        float t2418 = t2395 - t2406;
        memory[83996148 + t2417] = t2418;
        int t2420 = t1592 + t2393;
        int t2421 = t2420 + 1024;
        float t2422 = t2398 - t2409;
        memory[83996148 + t2421] = t2422;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2425 = 0; _pr2425 < 512; _pr2425++) {
        float t2426 = (float)_pr2425;
        float t2427 = (t2426 * 0.0625);
        float t2428 = metal::floor(t2427);
        float t2429 = t2428 * 16.0;
        float t2430 = t2426 - t2429;
        float t2431 = t2428 * 32.0;
        float t2432 = t2431 + t2430;
        float t2433 = t2432 + 16.0;
        float t2434 = -6.283185 * t2430;
        float t2435 = (t2434 * 0.03125);
        float t2436 = metal::cos(t2435);
        float t2437 = metal::sin(t2435);
        int t2438 = (int)t2432;
        int t2439 = (int)t2433;
        int t2440 = t1592 + t2438;
        float t2441 = memory[83996148 + t2440];
        int t2442 = t1592 + t2438;
        int t2443 = t2442 + 1024;
        float t2444 = memory[83996148 + t2443];
        int t2445 = t1592 + t2439;
        float t2446 = memory[83996148 + t2445];
        int t2447 = t1592 + t2439;
        int t2448 = t2447 + 1024;
        float t2449 = memory[83996148 + t2448];
        float t2450 = t2436 * t2446;
        float t2451 = t2437 * t2449;
        float t2452 = t2450 - t2451;
        float t2453 = t2436 * t2449;
        float t2454 = t2437 * t2446;
        float t2455 = t2453 + t2454;
        int t2456 = t1592 + t2438;
        float t2457 = t2441 + t2452;
        memory[83996148 + t2456] = t2457;
        int t2459 = t1592 + t2438;
        int t2460 = t2459 + 1024;
        float t2461 = t2444 + t2455;
        memory[83996148 + t2460] = t2461;
        int t2463 = t1592 + t2439;
        float t2464 = t2441 - t2452;
        memory[83996148 + t2463] = t2464;
        int t2466 = t1592 + t2439;
        int t2467 = t2466 + 1024;
        float t2468 = t2444 - t2455;
        memory[83996148 + t2467] = t2468;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2471 = 0; _pr2471 < 512; _pr2471++) {
        float t2472 = (float)_pr2471;
        float t2473 = (t2472 * 0.03125);
        float t2474 = metal::floor(t2473);
        float t2475 = t2474 * 32.0;
        float t2476 = t2472 - t2475;
        float t2477 = t2474 * 64.0;
        float t2478 = t2477 + t2476;
        float t2479 = t2478 + 32.0;
        float t2480 = -6.283185 * t2476;
        float t2481 = (t2480 * 0.015625);
        float t2482 = metal::cos(t2481);
        float t2483 = metal::sin(t2481);
        int t2484 = (int)t2478;
        int t2485 = (int)t2479;
        int t2486 = t1592 + t2484;
        float t2487 = memory[83996148 + t2486];
        int t2488 = t1592 + t2484;
        int t2489 = t2488 + 1024;
        float t2490 = memory[83996148 + t2489];
        int t2491 = t1592 + t2485;
        float t2492 = memory[83996148 + t2491];
        int t2493 = t1592 + t2485;
        int t2494 = t2493 + 1024;
        float t2495 = memory[83996148 + t2494];
        float t2496 = t2482 * t2492;
        float t2497 = t2483 * t2495;
        float t2498 = t2496 - t2497;
        float t2499 = t2482 * t2495;
        float t2500 = t2483 * t2492;
        float t2501 = t2499 + t2500;
        int t2502 = t1592 + t2484;
        float t2503 = t2487 + t2498;
        memory[83996148 + t2502] = t2503;
        int t2505 = t1592 + t2484;
        int t2506 = t2505 + 1024;
        float t2507 = t2490 + t2501;
        memory[83996148 + t2506] = t2507;
        int t2509 = t1592 + t2485;
        float t2510 = t2487 - t2498;
        memory[83996148 + t2509] = t2510;
        int t2512 = t1592 + t2485;
        int t2513 = t2512 + 1024;
        float t2514 = t2490 - t2501;
        memory[83996148 + t2513] = t2514;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2517 = 0; _pr2517 < 512; _pr2517++) {
        float t2518 = (float)_pr2517;
        float t2519 = (t2518 * 0.015625);
        float t2520 = metal::floor(t2519);
        float t2521 = t2520 * 64.0;
        float t2522 = t2518 - t2521;
        float t2523 = t2520 * 128.0;
        float t2524 = t2523 + t2522;
        float t2525 = t2524 + 64.0;
        float t2526 = -6.283185 * t2522;
        float t2527 = (t2526 * 0.0078125);
        float t2528 = metal::cos(t2527);
        float t2529 = metal::sin(t2527);
        int t2530 = (int)t2524;
        int t2531 = (int)t2525;
        int t2532 = t1592 + t2530;
        float t2533 = memory[83996148 + t2532];
        int t2534 = t1592 + t2530;
        int t2535 = t2534 + 1024;
        float t2536 = memory[83996148 + t2535];
        int t2537 = t1592 + t2531;
        float t2538 = memory[83996148 + t2537];
        int t2539 = t1592 + t2531;
        int t2540 = t2539 + 1024;
        float t2541 = memory[83996148 + t2540];
        float t2542 = t2528 * t2538;
        float t2543 = t2529 * t2541;
        float t2544 = t2542 - t2543;
        float t2545 = t2528 * t2541;
        float t2546 = t2529 * t2538;
        float t2547 = t2545 + t2546;
        int t2548 = t1592 + t2530;
        float t2549 = t2533 + t2544;
        memory[83996148 + t2548] = t2549;
        int t2551 = t1592 + t2530;
        int t2552 = t2551 + 1024;
        float t2553 = t2536 + t2547;
        memory[83996148 + t2552] = t2553;
        int t2555 = t1592 + t2531;
        float t2556 = t2533 - t2544;
        memory[83996148 + t2555] = t2556;
        int t2558 = t1592 + t2531;
        int t2559 = t2558 + 1024;
        float t2560 = t2536 - t2547;
        memory[83996148 + t2559] = t2560;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2563 = 0; _pr2563 < 512; _pr2563++) {
        float t2564 = (float)_pr2563;
        float t2565 = (t2564 * 0.0078125);
        float t2566 = metal::floor(t2565);
        float t2567 = t2566 * 128.0;
        float t2568 = t2564 - t2567;
        float t2569 = t2566 * 256.0;
        float t2570 = t2569 + t2568;
        float t2571 = t2570 + 128.0;
        float t2572 = -6.283185 * t2568;
        float t2573 = (t2572 * 0.00390625);
        float t2574 = metal::cos(t2573);
        float t2575 = metal::sin(t2573);
        int t2576 = (int)t2570;
        int t2577 = (int)t2571;
        int t2578 = t1592 + t2576;
        float t2579 = memory[83996148 + t2578];
        int t2580 = t1592 + t2576;
        int t2581 = t2580 + 1024;
        float t2582 = memory[83996148 + t2581];
        int t2583 = t1592 + t2577;
        float t2584 = memory[83996148 + t2583];
        int t2585 = t1592 + t2577;
        int t2586 = t2585 + 1024;
        float t2587 = memory[83996148 + t2586];
        float t2588 = t2574 * t2584;
        float t2589 = t2575 * t2587;
        float t2590 = t2588 - t2589;
        float t2591 = t2574 * t2587;
        float t2592 = t2575 * t2584;
        float t2593 = t2591 + t2592;
        int t2594 = t1592 + t2576;
        float t2595 = t2579 + t2590;
        memory[83996148 + t2594] = t2595;
        int t2597 = t1592 + t2576;
        int t2598 = t2597 + 1024;
        float t2599 = t2582 + t2593;
        memory[83996148 + t2598] = t2599;
        int t2601 = t1592 + t2577;
        float t2602 = t2579 - t2590;
        memory[83996148 + t2601] = t2602;
        int t2604 = t1592 + t2577;
        int t2605 = t2604 + 1024;
        float t2606 = t2582 - t2593;
        memory[83996148 + t2605] = t2606;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2609 = 0; _pr2609 < 512; _pr2609++) {
        float t2610 = (float)_pr2609;
        float t2611 = (t2610 * 0.00390625);
        float t2612 = metal::floor(t2611);
        float t2613 = t2612 * 256.0;
        float t2614 = t2610 - t2613;
        float t2615 = t2612 * 512.0;
        float t2616 = t2615 + t2614;
        float t2617 = t2616 + 256.0;
        float t2618 = -6.283185 * t2614;
        float t2619 = (t2618 * 0.001953125);
        float t2620 = metal::cos(t2619);
        float t2621 = metal::sin(t2619);
        int t2622 = (int)t2616;
        int t2623 = (int)t2617;
        int t2624 = t1592 + t2622;
        float t2625 = memory[83996148 + t2624];
        int t2626 = t1592 + t2622;
        int t2627 = t2626 + 1024;
        float t2628 = memory[83996148 + t2627];
        int t2629 = t1592 + t2623;
        float t2630 = memory[83996148 + t2629];
        int t2631 = t1592 + t2623;
        int t2632 = t2631 + 1024;
        float t2633 = memory[83996148 + t2632];
        float t2634 = t2620 * t2630;
        float t2635 = t2621 * t2633;
        float t2636 = t2634 - t2635;
        float t2637 = t2620 * t2633;
        float t2638 = t2621 * t2630;
        float t2639 = t2637 + t2638;
        int t2640 = t1592 + t2622;
        float t2641 = t2625 + t2636;
        memory[83996148 + t2640] = t2641;
        int t2643 = t1592 + t2622;
        int t2644 = t2643 + 1024;
        float t2645 = t2628 + t2639;
        memory[83996148 + t2644] = t2645;
        int t2647 = t1592 + t2623;
        float t2648 = t2625 - t2636;
        memory[83996148 + t2647] = t2648;
        int t2650 = t1592 + t2623;
        int t2651 = t2650 + 1024;
        float t2652 = t2628 - t2639;
        memory[83996148 + t2651] = t2652;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2655 = 0; _pr2655 < 512; _pr2655++) {
        float t2656 = (float)_pr2655;
        float t2657 = (t2656 * 0.001953125);
        float t2658 = metal::floor(t2657);
        float t2659 = t2658 * 512.0;
        float t2660 = t2656 - t2659;
        float t2661 = t2658 * 1024.0;
        float t2662 = t2661 + t2660;
        float t2663 = t2662 + 512.0;
        float t2664 = -6.283185 * t2660;
        float t2665 = (t2664 * 0.0009765625);
        float t2666 = metal::cos(t2665);
        float t2667 = metal::sin(t2665);
        int t2668 = (int)t2662;
        int t2669 = (int)t2663;
        int t2670 = t1592 + t2668;
        float t2671 = memory[83996148 + t2670];
        int t2672 = t1592 + t2668;
        int t2673 = t2672 + 1024;
        float t2674 = memory[83996148 + t2673];
        int t2675 = t1592 + t2669;
        float t2676 = memory[83996148 + t2675];
        int t2677 = t1592 + t2669;
        int t2678 = t2677 + 1024;
        float t2679 = memory[83996148 + t2678];
        float t2680 = t2666 * t2676;
        float t2681 = t2667 * t2679;
        float t2682 = t2680 - t2681;
        float t2683 = t2666 * t2679;
        float t2684 = t2667 * t2676;
        float t2685 = t2683 + t2684;
        int t2686 = t1592 + t2668;
        float t2687 = t2671 + t2682;
        memory[83996148 + t2686] = t2687;
        int t2689 = t1592 + t2668;
        int t2690 = t2689 + 1024;
        float t2691 = t2674 + t2685;
        memory[83996148 + t2690] = t2691;
        int t2693 = t1592 + t2669;
        float t2694 = t2671 - t2682;
        memory[83996148 + t2693] = t2694;
        int t2696 = t1592 + t2669;
        int t2697 = t2696 + 1024;
        float t2698 = t2674 - t2685;
        memory[83996148 + t2697] = t2698;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2701 = 0; _pr2701 < 513; _pr2701++) {
        int t2702 = t1592 + _pr2701;
        float t2703 = memory[50441716 + t2702];
        int t2704 = t1592 + _pr2701;
        int t2705 = t2704 + 1024;
        float t2706 = memory[50441716 + t2705];
        float t2707 = t2703 * t2703;
        float t2708 = t2706 * t2706;
        float t2709 = t2707 + t2708;
        float t2710 = metal::sqrt(t2709);
        int t2711 = t1593 + _pr2701;
        memory[117550580 + t2711] = t2710;
        int t2713 = t1592 + _pr2701;
        float t2714 = memory[83996148 + t2713];
        int t2715 = t1592 + _pr2701;
        int t2716 = t2715 + 1024;
        float t2717 = memory[83996148 + t2716];
        float t2718 = t2714 * t2714;
        float t2719 = t2717 * t2717;
        float t2720 = t2718 + t2719;
        float t2721 = metal::sqrt(t2720);
        int t2722 = t1593 + _pr2701;
        memory[125955572 + t2722] = t2721;
        float t2724 = t2710 - t2721;
        int t2725 = t1593 + _pr2701;
        float t2726 = t2724 * t2724;
        memory[134360564 + t2725] = t2726;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2729 = 0; t2729 < 513; t2729++) {
        int t2730 = t1593 + t2729;
        float t2731 = memory[134360564 + t2730];
        float t2732 = t[15*frameCount + id] + t2731;
        t[15*frameCount + id] = t2732;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2743), value: global(2743)) */
  float t5720 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5720)) {
    /* loadGlobal(1595) - handled in variable access */
    /* loadGlobal(1573) - handled in variable access */
    int t2735 = id;
    int t2736 = t2735 / 61;
    uint _frameIndex = (uint)(t2736);
    int t2737 = t2736 * 61;
    int t2738 = t2735 - t2737;
    float t2739 = (t[15*frameCount + _frameIndex] * 6.1035156e-05);
    float t2740 = t[13*frameCount + _frameIndex] + t2739;
    float t2741 = t2740 * 0.5;
    float t2742 = t2741;
    t[16*frameCount + _frameIndex] = t2742;
    float t2744 = t2741;
    float t2745 = t2740;
    float t2746 = (t[15*frameCount + _frameIndex] * 3.7252903e-09);
    float t2747 = -0.5 * t2746;
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
    /* loadGlobal(1574) - handled in variable access */
    /* loadGlobal(512) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    int t2748 = id;
    int t2749 = t2748 * 2048;
    int t2750 = t2748 * 513;
    int t2751 = t2748 * 2048;
    float t2752 = t[14*frameCount + id] == 0.0;
    if (t2752) {
      for (uint _pr2754 = 0; _pr2754 < 513; _pr2754++) {
        int t2755 = t2750 + _pr2754;
        float t2756 = memory[117550580 + t2755];
        int t2757 = t2750 + _pr2754;
        float t2758 = memory[125955572 + t2757];
        int t2759 = t2749 + _pr2754;
        float t2760 = memory[50441716 + t2759];
        int t2761 = t2749 + _pr2754;
        int t2762 = t2761 + 1024;
        float t2763 = memory[50441716 + t2762];
        int t2764 = t2749 + _pr2754;
        float t2765 = memory[83996148 + t2764];
        int t2766 = t2749 + _pr2754;
        int t2767 = t2766 + 1024;
        float t2768 = memory[83996148 + t2767];
        float t2769 = t2756 - t2758;
        float t2770 = 2.0 * t2769;
        float t2771 = t2770 * 3.0517578e-05;
        float t2772 = t2756 - t2758;
        float t2773 = -2.0 * t2772;
        float t2774 = t2773 * 3.0517578e-05;
        float t2775 = metal::max(t2756, 1e-08);
        float t2776 = metal::max(t2758, 1e-08);
        float t2777 = t2771 * t2760;
        float t2778 = t2777 / t2775;
        float t2779 = t2771 * t2763;
        float t2780 = t2779 / t2775;
        float t2781 = t2774 * t2765;
        float t2782 = t2781 / t2776;
        float t2783 = t2774 * t2768;
        float t2784 = t2783 / t2776;
        int t2785 = t2751 + _pr2754;
        memory[142765556 + t2785] = t2778;
        int t2787 = t2751 + _pr2754;
        int t2788 = t2787 + 1024;
        memory[142765556 + t2788] = t2780;
        int t2790 = t2751 + _pr2754;
        memory[176319988 + t2790] = t2782;
        int t2792 = t2751 + _pr2754;
        int t2793 = t2792 + 1024;
        memory[176319988 + t2793] = t2784;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2796 = 0; _pr2796 < 511; _pr2796++) {
        int t2797 = _pr2796 + 513;
        int t2798 = t2751 + t2797;
        memory[142765556 + t2798] = 0.0;
        int t2800 = t2751 + t2797;
        int t2801 = t2800 + 1024;
        memory[142765556 + t2801] = 0.0;
        int t2803 = t2751 + t2797;
        memory[176319988 + t2803] = 0.0;
        int t2805 = t2751 + t2797;
        int t2806 = t2805 + 1024;
        memory[176319988 + t2806] = 0.0;
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
    /* loadGlobal(1574) - handled in variable access */
    int t2810 = id;
    int t2811 = t2810 * 2048;
    int t2812 = t2810 * 1024;
    float t2813 = t[14*frameCount + id] == 0.0;
    if (t2813) {
      for (uint t2815 = 0; t2815 < 1024; t2815++) {
        float t2816 = (float)t2815;
        float t2817 = (t2816 - metal::floor(t2816 / 2.0) * 2.0);
        float t2818 = t2817;
        float t2819 = (t2816 * 0.5);
        float t2820 = metal::floor(t2819);
        float t2821 = t2818 * 2.0;
        float t2822 = (t2820 - metal::floor(t2820 / 2.0) * 2.0);
        float t2823 = t2821 + t2822;
        float t2824 = (t2820 * 0.5);
        float t2825 = metal::floor(t2824);
        float t2826 = t2823 * 2.0;
        float t2827 = (t2825 - metal::floor(t2825 / 2.0) * 2.0);
        float t2828 = t2826 + t2827;
        float t2829 = (t2825 * 0.5);
        float t2830 = metal::floor(t2829);
        float t2831 = t2828 * 2.0;
        float t2832 = (t2830 - metal::floor(t2830 / 2.0) * 2.0);
        float t2833 = t2831 + t2832;
        float t2834 = (t2830 * 0.5);
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
        float t2866 = (float)t2815;
        float t2867 = t2866 < t2863;
        int t2868 = (int)t2863;
        int t2869 = t2811 + t2815;
        float t2870 = memory[142765556 + t2869];
        int t2871 = t2811 + t2815;
        int t2872 = t2871 + 1024;
        float t2873 = memory[142765556 + t2872];
        int t2874 = t2811 + t2868;
        float t2875 = memory[142765556 + t2874];
        int t2876 = t2811 + t2868;
        int t2877 = t2876 + 1024;
        float t2878 = memory[142765556 + t2877];
        float t2879 = metal::select(t2870, t2875, t2867 > 0.0);
        float t2880 = metal::select(t2873, t2878, t2867 > 0.0);
        float t2881 = metal::select(t2875, t2870, t2867 > 0.0);
        float t2882 = metal::select(t2878, t2873, t2867 > 0.0);
        int t2883 = t2811 + t2815;
        memory[142765556 + t2883] = t2879;
        int t2885 = t2811 + t2815;
        int t2886 = t2885 + 1024;
        memory[142765556 + t2886] = t2880;
        int t2888 = t2811 + t2868;
        memory[142765556 + t2888] = t2881;
        int t2890 = t2811 + t2868;
        int t2891 = t2890 + 1024;
        memory[142765556 + t2891] = t2882;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2894 = 0; _pr2894 < 512; _pr2894++) {
        float t2895 = (float)_pr2894;
        float t2896 = t2895;
        float t2897 = metal::floor(t2896);
        float t2898 = t2897;
        float t2899 = t2895 - t2898;
        float t2900 = t2897 * 2.0;
        float t2901 = t2900 + t2899;
        float t2902 = t2901 + 1.0;
        float t2903 = 6.283185 * t2899;
        float t2904 = (t2903 * 0.5);
        float t2905 = metal::cos(t2904);
        float t2906 = metal::sin(t2904);
        int t2907 = (int)t2901;
        int t2908 = (int)t2902;
        int t2909 = t2811 + t2907;
        float t2910 = memory[142765556 + t2909];
        int t2911 = t2811 + t2907;
        int t2912 = t2911 + 1024;
        float t2913 = memory[142765556 + t2912];
        int t2914 = t2811 + t2908;
        float t2915 = memory[142765556 + t2914];
        int t2916 = t2811 + t2908;
        int t2917 = t2916 + 1024;
        float t2918 = memory[142765556 + t2917];
        float t2919 = t2905 * t2915;
        float t2920 = t2906 * t2918;
        float t2921 = t2919 - t2920;
        float t2922 = t2905 * t2918;
        float t2923 = t2906 * t2915;
        float t2924 = t2922 + t2923;
        int t2925 = t2811 + t2907;
        float t2926 = t2910 + t2921;
        memory[142765556 + t2925] = t2926;
        int t2928 = t2811 + t2907;
        int t2929 = t2928 + 1024;
        float t2930 = t2913 + t2924;
        memory[142765556 + t2929] = t2930;
        int t2932 = t2811 + t2908;
        float t2933 = t2910 - t2921;
        memory[142765556 + t2932] = t2933;
        int t2935 = t2811 + t2908;
        int t2936 = t2935 + 1024;
        float t2937 = t2913 - t2924;
        memory[142765556 + t2936] = t2937;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2940 = 0; _pr2940 < 512; _pr2940++) {
        float t2941 = (float)_pr2940;
        float t2942 = (t2941 * 0.5);
        float t2943 = metal::floor(t2942);
        float t2944 = t2943 * 2.0;
        float t2945 = t2941 - t2944;
        float t2946 = t2943 * 4.0;
        float t2947 = t2946 + t2945;
        float t2948 = t2947 + 2.0;
        float t2949 = 6.283185 * t2945;
        float t2950 = (t2949 * 0.25);
        float t2951 = metal::cos(t2950);
        float t2952 = metal::sin(t2950);
        int t2953 = (int)t2947;
        int t2954 = (int)t2948;
        int t2955 = t2811 + t2953;
        float t2956 = memory[142765556 + t2955];
        int t2957 = t2811 + t2953;
        int t2958 = t2957 + 1024;
        float t2959 = memory[142765556 + t2958];
        int t2960 = t2811 + t2954;
        float t2961 = memory[142765556 + t2960];
        int t2962 = t2811 + t2954;
        int t2963 = t2962 + 1024;
        float t2964 = memory[142765556 + t2963];
        float t2965 = t2951 * t2961;
        float t2966 = t2952 * t2964;
        float t2967 = t2965 - t2966;
        float t2968 = t2951 * t2964;
        float t2969 = t2952 * t2961;
        float t2970 = t2968 + t2969;
        int t2971 = t2811 + t2953;
        float t2972 = t2956 + t2967;
        memory[142765556 + t2971] = t2972;
        int t2974 = t2811 + t2953;
        int t2975 = t2974 + 1024;
        float t2976 = t2959 + t2970;
        memory[142765556 + t2975] = t2976;
        int t2978 = t2811 + t2954;
        float t2979 = t2956 - t2967;
        memory[142765556 + t2978] = t2979;
        int t2981 = t2811 + t2954;
        int t2982 = t2981 + 1024;
        float t2983 = t2959 - t2970;
        memory[142765556 + t2982] = t2983;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2986 = 0; _pr2986 < 512; _pr2986++) {
        float t2987 = (float)_pr2986;
        float t2988 = (t2987 * 0.25);
        float t2989 = metal::floor(t2988);
        float t2990 = t2989 * 4.0;
        float t2991 = t2987 - t2990;
        float t2992 = t2989 * 8.0;
        float t2993 = t2992 + t2991;
        float t2994 = t2993 + 4.0;
        float t2995 = 6.283185 * t2991;
        float t2996 = (t2995 * 0.125);
        float t2997 = metal::cos(t2996);
        float t2998 = metal::sin(t2996);
        int t2999 = (int)t2993;
        int t3000 = (int)t2994;
        int t3001 = t2811 + t2999;
        float t3002 = memory[142765556 + t3001];
        int t3003 = t2811 + t2999;
        int t3004 = t3003 + 1024;
        float t3005 = memory[142765556 + t3004];
        int t3006 = t2811 + t3000;
        float t3007 = memory[142765556 + t3006];
        int t3008 = t2811 + t3000;
        int t3009 = t3008 + 1024;
        float t3010 = memory[142765556 + t3009];
        float t3011 = t2997 * t3007;
        float t3012 = t2998 * t3010;
        float t3013 = t3011 - t3012;
        float t3014 = t2997 * t3010;
        float t3015 = t2998 * t3007;
        float t3016 = t3014 + t3015;
        int t3017 = t2811 + t2999;
        float t3018 = t3002 + t3013;
        memory[142765556 + t3017] = t3018;
        int t3020 = t2811 + t2999;
        int t3021 = t3020 + 1024;
        float t3022 = t3005 + t3016;
        memory[142765556 + t3021] = t3022;
        int t3024 = t2811 + t3000;
        float t3025 = t3002 - t3013;
        memory[142765556 + t3024] = t3025;
        int t3027 = t2811 + t3000;
        int t3028 = t3027 + 1024;
        float t3029 = t3005 - t3016;
        memory[142765556 + t3028] = t3029;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3032 = 0; _pr3032 < 512; _pr3032++) {
        float t3033 = (float)_pr3032;
        float t3034 = (t3033 * 0.125);
        float t3035 = metal::floor(t3034);
        float t3036 = t3035 * 8.0;
        float t3037 = t3033 - t3036;
        float t3038 = t3035 * 16.0;
        float t3039 = t3038 + t3037;
        float t3040 = t3039 + 8.0;
        float t3041 = 6.283185 * t3037;
        float t3042 = (t3041 * 0.0625);
        float t3043 = metal::cos(t3042);
        float t3044 = metal::sin(t3042);
        int t3045 = (int)t3039;
        int t3046 = (int)t3040;
        int t3047 = t2811 + t3045;
        float t3048 = memory[142765556 + t3047];
        int t3049 = t2811 + t3045;
        int t3050 = t3049 + 1024;
        float t3051 = memory[142765556 + t3050];
        int t3052 = t2811 + t3046;
        float t3053 = memory[142765556 + t3052];
        int t3054 = t2811 + t3046;
        int t3055 = t3054 + 1024;
        float t3056 = memory[142765556 + t3055];
        float t3057 = t3043 * t3053;
        float t3058 = t3044 * t3056;
        float t3059 = t3057 - t3058;
        float t3060 = t3043 * t3056;
        float t3061 = t3044 * t3053;
        float t3062 = t3060 + t3061;
        int t3063 = t2811 + t3045;
        float t3064 = t3048 + t3059;
        memory[142765556 + t3063] = t3064;
        int t3066 = t2811 + t3045;
        int t3067 = t3066 + 1024;
        float t3068 = t3051 + t3062;
        memory[142765556 + t3067] = t3068;
        int t3070 = t2811 + t3046;
        float t3071 = t3048 - t3059;
        memory[142765556 + t3070] = t3071;
        int t3073 = t2811 + t3046;
        int t3074 = t3073 + 1024;
        float t3075 = t3051 - t3062;
        memory[142765556 + t3074] = t3075;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3078 = 0; _pr3078 < 512; _pr3078++) {
        float t3079 = (float)_pr3078;
        float t3080 = (t3079 * 0.0625);
        float t3081 = metal::floor(t3080);
        float t3082 = t3081 * 16.0;
        float t3083 = t3079 - t3082;
        float t3084 = t3081 * 32.0;
        float t3085 = t3084 + t3083;
        float t3086 = t3085 + 16.0;
        float t3087 = 6.283185 * t3083;
        float t3088 = (t3087 * 0.03125);
        float t3089 = metal::cos(t3088);
        float t3090 = metal::sin(t3088);
        int t3091 = (int)t3085;
        int t3092 = (int)t3086;
        int t3093 = t2811 + t3091;
        float t3094 = memory[142765556 + t3093];
        int t3095 = t2811 + t3091;
        int t3096 = t3095 + 1024;
        float t3097 = memory[142765556 + t3096];
        int t3098 = t2811 + t3092;
        float t3099 = memory[142765556 + t3098];
        int t3100 = t2811 + t3092;
        int t3101 = t3100 + 1024;
        float t3102 = memory[142765556 + t3101];
        float t3103 = t3089 * t3099;
        float t3104 = t3090 * t3102;
        float t3105 = t3103 - t3104;
        float t3106 = t3089 * t3102;
        float t3107 = t3090 * t3099;
        float t3108 = t3106 + t3107;
        int t3109 = t2811 + t3091;
        float t3110 = t3094 + t3105;
        memory[142765556 + t3109] = t3110;
        int t3112 = t2811 + t3091;
        int t3113 = t3112 + 1024;
        float t3114 = t3097 + t3108;
        memory[142765556 + t3113] = t3114;
        int t3116 = t2811 + t3092;
        float t3117 = t3094 - t3105;
        memory[142765556 + t3116] = t3117;
        int t3119 = t2811 + t3092;
        int t3120 = t3119 + 1024;
        float t3121 = t3097 - t3108;
        memory[142765556 + t3120] = t3121;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3124 = 0; _pr3124 < 512; _pr3124++) {
        float t3125 = (float)_pr3124;
        float t3126 = (t3125 * 0.03125);
        float t3127 = metal::floor(t3126);
        float t3128 = t3127 * 32.0;
        float t3129 = t3125 - t3128;
        float t3130 = t3127 * 64.0;
        float t3131 = t3130 + t3129;
        float t3132 = t3131 + 32.0;
        float t3133 = 6.283185 * t3129;
        float t3134 = (t3133 * 0.015625);
        float t3135 = metal::cos(t3134);
        float t3136 = metal::sin(t3134);
        int t3137 = (int)t3131;
        int t3138 = (int)t3132;
        int t3139 = t2811 + t3137;
        float t3140 = memory[142765556 + t3139];
        int t3141 = t2811 + t3137;
        int t3142 = t3141 + 1024;
        float t3143 = memory[142765556 + t3142];
        int t3144 = t2811 + t3138;
        float t3145 = memory[142765556 + t3144];
        int t3146 = t2811 + t3138;
        int t3147 = t3146 + 1024;
        float t3148 = memory[142765556 + t3147];
        float t3149 = t3135 * t3145;
        float t3150 = t3136 * t3148;
        float t3151 = t3149 - t3150;
        float t3152 = t3135 * t3148;
        float t3153 = t3136 * t3145;
        float t3154 = t3152 + t3153;
        int t3155 = t2811 + t3137;
        float t3156 = t3140 + t3151;
        memory[142765556 + t3155] = t3156;
        int t3158 = t2811 + t3137;
        int t3159 = t3158 + 1024;
        float t3160 = t3143 + t3154;
        memory[142765556 + t3159] = t3160;
        int t3162 = t2811 + t3138;
        float t3163 = t3140 - t3151;
        memory[142765556 + t3162] = t3163;
        int t3165 = t2811 + t3138;
        int t3166 = t3165 + 1024;
        float t3167 = t3143 - t3154;
        memory[142765556 + t3166] = t3167;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3170 = 0; _pr3170 < 512; _pr3170++) {
        float t3171 = (float)_pr3170;
        float t3172 = (t3171 * 0.015625);
        float t3173 = metal::floor(t3172);
        float t3174 = t3173 * 64.0;
        float t3175 = t3171 - t3174;
        float t3176 = t3173 * 128.0;
        float t3177 = t3176 + t3175;
        float t3178 = t3177 + 64.0;
        float t3179 = 6.283185 * t3175;
        float t3180 = (t3179 * 0.0078125);
        float t3181 = metal::cos(t3180);
        float t3182 = metal::sin(t3180);
        int t3183 = (int)t3177;
        int t3184 = (int)t3178;
        int t3185 = t2811 + t3183;
        float t3186 = memory[142765556 + t3185];
        int t3187 = t2811 + t3183;
        int t3188 = t3187 + 1024;
        float t3189 = memory[142765556 + t3188];
        int t3190 = t2811 + t3184;
        float t3191 = memory[142765556 + t3190];
        int t3192 = t2811 + t3184;
        int t3193 = t3192 + 1024;
        float t3194 = memory[142765556 + t3193];
        float t3195 = t3181 * t3191;
        float t3196 = t3182 * t3194;
        float t3197 = t3195 - t3196;
        float t3198 = t3181 * t3194;
        float t3199 = t3182 * t3191;
        float t3200 = t3198 + t3199;
        int t3201 = t2811 + t3183;
        float t3202 = t3186 + t3197;
        memory[142765556 + t3201] = t3202;
        int t3204 = t2811 + t3183;
        int t3205 = t3204 + 1024;
        float t3206 = t3189 + t3200;
        memory[142765556 + t3205] = t3206;
        int t3208 = t2811 + t3184;
        float t3209 = t3186 - t3197;
        memory[142765556 + t3208] = t3209;
        int t3211 = t2811 + t3184;
        int t3212 = t3211 + 1024;
        float t3213 = t3189 - t3200;
        memory[142765556 + t3212] = t3213;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3216 = 0; _pr3216 < 512; _pr3216++) {
        float t3217 = (float)_pr3216;
        float t3218 = (t3217 * 0.0078125);
        float t3219 = metal::floor(t3218);
        float t3220 = t3219 * 128.0;
        float t3221 = t3217 - t3220;
        float t3222 = t3219 * 256.0;
        float t3223 = t3222 + t3221;
        float t3224 = t3223 + 128.0;
        float t3225 = 6.283185 * t3221;
        float t3226 = (t3225 * 0.00390625);
        float t3227 = metal::cos(t3226);
        float t3228 = metal::sin(t3226);
        int t3229 = (int)t3223;
        int t3230 = (int)t3224;
        int t3231 = t2811 + t3229;
        float t3232 = memory[142765556 + t3231];
        int t3233 = t2811 + t3229;
        int t3234 = t3233 + 1024;
        float t3235 = memory[142765556 + t3234];
        int t3236 = t2811 + t3230;
        float t3237 = memory[142765556 + t3236];
        int t3238 = t2811 + t3230;
        int t3239 = t3238 + 1024;
        float t3240 = memory[142765556 + t3239];
        float t3241 = t3227 * t3237;
        float t3242 = t3228 * t3240;
        float t3243 = t3241 - t3242;
        float t3244 = t3227 * t3240;
        float t3245 = t3228 * t3237;
        float t3246 = t3244 + t3245;
        int t3247 = t2811 + t3229;
        float t3248 = t3232 + t3243;
        memory[142765556 + t3247] = t3248;
        int t3250 = t2811 + t3229;
        int t3251 = t3250 + 1024;
        float t3252 = t3235 + t3246;
        memory[142765556 + t3251] = t3252;
        int t3254 = t2811 + t3230;
        float t3255 = t3232 - t3243;
        memory[142765556 + t3254] = t3255;
        int t3257 = t2811 + t3230;
        int t3258 = t3257 + 1024;
        float t3259 = t3235 - t3246;
        memory[142765556 + t3258] = t3259;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3262 = 0; _pr3262 < 512; _pr3262++) {
        float t3263 = (float)_pr3262;
        float t3264 = (t3263 * 0.00390625);
        float t3265 = metal::floor(t3264);
        float t3266 = t3265 * 256.0;
        float t3267 = t3263 - t3266;
        float t3268 = t3265 * 512.0;
        float t3269 = t3268 + t3267;
        float t3270 = t3269 + 256.0;
        float t3271 = 6.283185 * t3267;
        float t3272 = (t3271 * 0.001953125);
        float t3273 = metal::cos(t3272);
        float t3274 = metal::sin(t3272);
        int t3275 = (int)t3269;
        int t3276 = (int)t3270;
        int t3277 = t2811 + t3275;
        float t3278 = memory[142765556 + t3277];
        int t3279 = t2811 + t3275;
        int t3280 = t3279 + 1024;
        float t3281 = memory[142765556 + t3280];
        int t3282 = t2811 + t3276;
        float t3283 = memory[142765556 + t3282];
        int t3284 = t2811 + t3276;
        int t3285 = t3284 + 1024;
        float t3286 = memory[142765556 + t3285];
        float t3287 = t3273 * t3283;
        float t3288 = t3274 * t3286;
        float t3289 = t3287 - t3288;
        float t3290 = t3273 * t3286;
        float t3291 = t3274 * t3283;
        float t3292 = t3290 + t3291;
        int t3293 = t2811 + t3275;
        float t3294 = t3278 + t3289;
        memory[142765556 + t3293] = t3294;
        int t3296 = t2811 + t3275;
        int t3297 = t3296 + 1024;
        float t3298 = t3281 + t3292;
        memory[142765556 + t3297] = t3298;
        int t3300 = t2811 + t3276;
        float t3301 = t3278 - t3289;
        memory[142765556 + t3300] = t3301;
        int t3303 = t2811 + t3276;
        int t3304 = t3303 + 1024;
        float t3305 = t3281 - t3292;
        memory[142765556 + t3304] = t3305;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3308 = 0; _pr3308 < 512; _pr3308++) {
        float t3309 = (float)_pr3308;
        float t3310 = (t3309 * 0.001953125);
        float t3311 = metal::floor(t3310);
        float t3312 = t3311 * 512.0;
        float t3313 = t3309 - t3312;
        float t3314 = t3311 * 1024.0;
        float t3315 = t3314 + t3313;
        float t3316 = t3315 + 512.0;
        float t3317 = 6.283185 * t3313;
        float t3318 = (t3317 * 0.0009765625);
        float t3319 = metal::cos(t3318);
        float t3320 = metal::sin(t3318);
        int t3321 = (int)t3315;
        int t3322 = (int)t3316;
        int t3323 = t2811 + t3321;
        float t3324 = memory[142765556 + t3323];
        int t3325 = t2811 + t3321;
        int t3326 = t3325 + 1024;
        float t3327 = memory[142765556 + t3326];
        int t3328 = t2811 + t3322;
        float t3329 = memory[142765556 + t3328];
        int t3330 = t2811 + t3322;
        int t3331 = t3330 + 1024;
        float t3332 = memory[142765556 + t3331];
        float t3333 = t3319 * t3329;
        float t3334 = t3320 * t3332;
        float t3335 = t3333 - t3334;
        float t3336 = t3319 * t3332;
        float t3337 = t3320 * t3329;
        float t3338 = t3336 + t3337;
        int t3339 = t2811 + t3321;
        float t3340 = t3324 + t3335;
        memory[142765556 + t3339] = t3340;
        int t3342 = t2811 + t3321;
        int t3343 = t3342 + 1024;
        float t3344 = t3327 + t3338;
        memory[142765556 + t3343] = t3344;
        int t3346 = t2811 + t3322;
        float t3347 = t3324 - t3335;
        memory[142765556 + t3346] = t3347;
        int t3349 = t2811 + t3322;
        int t3350 = t3349 + 1024;
        float t3351 = t3327 - t3338;
        memory[142765556 + t3350] = t3351;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3354 = 0; _pr3354 < 1024; _pr3354++) {
        int t3355 = t2811 + _pr3354;
        float t3356 = memory[142765556 + t3355];
        float t3357 = t3356 * 1.9036306e-06;
        float t3358 = memory[48884 + (int)_pr3354];
        int t3359 = t2812 + _pr3354;
        float t3360 = t3357 * t3358;
        memory[50441716 + t3359] = t3360;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t3363 = 0; t3363 < 1024; t3363++) {
        float t3364 = (float)t3363;
        float t3365 = (t3364 - metal::floor(t3364 / 2.0) * 2.0);
        float t3366 = t3365;
        float t3367 = (t3364 * 0.5);
        float t3368 = metal::floor(t3367);
        float t3369 = t3366 * 2.0;
        float t3370 = (t3368 - metal::floor(t3368 / 2.0) * 2.0);
        float t3371 = t3369 + t3370;
        float t3372 = (t3368 * 0.5);
        float t3373 = metal::floor(t3372);
        float t3374 = t3371 * 2.0;
        float t3375 = (t3373 - metal::floor(t3373 / 2.0) * 2.0);
        float t3376 = t3374 + t3375;
        float t3377 = (t3373 * 0.5);
        float t3378 = metal::floor(t3377);
        float t3379 = t3376 * 2.0;
        float t3380 = (t3378 - metal::floor(t3378 / 2.0) * 2.0);
        float t3381 = t3379 + t3380;
        float t3382 = (t3378 * 0.5);
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
        float t3414 = (float)t3363;
        float t3415 = t3414 < t3411;
        int t3416 = (int)t3411;
        int t3417 = t2811 + t3363;
        float t3418 = memory[176319988 + t3417];
        int t3419 = t2811 + t3363;
        int t3420 = t3419 + 1024;
        float t3421 = memory[176319988 + t3420];
        int t3422 = t2811 + t3416;
        float t3423 = memory[176319988 + t3422];
        int t3424 = t2811 + t3416;
        int t3425 = t3424 + 1024;
        float t3426 = memory[176319988 + t3425];
        float t3427 = metal::select(t3418, t3423, t3415 > 0.0);
        float t3428 = metal::select(t3421, t3426, t3415 > 0.0);
        float t3429 = metal::select(t3423, t3418, t3415 > 0.0);
        float t3430 = metal::select(t3426, t3421, t3415 > 0.0);
        int t3431 = t2811 + t3363;
        memory[176319988 + t3431] = t3427;
        int t3433 = t2811 + t3363;
        int t3434 = t3433 + 1024;
        memory[176319988 + t3434] = t3428;
        int t3436 = t2811 + t3416;
        memory[176319988 + t3436] = t3429;
        int t3438 = t2811 + t3416;
        int t3439 = t3438 + 1024;
        memory[176319988 + t3439] = t3430;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3442 = 0; _pr3442 < 512; _pr3442++) {
        float t3443 = (float)_pr3442;
        float t3444 = t3443;
        float t3445 = metal::floor(t3444);
        float t3446 = t3445;
        float t3447 = t3443 - t3446;
        float t3448 = t3445 * 2.0;
        float t3449 = t3448 + t3447;
        float t3450 = t3449 + 1.0;
        float t3451 = 6.283185 * t3447;
        float t3452 = (t3451 * 0.5);
        float t3453 = metal::cos(t3452);
        float t3454 = metal::sin(t3452);
        int t3455 = (int)t3449;
        int t3456 = (int)t3450;
        int t3457 = t2811 + t3455;
        float t3458 = memory[176319988 + t3457];
        int t3459 = t2811 + t3455;
        int t3460 = t3459 + 1024;
        float t3461 = memory[176319988 + t3460];
        int t3462 = t2811 + t3456;
        float t3463 = memory[176319988 + t3462];
        int t3464 = t2811 + t3456;
        int t3465 = t3464 + 1024;
        float t3466 = memory[176319988 + t3465];
        float t3467 = t3453 * t3463;
        float t3468 = t3454 * t3466;
        float t3469 = t3467 - t3468;
        float t3470 = t3453 * t3466;
        float t3471 = t3454 * t3463;
        float t3472 = t3470 + t3471;
        int t3473 = t2811 + t3455;
        float t3474 = t3458 + t3469;
        memory[176319988 + t3473] = t3474;
        int t3476 = t2811 + t3455;
        int t3477 = t3476 + 1024;
        float t3478 = t3461 + t3472;
        memory[176319988 + t3477] = t3478;
        int t3480 = t2811 + t3456;
        float t3481 = t3458 - t3469;
        memory[176319988 + t3480] = t3481;
        int t3483 = t2811 + t3456;
        int t3484 = t3483 + 1024;
        float t3485 = t3461 - t3472;
        memory[176319988 + t3484] = t3485;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3488 = 0; _pr3488 < 512; _pr3488++) {
        float t3489 = (float)_pr3488;
        float t3490 = (t3489 * 0.5);
        float t3491 = metal::floor(t3490);
        float t3492 = t3491 * 2.0;
        float t3493 = t3489 - t3492;
        float t3494 = t3491 * 4.0;
        float t3495 = t3494 + t3493;
        float t3496 = t3495 + 2.0;
        float t3497 = 6.283185 * t3493;
        float t3498 = (t3497 * 0.25);
        float t3499 = metal::cos(t3498);
        float t3500 = metal::sin(t3498);
        int t3501 = (int)t3495;
        int t3502 = (int)t3496;
        int t3503 = t2811 + t3501;
        float t3504 = memory[176319988 + t3503];
        int t3505 = t2811 + t3501;
        int t3506 = t3505 + 1024;
        float t3507 = memory[176319988 + t3506];
        int t3508 = t2811 + t3502;
        float t3509 = memory[176319988 + t3508];
        int t3510 = t2811 + t3502;
        int t3511 = t3510 + 1024;
        float t3512 = memory[176319988 + t3511];
        float t3513 = t3499 * t3509;
        float t3514 = t3500 * t3512;
        float t3515 = t3513 - t3514;
        float t3516 = t3499 * t3512;
        float t3517 = t3500 * t3509;
        float t3518 = t3516 + t3517;
        int t3519 = t2811 + t3501;
        float t3520 = t3504 + t3515;
        memory[176319988 + t3519] = t3520;
        int t3522 = t2811 + t3501;
        int t3523 = t3522 + 1024;
        float t3524 = t3507 + t3518;
        memory[176319988 + t3523] = t3524;
        int t3526 = t2811 + t3502;
        float t3527 = t3504 - t3515;
        memory[176319988 + t3526] = t3527;
        int t3529 = t2811 + t3502;
        int t3530 = t3529 + 1024;
        float t3531 = t3507 - t3518;
        memory[176319988 + t3530] = t3531;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3534 = 0; _pr3534 < 512; _pr3534++) {
        float t3535 = (float)_pr3534;
        float t3536 = (t3535 * 0.25);
        float t3537 = metal::floor(t3536);
        float t3538 = t3537 * 4.0;
        float t3539 = t3535 - t3538;
        float t3540 = t3537 * 8.0;
        float t3541 = t3540 + t3539;
        float t3542 = t3541 + 4.0;
        float t3543 = 6.283185 * t3539;
        float t3544 = (t3543 * 0.125);
        float t3545 = metal::cos(t3544);
        float t3546 = metal::sin(t3544);
        int t3547 = (int)t3541;
        int t3548 = (int)t3542;
        int t3549 = t2811 + t3547;
        float t3550 = memory[176319988 + t3549];
        int t3551 = t2811 + t3547;
        int t3552 = t3551 + 1024;
        float t3553 = memory[176319988 + t3552];
        int t3554 = t2811 + t3548;
        float t3555 = memory[176319988 + t3554];
        int t3556 = t2811 + t3548;
        int t3557 = t3556 + 1024;
        float t3558 = memory[176319988 + t3557];
        float t3559 = t3545 * t3555;
        float t3560 = t3546 * t3558;
        float t3561 = t3559 - t3560;
        float t3562 = t3545 * t3558;
        float t3563 = t3546 * t3555;
        float t3564 = t3562 + t3563;
        int t3565 = t2811 + t3547;
        float t3566 = t3550 + t3561;
        memory[176319988 + t3565] = t3566;
        int t3568 = t2811 + t3547;
        int t3569 = t3568 + 1024;
        float t3570 = t3553 + t3564;
        memory[176319988 + t3569] = t3570;
        int t3572 = t2811 + t3548;
        float t3573 = t3550 - t3561;
        memory[176319988 + t3572] = t3573;
        int t3575 = t2811 + t3548;
        int t3576 = t3575 + 1024;
        float t3577 = t3553 - t3564;
        memory[176319988 + t3576] = t3577;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3580 = 0; _pr3580 < 512; _pr3580++) {
        float t3581 = (float)_pr3580;
        float t3582 = (t3581 * 0.125);
        float t3583 = metal::floor(t3582);
        float t3584 = t3583 * 8.0;
        float t3585 = t3581 - t3584;
        float t3586 = t3583 * 16.0;
        float t3587 = t3586 + t3585;
        float t3588 = t3587 + 8.0;
        float t3589 = 6.283185 * t3585;
        float t3590 = (t3589 * 0.0625);
        float t3591 = metal::cos(t3590);
        float t3592 = metal::sin(t3590);
        int t3593 = (int)t3587;
        int t3594 = (int)t3588;
        int t3595 = t2811 + t3593;
        float t3596 = memory[176319988 + t3595];
        int t3597 = t2811 + t3593;
        int t3598 = t3597 + 1024;
        float t3599 = memory[176319988 + t3598];
        int t3600 = t2811 + t3594;
        float t3601 = memory[176319988 + t3600];
        int t3602 = t2811 + t3594;
        int t3603 = t3602 + 1024;
        float t3604 = memory[176319988 + t3603];
        float t3605 = t3591 * t3601;
        float t3606 = t3592 * t3604;
        float t3607 = t3605 - t3606;
        float t3608 = t3591 * t3604;
        float t3609 = t3592 * t3601;
        float t3610 = t3608 + t3609;
        int t3611 = t2811 + t3593;
        float t3612 = t3596 + t3607;
        memory[176319988 + t3611] = t3612;
        int t3614 = t2811 + t3593;
        int t3615 = t3614 + 1024;
        float t3616 = t3599 + t3610;
        memory[176319988 + t3615] = t3616;
        int t3618 = t2811 + t3594;
        float t3619 = t3596 - t3607;
        memory[176319988 + t3618] = t3619;
        int t3621 = t2811 + t3594;
        int t3622 = t3621 + 1024;
        float t3623 = t3599 - t3610;
        memory[176319988 + t3622] = t3623;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3626 = 0; _pr3626 < 512; _pr3626++) {
        float t3627 = (float)_pr3626;
        float t3628 = (t3627 * 0.0625);
        float t3629 = metal::floor(t3628);
        float t3630 = t3629 * 16.0;
        float t3631 = t3627 - t3630;
        float t3632 = t3629 * 32.0;
        float t3633 = t3632 + t3631;
        float t3634 = t3633 + 16.0;
        float t3635 = 6.283185 * t3631;
        float t3636 = (t3635 * 0.03125);
        float t3637 = metal::cos(t3636);
        float t3638 = metal::sin(t3636);
        int t3639 = (int)t3633;
        int t3640 = (int)t3634;
        int t3641 = t2811 + t3639;
        float t3642 = memory[176319988 + t3641];
        int t3643 = t2811 + t3639;
        int t3644 = t3643 + 1024;
        float t3645 = memory[176319988 + t3644];
        int t3646 = t2811 + t3640;
        float t3647 = memory[176319988 + t3646];
        int t3648 = t2811 + t3640;
        int t3649 = t3648 + 1024;
        float t3650 = memory[176319988 + t3649];
        float t3651 = t3637 * t3647;
        float t3652 = t3638 * t3650;
        float t3653 = t3651 - t3652;
        float t3654 = t3637 * t3650;
        float t3655 = t3638 * t3647;
        float t3656 = t3654 + t3655;
        int t3657 = t2811 + t3639;
        float t3658 = t3642 + t3653;
        memory[176319988 + t3657] = t3658;
        int t3660 = t2811 + t3639;
        int t3661 = t3660 + 1024;
        float t3662 = t3645 + t3656;
        memory[176319988 + t3661] = t3662;
        int t3664 = t2811 + t3640;
        float t3665 = t3642 - t3653;
        memory[176319988 + t3664] = t3665;
        int t3667 = t2811 + t3640;
        int t3668 = t3667 + 1024;
        float t3669 = t3645 - t3656;
        memory[176319988 + t3668] = t3669;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3672 = 0; _pr3672 < 512; _pr3672++) {
        float t3673 = (float)_pr3672;
        float t3674 = (t3673 * 0.03125);
        float t3675 = metal::floor(t3674);
        float t3676 = t3675 * 32.0;
        float t3677 = t3673 - t3676;
        float t3678 = t3675 * 64.0;
        float t3679 = t3678 + t3677;
        float t3680 = t3679 + 32.0;
        float t3681 = 6.283185 * t3677;
        float t3682 = (t3681 * 0.015625);
        float t3683 = metal::cos(t3682);
        float t3684 = metal::sin(t3682);
        int t3685 = (int)t3679;
        int t3686 = (int)t3680;
        int t3687 = t2811 + t3685;
        float t3688 = memory[176319988 + t3687];
        int t3689 = t2811 + t3685;
        int t3690 = t3689 + 1024;
        float t3691 = memory[176319988 + t3690];
        int t3692 = t2811 + t3686;
        float t3693 = memory[176319988 + t3692];
        int t3694 = t2811 + t3686;
        int t3695 = t3694 + 1024;
        float t3696 = memory[176319988 + t3695];
        float t3697 = t3683 * t3693;
        float t3698 = t3684 * t3696;
        float t3699 = t3697 - t3698;
        float t3700 = t3683 * t3696;
        float t3701 = t3684 * t3693;
        float t3702 = t3700 + t3701;
        int t3703 = t2811 + t3685;
        float t3704 = t3688 + t3699;
        memory[176319988 + t3703] = t3704;
        int t3706 = t2811 + t3685;
        int t3707 = t3706 + 1024;
        float t3708 = t3691 + t3702;
        memory[176319988 + t3707] = t3708;
        int t3710 = t2811 + t3686;
        float t3711 = t3688 - t3699;
        memory[176319988 + t3710] = t3711;
        int t3713 = t2811 + t3686;
        int t3714 = t3713 + 1024;
        float t3715 = t3691 - t3702;
        memory[176319988 + t3714] = t3715;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3718 = 0; _pr3718 < 512; _pr3718++) {
        float t3719 = (float)_pr3718;
        float t3720 = (t3719 * 0.015625);
        float t3721 = metal::floor(t3720);
        float t3722 = t3721 * 64.0;
        float t3723 = t3719 - t3722;
        float t3724 = t3721 * 128.0;
        float t3725 = t3724 + t3723;
        float t3726 = t3725 + 64.0;
        float t3727 = 6.283185 * t3723;
        float t3728 = (t3727 * 0.0078125);
        float t3729 = metal::cos(t3728);
        float t3730 = metal::sin(t3728);
        int t3731 = (int)t3725;
        int t3732 = (int)t3726;
        int t3733 = t2811 + t3731;
        float t3734 = memory[176319988 + t3733];
        int t3735 = t2811 + t3731;
        int t3736 = t3735 + 1024;
        float t3737 = memory[176319988 + t3736];
        int t3738 = t2811 + t3732;
        float t3739 = memory[176319988 + t3738];
        int t3740 = t2811 + t3732;
        int t3741 = t3740 + 1024;
        float t3742 = memory[176319988 + t3741];
        float t3743 = t3729 * t3739;
        float t3744 = t3730 * t3742;
        float t3745 = t3743 - t3744;
        float t3746 = t3729 * t3742;
        float t3747 = t3730 * t3739;
        float t3748 = t3746 + t3747;
        int t3749 = t2811 + t3731;
        float t3750 = t3734 + t3745;
        memory[176319988 + t3749] = t3750;
        int t3752 = t2811 + t3731;
        int t3753 = t3752 + 1024;
        float t3754 = t3737 + t3748;
        memory[176319988 + t3753] = t3754;
        int t3756 = t2811 + t3732;
        float t3757 = t3734 - t3745;
        memory[176319988 + t3756] = t3757;
        int t3759 = t2811 + t3732;
        int t3760 = t3759 + 1024;
        float t3761 = t3737 - t3748;
        memory[176319988 + t3760] = t3761;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3764 = 0; _pr3764 < 512; _pr3764++) {
        float t3765 = (float)_pr3764;
        float t3766 = (t3765 * 0.0078125);
        float t3767 = metal::floor(t3766);
        float t3768 = t3767 * 128.0;
        float t3769 = t3765 - t3768;
        float t3770 = t3767 * 256.0;
        float t3771 = t3770 + t3769;
        float t3772 = t3771 + 128.0;
        float t3773 = 6.283185 * t3769;
        float t3774 = (t3773 * 0.00390625);
        float t3775 = metal::cos(t3774);
        float t3776 = metal::sin(t3774);
        int t3777 = (int)t3771;
        int t3778 = (int)t3772;
        int t3779 = t2811 + t3777;
        float t3780 = memory[176319988 + t3779];
        int t3781 = t2811 + t3777;
        int t3782 = t3781 + 1024;
        float t3783 = memory[176319988 + t3782];
        int t3784 = t2811 + t3778;
        float t3785 = memory[176319988 + t3784];
        int t3786 = t2811 + t3778;
        int t3787 = t3786 + 1024;
        float t3788 = memory[176319988 + t3787];
        float t3789 = t3775 * t3785;
        float t3790 = t3776 * t3788;
        float t3791 = t3789 - t3790;
        float t3792 = t3775 * t3788;
        float t3793 = t3776 * t3785;
        float t3794 = t3792 + t3793;
        int t3795 = t2811 + t3777;
        float t3796 = t3780 + t3791;
        memory[176319988 + t3795] = t3796;
        int t3798 = t2811 + t3777;
        int t3799 = t3798 + 1024;
        float t3800 = t3783 + t3794;
        memory[176319988 + t3799] = t3800;
        int t3802 = t2811 + t3778;
        float t3803 = t3780 - t3791;
        memory[176319988 + t3802] = t3803;
        int t3805 = t2811 + t3778;
        int t3806 = t3805 + 1024;
        float t3807 = t3783 - t3794;
        memory[176319988 + t3806] = t3807;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3810 = 0; _pr3810 < 512; _pr3810++) {
        float t3811 = (float)_pr3810;
        float t3812 = (t3811 * 0.00390625);
        float t3813 = metal::floor(t3812);
        float t3814 = t3813 * 256.0;
        float t3815 = t3811 - t3814;
        float t3816 = t3813 * 512.0;
        float t3817 = t3816 + t3815;
        float t3818 = t3817 + 256.0;
        float t3819 = 6.283185 * t3815;
        float t3820 = (t3819 * 0.001953125);
        float t3821 = metal::cos(t3820);
        float t3822 = metal::sin(t3820);
        int t3823 = (int)t3817;
        int t3824 = (int)t3818;
        int t3825 = t2811 + t3823;
        float t3826 = memory[176319988 + t3825];
        int t3827 = t2811 + t3823;
        int t3828 = t3827 + 1024;
        float t3829 = memory[176319988 + t3828];
        int t3830 = t2811 + t3824;
        float t3831 = memory[176319988 + t3830];
        int t3832 = t2811 + t3824;
        int t3833 = t3832 + 1024;
        float t3834 = memory[176319988 + t3833];
        float t3835 = t3821 * t3831;
        float t3836 = t3822 * t3834;
        float t3837 = t3835 - t3836;
        float t3838 = t3821 * t3834;
        float t3839 = t3822 * t3831;
        float t3840 = t3838 + t3839;
        int t3841 = t2811 + t3823;
        float t3842 = t3826 + t3837;
        memory[176319988 + t3841] = t3842;
        int t3844 = t2811 + t3823;
        int t3845 = t3844 + 1024;
        float t3846 = t3829 + t3840;
        memory[176319988 + t3845] = t3846;
        int t3848 = t2811 + t3824;
        float t3849 = t3826 - t3837;
        memory[176319988 + t3848] = t3849;
        int t3851 = t2811 + t3824;
        int t3852 = t3851 + 1024;
        float t3853 = t3829 - t3840;
        memory[176319988 + t3852] = t3853;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3856 = 0; _pr3856 < 512; _pr3856++) {
        float t3857 = (float)_pr3856;
        float t3858 = (t3857 * 0.001953125);
        float t3859 = metal::floor(t3858);
        float t3860 = t3859 * 512.0;
        float t3861 = t3857 - t3860;
        float t3862 = t3859 * 1024.0;
        float t3863 = t3862 + t3861;
        float t3864 = t3863 + 512.0;
        float t3865 = 6.283185 * t3861;
        float t3866 = (t3865 * 0.0009765625);
        float t3867 = metal::cos(t3866);
        float t3868 = metal::sin(t3866);
        int t3869 = (int)t3863;
        int t3870 = (int)t3864;
        int t3871 = t2811 + t3869;
        float t3872 = memory[176319988 + t3871];
        int t3873 = t2811 + t3869;
        int t3874 = t3873 + 1024;
        float t3875 = memory[176319988 + t3874];
        int t3876 = t2811 + t3870;
        float t3877 = memory[176319988 + t3876];
        int t3878 = t2811 + t3870;
        int t3879 = t3878 + 1024;
        float t3880 = memory[176319988 + t3879];
        float t3881 = t3867 * t3877;
        float t3882 = t3868 * t3880;
        float t3883 = t3881 - t3882;
        float t3884 = t3867 * t3880;
        float t3885 = t3868 * t3877;
        float t3886 = t3884 + t3885;
        int t3887 = t2811 + t3869;
        float t3888 = t3872 + t3883;
        memory[176319988 + t3887] = t3888;
        int t3890 = t2811 + t3869;
        int t3891 = t3890 + 1024;
        float t3892 = t3875 + t3886;
        memory[176319988 + t3891] = t3892;
        int t3894 = t2811 + t3870;
        float t3895 = t3872 - t3883;
        memory[176319988 + t3894] = t3895;
        int t3897 = t2811 + t3870;
        int t3898 = t3897 + 1024;
        float t3899 = t3875 - t3886;
        memory[176319988 + t3898] = t3899;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3902 = 0; _pr3902 < 1024; _pr3902++) {
        int t3903 = t2811 + _pr3902;
        float t3904 = memory[176319988 + t3903];
        float t3905 = t3904 * 1.9036306e-06;
        float t3906 = memory[48884 + (int)_pr3902];
        int t3907 = t2812 + _pr3902;
        float t3908 = t3905 * t3906;
        memory[83996148 + t3907] = t3908;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t3912 = t[14*frameCount + id] > 0.0;
    if (t3912) {
      for (uint _pr3914 = 0; _pr3914 < 1024; _pr3914++) {
        int t3915 = t2812 + _pr3914;
        memory[50441716 + t3915] = 0.0;
        int t3917 = t2812 + _pr3914;
        memory[83996148 + t3917] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3938), value: global(3938)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1574) - handled in variable access */
    int t3921 = id;
    float t3922 = 0.0;
    for (uint t3923 = 0; t3923 < 1024; t3923++) {
      float t3924 = (float)t3923;
      float t3925 = (float)t3921;
      float t3926 = t3925 + t3924;
      int t3927 = 1023 - t3923;
      float t3928 = frameCount - 1.0;
      float t3929 = metal::min(t3926, t3928);
      int t3930 = (int)t3929;
      int t3931 = t3930 * 1024;
      int t3932 = t3931 + t3927;
      float t3933 = memory[50441716 + t3932];
      float t3934 = t3926 < frameCount;
      float t3935 = metal::select(0.0, t3933, t3934 > 0.0);
      float t3936 = t3922 + t3935;
      t3922 = t3936;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[17*frameCount + id] = (t3922 * 0.0013797212);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3956), value: global(3956)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1574) - handled in variable access */
    int t3939 = id;
    float t3940 = 0.0;
    for (uint t3941 = 0; t3941 < 1024; t3941++) {
      float t3942 = (float)t3941;
      float t3943 = (float)t3939;
      float t3944 = t3943 + t3942;
      int t3945 = 1023 - t3941;
      float t3946 = frameCount - 1.0;
      float t3947 = metal::min(t3944, t3946);
      int t3948 = (int)t3947;
      int t3949 = t3948 * 1024;
      int t3950 = t3949 + t3945;
      float t3951 = memory[83996148 + t3950];
      float t3952 = t3944 < frameCount;
      float t3953 = metal::select(0.0, t3951, t3952 > 0.0);
      float t3954 = t3940 + t3953;
      t3940 = t3954;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[18*frameCount + id] = (t3940 * 0.0013797212);
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
  float t5721 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5721)) {
    /* loadGlobal(534) - handled in variable access */
    int t3957 = id;
    int t3958 = t3957 / 61;
    uint _frameIndex = (uint)(t3958);
    int t3959 = t3958 * 61;
    int t3960 = t3957 - t3959;
    float t3961 = (t[12*frameCount + _frameIndex] * 3.7252903e-09);
    float t3962 = -0.5 * t3961;
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
    /* loadGlobal(513) - handled in variable access */
    /* loadGlobal(512) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    int t3963 = id;
    int t3964 = t3963 * 1024;
    int t3965 = t3963 * 257;
    int t3966 = t3963 * 1024;
    float t3967 = t[11*frameCount + id] == 0.0;
    if (t3967) {
      for (uint _pr3969 = 0; _pr3969 < 257; _pr3969++) {
        int t3970 = t3965 + _pr3969;
        float t3971 = memory[37809652 + t3970];
        int t3972 = t3965 + _pr3969;
        float t3973 = memory[42020340 + t3972];
        int t3974 = t3964 + _pr3969;
        float t3975 = memory[4255220 + t3974];
        int t3976 = t3964 + _pr3969;
        int t3977 = t3976 + 512;
        float t3978 = memory[4255220 + t3977];
        int t3979 = t3964 + _pr3969;
        float t3980 = memory[21032436 + t3979];
        int t3981 = t3964 + _pr3969;
        int t3982 = t3981 + 512;
        float t3983 = memory[21032436 + t3982];
        float t3984 = t3971 - t3973;
        float t3985 = 2.0 * t3984;
        float t3986 = t3985 * 3.0517578e-05;
        float t3987 = t3971 - t3973;
        float t3988 = -2.0 * t3987;
        float t3989 = t3988 * 3.0517578e-05;
        float t3990 = metal::max(t3971, 1e-08);
        float t3991 = metal::max(t3973, 1e-08);
        float t3992 = t3986 * t3975;
        float t3993 = t3992 / t3990;
        float t3994 = t3986 * t3978;
        float t3995 = t3994 / t3990;
        float t3996 = t3989 * t3980;
        float t3997 = t3996 / t3991;
        float t3998 = t3989 * t3983;
        float t3999 = t3998 / t3991;
        int t4000 = t3966 + _pr3969;
        memory[50441716 + t4000] = t3993;
        int t4002 = t3966 + _pr3969;
        int t4003 = t4002 + 512;
        memory[50441716 + t4003] = t3995;
        int t4005 = t3966 + _pr3969;
        memory[83996148 + t4005] = t3997;
        int t4007 = t3966 + _pr3969;
        int t4008 = t4007 + 512;
        memory[83996148 + t4008] = t3999;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4011 = 0; _pr4011 < 255; _pr4011++) {
        int t4012 = _pr4011 + 257;
        int t4013 = t3966 + t4012;
        memory[50441716 + t4013] = 0.0;
        int t4015 = t3966 + t4012;
        int t4016 = t4015 + 512;
        memory[50441716 + t4016] = 0.0;
        int t4018 = t3966 + t4012;
        memory[83996148 + t4018] = 0.0;
        int t4020 = t3966 + t4012;
        int t4021 = t4020 + 512;
        memory[83996148 + t4021] = 0.0;
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
    /* loadGlobal(513) - handled in variable access */
    int t4025 = id;
    int t4026 = t4025 * 1024;
    int t4027 = t4025 * 512;
    float t4028 = t[11*frameCount + id] == 0.0;
    if (t4028) {
      for (uint t4030 = 0; t4030 < 512; t4030++) {
        float t4031 = (float)t4030;
        float t4032 = (t4031 - metal::floor(t4031 / 2.0) * 2.0);
        float t4033 = t4032;
        float t4034 = (t4031 * 0.5);
        float t4035 = metal::floor(t4034);
        float t4036 = t4033 * 2.0;
        float t4037 = (t4035 - metal::floor(t4035 / 2.0) * 2.0);
        float t4038 = t4036 + t4037;
        float t4039 = (t4035 * 0.5);
        float t4040 = metal::floor(t4039);
        float t4041 = t4038 * 2.0;
        float t4042 = (t4040 - metal::floor(t4040 / 2.0) * 2.0);
        float t4043 = t4041 + t4042;
        float t4044 = (t4040 * 0.5);
        float t4045 = metal::floor(t4044);
        float t4046 = t4043 * 2.0;
        float t4047 = (t4045 - metal::floor(t4045 / 2.0) * 2.0);
        float t4048 = t4046 + t4047;
        float t4049 = (t4045 * 0.5);
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
        float t4076 = (float)t4030;
        float t4077 = t4076 < t4073;
        int t4078 = (int)t4073;
        int t4079 = t4026 + t4030;
        float t4080 = memory[50441716 + t4079];
        int t4081 = t4026 + t4030;
        int t4082 = t4081 + 512;
        float t4083 = memory[50441716 + t4082];
        int t4084 = t4026 + t4078;
        float t4085 = memory[50441716 + t4084];
        int t4086 = t4026 + t4078;
        int t4087 = t4086 + 512;
        float t4088 = memory[50441716 + t4087];
        float t4089 = metal::select(t4080, t4085, t4077 > 0.0);
        float t4090 = metal::select(t4083, t4088, t4077 > 0.0);
        float t4091 = metal::select(t4085, t4080, t4077 > 0.0);
        float t4092 = metal::select(t4088, t4083, t4077 > 0.0);
        int t4093 = t4026 + t4030;
        memory[50441716 + t4093] = t4089;
        int t4095 = t4026 + t4030;
        int t4096 = t4095 + 512;
        memory[50441716 + t4096] = t4090;
        int t4098 = t4026 + t4078;
        memory[50441716 + t4098] = t4091;
        int t4100 = t4026 + t4078;
        int t4101 = t4100 + 512;
        memory[50441716 + t4101] = t4092;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4104 = 0; _pr4104 < 256; _pr4104++) {
        float t4105 = (float)_pr4104;
        float t4106 = t4105;
        float t4107 = metal::floor(t4106);
        float t4108 = t4107;
        float t4109 = t4105 - t4108;
        float t4110 = t4107 * 2.0;
        float t4111 = t4110 + t4109;
        float t4112 = t4111 + 1.0;
        float t4113 = 6.283185 * t4109;
        float t4114 = (t4113 * 0.5);
        float t4115 = metal::cos(t4114);
        float t4116 = metal::sin(t4114);
        int t4117 = (int)t4111;
        int t4118 = (int)t4112;
        int t4119 = t4026 + t4117;
        float t4120 = memory[50441716 + t4119];
        int t4121 = t4026 + t4117;
        int t4122 = t4121 + 512;
        float t4123 = memory[50441716 + t4122];
        int t4124 = t4026 + t4118;
        float t4125 = memory[50441716 + t4124];
        int t4126 = t4026 + t4118;
        int t4127 = t4126 + 512;
        float t4128 = memory[50441716 + t4127];
        float t4129 = t4115 * t4125;
        float t4130 = t4116 * t4128;
        float t4131 = t4129 - t4130;
        float t4132 = t4115 * t4128;
        float t4133 = t4116 * t4125;
        float t4134 = t4132 + t4133;
        int t4135 = t4026 + t4117;
        float t4136 = t4120 + t4131;
        memory[50441716 + t4135] = t4136;
        int t4138 = t4026 + t4117;
        int t4139 = t4138 + 512;
        float t4140 = t4123 + t4134;
        memory[50441716 + t4139] = t4140;
        int t4142 = t4026 + t4118;
        float t4143 = t4120 - t4131;
        memory[50441716 + t4142] = t4143;
        int t4145 = t4026 + t4118;
        int t4146 = t4145 + 512;
        float t4147 = t4123 - t4134;
        memory[50441716 + t4146] = t4147;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4150 = 0; _pr4150 < 256; _pr4150++) {
        float t4151 = (float)_pr4150;
        float t4152 = (t4151 * 0.5);
        float t4153 = metal::floor(t4152);
        float t4154 = t4153 * 2.0;
        float t4155 = t4151 - t4154;
        float t4156 = t4153 * 4.0;
        float t4157 = t4156 + t4155;
        float t4158 = t4157 + 2.0;
        float t4159 = 6.283185 * t4155;
        float t4160 = (t4159 * 0.25);
        float t4161 = metal::cos(t4160);
        float t4162 = metal::sin(t4160);
        int t4163 = (int)t4157;
        int t4164 = (int)t4158;
        int t4165 = t4026 + t4163;
        float t4166 = memory[50441716 + t4165];
        int t4167 = t4026 + t4163;
        int t4168 = t4167 + 512;
        float t4169 = memory[50441716 + t4168];
        int t4170 = t4026 + t4164;
        float t4171 = memory[50441716 + t4170];
        int t4172 = t4026 + t4164;
        int t4173 = t4172 + 512;
        float t4174 = memory[50441716 + t4173];
        float t4175 = t4161 * t4171;
        float t4176 = t4162 * t4174;
        float t4177 = t4175 - t4176;
        float t4178 = t4161 * t4174;
        float t4179 = t4162 * t4171;
        float t4180 = t4178 + t4179;
        int t4181 = t4026 + t4163;
        float t4182 = t4166 + t4177;
        memory[50441716 + t4181] = t4182;
        int t4184 = t4026 + t4163;
        int t4185 = t4184 + 512;
        float t4186 = t4169 + t4180;
        memory[50441716 + t4185] = t4186;
        int t4188 = t4026 + t4164;
        float t4189 = t4166 - t4177;
        memory[50441716 + t4188] = t4189;
        int t4191 = t4026 + t4164;
        int t4192 = t4191 + 512;
        float t4193 = t4169 - t4180;
        memory[50441716 + t4192] = t4193;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4196 = 0; _pr4196 < 256; _pr4196++) {
        float t4197 = (float)_pr4196;
        float t4198 = (t4197 * 0.25);
        float t4199 = metal::floor(t4198);
        float t4200 = t4199 * 4.0;
        float t4201 = t4197 - t4200;
        float t4202 = t4199 * 8.0;
        float t4203 = t4202 + t4201;
        float t4204 = t4203 + 4.0;
        float t4205 = 6.283185 * t4201;
        float t4206 = (t4205 * 0.125);
        float t4207 = metal::cos(t4206);
        float t4208 = metal::sin(t4206);
        int t4209 = (int)t4203;
        int t4210 = (int)t4204;
        int t4211 = t4026 + t4209;
        float t4212 = memory[50441716 + t4211];
        int t4213 = t4026 + t4209;
        int t4214 = t4213 + 512;
        float t4215 = memory[50441716 + t4214];
        int t4216 = t4026 + t4210;
        float t4217 = memory[50441716 + t4216];
        int t4218 = t4026 + t4210;
        int t4219 = t4218 + 512;
        float t4220 = memory[50441716 + t4219];
        float t4221 = t4207 * t4217;
        float t4222 = t4208 * t4220;
        float t4223 = t4221 - t4222;
        float t4224 = t4207 * t4220;
        float t4225 = t4208 * t4217;
        float t4226 = t4224 + t4225;
        int t4227 = t4026 + t4209;
        float t4228 = t4212 + t4223;
        memory[50441716 + t4227] = t4228;
        int t4230 = t4026 + t4209;
        int t4231 = t4230 + 512;
        float t4232 = t4215 + t4226;
        memory[50441716 + t4231] = t4232;
        int t4234 = t4026 + t4210;
        float t4235 = t4212 - t4223;
        memory[50441716 + t4234] = t4235;
        int t4237 = t4026 + t4210;
        int t4238 = t4237 + 512;
        float t4239 = t4215 - t4226;
        memory[50441716 + t4238] = t4239;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4242 = 0; _pr4242 < 256; _pr4242++) {
        float t4243 = (float)_pr4242;
        float t4244 = (t4243 * 0.125);
        float t4245 = metal::floor(t4244);
        float t4246 = t4245 * 8.0;
        float t4247 = t4243 - t4246;
        float t4248 = t4245 * 16.0;
        float t4249 = t4248 + t4247;
        float t4250 = t4249 + 8.0;
        float t4251 = 6.283185 * t4247;
        float t4252 = (t4251 * 0.0625);
        float t4253 = metal::cos(t4252);
        float t4254 = metal::sin(t4252);
        int t4255 = (int)t4249;
        int t4256 = (int)t4250;
        int t4257 = t4026 + t4255;
        float t4258 = memory[50441716 + t4257];
        int t4259 = t4026 + t4255;
        int t4260 = t4259 + 512;
        float t4261 = memory[50441716 + t4260];
        int t4262 = t4026 + t4256;
        float t4263 = memory[50441716 + t4262];
        int t4264 = t4026 + t4256;
        int t4265 = t4264 + 512;
        float t4266 = memory[50441716 + t4265];
        float t4267 = t4253 * t4263;
        float t4268 = t4254 * t4266;
        float t4269 = t4267 - t4268;
        float t4270 = t4253 * t4266;
        float t4271 = t4254 * t4263;
        float t4272 = t4270 + t4271;
        int t4273 = t4026 + t4255;
        float t4274 = t4258 + t4269;
        memory[50441716 + t4273] = t4274;
        int t4276 = t4026 + t4255;
        int t4277 = t4276 + 512;
        float t4278 = t4261 + t4272;
        memory[50441716 + t4277] = t4278;
        int t4280 = t4026 + t4256;
        float t4281 = t4258 - t4269;
        memory[50441716 + t4280] = t4281;
        int t4283 = t4026 + t4256;
        int t4284 = t4283 + 512;
        float t4285 = t4261 - t4272;
        memory[50441716 + t4284] = t4285;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4288 = 0; _pr4288 < 256; _pr4288++) {
        float t4289 = (float)_pr4288;
        float t4290 = (t4289 * 0.0625);
        float t4291 = metal::floor(t4290);
        float t4292 = t4291 * 16.0;
        float t4293 = t4289 - t4292;
        float t4294 = t4291 * 32.0;
        float t4295 = t4294 + t4293;
        float t4296 = t4295 + 16.0;
        float t4297 = 6.283185 * t4293;
        float t4298 = (t4297 * 0.03125);
        float t4299 = metal::cos(t4298);
        float t4300 = metal::sin(t4298);
        int t4301 = (int)t4295;
        int t4302 = (int)t4296;
        int t4303 = t4026 + t4301;
        float t4304 = memory[50441716 + t4303];
        int t4305 = t4026 + t4301;
        int t4306 = t4305 + 512;
        float t4307 = memory[50441716 + t4306];
        int t4308 = t4026 + t4302;
        float t4309 = memory[50441716 + t4308];
        int t4310 = t4026 + t4302;
        int t4311 = t4310 + 512;
        float t4312 = memory[50441716 + t4311];
        float t4313 = t4299 * t4309;
        float t4314 = t4300 * t4312;
        float t4315 = t4313 - t4314;
        float t4316 = t4299 * t4312;
        float t4317 = t4300 * t4309;
        float t4318 = t4316 + t4317;
        int t4319 = t4026 + t4301;
        float t4320 = t4304 + t4315;
        memory[50441716 + t4319] = t4320;
        int t4322 = t4026 + t4301;
        int t4323 = t4322 + 512;
        float t4324 = t4307 + t4318;
        memory[50441716 + t4323] = t4324;
        int t4326 = t4026 + t4302;
        float t4327 = t4304 - t4315;
        memory[50441716 + t4326] = t4327;
        int t4329 = t4026 + t4302;
        int t4330 = t4329 + 512;
        float t4331 = t4307 - t4318;
        memory[50441716 + t4330] = t4331;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4334 = 0; _pr4334 < 256; _pr4334++) {
        float t4335 = (float)_pr4334;
        float t4336 = (t4335 * 0.03125);
        float t4337 = metal::floor(t4336);
        float t4338 = t4337 * 32.0;
        float t4339 = t4335 - t4338;
        float t4340 = t4337 * 64.0;
        float t4341 = t4340 + t4339;
        float t4342 = t4341 + 32.0;
        float t4343 = 6.283185 * t4339;
        float t4344 = (t4343 * 0.015625);
        float t4345 = metal::cos(t4344);
        float t4346 = metal::sin(t4344);
        int t4347 = (int)t4341;
        int t4348 = (int)t4342;
        int t4349 = t4026 + t4347;
        float t4350 = memory[50441716 + t4349];
        int t4351 = t4026 + t4347;
        int t4352 = t4351 + 512;
        float t4353 = memory[50441716 + t4352];
        int t4354 = t4026 + t4348;
        float t4355 = memory[50441716 + t4354];
        int t4356 = t4026 + t4348;
        int t4357 = t4356 + 512;
        float t4358 = memory[50441716 + t4357];
        float t4359 = t4345 * t4355;
        float t4360 = t4346 * t4358;
        float t4361 = t4359 - t4360;
        float t4362 = t4345 * t4358;
        float t4363 = t4346 * t4355;
        float t4364 = t4362 + t4363;
        int t4365 = t4026 + t4347;
        float t4366 = t4350 + t4361;
        memory[50441716 + t4365] = t4366;
        int t4368 = t4026 + t4347;
        int t4369 = t4368 + 512;
        float t4370 = t4353 + t4364;
        memory[50441716 + t4369] = t4370;
        int t4372 = t4026 + t4348;
        float t4373 = t4350 - t4361;
        memory[50441716 + t4372] = t4373;
        int t4375 = t4026 + t4348;
        int t4376 = t4375 + 512;
        float t4377 = t4353 - t4364;
        memory[50441716 + t4376] = t4377;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4380 = 0; _pr4380 < 256; _pr4380++) {
        float t4381 = (float)_pr4380;
        float t4382 = (t4381 * 0.015625);
        float t4383 = metal::floor(t4382);
        float t4384 = t4383 * 64.0;
        float t4385 = t4381 - t4384;
        float t4386 = t4383 * 128.0;
        float t4387 = t4386 + t4385;
        float t4388 = t4387 + 64.0;
        float t4389 = 6.283185 * t4385;
        float t4390 = (t4389 * 0.0078125);
        float t4391 = metal::cos(t4390);
        float t4392 = metal::sin(t4390);
        int t4393 = (int)t4387;
        int t4394 = (int)t4388;
        int t4395 = t4026 + t4393;
        float t4396 = memory[50441716 + t4395];
        int t4397 = t4026 + t4393;
        int t4398 = t4397 + 512;
        float t4399 = memory[50441716 + t4398];
        int t4400 = t4026 + t4394;
        float t4401 = memory[50441716 + t4400];
        int t4402 = t4026 + t4394;
        int t4403 = t4402 + 512;
        float t4404 = memory[50441716 + t4403];
        float t4405 = t4391 * t4401;
        float t4406 = t4392 * t4404;
        float t4407 = t4405 - t4406;
        float t4408 = t4391 * t4404;
        float t4409 = t4392 * t4401;
        float t4410 = t4408 + t4409;
        int t4411 = t4026 + t4393;
        float t4412 = t4396 + t4407;
        memory[50441716 + t4411] = t4412;
        int t4414 = t4026 + t4393;
        int t4415 = t4414 + 512;
        float t4416 = t4399 + t4410;
        memory[50441716 + t4415] = t4416;
        int t4418 = t4026 + t4394;
        float t4419 = t4396 - t4407;
        memory[50441716 + t4418] = t4419;
        int t4421 = t4026 + t4394;
        int t4422 = t4421 + 512;
        float t4423 = t4399 - t4410;
        memory[50441716 + t4422] = t4423;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4426 = 0; _pr4426 < 256; _pr4426++) {
        float t4427 = (float)_pr4426;
        float t4428 = (t4427 * 0.0078125);
        float t4429 = metal::floor(t4428);
        float t4430 = t4429 * 128.0;
        float t4431 = t4427 - t4430;
        float t4432 = t4429 * 256.0;
        float t4433 = t4432 + t4431;
        float t4434 = t4433 + 128.0;
        float t4435 = 6.283185 * t4431;
        float t4436 = (t4435 * 0.00390625);
        float t4437 = metal::cos(t4436);
        float t4438 = metal::sin(t4436);
        int t4439 = (int)t4433;
        int t4440 = (int)t4434;
        int t4441 = t4026 + t4439;
        float t4442 = memory[50441716 + t4441];
        int t4443 = t4026 + t4439;
        int t4444 = t4443 + 512;
        float t4445 = memory[50441716 + t4444];
        int t4446 = t4026 + t4440;
        float t4447 = memory[50441716 + t4446];
        int t4448 = t4026 + t4440;
        int t4449 = t4448 + 512;
        float t4450 = memory[50441716 + t4449];
        float t4451 = t4437 * t4447;
        float t4452 = t4438 * t4450;
        float t4453 = t4451 - t4452;
        float t4454 = t4437 * t4450;
        float t4455 = t4438 * t4447;
        float t4456 = t4454 + t4455;
        int t4457 = t4026 + t4439;
        float t4458 = t4442 + t4453;
        memory[50441716 + t4457] = t4458;
        int t4460 = t4026 + t4439;
        int t4461 = t4460 + 512;
        float t4462 = t4445 + t4456;
        memory[50441716 + t4461] = t4462;
        int t4464 = t4026 + t4440;
        float t4465 = t4442 - t4453;
        memory[50441716 + t4464] = t4465;
        int t4467 = t4026 + t4440;
        int t4468 = t4467 + 512;
        float t4469 = t4445 - t4456;
        memory[50441716 + t4468] = t4469;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4472 = 0; _pr4472 < 256; _pr4472++) {
        float t4473 = (float)_pr4472;
        float t4474 = (t4473 * 0.00390625);
        float t4475 = metal::floor(t4474);
        float t4476 = t4475 * 256.0;
        float t4477 = t4473 - t4476;
        float t4478 = t4475 * 512.0;
        float t4479 = t4478 + t4477;
        float t4480 = t4479 + 256.0;
        float t4481 = 6.283185 * t4477;
        float t4482 = (t4481 * 0.001953125);
        float t4483 = metal::cos(t4482);
        float t4484 = metal::sin(t4482);
        int t4485 = (int)t4479;
        int t4486 = (int)t4480;
        int t4487 = t4026 + t4485;
        float t4488 = memory[50441716 + t4487];
        int t4489 = t4026 + t4485;
        int t4490 = t4489 + 512;
        float t4491 = memory[50441716 + t4490];
        int t4492 = t4026 + t4486;
        float t4493 = memory[50441716 + t4492];
        int t4494 = t4026 + t4486;
        int t4495 = t4494 + 512;
        float t4496 = memory[50441716 + t4495];
        float t4497 = t4483 * t4493;
        float t4498 = t4484 * t4496;
        float t4499 = t4497 - t4498;
        float t4500 = t4483 * t4496;
        float t4501 = t4484 * t4493;
        float t4502 = t4500 + t4501;
        int t4503 = t4026 + t4485;
        float t4504 = t4488 + t4499;
        memory[50441716 + t4503] = t4504;
        int t4506 = t4026 + t4485;
        int t4507 = t4506 + 512;
        float t4508 = t4491 + t4502;
        memory[50441716 + t4507] = t4508;
        int t4510 = t4026 + t4486;
        float t4511 = t4488 - t4499;
        memory[50441716 + t4510] = t4511;
        int t4513 = t4026 + t4486;
        int t4514 = t4513 + 512;
        float t4515 = t4491 - t4502;
        memory[50441716 + t4514] = t4515;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4518 = 0; _pr4518 < 512; _pr4518++) {
        int t4519 = t4026 + _pr4518;
        float t4520 = memory[50441716 + t4519];
        float t4521 = t4520 * 7.599708e-06;
        float t4522 = memory[25460 + (int)_pr4518];
        int t4523 = t4027 + _pr4518;
        float t4524 = t4521 * t4522;
        memory[117550580 + t4523] = t4524;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t4527 = 0; t4527 < 512; t4527++) {
        float t4528 = (float)t4527;
        float t4529 = (t4528 - metal::floor(t4528 / 2.0) * 2.0);
        float t4530 = t4529;
        float t4531 = (t4528 * 0.5);
        float t4532 = metal::floor(t4531);
        float t4533 = t4530 * 2.0;
        float t4534 = (t4532 - metal::floor(t4532 / 2.0) * 2.0);
        float t4535 = t4533 + t4534;
        float t4536 = (t4532 * 0.5);
        float t4537 = metal::floor(t4536);
        float t4538 = t4535 * 2.0;
        float t4539 = (t4537 - metal::floor(t4537 / 2.0) * 2.0);
        float t4540 = t4538 + t4539;
        float t4541 = (t4537 * 0.5);
        float t4542 = metal::floor(t4541);
        float t4543 = t4540 * 2.0;
        float t4544 = (t4542 - metal::floor(t4542 / 2.0) * 2.0);
        float t4545 = t4543 + t4544;
        float t4546 = (t4542 * 0.5);
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
        float t4573 = (float)t4527;
        float t4574 = t4573 < t4570;
        int t4575 = (int)t4570;
        int t4576 = t4026 + t4527;
        float t4577 = memory[83996148 + t4576];
        int t4578 = t4026 + t4527;
        int t4579 = t4578 + 512;
        float t4580 = memory[83996148 + t4579];
        int t4581 = t4026 + t4575;
        float t4582 = memory[83996148 + t4581];
        int t4583 = t4026 + t4575;
        int t4584 = t4583 + 512;
        float t4585 = memory[83996148 + t4584];
        float t4586 = metal::select(t4577, t4582, t4574 > 0.0);
        float t4587 = metal::select(t4580, t4585, t4574 > 0.0);
        float t4588 = metal::select(t4582, t4577, t4574 > 0.0);
        float t4589 = metal::select(t4585, t4580, t4574 > 0.0);
        int t4590 = t4026 + t4527;
        memory[83996148 + t4590] = t4586;
        int t4592 = t4026 + t4527;
        int t4593 = t4592 + 512;
        memory[83996148 + t4593] = t4587;
        int t4595 = t4026 + t4575;
        memory[83996148 + t4595] = t4588;
        int t4597 = t4026 + t4575;
        int t4598 = t4597 + 512;
        memory[83996148 + t4598] = t4589;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4601 = 0; _pr4601 < 256; _pr4601++) {
        float t4602 = (float)_pr4601;
        float t4603 = t4602;
        float t4604 = metal::floor(t4603);
        float t4605 = t4604;
        float t4606 = t4602 - t4605;
        float t4607 = t4604 * 2.0;
        float t4608 = t4607 + t4606;
        float t4609 = t4608 + 1.0;
        float t4610 = 6.283185 * t4606;
        float t4611 = (t4610 * 0.5);
        float t4612 = metal::cos(t4611);
        float t4613 = metal::sin(t4611);
        int t4614 = (int)t4608;
        int t4615 = (int)t4609;
        int t4616 = t4026 + t4614;
        float t4617 = memory[83996148 + t4616];
        int t4618 = t4026 + t4614;
        int t4619 = t4618 + 512;
        float t4620 = memory[83996148 + t4619];
        int t4621 = t4026 + t4615;
        float t4622 = memory[83996148 + t4621];
        int t4623 = t4026 + t4615;
        int t4624 = t4623 + 512;
        float t4625 = memory[83996148 + t4624];
        float t4626 = t4612 * t4622;
        float t4627 = t4613 * t4625;
        float t4628 = t4626 - t4627;
        float t4629 = t4612 * t4625;
        float t4630 = t4613 * t4622;
        float t4631 = t4629 + t4630;
        int t4632 = t4026 + t4614;
        float t4633 = t4617 + t4628;
        memory[83996148 + t4632] = t4633;
        int t4635 = t4026 + t4614;
        int t4636 = t4635 + 512;
        float t4637 = t4620 + t4631;
        memory[83996148 + t4636] = t4637;
        int t4639 = t4026 + t4615;
        float t4640 = t4617 - t4628;
        memory[83996148 + t4639] = t4640;
        int t4642 = t4026 + t4615;
        int t4643 = t4642 + 512;
        float t4644 = t4620 - t4631;
        memory[83996148 + t4643] = t4644;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4647 = 0; _pr4647 < 256; _pr4647++) {
        float t4648 = (float)_pr4647;
        float t4649 = (t4648 * 0.5);
        float t4650 = metal::floor(t4649);
        float t4651 = t4650 * 2.0;
        float t4652 = t4648 - t4651;
        float t4653 = t4650 * 4.0;
        float t4654 = t4653 + t4652;
        float t4655 = t4654 + 2.0;
        float t4656 = 6.283185 * t4652;
        float t4657 = (t4656 * 0.25);
        float t4658 = metal::cos(t4657);
        float t4659 = metal::sin(t4657);
        int t4660 = (int)t4654;
        int t4661 = (int)t4655;
        int t4662 = t4026 + t4660;
        float t4663 = memory[83996148 + t4662];
        int t4664 = t4026 + t4660;
        int t4665 = t4664 + 512;
        float t4666 = memory[83996148 + t4665];
        int t4667 = t4026 + t4661;
        float t4668 = memory[83996148 + t4667];
        int t4669 = t4026 + t4661;
        int t4670 = t4669 + 512;
        float t4671 = memory[83996148 + t4670];
        float t4672 = t4658 * t4668;
        float t4673 = t4659 * t4671;
        float t4674 = t4672 - t4673;
        float t4675 = t4658 * t4671;
        float t4676 = t4659 * t4668;
        float t4677 = t4675 + t4676;
        int t4678 = t4026 + t4660;
        float t4679 = t4663 + t4674;
        memory[83996148 + t4678] = t4679;
        int t4681 = t4026 + t4660;
        int t4682 = t4681 + 512;
        float t4683 = t4666 + t4677;
        memory[83996148 + t4682] = t4683;
        int t4685 = t4026 + t4661;
        float t4686 = t4663 - t4674;
        memory[83996148 + t4685] = t4686;
        int t4688 = t4026 + t4661;
        int t4689 = t4688 + 512;
        float t4690 = t4666 - t4677;
        memory[83996148 + t4689] = t4690;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4693 = 0; _pr4693 < 256; _pr4693++) {
        float t4694 = (float)_pr4693;
        float t4695 = (t4694 * 0.25);
        float t4696 = metal::floor(t4695);
        float t4697 = t4696 * 4.0;
        float t4698 = t4694 - t4697;
        float t4699 = t4696 * 8.0;
        float t4700 = t4699 + t4698;
        float t4701 = t4700 + 4.0;
        float t4702 = 6.283185 * t4698;
        float t4703 = (t4702 * 0.125);
        float t4704 = metal::cos(t4703);
        float t4705 = metal::sin(t4703);
        int t4706 = (int)t4700;
        int t4707 = (int)t4701;
        int t4708 = t4026 + t4706;
        float t4709 = memory[83996148 + t4708];
        int t4710 = t4026 + t4706;
        int t4711 = t4710 + 512;
        float t4712 = memory[83996148 + t4711];
        int t4713 = t4026 + t4707;
        float t4714 = memory[83996148 + t4713];
        int t4715 = t4026 + t4707;
        int t4716 = t4715 + 512;
        float t4717 = memory[83996148 + t4716];
        float t4718 = t4704 * t4714;
        float t4719 = t4705 * t4717;
        float t4720 = t4718 - t4719;
        float t4721 = t4704 * t4717;
        float t4722 = t4705 * t4714;
        float t4723 = t4721 + t4722;
        int t4724 = t4026 + t4706;
        float t4725 = t4709 + t4720;
        memory[83996148 + t4724] = t4725;
        int t4727 = t4026 + t4706;
        int t4728 = t4727 + 512;
        float t4729 = t4712 + t4723;
        memory[83996148 + t4728] = t4729;
        int t4731 = t4026 + t4707;
        float t4732 = t4709 - t4720;
        memory[83996148 + t4731] = t4732;
        int t4734 = t4026 + t4707;
        int t4735 = t4734 + 512;
        float t4736 = t4712 - t4723;
        memory[83996148 + t4735] = t4736;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4739 = 0; _pr4739 < 256; _pr4739++) {
        float t4740 = (float)_pr4739;
        float t4741 = (t4740 * 0.125);
        float t4742 = metal::floor(t4741);
        float t4743 = t4742 * 8.0;
        float t4744 = t4740 - t4743;
        float t4745 = t4742 * 16.0;
        float t4746 = t4745 + t4744;
        float t4747 = t4746 + 8.0;
        float t4748 = 6.283185 * t4744;
        float t4749 = (t4748 * 0.0625);
        float t4750 = metal::cos(t4749);
        float t4751 = metal::sin(t4749);
        int t4752 = (int)t4746;
        int t4753 = (int)t4747;
        int t4754 = t4026 + t4752;
        float t4755 = memory[83996148 + t4754];
        int t4756 = t4026 + t4752;
        int t4757 = t4756 + 512;
        float t4758 = memory[83996148 + t4757];
        int t4759 = t4026 + t4753;
        float t4760 = memory[83996148 + t4759];
        int t4761 = t4026 + t4753;
        int t4762 = t4761 + 512;
        float t4763 = memory[83996148 + t4762];
        float t4764 = t4750 * t4760;
        float t4765 = t4751 * t4763;
        float t4766 = t4764 - t4765;
        float t4767 = t4750 * t4763;
        float t4768 = t4751 * t4760;
        float t4769 = t4767 + t4768;
        int t4770 = t4026 + t4752;
        float t4771 = t4755 + t4766;
        memory[83996148 + t4770] = t4771;
        int t4773 = t4026 + t4752;
        int t4774 = t4773 + 512;
        float t4775 = t4758 + t4769;
        memory[83996148 + t4774] = t4775;
        int t4777 = t4026 + t4753;
        float t4778 = t4755 - t4766;
        memory[83996148 + t4777] = t4778;
        int t4780 = t4026 + t4753;
        int t4781 = t4780 + 512;
        float t4782 = t4758 - t4769;
        memory[83996148 + t4781] = t4782;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4785 = 0; _pr4785 < 256; _pr4785++) {
        float t4786 = (float)_pr4785;
        float t4787 = (t4786 * 0.0625);
        float t4788 = metal::floor(t4787);
        float t4789 = t4788 * 16.0;
        float t4790 = t4786 - t4789;
        float t4791 = t4788 * 32.0;
        float t4792 = t4791 + t4790;
        float t4793 = t4792 + 16.0;
        float t4794 = 6.283185 * t4790;
        float t4795 = (t4794 * 0.03125);
        float t4796 = metal::cos(t4795);
        float t4797 = metal::sin(t4795);
        int t4798 = (int)t4792;
        int t4799 = (int)t4793;
        int t4800 = t4026 + t4798;
        float t4801 = memory[83996148 + t4800];
        int t4802 = t4026 + t4798;
        int t4803 = t4802 + 512;
        float t4804 = memory[83996148 + t4803];
        int t4805 = t4026 + t4799;
        float t4806 = memory[83996148 + t4805];
        int t4807 = t4026 + t4799;
        int t4808 = t4807 + 512;
        float t4809 = memory[83996148 + t4808];
        float t4810 = t4796 * t4806;
        float t4811 = t4797 * t4809;
        float t4812 = t4810 - t4811;
        float t4813 = t4796 * t4809;
        float t4814 = t4797 * t4806;
        float t4815 = t4813 + t4814;
        int t4816 = t4026 + t4798;
        float t4817 = t4801 + t4812;
        memory[83996148 + t4816] = t4817;
        int t4819 = t4026 + t4798;
        int t4820 = t4819 + 512;
        float t4821 = t4804 + t4815;
        memory[83996148 + t4820] = t4821;
        int t4823 = t4026 + t4799;
        float t4824 = t4801 - t4812;
        memory[83996148 + t4823] = t4824;
        int t4826 = t4026 + t4799;
        int t4827 = t4826 + 512;
        float t4828 = t4804 - t4815;
        memory[83996148 + t4827] = t4828;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4831 = 0; _pr4831 < 256; _pr4831++) {
        float t4832 = (float)_pr4831;
        float t4833 = (t4832 * 0.03125);
        float t4834 = metal::floor(t4833);
        float t4835 = t4834 * 32.0;
        float t4836 = t4832 - t4835;
        float t4837 = t4834 * 64.0;
        float t4838 = t4837 + t4836;
        float t4839 = t4838 + 32.0;
        float t4840 = 6.283185 * t4836;
        float t4841 = (t4840 * 0.015625);
        float t4842 = metal::cos(t4841);
        float t4843 = metal::sin(t4841);
        int t4844 = (int)t4838;
        int t4845 = (int)t4839;
        int t4846 = t4026 + t4844;
        float t4847 = memory[83996148 + t4846];
        int t4848 = t4026 + t4844;
        int t4849 = t4848 + 512;
        float t4850 = memory[83996148 + t4849];
        int t4851 = t4026 + t4845;
        float t4852 = memory[83996148 + t4851];
        int t4853 = t4026 + t4845;
        int t4854 = t4853 + 512;
        float t4855 = memory[83996148 + t4854];
        float t4856 = t4842 * t4852;
        float t4857 = t4843 * t4855;
        float t4858 = t4856 - t4857;
        float t4859 = t4842 * t4855;
        float t4860 = t4843 * t4852;
        float t4861 = t4859 + t4860;
        int t4862 = t4026 + t4844;
        float t4863 = t4847 + t4858;
        memory[83996148 + t4862] = t4863;
        int t4865 = t4026 + t4844;
        int t4866 = t4865 + 512;
        float t4867 = t4850 + t4861;
        memory[83996148 + t4866] = t4867;
        int t4869 = t4026 + t4845;
        float t4870 = t4847 - t4858;
        memory[83996148 + t4869] = t4870;
        int t4872 = t4026 + t4845;
        int t4873 = t4872 + 512;
        float t4874 = t4850 - t4861;
        memory[83996148 + t4873] = t4874;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4877 = 0; _pr4877 < 256; _pr4877++) {
        float t4878 = (float)_pr4877;
        float t4879 = (t4878 * 0.015625);
        float t4880 = metal::floor(t4879);
        float t4881 = t4880 * 64.0;
        float t4882 = t4878 - t4881;
        float t4883 = t4880 * 128.0;
        float t4884 = t4883 + t4882;
        float t4885 = t4884 + 64.0;
        float t4886 = 6.283185 * t4882;
        float t4887 = (t4886 * 0.0078125);
        float t4888 = metal::cos(t4887);
        float t4889 = metal::sin(t4887);
        int t4890 = (int)t4884;
        int t4891 = (int)t4885;
        int t4892 = t4026 + t4890;
        float t4893 = memory[83996148 + t4892];
        int t4894 = t4026 + t4890;
        int t4895 = t4894 + 512;
        float t4896 = memory[83996148 + t4895];
        int t4897 = t4026 + t4891;
        float t4898 = memory[83996148 + t4897];
        int t4899 = t4026 + t4891;
        int t4900 = t4899 + 512;
        float t4901 = memory[83996148 + t4900];
        float t4902 = t4888 * t4898;
        float t4903 = t4889 * t4901;
        float t4904 = t4902 - t4903;
        float t4905 = t4888 * t4901;
        float t4906 = t4889 * t4898;
        float t4907 = t4905 + t4906;
        int t4908 = t4026 + t4890;
        float t4909 = t4893 + t4904;
        memory[83996148 + t4908] = t4909;
        int t4911 = t4026 + t4890;
        int t4912 = t4911 + 512;
        float t4913 = t4896 + t4907;
        memory[83996148 + t4912] = t4913;
        int t4915 = t4026 + t4891;
        float t4916 = t4893 - t4904;
        memory[83996148 + t4915] = t4916;
        int t4918 = t4026 + t4891;
        int t4919 = t4918 + 512;
        float t4920 = t4896 - t4907;
        memory[83996148 + t4919] = t4920;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4923 = 0; _pr4923 < 256; _pr4923++) {
        float t4924 = (float)_pr4923;
        float t4925 = (t4924 * 0.0078125);
        float t4926 = metal::floor(t4925);
        float t4927 = t4926 * 128.0;
        float t4928 = t4924 - t4927;
        float t4929 = t4926 * 256.0;
        float t4930 = t4929 + t4928;
        float t4931 = t4930 + 128.0;
        float t4932 = 6.283185 * t4928;
        float t4933 = (t4932 * 0.00390625);
        float t4934 = metal::cos(t4933);
        float t4935 = metal::sin(t4933);
        int t4936 = (int)t4930;
        int t4937 = (int)t4931;
        int t4938 = t4026 + t4936;
        float t4939 = memory[83996148 + t4938];
        int t4940 = t4026 + t4936;
        int t4941 = t4940 + 512;
        float t4942 = memory[83996148 + t4941];
        int t4943 = t4026 + t4937;
        float t4944 = memory[83996148 + t4943];
        int t4945 = t4026 + t4937;
        int t4946 = t4945 + 512;
        float t4947 = memory[83996148 + t4946];
        float t4948 = t4934 * t4944;
        float t4949 = t4935 * t4947;
        float t4950 = t4948 - t4949;
        float t4951 = t4934 * t4947;
        float t4952 = t4935 * t4944;
        float t4953 = t4951 + t4952;
        int t4954 = t4026 + t4936;
        float t4955 = t4939 + t4950;
        memory[83996148 + t4954] = t4955;
        int t4957 = t4026 + t4936;
        int t4958 = t4957 + 512;
        float t4959 = t4942 + t4953;
        memory[83996148 + t4958] = t4959;
        int t4961 = t4026 + t4937;
        float t4962 = t4939 - t4950;
        memory[83996148 + t4961] = t4962;
        int t4964 = t4026 + t4937;
        int t4965 = t4964 + 512;
        float t4966 = t4942 - t4953;
        memory[83996148 + t4965] = t4966;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4969 = 0; _pr4969 < 256; _pr4969++) {
        float t4970 = (float)_pr4969;
        float t4971 = (t4970 * 0.00390625);
        float t4972 = metal::floor(t4971);
        float t4973 = t4972 * 256.0;
        float t4974 = t4970 - t4973;
        float t4975 = t4972 * 512.0;
        float t4976 = t4975 + t4974;
        float t4977 = t4976 + 256.0;
        float t4978 = 6.283185 * t4974;
        float t4979 = (t4978 * 0.001953125);
        float t4980 = metal::cos(t4979);
        float t4981 = metal::sin(t4979);
        int t4982 = (int)t4976;
        int t4983 = (int)t4977;
        int t4984 = t4026 + t4982;
        float t4985 = memory[83996148 + t4984];
        int t4986 = t4026 + t4982;
        int t4987 = t4986 + 512;
        float t4988 = memory[83996148 + t4987];
        int t4989 = t4026 + t4983;
        float t4990 = memory[83996148 + t4989];
        int t4991 = t4026 + t4983;
        int t4992 = t4991 + 512;
        float t4993 = memory[83996148 + t4992];
        float t4994 = t4980 * t4990;
        float t4995 = t4981 * t4993;
        float t4996 = t4994 - t4995;
        float t4997 = t4980 * t4993;
        float t4998 = t4981 * t4990;
        float t4999 = t4997 + t4998;
        int t5000 = t4026 + t4982;
        float t5001 = t4985 + t4996;
        memory[83996148 + t5000] = t5001;
        int t5003 = t4026 + t4982;
        int t5004 = t5003 + 512;
        float t5005 = t4988 + t4999;
        memory[83996148 + t5004] = t5005;
        int t5007 = t4026 + t4983;
        float t5008 = t4985 - t4996;
        memory[83996148 + t5007] = t5008;
        int t5010 = t4026 + t4983;
        int t5011 = t5010 + 512;
        float t5012 = t4988 - t4999;
        memory[83996148 + t5011] = t5012;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr5015 = 0; _pr5015 < 512; _pr5015++) {
        int t5016 = t4026 + _pr5015;
        float t5017 = memory[83996148 + t5016];
        float t5018 = t5017 * 7.599708e-06;
        float t5019 = memory[25460 + (int)_pr5015];
        int t5020 = t4027 + _pr5015;
        float t5021 = t5018 * t5019;
        memory[125955572 + t5020] = t5021;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t5025 = t[11*frameCount + id] > 0.0;
    if (t5025) {
      for (uint _pr5027 = 0; _pr5027 < 512; _pr5027++) {
        int t5028 = t4027 + _pr5027;
        memory[117550580 + t5028] = 0.0;
        int t5030 = t4027 + _pr5027;
        memory[125955572 + t5030] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5051), value: global(5051)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(513) - handled in variable access */
    int t5034 = id;
    float t5035 = 0.0;
    for (uint t5036 = 0; t5036 < 512; t5036++) {
      float t5037 = (float)t5036;
      float t5038 = (float)t5034;
      float t5039 = t5038 + t5037;
      int t5040 = 511 - t5036;
      float t5041 = frameCount - 1.0;
      float t5042 = metal::min(t5039, t5041);
      int t5043 = (int)t5042;
      int t5044 = t5043 * 512;
      int t5045 = t5044 + t5040;
      float t5046 = memory[117550580 + t5045];
      float t5047 = t5039 < frameCount;
      float t5048 = metal::select(0.0, t5046, t5047 > 0.0);
      float t5049 = t5035 + t5048;
      t5035 = t5049;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[19*frameCount + id] = (t5035 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5069), value: global(5069)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(513) - handled in variable access */
    int t5052 = id;
    float t5053 = 0.0;
    for (uint t5054 = 0; t5054 < 512; t5054++) {
      float t5055 = (float)t5054;
      float t5056 = (float)t5052;
      float t5057 = t5056 + t5055;
      int t5058 = 511 - t5054;
      float t5059 = frameCount - 1.0;
      float t5060 = metal::min(t5057, t5059);
      int t5061 = (int)t5060;
      int t5062 = t5061 * 512;
      int t5063 = t5062 + t5058;
      float t5064 = memory[125955572 + t5063];
      float t5065 = t5057 < frameCount;
      float t5066 = metal::select(0.0, t5064, t5065 > 0.0);
      float t5067 = t5053 + t5066;
      t5053 = t5067;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[20*frameCount + id] = (t5053 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5090), value: global(5090)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5089), value: global(5089)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5074), value: global(5074)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5069) - handled in variable access */
    /* loadGlobal(5051) - handled in variable access */
    /* loadGlobal(3956) - handled in variable access */
    /* loadGlobal(3938) - handled in variable access */
    /* loadGlobal(458) - handled in variable access */
    /* loadGlobal(457) - handled in variable access */
    /* loadGlobal(439) - handled in variable access */
    /* loadGlobal(309) - handled in variable access */
    float t5070 = t[17*frameCount + id] + t[19*frameCount + id];
    float t5071 = t[18*frameCount + id] + t[20*frameCount + id];
    float t5072 = 0.015625 * t5070;
    float t5073 = t[7*frameCount + id] * t5070;
    t[21*frameCount + id] = t[6*frameCount + id] * t5072;
    float t5075 = t[5*frameCount + id] * t5072;
    float t5076 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t5077 = t5076 < 0.0;
    float t5078 = t5076 + 61.0;
    float t5079 = metal::select(t5076, t5078, t5077 > 0.0);
    float t5080 = t5079;
    float t5081 = metal::floor(t5080);
    float t5082 = t5080 - t5081;
    float t5083 = t5081 + 1.0;
    float t5084 = t5083 >= 61.0;
    float t5085 = metal::select(t5083, 0.0, t5084 > 0.0);
    float t5086 = 1.0 - t5082;
    float t5087 = t5075 * t5086;
    float t5088 = t5075 * t5082;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5081], t5087, metal::memory_order_relaxed);
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209874420 + (int)t5085], t5088, metal::memory_order_relaxed);
    for (uint t5091 = 0; t5091 < 61; t5091++) {
      float t5092 = memory[209874420 + (int)t5091];
      float t5093 = memory[60788 + (int)t5091];
      float t5094 = t5092 / t5093;
      float t5095 = memory[60788 + (int)t5091];
      float t5096 = memory[60788 + (int)t5091];
      float t5097 = t5095 * t5096;
      float t5098 = 1.0 / t5097;
      float t5099 = memory[209874420 + (int)t5091];
      float t5100 = t5099 * -1.0;
      float t5101 = t5100 * t5098;
      float t5102 = t5094 + t5101;
      float t5103 = memory[60596 + (int)t5091];
      float t5104 = metal::exp(t5103);
      float t5105 = t5104 * t5101;
      float t5106 = -1.0 * t5105;
      int t5107 = id;
      int t5108 = t5107 * 61;
      int t5109 = t5108 + t5091;
      memory[3206644 + t5109] = t5106;
      float t5111 = memory[60724 + (int)t5091];
      float t5112 = t5111 * t5105;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5113 = 0; t5113 < 1; t5113++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=201, axis=0, in=[61, 1], out=[1], inFA=true, outFA=true), value: empty) */
      float t5114 = 0.0;
      int t5115 = t5113;
      int t5116 = t5115;
      int t5117 = t5113 - t5116;
      int t5118 = t5115;
      int t5119 = t5118;
      for (uint t5120 = 0; t5120 < 61; t5120++) {
        int t5121 = t5120;
        int t5122 = t5119 + t5121;
        int t5123 = id;
        int t5124 = t5123 * 61;
        int t5125 = t5124 + t5122;
        float t5126 = memory[3206644 + t5125];
        float t5127 = t5114 + t5126;
        t5114 = t5127;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5129 = id;
      int t5130 = t5129;
      int t5131 = t5130 + t5113;
      memory[60916 + t5131] = t5114;
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
    for (uint t5133 = 0; t5133 < 3904; t5133++) {
      /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=203, axis=2, in=[61, 1], out=[61, 1, 64], inFA=true, outFA=true), value: empty) */
      int t5134 = t5133 / 64;
      int t5135 = t5134 % 61;
      int t5136 = t5135 * 1;
      int t5137 = 0 + t5136;
      int t5138 = t5133 / 64;
      int t5139 = t5138 % 1;
      int t5140 = t5139 * 1;
      int t5141 = t5137 + t5140;
      float t5142 = (float)t5141;
      int t5143 = id;
      int t5144 = t5143 * 61;
      float t5145 = t5144 + t5142;
      int t5146 = (int)t5145;
      float t5147 = memory[3206644 + t5146];
      float t5148 = (float)t5133;
      int t5149 = id;
      int t5150 = t5149 * 3904;
      float t5151 = t5150 + t5148;
      int t5152 = (int)t5151;
      memory[273837620 + t5152] = t5147;
      int t5154 = t5133 / 64;
      int t5155 = t5154 * 64;
      int t5156 = t5133 - t5155;
      int t5157 = t5156 / 64;
      int t5158 = t5157 * 64;
      int t5159 = t5156 - t5158;
      int t5160 = t5159 / 64;
      int t5161 = t5160 * 64;
      int t5162 = t5159 - t5161;
      float t5163 = memory[8576 + t5162];
      int t5164 = id;
      int t5165 = t5164 * 3904;
      int t5166 = t5165 + t5133;
      float t5167 = memory[273837620 + t5166];
      float t5168 = t5163 * t5167;
      int t5169 = id;
      int t5170 = t5169 * 3904;
      int t5171 = t5170 + t5133;
      memory[209874484 + t5171] = t5168;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5173 = 0; t5173 < 64; t5173++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=206, axis=0, in=[61, 1, 64], out=[1, 64], inFA=true, outFA=true), value: empty) */
      float t5174 = 0.0;
      int t5175 = t5173 / 64;
      int t5176 = t5175 * 64;
      int t5177 = t5173 - t5176;
      int t5178 = t5177;
      int t5179 = t5178;
      int t5180 = t5177 - t5179;
      int t5181 = t5175 * 64;
      int t5182 = t5181;
      int t5183 = t5178;
      int t5184 = t5182 + t5183;
      for (uint t5185 = 0; t5185 < 61; t5185++) {
        int t5186 = t5185 * 64;
        int t5187 = t5184 + t5186;
        int t5188 = t5185 * 64;
        int t5189 = t5188 + t5178;
        float t5190 = memory[41076 + t5189];
        float t5191 = t5185 + 0.0;
        int t5192 = id;
        int t5193 = t5192 * 61;
        float t5194 = t5193 + t5191;
        int t5195 = (int)t5194;
        float t5196 = memory[3206644 + t5195];
        float t5197 = t5190 * t5196;
        float t5198 = t5174 + t5197;
        t5174 = t5198;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5200 = id;
      int t5201 = t5200 * 64;
      int t5202 = t5201 + t5173;
      memory[37809652 + t5202] = t5174;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5204), value: global(5204)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5074) - handled in variable access */
    /* loadGlobal(431) - handled in variable access */
    /* loadGlobal(349) - handled in variable access */
    t[24*frameCount + id] = t[3*frameCount + id] * t[21*frameCount + id];
    float t5205 = t[4*frameCount + id] * t[21*frameCount + id];
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
  float t5722 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5722)) {
    /* loadGlobal(5204) - handled in variable access */
    int t5206 = id;
    int t5207 = t5206 / 64;
    uint _frameIndex = (uint)(t5207);
    int t5208 = t5207 * 64;
    int t5209 = t5206 - t5208;
    int t5210 = t5207 * 64;
    int t5211 = t5210 + t5209;
    memory[3206644 + t5211] = t[24*frameCount + _frameIndex];
    int t5213 = _frameIndex;
    int t5214 = t5213 * 64;
    int t5215 = t5214 + t5209;
    float t5216 = memory[1109492 + t5215];
    int t5217 = _frameIndex;
    int t5218 = t5217 * 64;
    int t5219 = t5218 + t5209;
    float t5220 = memory[3206644 + t5219];
    float t5221 = t5216 * t5220;
    int t5222 = _frameIndex;
    int t5223 = t5222 * 64;
    int t5224 = t5223 + t5209;
    float t5225 = memory[2158068 + t5224];
    int t5226 = _frameIndex;
    int t5227 = t5226 * 64;
    int t5228 = t5227 + t5209;
    float t5229 = memory[3206644 + t5228];
    float t5230 = t5225 * t5229;
    int t5231 = _frameIndex;
    int t5232 = t5231 * 64;
    int t5233 = t5232 + t5209;
    memory[42020340 + t5233] = t5230;
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
  float t5723 = frameCount * 3904.0;
  if (id >= 0 && id < (uint)(t5723)) {
    /* loadGlobal(309) - handled in variable access */
    int t5235 = id;
    int t5236 = t5235 / 3904;
    uint _frameIndex = (uint)(t5236);
    int t5237 = t5236 * 3904;
    int t5238 = t5235 - t5237;
    float t5239 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5240 = t5239 < 0.0;
    float t5241 = t5239 + 61.0;
    float t5242 = metal::select(t5239, t5241, t5240 > 0.0);
    float t5243 = metal::floor(t5242);
    float t5244 = t5243 + 1.0;
    float t5245 = t5244 >= 61.0;
    float t5246 = metal::select(t5244, 0.0, t5245 > 0.0);
    float t5247 = t5242 - t5243;
    int t5248 = _frameIndex;
    memory[3206644 + t5248] = t5243;
    memory[46231028 + t5248] = t5247;
    float t5251 = t5248 + 16384.0;
    int t5252 = (int)t5251;
    memory[3206644 + t5252] = t5246;
    float t5254 = 1.0 - t5247;
    float t5255 = t5248 * 64.0;
    for (uint _pr5256 = 0; _pr5256 < 64; _pr5256++) {
      float t5257 = (float)_pr5256;
      float t5258 = t5255 + t5257;
      int t5259 = (int)t5258;
      float t5260 = memory[42020340 + t5259];
      float t5261 = t5255 + t5257;
      float t5262 = t5260 * t5254;
      int t5263 = (int)t5261;
      memory[1109492 + t5263] = t5262;
      float t5265 = t5260 * t5247;
      int t5266 = (int)t5261;
      memory[2158068 + t5266] = t5265;
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
  if (id < 3904) { uint _pr5269 = id;
    int t5270 = _pr5269 / 64;
    int t5271 = t5270 * 64;
    int t5272 = _pr5269 - t5271;
    float t5273 = (float)t5270;
    float t5274 = (float)t5272;
    float t5275 = 0.0;
    for (uint t5276 = 0; t5276 < 16384; t5276++) {
      float t5277 = (float)t5276;
      float t5278 = t5277 < frameCount;
      float t5279 = t5277 * 64.0;
      float t5280 = t5279 + t5274;
      float t5281 = memory[3206644 + (int)t5276];
      float t5282 = t5281 - t5273;
      float t5283 = metal::abs(t5282);
      float t5284 = t5283 < 0.5;
      int t5285 = (int)t5280;
      float t5286 = memory[1109492 + t5285];
      float t5287 = t5278 * t5284;
      float t5288 = t5287 > 0.0;
      float t5289 = metal::select(0.0, t5286, t5288 > 0.0);
      float t5290 = t5275 + t5289;
      t5275 = t5290;
      float t5291 = t5277 + 16384.0;
      int t5292 = (int)t5291;
      float t5293 = memory[3206644 + t5292];
      float t5294 = t5293 - t5273;
      float t5295 = metal::abs(t5294);
      float t5296 = t5295 < 0.5;
      int t5297 = (int)t5280;
      float t5298 = memory[2158068 + t5297];
      float t5299 = t5278 * t5296;
      float t5300 = t5299 > 0.0;
      float t5301 = metal::select(0.0, t5298, t5300 > 0.0);
      float t5302 = t5275 + t5301;
      t5275 = t5302;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t5304 = t5273 * 64.0;
    float t5305 = t5304 + t5274;
    int t5306 = (int)t5305;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[337800756 + t5306], t5275, metal::memory_order_relaxed);
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
    for (uint t5309 = 0; t5309 < 3904; t5309++) {
      float t5310 = memory[337800756 + (int)t5309];
      float t5311 = memory[52788 + (int)t5309];
      float t5312 = t5310 / t5311;
      float t5313 = memory[52788 + (int)t5309];
      float t5314 = memory[52788 + (int)t5309];
      float t5315 = t5313 * t5314;
      float t5316 = 1.0 / t5315;
      float t5317 = memory[337800756 + (int)t5309];
      float t5318 = t5317 * -1.0;
      float t5319 = t5318 * t5316;
      float t5320 = t5312 + t5319;
      float t5321 = memory[44980 + (int)t5309];
      float t5322 = metal::exp(t5321);
      float t5323 = t5322 * t5319;
      float t5324 = -1.0 * t5323;
      int t5325 = id;
      int t5326 = t5325 * 3904;
      int t5327 = t5326 + t5309;
      memory[273837620 + t5327] = t5324;
      float t5329 = memory[56692 + (int)t5309];
      float t5330 = t5329 * t5323;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5331 = 0; t5331 < 64; t5331++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=232, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5332 = 0.0;
      int t5333 = t5331;
      int t5334 = t5333;
      int t5335 = t5331 - t5334;
      int t5336 = t5333;
      int t5337 = t5336;
      for (uint t5338 = 0; t5338 < 61; t5338++) {
        int t5339 = t5338 * 64;
        int t5340 = t5337 + t5339;
        int t5341 = id;
        int t5342 = t5341 * 3904;
        int t5343 = t5342 + t5340;
        float t5344 = memory[273837620 + t5343];
        float t5345 = t5332 + t5344;
        t5332 = t5345;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5347 = id;
      int t5348 = t5347 * 64;
      int t5349 = t5348 + t5331;
      memory[1109492 + t5349] = t5332;
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
    for (uint t5351 = 0; t5351 < 3904; t5351++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=237, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5352 = 0.0;
      int t5353 = t5351 / 64;
      int t5354 = t5353 * 64;
      int t5355 = t5351 - t5354;
      int t5356 = t5355;
      int t5357 = t5356;
      int t5358 = t5355 - t5357;
      int t5359 = t5353 * 4096;
      int t5360 = t5359;
      int t5361 = t5356;
      int t5362 = t5360 + t5361;
      for (uint t5363 = 0; t5363 < 64; t5363++) {
        int t5364 = t5363 * 64;
        int t5365 = t5362 + t5364;
        int t5366 = t5363 * 64;
        int t5367 = t5366 + t5356;
        int t5368 = t5367 / 64;
        int t5369 = t5368 * 64;
        int t5370 = t5367 - t5369;
        int t5371 = t5370 * 64;
        int t5372 = t5368 + t5371;
        float t5373 = memory[4416 + t5372];
        int t5374 = t5353 * 64;
        int t5375 = t5374 + t5363;
        int t5376 = id;
        int t5377 = t5376 * 3904;
        int t5378 = t5377 + t5375;
        float t5379 = memory[273837620 + t5378];
        float t5380 = t5373 * t5379;
        float t5381 = t5352 + t5380;
        t5352 = t5381;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5383 = id;
      int t5384 = t5383 * 3904;
      int t5385 = t5384 + t5351;
      memory[337804660 + t5385] = t5352;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5387 = 0; t5387 < 4096; t5387++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=239, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5388 = 0.0;
      int t5389 = t5387 / 64;
      int t5390 = t5389 * 64;
      int t5391 = t5387 - t5390;
      int t5392 = t5391;
      int t5393 = t5392;
      int t5394 = t5391 - t5393;
      int t5395 = t5389 * 64;
      int t5396 = t5395;
      int t5397 = t5392;
      int t5398 = t5396 + t5397;
      for (uint t5399 = 0; t5399 < 61; t5399++) {
        int t5400 = t5399 * 4096;
        int t5401 = t5398 + t5400;
        int t5402 = t5399 * 64;
        int t5403 = t5402 + t5392;
        float t5404 = memory[41076 + t5403];
        int t5405 = t5399 * 64;
        int t5406 = t5405 + t5389;
        int t5407 = id;
        int t5408 = t5407 * 3904;
        int t5409 = t5408 + t5406;
        float t5410 = memory[273837620 + t5409];
        float t5411 = t5404 * t5410;
        float t5412 = t5388 + t5411;
        t5388 = t5412;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5414 = id;
      int t5415 = t5414 * 4096;
      int t5416 = t5415 + t5387;
      memory[401767796 + t5416] = t5388;
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
    for (uint t5418 = 0; t5418 < 3904; t5418++) {
      int t5419 = id;
      int t5420 = t5419 * 3904;
      int t5421 = t5420 + t5418;
      float t5422 = memory[209874484 + t5421];
      int t5423 = id;
      int t5424 = t5423 * 3904;
      int t5425 = t5424 + t5418;
      float t5426 = memory[337804660 + t5425];
      float t5427 = t5422 + t5426;
      float t5428 = memory[37172 + (int)t5418];
      float t5429 = metal::tanh(t5428);
      float t5430 = t5429 * t5429;
      float t5431 = 1.0 - t5430;
      float t5432 = t5431 * t5427;
      int t5433 = id;
      int t5434 = t5433 * 3904;
      int t5435 = t5434 + t5418;
      memory[468876660 + t5435] = t5432;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5437 = 0; t5437 < 64; t5437++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=250, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5438 = 0.0;
      int t5439 = t5437;
      int t5440 = t5439;
      int t5441 = t5437 - t5440;
      int t5442 = t5439;
      int t5443 = t5442;
      for (uint t5444 = 0; t5444 < 61; t5444++) {
        int t5445 = t5444 * 64;
        int t5446 = t5443 + t5445;
        int t5447 = t5444 * 64;
        int t5448 = t5447 + t5439;
        float t5449 = memory[25460 + t5448];
        int t5450 = t5444 * 64;
        int t5451 = t5450 + t5439;
        int t5452 = id;
        int t5453 = t5452 * 3904;
        int t5454 = t5453 + t5451;
        float t5455 = memory[273837620 + t5454];
        float t5456 = t5449 * t5455;
        float t5457 = t5438 + t5456;
        t5438 = t5457;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5459 = id;
      int t5460 = t5459 * 64;
      int t5461 = t5460 + t5437;
      memory[2158068 + t5461] = t5438;
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
    for (uint t5463 = 0; t5463 < 3904; t5463++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=255, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5464 = 0.0;
      int t5465 = t5463 / 64;
      int t5466 = t5465 * 64;
      int t5467 = t5463 - t5466;
      int t5468 = t5467;
      int t5469 = t5468;
      int t5470 = t5467 - t5469;
      int t5471 = t5465 * 4096;
      int t5472 = t5471;
      int t5473 = t5468;
      int t5474 = t5472 + t5473;
      for (uint t5475 = 0; t5475 < 64; t5475++) {
        int t5476 = t5475 * 64;
        int t5477 = t5474 + t5476;
        int t5478 = t5475 * 64;
        int t5479 = t5478 + t5468;
        int t5480 = t5479 / 64;
        int t5481 = t5480 * 64;
        int t5482 = t5479 - t5481;
        int t5483 = t5482 * 64;
        int t5484 = t5480 + t5483;
        float t5485 = memory[256 + t5484];
        int t5486 = t5465 * 64;
        int t5487 = t5486 + t5475;
        int t5488 = id;
        int t5489 = t5488 * 3904;
        int t5490 = t5489 + t5487;
        float t5491 = memory[468876660 + t5490];
        float t5492 = t5485 * t5491;
        float t5493 = t5464 + t5492;
        t5464 = t5493;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5495 = id;
      int t5496 = t5495 * 3904;
      int t5497 = t5496 + t5463;
      memory[209874484 + t5497] = t5464;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5499 = 0; t5499 < 4096; t5499++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=257, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5500 = 0.0;
      int t5501 = t5499 / 64;
      int t5502 = t5501 * 64;
      int t5503 = t5499 - t5502;
      int t5504 = t5503;
      int t5505 = t5504;
      int t5506 = t5503 - t5505;
      int t5507 = t5501 * 64;
      int t5508 = t5507;
      int t5509 = t5504;
      int t5510 = t5508 + t5509;
      for (uint t5511 = 0; t5511 < 61; t5511++) {
        int t5512 = t5511 * 4096;
        int t5513 = t5510 + t5512;
        int t5514 = t5511 * 64;
        int t5515 = t5514 + t5504;
        float t5516 = memory[29364 + t5515];
        int t5517 = t5511 * 64;
        int t5518 = t5517 + t5501;
        int t5519 = id;
        int t5520 = t5519 * 3904;
        int t5521 = t5520 + t5518;
        float t5522 = memory[468876660 + t5521];
        float t5523 = t5516 * t5522;
        float t5524 = t5500 + t5523;
        t5500 = t5524;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5526 = id;
      int t5527 = t5526 * 4096;
      int t5528 = t5527 + t5499;
      memory[532839796 + t5528] = t5500;
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
    for (uint t5530 = 0; t5530 < 3904; t5530++) {
      float t5531 = memory[33268 + (int)t5530];
      float t5532 = metal::tanh(t5531);
      float t5533 = t5532 * t5532;
      float t5534 = 1.0 - t5533;
      int t5535 = id;
      int t5536 = t5535 * 3904;
      int t5537 = t5536 + t5530;
      float t5538 = memory[209874484 + t5537];
      float t5539 = t5534 * t5538;
      int t5540 = id;
      int t5541 = t5540 * 3904;
      int t5542 = t5541 + t5530;
      memory[273837620 + t5542] = t5539;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5544 = 0; t5544 < 64; t5544++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=267, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5545 = 0.0;
      int t5546 = t5544;
      int t5547 = t5546;
      int t5548 = t5544 - t5547;
      int t5549 = t5546;
      int t5550 = t5549;
      for (uint t5551 = 0; t5551 < 61; t5551++) {
        int t5552 = t5551 * 64;
        int t5553 = t5550 + t5552;
        int t5554 = t5551 * 64;
        int t5555 = t5554 + t5546;
        float t5556 = memory[25460 + t5555];
        int t5557 = t5551 * 64;
        int t5558 = t5557 + t5546;
        int t5559 = id;
        int t5560 = t5559 * 3904;
        int t5561 = t5560 + t5558;
        float t5562 = memory[209874484 + t5561];
        float t5563 = t5556 * t5562;
        float t5564 = t5545 + t5563;
        t5545 = t5564;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5566 = id;
      int t5567 = t5566 * 64;
      int t5568 = t5567 + t5544;
      memory[3206644 + t5568] = t5545;
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
    for (uint t5570 = 0; t5570 < 183; t5570++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=272, axis=1, in=[61, 64, 3], out=[61, 3], inFA=true, outFA=true), value: empty) */
      float t5571 = 0.0;
      int t5572 = t5570 / 3;
      int t5573 = t5572 * 3;
      int t5574 = t5570 - t5573;
      int t5575 = t5574;
      int t5576 = t5575;
      int t5577 = t5574 - t5576;
      int t5578 = t5572 * 192;
      int t5579 = t5578;
      int t5580 = t5575;
      int t5581 = t5579 + t5580;
      for (uint t5582 = 0; t5582 < 64; t5582++) {
        int t5583 = t5582 * 3;
        int t5584 = t5581 + t5583;
        int t5585 = t5582 * 3;
        int t5586 = t5585 + t5575;
        int t5587 = t5586 / 3;
        int t5588 = t5587 * 3;
        int t5589 = t5586 - t5588;
        int t5590 = t5589 * 64;
        int t5591 = t5587 + t5590;
        float t5592 = memory[0 + t5591];
        int t5593 = t5572 * 64;
        int t5594 = t5593 + t5582;
        int t5595 = id;
        int t5596 = t5595 * 3904;
        int t5597 = t5596 + t5594;
        float t5598 = memory[273837620 + t5597];
        float t5599 = t5592 * t5598;
        float t5600 = t5571 + t5599;
        t5571 = t5600;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5602 = id;
      int t5603 = t5602 * 183;
      int t5604 = t5603 + t5570;
      memory[46231028 + t5604] = t5571;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 3]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5606 = 0; t5606 < 192; t5606++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=274, axis=0, in=[61, 64, 3], out=[64, 3], inFA=true, outFA=true), value: empty) */
      float t5607 = 0.0;
      int t5608 = t5606 / 3;
      int t5609 = t5608 * 3;
      int t5610 = t5606 - t5609;
      int t5611 = t5610;
      int t5612 = t5611;
      int t5613 = t5610 - t5612;
      int t5614 = t5608 * 3;
      int t5615 = t5614;
      int t5616 = t5611;
      int t5617 = t5615 + t5616;
      for (uint t5618 = 0; t5618 < 61; t5618++) {
        int t5619 = t5618 * 192;
        int t5620 = t5617 + t5619;
        int t5621 = t5618 * 3;
        int t5622 = t5621 + t5611;
        float t5623 = memory[8706 + t5622];
        int t5624 = t5618 * 64;
        int t5625 = t5624 + t5608;
        int t5626 = id;
        int t5627 = t5626 * 3904;
        int t5628 = t5627 + t5625;
        float t5629 = memory[273837620 + t5628];
        float t5630 = t5623 * t5629;
        float t5631 = t5607 + t5630;
        t5607 = t5631;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5633 = id;
      int t5634 = t5633 * 192;
      int t5635 = t5634 + t5606;
      memory[42020340 + t5635] = t5607;
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
  if (id < 192) { uint _pr5637 = id;
    float t5638 = 0.0;
    for (uint t5639 = 0; t5639 < 16384; t5639++) {
      int t5640 = t5639 * 192;
      int t5641 = t5640 + _pr5637;
      float t5642 = memory[42020340 + t5641];
      float t5643 = t5638 + t5642;
      t5638 = t5643;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5637] = t5638;
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
  if (id < 64) { uint _pr5647 = id;
    float t5648 = 0.0;
    for (uint t5649 = 0; t5649 < 16384; t5649++) {
      int t5650 = t5649 * 64;
      int t5651 = t5650 + _pr5647;
      float t5652 = memory[3206644 + t5651];
      float t5653 = t5648 + t5652;
      t5648 = t5653;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5647] = t5648;
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
  if (id < 4096) { uint _pr5657 = id;
    float t5658 = 0.0;
    for (uint t5659 = 0; t5659 < 16384; t5659++) {
      int t5660 = t5659 * 4096;
      int t5661 = t5660 + _pr5657;
      float t5662 = memory[532839796 + t5661];
      float t5663 = t5658 + t5662;
      t5658 = t5663;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[3206644 + (int)_pr5657] = t5658;
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
  if (id < 64) { uint _pr5667 = id;
    float t5668 = 0.0;
    for (uint t5669 = 0; t5669 < 16384; t5669++) {
      int t5670 = t5669 * 64;
      int t5671 = t5670 + _pr5667;
      float t5672 = memory[2158068 + t5671];
      float t5673 = t5668 + t5672;
      t5668 = t5673;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5667] = t5668;
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
  if (id < 4096) { uint _pr5677 = id;
    float t5678 = 0.0;
    for (uint t5679 = 0; t5679 < 16384; t5679++) {
      int t5680 = t5679 * 4096;
      int t5681 = t5680 + _pr5677;
      float t5682 = memory[401767796 + t5681];
      float t5683 = t5678 + t5682;
      t5678 = t5683;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[2158068 + (int)_pr5677] = t5678;
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
  if (id < 64) { uint _pr5687 = id;
    float t5688 = 0.0;
    for (uint t5689 = 0; t5689 < 16384; t5689++) {
      int t5690 = t5689 * 64;
      int t5691 = t5690 + _pr5687;
      float t5692 = memory[1109492 + t5691];
      float t5693 = t5688 + t5692;
      t5688 = t5693;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5687] = t5688;
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
  if (id < 64) { uint _pr5697 = id;
    float t5698 = 0.0;
    for (uint t5699 = 0; t5699 < 16384; t5699++) {
      int t5700 = t5699 * 64;
      int t5701 = t5700 + _pr5697;
      float t5702 = memory[37809652 + t5701];
      float t5703 = t5698 + t5702;
      t5698 = t5703;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[25460 + (int)_pr5697] = t5698;
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
  if (id < 1) { uint _pr5707 = id;
    float t5708 = 0.0;
    for (uint t5709 = 0; t5709 < 16384; t5709++) {
      int t5710 = t5709;
      int t5711 = t5710 + _pr5707;
      float t5712 = memory[60916 + t5711];
      float t5713 = t5708 + t5712;
      t5708 = t5713;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[60596 + (int)_pr5707] = t5708;
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
    /* loadGlobal(5090) - handled in variable access */
    /* loadGlobal(5089) - handled in variable access */
    /* loadGlobal(2743) - handled in variable access */
    outputs[0 * frameCount + id] = t[16*frameCount + id];
  }
  #pragma clang diagnostic pop
}

