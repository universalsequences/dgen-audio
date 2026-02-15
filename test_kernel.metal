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
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t76 = 0; t76 < 3904; t76++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=17, axis=2, in=[61, 64, 3], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t77 = 0.0;
      int t78 = t76 / 64;
      int t79 = t78 * 64;
      int t80 = t76 - t79;
      int t81 = t80;
      int t82 = t81;
      int t83 = t80 - t82;
      int t84 = t78 * 192;
      int t85 = t84;
      int t86 = t81 * 3;
      int t87 = t85 + t86;
      for (uint t88 = 0; t88 < 3; t88++) {
        int t89 = t88;
        int t90 = t87 + t89;
        int t91 = t78 * 3;
        int t92 = t91 + t88;
        float t93 = memory[4546 + t92];
        int t94 = t81 * 3;
        int t95 = t94 + t88;
        int t96 = t95 / 3;
        int t97 = t96 * 3;
        int t98 = t95 - t97;
        int t99 = t98 * 64;
        int t100 = t96 + t99;
        float t101 = memory[0 + t100];
        float t102 = t93 * t101;
        float t103 = t77 + t102;
        t77 = t103;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[21300 + (int)t76] = t77;
      float t106 = memory[21300 + (int)t76];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t107 = t76 / 64;
      int t108 = t107 * 64;
      int t109 = t76 - t108;
      int t110 = t109;
      float t111 = memory[192 + t110];
      float t112 = t106 + t111;
      memory[25204 + (int)t76] = t112;
      float t114 = metal::tanh(t112);
      memory[29108 + (int)t76] = t114;
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
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t116 = 0; t116 < 3904; t116++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=24, axis=2, in=[61, 64, 64], out=[61, 64], inFA=false, outFA=false), value: empty) */
      float t117 = 0.0;
      int t118 = t116 / 64;
      int t119 = t118 * 64;
      int t120 = t116 - t119;
      int t121 = t120;
      int t122 = t121;
      int t123 = t120 - t122;
      int t124 = t118 * 4096;
      int t125 = t124;
      int t126 = t121 * 64;
      int t127 = t125 + t126;
      for (uint t128 = 0; t128 < 64; t128++) {
        int t129 = t128;
        int t130 = t127 + t129;
        int t131 = t118 * 64;
        int t132 = t131 + t128;
        float t133 = memory[29108 + t132];
        int t134 = t121 * 64;
        int t135 = t134 + t128;
        int t136 = t135 / 64;
        int t137 = t136 * 64;
        int t138 = t135 - t137;
        int t139 = t138 * 64;
        int t140 = t136 + t139;
        float t141 = memory[256 + t140];
        float t142 = t133 * t141;
        float t143 = t117 + t142;
        t117 = t143;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[21300 + (int)t116] = t117;
      float t146 = memory[21300 + (int)t116];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t147 = t116 / 64;
      int t148 = t147 * 64;
      int t149 = t116 - t148;
      int t150 = t149;
      float t151 = memory[4352 + t150];
      float t152 = t146 + t151;
      memory[40820 + (int)t116] = t152;
      float t154 = t152 * -1.0;
      memory[44724 + (int)t116] = t154;
      float t156 = metal::exp(t154);
      float t157 = 1.0 + t156;
      memory[36916 + (int)t116] = t157;
      float t159 = 1.0 / t157;
      memory[33012 + (int)t116] = t159;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
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
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t161 = 0; t161 < 61; t161++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=36, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t162 = 0.0;
      int t163 = t161;
      int t164 = t163;
      int t165 = t161 - t164;
      int t166 = t165;
      int t167 = t166;
      int t168 = t165 - t167;
      int t169 = t163 * 64;
      int t170 = t169;
      int t171 = t166 * 64;
      int t172 = t170 + t171;
      for (uint t173 = 0; t173 < 64; t173++) {
        int t174 = t173;
        int t175 = t172 + t174;
        int t176 = t163 * 64;
        int t177 = t176 + t173;
        float t178 = memory[29108 + t177];
        int t179 = t173 / 64;
        int t180 = t179 * 64;
        int t181 = t173 - t180;
        float t182 = memory[4416 + t181];
        float t183 = t178 * t182;
        float t184 = t162 + t183;
        t162 = t184;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[21300 + (int)t161] = t162;
      float t187 = memory[21300 + (int)t161];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t188 = t161;
      int t189 = t188;
      int t190 = t161 - t189;
      float t191 = memory[4480 + (int)0.0];
      float t192 = t187 + t191;
      memory[48756 + (int)t161] = t192;
      float t194 = t192 * -1.0;
      memory[48692 + (int)t161] = t194;
      float t196 = metal::exp(t194);
      float t197 = 1.0 + t196;
      memory[48628 + (int)t161] = t197;
      float t199 = 1.0 / t197;
      memory[48820 + (int)t161] = t199;
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
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t201 = 0; t201 < 61; t201++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=48, axis=2, in=[61, 1, 64], out=[61, 1], inFA=false, outFA=false), value: empty) */
      float t202 = 0.0;
      int t203 = t201;
      int t204 = t203;
      int t205 = t201 - t204;
      int t206 = t205;
      int t207 = t206;
      int t208 = t205 - t207;
      int t209 = t203 * 64;
      int t210 = t209;
      int t211 = t206 * 64;
      int t212 = t210 + t211;
      for (uint t213 = 0; t213 < 64; t213++) {
        int t214 = t213;
        int t215 = t212 + t214;
        int t216 = t203 * 64;
        int t217 = t216 + t213;
        float t218 = memory[29108 + t217];
        int t219 = t213 / 64;
        int t220 = t219 * 64;
        int t221 = t213 - t220;
        float t222 = memory[4481 + t221];
        float t223 = t218 * t222;
        float t224 = t202 + t223;
        t202 = t224;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      memory[21300 + (int)t201] = t202;
      float t227 = memory[21300 + (int)t201];
      /* [1mUOp[0m(op: [38;5;51mbroadcastAccess[0m, value: empty) */
      int t228 = t201;
      int t229 = t228;
      int t230 = t201 - t229;
      float t231 = memory[4545 + (int)0.0];
      float t232 = t227 + t231;
      float t233 = t232 * -1.0;
      float t234 = metal::exp(t233);
      float t235 = 1.0 + t234;
      float t236 = 1.0 / t235;
      memory[48884 + (int)t201] = t236;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 4
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_4(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(238), value: global(238)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[0*frameCount + i] = memory[532827828];
      float t239 = t[0*frameCount + i] + 0.003662333;
      float t240 = metal::select(t239, 0.0, 0.0 > 0.0);
      float t241 = t240;
      float t242 = (t241 * 0.016666668);
      float t243 = metal::floor(t242);
      float t244 = t243 * 60.0;
      float t245 = t240 - t244;
      memory[532827828] = t245;
      float t247 = t245 >= 60.0;
      if (t247) {
        float t249 = t245 - 60.0;
        memory[532827828] = t249;
      }
      if (0.0) {
        memory[532827828] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 5
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(296), value: global(296)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(276), value: global(276)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(256), value: global(256)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(238) - handled in variable access */
    float t255 = metal::min(t[0*frameCount + id], 59.9999);
    t[1*frameCount + id] = metal::max(t255, 0.0);
    float t257 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t258 = t257 < 0.0;
    float t259 = t257 + 61.0;
    float t260 = metal::select(t257, t259, t258 > 0.0);
    float t261 = t260;
    float t262 = metal::floor(t261);
    float t263 = t261 - t262;
    float t264 = t262 + 1.0;
    float t265 = t264 >= 61.0;
    float t266 = metal::select(t264, 0.0, t265 > 0.0);
    int t267 = (int)t262;
    float t268 = memory[21113 + t267];
    int t269 = (int)t266;
    float t270 = memory[21113 + t269];
    float t271 = 1.0 - t263;
    float t272 = t268 * t271;
    float t273 = t270 * t263;
    float t274 = t272 + t273;
    float t275 = metal::max(t274, 20.0);
    t[2*frameCount + id] = metal::min(t275, 500.0);
    float t277 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t278 = t277 < 0.0;
    float t279 = t277 + 61.0;
    float t280 = metal::select(t277, t279, t278 > 0.0);
    float t281 = t280;
    float t282 = metal::floor(t281);
    float t283 = t281 - t282;
    float t284 = t282 + 1.0;
    float t285 = t284 >= 61.0;
    float t286 = metal::select(t284, 0.0, t285 > 0.0);
    int t287 = (int)t282;
    float t288 = memory[21174 + t287];
    int t289 = (int)t286;
    float t290 = memory[21174 + t289];
    float t291 = 1.0 - t283;
    float t292 = t288 * t291;
    float t293 = t290 * t283;
    float t294 = t292 + t293;
    float t295 = metal::min(t294, 1.0);
    t[3*frameCount + id] = metal::max(t295, 0.0);
  }
  #pragma clang diagnostic pop
}



// KERNEL 6
// Kind: simd
// ThreadCountScale Optional(64)
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
  float t5548 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5548)) {
    /* loadGlobal(276) - handled in variable access */
    /* loadGlobal(256) - handled in variable access */
    int t297 = id;
    int t298 = t297 / 64;
    uint _frameIndex = (uint)(t298);
    int t299 = t298 * 64;
    int t300 = t297 - t299;
    float t301 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t302 = t301 < 0.0;
    float t303 = t301 + 61.0;
    float t304 = metal::select(t301, t303, t302 > 0.0);
    float t305 = metal::floor(t304);
    float t306 = t305 + 1.0;
    float t307 = t306 >= 61.0;
    float t308 = metal::select(t306, 0.0, t307 > 0.0);
    float t309 = t304 - t305;
    float t310 = 1.0 - t309;
    float t311 = t298 * 64.0;
    float t312 = (float)t300;
    float t313 = t305 * 64.0;
    float t314 = t313 + t312;
    int t315 = (int)t314;
    float t316 = memory[33012 + t315];
    float t317 = t308 * 64.0;
    float t318 = t317 + t312;
    int t319 = (int)t318;
    float t320 = memory[33012 + t319];
    float t321 = t310 * t316;
    float t322 = t309 * t320;
    float t323 = t321 + t322;
    float t324 = t311 + t312;
    int t325 = (int)t324;
    memory[48948 + t325] = t323;
    int t327 = (int)t324;
    memory[2146100 + t327] = t323;
    float t329 = memory[21235 + t300];
    float t330 = t329 * t[2*frameCount + _frameIndex];
    int t331 = _frameIndex;
    int t332 = t331 * 64;
    int t333 = t332 + t300;
    memory[1097524 + t333] = t330;
  }
  #pragma clang diagnostic pop
}



// KERNEL 7
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_7(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(64)) {
    for (uint i = 0; i < frameCount; i += 1) {
      int t335 = id;
      int t336 = i;
      int t337 = t336 * 64;
      int t338 = t337 + t335;
      float t339 = memory[1097524 + t338];
      float t340 = (t339 * 6.25e-05);
      float t341 = memory[21300 + t335];
      float t342 = t341 + t340;
      float t343 = metal::select(t342, 0.0, 0.0 > 0.0);
      float t344 = metal::floor(t343);
      float t345 = t343 - t344;
      float t346 = t345 >= 1.0;
      float t347 = t345 - 1.0;
      float t348 = metal::select(t345, t347, t346 > 0.0);
      float t349 = metal::select(t348, 0.0, 0.0 > 0.0);
      memory[21300 + t335] = t349;
      int t351 = i;
      int t352 = t351 * 64;
      int t353 = t352 + t335;
      memory[48948 + t353] = t341;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 8
// Kind: simd
// ThreadCountScale Optional(64)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_8(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5549 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5549)) {
    int t355 = id;
    int t356 = t355 / 64;
    uint _frameIndex = (uint)(t356);
    int t357 = t356 * 64;
    int t358 = t355 - t357;
    int t359 = _frameIndex;
    int t360 = t359 * 64;
    int t361 = t360 + t358;
    float t362 = memory[48948 + t361];
    float t363 = t362 * 6.283185;
    float t364 = metal::sin(t363);
    int t365 = _frameIndex;
    int t366 = t365 * 64;
    int t367 = t366 + t358;
    memory[3194676 + t367] = t364;
    int t369 = _frameIndex;
    int t370 = t369 * 64;
    int t371 = t370 + t358;
    float t372 = memory[2146100 + t371];
    float t373 = t364 * t372;
    int t374 = _frameIndex;
    int t375 = t374 * 64;
    int t376 = t375 + t358;
    memory[1097524 + t376] = t373;
  }
  #pragma clang diagnostic pop
}



// KERNEL 9
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_9(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(378), value: global(378)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    t[4*frameCount + id] = 0.0;
    for (uint t379 = 0; t379 < 64; t379++) {
      int t380 = id;
      int t381 = t380 * 64;
      int t382 = t381 + t379;
      float t383 = memory[1097524 + t382];
      float t384 = t[4*frameCount + id] + t383;
      t[4*frameCount + id] = t384;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(406), value: global(406)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(405), value: global(405)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(404), value: global(404)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(386), value: global(386)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(378) - handled in variable access */
    /* loadGlobal(296) - handled in variable access */
    /* loadGlobal(256) - handled in variable access */
    t[5*frameCount + id] = t[4*frameCount + id] * t[3*frameCount + id];
    float t387 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t388 = t387 < 0.0;
    float t389 = t387 + 61.0;
    float t390 = metal::select(t387, t389, t388 > 0.0);
    float t391 = t390;
    float t392 = metal::floor(t391);
    float t393 = t391 - t392;
    float t394 = t392 + 1.0;
    float t395 = t394 >= 61.0;
    float t396 = metal::select(t394, 0.0, t395 > 0.0);
    int t397 = (int)t392;
    float t398 = memory[48820 + t397];
    int t399 = (int)t396;
    float t400 = memory[48820 + t399];
    float t401 = 1.0 - t393;
    float t402 = t398 * t401;
    float t403 = t400 * t393;
    t[6*frameCount + id] = t402 + t403;
    t[7*frameCount + id] = t[5*frameCount + id] * t[6*frameCount + id];
    t[8*frameCount + id] = t[7*frameCount + id] * 0.015625;
    float t407 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t408 = t407 < 0.0;
    float t409 = t407 + 61.0;
    float t410 = metal::select(t407, t409, t408 > 0.0);
    float t411 = t410;
    float t412 = metal::floor(t411);
    float t413 = t411 - t412;
    float t414 = t412 + 1.0;
    float t415 = t414 >= 61.0;
    float t416 = metal::select(t414, 0.0, t415 > 0.0);
    int t417 = (int)t412;
    float t418 = memory[48884 + t417];
    int t419 = (int)t416;
    float t420 = memory[48884 + t419];
    float t421 = 1.0 - t413;
    float t422 = t418 * t421;
    float t423 = t420 * t413;
    float t424 = t422 + t423;
  }
  #pragma clang diagnostic pop
}



// KERNEL 11
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(425), value: global(425)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[9*frameCount + i] = memory[532827829];
      float t426 = t[9*frameCount + i] + 1.0;
      float t427 = metal::select(t426, 0.0, 0.0 > 0.0);
      float t428 = t427;
      float t429 = (t428 * 6.1035156e-05);
      float t430 = metal::floor(t429);
      float t431 = t430 * 16384.0;
      float t432 = t427 - t431;
      memory[532827829] = t432;
      float t434 = t432 >= 16384.0;
      if (t434) {
        float t436 = t432 - 16384.0;
        memory[532827829] = t436;
      }
      if (0.0) {
        memory[532827829] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 12
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(459), value: global(459)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(425) - handled in variable access */
    float t442 = (t[9*frameCount + id] - metal::floor(t[9*frameCount + id] / 16384.0) * 16384.0);
    float t443 = t442 < 0.0;
    float t444 = t442 + 16384.0;
    float t445 = metal::select(t442, t444, t443 > 0.0);
    float t446 = t445;
    float t447 = metal::floor(t446);
    float t448 = t446 - t447;
    float t449 = t447 + 1.0;
    float t450 = t449 >= 16384.0;
    float t451 = metal::select(t449, 0.0, t450 > 0.0);
    int t452 = (int)t447;
    float t453 = memory[4729 + t452];
    int t454 = (int)t451;
    float t455 = memory[4729 + t454];
    float t456 = 1.0 - t448;
    float t457 = t453 * t456;
    float t458 = t455 * t448;
    t[10*frameCount + id] = t457 + t458;
  }
  #pragma clang diagnostic pop
}



// KERNEL 13
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(460), value: global(460)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[11*frameCount + i] = memory[532827830];
      float t461 = t[11*frameCount + i] + 1.0;
      float t462 = metal::select(t461, 0.0, 0.0 > 0.0);
      float t463 = t462;
      float t464 = (t463 * 0.0078125);
      float t465 = metal::floor(t464);
      float t466 = t465 * 128.0;
      float t467 = t462 - t466;
      memory[532827830] = t467;
      float t469 = t467 >= 128.0;
      if (t469) {
        float t471 = t467 - 128.0;
        memory[532827830] = t471;
      }
      if (0.0) {
        memory[532827830] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 14
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(481), value: global(481)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(460) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    /* loadGlobal(406) - handled in variable access */
    int t477 = id;
    int t478 = t477 * 1024;
    int t479 = t477 * 257;
    float t480 = t[11*frameCount + id] == 0.0;
    t[12*frameCount + id] = 0.0;
    if (t480) {
      for (uint _pr483 = 0; _pr483 < 512; _pr483++) {
        float t484 = (float)_pr483;
        float t485 = 6.283185 * t484;
        float t486 = (t485 * 0.0019569471);
        float t487 = metal::cos(t486);
        float t488 = 1.0 - t487;
        float t489 = 0.5 * t488;
        float t490 = (float)t477;
        float t491 = t490 - 511.0;
        float t492 = t491 + t484;
        float t493 = (t492 < 0 || t492 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t492];
        float t494 = (t492 < 0 || t492 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t492];
        int t495 = t478 + _pr483;
        float t496 = t493 * t489;
        memory[4243252 + t495] = t496;
        int t498 = t478 + _pr483;
        int t499 = t498 + 512;
        memory[4243252 + t499] = 0.0;
        int t501 = t478 + _pr483;
        float t502 = t494 * t489;
        memory[21020468 + t501] = t502;
        int t504 = t478 + _pr483;
        int t505 = t504 + 512;
        memory[21020468 + t505] = 0.0;
        memory[21300 + (int)_pr483] = t489;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t509 = 0; t509 < 512; t509++) {
        float t510 = (float)t509;
        float t511 = (t510 - metal::floor(t510 / 2.0) * 2.0);
        float t512 = t511;
        float t513 = (t510 * 0.5);
        float t514 = metal::floor(t513);
        float t515 = t512 * 2.0;
        float t516 = (t514 - metal::floor(t514 / 2.0) * 2.0);
        float t517 = t515 + t516;
        float t518 = (t514 * 0.5);
        float t519 = metal::floor(t518);
        float t520 = t517 * 2.0;
        float t521 = (t519 - metal::floor(t519 / 2.0) * 2.0);
        float t522 = t520 + t521;
        float t523 = (t519 * 0.5);
        float t524 = metal::floor(t523);
        float t525 = t522 * 2.0;
        float t526 = (t524 - metal::floor(t524 / 2.0) * 2.0);
        float t527 = t525 + t526;
        float t528 = (t524 * 0.5);
        float t529 = metal::floor(t528);
        float t530 = t527 * 2.0;
        float t531 = (t529 - metal::floor(t529 / 2.0) * 2.0);
        float t532 = t530 + t531;
        float t533 = (t529 * 0.5);
        float t534 = metal::floor(t533);
        float t535 = t532 * 2.0;
        float t536 = (t534 - metal::floor(t534 / 2.0) * 2.0);
        float t537 = t535 + t536;
        float t538 = (t534 * 0.5);
        float t539 = metal::floor(t538);
        float t540 = t537 * 2.0;
        float t541 = (t539 - metal::floor(t539 / 2.0) * 2.0);
        float t542 = t540 + t541;
        float t543 = (t539 * 0.5);
        float t544 = metal::floor(t543);
        float t545 = t542 * 2.0;
        float t546 = (t544 - metal::floor(t544 / 2.0) * 2.0);
        float t547 = t545 + t546;
        float t548 = (t544 * 0.5);
        float t549 = metal::floor(t548);
        float t550 = t547 * 2.0;
        float t551 = (t549 - metal::floor(t549 / 2.0) * 2.0);
        float t552 = t550 + t551;
        float t553 = (t549 * 0.5);
        float t554 = metal::floor(t553);
        float t555 = (float)t509;
        float t556 = t555 < t552;
        int t557 = (int)t552;
        int t558 = t478 + t509;
        float t559 = memory[4243252 + t558];
        int t560 = t478 + t509;
        int t561 = t560 + 512;
        float t562 = memory[4243252 + t561];
        int t563 = t478 + t557;
        float t564 = memory[4243252 + t563];
        int t565 = t478 + t557;
        int t566 = t565 + 512;
        float t567 = memory[4243252 + t566];
        float t568 = metal::select(t559, t564, t556 > 0.0);
        float t569 = metal::select(t562, t567, t556 > 0.0);
        float t570 = metal::select(t564, t559, t556 > 0.0);
        float t571 = metal::select(t567, t562, t556 > 0.0);
        int t572 = t478 + t509;
        memory[4243252 + t572] = t568;
        int t574 = t478 + t509;
        int t575 = t574 + 512;
        memory[4243252 + t575] = t569;
        int t577 = t478 + t557;
        memory[4243252 + t577] = t570;
        int t579 = t478 + t557;
        int t580 = t579 + 512;
        memory[4243252 + t580] = t571;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr583 = 0; _pr583 < 256; _pr583++) {
        float t584 = (float)_pr583;
        float t585 = t584;
        float t586 = metal::floor(t585);
        float t587 = t586;
        float t588 = t584 - t587;
        float t589 = t586 * 2.0;
        float t590 = t589 + t588;
        float t591 = t590 + 1.0;
        float t592 = -6.283185 * t588;
        float t593 = (t592 * 0.5);
        float t594 = metal::cos(t593);
        float t595 = metal::sin(t593);
        int t596 = (int)t590;
        int t597 = (int)t591;
        int t598 = t478 + t596;
        float t599 = memory[4243252 + t598];
        int t600 = t478 + t596;
        int t601 = t600 + 512;
        float t602 = memory[4243252 + t601];
        int t603 = t478 + t597;
        float t604 = memory[4243252 + t603];
        int t605 = t478 + t597;
        int t606 = t605 + 512;
        float t607 = memory[4243252 + t606];
        float t608 = t594 * t604;
        float t609 = t595 * t607;
        float t610 = t608 - t609;
        float t611 = t594 * t607;
        float t612 = t595 * t604;
        float t613 = t611 + t612;
        int t614 = t478 + t596;
        float t615 = t599 + t610;
        memory[4243252 + t614] = t615;
        int t617 = t478 + t596;
        int t618 = t617 + 512;
        float t619 = t602 + t613;
        memory[4243252 + t618] = t619;
        int t621 = t478 + t597;
        float t622 = t599 - t610;
        memory[4243252 + t621] = t622;
        int t624 = t478 + t597;
        int t625 = t624 + 512;
        float t626 = t602 - t613;
        memory[4243252 + t625] = t626;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr629 = 0; _pr629 < 256; _pr629++) {
        float t630 = (float)_pr629;
        float t631 = (t630 * 0.5);
        float t632 = metal::floor(t631);
        float t633 = t632 * 2.0;
        float t634 = t630 - t633;
        float t635 = t632 * 4.0;
        float t636 = t635 + t634;
        float t637 = t636 + 2.0;
        float t638 = -6.283185 * t634;
        float t639 = (t638 * 0.25);
        float t640 = metal::cos(t639);
        float t641 = metal::sin(t639);
        int t642 = (int)t636;
        int t643 = (int)t637;
        int t644 = t478 + t642;
        float t645 = memory[4243252 + t644];
        int t646 = t478 + t642;
        int t647 = t646 + 512;
        float t648 = memory[4243252 + t647];
        int t649 = t478 + t643;
        float t650 = memory[4243252 + t649];
        int t651 = t478 + t643;
        int t652 = t651 + 512;
        float t653 = memory[4243252 + t652];
        float t654 = t640 * t650;
        float t655 = t641 * t653;
        float t656 = t654 - t655;
        float t657 = t640 * t653;
        float t658 = t641 * t650;
        float t659 = t657 + t658;
        int t660 = t478 + t642;
        float t661 = t645 + t656;
        memory[4243252 + t660] = t661;
        int t663 = t478 + t642;
        int t664 = t663 + 512;
        float t665 = t648 + t659;
        memory[4243252 + t664] = t665;
        int t667 = t478 + t643;
        float t668 = t645 - t656;
        memory[4243252 + t667] = t668;
        int t670 = t478 + t643;
        int t671 = t670 + 512;
        float t672 = t648 - t659;
        memory[4243252 + t671] = t672;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr675 = 0; _pr675 < 256; _pr675++) {
        float t676 = (float)_pr675;
        float t677 = (t676 * 0.25);
        float t678 = metal::floor(t677);
        float t679 = t678 * 4.0;
        float t680 = t676 - t679;
        float t681 = t678 * 8.0;
        float t682 = t681 + t680;
        float t683 = t682 + 4.0;
        float t684 = -6.283185 * t680;
        float t685 = (t684 * 0.125);
        float t686 = metal::cos(t685);
        float t687 = metal::sin(t685);
        int t688 = (int)t682;
        int t689 = (int)t683;
        int t690 = t478 + t688;
        float t691 = memory[4243252 + t690];
        int t692 = t478 + t688;
        int t693 = t692 + 512;
        float t694 = memory[4243252 + t693];
        int t695 = t478 + t689;
        float t696 = memory[4243252 + t695];
        int t697 = t478 + t689;
        int t698 = t697 + 512;
        float t699 = memory[4243252 + t698];
        float t700 = t686 * t696;
        float t701 = t687 * t699;
        float t702 = t700 - t701;
        float t703 = t686 * t699;
        float t704 = t687 * t696;
        float t705 = t703 + t704;
        int t706 = t478 + t688;
        float t707 = t691 + t702;
        memory[4243252 + t706] = t707;
        int t709 = t478 + t688;
        int t710 = t709 + 512;
        float t711 = t694 + t705;
        memory[4243252 + t710] = t711;
        int t713 = t478 + t689;
        float t714 = t691 - t702;
        memory[4243252 + t713] = t714;
        int t716 = t478 + t689;
        int t717 = t716 + 512;
        float t718 = t694 - t705;
        memory[4243252 + t717] = t718;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr721 = 0; _pr721 < 256; _pr721++) {
        float t722 = (float)_pr721;
        float t723 = (t722 * 0.125);
        float t724 = metal::floor(t723);
        float t725 = t724 * 8.0;
        float t726 = t722 - t725;
        float t727 = t724 * 16.0;
        float t728 = t727 + t726;
        float t729 = t728 + 8.0;
        float t730 = -6.283185 * t726;
        float t731 = (t730 * 0.0625);
        float t732 = metal::cos(t731);
        float t733 = metal::sin(t731);
        int t734 = (int)t728;
        int t735 = (int)t729;
        int t736 = t478 + t734;
        float t737 = memory[4243252 + t736];
        int t738 = t478 + t734;
        int t739 = t738 + 512;
        float t740 = memory[4243252 + t739];
        int t741 = t478 + t735;
        float t742 = memory[4243252 + t741];
        int t743 = t478 + t735;
        int t744 = t743 + 512;
        float t745 = memory[4243252 + t744];
        float t746 = t732 * t742;
        float t747 = t733 * t745;
        float t748 = t746 - t747;
        float t749 = t732 * t745;
        float t750 = t733 * t742;
        float t751 = t749 + t750;
        int t752 = t478 + t734;
        float t753 = t737 + t748;
        memory[4243252 + t752] = t753;
        int t755 = t478 + t734;
        int t756 = t755 + 512;
        float t757 = t740 + t751;
        memory[4243252 + t756] = t757;
        int t759 = t478 + t735;
        float t760 = t737 - t748;
        memory[4243252 + t759] = t760;
        int t762 = t478 + t735;
        int t763 = t762 + 512;
        float t764 = t740 - t751;
        memory[4243252 + t763] = t764;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr767 = 0; _pr767 < 256; _pr767++) {
        float t768 = (float)_pr767;
        float t769 = (t768 * 0.0625);
        float t770 = metal::floor(t769);
        float t771 = t770 * 16.0;
        float t772 = t768 - t771;
        float t773 = t770 * 32.0;
        float t774 = t773 + t772;
        float t775 = t774 + 16.0;
        float t776 = -6.283185 * t772;
        float t777 = (t776 * 0.03125);
        float t778 = metal::cos(t777);
        float t779 = metal::sin(t777);
        int t780 = (int)t774;
        int t781 = (int)t775;
        int t782 = t478 + t780;
        float t783 = memory[4243252 + t782];
        int t784 = t478 + t780;
        int t785 = t784 + 512;
        float t786 = memory[4243252 + t785];
        int t787 = t478 + t781;
        float t788 = memory[4243252 + t787];
        int t789 = t478 + t781;
        int t790 = t789 + 512;
        float t791 = memory[4243252 + t790];
        float t792 = t778 * t788;
        float t793 = t779 * t791;
        float t794 = t792 - t793;
        float t795 = t778 * t791;
        float t796 = t779 * t788;
        float t797 = t795 + t796;
        int t798 = t478 + t780;
        float t799 = t783 + t794;
        memory[4243252 + t798] = t799;
        int t801 = t478 + t780;
        int t802 = t801 + 512;
        float t803 = t786 + t797;
        memory[4243252 + t802] = t803;
        int t805 = t478 + t781;
        float t806 = t783 - t794;
        memory[4243252 + t805] = t806;
        int t808 = t478 + t781;
        int t809 = t808 + 512;
        float t810 = t786 - t797;
        memory[4243252 + t809] = t810;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr813 = 0; _pr813 < 256; _pr813++) {
        float t814 = (float)_pr813;
        float t815 = (t814 * 0.03125);
        float t816 = metal::floor(t815);
        float t817 = t816 * 32.0;
        float t818 = t814 - t817;
        float t819 = t816 * 64.0;
        float t820 = t819 + t818;
        float t821 = t820 + 32.0;
        float t822 = -6.283185 * t818;
        float t823 = (t822 * 0.015625);
        float t824 = metal::cos(t823);
        float t825 = metal::sin(t823);
        int t826 = (int)t820;
        int t827 = (int)t821;
        int t828 = t478 + t826;
        float t829 = memory[4243252 + t828];
        int t830 = t478 + t826;
        int t831 = t830 + 512;
        float t832 = memory[4243252 + t831];
        int t833 = t478 + t827;
        float t834 = memory[4243252 + t833];
        int t835 = t478 + t827;
        int t836 = t835 + 512;
        float t837 = memory[4243252 + t836];
        float t838 = t824 * t834;
        float t839 = t825 * t837;
        float t840 = t838 - t839;
        float t841 = t824 * t837;
        float t842 = t825 * t834;
        float t843 = t841 + t842;
        int t844 = t478 + t826;
        float t845 = t829 + t840;
        memory[4243252 + t844] = t845;
        int t847 = t478 + t826;
        int t848 = t847 + 512;
        float t849 = t832 + t843;
        memory[4243252 + t848] = t849;
        int t851 = t478 + t827;
        float t852 = t829 - t840;
        memory[4243252 + t851] = t852;
        int t854 = t478 + t827;
        int t855 = t854 + 512;
        float t856 = t832 - t843;
        memory[4243252 + t855] = t856;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr859 = 0; _pr859 < 256; _pr859++) {
        float t860 = (float)_pr859;
        float t861 = (t860 * 0.015625);
        float t862 = metal::floor(t861);
        float t863 = t862 * 64.0;
        float t864 = t860 - t863;
        float t865 = t862 * 128.0;
        float t866 = t865 + t864;
        float t867 = t866 + 64.0;
        float t868 = -6.283185 * t864;
        float t869 = (t868 * 0.0078125);
        float t870 = metal::cos(t869);
        float t871 = metal::sin(t869);
        int t872 = (int)t866;
        int t873 = (int)t867;
        int t874 = t478 + t872;
        float t875 = memory[4243252 + t874];
        int t876 = t478 + t872;
        int t877 = t876 + 512;
        float t878 = memory[4243252 + t877];
        int t879 = t478 + t873;
        float t880 = memory[4243252 + t879];
        int t881 = t478 + t873;
        int t882 = t881 + 512;
        float t883 = memory[4243252 + t882];
        float t884 = t870 * t880;
        float t885 = t871 * t883;
        float t886 = t884 - t885;
        float t887 = t870 * t883;
        float t888 = t871 * t880;
        float t889 = t887 + t888;
        int t890 = t478 + t872;
        float t891 = t875 + t886;
        memory[4243252 + t890] = t891;
        int t893 = t478 + t872;
        int t894 = t893 + 512;
        float t895 = t878 + t889;
        memory[4243252 + t894] = t895;
        int t897 = t478 + t873;
        float t898 = t875 - t886;
        memory[4243252 + t897] = t898;
        int t900 = t478 + t873;
        int t901 = t900 + 512;
        float t902 = t878 - t889;
        memory[4243252 + t901] = t902;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr905 = 0; _pr905 < 256; _pr905++) {
        float t906 = (float)_pr905;
        float t907 = (t906 * 0.0078125);
        float t908 = metal::floor(t907);
        float t909 = t908 * 128.0;
        float t910 = t906 - t909;
        float t911 = t908 * 256.0;
        float t912 = t911 + t910;
        float t913 = t912 + 128.0;
        float t914 = -6.283185 * t910;
        float t915 = (t914 * 0.00390625);
        float t916 = metal::cos(t915);
        float t917 = metal::sin(t915);
        int t918 = (int)t912;
        int t919 = (int)t913;
        int t920 = t478 + t918;
        float t921 = memory[4243252 + t920];
        int t922 = t478 + t918;
        int t923 = t922 + 512;
        float t924 = memory[4243252 + t923];
        int t925 = t478 + t919;
        float t926 = memory[4243252 + t925];
        int t927 = t478 + t919;
        int t928 = t927 + 512;
        float t929 = memory[4243252 + t928];
        float t930 = t916 * t926;
        float t931 = t917 * t929;
        float t932 = t930 - t931;
        float t933 = t916 * t929;
        float t934 = t917 * t926;
        float t935 = t933 + t934;
        int t936 = t478 + t918;
        float t937 = t921 + t932;
        memory[4243252 + t936] = t937;
        int t939 = t478 + t918;
        int t940 = t939 + 512;
        float t941 = t924 + t935;
        memory[4243252 + t940] = t941;
        int t943 = t478 + t919;
        float t944 = t921 - t932;
        memory[4243252 + t943] = t944;
        int t946 = t478 + t919;
        int t947 = t946 + 512;
        float t948 = t924 - t935;
        memory[4243252 + t947] = t948;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr951 = 0; _pr951 < 256; _pr951++) {
        float t952 = (float)_pr951;
        float t953 = (t952 * 0.00390625);
        float t954 = metal::floor(t953);
        float t955 = t954 * 256.0;
        float t956 = t952 - t955;
        float t957 = t954 * 512.0;
        float t958 = t957 + t956;
        float t959 = t958 + 256.0;
        float t960 = -6.283185 * t956;
        float t961 = (t960 * 0.001953125);
        float t962 = metal::cos(t961);
        float t963 = metal::sin(t961);
        int t964 = (int)t958;
        int t965 = (int)t959;
        int t966 = t478 + t964;
        float t967 = memory[4243252 + t966];
        int t968 = t478 + t964;
        int t969 = t968 + 512;
        float t970 = memory[4243252 + t969];
        int t971 = t478 + t965;
        float t972 = memory[4243252 + t971];
        int t973 = t478 + t965;
        int t974 = t973 + 512;
        float t975 = memory[4243252 + t974];
        float t976 = t962 * t972;
        float t977 = t963 * t975;
        float t978 = t976 - t977;
        float t979 = t962 * t975;
        float t980 = t963 * t972;
        float t981 = t979 + t980;
        int t982 = t478 + t964;
        float t983 = t967 + t978;
        memory[4243252 + t982] = t983;
        int t985 = t478 + t964;
        int t986 = t985 + 512;
        float t987 = t970 + t981;
        memory[4243252 + t986] = t987;
        int t989 = t478 + t965;
        float t990 = t967 - t978;
        memory[4243252 + t989] = t990;
        int t992 = t478 + t965;
        int t993 = t992 + 512;
        float t994 = t970 - t981;
        memory[4243252 + t993] = t994;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t997 = 0; t997 < 512; t997++) {
        float t998 = (float)t997;
        float t999 = (t998 - metal::floor(t998 / 2.0) * 2.0);
        float t1000 = t999;
        float t1001 = (t998 * 0.5);
        float t1002 = metal::floor(t1001);
        float t1003 = t1000 * 2.0;
        float t1004 = (t1002 - metal::floor(t1002 / 2.0) * 2.0);
        float t1005 = t1003 + t1004;
        float t1006 = (t1002 * 0.5);
        float t1007 = metal::floor(t1006);
        float t1008 = t1005 * 2.0;
        float t1009 = (t1007 - metal::floor(t1007 / 2.0) * 2.0);
        float t1010 = t1008 + t1009;
        float t1011 = (t1007 * 0.5);
        float t1012 = metal::floor(t1011);
        float t1013 = t1010 * 2.0;
        float t1014 = (t1012 - metal::floor(t1012 / 2.0) * 2.0);
        float t1015 = t1013 + t1014;
        float t1016 = (t1012 * 0.5);
        float t1017 = metal::floor(t1016);
        float t1018 = t1015 * 2.0;
        float t1019 = (t1017 - metal::floor(t1017 / 2.0) * 2.0);
        float t1020 = t1018 + t1019;
        float t1021 = (t1017 * 0.5);
        float t1022 = metal::floor(t1021);
        float t1023 = t1020 * 2.0;
        float t1024 = (t1022 - metal::floor(t1022 / 2.0) * 2.0);
        float t1025 = t1023 + t1024;
        float t1026 = (t1022 * 0.5);
        float t1027 = metal::floor(t1026);
        float t1028 = t1025 * 2.0;
        float t1029 = (t1027 - metal::floor(t1027 / 2.0) * 2.0);
        float t1030 = t1028 + t1029;
        float t1031 = (t1027 * 0.5);
        float t1032 = metal::floor(t1031);
        float t1033 = t1030 * 2.0;
        float t1034 = (t1032 - metal::floor(t1032 / 2.0) * 2.0);
        float t1035 = t1033 + t1034;
        float t1036 = (t1032 * 0.5);
        float t1037 = metal::floor(t1036);
        float t1038 = t1035 * 2.0;
        float t1039 = (t1037 - metal::floor(t1037 / 2.0) * 2.0);
        float t1040 = t1038 + t1039;
        float t1041 = (t1037 * 0.5);
        float t1042 = metal::floor(t1041);
        float t1043 = (float)t997;
        float t1044 = t1043 < t1040;
        int t1045 = (int)t1040;
        int t1046 = t478 + t997;
        float t1047 = memory[21020468 + t1046];
        int t1048 = t478 + t997;
        int t1049 = t1048 + 512;
        float t1050 = memory[21020468 + t1049];
        int t1051 = t478 + t1045;
        float t1052 = memory[21020468 + t1051];
        int t1053 = t478 + t1045;
        int t1054 = t1053 + 512;
        float t1055 = memory[21020468 + t1054];
        float t1056 = metal::select(t1047, t1052, t1044 > 0.0);
        float t1057 = metal::select(t1050, t1055, t1044 > 0.0);
        float t1058 = metal::select(t1052, t1047, t1044 > 0.0);
        float t1059 = metal::select(t1055, t1050, t1044 > 0.0);
        int t1060 = t478 + t997;
        memory[21020468 + t1060] = t1056;
        int t1062 = t478 + t997;
        int t1063 = t1062 + 512;
        memory[21020468 + t1063] = t1057;
        int t1065 = t478 + t1045;
        memory[21020468 + t1065] = t1058;
        int t1067 = t478 + t1045;
        int t1068 = t1067 + 512;
        memory[21020468 + t1068] = t1059;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1071 = 0; _pr1071 < 256; _pr1071++) {
        float t1072 = (float)_pr1071;
        float t1073 = t1072;
        float t1074 = metal::floor(t1073);
        float t1075 = t1074;
        float t1076 = t1072 - t1075;
        float t1077 = t1074 * 2.0;
        float t1078 = t1077 + t1076;
        float t1079 = t1078 + 1.0;
        float t1080 = -6.283185 * t1076;
        float t1081 = (t1080 * 0.5);
        float t1082 = metal::cos(t1081);
        float t1083 = metal::sin(t1081);
        int t1084 = (int)t1078;
        int t1085 = (int)t1079;
        int t1086 = t478 + t1084;
        float t1087 = memory[21020468 + t1086];
        int t1088 = t478 + t1084;
        int t1089 = t1088 + 512;
        float t1090 = memory[21020468 + t1089];
        int t1091 = t478 + t1085;
        float t1092 = memory[21020468 + t1091];
        int t1093 = t478 + t1085;
        int t1094 = t1093 + 512;
        float t1095 = memory[21020468 + t1094];
        float t1096 = t1082 * t1092;
        float t1097 = t1083 * t1095;
        float t1098 = t1096 - t1097;
        float t1099 = t1082 * t1095;
        float t1100 = t1083 * t1092;
        float t1101 = t1099 + t1100;
        int t1102 = t478 + t1084;
        float t1103 = t1087 + t1098;
        memory[21020468 + t1102] = t1103;
        int t1105 = t478 + t1084;
        int t1106 = t1105 + 512;
        float t1107 = t1090 + t1101;
        memory[21020468 + t1106] = t1107;
        int t1109 = t478 + t1085;
        float t1110 = t1087 - t1098;
        memory[21020468 + t1109] = t1110;
        int t1112 = t478 + t1085;
        int t1113 = t1112 + 512;
        float t1114 = t1090 - t1101;
        memory[21020468 + t1113] = t1114;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1117 = 0; _pr1117 < 256; _pr1117++) {
        float t1118 = (float)_pr1117;
        float t1119 = (t1118 * 0.5);
        float t1120 = metal::floor(t1119);
        float t1121 = t1120 * 2.0;
        float t1122 = t1118 - t1121;
        float t1123 = t1120 * 4.0;
        float t1124 = t1123 + t1122;
        float t1125 = t1124 + 2.0;
        float t1126 = -6.283185 * t1122;
        float t1127 = (t1126 * 0.25);
        float t1128 = metal::cos(t1127);
        float t1129 = metal::sin(t1127);
        int t1130 = (int)t1124;
        int t1131 = (int)t1125;
        int t1132 = t478 + t1130;
        float t1133 = memory[21020468 + t1132];
        int t1134 = t478 + t1130;
        int t1135 = t1134 + 512;
        float t1136 = memory[21020468 + t1135];
        int t1137 = t478 + t1131;
        float t1138 = memory[21020468 + t1137];
        int t1139 = t478 + t1131;
        int t1140 = t1139 + 512;
        float t1141 = memory[21020468 + t1140];
        float t1142 = t1128 * t1138;
        float t1143 = t1129 * t1141;
        float t1144 = t1142 - t1143;
        float t1145 = t1128 * t1141;
        float t1146 = t1129 * t1138;
        float t1147 = t1145 + t1146;
        int t1148 = t478 + t1130;
        float t1149 = t1133 + t1144;
        memory[21020468 + t1148] = t1149;
        int t1151 = t478 + t1130;
        int t1152 = t1151 + 512;
        float t1153 = t1136 + t1147;
        memory[21020468 + t1152] = t1153;
        int t1155 = t478 + t1131;
        float t1156 = t1133 - t1144;
        memory[21020468 + t1155] = t1156;
        int t1158 = t478 + t1131;
        int t1159 = t1158 + 512;
        float t1160 = t1136 - t1147;
        memory[21020468 + t1159] = t1160;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1163 = 0; _pr1163 < 256; _pr1163++) {
        float t1164 = (float)_pr1163;
        float t1165 = (t1164 * 0.25);
        float t1166 = metal::floor(t1165);
        float t1167 = t1166 * 4.0;
        float t1168 = t1164 - t1167;
        float t1169 = t1166 * 8.0;
        float t1170 = t1169 + t1168;
        float t1171 = t1170 + 4.0;
        float t1172 = -6.283185 * t1168;
        float t1173 = (t1172 * 0.125);
        float t1174 = metal::cos(t1173);
        float t1175 = metal::sin(t1173);
        int t1176 = (int)t1170;
        int t1177 = (int)t1171;
        int t1178 = t478 + t1176;
        float t1179 = memory[21020468 + t1178];
        int t1180 = t478 + t1176;
        int t1181 = t1180 + 512;
        float t1182 = memory[21020468 + t1181];
        int t1183 = t478 + t1177;
        float t1184 = memory[21020468 + t1183];
        int t1185 = t478 + t1177;
        int t1186 = t1185 + 512;
        float t1187 = memory[21020468 + t1186];
        float t1188 = t1174 * t1184;
        float t1189 = t1175 * t1187;
        float t1190 = t1188 - t1189;
        float t1191 = t1174 * t1187;
        float t1192 = t1175 * t1184;
        float t1193 = t1191 + t1192;
        int t1194 = t478 + t1176;
        float t1195 = t1179 + t1190;
        memory[21020468 + t1194] = t1195;
        int t1197 = t478 + t1176;
        int t1198 = t1197 + 512;
        float t1199 = t1182 + t1193;
        memory[21020468 + t1198] = t1199;
        int t1201 = t478 + t1177;
        float t1202 = t1179 - t1190;
        memory[21020468 + t1201] = t1202;
        int t1204 = t478 + t1177;
        int t1205 = t1204 + 512;
        float t1206 = t1182 - t1193;
        memory[21020468 + t1205] = t1206;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1209 = 0; _pr1209 < 256; _pr1209++) {
        float t1210 = (float)_pr1209;
        float t1211 = (t1210 * 0.125);
        float t1212 = metal::floor(t1211);
        float t1213 = t1212 * 8.0;
        float t1214 = t1210 - t1213;
        float t1215 = t1212 * 16.0;
        float t1216 = t1215 + t1214;
        float t1217 = t1216 + 8.0;
        float t1218 = -6.283185 * t1214;
        float t1219 = (t1218 * 0.0625);
        float t1220 = metal::cos(t1219);
        float t1221 = metal::sin(t1219);
        int t1222 = (int)t1216;
        int t1223 = (int)t1217;
        int t1224 = t478 + t1222;
        float t1225 = memory[21020468 + t1224];
        int t1226 = t478 + t1222;
        int t1227 = t1226 + 512;
        float t1228 = memory[21020468 + t1227];
        int t1229 = t478 + t1223;
        float t1230 = memory[21020468 + t1229];
        int t1231 = t478 + t1223;
        int t1232 = t1231 + 512;
        float t1233 = memory[21020468 + t1232];
        float t1234 = t1220 * t1230;
        float t1235 = t1221 * t1233;
        float t1236 = t1234 - t1235;
        float t1237 = t1220 * t1233;
        float t1238 = t1221 * t1230;
        float t1239 = t1237 + t1238;
        int t1240 = t478 + t1222;
        float t1241 = t1225 + t1236;
        memory[21020468 + t1240] = t1241;
        int t1243 = t478 + t1222;
        int t1244 = t1243 + 512;
        float t1245 = t1228 + t1239;
        memory[21020468 + t1244] = t1245;
        int t1247 = t478 + t1223;
        float t1248 = t1225 - t1236;
        memory[21020468 + t1247] = t1248;
        int t1250 = t478 + t1223;
        int t1251 = t1250 + 512;
        float t1252 = t1228 - t1239;
        memory[21020468 + t1251] = t1252;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1255 = 0; _pr1255 < 256; _pr1255++) {
        float t1256 = (float)_pr1255;
        float t1257 = (t1256 * 0.0625);
        float t1258 = metal::floor(t1257);
        float t1259 = t1258 * 16.0;
        float t1260 = t1256 - t1259;
        float t1261 = t1258 * 32.0;
        float t1262 = t1261 + t1260;
        float t1263 = t1262 + 16.0;
        float t1264 = -6.283185 * t1260;
        float t1265 = (t1264 * 0.03125);
        float t1266 = metal::cos(t1265);
        float t1267 = metal::sin(t1265);
        int t1268 = (int)t1262;
        int t1269 = (int)t1263;
        int t1270 = t478 + t1268;
        float t1271 = memory[21020468 + t1270];
        int t1272 = t478 + t1268;
        int t1273 = t1272 + 512;
        float t1274 = memory[21020468 + t1273];
        int t1275 = t478 + t1269;
        float t1276 = memory[21020468 + t1275];
        int t1277 = t478 + t1269;
        int t1278 = t1277 + 512;
        float t1279 = memory[21020468 + t1278];
        float t1280 = t1266 * t1276;
        float t1281 = t1267 * t1279;
        float t1282 = t1280 - t1281;
        float t1283 = t1266 * t1279;
        float t1284 = t1267 * t1276;
        float t1285 = t1283 + t1284;
        int t1286 = t478 + t1268;
        float t1287 = t1271 + t1282;
        memory[21020468 + t1286] = t1287;
        int t1289 = t478 + t1268;
        int t1290 = t1289 + 512;
        float t1291 = t1274 + t1285;
        memory[21020468 + t1290] = t1291;
        int t1293 = t478 + t1269;
        float t1294 = t1271 - t1282;
        memory[21020468 + t1293] = t1294;
        int t1296 = t478 + t1269;
        int t1297 = t1296 + 512;
        float t1298 = t1274 - t1285;
        memory[21020468 + t1297] = t1298;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1301 = 0; _pr1301 < 256; _pr1301++) {
        float t1302 = (float)_pr1301;
        float t1303 = (t1302 * 0.03125);
        float t1304 = metal::floor(t1303);
        float t1305 = t1304 * 32.0;
        float t1306 = t1302 - t1305;
        float t1307 = t1304 * 64.0;
        float t1308 = t1307 + t1306;
        float t1309 = t1308 + 32.0;
        float t1310 = -6.283185 * t1306;
        float t1311 = (t1310 * 0.015625);
        float t1312 = metal::cos(t1311);
        float t1313 = metal::sin(t1311);
        int t1314 = (int)t1308;
        int t1315 = (int)t1309;
        int t1316 = t478 + t1314;
        float t1317 = memory[21020468 + t1316];
        int t1318 = t478 + t1314;
        int t1319 = t1318 + 512;
        float t1320 = memory[21020468 + t1319];
        int t1321 = t478 + t1315;
        float t1322 = memory[21020468 + t1321];
        int t1323 = t478 + t1315;
        int t1324 = t1323 + 512;
        float t1325 = memory[21020468 + t1324];
        float t1326 = t1312 * t1322;
        float t1327 = t1313 * t1325;
        float t1328 = t1326 - t1327;
        float t1329 = t1312 * t1325;
        float t1330 = t1313 * t1322;
        float t1331 = t1329 + t1330;
        int t1332 = t478 + t1314;
        float t1333 = t1317 + t1328;
        memory[21020468 + t1332] = t1333;
        int t1335 = t478 + t1314;
        int t1336 = t1335 + 512;
        float t1337 = t1320 + t1331;
        memory[21020468 + t1336] = t1337;
        int t1339 = t478 + t1315;
        float t1340 = t1317 - t1328;
        memory[21020468 + t1339] = t1340;
        int t1342 = t478 + t1315;
        int t1343 = t1342 + 512;
        float t1344 = t1320 - t1331;
        memory[21020468 + t1343] = t1344;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1347 = 0; _pr1347 < 256; _pr1347++) {
        float t1348 = (float)_pr1347;
        float t1349 = (t1348 * 0.015625);
        float t1350 = metal::floor(t1349);
        float t1351 = t1350 * 64.0;
        float t1352 = t1348 - t1351;
        float t1353 = t1350 * 128.0;
        float t1354 = t1353 + t1352;
        float t1355 = t1354 + 64.0;
        float t1356 = -6.283185 * t1352;
        float t1357 = (t1356 * 0.0078125);
        float t1358 = metal::cos(t1357);
        float t1359 = metal::sin(t1357);
        int t1360 = (int)t1354;
        int t1361 = (int)t1355;
        int t1362 = t478 + t1360;
        float t1363 = memory[21020468 + t1362];
        int t1364 = t478 + t1360;
        int t1365 = t1364 + 512;
        float t1366 = memory[21020468 + t1365];
        int t1367 = t478 + t1361;
        float t1368 = memory[21020468 + t1367];
        int t1369 = t478 + t1361;
        int t1370 = t1369 + 512;
        float t1371 = memory[21020468 + t1370];
        float t1372 = t1358 * t1368;
        float t1373 = t1359 * t1371;
        float t1374 = t1372 - t1373;
        float t1375 = t1358 * t1371;
        float t1376 = t1359 * t1368;
        float t1377 = t1375 + t1376;
        int t1378 = t478 + t1360;
        float t1379 = t1363 + t1374;
        memory[21020468 + t1378] = t1379;
        int t1381 = t478 + t1360;
        int t1382 = t1381 + 512;
        float t1383 = t1366 + t1377;
        memory[21020468 + t1382] = t1383;
        int t1385 = t478 + t1361;
        float t1386 = t1363 - t1374;
        memory[21020468 + t1385] = t1386;
        int t1388 = t478 + t1361;
        int t1389 = t1388 + 512;
        float t1390 = t1366 - t1377;
        memory[21020468 + t1389] = t1390;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1393 = 0; _pr1393 < 256; _pr1393++) {
        float t1394 = (float)_pr1393;
        float t1395 = (t1394 * 0.0078125);
        float t1396 = metal::floor(t1395);
        float t1397 = t1396 * 128.0;
        float t1398 = t1394 - t1397;
        float t1399 = t1396 * 256.0;
        float t1400 = t1399 + t1398;
        float t1401 = t1400 + 128.0;
        float t1402 = -6.283185 * t1398;
        float t1403 = (t1402 * 0.00390625);
        float t1404 = metal::cos(t1403);
        float t1405 = metal::sin(t1403);
        int t1406 = (int)t1400;
        int t1407 = (int)t1401;
        int t1408 = t478 + t1406;
        float t1409 = memory[21020468 + t1408];
        int t1410 = t478 + t1406;
        int t1411 = t1410 + 512;
        float t1412 = memory[21020468 + t1411];
        int t1413 = t478 + t1407;
        float t1414 = memory[21020468 + t1413];
        int t1415 = t478 + t1407;
        int t1416 = t1415 + 512;
        float t1417 = memory[21020468 + t1416];
        float t1418 = t1404 * t1414;
        float t1419 = t1405 * t1417;
        float t1420 = t1418 - t1419;
        float t1421 = t1404 * t1417;
        float t1422 = t1405 * t1414;
        float t1423 = t1421 + t1422;
        int t1424 = t478 + t1406;
        float t1425 = t1409 + t1420;
        memory[21020468 + t1424] = t1425;
        int t1427 = t478 + t1406;
        int t1428 = t1427 + 512;
        float t1429 = t1412 + t1423;
        memory[21020468 + t1428] = t1429;
        int t1431 = t478 + t1407;
        float t1432 = t1409 - t1420;
        memory[21020468 + t1431] = t1432;
        int t1434 = t478 + t1407;
        int t1435 = t1434 + 512;
        float t1436 = t1412 - t1423;
        memory[21020468 + t1435] = t1436;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1439 = 0; _pr1439 < 256; _pr1439++) {
        float t1440 = (float)_pr1439;
        float t1441 = (t1440 * 0.00390625);
        float t1442 = metal::floor(t1441);
        float t1443 = t1442 * 256.0;
        float t1444 = t1440 - t1443;
        float t1445 = t1442 * 512.0;
        float t1446 = t1445 + t1444;
        float t1447 = t1446 + 256.0;
        float t1448 = -6.283185 * t1444;
        float t1449 = (t1448 * 0.001953125);
        float t1450 = metal::cos(t1449);
        float t1451 = metal::sin(t1449);
        int t1452 = (int)t1446;
        int t1453 = (int)t1447;
        int t1454 = t478 + t1452;
        float t1455 = memory[21020468 + t1454];
        int t1456 = t478 + t1452;
        int t1457 = t1456 + 512;
        float t1458 = memory[21020468 + t1457];
        int t1459 = t478 + t1453;
        float t1460 = memory[21020468 + t1459];
        int t1461 = t478 + t1453;
        int t1462 = t1461 + 512;
        float t1463 = memory[21020468 + t1462];
        float t1464 = t1450 * t1460;
        float t1465 = t1451 * t1463;
        float t1466 = t1464 - t1465;
        float t1467 = t1450 * t1463;
        float t1468 = t1451 * t1460;
        float t1469 = t1467 + t1468;
        int t1470 = t478 + t1452;
        float t1471 = t1455 + t1466;
        memory[21020468 + t1470] = t1471;
        int t1473 = t478 + t1452;
        int t1474 = t1473 + 512;
        float t1475 = t1458 + t1469;
        memory[21020468 + t1474] = t1475;
        int t1477 = t478 + t1453;
        float t1478 = t1455 - t1466;
        memory[21020468 + t1477] = t1478;
        int t1480 = t478 + t1453;
        int t1481 = t1480 + 512;
        float t1482 = t1458 - t1469;
        memory[21020468 + t1481] = t1482;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1485 = 0; _pr1485 < 257; _pr1485++) {
        int t1486 = t478 + _pr1485;
        float t1487 = memory[4243252 + t1486];
        int t1488 = t478 + _pr1485;
        int t1489 = t1488 + 512;
        float t1490 = memory[4243252 + t1489];
        float t1491 = t1487 * t1487;
        float t1492 = t1490 * t1490;
        float t1493 = t1491 + t1492;
        float t1494 = metal::sqrt(t1493);
        int t1495 = t479 + _pr1485;
        memory[37797684 + t1495] = t1494;
        int t1497 = t478 + _pr1485;
        float t1498 = memory[21020468 + t1497];
        int t1499 = t478 + _pr1485;
        int t1500 = t1499 + 512;
        float t1501 = memory[21020468 + t1500];
        float t1502 = t1498 * t1498;
        float t1503 = t1501 * t1501;
        float t1504 = t1502 + t1503;
        float t1505 = metal::sqrt(t1504);
        int t1506 = t479 + _pr1485;
        memory[42008372 + t1506] = t1505;
        float t1508 = t1494 - t1505;
        int t1509 = t479 + _pr1485;
        float t1510 = t1508 * t1508;
        memory[46219060 + t1509] = t1510;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1513 = 0; t1513 < 257; t1513++) {
        int t1514 = t479 + t1513;
        float t1515 = memory[46219060 + t1514];
        float t1516 = t[12*frameCount + id] + t1515;
        t[12*frameCount + id] = t1516;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 15
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_15(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1520), value: global(1520)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(481) - handled in variable access */
    float t1519 = (t[12*frameCount + id] * 6.1035156e-05);
    t[13*frameCount + id] = t1519;
  }
  #pragma clang diagnostic pop
}



// KERNEL 16
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize Optional(1)
// ThreadCount nil
kernel void kernel_16(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1521), value: global(1521)) */
  if (id >= 0 && id < (uint)(1)) {
    for (uint i = 0; i < frameCount; i += 1) {
      t[14*frameCount + i] = memory[532827831];
      float t1522 = t[14*frameCount + i] + 1.0;
      float t1523 = metal::select(t1522, 0.0, 0.0 > 0.0);
      float t1524 = t1523;
      float t1525 = (t1524 * 0.00390625);
      float t1526 = metal::floor(t1525);
      float t1527 = t1526 * 256.0;
      float t1528 = t1523 - t1527;
      memory[532827831] = t1528;
      float t1530 = t1528 >= 256.0;
      if (t1530) {
        float t1532 = t1528 - 256.0;
        memory[532827831] = t1532;
      }
      if (0.0) {
        memory[532827831] = 0.0;
      }
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 17
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(1542), value: global(1542)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1521) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    /* loadGlobal(406) - handled in variable access */
    int t1538 = id;
    int t1539 = t1538 * 2048;
    int t1540 = t1538 * 513;
    float t1541 = t[14*frameCount + id] == 0.0;
    t[15*frameCount + id] = 0.0;
    if (t1541) {
      for (uint _pr1544 = 0; _pr1544 < 1024; _pr1544++) {
        float t1545 = (float)_pr1544;
        float t1546 = 6.283185 * t1545;
        float t1547 = (t1546 * 0.0009775171);
        float t1548 = metal::cos(t1547);
        float t1549 = 1.0 - t1548;
        float t1550 = 0.5 * t1549;
        float t1551 = (float)t1538;
        float t1552 = t1551 - 1023.0;
        float t1553 = t1552 + t1545;
        float t1554 = (t1553 < 0 || t1553 >= frameCount) ? 0.0 : t[8 * frameCount + (int)t1553];
        float t1555 = (t1553 < 0 || t1553 >= frameCount) ? 0.0 : t[10 * frameCount + (int)t1553];
        int t1556 = t1539 + _pr1544;
        float t1557 = t1554 * t1550;
        memory[50429748 + t1556] = t1557;
        int t1559 = t1539 + _pr1544;
        int t1560 = t1559 + 1024;
        memory[50429748 + t1560] = 0.0;
        int t1562 = t1539 + _pr1544;
        float t1563 = t1555 * t1550;
        memory[83984180 + t1562] = t1563;
        int t1565 = t1539 + _pr1544;
        int t1566 = t1565 + 1024;
        memory[83984180 + t1566] = 0.0;
        memory[33012 + (int)_pr1544] = t1550;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t1570 = 0; t1570 < 1024; t1570++) {
        float t1571 = (float)t1570;
        float t1572 = (t1571 - metal::floor(t1571 / 2.0) * 2.0);
        float t1573 = t1572;
        float t1574 = (t1571 * 0.5);
        float t1575 = metal::floor(t1574);
        float t1576 = t1573 * 2.0;
        float t1577 = (t1575 - metal::floor(t1575 / 2.0) * 2.0);
        float t1578 = t1576 + t1577;
        float t1579 = (t1575 * 0.5);
        float t1580 = metal::floor(t1579);
        float t1581 = t1578 * 2.0;
        float t1582 = (t1580 - metal::floor(t1580 / 2.0) * 2.0);
        float t1583 = t1581 + t1582;
        float t1584 = (t1580 * 0.5);
        float t1585 = metal::floor(t1584);
        float t1586 = t1583 * 2.0;
        float t1587 = (t1585 - metal::floor(t1585 / 2.0) * 2.0);
        float t1588 = t1586 + t1587;
        float t1589 = (t1585 * 0.5);
        float t1590 = metal::floor(t1589);
        float t1591 = t1588 * 2.0;
        float t1592 = (t1590 - metal::floor(t1590 / 2.0) * 2.0);
        float t1593 = t1591 + t1592;
        float t1594 = (t1590 * 0.5);
        float t1595 = metal::floor(t1594);
        float t1596 = t1593 * 2.0;
        float t1597 = (t1595 - metal::floor(t1595 / 2.0) * 2.0);
        float t1598 = t1596 + t1597;
        float t1599 = (t1595 * 0.5);
        float t1600 = metal::floor(t1599);
        float t1601 = t1598 * 2.0;
        float t1602 = (t1600 - metal::floor(t1600 / 2.0) * 2.0);
        float t1603 = t1601 + t1602;
        float t1604 = (t1600 * 0.5);
        float t1605 = metal::floor(t1604);
        float t1606 = t1603 * 2.0;
        float t1607 = (t1605 - metal::floor(t1605 / 2.0) * 2.0);
        float t1608 = t1606 + t1607;
        float t1609 = (t1605 * 0.5);
        float t1610 = metal::floor(t1609);
        float t1611 = t1608 * 2.0;
        float t1612 = (t1610 - metal::floor(t1610 / 2.0) * 2.0);
        float t1613 = t1611 + t1612;
        float t1614 = (t1610 * 0.5);
        float t1615 = metal::floor(t1614);
        float t1616 = t1613 * 2.0;
        float t1617 = (t1615 - metal::floor(t1615 / 2.0) * 2.0);
        float t1618 = t1616 + t1617;
        float t1619 = (t1615 * 0.5);
        float t1620 = metal::floor(t1619);
        float t1621 = (float)t1570;
        float t1622 = t1621 < t1618;
        int t1623 = (int)t1618;
        int t1624 = t1539 + t1570;
        float t1625 = memory[50429748 + t1624];
        int t1626 = t1539 + t1570;
        int t1627 = t1626 + 1024;
        float t1628 = memory[50429748 + t1627];
        int t1629 = t1539 + t1623;
        float t1630 = memory[50429748 + t1629];
        int t1631 = t1539 + t1623;
        int t1632 = t1631 + 1024;
        float t1633 = memory[50429748 + t1632];
        float t1634 = metal::select(t1625, t1630, t1622 > 0.0);
        float t1635 = metal::select(t1628, t1633, t1622 > 0.0);
        float t1636 = metal::select(t1630, t1625, t1622 > 0.0);
        float t1637 = metal::select(t1633, t1628, t1622 > 0.0);
        int t1638 = t1539 + t1570;
        memory[50429748 + t1638] = t1634;
        int t1640 = t1539 + t1570;
        int t1641 = t1640 + 1024;
        memory[50429748 + t1641] = t1635;
        int t1643 = t1539 + t1623;
        memory[50429748 + t1643] = t1636;
        int t1645 = t1539 + t1623;
        int t1646 = t1645 + 1024;
        memory[50429748 + t1646] = t1637;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1649 = 0; _pr1649 < 512; _pr1649++) {
        float t1650 = (float)_pr1649;
        float t1651 = t1650;
        float t1652 = metal::floor(t1651);
        float t1653 = t1652;
        float t1654 = t1650 - t1653;
        float t1655 = t1652 * 2.0;
        float t1656 = t1655 + t1654;
        float t1657 = t1656 + 1.0;
        float t1658 = -6.283185 * t1654;
        float t1659 = (t1658 * 0.5);
        float t1660 = metal::cos(t1659);
        float t1661 = metal::sin(t1659);
        int t1662 = (int)t1656;
        int t1663 = (int)t1657;
        int t1664 = t1539 + t1662;
        float t1665 = memory[50429748 + t1664];
        int t1666 = t1539 + t1662;
        int t1667 = t1666 + 1024;
        float t1668 = memory[50429748 + t1667];
        int t1669 = t1539 + t1663;
        float t1670 = memory[50429748 + t1669];
        int t1671 = t1539 + t1663;
        int t1672 = t1671 + 1024;
        float t1673 = memory[50429748 + t1672];
        float t1674 = t1660 * t1670;
        float t1675 = t1661 * t1673;
        float t1676 = t1674 - t1675;
        float t1677 = t1660 * t1673;
        float t1678 = t1661 * t1670;
        float t1679 = t1677 + t1678;
        int t1680 = t1539 + t1662;
        float t1681 = t1665 + t1676;
        memory[50429748 + t1680] = t1681;
        int t1683 = t1539 + t1662;
        int t1684 = t1683 + 1024;
        float t1685 = t1668 + t1679;
        memory[50429748 + t1684] = t1685;
        int t1687 = t1539 + t1663;
        float t1688 = t1665 - t1676;
        memory[50429748 + t1687] = t1688;
        int t1690 = t1539 + t1663;
        int t1691 = t1690 + 1024;
        float t1692 = t1668 - t1679;
        memory[50429748 + t1691] = t1692;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1695 = 0; _pr1695 < 512; _pr1695++) {
        float t1696 = (float)_pr1695;
        float t1697 = (t1696 * 0.5);
        float t1698 = metal::floor(t1697);
        float t1699 = t1698 * 2.0;
        float t1700 = t1696 - t1699;
        float t1701 = t1698 * 4.0;
        float t1702 = t1701 + t1700;
        float t1703 = t1702 + 2.0;
        float t1704 = -6.283185 * t1700;
        float t1705 = (t1704 * 0.25);
        float t1706 = metal::cos(t1705);
        float t1707 = metal::sin(t1705);
        int t1708 = (int)t1702;
        int t1709 = (int)t1703;
        int t1710 = t1539 + t1708;
        float t1711 = memory[50429748 + t1710];
        int t1712 = t1539 + t1708;
        int t1713 = t1712 + 1024;
        float t1714 = memory[50429748 + t1713];
        int t1715 = t1539 + t1709;
        float t1716 = memory[50429748 + t1715];
        int t1717 = t1539 + t1709;
        int t1718 = t1717 + 1024;
        float t1719 = memory[50429748 + t1718];
        float t1720 = t1706 * t1716;
        float t1721 = t1707 * t1719;
        float t1722 = t1720 - t1721;
        float t1723 = t1706 * t1719;
        float t1724 = t1707 * t1716;
        float t1725 = t1723 + t1724;
        int t1726 = t1539 + t1708;
        float t1727 = t1711 + t1722;
        memory[50429748 + t1726] = t1727;
        int t1729 = t1539 + t1708;
        int t1730 = t1729 + 1024;
        float t1731 = t1714 + t1725;
        memory[50429748 + t1730] = t1731;
        int t1733 = t1539 + t1709;
        float t1734 = t1711 - t1722;
        memory[50429748 + t1733] = t1734;
        int t1736 = t1539 + t1709;
        int t1737 = t1736 + 1024;
        float t1738 = t1714 - t1725;
        memory[50429748 + t1737] = t1738;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1741 = 0; _pr1741 < 512; _pr1741++) {
        float t1742 = (float)_pr1741;
        float t1743 = (t1742 * 0.25);
        float t1744 = metal::floor(t1743);
        float t1745 = t1744 * 4.0;
        float t1746 = t1742 - t1745;
        float t1747 = t1744 * 8.0;
        float t1748 = t1747 + t1746;
        float t1749 = t1748 + 4.0;
        float t1750 = -6.283185 * t1746;
        float t1751 = (t1750 * 0.125);
        float t1752 = metal::cos(t1751);
        float t1753 = metal::sin(t1751);
        int t1754 = (int)t1748;
        int t1755 = (int)t1749;
        int t1756 = t1539 + t1754;
        float t1757 = memory[50429748 + t1756];
        int t1758 = t1539 + t1754;
        int t1759 = t1758 + 1024;
        float t1760 = memory[50429748 + t1759];
        int t1761 = t1539 + t1755;
        float t1762 = memory[50429748 + t1761];
        int t1763 = t1539 + t1755;
        int t1764 = t1763 + 1024;
        float t1765 = memory[50429748 + t1764];
        float t1766 = t1752 * t1762;
        float t1767 = t1753 * t1765;
        float t1768 = t1766 - t1767;
        float t1769 = t1752 * t1765;
        float t1770 = t1753 * t1762;
        float t1771 = t1769 + t1770;
        int t1772 = t1539 + t1754;
        float t1773 = t1757 + t1768;
        memory[50429748 + t1772] = t1773;
        int t1775 = t1539 + t1754;
        int t1776 = t1775 + 1024;
        float t1777 = t1760 + t1771;
        memory[50429748 + t1776] = t1777;
        int t1779 = t1539 + t1755;
        float t1780 = t1757 - t1768;
        memory[50429748 + t1779] = t1780;
        int t1782 = t1539 + t1755;
        int t1783 = t1782 + 1024;
        float t1784 = t1760 - t1771;
        memory[50429748 + t1783] = t1784;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1787 = 0; _pr1787 < 512; _pr1787++) {
        float t1788 = (float)_pr1787;
        float t1789 = (t1788 * 0.125);
        float t1790 = metal::floor(t1789);
        float t1791 = t1790 * 8.0;
        float t1792 = t1788 - t1791;
        float t1793 = t1790 * 16.0;
        float t1794 = t1793 + t1792;
        float t1795 = t1794 + 8.0;
        float t1796 = -6.283185 * t1792;
        float t1797 = (t1796 * 0.0625);
        float t1798 = metal::cos(t1797);
        float t1799 = metal::sin(t1797);
        int t1800 = (int)t1794;
        int t1801 = (int)t1795;
        int t1802 = t1539 + t1800;
        float t1803 = memory[50429748 + t1802];
        int t1804 = t1539 + t1800;
        int t1805 = t1804 + 1024;
        float t1806 = memory[50429748 + t1805];
        int t1807 = t1539 + t1801;
        float t1808 = memory[50429748 + t1807];
        int t1809 = t1539 + t1801;
        int t1810 = t1809 + 1024;
        float t1811 = memory[50429748 + t1810];
        float t1812 = t1798 * t1808;
        float t1813 = t1799 * t1811;
        float t1814 = t1812 - t1813;
        float t1815 = t1798 * t1811;
        float t1816 = t1799 * t1808;
        float t1817 = t1815 + t1816;
        int t1818 = t1539 + t1800;
        float t1819 = t1803 + t1814;
        memory[50429748 + t1818] = t1819;
        int t1821 = t1539 + t1800;
        int t1822 = t1821 + 1024;
        float t1823 = t1806 + t1817;
        memory[50429748 + t1822] = t1823;
        int t1825 = t1539 + t1801;
        float t1826 = t1803 - t1814;
        memory[50429748 + t1825] = t1826;
        int t1828 = t1539 + t1801;
        int t1829 = t1828 + 1024;
        float t1830 = t1806 - t1817;
        memory[50429748 + t1829] = t1830;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1833 = 0; _pr1833 < 512; _pr1833++) {
        float t1834 = (float)_pr1833;
        float t1835 = (t1834 * 0.0625);
        float t1836 = metal::floor(t1835);
        float t1837 = t1836 * 16.0;
        float t1838 = t1834 - t1837;
        float t1839 = t1836 * 32.0;
        float t1840 = t1839 + t1838;
        float t1841 = t1840 + 16.0;
        float t1842 = -6.283185 * t1838;
        float t1843 = (t1842 * 0.03125);
        float t1844 = metal::cos(t1843);
        float t1845 = metal::sin(t1843);
        int t1846 = (int)t1840;
        int t1847 = (int)t1841;
        int t1848 = t1539 + t1846;
        float t1849 = memory[50429748 + t1848];
        int t1850 = t1539 + t1846;
        int t1851 = t1850 + 1024;
        float t1852 = memory[50429748 + t1851];
        int t1853 = t1539 + t1847;
        float t1854 = memory[50429748 + t1853];
        int t1855 = t1539 + t1847;
        int t1856 = t1855 + 1024;
        float t1857 = memory[50429748 + t1856];
        float t1858 = t1844 * t1854;
        float t1859 = t1845 * t1857;
        float t1860 = t1858 - t1859;
        float t1861 = t1844 * t1857;
        float t1862 = t1845 * t1854;
        float t1863 = t1861 + t1862;
        int t1864 = t1539 + t1846;
        float t1865 = t1849 + t1860;
        memory[50429748 + t1864] = t1865;
        int t1867 = t1539 + t1846;
        int t1868 = t1867 + 1024;
        float t1869 = t1852 + t1863;
        memory[50429748 + t1868] = t1869;
        int t1871 = t1539 + t1847;
        float t1872 = t1849 - t1860;
        memory[50429748 + t1871] = t1872;
        int t1874 = t1539 + t1847;
        int t1875 = t1874 + 1024;
        float t1876 = t1852 - t1863;
        memory[50429748 + t1875] = t1876;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1879 = 0; _pr1879 < 512; _pr1879++) {
        float t1880 = (float)_pr1879;
        float t1881 = (t1880 * 0.03125);
        float t1882 = metal::floor(t1881);
        float t1883 = t1882 * 32.0;
        float t1884 = t1880 - t1883;
        float t1885 = t1882 * 64.0;
        float t1886 = t1885 + t1884;
        float t1887 = t1886 + 32.0;
        float t1888 = -6.283185 * t1884;
        float t1889 = (t1888 * 0.015625);
        float t1890 = metal::cos(t1889);
        float t1891 = metal::sin(t1889);
        int t1892 = (int)t1886;
        int t1893 = (int)t1887;
        int t1894 = t1539 + t1892;
        float t1895 = memory[50429748 + t1894];
        int t1896 = t1539 + t1892;
        int t1897 = t1896 + 1024;
        float t1898 = memory[50429748 + t1897];
        int t1899 = t1539 + t1893;
        float t1900 = memory[50429748 + t1899];
        int t1901 = t1539 + t1893;
        int t1902 = t1901 + 1024;
        float t1903 = memory[50429748 + t1902];
        float t1904 = t1890 * t1900;
        float t1905 = t1891 * t1903;
        float t1906 = t1904 - t1905;
        float t1907 = t1890 * t1903;
        float t1908 = t1891 * t1900;
        float t1909 = t1907 + t1908;
        int t1910 = t1539 + t1892;
        float t1911 = t1895 + t1906;
        memory[50429748 + t1910] = t1911;
        int t1913 = t1539 + t1892;
        int t1914 = t1913 + 1024;
        float t1915 = t1898 + t1909;
        memory[50429748 + t1914] = t1915;
        int t1917 = t1539 + t1893;
        float t1918 = t1895 - t1906;
        memory[50429748 + t1917] = t1918;
        int t1920 = t1539 + t1893;
        int t1921 = t1920 + 1024;
        float t1922 = t1898 - t1909;
        memory[50429748 + t1921] = t1922;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1925 = 0; _pr1925 < 512; _pr1925++) {
        float t1926 = (float)_pr1925;
        float t1927 = (t1926 * 0.015625);
        float t1928 = metal::floor(t1927);
        float t1929 = t1928 * 64.0;
        float t1930 = t1926 - t1929;
        float t1931 = t1928 * 128.0;
        float t1932 = t1931 + t1930;
        float t1933 = t1932 + 64.0;
        float t1934 = -6.283185 * t1930;
        float t1935 = (t1934 * 0.0078125);
        float t1936 = metal::cos(t1935);
        float t1937 = metal::sin(t1935);
        int t1938 = (int)t1932;
        int t1939 = (int)t1933;
        int t1940 = t1539 + t1938;
        float t1941 = memory[50429748 + t1940];
        int t1942 = t1539 + t1938;
        int t1943 = t1942 + 1024;
        float t1944 = memory[50429748 + t1943];
        int t1945 = t1539 + t1939;
        float t1946 = memory[50429748 + t1945];
        int t1947 = t1539 + t1939;
        int t1948 = t1947 + 1024;
        float t1949 = memory[50429748 + t1948];
        float t1950 = t1936 * t1946;
        float t1951 = t1937 * t1949;
        float t1952 = t1950 - t1951;
        float t1953 = t1936 * t1949;
        float t1954 = t1937 * t1946;
        float t1955 = t1953 + t1954;
        int t1956 = t1539 + t1938;
        float t1957 = t1941 + t1952;
        memory[50429748 + t1956] = t1957;
        int t1959 = t1539 + t1938;
        int t1960 = t1959 + 1024;
        float t1961 = t1944 + t1955;
        memory[50429748 + t1960] = t1961;
        int t1963 = t1539 + t1939;
        float t1964 = t1941 - t1952;
        memory[50429748 + t1963] = t1964;
        int t1966 = t1539 + t1939;
        int t1967 = t1966 + 1024;
        float t1968 = t1944 - t1955;
        memory[50429748 + t1967] = t1968;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr1971 = 0; _pr1971 < 512; _pr1971++) {
        float t1972 = (float)_pr1971;
        float t1973 = (t1972 * 0.0078125);
        float t1974 = metal::floor(t1973);
        float t1975 = t1974 * 128.0;
        float t1976 = t1972 - t1975;
        float t1977 = t1974 * 256.0;
        float t1978 = t1977 + t1976;
        float t1979 = t1978 + 128.0;
        float t1980 = -6.283185 * t1976;
        float t1981 = (t1980 * 0.00390625);
        float t1982 = metal::cos(t1981);
        float t1983 = metal::sin(t1981);
        int t1984 = (int)t1978;
        int t1985 = (int)t1979;
        int t1986 = t1539 + t1984;
        float t1987 = memory[50429748 + t1986];
        int t1988 = t1539 + t1984;
        int t1989 = t1988 + 1024;
        float t1990 = memory[50429748 + t1989];
        int t1991 = t1539 + t1985;
        float t1992 = memory[50429748 + t1991];
        int t1993 = t1539 + t1985;
        int t1994 = t1993 + 1024;
        float t1995 = memory[50429748 + t1994];
        float t1996 = t1982 * t1992;
        float t1997 = t1983 * t1995;
        float t1998 = t1996 - t1997;
        float t1999 = t1982 * t1995;
        float t2000 = t1983 * t1992;
        float t2001 = t1999 + t2000;
        int t2002 = t1539 + t1984;
        float t2003 = t1987 + t1998;
        memory[50429748 + t2002] = t2003;
        int t2005 = t1539 + t1984;
        int t2006 = t2005 + 1024;
        float t2007 = t1990 + t2001;
        memory[50429748 + t2006] = t2007;
        int t2009 = t1539 + t1985;
        float t2010 = t1987 - t1998;
        memory[50429748 + t2009] = t2010;
        int t2012 = t1539 + t1985;
        int t2013 = t2012 + 1024;
        float t2014 = t1990 - t2001;
        memory[50429748 + t2013] = t2014;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2017 = 0; _pr2017 < 512; _pr2017++) {
        float t2018 = (float)_pr2017;
        float t2019 = (t2018 * 0.00390625);
        float t2020 = metal::floor(t2019);
        float t2021 = t2020 * 256.0;
        float t2022 = t2018 - t2021;
        float t2023 = t2020 * 512.0;
        float t2024 = t2023 + t2022;
        float t2025 = t2024 + 256.0;
        float t2026 = -6.283185 * t2022;
        float t2027 = (t2026 * 0.001953125);
        float t2028 = metal::cos(t2027);
        float t2029 = metal::sin(t2027);
        int t2030 = (int)t2024;
        int t2031 = (int)t2025;
        int t2032 = t1539 + t2030;
        float t2033 = memory[50429748 + t2032];
        int t2034 = t1539 + t2030;
        int t2035 = t2034 + 1024;
        float t2036 = memory[50429748 + t2035];
        int t2037 = t1539 + t2031;
        float t2038 = memory[50429748 + t2037];
        int t2039 = t1539 + t2031;
        int t2040 = t2039 + 1024;
        float t2041 = memory[50429748 + t2040];
        float t2042 = t2028 * t2038;
        float t2043 = t2029 * t2041;
        float t2044 = t2042 - t2043;
        float t2045 = t2028 * t2041;
        float t2046 = t2029 * t2038;
        float t2047 = t2045 + t2046;
        int t2048 = t1539 + t2030;
        float t2049 = t2033 + t2044;
        memory[50429748 + t2048] = t2049;
        int t2051 = t1539 + t2030;
        int t2052 = t2051 + 1024;
        float t2053 = t2036 + t2047;
        memory[50429748 + t2052] = t2053;
        int t2055 = t1539 + t2031;
        float t2056 = t2033 - t2044;
        memory[50429748 + t2055] = t2056;
        int t2058 = t1539 + t2031;
        int t2059 = t2058 + 1024;
        float t2060 = t2036 - t2047;
        memory[50429748 + t2059] = t2060;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2063 = 0; _pr2063 < 512; _pr2063++) {
        float t2064 = (float)_pr2063;
        float t2065 = (t2064 * 0.001953125);
        float t2066 = metal::floor(t2065);
        float t2067 = t2066 * 512.0;
        float t2068 = t2064 - t2067;
        float t2069 = t2066 * 1024.0;
        float t2070 = t2069 + t2068;
        float t2071 = t2070 + 512.0;
        float t2072 = -6.283185 * t2068;
        float t2073 = (t2072 * 0.0009765625);
        float t2074 = metal::cos(t2073);
        float t2075 = metal::sin(t2073);
        int t2076 = (int)t2070;
        int t2077 = (int)t2071;
        int t2078 = t1539 + t2076;
        float t2079 = memory[50429748 + t2078];
        int t2080 = t1539 + t2076;
        int t2081 = t2080 + 1024;
        float t2082 = memory[50429748 + t2081];
        int t2083 = t1539 + t2077;
        float t2084 = memory[50429748 + t2083];
        int t2085 = t1539 + t2077;
        int t2086 = t2085 + 1024;
        float t2087 = memory[50429748 + t2086];
        float t2088 = t2074 * t2084;
        float t2089 = t2075 * t2087;
        float t2090 = t2088 - t2089;
        float t2091 = t2074 * t2087;
        float t2092 = t2075 * t2084;
        float t2093 = t2091 + t2092;
        int t2094 = t1539 + t2076;
        float t2095 = t2079 + t2090;
        memory[50429748 + t2094] = t2095;
        int t2097 = t1539 + t2076;
        int t2098 = t2097 + 1024;
        float t2099 = t2082 + t2093;
        memory[50429748 + t2098] = t2099;
        int t2101 = t1539 + t2077;
        float t2102 = t2079 - t2090;
        memory[50429748 + t2101] = t2102;
        int t2104 = t1539 + t2077;
        int t2105 = t2104 + 1024;
        float t2106 = t2082 - t2093;
        memory[50429748 + t2105] = t2106;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2109 = 0; t2109 < 1024; t2109++) {
        float t2110 = (float)t2109;
        float t2111 = (t2110 - metal::floor(t2110 / 2.0) * 2.0);
        float t2112 = t2111;
        float t2113 = (t2110 * 0.5);
        float t2114 = metal::floor(t2113);
        float t2115 = t2112 * 2.0;
        float t2116 = (t2114 - metal::floor(t2114 / 2.0) * 2.0);
        float t2117 = t2115 + t2116;
        float t2118 = (t2114 * 0.5);
        float t2119 = metal::floor(t2118);
        float t2120 = t2117 * 2.0;
        float t2121 = (t2119 - metal::floor(t2119 / 2.0) * 2.0);
        float t2122 = t2120 + t2121;
        float t2123 = (t2119 * 0.5);
        float t2124 = metal::floor(t2123);
        float t2125 = t2122 * 2.0;
        float t2126 = (t2124 - metal::floor(t2124 / 2.0) * 2.0);
        float t2127 = t2125 + t2126;
        float t2128 = (t2124 * 0.5);
        float t2129 = metal::floor(t2128);
        float t2130 = t2127 * 2.0;
        float t2131 = (t2129 - metal::floor(t2129 / 2.0) * 2.0);
        float t2132 = t2130 + t2131;
        float t2133 = (t2129 * 0.5);
        float t2134 = metal::floor(t2133);
        float t2135 = t2132 * 2.0;
        float t2136 = (t2134 - metal::floor(t2134 / 2.0) * 2.0);
        float t2137 = t2135 + t2136;
        float t2138 = (t2134 * 0.5);
        float t2139 = metal::floor(t2138);
        float t2140 = t2137 * 2.0;
        float t2141 = (t2139 - metal::floor(t2139 / 2.0) * 2.0);
        float t2142 = t2140 + t2141;
        float t2143 = (t2139 * 0.5);
        float t2144 = metal::floor(t2143);
        float t2145 = t2142 * 2.0;
        float t2146 = (t2144 - metal::floor(t2144 / 2.0) * 2.0);
        float t2147 = t2145 + t2146;
        float t2148 = (t2144 * 0.5);
        float t2149 = metal::floor(t2148);
        float t2150 = t2147 * 2.0;
        float t2151 = (t2149 - metal::floor(t2149 / 2.0) * 2.0);
        float t2152 = t2150 + t2151;
        float t2153 = (t2149 * 0.5);
        float t2154 = metal::floor(t2153);
        float t2155 = t2152 * 2.0;
        float t2156 = (t2154 - metal::floor(t2154 / 2.0) * 2.0);
        float t2157 = t2155 + t2156;
        float t2158 = (t2154 * 0.5);
        float t2159 = metal::floor(t2158);
        float t2160 = (float)t2109;
        float t2161 = t2160 < t2157;
        int t2162 = (int)t2157;
        int t2163 = t1539 + t2109;
        float t2164 = memory[83984180 + t2163];
        int t2165 = t1539 + t2109;
        int t2166 = t2165 + 1024;
        float t2167 = memory[83984180 + t2166];
        int t2168 = t1539 + t2162;
        float t2169 = memory[83984180 + t2168];
        int t2170 = t1539 + t2162;
        int t2171 = t2170 + 1024;
        float t2172 = memory[83984180 + t2171];
        float t2173 = metal::select(t2164, t2169, t2161 > 0.0);
        float t2174 = metal::select(t2167, t2172, t2161 > 0.0);
        float t2175 = metal::select(t2169, t2164, t2161 > 0.0);
        float t2176 = metal::select(t2172, t2167, t2161 > 0.0);
        int t2177 = t1539 + t2109;
        memory[83984180 + t2177] = t2173;
        int t2179 = t1539 + t2109;
        int t2180 = t2179 + 1024;
        memory[83984180 + t2180] = t2174;
        int t2182 = t1539 + t2162;
        memory[83984180 + t2182] = t2175;
        int t2184 = t1539 + t2162;
        int t2185 = t2184 + 1024;
        memory[83984180 + t2185] = t2176;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2188 = 0; _pr2188 < 512; _pr2188++) {
        float t2189 = (float)_pr2188;
        float t2190 = t2189;
        float t2191 = metal::floor(t2190);
        float t2192 = t2191;
        float t2193 = t2189 - t2192;
        float t2194 = t2191 * 2.0;
        float t2195 = t2194 + t2193;
        float t2196 = t2195 + 1.0;
        float t2197 = -6.283185 * t2193;
        float t2198 = (t2197 * 0.5);
        float t2199 = metal::cos(t2198);
        float t2200 = metal::sin(t2198);
        int t2201 = (int)t2195;
        int t2202 = (int)t2196;
        int t2203 = t1539 + t2201;
        float t2204 = memory[83984180 + t2203];
        int t2205 = t1539 + t2201;
        int t2206 = t2205 + 1024;
        float t2207 = memory[83984180 + t2206];
        int t2208 = t1539 + t2202;
        float t2209 = memory[83984180 + t2208];
        int t2210 = t1539 + t2202;
        int t2211 = t2210 + 1024;
        float t2212 = memory[83984180 + t2211];
        float t2213 = t2199 * t2209;
        float t2214 = t2200 * t2212;
        float t2215 = t2213 - t2214;
        float t2216 = t2199 * t2212;
        float t2217 = t2200 * t2209;
        float t2218 = t2216 + t2217;
        int t2219 = t1539 + t2201;
        float t2220 = t2204 + t2215;
        memory[83984180 + t2219] = t2220;
        int t2222 = t1539 + t2201;
        int t2223 = t2222 + 1024;
        float t2224 = t2207 + t2218;
        memory[83984180 + t2223] = t2224;
        int t2226 = t1539 + t2202;
        float t2227 = t2204 - t2215;
        memory[83984180 + t2226] = t2227;
        int t2229 = t1539 + t2202;
        int t2230 = t2229 + 1024;
        float t2231 = t2207 - t2218;
        memory[83984180 + t2230] = t2231;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2234 = 0; _pr2234 < 512; _pr2234++) {
        float t2235 = (float)_pr2234;
        float t2236 = (t2235 * 0.5);
        float t2237 = metal::floor(t2236);
        float t2238 = t2237 * 2.0;
        float t2239 = t2235 - t2238;
        float t2240 = t2237 * 4.0;
        float t2241 = t2240 + t2239;
        float t2242 = t2241 + 2.0;
        float t2243 = -6.283185 * t2239;
        float t2244 = (t2243 * 0.25);
        float t2245 = metal::cos(t2244);
        float t2246 = metal::sin(t2244);
        int t2247 = (int)t2241;
        int t2248 = (int)t2242;
        int t2249 = t1539 + t2247;
        float t2250 = memory[83984180 + t2249];
        int t2251 = t1539 + t2247;
        int t2252 = t2251 + 1024;
        float t2253 = memory[83984180 + t2252];
        int t2254 = t1539 + t2248;
        float t2255 = memory[83984180 + t2254];
        int t2256 = t1539 + t2248;
        int t2257 = t2256 + 1024;
        float t2258 = memory[83984180 + t2257];
        float t2259 = t2245 * t2255;
        float t2260 = t2246 * t2258;
        float t2261 = t2259 - t2260;
        float t2262 = t2245 * t2258;
        float t2263 = t2246 * t2255;
        float t2264 = t2262 + t2263;
        int t2265 = t1539 + t2247;
        float t2266 = t2250 + t2261;
        memory[83984180 + t2265] = t2266;
        int t2268 = t1539 + t2247;
        int t2269 = t2268 + 1024;
        float t2270 = t2253 + t2264;
        memory[83984180 + t2269] = t2270;
        int t2272 = t1539 + t2248;
        float t2273 = t2250 - t2261;
        memory[83984180 + t2272] = t2273;
        int t2275 = t1539 + t2248;
        int t2276 = t2275 + 1024;
        float t2277 = t2253 - t2264;
        memory[83984180 + t2276] = t2277;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2280 = 0; _pr2280 < 512; _pr2280++) {
        float t2281 = (float)_pr2280;
        float t2282 = (t2281 * 0.25);
        float t2283 = metal::floor(t2282);
        float t2284 = t2283 * 4.0;
        float t2285 = t2281 - t2284;
        float t2286 = t2283 * 8.0;
        float t2287 = t2286 + t2285;
        float t2288 = t2287 + 4.0;
        float t2289 = -6.283185 * t2285;
        float t2290 = (t2289 * 0.125);
        float t2291 = metal::cos(t2290);
        float t2292 = metal::sin(t2290);
        int t2293 = (int)t2287;
        int t2294 = (int)t2288;
        int t2295 = t1539 + t2293;
        float t2296 = memory[83984180 + t2295];
        int t2297 = t1539 + t2293;
        int t2298 = t2297 + 1024;
        float t2299 = memory[83984180 + t2298];
        int t2300 = t1539 + t2294;
        float t2301 = memory[83984180 + t2300];
        int t2302 = t1539 + t2294;
        int t2303 = t2302 + 1024;
        float t2304 = memory[83984180 + t2303];
        float t2305 = t2291 * t2301;
        float t2306 = t2292 * t2304;
        float t2307 = t2305 - t2306;
        float t2308 = t2291 * t2304;
        float t2309 = t2292 * t2301;
        float t2310 = t2308 + t2309;
        int t2311 = t1539 + t2293;
        float t2312 = t2296 + t2307;
        memory[83984180 + t2311] = t2312;
        int t2314 = t1539 + t2293;
        int t2315 = t2314 + 1024;
        float t2316 = t2299 + t2310;
        memory[83984180 + t2315] = t2316;
        int t2318 = t1539 + t2294;
        float t2319 = t2296 - t2307;
        memory[83984180 + t2318] = t2319;
        int t2321 = t1539 + t2294;
        int t2322 = t2321 + 1024;
        float t2323 = t2299 - t2310;
        memory[83984180 + t2322] = t2323;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2326 = 0; _pr2326 < 512; _pr2326++) {
        float t2327 = (float)_pr2326;
        float t2328 = (t2327 * 0.125);
        float t2329 = metal::floor(t2328);
        float t2330 = t2329 * 8.0;
        float t2331 = t2327 - t2330;
        float t2332 = t2329 * 16.0;
        float t2333 = t2332 + t2331;
        float t2334 = t2333 + 8.0;
        float t2335 = -6.283185 * t2331;
        float t2336 = (t2335 * 0.0625);
        float t2337 = metal::cos(t2336);
        float t2338 = metal::sin(t2336);
        int t2339 = (int)t2333;
        int t2340 = (int)t2334;
        int t2341 = t1539 + t2339;
        float t2342 = memory[83984180 + t2341];
        int t2343 = t1539 + t2339;
        int t2344 = t2343 + 1024;
        float t2345 = memory[83984180 + t2344];
        int t2346 = t1539 + t2340;
        float t2347 = memory[83984180 + t2346];
        int t2348 = t1539 + t2340;
        int t2349 = t2348 + 1024;
        float t2350 = memory[83984180 + t2349];
        float t2351 = t2337 * t2347;
        float t2352 = t2338 * t2350;
        float t2353 = t2351 - t2352;
        float t2354 = t2337 * t2350;
        float t2355 = t2338 * t2347;
        float t2356 = t2354 + t2355;
        int t2357 = t1539 + t2339;
        float t2358 = t2342 + t2353;
        memory[83984180 + t2357] = t2358;
        int t2360 = t1539 + t2339;
        int t2361 = t2360 + 1024;
        float t2362 = t2345 + t2356;
        memory[83984180 + t2361] = t2362;
        int t2364 = t1539 + t2340;
        float t2365 = t2342 - t2353;
        memory[83984180 + t2364] = t2365;
        int t2367 = t1539 + t2340;
        int t2368 = t2367 + 1024;
        float t2369 = t2345 - t2356;
        memory[83984180 + t2368] = t2369;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2372 = 0; _pr2372 < 512; _pr2372++) {
        float t2373 = (float)_pr2372;
        float t2374 = (t2373 * 0.0625);
        float t2375 = metal::floor(t2374);
        float t2376 = t2375 * 16.0;
        float t2377 = t2373 - t2376;
        float t2378 = t2375 * 32.0;
        float t2379 = t2378 + t2377;
        float t2380 = t2379 + 16.0;
        float t2381 = -6.283185 * t2377;
        float t2382 = (t2381 * 0.03125);
        float t2383 = metal::cos(t2382);
        float t2384 = metal::sin(t2382);
        int t2385 = (int)t2379;
        int t2386 = (int)t2380;
        int t2387 = t1539 + t2385;
        float t2388 = memory[83984180 + t2387];
        int t2389 = t1539 + t2385;
        int t2390 = t2389 + 1024;
        float t2391 = memory[83984180 + t2390];
        int t2392 = t1539 + t2386;
        float t2393 = memory[83984180 + t2392];
        int t2394 = t1539 + t2386;
        int t2395 = t2394 + 1024;
        float t2396 = memory[83984180 + t2395];
        float t2397 = t2383 * t2393;
        float t2398 = t2384 * t2396;
        float t2399 = t2397 - t2398;
        float t2400 = t2383 * t2396;
        float t2401 = t2384 * t2393;
        float t2402 = t2400 + t2401;
        int t2403 = t1539 + t2385;
        float t2404 = t2388 + t2399;
        memory[83984180 + t2403] = t2404;
        int t2406 = t1539 + t2385;
        int t2407 = t2406 + 1024;
        float t2408 = t2391 + t2402;
        memory[83984180 + t2407] = t2408;
        int t2410 = t1539 + t2386;
        float t2411 = t2388 - t2399;
        memory[83984180 + t2410] = t2411;
        int t2413 = t1539 + t2386;
        int t2414 = t2413 + 1024;
        float t2415 = t2391 - t2402;
        memory[83984180 + t2414] = t2415;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2418 = 0; _pr2418 < 512; _pr2418++) {
        float t2419 = (float)_pr2418;
        float t2420 = (t2419 * 0.03125);
        float t2421 = metal::floor(t2420);
        float t2422 = t2421 * 32.0;
        float t2423 = t2419 - t2422;
        float t2424 = t2421 * 64.0;
        float t2425 = t2424 + t2423;
        float t2426 = t2425 + 32.0;
        float t2427 = -6.283185 * t2423;
        float t2428 = (t2427 * 0.015625);
        float t2429 = metal::cos(t2428);
        float t2430 = metal::sin(t2428);
        int t2431 = (int)t2425;
        int t2432 = (int)t2426;
        int t2433 = t1539 + t2431;
        float t2434 = memory[83984180 + t2433];
        int t2435 = t1539 + t2431;
        int t2436 = t2435 + 1024;
        float t2437 = memory[83984180 + t2436];
        int t2438 = t1539 + t2432;
        float t2439 = memory[83984180 + t2438];
        int t2440 = t1539 + t2432;
        int t2441 = t2440 + 1024;
        float t2442 = memory[83984180 + t2441];
        float t2443 = t2429 * t2439;
        float t2444 = t2430 * t2442;
        float t2445 = t2443 - t2444;
        float t2446 = t2429 * t2442;
        float t2447 = t2430 * t2439;
        float t2448 = t2446 + t2447;
        int t2449 = t1539 + t2431;
        float t2450 = t2434 + t2445;
        memory[83984180 + t2449] = t2450;
        int t2452 = t1539 + t2431;
        int t2453 = t2452 + 1024;
        float t2454 = t2437 + t2448;
        memory[83984180 + t2453] = t2454;
        int t2456 = t1539 + t2432;
        float t2457 = t2434 - t2445;
        memory[83984180 + t2456] = t2457;
        int t2459 = t1539 + t2432;
        int t2460 = t2459 + 1024;
        float t2461 = t2437 - t2448;
        memory[83984180 + t2460] = t2461;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2464 = 0; _pr2464 < 512; _pr2464++) {
        float t2465 = (float)_pr2464;
        float t2466 = (t2465 * 0.015625);
        float t2467 = metal::floor(t2466);
        float t2468 = t2467 * 64.0;
        float t2469 = t2465 - t2468;
        float t2470 = t2467 * 128.0;
        float t2471 = t2470 + t2469;
        float t2472 = t2471 + 64.0;
        float t2473 = -6.283185 * t2469;
        float t2474 = (t2473 * 0.0078125);
        float t2475 = metal::cos(t2474);
        float t2476 = metal::sin(t2474);
        int t2477 = (int)t2471;
        int t2478 = (int)t2472;
        int t2479 = t1539 + t2477;
        float t2480 = memory[83984180 + t2479];
        int t2481 = t1539 + t2477;
        int t2482 = t2481 + 1024;
        float t2483 = memory[83984180 + t2482];
        int t2484 = t1539 + t2478;
        float t2485 = memory[83984180 + t2484];
        int t2486 = t1539 + t2478;
        int t2487 = t2486 + 1024;
        float t2488 = memory[83984180 + t2487];
        float t2489 = t2475 * t2485;
        float t2490 = t2476 * t2488;
        float t2491 = t2489 - t2490;
        float t2492 = t2475 * t2488;
        float t2493 = t2476 * t2485;
        float t2494 = t2492 + t2493;
        int t2495 = t1539 + t2477;
        float t2496 = t2480 + t2491;
        memory[83984180 + t2495] = t2496;
        int t2498 = t1539 + t2477;
        int t2499 = t2498 + 1024;
        float t2500 = t2483 + t2494;
        memory[83984180 + t2499] = t2500;
        int t2502 = t1539 + t2478;
        float t2503 = t2480 - t2491;
        memory[83984180 + t2502] = t2503;
        int t2505 = t1539 + t2478;
        int t2506 = t2505 + 1024;
        float t2507 = t2483 - t2494;
        memory[83984180 + t2506] = t2507;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2510 = 0; _pr2510 < 512; _pr2510++) {
        float t2511 = (float)_pr2510;
        float t2512 = (t2511 * 0.0078125);
        float t2513 = metal::floor(t2512);
        float t2514 = t2513 * 128.0;
        float t2515 = t2511 - t2514;
        float t2516 = t2513 * 256.0;
        float t2517 = t2516 + t2515;
        float t2518 = t2517 + 128.0;
        float t2519 = -6.283185 * t2515;
        float t2520 = (t2519 * 0.00390625);
        float t2521 = metal::cos(t2520);
        float t2522 = metal::sin(t2520);
        int t2523 = (int)t2517;
        int t2524 = (int)t2518;
        int t2525 = t1539 + t2523;
        float t2526 = memory[83984180 + t2525];
        int t2527 = t1539 + t2523;
        int t2528 = t2527 + 1024;
        float t2529 = memory[83984180 + t2528];
        int t2530 = t1539 + t2524;
        float t2531 = memory[83984180 + t2530];
        int t2532 = t1539 + t2524;
        int t2533 = t2532 + 1024;
        float t2534 = memory[83984180 + t2533];
        float t2535 = t2521 * t2531;
        float t2536 = t2522 * t2534;
        float t2537 = t2535 - t2536;
        float t2538 = t2521 * t2534;
        float t2539 = t2522 * t2531;
        float t2540 = t2538 + t2539;
        int t2541 = t1539 + t2523;
        float t2542 = t2526 + t2537;
        memory[83984180 + t2541] = t2542;
        int t2544 = t1539 + t2523;
        int t2545 = t2544 + 1024;
        float t2546 = t2529 + t2540;
        memory[83984180 + t2545] = t2546;
        int t2548 = t1539 + t2524;
        float t2549 = t2526 - t2537;
        memory[83984180 + t2548] = t2549;
        int t2551 = t1539 + t2524;
        int t2552 = t2551 + 1024;
        float t2553 = t2529 - t2540;
        memory[83984180 + t2552] = t2553;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2556 = 0; _pr2556 < 512; _pr2556++) {
        float t2557 = (float)_pr2556;
        float t2558 = (t2557 * 0.00390625);
        float t2559 = metal::floor(t2558);
        float t2560 = t2559 * 256.0;
        float t2561 = t2557 - t2560;
        float t2562 = t2559 * 512.0;
        float t2563 = t2562 + t2561;
        float t2564 = t2563 + 256.0;
        float t2565 = -6.283185 * t2561;
        float t2566 = (t2565 * 0.001953125);
        float t2567 = metal::cos(t2566);
        float t2568 = metal::sin(t2566);
        int t2569 = (int)t2563;
        int t2570 = (int)t2564;
        int t2571 = t1539 + t2569;
        float t2572 = memory[83984180 + t2571];
        int t2573 = t1539 + t2569;
        int t2574 = t2573 + 1024;
        float t2575 = memory[83984180 + t2574];
        int t2576 = t1539 + t2570;
        float t2577 = memory[83984180 + t2576];
        int t2578 = t1539 + t2570;
        int t2579 = t2578 + 1024;
        float t2580 = memory[83984180 + t2579];
        float t2581 = t2567 * t2577;
        float t2582 = t2568 * t2580;
        float t2583 = t2581 - t2582;
        float t2584 = t2567 * t2580;
        float t2585 = t2568 * t2577;
        float t2586 = t2584 + t2585;
        int t2587 = t1539 + t2569;
        float t2588 = t2572 + t2583;
        memory[83984180 + t2587] = t2588;
        int t2590 = t1539 + t2569;
        int t2591 = t2590 + 1024;
        float t2592 = t2575 + t2586;
        memory[83984180 + t2591] = t2592;
        int t2594 = t1539 + t2570;
        float t2595 = t2572 - t2583;
        memory[83984180 + t2594] = t2595;
        int t2597 = t1539 + t2570;
        int t2598 = t2597 + 1024;
        float t2599 = t2575 - t2586;
        memory[83984180 + t2598] = t2599;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2602 = 0; _pr2602 < 512; _pr2602++) {
        float t2603 = (float)_pr2602;
        float t2604 = (t2603 * 0.001953125);
        float t2605 = metal::floor(t2604);
        float t2606 = t2605 * 512.0;
        float t2607 = t2603 - t2606;
        float t2608 = t2605 * 1024.0;
        float t2609 = t2608 + t2607;
        float t2610 = t2609 + 512.0;
        float t2611 = -6.283185 * t2607;
        float t2612 = (t2611 * 0.0009765625);
        float t2613 = metal::cos(t2612);
        float t2614 = metal::sin(t2612);
        int t2615 = (int)t2609;
        int t2616 = (int)t2610;
        int t2617 = t1539 + t2615;
        float t2618 = memory[83984180 + t2617];
        int t2619 = t1539 + t2615;
        int t2620 = t2619 + 1024;
        float t2621 = memory[83984180 + t2620];
        int t2622 = t1539 + t2616;
        float t2623 = memory[83984180 + t2622];
        int t2624 = t1539 + t2616;
        int t2625 = t2624 + 1024;
        float t2626 = memory[83984180 + t2625];
        float t2627 = t2613 * t2623;
        float t2628 = t2614 * t2626;
        float t2629 = t2627 - t2628;
        float t2630 = t2613 * t2626;
        float t2631 = t2614 * t2623;
        float t2632 = t2630 + t2631;
        int t2633 = t1539 + t2615;
        float t2634 = t2618 + t2629;
        memory[83984180 + t2633] = t2634;
        int t2636 = t1539 + t2615;
        int t2637 = t2636 + 1024;
        float t2638 = t2621 + t2632;
        memory[83984180 + t2637] = t2638;
        int t2640 = t1539 + t2616;
        float t2641 = t2618 - t2629;
        memory[83984180 + t2640] = t2641;
        int t2643 = t1539 + t2616;
        int t2644 = t2643 + 1024;
        float t2645 = t2621 - t2632;
        memory[83984180 + t2644] = t2645;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2648 = 0; _pr2648 < 513; _pr2648++) {
        int t2649 = t1539 + _pr2648;
        float t2650 = memory[50429748 + t2649];
        int t2651 = t1539 + _pr2648;
        int t2652 = t2651 + 1024;
        float t2653 = memory[50429748 + t2652];
        float t2654 = t2650 * t2650;
        float t2655 = t2653 * t2653;
        float t2656 = t2654 + t2655;
        float t2657 = metal::sqrt(t2656);
        int t2658 = t1540 + _pr2648;
        memory[117538612 + t2658] = t2657;
        int t2660 = t1539 + _pr2648;
        float t2661 = memory[83984180 + t2660];
        int t2662 = t1539 + _pr2648;
        int t2663 = t2662 + 1024;
        float t2664 = memory[83984180 + t2663];
        float t2665 = t2661 * t2661;
        float t2666 = t2664 * t2664;
        float t2667 = t2665 + t2666;
        float t2668 = metal::sqrt(t2667);
        int t2669 = t1540 + _pr2648;
        memory[125943604 + t2669] = t2668;
        float t2671 = t2657 - t2668;
        int t2672 = t1540 + _pr2648;
        float t2673 = t2671 * t2671;
        memory[134348596 + t2672] = t2673;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t2676 = 0; t2676 < 513; t2676++) {
        int t2677 = t1540 + t2676;
        float t2678 = memory[134348596 + t2677];
        float t2679 = t[15*frameCount + id] + t2678;
        t[15*frameCount + id] = t2679;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
  }
  #pragma clang diagnostic pop
}



// KERNEL 18
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_18(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(2690), value: global(2690)) */
  float t5550 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5550)) {
    /* loadGlobal(1542) - handled in variable access */
    /* loadGlobal(1520) - handled in variable access */
    int t2682 = id;
    int t2683 = t2682 / 61;
    uint _frameIndex = (uint)(t2683);
    int t2684 = t2683 * 61;
    int t2685 = t2682 - t2684;
    float t2686 = (t[15*frameCount + _frameIndex] * 6.1035156e-05);
    float t2687 = t[13*frameCount + _frameIndex] + t2686;
    float t2688 = t2687 * 0.5;
    float t2689 = t2688;
    t[16*frameCount + _frameIndex] = t2689;
    float t2691 = t2688;
    float t2692 = t2687;
    float t2693 = (t[15*frameCount + _frameIndex] * 3.7252903e-09);
    float t2694 = -0.5 * t2693;
  }
  #pragma clang diagnostic pop
}



// KERNEL 19
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_19(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1521) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    /* loadGlobal(406) - handled in variable access */
    int t2695 = id;
    int t2696 = t2695 * 2048;
    int t2697 = t2695 * 513;
    int t2698 = t2695 * 2048;
    float t2699 = t[14*frameCount + id] == 0.0;
    if (t2699) {
      for (uint _pr2701 = 0; _pr2701 < 513; _pr2701++) {
        int t2702 = t2697 + _pr2701;
        float t2703 = memory[117538612 + t2702];
        int t2704 = t2697 + _pr2701;
        float t2705 = memory[125943604 + t2704];
        int t2706 = t2696 + _pr2701;
        float t2707 = memory[50429748 + t2706];
        int t2708 = t2696 + _pr2701;
        int t2709 = t2708 + 1024;
        float t2710 = memory[50429748 + t2709];
        int t2711 = t2696 + _pr2701;
        float t2712 = memory[83984180 + t2711];
        int t2713 = t2696 + _pr2701;
        int t2714 = t2713 + 1024;
        float t2715 = memory[83984180 + t2714];
        float t2716 = t2703 - t2705;
        float t2717 = 2.0 * t2716;
        float t2718 = t2717 * 3.0517578e-05;
        float t2719 = t2703 - t2705;
        float t2720 = -2.0 * t2719;
        float t2721 = t2720 * 3.0517578e-05;
        float t2722 = metal::max(t2703, 1e-08);
        float t2723 = metal::max(t2705, 1e-08);
        float t2724 = t2718 * t2707;
        float t2725 = t2724 / t2722;
        float t2726 = t2718 * t2710;
        float t2727 = t2726 / t2722;
        float t2728 = t2721 * t2712;
        float t2729 = t2728 / t2723;
        float t2730 = t2721 * t2715;
        float t2731 = t2730 / t2723;
        int t2732 = t2698 + _pr2701;
        memory[142753588 + t2732] = t2725;
        int t2734 = t2698 + _pr2701;
        int t2735 = t2734 + 1024;
        memory[142753588 + t2735] = t2727;
        int t2737 = t2698 + _pr2701;
        memory[176308020 + t2737] = t2729;
        int t2739 = t2698 + _pr2701;
        int t2740 = t2739 + 1024;
        memory[176308020 + t2740] = t2731;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2743 = 0; _pr2743 < 511; _pr2743++) {
        int t2744 = _pr2743 + 513;
        int t2745 = t2698 + t2744;
        memory[142753588 + t2745] = 0.0;
        int t2747 = t2698 + t2744;
        int t2748 = t2747 + 1024;
        memory[142753588 + t2748] = 0.0;
        int t2750 = t2698 + t2744;
        memory[176308020 + t2750] = 0.0;
        int t2752 = t2698 + t2744;
        int t2753 = t2752 + 1024;
        memory[176308020 + t2753] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
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
    /* loadGlobal(1521) - handled in variable access */
    int t2757 = id;
    int t2758 = t2757 * 2048;
    int t2759 = t2757 * 1024;
    float t2760 = t[14*frameCount + id] == 0.0;
    if (t2760) {
      for (uint t2762 = 0; t2762 < 1024; t2762++) {
        float t2763 = (float)t2762;
        float t2764 = (t2763 - metal::floor(t2763 / 2.0) * 2.0);
        float t2765 = t2764;
        float t2766 = (t2763 * 0.5);
        float t2767 = metal::floor(t2766);
        float t2768 = t2765 * 2.0;
        float t2769 = (t2767 - metal::floor(t2767 / 2.0) * 2.0);
        float t2770 = t2768 + t2769;
        float t2771 = (t2767 * 0.5);
        float t2772 = metal::floor(t2771);
        float t2773 = t2770 * 2.0;
        float t2774 = (t2772 - metal::floor(t2772 / 2.0) * 2.0);
        float t2775 = t2773 + t2774;
        float t2776 = (t2772 * 0.5);
        float t2777 = metal::floor(t2776);
        float t2778 = t2775 * 2.0;
        float t2779 = (t2777 - metal::floor(t2777 / 2.0) * 2.0);
        float t2780 = t2778 + t2779;
        float t2781 = (t2777 * 0.5);
        float t2782 = metal::floor(t2781);
        float t2783 = t2780 * 2.0;
        float t2784 = (t2782 - metal::floor(t2782 / 2.0) * 2.0);
        float t2785 = t2783 + t2784;
        float t2786 = (t2782 * 0.5);
        float t2787 = metal::floor(t2786);
        float t2788 = t2785 * 2.0;
        float t2789 = (t2787 - metal::floor(t2787 / 2.0) * 2.0);
        float t2790 = t2788 + t2789;
        float t2791 = (t2787 * 0.5);
        float t2792 = metal::floor(t2791);
        float t2793 = t2790 * 2.0;
        float t2794 = (t2792 - metal::floor(t2792 / 2.0) * 2.0);
        float t2795 = t2793 + t2794;
        float t2796 = (t2792 * 0.5);
        float t2797 = metal::floor(t2796);
        float t2798 = t2795 * 2.0;
        float t2799 = (t2797 - metal::floor(t2797 / 2.0) * 2.0);
        float t2800 = t2798 + t2799;
        float t2801 = (t2797 * 0.5);
        float t2802 = metal::floor(t2801);
        float t2803 = t2800 * 2.0;
        float t2804 = (t2802 - metal::floor(t2802 / 2.0) * 2.0);
        float t2805 = t2803 + t2804;
        float t2806 = (t2802 * 0.5);
        float t2807 = metal::floor(t2806);
        float t2808 = t2805 * 2.0;
        float t2809 = (t2807 - metal::floor(t2807 / 2.0) * 2.0);
        float t2810 = t2808 + t2809;
        float t2811 = (t2807 * 0.5);
        float t2812 = metal::floor(t2811);
        float t2813 = (float)t2762;
        float t2814 = t2813 < t2810;
        int t2815 = (int)t2810;
        int t2816 = t2758 + t2762;
        float t2817 = memory[142753588 + t2816];
        int t2818 = t2758 + t2762;
        int t2819 = t2818 + 1024;
        float t2820 = memory[142753588 + t2819];
        int t2821 = t2758 + t2815;
        float t2822 = memory[142753588 + t2821];
        int t2823 = t2758 + t2815;
        int t2824 = t2823 + 1024;
        float t2825 = memory[142753588 + t2824];
        float t2826 = metal::select(t2817, t2822, t2814 > 0.0);
        float t2827 = metal::select(t2820, t2825, t2814 > 0.0);
        float t2828 = metal::select(t2822, t2817, t2814 > 0.0);
        float t2829 = metal::select(t2825, t2820, t2814 > 0.0);
        int t2830 = t2758 + t2762;
        memory[142753588 + t2830] = t2826;
        int t2832 = t2758 + t2762;
        int t2833 = t2832 + 1024;
        memory[142753588 + t2833] = t2827;
        int t2835 = t2758 + t2815;
        memory[142753588 + t2835] = t2828;
        int t2837 = t2758 + t2815;
        int t2838 = t2837 + 1024;
        memory[142753588 + t2838] = t2829;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2841 = 0; _pr2841 < 512; _pr2841++) {
        float t2842 = (float)_pr2841;
        float t2843 = t2842;
        float t2844 = metal::floor(t2843);
        float t2845 = t2844;
        float t2846 = t2842 - t2845;
        float t2847 = t2844 * 2.0;
        float t2848 = t2847 + t2846;
        float t2849 = t2848 + 1.0;
        float t2850 = 6.283185 * t2846;
        float t2851 = (t2850 * 0.5);
        float t2852 = metal::cos(t2851);
        float t2853 = metal::sin(t2851);
        int t2854 = (int)t2848;
        int t2855 = (int)t2849;
        int t2856 = t2758 + t2854;
        float t2857 = memory[142753588 + t2856];
        int t2858 = t2758 + t2854;
        int t2859 = t2858 + 1024;
        float t2860 = memory[142753588 + t2859];
        int t2861 = t2758 + t2855;
        float t2862 = memory[142753588 + t2861];
        int t2863 = t2758 + t2855;
        int t2864 = t2863 + 1024;
        float t2865 = memory[142753588 + t2864];
        float t2866 = t2852 * t2862;
        float t2867 = t2853 * t2865;
        float t2868 = t2866 - t2867;
        float t2869 = t2852 * t2865;
        float t2870 = t2853 * t2862;
        float t2871 = t2869 + t2870;
        int t2872 = t2758 + t2854;
        float t2873 = t2857 + t2868;
        memory[142753588 + t2872] = t2873;
        int t2875 = t2758 + t2854;
        int t2876 = t2875 + 1024;
        float t2877 = t2860 + t2871;
        memory[142753588 + t2876] = t2877;
        int t2879 = t2758 + t2855;
        float t2880 = t2857 - t2868;
        memory[142753588 + t2879] = t2880;
        int t2882 = t2758 + t2855;
        int t2883 = t2882 + 1024;
        float t2884 = t2860 - t2871;
        memory[142753588 + t2883] = t2884;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2887 = 0; _pr2887 < 512; _pr2887++) {
        float t2888 = (float)_pr2887;
        float t2889 = (t2888 * 0.5);
        float t2890 = metal::floor(t2889);
        float t2891 = t2890 * 2.0;
        float t2892 = t2888 - t2891;
        float t2893 = t2890 * 4.0;
        float t2894 = t2893 + t2892;
        float t2895 = t2894 + 2.0;
        float t2896 = 6.283185 * t2892;
        float t2897 = (t2896 * 0.25);
        float t2898 = metal::cos(t2897);
        float t2899 = metal::sin(t2897);
        int t2900 = (int)t2894;
        int t2901 = (int)t2895;
        int t2902 = t2758 + t2900;
        float t2903 = memory[142753588 + t2902];
        int t2904 = t2758 + t2900;
        int t2905 = t2904 + 1024;
        float t2906 = memory[142753588 + t2905];
        int t2907 = t2758 + t2901;
        float t2908 = memory[142753588 + t2907];
        int t2909 = t2758 + t2901;
        int t2910 = t2909 + 1024;
        float t2911 = memory[142753588 + t2910];
        float t2912 = t2898 * t2908;
        float t2913 = t2899 * t2911;
        float t2914 = t2912 - t2913;
        float t2915 = t2898 * t2911;
        float t2916 = t2899 * t2908;
        float t2917 = t2915 + t2916;
        int t2918 = t2758 + t2900;
        float t2919 = t2903 + t2914;
        memory[142753588 + t2918] = t2919;
        int t2921 = t2758 + t2900;
        int t2922 = t2921 + 1024;
        float t2923 = t2906 + t2917;
        memory[142753588 + t2922] = t2923;
        int t2925 = t2758 + t2901;
        float t2926 = t2903 - t2914;
        memory[142753588 + t2925] = t2926;
        int t2928 = t2758 + t2901;
        int t2929 = t2928 + 1024;
        float t2930 = t2906 - t2917;
        memory[142753588 + t2929] = t2930;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2933 = 0; _pr2933 < 512; _pr2933++) {
        float t2934 = (float)_pr2933;
        float t2935 = (t2934 * 0.25);
        float t2936 = metal::floor(t2935);
        float t2937 = t2936 * 4.0;
        float t2938 = t2934 - t2937;
        float t2939 = t2936 * 8.0;
        float t2940 = t2939 + t2938;
        float t2941 = t2940 + 4.0;
        float t2942 = 6.283185 * t2938;
        float t2943 = (t2942 * 0.125);
        float t2944 = metal::cos(t2943);
        float t2945 = metal::sin(t2943);
        int t2946 = (int)t2940;
        int t2947 = (int)t2941;
        int t2948 = t2758 + t2946;
        float t2949 = memory[142753588 + t2948];
        int t2950 = t2758 + t2946;
        int t2951 = t2950 + 1024;
        float t2952 = memory[142753588 + t2951];
        int t2953 = t2758 + t2947;
        float t2954 = memory[142753588 + t2953];
        int t2955 = t2758 + t2947;
        int t2956 = t2955 + 1024;
        float t2957 = memory[142753588 + t2956];
        float t2958 = t2944 * t2954;
        float t2959 = t2945 * t2957;
        float t2960 = t2958 - t2959;
        float t2961 = t2944 * t2957;
        float t2962 = t2945 * t2954;
        float t2963 = t2961 + t2962;
        int t2964 = t2758 + t2946;
        float t2965 = t2949 + t2960;
        memory[142753588 + t2964] = t2965;
        int t2967 = t2758 + t2946;
        int t2968 = t2967 + 1024;
        float t2969 = t2952 + t2963;
        memory[142753588 + t2968] = t2969;
        int t2971 = t2758 + t2947;
        float t2972 = t2949 - t2960;
        memory[142753588 + t2971] = t2972;
        int t2974 = t2758 + t2947;
        int t2975 = t2974 + 1024;
        float t2976 = t2952 - t2963;
        memory[142753588 + t2975] = t2976;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr2979 = 0; _pr2979 < 512; _pr2979++) {
        float t2980 = (float)_pr2979;
        float t2981 = (t2980 * 0.125);
        float t2982 = metal::floor(t2981);
        float t2983 = t2982 * 8.0;
        float t2984 = t2980 - t2983;
        float t2985 = t2982 * 16.0;
        float t2986 = t2985 + t2984;
        float t2987 = t2986 + 8.0;
        float t2988 = 6.283185 * t2984;
        float t2989 = (t2988 * 0.0625);
        float t2990 = metal::cos(t2989);
        float t2991 = metal::sin(t2989);
        int t2992 = (int)t2986;
        int t2993 = (int)t2987;
        int t2994 = t2758 + t2992;
        float t2995 = memory[142753588 + t2994];
        int t2996 = t2758 + t2992;
        int t2997 = t2996 + 1024;
        float t2998 = memory[142753588 + t2997];
        int t2999 = t2758 + t2993;
        float t3000 = memory[142753588 + t2999];
        int t3001 = t2758 + t2993;
        int t3002 = t3001 + 1024;
        float t3003 = memory[142753588 + t3002];
        float t3004 = t2990 * t3000;
        float t3005 = t2991 * t3003;
        float t3006 = t3004 - t3005;
        float t3007 = t2990 * t3003;
        float t3008 = t2991 * t3000;
        float t3009 = t3007 + t3008;
        int t3010 = t2758 + t2992;
        float t3011 = t2995 + t3006;
        memory[142753588 + t3010] = t3011;
        int t3013 = t2758 + t2992;
        int t3014 = t3013 + 1024;
        float t3015 = t2998 + t3009;
        memory[142753588 + t3014] = t3015;
        int t3017 = t2758 + t2993;
        float t3018 = t2995 - t3006;
        memory[142753588 + t3017] = t3018;
        int t3020 = t2758 + t2993;
        int t3021 = t3020 + 1024;
        float t3022 = t2998 - t3009;
        memory[142753588 + t3021] = t3022;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3025 = 0; _pr3025 < 512; _pr3025++) {
        float t3026 = (float)_pr3025;
        float t3027 = (t3026 * 0.0625);
        float t3028 = metal::floor(t3027);
        float t3029 = t3028 * 16.0;
        float t3030 = t3026 - t3029;
        float t3031 = t3028 * 32.0;
        float t3032 = t3031 + t3030;
        float t3033 = t3032 + 16.0;
        float t3034 = 6.283185 * t3030;
        float t3035 = (t3034 * 0.03125);
        float t3036 = metal::cos(t3035);
        float t3037 = metal::sin(t3035);
        int t3038 = (int)t3032;
        int t3039 = (int)t3033;
        int t3040 = t2758 + t3038;
        float t3041 = memory[142753588 + t3040];
        int t3042 = t2758 + t3038;
        int t3043 = t3042 + 1024;
        float t3044 = memory[142753588 + t3043];
        int t3045 = t2758 + t3039;
        float t3046 = memory[142753588 + t3045];
        int t3047 = t2758 + t3039;
        int t3048 = t3047 + 1024;
        float t3049 = memory[142753588 + t3048];
        float t3050 = t3036 * t3046;
        float t3051 = t3037 * t3049;
        float t3052 = t3050 - t3051;
        float t3053 = t3036 * t3049;
        float t3054 = t3037 * t3046;
        float t3055 = t3053 + t3054;
        int t3056 = t2758 + t3038;
        float t3057 = t3041 + t3052;
        memory[142753588 + t3056] = t3057;
        int t3059 = t2758 + t3038;
        int t3060 = t3059 + 1024;
        float t3061 = t3044 + t3055;
        memory[142753588 + t3060] = t3061;
        int t3063 = t2758 + t3039;
        float t3064 = t3041 - t3052;
        memory[142753588 + t3063] = t3064;
        int t3066 = t2758 + t3039;
        int t3067 = t3066 + 1024;
        float t3068 = t3044 - t3055;
        memory[142753588 + t3067] = t3068;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3071 = 0; _pr3071 < 512; _pr3071++) {
        float t3072 = (float)_pr3071;
        float t3073 = (t3072 * 0.03125);
        float t3074 = metal::floor(t3073);
        float t3075 = t3074 * 32.0;
        float t3076 = t3072 - t3075;
        float t3077 = t3074 * 64.0;
        float t3078 = t3077 + t3076;
        float t3079 = t3078 + 32.0;
        float t3080 = 6.283185 * t3076;
        float t3081 = (t3080 * 0.015625);
        float t3082 = metal::cos(t3081);
        float t3083 = metal::sin(t3081);
        int t3084 = (int)t3078;
        int t3085 = (int)t3079;
        int t3086 = t2758 + t3084;
        float t3087 = memory[142753588 + t3086];
        int t3088 = t2758 + t3084;
        int t3089 = t3088 + 1024;
        float t3090 = memory[142753588 + t3089];
        int t3091 = t2758 + t3085;
        float t3092 = memory[142753588 + t3091];
        int t3093 = t2758 + t3085;
        int t3094 = t3093 + 1024;
        float t3095 = memory[142753588 + t3094];
        float t3096 = t3082 * t3092;
        float t3097 = t3083 * t3095;
        float t3098 = t3096 - t3097;
        float t3099 = t3082 * t3095;
        float t3100 = t3083 * t3092;
        float t3101 = t3099 + t3100;
        int t3102 = t2758 + t3084;
        float t3103 = t3087 + t3098;
        memory[142753588 + t3102] = t3103;
        int t3105 = t2758 + t3084;
        int t3106 = t3105 + 1024;
        float t3107 = t3090 + t3101;
        memory[142753588 + t3106] = t3107;
        int t3109 = t2758 + t3085;
        float t3110 = t3087 - t3098;
        memory[142753588 + t3109] = t3110;
        int t3112 = t2758 + t3085;
        int t3113 = t3112 + 1024;
        float t3114 = t3090 - t3101;
        memory[142753588 + t3113] = t3114;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3117 = 0; _pr3117 < 512; _pr3117++) {
        float t3118 = (float)_pr3117;
        float t3119 = (t3118 * 0.015625);
        float t3120 = metal::floor(t3119);
        float t3121 = t3120 * 64.0;
        float t3122 = t3118 - t3121;
        float t3123 = t3120 * 128.0;
        float t3124 = t3123 + t3122;
        float t3125 = t3124 + 64.0;
        float t3126 = 6.283185 * t3122;
        float t3127 = (t3126 * 0.0078125);
        float t3128 = metal::cos(t3127);
        float t3129 = metal::sin(t3127);
        int t3130 = (int)t3124;
        int t3131 = (int)t3125;
        int t3132 = t2758 + t3130;
        float t3133 = memory[142753588 + t3132];
        int t3134 = t2758 + t3130;
        int t3135 = t3134 + 1024;
        float t3136 = memory[142753588 + t3135];
        int t3137 = t2758 + t3131;
        float t3138 = memory[142753588 + t3137];
        int t3139 = t2758 + t3131;
        int t3140 = t3139 + 1024;
        float t3141 = memory[142753588 + t3140];
        float t3142 = t3128 * t3138;
        float t3143 = t3129 * t3141;
        float t3144 = t3142 - t3143;
        float t3145 = t3128 * t3141;
        float t3146 = t3129 * t3138;
        float t3147 = t3145 + t3146;
        int t3148 = t2758 + t3130;
        float t3149 = t3133 + t3144;
        memory[142753588 + t3148] = t3149;
        int t3151 = t2758 + t3130;
        int t3152 = t3151 + 1024;
        float t3153 = t3136 + t3147;
        memory[142753588 + t3152] = t3153;
        int t3155 = t2758 + t3131;
        float t3156 = t3133 - t3144;
        memory[142753588 + t3155] = t3156;
        int t3158 = t2758 + t3131;
        int t3159 = t3158 + 1024;
        float t3160 = t3136 - t3147;
        memory[142753588 + t3159] = t3160;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3163 = 0; _pr3163 < 512; _pr3163++) {
        float t3164 = (float)_pr3163;
        float t3165 = (t3164 * 0.0078125);
        float t3166 = metal::floor(t3165);
        float t3167 = t3166 * 128.0;
        float t3168 = t3164 - t3167;
        float t3169 = t3166 * 256.0;
        float t3170 = t3169 + t3168;
        float t3171 = t3170 + 128.0;
        float t3172 = 6.283185 * t3168;
        float t3173 = (t3172 * 0.00390625);
        float t3174 = metal::cos(t3173);
        float t3175 = metal::sin(t3173);
        int t3176 = (int)t3170;
        int t3177 = (int)t3171;
        int t3178 = t2758 + t3176;
        float t3179 = memory[142753588 + t3178];
        int t3180 = t2758 + t3176;
        int t3181 = t3180 + 1024;
        float t3182 = memory[142753588 + t3181];
        int t3183 = t2758 + t3177;
        float t3184 = memory[142753588 + t3183];
        int t3185 = t2758 + t3177;
        int t3186 = t3185 + 1024;
        float t3187 = memory[142753588 + t3186];
        float t3188 = t3174 * t3184;
        float t3189 = t3175 * t3187;
        float t3190 = t3188 - t3189;
        float t3191 = t3174 * t3187;
        float t3192 = t3175 * t3184;
        float t3193 = t3191 + t3192;
        int t3194 = t2758 + t3176;
        float t3195 = t3179 + t3190;
        memory[142753588 + t3194] = t3195;
        int t3197 = t2758 + t3176;
        int t3198 = t3197 + 1024;
        float t3199 = t3182 + t3193;
        memory[142753588 + t3198] = t3199;
        int t3201 = t2758 + t3177;
        float t3202 = t3179 - t3190;
        memory[142753588 + t3201] = t3202;
        int t3204 = t2758 + t3177;
        int t3205 = t3204 + 1024;
        float t3206 = t3182 - t3193;
        memory[142753588 + t3205] = t3206;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3209 = 0; _pr3209 < 512; _pr3209++) {
        float t3210 = (float)_pr3209;
        float t3211 = (t3210 * 0.00390625);
        float t3212 = metal::floor(t3211);
        float t3213 = t3212 * 256.0;
        float t3214 = t3210 - t3213;
        float t3215 = t3212 * 512.0;
        float t3216 = t3215 + t3214;
        float t3217 = t3216 + 256.0;
        float t3218 = 6.283185 * t3214;
        float t3219 = (t3218 * 0.001953125);
        float t3220 = metal::cos(t3219);
        float t3221 = metal::sin(t3219);
        int t3222 = (int)t3216;
        int t3223 = (int)t3217;
        int t3224 = t2758 + t3222;
        float t3225 = memory[142753588 + t3224];
        int t3226 = t2758 + t3222;
        int t3227 = t3226 + 1024;
        float t3228 = memory[142753588 + t3227];
        int t3229 = t2758 + t3223;
        float t3230 = memory[142753588 + t3229];
        int t3231 = t2758 + t3223;
        int t3232 = t3231 + 1024;
        float t3233 = memory[142753588 + t3232];
        float t3234 = t3220 * t3230;
        float t3235 = t3221 * t3233;
        float t3236 = t3234 - t3235;
        float t3237 = t3220 * t3233;
        float t3238 = t3221 * t3230;
        float t3239 = t3237 + t3238;
        int t3240 = t2758 + t3222;
        float t3241 = t3225 + t3236;
        memory[142753588 + t3240] = t3241;
        int t3243 = t2758 + t3222;
        int t3244 = t3243 + 1024;
        float t3245 = t3228 + t3239;
        memory[142753588 + t3244] = t3245;
        int t3247 = t2758 + t3223;
        float t3248 = t3225 - t3236;
        memory[142753588 + t3247] = t3248;
        int t3250 = t2758 + t3223;
        int t3251 = t3250 + 1024;
        float t3252 = t3228 - t3239;
        memory[142753588 + t3251] = t3252;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3255 = 0; _pr3255 < 512; _pr3255++) {
        float t3256 = (float)_pr3255;
        float t3257 = (t3256 * 0.001953125);
        float t3258 = metal::floor(t3257);
        float t3259 = t3258 * 512.0;
        float t3260 = t3256 - t3259;
        float t3261 = t3258 * 1024.0;
        float t3262 = t3261 + t3260;
        float t3263 = t3262 + 512.0;
        float t3264 = 6.283185 * t3260;
        float t3265 = (t3264 * 0.0009765625);
        float t3266 = metal::cos(t3265);
        float t3267 = metal::sin(t3265);
        int t3268 = (int)t3262;
        int t3269 = (int)t3263;
        int t3270 = t2758 + t3268;
        float t3271 = memory[142753588 + t3270];
        int t3272 = t2758 + t3268;
        int t3273 = t3272 + 1024;
        float t3274 = memory[142753588 + t3273];
        int t3275 = t2758 + t3269;
        float t3276 = memory[142753588 + t3275];
        int t3277 = t2758 + t3269;
        int t3278 = t3277 + 1024;
        float t3279 = memory[142753588 + t3278];
        float t3280 = t3266 * t3276;
        float t3281 = t3267 * t3279;
        float t3282 = t3280 - t3281;
        float t3283 = t3266 * t3279;
        float t3284 = t3267 * t3276;
        float t3285 = t3283 + t3284;
        int t3286 = t2758 + t3268;
        float t3287 = t3271 + t3282;
        memory[142753588 + t3286] = t3287;
        int t3289 = t2758 + t3268;
        int t3290 = t3289 + 1024;
        float t3291 = t3274 + t3285;
        memory[142753588 + t3290] = t3291;
        int t3293 = t2758 + t3269;
        float t3294 = t3271 - t3282;
        memory[142753588 + t3293] = t3294;
        int t3296 = t2758 + t3269;
        int t3297 = t3296 + 1024;
        float t3298 = t3274 - t3285;
        memory[142753588 + t3297] = t3298;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3301 = 0; _pr3301 < 1024; _pr3301++) {
        int t3302 = t2758 + _pr3301;
        float t3303 = memory[142753588 + t3302];
        float t3304 = t3303 * 1.9036306e-06;
        float t3305 = memory[33012 + (int)_pr3301];
        int t3306 = t2759 + _pr3301;
        float t3307 = t3304 * t3305;
        memory[50429748 + t3306] = t3307;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t3310 = 0; t3310 < 1024; t3310++) {
        float t3311 = (float)t3310;
        float t3312 = (t3311 - metal::floor(t3311 / 2.0) * 2.0);
        float t3313 = t3312;
        float t3314 = (t3311 * 0.5);
        float t3315 = metal::floor(t3314);
        float t3316 = t3313 * 2.0;
        float t3317 = (t3315 - metal::floor(t3315 / 2.0) * 2.0);
        float t3318 = t3316 + t3317;
        float t3319 = (t3315 * 0.5);
        float t3320 = metal::floor(t3319);
        float t3321 = t3318 * 2.0;
        float t3322 = (t3320 - metal::floor(t3320 / 2.0) * 2.0);
        float t3323 = t3321 + t3322;
        float t3324 = (t3320 * 0.5);
        float t3325 = metal::floor(t3324);
        float t3326 = t3323 * 2.0;
        float t3327 = (t3325 - metal::floor(t3325 / 2.0) * 2.0);
        float t3328 = t3326 + t3327;
        float t3329 = (t3325 * 0.5);
        float t3330 = metal::floor(t3329);
        float t3331 = t3328 * 2.0;
        float t3332 = (t3330 - metal::floor(t3330 / 2.0) * 2.0);
        float t3333 = t3331 + t3332;
        float t3334 = (t3330 * 0.5);
        float t3335 = metal::floor(t3334);
        float t3336 = t3333 * 2.0;
        float t3337 = (t3335 - metal::floor(t3335 / 2.0) * 2.0);
        float t3338 = t3336 + t3337;
        float t3339 = (t3335 * 0.5);
        float t3340 = metal::floor(t3339);
        float t3341 = t3338 * 2.0;
        float t3342 = (t3340 - metal::floor(t3340 / 2.0) * 2.0);
        float t3343 = t3341 + t3342;
        float t3344 = (t3340 * 0.5);
        float t3345 = metal::floor(t3344);
        float t3346 = t3343 * 2.0;
        float t3347 = (t3345 - metal::floor(t3345 / 2.0) * 2.0);
        float t3348 = t3346 + t3347;
        float t3349 = (t3345 * 0.5);
        float t3350 = metal::floor(t3349);
        float t3351 = t3348 * 2.0;
        float t3352 = (t3350 - metal::floor(t3350 / 2.0) * 2.0);
        float t3353 = t3351 + t3352;
        float t3354 = (t3350 * 0.5);
        float t3355 = metal::floor(t3354);
        float t3356 = t3353 * 2.0;
        float t3357 = (t3355 - metal::floor(t3355 / 2.0) * 2.0);
        float t3358 = t3356 + t3357;
        float t3359 = (t3355 * 0.5);
        float t3360 = metal::floor(t3359);
        float t3361 = (float)t3310;
        float t3362 = t3361 < t3358;
        int t3363 = (int)t3358;
        int t3364 = t2758 + t3310;
        float t3365 = memory[176308020 + t3364];
        int t3366 = t2758 + t3310;
        int t3367 = t3366 + 1024;
        float t3368 = memory[176308020 + t3367];
        int t3369 = t2758 + t3363;
        float t3370 = memory[176308020 + t3369];
        int t3371 = t2758 + t3363;
        int t3372 = t3371 + 1024;
        float t3373 = memory[176308020 + t3372];
        float t3374 = metal::select(t3365, t3370, t3362 > 0.0);
        float t3375 = metal::select(t3368, t3373, t3362 > 0.0);
        float t3376 = metal::select(t3370, t3365, t3362 > 0.0);
        float t3377 = metal::select(t3373, t3368, t3362 > 0.0);
        int t3378 = t2758 + t3310;
        memory[176308020 + t3378] = t3374;
        int t3380 = t2758 + t3310;
        int t3381 = t3380 + 1024;
        memory[176308020 + t3381] = t3375;
        int t3383 = t2758 + t3363;
        memory[176308020 + t3383] = t3376;
        int t3385 = t2758 + t3363;
        int t3386 = t3385 + 1024;
        memory[176308020 + t3386] = t3377;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3389 = 0; _pr3389 < 512; _pr3389++) {
        float t3390 = (float)_pr3389;
        float t3391 = t3390;
        float t3392 = metal::floor(t3391);
        float t3393 = t3392;
        float t3394 = t3390 - t3393;
        float t3395 = t3392 * 2.0;
        float t3396 = t3395 + t3394;
        float t3397 = t3396 + 1.0;
        float t3398 = 6.283185 * t3394;
        float t3399 = (t3398 * 0.5);
        float t3400 = metal::cos(t3399);
        float t3401 = metal::sin(t3399);
        int t3402 = (int)t3396;
        int t3403 = (int)t3397;
        int t3404 = t2758 + t3402;
        float t3405 = memory[176308020 + t3404];
        int t3406 = t2758 + t3402;
        int t3407 = t3406 + 1024;
        float t3408 = memory[176308020 + t3407];
        int t3409 = t2758 + t3403;
        float t3410 = memory[176308020 + t3409];
        int t3411 = t2758 + t3403;
        int t3412 = t3411 + 1024;
        float t3413 = memory[176308020 + t3412];
        float t3414 = t3400 * t3410;
        float t3415 = t3401 * t3413;
        float t3416 = t3414 - t3415;
        float t3417 = t3400 * t3413;
        float t3418 = t3401 * t3410;
        float t3419 = t3417 + t3418;
        int t3420 = t2758 + t3402;
        float t3421 = t3405 + t3416;
        memory[176308020 + t3420] = t3421;
        int t3423 = t2758 + t3402;
        int t3424 = t3423 + 1024;
        float t3425 = t3408 + t3419;
        memory[176308020 + t3424] = t3425;
        int t3427 = t2758 + t3403;
        float t3428 = t3405 - t3416;
        memory[176308020 + t3427] = t3428;
        int t3430 = t2758 + t3403;
        int t3431 = t3430 + 1024;
        float t3432 = t3408 - t3419;
        memory[176308020 + t3431] = t3432;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3435 = 0; _pr3435 < 512; _pr3435++) {
        float t3436 = (float)_pr3435;
        float t3437 = (t3436 * 0.5);
        float t3438 = metal::floor(t3437);
        float t3439 = t3438 * 2.0;
        float t3440 = t3436 - t3439;
        float t3441 = t3438 * 4.0;
        float t3442 = t3441 + t3440;
        float t3443 = t3442 + 2.0;
        float t3444 = 6.283185 * t3440;
        float t3445 = (t3444 * 0.25);
        float t3446 = metal::cos(t3445);
        float t3447 = metal::sin(t3445);
        int t3448 = (int)t3442;
        int t3449 = (int)t3443;
        int t3450 = t2758 + t3448;
        float t3451 = memory[176308020 + t3450];
        int t3452 = t2758 + t3448;
        int t3453 = t3452 + 1024;
        float t3454 = memory[176308020 + t3453];
        int t3455 = t2758 + t3449;
        float t3456 = memory[176308020 + t3455];
        int t3457 = t2758 + t3449;
        int t3458 = t3457 + 1024;
        float t3459 = memory[176308020 + t3458];
        float t3460 = t3446 * t3456;
        float t3461 = t3447 * t3459;
        float t3462 = t3460 - t3461;
        float t3463 = t3446 * t3459;
        float t3464 = t3447 * t3456;
        float t3465 = t3463 + t3464;
        int t3466 = t2758 + t3448;
        float t3467 = t3451 + t3462;
        memory[176308020 + t3466] = t3467;
        int t3469 = t2758 + t3448;
        int t3470 = t3469 + 1024;
        float t3471 = t3454 + t3465;
        memory[176308020 + t3470] = t3471;
        int t3473 = t2758 + t3449;
        float t3474 = t3451 - t3462;
        memory[176308020 + t3473] = t3474;
        int t3476 = t2758 + t3449;
        int t3477 = t3476 + 1024;
        float t3478 = t3454 - t3465;
        memory[176308020 + t3477] = t3478;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3481 = 0; _pr3481 < 512; _pr3481++) {
        float t3482 = (float)_pr3481;
        float t3483 = (t3482 * 0.25);
        float t3484 = metal::floor(t3483);
        float t3485 = t3484 * 4.0;
        float t3486 = t3482 - t3485;
        float t3487 = t3484 * 8.0;
        float t3488 = t3487 + t3486;
        float t3489 = t3488 + 4.0;
        float t3490 = 6.283185 * t3486;
        float t3491 = (t3490 * 0.125);
        float t3492 = metal::cos(t3491);
        float t3493 = metal::sin(t3491);
        int t3494 = (int)t3488;
        int t3495 = (int)t3489;
        int t3496 = t2758 + t3494;
        float t3497 = memory[176308020 + t3496];
        int t3498 = t2758 + t3494;
        int t3499 = t3498 + 1024;
        float t3500 = memory[176308020 + t3499];
        int t3501 = t2758 + t3495;
        float t3502 = memory[176308020 + t3501];
        int t3503 = t2758 + t3495;
        int t3504 = t3503 + 1024;
        float t3505 = memory[176308020 + t3504];
        float t3506 = t3492 * t3502;
        float t3507 = t3493 * t3505;
        float t3508 = t3506 - t3507;
        float t3509 = t3492 * t3505;
        float t3510 = t3493 * t3502;
        float t3511 = t3509 + t3510;
        int t3512 = t2758 + t3494;
        float t3513 = t3497 + t3508;
        memory[176308020 + t3512] = t3513;
        int t3515 = t2758 + t3494;
        int t3516 = t3515 + 1024;
        float t3517 = t3500 + t3511;
        memory[176308020 + t3516] = t3517;
        int t3519 = t2758 + t3495;
        float t3520 = t3497 - t3508;
        memory[176308020 + t3519] = t3520;
        int t3522 = t2758 + t3495;
        int t3523 = t3522 + 1024;
        float t3524 = t3500 - t3511;
        memory[176308020 + t3523] = t3524;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3527 = 0; _pr3527 < 512; _pr3527++) {
        float t3528 = (float)_pr3527;
        float t3529 = (t3528 * 0.125);
        float t3530 = metal::floor(t3529);
        float t3531 = t3530 * 8.0;
        float t3532 = t3528 - t3531;
        float t3533 = t3530 * 16.0;
        float t3534 = t3533 + t3532;
        float t3535 = t3534 + 8.0;
        float t3536 = 6.283185 * t3532;
        float t3537 = (t3536 * 0.0625);
        float t3538 = metal::cos(t3537);
        float t3539 = metal::sin(t3537);
        int t3540 = (int)t3534;
        int t3541 = (int)t3535;
        int t3542 = t2758 + t3540;
        float t3543 = memory[176308020 + t3542];
        int t3544 = t2758 + t3540;
        int t3545 = t3544 + 1024;
        float t3546 = memory[176308020 + t3545];
        int t3547 = t2758 + t3541;
        float t3548 = memory[176308020 + t3547];
        int t3549 = t2758 + t3541;
        int t3550 = t3549 + 1024;
        float t3551 = memory[176308020 + t3550];
        float t3552 = t3538 * t3548;
        float t3553 = t3539 * t3551;
        float t3554 = t3552 - t3553;
        float t3555 = t3538 * t3551;
        float t3556 = t3539 * t3548;
        float t3557 = t3555 + t3556;
        int t3558 = t2758 + t3540;
        float t3559 = t3543 + t3554;
        memory[176308020 + t3558] = t3559;
        int t3561 = t2758 + t3540;
        int t3562 = t3561 + 1024;
        float t3563 = t3546 + t3557;
        memory[176308020 + t3562] = t3563;
        int t3565 = t2758 + t3541;
        float t3566 = t3543 - t3554;
        memory[176308020 + t3565] = t3566;
        int t3568 = t2758 + t3541;
        int t3569 = t3568 + 1024;
        float t3570 = t3546 - t3557;
        memory[176308020 + t3569] = t3570;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3573 = 0; _pr3573 < 512; _pr3573++) {
        float t3574 = (float)_pr3573;
        float t3575 = (t3574 * 0.0625);
        float t3576 = metal::floor(t3575);
        float t3577 = t3576 * 16.0;
        float t3578 = t3574 - t3577;
        float t3579 = t3576 * 32.0;
        float t3580 = t3579 + t3578;
        float t3581 = t3580 + 16.0;
        float t3582 = 6.283185 * t3578;
        float t3583 = (t3582 * 0.03125);
        float t3584 = metal::cos(t3583);
        float t3585 = metal::sin(t3583);
        int t3586 = (int)t3580;
        int t3587 = (int)t3581;
        int t3588 = t2758 + t3586;
        float t3589 = memory[176308020 + t3588];
        int t3590 = t2758 + t3586;
        int t3591 = t3590 + 1024;
        float t3592 = memory[176308020 + t3591];
        int t3593 = t2758 + t3587;
        float t3594 = memory[176308020 + t3593];
        int t3595 = t2758 + t3587;
        int t3596 = t3595 + 1024;
        float t3597 = memory[176308020 + t3596];
        float t3598 = t3584 * t3594;
        float t3599 = t3585 * t3597;
        float t3600 = t3598 - t3599;
        float t3601 = t3584 * t3597;
        float t3602 = t3585 * t3594;
        float t3603 = t3601 + t3602;
        int t3604 = t2758 + t3586;
        float t3605 = t3589 + t3600;
        memory[176308020 + t3604] = t3605;
        int t3607 = t2758 + t3586;
        int t3608 = t3607 + 1024;
        float t3609 = t3592 + t3603;
        memory[176308020 + t3608] = t3609;
        int t3611 = t2758 + t3587;
        float t3612 = t3589 - t3600;
        memory[176308020 + t3611] = t3612;
        int t3614 = t2758 + t3587;
        int t3615 = t3614 + 1024;
        float t3616 = t3592 - t3603;
        memory[176308020 + t3615] = t3616;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3619 = 0; _pr3619 < 512; _pr3619++) {
        float t3620 = (float)_pr3619;
        float t3621 = (t3620 * 0.03125);
        float t3622 = metal::floor(t3621);
        float t3623 = t3622 * 32.0;
        float t3624 = t3620 - t3623;
        float t3625 = t3622 * 64.0;
        float t3626 = t3625 + t3624;
        float t3627 = t3626 + 32.0;
        float t3628 = 6.283185 * t3624;
        float t3629 = (t3628 * 0.015625);
        float t3630 = metal::cos(t3629);
        float t3631 = metal::sin(t3629);
        int t3632 = (int)t3626;
        int t3633 = (int)t3627;
        int t3634 = t2758 + t3632;
        float t3635 = memory[176308020 + t3634];
        int t3636 = t2758 + t3632;
        int t3637 = t3636 + 1024;
        float t3638 = memory[176308020 + t3637];
        int t3639 = t2758 + t3633;
        float t3640 = memory[176308020 + t3639];
        int t3641 = t2758 + t3633;
        int t3642 = t3641 + 1024;
        float t3643 = memory[176308020 + t3642];
        float t3644 = t3630 * t3640;
        float t3645 = t3631 * t3643;
        float t3646 = t3644 - t3645;
        float t3647 = t3630 * t3643;
        float t3648 = t3631 * t3640;
        float t3649 = t3647 + t3648;
        int t3650 = t2758 + t3632;
        float t3651 = t3635 + t3646;
        memory[176308020 + t3650] = t3651;
        int t3653 = t2758 + t3632;
        int t3654 = t3653 + 1024;
        float t3655 = t3638 + t3649;
        memory[176308020 + t3654] = t3655;
        int t3657 = t2758 + t3633;
        float t3658 = t3635 - t3646;
        memory[176308020 + t3657] = t3658;
        int t3660 = t2758 + t3633;
        int t3661 = t3660 + 1024;
        float t3662 = t3638 - t3649;
        memory[176308020 + t3661] = t3662;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3665 = 0; _pr3665 < 512; _pr3665++) {
        float t3666 = (float)_pr3665;
        float t3667 = (t3666 * 0.015625);
        float t3668 = metal::floor(t3667);
        float t3669 = t3668 * 64.0;
        float t3670 = t3666 - t3669;
        float t3671 = t3668 * 128.0;
        float t3672 = t3671 + t3670;
        float t3673 = t3672 + 64.0;
        float t3674 = 6.283185 * t3670;
        float t3675 = (t3674 * 0.0078125);
        float t3676 = metal::cos(t3675);
        float t3677 = metal::sin(t3675);
        int t3678 = (int)t3672;
        int t3679 = (int)t3673;
        int t3680 = t2758 + t3678;
        float t3681 = memory[176308020 + t3680];
        int t3682 = t2758 + t3678;
        int t3683 = t3682 + 1024;
        float t3684 = memory[176308020 + t3683];
        int t3685 = t2758 + t3679;
        float t3686 = memory[176308020 + t3685];
        int t3687 = t2758 + t3679;
        int t3688 = t3687 + 1024;
        float t3689 = memory[176308020 + t3688];
        float t3690 = t3676 * t3686;
        float t3691 = t3677 * t3689;
        float t3692 = t3690 - t3691;
        float t3693 = t3676 * t3689;
        float t3694 = t3677 * t3686;
        float t3695 = t3693 + t3694;
        int t3696 = t2758 + t3678;
        float t3697 = t3681 + t3692;
        memory[176308020 + t3696] = t3697;
        int t3699 = t2758 + t3678;
        int t3700 = t3699 + 1024;
        float t3701 = t3684 + t3695;
        memory[176308020 + t3700] = t3701;
        int t3703 = t2758 + t3679;
        float t3704 = t3681 - t3692;
        memory[176308020 + t3703] = t3704;
        int t3706 = t2758 + t3679;
        int t3707 = t3706 + 1024;
        float t3708 = t3684 - t3695;
        memory[176308020 + t3707] = t3708;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3711 = 0; _pr3711 < 512; _pr3711++) {
        float t3712 = (float)_pr3711;
        float t3713 = (t3712 * 0.0078125);
        float t3714 = metal::floor(t3713);
        float t3715 = t3714 * 128.0;
        float t3716 = t3712 - t3715;
        float t3717 = t3714 * 256.0;
        float t3718 = t3717 + t3716;
        float t3719 = t3718 + 128.0;
        float t3720 = 6.283185 * t3716;
        float t3721 = (t3720 * 0.00390625);
        float t3722 = metal::cos(t3721);
        float t3723 = metal::sin(t3721);
        int t3724 = (int)t3718;
        int t3725 = (int)t3719;
        int t3726 = t2758 + t3724;
        float t3727 = memory[176308020 + t3726];
        int t3728 = t2758 + t3724;
        int t3729 = t3728 + 1024;
        float t3730 = memory[176308020 + t3729];
        int t3731 = t2758 + t3725;
        float t3732 = memory[176308020 + t3731];
        int t3733 = t2758 + t3725;
        int t3734 = t3733 + 1024;
        float t3735 = memory[176308020 + t3734];
        float t3736 = t3722 * t3732;
        float t3737 = t3723 * t3735;
        float t3738 = t3736 - t3737;
        float t3739 = t3722 * t3735;
        float t3740 = t3723 * t3732;
        float t3741 = t3739 + t3740;
        int t3742 = t2758 + t3724;
        float t3743 = t3727 + t3738;
        memory[176308020 + t3742] = t3743;
        int t3745 = t2758 + t3724;
        int t3746 = t3745 + 1024;
        float t3747 = t3730 + t3741;
        memory[176308020 + t3746] = t3747;
        int t3749 = t2758 + t3725;
        float t3750 = t3727 - t3738;
        memory[176308020 + t3749] = t3750;
        int t3752 = t2758 + t3725;
        int t3753 = t3752 + 1024;
        float t3754 = t3730 - t3741;
        memory[176308020 + t3753] = t3754;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3757 = 0; _pr3757 < 512; _pr3757++) {
        float t3758 = (float)_pr3757;
        float t3759 = (t3758 * 0.00390625);
        float t3760 = metal::floor(t3759);
        float t3761 = t3760 * 256.0;
        float t3762 = t3758 - t3761;
        float t3763 = t3760 * 512.0;
        float t3764 = t3763 + t3762;
        float t3765 = t3764 + 256.0;
        float t3766 = 6.283185 * t3762;
        float t3767 = (t3766 * 0.001953125);
        float t3768 = metal::cos(t3767);
        float t3769 = metal::sin(t3767);
        int t3770 = (int)t3764;
        int t3771 = (int)t3765;
        int t3772 = t2758 + t3770;
        float t3773 = memory[176308020 + t3772];
        int t3774 = t2758 + t3770;
        int t3775 = t3774 + 1024;
        float t3776 = memory[176308020 + t3775];
        int t3777 = t2758 + t3771;
        float t3778 = memory[176308020 + t3777];
        int t3779 = t2758 + t3771;
        int t3780 = t3779 + 1024;
        float t3781 = memory[176308020 + t3780];
        float t3782 = t3768 * t3778;
        float t3783 = t3769 * t3781;
        float t3784 = t3782 - t3783;
        float t3785 = t3768 * t3781;
        float t3786 = t3769 * t3778;
        float t3787 = t3785 + t3786;
        int t3788 = t2758 + t3770;
        float t3789 = t3773 + t3784;
        memory[176308020 + t3788] = t3789;
        int t3791 = t2758 + t3770;
        int t3792 = t3791 + 1024;
        float t3793 = t3776 + t3787;
        memory[176308020 + t3792] = t3793;
        int t3795 = t2758 + t3771;
        float t3796 = t3773 - t3784;
        memory[176308020 + t3795] = t3796;
        int t3798 = t2758 + t3771;
        int t3799 = t3798 + 1024;
        float t3800 = t3776 - t3787;
        memory[176308020 + t3799] = t3800;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3803 = 0; _pr3803 < 512; _pr3803++) {
        float t3804 = (float)_pr3803;
        float t3805 = (t3804 * 0.001953125);
        float t3806 = metal::floor(t3805);
        float t3807 = t3806 * 512.0;
        float t3808 = t3804 - t3807;
        float t3809 = t3806 * 1024.0;
        float t3810 = t3809 + t3808;
        float t3811 = t3810 + 512.0;
        float t3812 = 6.283185 * t3808;
        float t3813 = (t3812 * 0.0009765625);
        float t3814 = metal::cos(t3813);
        float t3815 = metal::sin(t3813);
        int t3816 = (int)t3810;
        int t3817 = (int)t3811;
        int t3818 = t2758 + t3816;
        float t3819 = memory[176308020 + t3818];
        int t3820 = t2758 + t3816;
        int t3821 = t3820 + 1024;
        float t3822 = memory[176308020 + t3821];
        int t3823 = t2758 + t3817;
        float t3824 = memory[176308020 + t3823];
        int t3825 = t2758 + t3817;
        int t3826 = t3825 + 1024;
        float t3827 = memory[176308020 + t3826];
        float t3828 = t3814 * t3824;
        float t3829 = t3815 * t3827;
        float t3830 = t3828 - t3829;
        float t3831 = t3814 * t3827;
        float t3832 = t3815 * t3824;
        float t3833 = t3831 + t3832;
        int t3834 = t2758 + t3816;
        float t3835 = t3819 + t3830;
        memory[176308020 + t3834] = t3835;
        int t3837 = t2758 + t3816;
        int t3838 = t3837 + 1024;
        float t3839 = t3822 + t3833;
        memory[176308020 + t3838] = t3839;
        int t3841 = t2758 + t3817;
        float t3842 = t3819 - t3830;
        memory[176308020 + t3841] = t3842;
        int t3844 = t2758 + t3817;
        int t3845 = t3844 + 1024;
        float t3846 = t3822 - t3833;
        memory[176308020 + t3845] = t3846;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3849 = 0; _pr3849 < 1024; _pr3849++) {
        int t3850 = t2758 + _pr3849;
        float t3851 = memory[176308020 + t3850];
        float t3852 = t3851 * 1.9036306e-06;
        float t3853 = memory[33012 + (int)_pr3849];
        int t3854 = t2759 + _pr3849;
        float t3855 = t3852 * t3853;
        memory[83984180 + t3854] = t3855;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t3859 = t[14*frameCount + id] > 0.0;
    if (t3859) {
      for (uint _pr3861 = 0; _pr3861 < 1024; _pr3861++) {
        int t3862 = t2759 + _pr3861;
        memory[50429748 + t3862] = 0.0;
        int t3864 = t2759 + _pr3861;
        memory[83984180 + t3864] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3885), value: global(3885)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1521) - handled in variable access */
    int t3868 = id;
    float t3869 = 0.0;
    for (uint t3870 = 0; t3870 < 1024; t3870++) {
      float t3871 = (float)t3870;
      float t3872 = (float)t3868;
      float t3873 = t3872 + t3871;
      int t3874 = 1023 - t3870;
      float t3875 = frameCount - 1.0;
      float t3876 = metal::min(t3873, t3875);
      int t3877 = (int)t3876;
      int t3878 = t3877 * 1024;
      int t3879 = t3878 + t3874;
      float t3880 = memory[50429748 + t3879];
      float t3881 = t3873 < frameCount;
      float t3882 = metal::select(0.0, t3880, t3881 > 0.0);
      float t3883 = t3869 + t3882;
      t3869 = t3883;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[17*frameCount + id] = (t3869 * 0.0013797212);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(3903), value: global(3903)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(1521) - handled in variable access */
    int t3886 = id;
    float t3887 = 0.0;
    for (uint t3888 = 0; t3888 < 1024; t3888++) {
      float t3889 = (float)t3888;
      float t3890 = (float)t3886;
      float t3891 = t3890 + t3889;
      int t3892 = 1023 - t3888;
      float t3893 = frameCount - 1.0;
      float t3894 = metal::min(t3891, t3893);
      int t3895 = (int)t3894;
      int t3896 = t3895 * 1024;
      int t3897 = t3896 + t3892;
      float t3898 = memory[83984180 + t3897];
      float t3899 = t3891 < frameCount;
      float t3900 = metal::select(0.0, t3898, t3899 > 0.0);
      float t3901 = t3887 + t3900;
      t3887 = t3901;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[18*frameCount + id] = (t3887 * 0.0013797212);
  }
  #pragma clang diagnostic pop
}



// KERNEL 23
// Kind: simd
// ThreadCountScale Optional(61)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_23(
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5551 = frameCount * 61.0;
  if (id >= 0 && id < (uint)(t5551)) {
    /* loadGlobal(481) - handled in variable access */
    int t3904 = id;
    int t3905 = t3904 / 61;
    uint _frameIndex = (uint)(t3905);
    int t3906 = t3905 * 61;
    int t3907 = t3904 - t3906;
    float t3908 = (t[12*frameCount + _frameIndex] * 3.7252903e-09);
    float t3909 = -0.5 * t3908;
  }
  #pragma clang diagnostic pop
}



// KERNEL 24
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_24(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(460) - handled in variable access */
    /* loadGlobal(459) - handled in variable access */
    /* loadGlobal(406) - handled in variable access */
    int t3910 = id;
    int t3911 = t3910 * 1024;
    int t3912 = t3910 * 257;
    int t3913 = t3910 * 1024;
    float t3914 = t[11*frameCount + id] == 0.0;
    if (t3914) {
      for (uint _pr3916 = 0; _pr3916 < 257; _pr3916++) {
        int t3917 = t3912 + _pr3916;
        float t3918 = memory[37797684 + t3917];
        int t3919 = t3912 + _pr3916;
        float t3920 = memory[42008372 + t3919];
        int t3921 = t3911 + _pr3916;
        float t3922 = memory[4243252 + t3921];
        int t3923 = t3911 + _pr3916;
        int t3924 = t3923 + 512;
        float t3925 = memory[4243252 + t3924];
        int t3926 = t3911 + _pr3916;
        float t3927 = memory[21020468 + t3926];
        int t3928 = t3911 + _pr3916;
        int t3929 = t3928 + 512;
        float t3930 = memory[21020468 + t3929];
        float t3931 = t3918 - t3920;
        float t3932 = 2.0 * t3931;
        float t3933 = t3932 * 3.0517578e-05;
        float t3934 = t3918 - t3920;
        float t3935 = -2.0 * t3934;
        float t3936 = t3935 * 3.0517578e-05;
        float t3937 = metal::max(t3918, 1e-08);
        float t3938 = metal::max(t3920, 1e-08);
        float t3939 = t3933 * t3922;
        float t3940 = t3939 / t3937;
        float t3941 = t3933 * t3925;
        float t3942 = t3941 / t3937;
        float t3943 = t3936 * t3927;
        float t3944 = t3943 / t3938;
        float t3945 = t3936 * t3930;
        float t3946 = t3945 / t3938;
        int t3947 = t3913 + _pr3916;
        memory[50429748 + t3947] = t3940;
        int t3949 = t3913 + _pr3916;
        int t3950 = t3949 + 512;
        memory[50429748 + t3950] = t3942;
        int t3952 = t3913 + _pr3916;
        memory[83984180 + t3952] = t3944;
        int t3954 = t3913 + _pr3916;
        int t3955 = t3954 + 512;
        memory[83984180 + t3955] = t3946;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr3958 = 0; _pr3958 < 255; _pr3958++) {
        int t3959 = _pr3958 + 257;
        int t3960 = t3913 + t3959;
        memory[50429748 + t3960] = 0.0;
        int t3962 = t3913 + t3959;
        int t3963 = t3962 + 512;
        memory[50429748 + t3963] = 0.0;
        int t3965 = t3913 + t3959;
        memory[83984180 + t3965] = 0.0;
        int t3967 = t3913 + t3959;
        int t3968 = t3967 + 512;
        memory[83984180 + t3968] = 0.0;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
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
    /* loadGlobal(460) - handled in variable access */
    int t3972 = id;
    int t3973 = t3972 * 1024;
    int t3974 = t3972 * 512;
    float t3975 = t[11*frameCount + id] == 0.0;
    if (t3975) {
      for (uint t3977 = 0; t3977 < 512; t3977++) {
        float t3978 = (float)t3977;
        float t3979 = (t3978 - metal::floor(t3978 / 2.0) * 2.0);
        float t3980 = t3979;
        float t3981 = (t3978 * 0.5);
        float t3982 = metal::floor(t3981);
        float t3983 = t3980 * 2.0;
        float t3984 = (t3982 - metal::floor(t3982 / 2.0) * 2.0);
        float t3985 = t3983 + t3984;
        float t3986 = (t3982 * 0.5);
        float t3987 = metal::floor(t3986);
        float t3988 = t3985 * 2.0;
        float t3989 = (t3987 - metal::floor(t3987 / 2.0) * 2.0);
        float t3990 = t3988 + t3989;
        float t3991 = (t3987 * 0.5);
        float t3992 = metal::floor(t3991);
        float t3993 = t3990 * 2.0;
        float t3994 = (t3992 - metal::floor(t3992 / 2.0) * 2.0);
        float t3995 = t3993 + t3994;
        float t3996 = (t3992 * 0.5);
        float t3997 = metal::floor(t3996);
        float t3998 = t3995 * 2.0;
        float t3999 = (t3997 - metal::floor(t3997 / 2.0) * 2.0);
        float t4000 = t3998 + t3999;
        float t4001 = (t3997 * 0.5);
        float t4002 = metal::floor(t4001);
        float t4003 = t4000 * 2.0;
        float t4004 = (t4002 - metal::floor(t4002 / 2.0) * 2.0);
        float t4005 = t4003 + t4004;
        float t4006 = (t4002 * 0.5);
        float t4007 = metal::floor(t4006);
        float t4008 = t4005 * 2.0;
        float t4009 = (t4007 - metal::floor(t4007 / 2.0) * 2.0);
        float t4010 = t4008 + t4009;
        float t4011 = (t4007 * 0.5);
        float t4012 = metal::floor(t4011);
        float t4013 = t4010 * 2.0;
        float t4014 = (t4012 - metal::floor(t4012 / 2.0) * 2.0);
        float t4015 = t4013 + t4014;
        float t4016 = (t4012 * 0.5);
        float t4017 = metal::floor(t4016);
        float t4018 = t4015 * 2.0;
        float t4019 = (t4017 - metal::floor(t4017 / 2.0) * 2.0);
        float t4020 = t4018 + t4019;
        float t4021 = (t4017 * 0.5);
        float t4022 = metal::floor(t4021);
        float t4023 = (float)t3977;
        float t4024 = t4023 < t4020;
        int t4025 = (int)t4020;
        int t4026 = t3973 + t3977;
        float t4027 = memory[50429748 + t4026];
        int t4028 = t3973 + t3977;
        int t4029 = t4028 + 512;
        float t4030 = memory[50429748 + t4029];
        int t4031 = t3973 + t4025;
        float t4032 = memory[50429748 + t4031];
        int t4033 = t3973 + t4025;
        int t4034 = t4033 + 512;
        float t4035 = memory[50429748 + t4034];
        float t4036 = metal::select(t4027, t4032, t4024 > 0.0);
        float t4037 = metal::select(t4030, t4035, t4024 > 0.0);
        float t4038 = metal::select(t4032, t4027, t4024 > 0.0);
        float t4039 = metal::select(t4035, t4030, t4024 > 0.0);
        int t4040 = t3973 + t3977;
        memory[50429748 + t4040] = t4036;
        int t4042 = t3973 + t3977;
        int t4043 = t4042 + 512;
        memory[50429748 + t4043] = t4037;
        int t4045 = t3973 + t4025;
        memory[50429748 + t4045] = t4038;
        int t4047 = t3973 + t4025;
        int t4048 = t4047 + 512;
        memory[50429748 + t4048] = t4039;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4051 = 0; _pr4051 < 256; _pr4051++) {
        float t4052 = (float)_pr4051;
        float t4053 = t4052;
        float t4054 = metal::floor(t4053);
        float t4055 = t4054;
        float t4056 = t4052 - t4055;
        float t4057 = t4054 * 2.0;
        float t4058 = t4057 + t4056;
        float t4059 = t4058 + 1.0;
        float t4060 = 6.283185 * t4056;
        float t4061 = (t4060 * 0.5);
        float t4062 = metal::cos(t4061);
        float t4063 = metal::sin(t4061);
        int t4064 = (int)t4058;
        int t4065 = (int)t4059;
        int t4066 = t3973 + t4064;
        float t4067 = memory[50429748 + t4066];
        int t4068 = t3973 + t4064;
        int t4069 = t4068 + 512;
        float t4070 = memory[50429748 + t4069];
        int t4071 = t3973 + t4065;
        float t4072 = memory[50429748 + t4071];
        int t4073 = t3973 + t4065;
        int t4074 = t4073 + 512;
        float t4075 = memory[50429748 + t4074];
        float t4076 = t4062 * t4072;
        float t4077 = t4063 * t4075;
        float t4078 = t4076 - t4077;
        float t4079 = t4062 * t4075;
        float t4080 = t4063 * t4072;
        float t4081 = t4079 + t4080;
        int t4082 = t3973 + t4064;
        float t4083 = t4067 + t4078;
        memory[50429748 + t4082] = t4083;
        int t4085 = t3973 + t4064;
        int t4086 = t4085 + 512;
        float t4087 = t4070 + t4081;
        memory[50429748 + t4086] = t4087;
        int t4089 = t3973 + t4065;
        float t4090 = t4067 - t4078;
        memory[50429748 + t4089] = t4090;
        int t4092 = t3973 + t4065;
        int t4093 = t4092 + 512;
        float t4094 = t4070 - t4081;
        memory[50429748 + t4093] = t4094;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4097 = 0; _pr4097 < 256; _pr4097++) {
        float t4098 = (float)_pr4097;
        float t4099 = (t4098 * 0.5);
        float t4100 = metal::floor(t4099);
        float t4101 = t4100 * 2.0;
        float t4102 = t4098 - t4101;
        float t4103 = t4100 * 4.0;
        float t4104 = t4103 + t4102;
        float t4105 = t4104 + 2.0;
        float t4106 = 6.283185 * t4102;
        float t4107 = (t4106 * 0.25);
        float t4108 = metal::cos(t4107);
        float t4109 = metal::sin(t4107);
        int t4110 = (int)t4104;
        int t4111 = (int)t4105;
        int t4112 = t3973 + t4110;
        float t4113 = memory[50429748 + t4112];
        int t4114 = t3973 + t4110;
        int t4115 = t4114 + 512;
        float t4116 = memory[50429748 + t4115];
        int t4117 = t3973 + t4111;
        float t4118 = memory[50429748 + t4117];
        int t4119 = t3973 + t4111;
        int t4120 = t4119 + 512;
        float t4121 = memory[50429748 + t4120];
        float t4122 = t4108 * t4118;
        float t4123 = t4109 * t4121;
        float t4124 = t4122 - t4123;
        float t4125 = t4108 * t4121;
        float t4126 = t4109 * t4118;
        float t4127 = t4125 + t4126;
        int t4128 = t3973 + t4110;
        float t4129 = t4113 + t4124;
        memory[50429748 + t4128] = t4129;
        int t4131 = t3973 + t4110;
        int t4132 = t4131 + 512;
        float t4133 = t4116 + t4127;
        memory[50429748 + t4132] = t4133;
        int t4135 = t3973 + t4111;
        float t4136 = t4113 - t4124;
        memory[50429748 + t4135] = t4136;
        int t4138 = t3973 + t4111;
        int t4139 = t4138 + 512;
        float t4140 = t4116 - t4127;
        memory[50429748 + t4139] = t4140;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4143 = 0; _pr4143 < 256; _pr4143++) {
        float t4144 = (float)_pr4143;
        float t4145 = (t4144 * 0.25);
        float t4146 = metal::floor(t4145);
        float t4147 = t4146 * 4.0;
        float t4148 = t4144 - t4147;
        float t4149 = t4146 * 8.0;
        float t4150 = t4149 + t4148;
        float t4151 = t4150 + 4.0;
        float t4152 = 6.283185 * t4148;
        float t4153 = (t4152 * 0.125);
        float t4154 = metal::cos(t4153);
        float t4155 = metal::sin(t4153);
        int t4156 = (int)t4150;
        int t4157 = (int)t4151;
        int t4158 = t3973 + t4156;
        float t4159 = memory[50429748 + t4158];
        int t4160 = t3973 + t4156;
        int t4161 = t4160 + 512;
        float t4162 = memory[50429748 + t4161];
        int t4163 = t3973 + t4157;
        float t4164 = memory[50429748 + t4163];
        int t4165 = t3973 + t4157;
        int t4166 = t4165 + 512;
        float t4167 = memory[50429748 + t4166];
        float t4168 = t4154 * t4164;
        float t4169 = t4155 * t4167;
        float t4170 = t4168 - t4169;
        float t4171 = t4154 * t4167;
        float t4172 = t4155 * t4164;
        float t4173 = t4171 + t4172;
        int t4174 = t3973 + t4156;
        float t4175 = t4159 + t4170;
        memory[50429748 + t4174] = t4175;
        int t4177 = t3973 + t4156;
        int t4178 = t4177 + 512;
        float t4179 = t4162 + t4173;
        memory[50429748 + t4178] = t4179;
        int t4181 = t3973 + t4157;
        float t4182 = t4159 - t4170;
        memory[50429748 + t4181] = t4182;
        int t4184 = t3973 + t4157;
        int t4185 = t4184 + 512;
        float t4186 = t4162 - t4173;
        memory[50429748 + t4185] = t4186;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4189 = 0; _pr4189 < 256; _pr4189++) {
        float t4190 = (float)_pr4189;
        float t4191 = (t4190 * 0.125);
        float t4192 = metal::floor(t4191);
        float t4193 = t4192 * 8.0;
        float t4194 = t4190 - t4193;
        float t4195 = t4192 * 16.0;
        float t4196 = t4195 + t4194;
        float t4197 = t4196 + 8.0;
        float t4198 = 6.283185 * t4194;
        float t4199 = (t4198 * 0.0625);
        float t4200 = metal::cos(t4199);
        float t4201 = metal::sin(t4199);
        int t4202 = (int)t4196;
        int t4203 = (int)t4197;
        int t4204 = t3973 + t4202;
        float t4205 = memory[50429748 + t4204];
        int t4206 = t3973 + t4202;
        int t4207 = t4206 + 512;
        float t4208 = memory[50429748 + t4207];
        int t4209 = t3973 + t4203;
        float t4210 = memory[50429748 + t4209];
        int t4211 = t3973 + t4203;
        int t4212 = t4211 + 512;
        float t4213 = memory[50429748 + t4212];
        float t4214 = t4200 * t4210;
        float t4215 = t4201 * t4213;
        float t4216 = t4214 - t4215;
        float t4217 = t4200 * t4213;
        float t4218 = t4201 * t4210;
        float t4219 = t4217 + t4218;
        int t4220 = t3973 + t4202;
        float t4221 = t4205 + t4216;
        memory[50429748 + t4220] = t4221;
        int t4223 = t3973 + t4202;
        int t4224 = t4223 + 512;
        float t4225 = t4208 + t4219;
        memory[50429748 + t4224] = t4225;
        int t4227 = t3973 + t4203;
        float t4228 = t4205 - t4216;
        memory[50429748 + t4227] = t4228;
        int t4230 = t3973 + t4203;
        int t4231 = t4230 + 512;
        float t4232 = t4208 - t4219;
        memory[50429748 + t4231] = t4232;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4235 = 0; _pr4235 < 256; _pr4235++) {
        float t4236 = (float)_pr4235;
        float t4237 = (t4236 * 0.0625);
        float t4238 = metal::floor(t4237);
        float t4239 = t4238 * 16.0;
        float t4240 = t4236 - t4239;
        float t4241 = t4238 * 32.0;
        float t4242 = t4241 + t4240;
        float t4243 = t4242 + 16.0;
        float t4244 = 6.283185 * t4240;
        float t4245 = (t4244 * 0.03125);
        float t4246 = metal::cos(t4245);
        float t4247 = metal::sin(t4245);
        int t4248 = (int)t4242;
        int t4249 = (int)t4243;
        int t4250 = t3973 + t4248;
        float t4251 = memory[50429748 + t4250];
        int t4252 = t3973 + t4248;
        int t4253 = t4252 + 512;
        float t4254 = memory[50429748 + t4253];
        int t4255 = t3973 + t4249;
        float t4256 = memory[50429748 + t4255];
        int t4257 = t3973 + t4249;
        int t4258 = t4257 + 512;
        float t4259 = memory[50429748 + t4258];
        float t4260 = t4246 * t4256;
        float t4261 = t4247 * t4259;
        float t4262 = t4260 - t4261;
        float t4263 = t4246 * t4259;
        float t4264 = t4247 * t4256;
        float t4265 = t4263 + t4264;
        int t4266 = t3973 + t4248;
        float t4267 = t4251 + t4262;
        memory[50429748 + t4266] = t4267;
        int t4269 = t3973 + t4248;
        int t4270 = t4269 + 512;
        float t4271 = t4254 + t4265;
        memory[50429748 + t4270] = t4271;
        int t4273 = t3973 + t4249;
        float t4274 = t4251 - t4262;
        memory[50429748 + t4273] = t4274;
        int t4276 = t3973 + t4249;
        int t4277 = t4276 + 512;
        float t4278 = t4254 - t4265;
        memory[50429748 + t4277] = t4278;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4281 = 0; _pr4281 < 256; _pr4281++) {
        float t4282 = (float)_pr4281;
        float t4283 = (t4282 * 0.03125);
        float t4284 = metal::floor(t4283);
        float t4285 = t4284 * 32.0;
        float t4286 = t4282 - t4285;
        float t4287 = t4284 * 64.0;
        float t4288 = t4287 + t4286;
        float t4289 = t4288 + 32.0;
        float t4290 = 6.283185 * t4286;
        float t4291 = (t4290 * 0.015625);
        float t4292 = metal::cos(t4291);
        float t4293 = metal::sin(t4291);
        int t4294 = (int)t4288;
        int t4295 = (int)t4289;
        int t4296 = t3973 + t4294;
        float t4297 = memory[50429748 + t4296];
        int t4298 = t3973 + t4294;
        int t4299 = t4298 + 512;
        float t4300 = memory[50429748 + t4299];
        int t4301 = t3973 + t4295;
        float t4302 = memory[50429748 + t4301];
        int t4303 = t3973 + t4295;
        int t4304 = t4303 + 512;
        float t4305 = memory[50429748 + t4304];
        float t4306 = t4292 * t4302;
        float t4307 = t4293 * t4305;
        float t4308 = t4306 - t4307;
        float t4309 = t4292 * t4305;
        float t4310 = t4293 * t4302;
        float t4311 = t4309 + t4310;
        int t4312 = t3973 + t4294;
        float t4313 = t4297 + t4308;
        memory[50429748 + t4312] = t4313;
        int t4315 = t3973 + t4294;
        int t4316 = t4315 + 512;
        float t4317 = t4300 + t4311;
        memory[50429748 + t4316] = t4317;
        int t4319 = t3973 + t4295;
        float t4320 = t4297 - t4308;
        memory[50429748 + t4319] = t4320;
        int t4322 = t3973 + t4295;
        int t4323 = t4322 + 512;
        float t4324 = t4300 - t4311;
        memory[50429748 + t4323] = t4324;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4327 = 0; _pr4327 < 256; _pr4327++) {
        float t4328 = (float)_pr4327;
        float t4329 = (t4328 * 0.015625);
        float t4330 = metal::floor(t4329);
        float t4331 = t4330 * 64.0;
        float t4332 = t4328 - t4331;
        float t4333 = t4330 * 128.0;
        float t4334 = t4333 + t4332;
        float t4335 = t4334 + 64.0;
        float t4336 = 6.283185 * t4332;
        float t4337 = (t4336 * 0.0078125);
        float t4338 = metal::cos(t4337);
        float t4339 = metal::sin(t4337);
        int t4340 = (int)t4334;
        int t4341 = (int)t4335;
        int t4342 = t3973 + t4340;
        float t4343 = memory[50429748 + t4342];
        int t4344 = t3973 + t4340;
        int t4345 = t4344 + 512;
        float t4346 = memory[50429748 + t4345];
        int t4347 = t3973 + t4341;
        float t4348 = memory[50429748 + t4347];
        int t4349 = t3973 + t4341;
        int t4350 = t4349 + 512;
        float t4351 = memory[50429748 + t4350];
        float t4352 = t4338 * t4348;
        float t4353 = t4339 * t4351;
        float t4354 = t4352 - t4353;
        float t4355 = t4338 * t4351;
        float t4356 = t4339 * t4348;
        float t4357 = t4355 + t4356;
        int t4358 = t3973 + t4340;
        float t4359 = t4343 + t4354;
        memory[50429748 + t4358] = t4359;
        int t4361 = t3973 + t4340;
        int t4362 = t4361 + 512;
        float t4363 = t4346 + t4357;
        memory[50429748 + t4362] = t4363;
        int t4365 = t3973 + t4341;
        float t4366 = t4343 - t4354;
        memory[50429748 + t4365] = t4366;
        int t4368 = t3973 + t4341;
        int t4369 = t4368 + 512;
        float t4370 = t4346 - t4357;
        memory[50429748 + t4369] = t4370;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4373 = 0; _pr4373 < 256; _pr4373++) {
        float t4374 = (float)_pr4373;
        float t4375 = (t4374 * 0.0078125);
        float t4376 = metal::floor(t4375);
        float t4377 = t4376 * 128.0;
        float t4378 = t4374 - t4377;
        float t4379 = t4376 * 256.0;
        float t4380 = t4379 + t4378;
        float t4381 = t4380 + 128.0;
        float t4382 = 6.283185 * t4378;
        float t4383 = (t4382 * 0.00390625);
        float t4384 = metal::cos(t4383);
        float t4385 = metal::sin(t4383);
        int t4386 = (int)t4380;
        int t4387 = (int)t4381;
        int t4388 = t3973 + t4386;
        float t4389 = memory[50429748 + t4388];
        int t4390 = t3973 + t4386;
        int t4391 = t4390 + 512;
        float t4392 = memory[50429748 + t4391];
        int t4393 = t3973 + t4387;
        float t4394 = memory[50429748 + t4393];
        int t4395 = t3973 + t4387;
        int t4396 = t4395 + 512;
        float t4397 = memory[50429748 + t4396];
        float t4398 = t4384 * t4394;
        float t4399 = t4385 * t4397;
        float t4400 = t4398 - t4399;
        float t4401 = t4384 * t4397;
        float t4402 = t4385 * t4394;
        float t4403 = t4401 + t4402;
        int t4404 = t3973 + t4386;
        float t4405 = t4389 + t4400;
        memory[50429748 + t4404] = t4405;
        int t4407 = t3973 + t4386;
        int t4408 = t4407 + 512;
        float t4409 = t4392 + t4403;
        memory[50429748 + t4408] = t4409;
        int t4411 = t3973 + t4387;
        float t4412 = t4389 - t4400;
        memory[50429748 + t4411] = t4412;
        int t4414 = t3973 + t4387;
        int t4415 = t4414 + 512;
        float t4416 = t4392 - t4403;
        memory[50429748 + t4415] = t4416;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4419 = 0; _pr4419 < 256; _pr4419++) {
        float t4420 = (float)_pr4419;
        float t4421 = (t4420 * 0.00390625);
        float t4422 = metal::floor(t4421);
        float t4423 = t4422 * 256.0;
        float t4424 = t4420 - t4423;
        float t4425 = t4422 * 512.0;
        float t4426 = t4425 + t4424;
        float t4427 = t4426 + 256.0;
        float t4428 = 6.283185 * t4424;
        float t4429 = (t4428 * 0.001953125);
        float t4430 = metal::cos(t4429);
        float t4431 = metal::sin(t4429);
        int t4432 = (int)t4426;
        int t4433 = (int)t4427;
        int t4434 = t3973 + t4432;
        float t4435 = memory[50429748 + t4434];
        int t4436 = t3973 + t4432;
        int t4437 = t4436 + 512;
        float t4438 = memory[50429748 + t4437];
        int t4439 = t3973 + t4433;
        float t4440 = memory[50429748 + t4439];
        int t4441 = t3973 + t4433;
        int t4442 = t4441 + 512;
        float t4443 = memory[50429748 + t4442];
        float t4444 = t4430 * t4440;
        float t4445 = t4431 * t4443;
        float t4446 = t4444 - t4445;
        float t4447 = t4430 * t4443;
        float t4448 = t4431 * t4440;
        float t4449 = t4447 + t4448;
        int t4450 = t3973 + t4432;
        float t4451 = t4435 + t4446;
        memory[50429748 + t4450] = t4451;
        int t4453 = t3973 + t4432;
        int t4454 = t4453 + 512;
        float t4455 = t4438 + t4449;
        memory[50429748 + t4454] = t4455;
        int t4457 = t3973 + t4433;
        float t4458 = t4435 - t4446;
        memory[50429748 + t4457] = t4458;
        int t4460 = t3973 + t4433;
        int t4461 = t4460 + 512;
        float t4462 = t4438 - t4449;
        memory[50429748 + t4461] = t4462;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4465 = 0; _pr4465 < 512; _pr4465++) {
        int t4466 = t3973 + _pr4465;
        float t4467 = memory[50429748 + t4466];
        float t4468 = t4467 * 7.599708e-06;
        float t4469 = memory[21300 + (int)_pr4465];
        int t4470 = t3974 + _pr4465;
        float t4471 = t4468 * t4469;
        memory[117538612 + t4470] = t4471;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint t4474 = 0; t4474 < 512; t4474++) {
        float t4475 = (float)t4474;
        float t4476 = (t4475 - metal::floor(t4475 / 2.0) * 2.0);
        float t4477 = t4476;
        float t4478 = (t4475 * 0.5);
        float t4479 = metal::floor(t4478);
        float t4480 = t4477 * 2.0;
        float t4481 = (t4479 - metal::floor(t4479 / 2.0) * 2.0);
        float t4482 = t4480 + t4481;
        float t4483 = (t4479 * 0.5);
        float t4484 = metal::floor(t4483);
        float t4485 = t4482 * 2.0;
        float t4486 = (t4484 - metal::floor(t4484 / 2.0) * 2.0);
        float t4487 = t4485 + t4486;
        float t4488 = (t4484 * 0.5);
        float t4489 = metal::floor(t4488);
        float t4490 = t4487 * 2.0;
        float t4491 = (t4489 - metal::floor(t4489 / 2.0) * 2.0);
        float t4492 = t4490 + t4491;
        float t4493 = (t4489 * 0.5);
        float t4494 = metal::floor(t4493);
        float t4495 = t4492 * 2.0;
        float t4496 = (t4494 - metal::floor(t4494 / 2.0) * 2.0);
        float t4497 = t4495 + t4496;
        float t4498 = (t4494 * 0.5);
        float t4499 = metal::floor(t4498);
        float t4500 = t4497 * 2.0;
        float t4501 = (t4499 - metal::floor(t4499 / 2.0) * 2.0);
        float t4502 = t4500 + t4501;
        float t4503 = (t4499 * 0.5);
        float t4504 = metal::floor(t4503);
        float t4505 = t4502 * 2.0;
        float t4506 = (t4504 - metal::floor(t4504 / 2.0) * 2.0);
        float t4507 = t4505 + t4506;
        float t4508 = (t4504 * 0.5);
        float t4509 = metal::floor(t4508);
        float t4510 = t4507 * 2.0;
        float t4511 = (t4509 - metal::floor(t4509 / 2.0) * 2.0);
        float t4512 = t4510 + t4511;
        float t4513 = (t4509 * 0.5);
        float t4514 = metal::floor(t4513);
        float t4515 = t4512 * 2.0;
        float t4516 = (t4514 - metal::floor(t4514 / 2.0) * 2.0);
        float t4517 = t4515 + t4516;
        float t4518 = (t4514 * 0.5);
        float t4519 = metal::floor(t4518);
        float t4520 = (float)t4474;
        float t4521 = t4520 < t4517;
        int t4522 = (int)t4517;
        int t4523 = t3973 + t4474;
        float t4524 = memory[83984180 + t4523];
        int t4525 = t3973 + t4474;
        int t4526 = t4525 + 512;
        float t4527 = memory[83984180 + t4526];
        int t4528 = t3973 + t4522;
        float t4529 = memory[83984180 + t4528];
        int t4530 = t3973 + t4522;
        int t4531 = t4530 + 512;
        float t4532 = memory[83984180 + t4531];
        float t4533 = metal::select(t4524, t4529, t4521 > 0.0);
        float t4534 = metal::select(t4527, t4532, t4521 > 0.0);
        float t4535 = metal::select(t4529, t4524, t4521 > 0.0);
        float t4536 = metal::select(t4532, t4527, t4521 > 0.0);
        int t4537 = t3973 + t4474;
        memory[83984180 + t4537] = t4533;
        int t4539 = t3973 + t4474;
        int t4540 = t4539 + 512;
        memory[83984180 + t4540] = t4534;
        int t4542 = t3973 + t4522;
        memory[83984180 + t4542] = t4535;
        int t4544 = t3973 + t4522;
        int t4545 = t4544 + 512;
        memory[83984180 + t4545] = t4536;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4548 = 0; _pr4548 < 256; _pr4548++) {
        float t4549 = (float)_pr4548;
        float t4550 = t4549;
        float t4551 = metal::floor(t4550);
        float t4552 = t4551;
        float t4553 = t4549 - t4552;
        float t4554 = t4551 * 2.0;
        float t4555 = t4554 + t4553;
        float t4556 = t4555 + 1.0;
        float t4557 = 6.283185 * t4553;
        float t4558 = (t4557 * 0.5);
        float t4559 = metal::cos(t4558);
        float t4560 = metal::sin(t4558);
        int t4561 = (int)t4555;
        int t4562 = (int)t4556;
        int t4563 = t3973 + t4561;
        float t4564 = memory[83984180 + t4563];
        int t4565 = t3973 + t4561;
        int t4566 = t4565 + 512;
        float t4567 = memory[83984180 + t4566];
        int t4568 = t3973 + t4562;
        float t4569 = memory[83984180 + t4568];
        int t4570 = t3973 + t4562;
        int t4571 = t4570 + 512;
        float t4572 = memory[83984180 + t4571];
        float t4573 = t4559 * t4569;
        float t4574 = t4560 * t4572;
        float t4575 = t4573 - t4574;
        float t4576 = t4559 * t4572;
        float t4577 = t4560 * t4569;
        float t4578 = t4576 + t4577;
        int t4579 = t3973 + t4561;
        float t4580 = t4564 + t4575;
        memory[83984180 + t4579] = t4580;
        int t4582 = t3973 + t4561;
        int t4583 = t4582 + 512;
        float t4584 = t4567 + t4578;
        memory[83984180 + t4583] = t4584;
        int t4586 = t3973 + t4562;
        float t4587 = t4564 - t4575;
        memory[83984180 + t4586] = t4587;
        int t4589 = t3973 + t4562;
        int t4590 = t4589 + 512;
        float t4591 = t4567 - t4578;
        memory[83984180 + t4590] = t4591;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4594 = 0; _pr4594 < 256; _pr4594++) {
        float t4595 = (float)_pr4594;
        float t4596 = (t4595 * 0.5);
        float t4597 = metal::floor(t4596);
        float t4598 = t4597 * 2.0;
        float t4599 = t4595 - t4598;
        float t4600 = t4597 * 4.0;
        float t4601 = t4600 + t4599;
        float t4602 = t4601 + 2.0;
        float t4603 = 6.283185 * t4599;
        float t4604 = (t4603 * 0.25);
        float t4605 = metal::cos(t4604);
        float t4606 = metal::sin(t4604);
        int t4607 = (int)t4601;
        int t4608 = (int)t4602;
        int t4609 = t3973 + t4607;
        float t4610 = memory[83984180 + t4609];
        int t4611 = t3973 + t4607;
        int t4612 = t4611 + 512;
        float t4613 = memory[83984180 + t4612];
        int t4614 = t3973 + t4608;
        float t4615 = memory[83984180 + t4614];
        int t4616 = t3973 + t4608;
        int t4617 = t4616 + 512;
        float t4618 = memory[83984180 + t4617];
        float t4619 = t4605 * t4615;
        float t4620 = t4606 * t4618;
        float t4621 = t4619 - t4620;
        float t4622 = t4605 * t4618;
        float t4623 = t4606 * t4615;
        float t4624 = t4622 + t4623;
        int t4625 = t3973 + t4607;
        float t4626 = t4610 + t4621;
        memory[83984180 + t4625] = t4626;
        int t4628 = t3973 + t4607;
        int t4629 = t4628 + 512;
        float t4630 = t4613 + t4624;
        memory[83984180 + t4629] = t4630;
        int t4632 = t3973 + t4608;
        float t4633 = t4610 - t4621;
        memory[83984180 + t4632] = t4633;
        int t4635 = t3973 + t4608;
        int t4636 = t4635 + 512;
        float t4637 = t4613 - t4624;
        memory[83984180 + t4636] = t4637;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4640 = 0; _pr4640 < 256; _pr4640++) {
        float t4641 = (float)_pr4640;
        float t4642 = (t4641 * 0.25);
        float t4643 = metal::floor(t4642);
        float t4644 = t4643 * 4.0;
        float t4645 = t4641 - t4644;
        float t4646 = t4643 * 8.0;
        float t4647 = t4646 + t4645;
        float t4648 = t4647 + 4.0;
        float t4649 = 6.283185 * t4645;
        float t4650 = (t4649 * 0.125);
        float t4651 = metal::cos(t4650);
        float t4652 = metal::sin(t4650);
        int t4653 = (int)t4647;
        int t4654 = (int)t4648;
        int t4655 = t3973 + t4653;
        float t4656 = memory[83984180 + t4655];
        int t4657 = t3973 + t4653;
        int t4658 = t4657 + 512;
        float t4659 = memory[83984180 + t4658];
        int t4660 = t3973 + t4654;
        float t4661 = memory[83984180 + t4660];
        int t4662 = t3973 + t4654;
        int t4663 = t4662 + 512;
        float t4664 = memory[83984180 + t4663];
        float t4665 = t4651 * t4661;
        float t4666 = t4652 * t4664;
        float t4667 = t4665 - t4666;
        float t4668 = t4651 * t4664;
        float t4669 = t4652 * t4661;
        float t4670 = t4668 + t4669;
        int t4671 = t3973 + t4653;
        float t4672 = t4656 + t4667;
        memory[83984180 + t4671] = t4672;
        int t4674 = t3973 + t4653;
        int t4675 = t4674 + 512;
        float t4676 = t4659 + t4670;
        memory[83984180 + t4675] = t4676;
        int t4678 = t3973 + t4654;
        float t4679 = t4656 - t4667;
        memory[83984180 + t4678] = t4679;
        int t4681 = t3973 + t4654;
        int t4682 = t4681 + 512;
        float t4683 = t4659 - t4670;
        memory[83984180 + t4682] = t4683;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4686 = 0; _pr4686 < 256; _pr4686++) {
        float t4687 = (float)_pr4686;
        float t4688 = (t4687 * 0.125);
        float t4689 = metal::floor(t4688);
        float t4690 = t4689 * 8.0;
        float t4691 = t4687 - t4690;
        float t4692 = t4689 * 16.0;
        float t4693 = t4692 + t4691;
        float t4694 = t4693 + 8.0;
        float t4695 = 6.283185 * t4691;
        float t4696 = (t4695 * 0.0625);
        float t4697 = metal::cos(t4696);
        float t4698 = metal::sin(t4696);
        int t4699 = (int)t4693;
        int t4700 = (int)t4694;
        int t4701 = t3973 + t4699;
        float t4702 = memory[83984180 + t4701];
        int t4703 = t3973 + t4699;
        int t4704 = t4703 + 512;
        float t4705 = memory[83984180 + t4704];
        int t4706 = t3973 + t4700;
        float t4707 = memory[83984180 + t4706];
        int t4708 = t3973 + t4700;
        int t4709 = t4708 + 512;
        float t4710 = memory[83984180 + t4709];
        float t4711 = t4697 * t4707;
        float t4712 = t4698 * t4710;
        float t4713 = t4711 - t4712;
        float t4714 = t4697 * t4710;
        float t4715 = t4698 * t4707;
        float t4716 = t4714 + t4715;
        int t4717 = t3973 + t4699;
        float t4718 = t4702 + t4713;
        memory[83984180 + t4717] = t4718;
        int t4720 = t3973 + t4699;
        int t4721 = t4720 + 512;
        float t4722 = t4705 + t4716;
        memory[83984180 + t4721] = t4722;
        int t4724 = t3973 + t4700;
        float t4725 = t4702 - t4713;
        memory[83984180 + t4724] = t4725;
        int t4727 = t3973 + t4700;
        int t4728 = t4727 + 512;
        float t4729 = t4705 - t4716;
        memory[83984180 + t4728] = t4729;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4732 = 0; _pr4732 < 256; _pr4732++) {
        float t4733 = (float)_pr4732;
        float t4734 = (t4733 * 0.0625);
        float t4735 = metal::floor(t4734);
        float t4736 = t4735 * 16.0;
        float t4737 = t4733 - t4736;
        float t4738 = t4735 * 32.0;
        float t4739 = t4738 + t4737;
        float t4740 = t4739 + 16.0;
        float t4741 = 6.283185 * t4737;
        float t4742 = (t4741 * 0.03125);
        float t4743 = metal::cos(t4742);
        float t4744 = metal::sin(t4742);
        int t4745 = (int)t4739;
        int t4746 = (int)t4740;
        int t4747 = t3973 + t4745;
        float t4748 = memory[83984180 + t4747];
        int t4749 = t3973 + t4745;
        int t4750 = t4749 + 512;
        float t4751 = memory[83984180 + t4750];
        int t4752 = t3973 + t4746;
        float t4753 = memory[83984180 + t4752];
        int t4754 = t3973 + t4746;
        int t4755 = t4754 + 512;
        float t4756 = memory[83984180 + t4755];
        float t4757 = t4743 * t4753;
        float t4758 = t4744 * t4756;
        float t4759 = t4757 - t4758;
        float t4760 = t4743 * t4756;
        float t4761 = t4744 * t4753;
        float t4762 = t4760 + t4761;
        int t4763 = t3973 + t4745;
        float t4764 = t4748 + t4759;
        memory[83984180 + t4763] = t4764;
        int t4766 = t3973 + t4745;
        int t4767 = t4766 + 512;
        float t4768 = t4751 + t4762;
        memory[83984180 + t4767] = t4768;
        int t4770 = t3973 + t4746;
        float t4771 = t4748 - t4759;
        memory[83984180 + t4770] = t4771;
        int t4773 = t3973 + t4746;
        int t4774 = t4773 + 512;
        float t4775 = t4751 - t4762;
        memory[83984180 + t4774] = t4775;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4778 = 0; _pr4778 < 256; _pr4778++) {
        float t4779 = (float)_pr4778;
        float t4780 = (t4779 * 0.03125);
        float t4781 = metal::floor(t4780);
        float t4782 = t4781 * 32.0;
        float t4783 = t4779 - t4782;
        float t4784 = t4781 * 64.0;
        float t4785 = t4784 + t4783;
        float t4786 = t4785 + 32.0;
        float t4787 = 6.283185 * t4783;
        float t4788 = (t4787 * 0.015625);
        float t4789 = metal::cos(t4788);
        float t4790 = metal::sin(t4788);
        int t4791 = (int)t4785;
        int t4792 = (int)t4786;
        int t4793 = t3973 + t4791;
        float t4794 = memory[83984180 + t4793];
        int t4795 = t3973 + t4791;
        int t4796 = t4795 + 512;
        float t4797 = memory[83984180 + t4796];
        int t4798 = t3973 + t4792;
        float t4799 = memory[83984180 + t4798];
        int t4800 = t3973 + t4792;
        int t4801 = t4800 + 512;
        float t4802 = memory[83984180 + t4801];
        float t4803 = t4789 * t4799;
        float t4804 = t4790 * t4802;
        float t4805 = t4803 - t4804;
        float t4806 = t4789 * t4802;
        float t4807 = t4790 * t4799;
        float t4808 = t4806 + t4807;
        int t4809 = t3973 + t4791;
        float t4810 = t4794 + t4805;
        memory[83984180 + t4809] = t4810;
        int t4812 = t3973 + t4791;
        int t4813 = t4812 + 512;
        float t4814 = t4797 + t4808;
        memory[83984180 + t4813] = t4814;
        int t4816 = t3973 + t4792;
        float t4817 = t4794 - t4805;
        memory[83984180 + t4816] = t4817;
        int t4819 = t3973 + t4792;
        int t4820 = t4819 + 512;
        float t4821 = t4797 - t4808;
        memory[83984180 + t4820] = t4821;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4824 = 0; _pr4824 < 256; _pr4824++) {
        float t4825 = (float)_pr4824;
        float t4826 = (t4825 * 0.015625);
        float t4827 = metal::floor(t4826);
        float t4828 = t4827 * 64.0;
        float t4829 = t4825 - t4828;
        float t4830 = t4827 * 128.0;
        float t4831 = t4830 + t4829;
        float t4832 = t4831 + 64.0;
        float t4833 = 6.283185 * t4829;
        float t4834 = (t4833 * 0.0078125);
        float t4835 = metal::cos(t4834);
        float t4836 = metal::sin(t4834);
        int t4837 = (int)t4831;
        int t4838 = (int)t4832;
        int t4839 = t3973 + t4837;
        float t4840 = memory[83984180 + t4839];
        int t4841 = t3973 + t4837;
        int t4842 = t4841 + 512;
        float t4843 = memory[83984180 + t4842];
        int t4844 = t3973 + t4838;
        float t4845 = memory[83984180 + t4844];
        int t4846 = t3973 + t4838;
        int t4847 = t4846 + 512;
        float t4848 = memory[83984180 + t4847];
        float t4849 = t4835 * t4845;
        float t4850 = t4836 * t4848;
        float t4851 = t4849 - t4850;
        float t4852 = t4835 * t4848;
        float t4853 = t4836 * t4845;
        float t4854 = t4852 + t4853;
        int t4855 = t3973 + t4837;
        float t4856 = t4840 + t4851;
        memory[83984180 + t4855] = t4856;
        int t4858 = t3973 + t4837;
        int t4859 = t4858 + 512;
        float t4860 = t4843 + t4854;
        memory[83984180 + t4859] = t4860;
        int t4862 = t3973 + t4838;
        float t4863 = t4840 - t4851;
        memory[83984180 + t4862] = t4863;
        int t4865 = t3973 + t4838;
        int t4866 = t4865 + 512;
        float t4867 = t4843 - t4854;
        memory[83984180 + t4866] = t4867;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4870 = 0; _pr4870 < 256; _pr4870++) {
        float t4871 = (float)_pr4870;
        float t4872 = (t4871 * 0.0078125);
        float t4873 = metal::floor(t4872);
        float t4874 = t4873 * 128.0;
        float t4875 = t4871 - t4874;
        float t4876 = t4873 * 256.0;
        float t4877 = t4876 + t4875;
        float t4878 = t4877 + 128.0;
        float t4879 = 6.283185 * t4875;
        float t4880 = (t4879 * 0.00390625);
        float t4881 = metal::cos(t4880);
        float t4882 = metal::sin(t4880);
        int t4883 = (int)t4877;
        int t4884 = (int)t4878;
        int t4885 = t3973 + t4883;
        float t4886 = memory[83984180 + t4885];
        int t4887 = t3973 + t4883;
        int t4888 = t4887 + 512;
        float t4889 = memory[83984180 + t4888];
        int t4890 = t3973 + t4884;
        float t4891 = memory[83984180 + t4890];
        int t4892 = t3973 + t4884;
        int t4893 = t4892 + 512;
        float t4894 = memory[83984180 + t4893];
        float t4895 = t4881 * t4891;
        float t4896 = t4882 * t4894;
        float t4897 = t4895 - t4896;
        float t4898 = t4881 * t4894;
        float t4899 = t4882 * t4891;
        float t4900 = t4898 + t4899;
        int t4901 = t3973 + t4883;
        float t4902 = t4886 + t4897;
        memory[83984180 + t4901] = t4902;
        int t4904 = t3973 + t4883;
        int t4905 = t4904 + 512;
        float t4906 = t4889 + t4900;
        memory[83984180 + t4905] = t4906;
        int t4908 = t3973 + t4884;
        float t4909 = t4886 - t4897;
        memory[83984180 + t4908] = t4909;
        int t4911 = t3973 + t4884;
        int t4912 = t4911 + 512;
        float t4913 = t4889 - t4900;
        memory[83984180 + t4912] = t4913;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4916 = 0; _pr4916 < 256; _pr4916++) {
        float t4917 = (float)_pr4916;
        float t4918 = (t4917 * 0.00390625);
        float t4919 = metal::floor(t4918);
        float t4920 = t4919 * 256.0;
        float t4921 = t4917 - t4920;
        float t4922 = t4919 * 512.0;
        float t4923 = t4922 + t4921;
        float t4924 = t4923 + 256.0;
        float t4925 = 6.283185 * t4921;
        float t4926 = (t4925 * 0.001953125);
        float t4927 = metal::cos(t4926);
        float t4928 = metal::sin(t4926);
        int t4929 = (int)t4923;
        int t4930 = (int)t4924;
        int t4931 = t3973 + t4929;
        float t4932 = memory[83984180 + t4931];
        int t4933 = t3973 + t4929;
        int t4934 = t4933 + 512;
        float t4935 = memory[83984180 + t4934];
        int t4936 = t3973 + t4930;
        float t4937 = memory[83984180 + t4936];
        int t4938 = t3973 + t4930;
        int t4939 = t4938 + 512;
        float t4940 = memory[83984180 + t4939];
        float t4941 = t4927 * t4937;
        float t4942 = t4928 * t4940;
        float t4943 = t4941 - t4942;
        float t4944 = t4927 * t4940;
        float t4945 = t4928 * t4937;
        float t4946 = t4944 + t4945;
        int t4947 = t3973 + t4929;
        float t4948 = t4932 + t4943;
        memory[83984180 + t4947] = t4948;
        int t4950 = t3973 + t4929;
        int t4951 = t4950 + 512;
        float t4952 = t4935 + t4946;
        memory[83984180 + t4951] = t4952;
        int t4954 = t3973 + t4930;
        float t4955 = t4932 - t4943;
        memory[83984180 + t4954] = t4955;
        int t4957 = t3973 + t4930;
        int t4958 = t4957 + 512;
        float t4959 = t4935 - t4946;
        memory[83984180 + t4958] = t4959;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      for (uint _pr4962 = 0; _pr4962 < 512; _pr4962++) {
        int t4963 = t3973 + _pr4962;
        float t4964 = memory[83984180 + t4963];
        float t4965 = t4964 * 7.599708e-06;
        float t4966 = memory[21300 + (int)_pr4962];
        int t4967 = t3974 + _pr4962;
        float t4968 = t4965 * t4966;
        memory[125943604 + t4967] = t4968;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    }
    float t4972 = t[11*frameCount + id] > 0.0;
    if (t4972) {
      for (uint _pr4974 = 0; _pr4974 < 512; _pr4974++) {
        int t4975 = t3974 + _pr4974;
        memory[117538612 + t4975] = 0.0;
        int t4977 = t3974 + _pr4974;
        memory[125943604 + t4977] = 0.0;
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(4998), value: global(4998)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(460) - handled in variable access */
    int t4981 = id;
    float t4982 = 0.0;
    for (uint t4983 = 0; t4983 < 512; t4983++) {
      float t4984 = (float)t4983;
      float t4985 = (float)t4981;
      float t4986 = t4985 + t4984;
      int t4987 = 511 - t4983;
      float t4988 = frameCount - 1.0;
      float t4989 = metal::min(t4986, t4988);
      int t4990 = (int)t4989;
      int t4991 = t4990 * 512;
      int t4992 = t4991 + t4987;
      float t4993 = memory[117538612 + t4992];
      float t4994 = t4986 < frameCount;
      float t4995 = metal::select(0.0, t4993, t4994 > 0.0);
      float t4996 = t4982 + t4995;
      t4982 = t4996;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[19*frameCount + id] = (t4982 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5016), value: global(5016)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(460) - handled in variable access */
    int t4999 = id;
    float t5000 = 0.0;
    for (uint t5001 = 0; t5001 < 512; t5001++) {
      float t5002 = (float)t5001;
      float t5003 = (float)t4999;
      float t5004 = t5003 + t5002;
      int t5005 = 511 - t5001;
      float t5006 = frameCount - 1.0;
      float t5007 = metal::min(t5004, t5006);
      int t5008 = (int)t5007;
      int t5009 = t5008 * 512;
      int t5010 = t5009 + t5005;
      float t5011 = memory[125943604 + t5010];
      float t5012 = t5004 < frameCount;
      float t5013 = metal::select(0.0, t5011, t5012 > 0.0);
      float t5014 = t5000 + t5013;
      t5000 = t5014;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    t[20*frameCount + id] = (t5000 * 0.0027567567);
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
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5037), value: global(5037)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5036), value: global(5036)) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5021), value: global(5021)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5016) - handled in variable access */
    /* loadGlobal(4998) - handled in variable access */
    /* loadGlobal(3903) - handled in variable access */
    /* loadGlobal(3885) - handled in variable access */
    /* loadGlobal(405) - handled in variable access */
    /* loadGlobal(404) - handled in variable access */
    /* loadGlobal(386) - handled in variable access */
    /* loadGlobal(256) - handled in variable access */
    float t5017 = t[17*frameCount + id] + t[19*frameCount + id];
    float t5018 = t[18*frameCount + id] + t[20*frameCount + id];
    float t5019 = 0.015625 * t5017;
    float t5020 = t[7*frameCount + id] * t5017;
    t[21*frameCount + id] = t[6*frameCount + id] * t5019;
    float t5022 = t[5*frameCount + id] * t5019;
    float t5023 = (t[1*frameCount + id] - metal::floor(t[1*frameCount + id] / 61.0) * 61.0);
    float t5024 = t5023 < 0.0;
    float t5025 = t5023 + 61.0;
    float t5026 = metal::select(t5023, t5025, t5024 > 0.0);
    float t5027 = t5026;
    float t5028 = metal::floor(t5027);
    float t5029 = t5027 - t5028;
    float t5030 = t5028 + 1.0;
    float t5031 = t5030 >= 61.0;
    float t5032 = metal::select(t5030, 0.0, t5031 > 0.0);
    float t5033 = 1.0 - t5029;
    float t5034 = t5022 * t5033;
    float t5035 = t5022 * t5029;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209862452 + (int)t5028], t5034, metal::memory_order_relaxed);
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[209862452 + (int)t5032], t5035, metal::memory_order_relaxed);
    for (uint t5038 = 0; t5038 < 61; t5038++) {
      float t5039 = memory[209862452 + (int)t5038];
      float t5040 = memory[48628 + (int)t5038];
      float t5041 = t5039 / t5040;
      float t5042 = memory[48628 + (int)t5038];
      float t5043 = memory[48628 + (int)t5038];
      float t5044 = t5042 * t5043;
      float t5045 = 1.0 / t5044;
      float t5046 = memory[209862452 + (int)t5038];
      float t5047 = t5046 * -1.0;
      float t5048 = t5047 * t5045;
      float t5049 = t5041 + t5048;
      float t5050 = memory[48692 + (int)t5038];
      float t5051 = metal::exp(t5050);
      float t5052 = t5051 * t5048;
      float t5053 = -1.0 * t5052;
      int t5054 = id;
      int t5055 = t5054 * 61;
      int t5056 = t5055 + t5038;
      memory[1097524 + t5056] = t5053;
      float t5058 = memory[48756 + (int)t5038];
      float t5059 = t5058 * t5052;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5060 = 0; t5060 < 1; t5060++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=192, axis=0, in=[61, 1], out=[1], inFA=true, outFA=true), value: empty) */
      float t5061 = 0.0;
      int t5062 = t5060;
      int t5063 = t5062;
      int t5064 = t5060 - t5063;
      int t5065 = t5062;
      int t5066 = t5065;
      for (uint t5067 = 0; t5067 < 61; t5067++) {
        int t5068 = t5067;
        int t5069 = t5066 + t5068;
        int t5070 = id;
        int t5071 = t5070 * 61;
        int t5072 = t5071 + t5069;
        float t5073 = memory[1097524 + t5072];
        float t5074 = t5061 + t5073;
        t5061 = t5074;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5076 = id;
      int t5077 = t5076;
      int t5078 = t5077 + t5060;
      memory[48948 + t5078] = t5061;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5080 = 0; t5080 < 3904; t5080++) {
      /* [1mUOp[0m(op: [38;5;51mexpandAxisMarker[0m(node=194, axis=2, in=[61, 1], out=[61, 1, 64], inFA=true, outFA=true), value: empty) */
      int t5081 = t5080 / 64;
      int t5082 = t5081 % 61;
      int t5083 = t5082 * 1;
      int t5084 = 0 + t5083;
      int t5085 = t5080 / 64;
      int t5086 = t5085 % 1;
      int t5087 = t5086 * 1;
      int t5088 = t5084 + t5087;
      float t5089 = (float)t5088;
      int t5090 = id;
      int t5091 = t5090 * 61;
      float t5092 = t5091 + t5089;
      int t5093 = (int)t5092;
      float t5094 = memory[1097524 + t5093];
      float t5095 = (float)t5080;
      int t5096 = id;
      int t5097 = t5096 * 3904;
      float t5098 = t5097 + t5095;
      int t5099 = (int)t5098;
      memory[209862516 + t5099] = t5094;
      int t5101 = t5080 / 64;
      int t5102 = t5101 * 64;
      int t5103 = t5080 - t5102;
      int t5104 = t5103 / 64;
      int t5105 = t5104 * 64;
      int t5106 = t5103 - t5105;
      int t5107 = t5106 / 64;
      int t5108 = t5107 * 64;
      int t5109 = t5106 - t5108;
      float t5110 = memory[4416 + t5109];
      int t5111 = id;
      int t5112 = t5111 * 3904;
      int t5113 = t5112 + t5080;
      float t5114 = memory[209862516 + t5113];
      float t5115 = t5110 * t5114;
      int t5116 = id;
      int t5117 = t5116 * 3904;
      int t5118 = t5117 + t5080;
      memory[273825652 + t5118] = t5115;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5120 = 0; t5120 < 64; t5120++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=197, axis=0, in=[61, 1, 64], out=[1, 64], inFA=true, outFA=true), value: empty) */
      float t5121 = 0.0;
      int t5122 = t5120 / 64;
      int t5123 = t5122 * 64;
      int t5124 = t5120 - t5123;
      int t5125 = t5124;
      int t5126 = t5125;
      int t5127 = t5124 - t5126;
      int t5128 = t5122 * 64;
      int t5129 = t5128;
      int t5130 = t5125;
      int t5131 = t5129 + t5130;
      for (uint t5132 = 0; t5132 < 61; t5132++) {
        int t5133 = t5132 * 64;
        int t5134 = t5131 + t5133;
        int t5135 = t5132 * 64;
        int t5136 = t5135 + t5125;
        float t5137 = memory[29108 + t5136];
        float t5138 = t5132 + 0.0;
        int t5139 = id;
        int t5140 = t5139 * 61;
        float t5141 = t5140 + t5138;
        int t5142 = (int)t5141;
        float t5143 = memory[1097524 + t5142];
        float t5144 = t5137 * t5143;
        float t5145 = t5121 + t5144;
        t5121 = t5145;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5147 = id;
      int t5148 = t5147 * 64;
      int t5149 = t5148 + t5120;
      memory[37797684 + t5149] = t5121;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 64]), value: empty) */
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
    device float *t [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  /* [1mUOp[0m(op: [38;5;201mdefineGlobal[0m(5151), value: global(5151)) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5021) - handled in variable access */
    /* loadGlobal(378) - handled in variable access */
    /* loadGlobal(296) - handled in variable access */
    t[24*frameCount + id] = t[3*frameCount + id] * t[21*frameCount + id];
    float t5152 = t[4*frameCount + id] * t[21*frameCount + id];
  }
  #pragma clang diagnostic pop
}



// KERNEL 31
// Kind: simd
// ThreadCountScale Optional(64)
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_31(
    device float *memory [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  float t5552 = frameCount * 64.0;
  if (id >= 0 && id < (uint)(t5552)) {
    /* loadGlobal(5151) - handled in variable access */
    int t5153 = id;
    int t5154 = t5153 / 64;
    uint _frameIndex = (uint)(t5154);
    int t5155 = t5154 * 64;
    int t5156 = t5153 - t5155;
    int t5157 = t5154 * 64;
    int t5158 = t5157 + t5156;
    memory[42008372 + t5158] = t[24*frameCount + _frameIndex];
    int t5160 = _frameIndex;
    int t5161 = t5160 * 64;
    int t5162 = t5161 + t5156;
    float t5163 = memory[2146100 + t5162];
    int t5164 = _frameIndex;
    int t5165 = t5164 * 64;
    int t5166 = t5165 + t5156;
    float t5167 = memory[42008372 + t5166];
    float t5168 = t5163 * t5167;
    int t5169 = _frameIndex;
    int t5170 = t5169 * 64;
    int t5171 = t5170 + t5156;
    float t5172 = memory[3194676 + t5171];
    int t5173 = _frameIndex;
    int t5174 = t5173 * 64;
    int t5175 = t5174 + t5156;
    float t5176 = memory[42008372 + t5175];
    float t5177 = t5172 * t5176;
    int t5178 = _frameIndex;
    int t5179 = t5178 * 64;
    int t5180 = t5179 + t5156;
    memory[1097524 + t5180] = t5177;
  }
  #pragma clang diagnostic pop
}



// KERNEL 32
// Kind: simd
// ThreadCountScale Optional(3904)
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
  float t5553 = frameCount * 3904.0;
  if (id >= 0 && id < (uint)(t5553)) {
    /* loadGlobal(256) - handled in variable access */
    int t5182 = id;
    int t5183 = t5182 / 3904;
    uint _frameIndex = (uint)(t5183);
    int t5184 = t5183 * 3904;
    int t5185 = t5182 - t5184;
    float t5186 = (t[1*frameCount + _frameIndex] - metal::floor(t[1*frameCount + _frameIndex] / 61.0) * 61.0);
    float t5187 = t5186 < 0.0;
    float t5188 = t5186 + 61.0;
    float t5189 = metal::select(t5186, t5188, t5187 > 0.0);
    float t5190 = metal::floor(t5189);
    float t5191 = t5190 + 1.0;
    float t5192 = t5191 >= 61.0;
    float t5193 = metal::select(t5191, 0.0, t5192 > 0.0);
    float t5194 = t5189 - t5190;
    int t5195 = _frameIndex;
    memory[42008372 + t5195] = t5190;
    memory[46219060 + t5195] = t5194;
    float t5198 = t5195 + 16384.0;
    int t5199 = (int)t5198;
    memory[42008372 + t5199] = t5193;
    float t5201 = 1.0 - t5194;
    float t5202 = t5195 * 64.0;
    for (uint _pr5203 = 0; _pr5203 < 64; _pr5203++) {
      float t5204 = (float)_pr5203;
      float t5205 = t5202 + t5204;
      int t5206 = (int)t5205;
      float t5207 = memory[1097524 + t5206];
      float t5208 = t5202 + t5204;
      float t5209 = t5207 * t5201;
      int t5210 = (int)t5208;
      memory[2146100 + t5210] = t5209;
      float t5212 = t5207 * t5194;
      int t5213 = (int)t5208;
      memory[3194676 + t5213] = t5212;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 33
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(3904)
kernel void kernel_33(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 3904) { uint _pr5216 = id;
    int t5217 = _pr5216 / 64;
    int t5218 = t5217 * 64;
    int t5219 = _pr5216 - t5218;
    float t5220 = (float)t5217;
    float t5221 = (float)t5219;
    float t5222 = 0.0;
    for (uint t5223 = 0; t5223 < 16384; t5223++) {
      float t5224 = (float)t5223;
      float t5225 = t5224 < frameCount;
      float t5226 = t5224 * 64.0;
      float t5227 = t5226 + t5221;
      float t5228 = memory[42008372 + (int)t5223];
      float t5229 = t5228 - t5220;
      float t5230 = metal::abs(t5229);
      float t5231 = t5230 < 0.5;
      int t5232 = (int)t5227;
      float t5233 = memory[2146100 + t5232];
      float t5234 = t5225 * t5231;
      float t5235 = t5234 > 0.0;
      float t5236 = metal::select(0.0, t5233, t5235 > 0.0);
      float t5237 = t5222 + t5236;
      t5222 = t5237;
      float t5238 = t5224 + 16384.0;
      int t5239 = (int)t5238;
      float t5240 = memory[42008372 + t5239];
      float t5241 = t5240 - t5220;
      float t5242 = metal::abs(t5241);
      float t5243 = t5242 < 0.5;
      int t5244 = (int)t5227;
      float t5245 = memory[3194676 + t5244];
      float t5246 = t5225 * t5243;
      float t5247 = t5246 > 0.0;
      float t5248 = metal::select(0.0, t5245, t5247 > 0.0);
      float t5249 = t5222 + t5248;
      t5222 = t5249;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    float t5251 = t5220 * 64.0;
    float t5252 = t5251 + t5221;
    int t5253 = (int)t5252;
    atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[337788788 + t5253], t5222, metal::memory_order_relaxed);
  }
  #pragma clang diagnostic pop
}



// KERNEL 34
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_34(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    for (uint t5256 = 0; t5256 < 3904; t5256++) {
      float t5257 = memory[337788788 + (int)t5256];
      float t5258 = memory[36916 + (int)t5256];
      float t5259 = t5257 / t5258;
      float t5260 = memory[36916 + (int)t5256];
      float t5261 = memory[36916 + (int)t5256];
      float t5262 = t5260 * t5261;
      float t5263 = 1.0 / t5262;
      float t5264 = memory[337788788 + (int)t5256];
      float t5265 = t5264 * -1.0;
      float t5266 = t5265 * t5263;
      float t5267 = t5259 + t5266;
      float t5268 = memory[44724 + (int)t5256];
      float t5269 = metal::exp(t5268);
      float t5270 = t5269 * t5266;
      float t5271 = -1.0 * t5270;
      int t5272 = id;
      int t5273 = t5272 * 3904;
      int t5274 = t5273 + t5256;
      memory[209862516 + t5274] = t5271;
      float t5276 = memory[40820 + (int)t5256];
      float t5277 = t5276 * t5270;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5278 = 0; t5278 < 64; t5278++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=223, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5279 = 0.0;
      int t5280 = t5278;
      int t5281 = t5280;
      int t5282 = t5278 - t5281;
      int t5283 = t5280;
      int t5284 = t5283;
      for (uint t5285 = 0; t5285 < 61; t5285++) {
        int t5286 = t5285 * 64;
        int t5287 = t5284 + t5286;
        int t5288 = id;
        int t5289 = t5288 * 3904;
        int t5290 = t5289 + t5287;
        float t5291 = memory[209862516 + t5290];
        float t5292 = t5279 + t5291;
        t5279 = t5292;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5294 = id;
      int t5295 = t5294 * 64;
      int t5296 = t5295 + t5278;
      memory[1097524 + t5296] = t5279;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
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
    for (uint t5298 = 0; t5298 < 3904; t5298++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=228, axis=1, in=[61, 64, 64], out=[61, 64], inFA=true, outFA=true), value: empty) */
      float t5299 = 0.0;
      int t5300 = t5298 / 64;
      int t5301 = t5300 * 64;
      int t5302 = t5298 - t5301;
      int t5303 = t5302;
      int t5304 = t5303;
      int t5305 = t5302 - t5304;
      int t5306 = t5300 * 4096;
      int t5307 = t5306;
      int t5308 = t5303;
      int t5309 = t5307 + t5308;
      for (uint t5310 = 0; t5310 < 64; t5310++) {
        int t5311 = t5310 * 64;
        int t5312 = t5309 + t5311;
        int t5313 = t5310 * 64;
        int t5314 = t5313 + t5303;
        int t5315 = t5314 / 64;
        int t5316 = t5315 * 64;
        int t5317 = t5314 - t5316;
        int t5318 = t5317 * 64;
        int t5319 = t5315 + t5318;
        float t5320 = memory[256 + t5319];
        int t5321 = t5300 * 64;
        int t5322 = t5321 + t5310;
        int t5323 = id;
        int t5324 = t5323 * 3904;
        int t5325 = t5324 + t5322;
        float t5326 = memory[209862516 + t5325];
        float t5327 = t5320 * t5326;
        float t5328 = t5299 + t5327;
        t5299 = t5328;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5330 = id;
      int t5331 = t5330 * 3904;
      int t5332 = t5331 + t5298;
      memory[337792692 + t5332] = t5299;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 64]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5334 = 0; t5334 < 4096; t5334++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=230, axis=0, in=[61, 64, 64], out=[64, 64], inFA=true, outFA=true), value: empty) */
      float t5335 = 0.0;
      int t5336 = t5334 / 64;
      int t5337 = t5336 * 64;
      int t5338 = t5334 - t5337;
      int t5339 = t5338;
      int t5340 = t5339;
      int t5341 = t5338 - t5340;
      int t5342 = t5336 * 64;
      int t5343 = t5342;
      int t5344 = t5339;
      int t5345 = t5343 + t5344;
      for (uint t5346 = 0; t5346 < 61; t5346++) {
        int t5347 = t5346 * 4096;
        int t5348 = t5345 + t5347;
        int t5349 = t5346 * 64;
        int t5350 = t5349 + t5339;
        float t5351 = memory[29108 + t5350];
        int t5352 = t5346 * 64;
        int t5353 = t5352 + t5336;
        int t5354 = id;
        int t5355 = t5354 * 3904;
        int t5356 = t5355 + t5353;
        float t5357 = memory[209862516 + t5356];
        float t5358 = t5351 * t5357;
        float t5359 = t5335 + t5358;
        t5335 = t5359;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5361 = id;
      int t5362 = t5361 * 4096;
      int t5363 = t5362 + t5334;
      memory[401755828 + t5363] = t5335;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([64, 64]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 64]), value: empty) */
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
    for (uint t5365 = 0; t5365 < 3904; t5365++) {
      int t5366 = id;
      int t5367 = t5366 * 3904;
      int t5368 = t5367 + t5365;
      float t5369 = memory[273825652 + t5368];
      int t5370 = id;
      int t5371 = t5370 * 3904;
      int t5372 = t5371 + t5365;
      float t5373 = memory[337792692 + t5372];
      float t5374 = t5369 + t5373;
      float t5375 = memory[25204 + (int)t5365];
      float t5376 = metal::tanh(t5375);
      float t5377 = t5376 * t5376;
      float t5378 = 1.0 - t5377;
      float t5379 = t5378 * t5374;
      int t5380 = id;
      int t5381 = t5380 * 3904;
      int t5382 = t5381 + t5365;
      memory[468864692 + t5382] = t5379;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5384 = 0; t5384 < 64; t5384++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=241, axis=0, in=[61, 64], out=[64], inFA=true, outFA=true), value: empty) */
      float t5385 = 0.0;
      int t5386 = t5384;
      int t5387 = t5386;
      int t5388 = t5384 - t5387;
      int t5389 = t5386;
      int t5390 = t5389;
      for (uint t5391 = 0; t5391 < 61; t5391++) {
        int t5392 = t5391 * 64;
        int t5393 = t5390 + t5392;
        int t5394 = t5391 * 64;
        int t5395 = t5394 + t5386;
        float t5396 = memory[21300 + t5395];
        int t5397 = t5391 * 64;
        int t5398 = t5397 + t5386;
        int t5399 = id;
        int t5400 = t5399 * 3904;
        int t5401 = t5400 + t5398;
        float t5402 = memory[209862516 + t5401];
        float t5403 = t5396 * t5402;
        float t5404 = t5385 + t5403;
        t5385 = t5404;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5406 = id;
      int t5407 = t5406 * 64;
      int t5408 = t5407 + t5384;
      memory[2146100 + t5408] = t5385;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64]), value: empty) */
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
    for (uint t5410 = 0; t5410 < 183; t5410++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=246, axis=1, in=[61, 64, 3], out=[61, 3], inFA=true, outFA=true), value: empty) */
      float t5411 = 0.0;
      int t5412 = t5410 / 3;
      int t5413 = t5412 * 3;
      int t5414 = t5410 - t5413;
      int t5415 = t5414;
      int t5416 = t5415;
      int t5417 = t5414 - t5416;
      int t5418 = t5412 * 192;
      int t5419 = t5418;
      int t5420 = t5415;
      int t5421 = t5419 + t5420;
      for (uint t5422 = 0; t5422 < 64; t5422++) {
        int t5423 = t5422 * 3;
        int t5424 = t5421 + t5423;
        int t5425 = t5422 * 3;
        int t5426 = t5425 + t5415;
        int t5427 = t5426 / 3;
        int t5428 = t5427 * 3;
        int t5429 = t5426 - t5428;
        int t5430 = t5429 * 64;
        int t5431 = t5427 + t5430;
        float t5432 = memory[0 + t5431];
        int t5433 = t5412 * 64;
        int t5434 = t5433 + t5422;
        int t5435 = id;
        int t5436 = t5435 * 3904;
        int t5437 = t5436 + t5434;
        float t5438 = memory[468864692 + t5437];
        float t5439 = t5432 * t5438;
        float t5440 = t5411 + t5439;
        t5411 = t5440;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5442 = id;
      int t5443 = t5442 * 183;
      int t5444 = t5443 + t5410;
      memory[42008372 + t5444] = t5411;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([61, 1, 3]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    for (uint t5446 = 0; t5446 < 192; t5446++) {
      /* [1mUOp[0m(op: [38;5;51msumAxisMarker[0m(node=248, axis=0, in=[61, 64, 3], out=[64, 3], inFA=true, outFA=true), value: empty) */
      float t5447 = 0.0;
      int t5448 = t5446 / 3;
      int t5449 = t5448 * 3;
      int t5450 = t5446 - t5449;
      int t5451 = t5450;
      int t5452 = t5451;
      int t5453 = t5450 - t5452;
      int t5454 = t5448 * 3;
      int t5455 = t5454;
      int t5456 = t5451;
      int t5457 = t5455 + t5456;
      for (uint t5458 = 0; t5458 < 61; t5458++) {
        int t5459 = t5458 * 192;
        int t5460 = t5457 + t5459;
        int t5461 = t5458 * 3;
        int t5462 = t5461 + t5451;
        float t5463 = memory[4546 + t5462];
        int t5464 = t5458 * 64;
        int t5465 = t5464 + t5448;
        int t5466 = id;
        int t5467 = t5466 * 3904;
        int t5468 = t5467 + t5465;
        float t5469 = memory[468864692 + t5468];
        float t5470 = t5463 * t5469;
        float t5471 = t5447 + t5470;
        t5447 = t5471;
      } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
      int t5473 = id;
      int t5474 = t5473 * 192;
      int t5475 = t5474 + t5446;
      memory[46219060 + t5475] = t5447;
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([1, 64, 3]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mreshape[0m([64, 3]), value: empty) */
      /* [1mUOp[0m(op: [38;5;51mtranspose[0m([1, 0]), value: empty) */
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
  }
  #pragma clang diagnostic pop
}



// KERNEL 38
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(192)
kernel void kernel_38(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 192) { uint _pr5477 = id;
    float t5478 = 0.0;
    for (uint t5479 = 0; t5479 < 16384; t5479++) {
      int t5480 = t5479 * 192;
      int t5481 = t5480 + _pr5477;
      float t5482 = memory[46219060 + t5481];
      float t5483 = t5478 + t5482;
      t5478 = t5483;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[21300 + (int)_pr5477] = t5478;
  }
  #pragma clang diagnostic pop
}



// KERNEL 39
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_39(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5487 = id;
    float t5488 = 0.0;
    for (uint t5489 = 0; t5489 < 16384; t5489++) {
      int t5490 = t5489 * 64;
      int t5491 = t5490 + _pr5487;
      float t5492 = memory[2146100 + t5491];
      float t5493 = t5488 + t5492;
      t5488 = t5493;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[21300 + (int)_pr5487] = t5488;
  }
  #pragma clang diagnostic pop
}



// KERNEL 40
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(4096)
kernel void kernel_40(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 4096) { uint _pr5497 = id;
    float t5498 = 0.0;
    for (uint t5499 = 0; t5499 < 16384; t5499++) {
      int t5500 = t5499 * 4096;
      int t5501 = t5500 + _pr5497;
      float t5502 = memory[401755828 + t5501];
      float t5503 = t5498 + t5502;
      t5498 = t5503;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[2146100 + (int)_pr5497] = t5498;
  }
  #pragma clang diagnostic pop
}



// KERNEL 41
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(64)
kernel void kernel_41(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 64) { uint _pr5507 = id;
    float t5508 = 0.0;
    for (uint t5509 = 0; t5509 < 16384; t5509++) {
      int t5510 = t5509 * 64;
      int t5511 = t5510 + _pr5507;
      float t5512 = memory[1097524 + t5511];
      float t5513 = t5508 + t5512;
      t5508 = t5513;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[21300 + (int)_pr5507] = t5508;
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
  if (id < 64) { uint _pr5517 = id;
    float t5518 = 0.0;
    for (uint t5519 = 0; t5519 < 16384; t5519++) {
      int t5520 = t5519 * 64;
      int t5521 = t5520 + _pr5517;
      float t5522 = memory[37797684 + t5521];
      float t5523 = t5518 + t5522;
      t5518 = t5523;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[21300 + (int)_pr5517] = t5518;
  }
  #pragma clang diagnostic pop
}



// KERNEL 43
// Kind: scalar
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount Optional(1)
kernel void kernel_43(
    device float *memory [[buffer(0)]],
    constant uint &frameCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  uint i = 0; // Static block - no frame loop
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id < 1) { uint _pr5527 = id;
    float t5528 = 0.0;
    for (uint t5529 = 0; t5529 < 16384; t5529++) {
      int t5530 = t5529;
      int t5531 = t5530 + _pr5527;
      float t5532 = memory[48948 + t5531];
      float t5533 = t5528 + t5532;
      t5528 = t5533;
    } atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);
    memory[48628 + (int)_pr5527] = t5528;
  }
  #pragma clang diagnostic pop
}



// KERNEL 44
// Kind: simd
// ThreadCountScale nil
// ThreadGroupSize nil
// ThreadCount nil
kernel void kernel_44(
    device float *outputs [[buffer(0)]],
    device float *t [[buffer(1)]],
    constant uint &frameCount [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-variable"
  /* [1mUOp[0m(op: [38;5;201mframeCount[0m, value: empty) */
  if (id >= 0 && id < (uint)(frameCount)) {
    /* loadGlobal(5037) - handled in variable access */
    /* loadGlobal(5036) - handled in variable access */
    /* loadGlobal(2690) - handled in variable access */
    outputs[0 * frameCount + id] = t[16*frameCount + id];
  }
  #pragma clang diagnostic pop
}

