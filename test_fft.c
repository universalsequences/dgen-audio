// KERNEL 0
// FrameOrder: sequential
// DispatchMode: singleThreaded
// Threads: 1, ThreadgroupSize: 1 (sequential, frame-looped)
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// Enable profiling only when DGEN_PROFILE is defined by build flags

float32x4_t vfmodq_f32(float32x4_t a, float32x4_t b) {
  // a - floor(a / b) * b  (faster and correct for positive ranges)
  float32x4_t q = vdivq_f32(a, b);
  float32x4_t q_floor = vrndmq_f32(q);  // floor
  return vsubq_f32(a, vmulq_f32(b, q_floor));
}

static inline uint32x4_t mask_nz_f32(float32x4_t x) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    // eq0 = (x == 0.0f)
    uint32x4_t eq0  = vceqq_f32(x, zero);
    // non-zero mask = bitwise NOT of eq0
    return vmvnq_u32(eq0);
}

static inline float32x4_t boolmask_to_float(uint32x4_t m) {
    float32x4_t ones  = vdupq_n_f32(1.0f);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    // Select 1.0f where mask bits are 1, else 0.0f
    return vbslq_f32(m, ones, zeros);
}

static inline float32x4_t simd_and_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = vandq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

static inline float32x4_t simd_or_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = vorrq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

static inline float32x4_t simd_xor_f32(float32x4_t a, float32x4_t b) {
    uint32x4_t a_nz = mask_nz_f32(a);
    uint32x4_t b_nz = mask_nz_f32(b);
    uint32x4_t m    = veorq_u32(a_nz, b_nz);
    return boolmask_to_float(m);
}

const int VOICE_COUNT = 1;
const int SCRATCH_STRIDE = 4096;
float t131_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t132_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t133_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t150_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t151_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t153_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t171_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t188_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t189_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
float t191_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};
// Memory size required: 176191487 floats

void setParamValue(int cellId, float val) {
  //memory[cellId] = val;
}

void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state, void * restrict buffers) {
  int frameCount = nframes;  // Use audiograph frame count parameter
  int i = 0;
  float32x4_t c1 = vdupq_n_f32(1.0f);
  float32x4_t c2 = vdupq_n_f32(0.0f);
  float32x4_t c3 = vdupq_n_f32(5119.0f);
  float32x4_t c4 = vdupq_n_f32(128.0f);
  float32x4_t c5 = vdupq_n_f32(2.0f);
  float32x4_t c6 = vdupq_n_f32(512.0f);
  float32x4_t c7 = vdupq_n_f32(256.0f);
  float32x4_t c8 = vdupq_n_f32(64.0f);
  float32x4_t c9 = vdupq_n_f32(32.0f);
  float32x4_t c10 = vdupq_n_f32(16.0f);
  float32x4_t c11 = vdupq_n_f32(8.0f);
  float32x4_t c12 = vdupq_n_f32(4.0f);
  float32x4_t c13 = vdupq_n_f32(1024.0f);
  float32x4_t c14 = vdupq_n_f32(0.0009765625f);
  float *memory = (float*)state;
  int voiceIndex = 0;
  if (voiceIndex < 0) voiceIndex = 0;
  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;
  int _scratchBase = voiceIndex * SCRATCH_STRIDE;
  float *t131 = t131_g + _scratchBase;
  float *t132 = t132_g + _scratchBase;
  float *t133 = t133_g + _scratchBase;
  float *t150 = t150_g + _scratchBase;
  float *t151 = t151_g + _scratchBase;
  float *t153 = t153_g + _scratchBase;
  float *t171 = t171_g + _scratchBase;
  float *t188 = t188_g + _scratchBase;
  float *t189 = t189_g + _scratchBase;
  float *t191 = t191_g + _scratchBase;
  /* frameCount available as function parameter */
  for (int i = 0; i < frameCount; i += 4) {
    /* t132 declared globally */
    /* t131 declared globally */
    float32x4_t simd131 = vld1q_f32(in[0] + i); vst1q_f32(t131 + i, simd131);
    float32x4_t simd132 = vld1q_f32(in[1] + i); vst1q_f32(t132 + i, simd132);
  }
  for (int i = 0; i < frameCount; i += 1) {
    /* t133 declared globally */
    t133[i] = memory[112222206];
    float t134 = t133[i] + 1.0;
    float t135 = 0.0 > 0.0f ? 0.0 : t134;
    float t136 = t135;
    float t137 = (t136 / 5119.0f);
    float t138 = floorf(t137);
    float t139 = t138 * 5119.0;
    float t140 = t135 - t139;
    memory[112222206] = t140;
    float t142 = t140 >= 5119.0;
    if (t142) {
      float t144 = t140 - 5119.0;
      memory[112222206] = t144;
    }
    if (0.0) {
      memory[112222206] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd133 = vld1q_f32(t133 + i); /* extra */
    t133[i] = t133[i];
    /* t150 declared globally */
    float32x4_t simd150 = vrndmq_f32(simd133); vst1q_f32(t150 + i, simd150);
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
    float32x4_t simd131 = vld1q_f32(t131 + i); /* extra */
    t131[i] = t131[i];
    /* t151 declared globally */
    for (int t1 = 0; t1 < 1024; t1+=1) {
      memory[112217087 + (int)t150[i]] = t131[i];
    }
    /* t153 declared globally */
    t153[i] = memory[112222207];
    float t154 = t153[i] + 1.0;
    float t155 = 0.0 > 0.0f ? 0.0 : t154;
    float t156 = t155;
    float t157 = (t156 / 128.0f);
    float t158 = floorf(t157);
    float t159 = t158 * 128.0;
    float t160 = t155 - t159;
    memory[112222207] = t160;
    float t162 = t160 >= 128.0;
    if (t162) {
      float t164 = t160 - 128.0;
      memory[112222207] = t164;
    }
    if (0.0) {
      memory[112222207] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    float32x4_t simd151 = vld1q_f32(t151 + i); /* extra */
    t151[i] = t151[i];
    for (int simd2 = 0; simd2 < 1024; simd2+=4) {
    }
  }
  for (int i = 0; i < frameCount; i += 1) {
    /* t171 declared globally */
    t171[i] = memory[10240];
    float t172 = t171[i] + 1.0;
    float t173 = 0.0 > 0.0f ? 0.0 : t172;
    float t174 = t173;
    float t175 = (t174 / 5119.0f);
    float t176 = floorf(t175);
    float t177 = t176 * 5119.0;
    float t178 = t173 - t177;
    memory[10240] = t178;
    float t180 = t178 >= 5119.0;
    if (t180) {
      float t182 = t178 - 5119.0;
      memory[10240] = t182;
    }
    if (0.0) {
      memory[10240] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 4) {
    float32x4_t simd171 = vld1q_f32(t171 + i); /* extra */
    t171[i] = t171[i];
    /* t188 declared globally */
    float32x4_t simd188 = vrndmq_f32(simd171); vst1q_f32(t188 + i, simd188);
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
    float32x4_t simd132 = vld1q_f32(t132 + i); /* extra */
    t132[i] = t132[i];
    /* t189 declared globally */
    for (int t3 = 0; t3 < 1024; t3+=1) {
      memory[112222208 + (int)t188[i]] = t132[i];
    }
    /* t191 declared globally */
    t191[i] = memory[10241];
    float t192 = t191[i] + 1.0;
    float t193 = 0.0 > 0.0f ? 0.0 : t192;
    float t194 = t193;
    float t195 = (t194 / 128.0f);
    float t196 = floorf(t195);
    float t197 = t196 * 128.0;
    float t198 = t193 - t197;
    memory[10241] = t198;
    float t200 = t198 >= 128.0;
    if (t200) {
      float t202 = t198 - 128.0;
      memory[10241] = t202;
    }
    if (0.0) {
      memory[10241] = 0.0;
    }
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    float32x4_t simd189 = vld1q_f32(t189 + i); /* extra */
    t189[i] = t189[i];
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
    for (int simd4 = 0; simd4 < 1024; simd4+=4) {
      /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), value: empty) */
      /*  [1mUOp [0m(op:  [38;5;51mtranspose [0m([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), value: empty) */
      /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
    }
  }
  for (int simd5 = 0; simd5 < 1024; simd5+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      for (int simd6 = 0; simd6 < 1024; simd6+=4) {
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
      }
    }
  }
  for (int t7 = 0; t7 < 1; t7+=1) {
  }
  for (int simd8 = 0; simd8 < 512; simd8+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
  }
  for (int t9 = 0; t9 < 1; t9+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
      for (int t214 = 0; t214 < 512; t214++) {
        int t215 = t214;
        int t216 = t215;
        int t217 = t214 - t216;
        int t218 = t215;
        int t219 = t218;
        int t220 = t217;
        int t221 = t219 + t220;
        int t222 = t221;
        int t223 = t222;
        int t224 = t221 - t223;
        int t225 = t224;
        int t226 = t225;
        int t227 = t224 - t226;
        int t228 = t225 + 1;
        int t229 = t222 * 2;
        int t230 = t229;
        int t231 = t228;
        int t232 = t230 + t231;
        int t233 = t227;
        int t234 = t232 + t233;
        int t235 = t234;
        int t236 = t235;
        int t237 = t236 / 512;
        int t238 = t237 * 512;
        int t239 = t236 - t238;
        int t240 = t239 / 256;
        int t241 = t240 * 256;
        int t242 = t239 - t241;
        int t243 = t242 / 128;
        int t244 = t243 * 128;
        int t245 = t242 - t244;
        int t246 = t245 / 64;
        int t247 = t246 * 64;
        int t248 = t245 - t247;
        int t249 = t248 / 32;
        int t250 = t249 * 32;
        int t251 = t248 - t250;
        int t252 = t251 / 16;
        int t253 = t252 * 16;
        int t254 = t251 - t253;
        int t255 = t254 / 8;
        int t256 = t255 * 8;
        int t257 = t254 - t256;
        int t258 = t257 / 4;
        int t259 = t258 * 4;
        int t260 = t257 - t259;
        int t261 = t260 / 2;
        int t262 = t261 * 2;
        int t263 = t260 - t262;
        int t264 = t263 * 512;
        int t265 = t264;
        int t266 = t261 * 256;
        int t267 = t265 + t266;
        int t268 = t258 * 128;
        int t269 = t267 + t268;
        int t270 = t255 * 64;
        int t271 = t269 + t270;
        int t272 = t252 * 32;
        int t273 = t271 + t272;
        int t274 = t249 * 16;
        int t275 = t273 + t274;
        int t276 = t246 * 8;
        int t277 = t275 + t276;
        int t278 = t243 * 4;
        int t279 = t277 + t278;
        int t280 = t240 * 2;
        int t281 = t279 + t280;
        int t282 = t237;
        int t283 = t281 + t282;
        int t284 = t283 / 1024;
        int t285 = t284 * 1024;
        int t286 = t283 - t285;
        float t287 = (int)t150[i];
        int t288 = t287 - 1024;
        int t289 = t288 + 1;
        int t290 = t289 + t286;
        int t291 = t290 + 5119;
        int t292 = t291 % 5119;
        int t293 = t284 * 5119;
        int t294 = t293 + t292;
        float t295 = memory[112217087 + t294];
        int t296 = t214;
        int t297 = t296;
        int t298 = t214 - t297;
        float t299 = memory[1024 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t300 = t295 * t299;
        int t301 = t214;
        int t302 = t301;
        int t303 = t214 - t302;
        int t304 = t301;
        int t305 = t304;
        int t306 = t301 - t305;
        int t307 = t306;
        int t308 = t307;
        int t309 = t306 - t308;
        int t310 = t304 * 2;
        int t311 = 1 + t310;
        float t312 = memory[0 + t311];
        int t313 = t214;
        int t314 = t313;
        int t315 = t214 - t314;
        float t316 = memory[1025 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t317 = t312 * t316;
        float t318 = t300 - t317;
        int t319 = t214;
        int t320 = t319;
        int t321 = t214 - t320;
        int t322 = t319;
        int t323 = t322;
        int t324 = t321;
        int t325 = t323 + t324;
        int t326 = t325;
        int t327 = t326;
        int t328 = t325 - t327;
        int t329 = t328;
        int t330 = t329;
        int t331 = t328 - t330;
        int t332 = t329 + 1;
        int t333 = t326 * 2;
        int t334 = t333;
        int t335 = t332;
        int t336 = t334 + t335;
        int t337 = t331;
        int t338 = t336 + t337;
        int t339 = t338;
        int t340 = t339;
        int t341 = t340 / 512;
        int t342 = t341 * 512;
        int t343 = t340 - t342;
        int t344 = t343 / 256;
        int t345 = t344 * 256;
        int t346 = t343 - t345;
        int t347 = t346 / 128;
        int t348 = t347 * 128;
        int t349 = t346 - t348;
        int t350 = t349 / 64;
        int t351 = t350 * 64;
        int t352 = t349 - t351;
        int t353 = t352 / 32;
        int t354 = t353 * 32;
        int t355 = t352 - t354;
        int t356 = t355 / 16;
        int t357 = t356 * 16;
        int t358 = t355 - t357;
        int t359 = t358 / 8;
        int t360 = t359 * 8;
        int t361 = t358 - t360;
        int t362 = t361 / 4;
        int t363 = t362 * 4;
        int t364 = t361 - t363;
        int t365 = t364 / 2;
        int t366 = t365 * 2;
        int t367 = t364 - t366;
        int t368 = t367 * 512;
        int t369 = t368;
        int t370 = t365 * 256;
        int t371 = t369 + t370;
        int t372 = t362 * 128;
        int t373 = t371 + t372;
        int t374 = t359 * 64;
        int t375 = t373 + t374;
        int t376 = t356 * 32;
        int t377 = t375 + t376;
        int t378 = t353 * 16;
        int t379 = t377 + t378;
        int t380 = t350 * 8;
        int t381 = t379 + t380;
        int t382 = t347 * 4;
        int t383 = t381 + t382;
        int t384 = t344 * 2;
        int t385 = t383 + t384;
        int t386 = t341;
        int t387 = t385 + t386;
        int t388 = t387 / 1024;
        int t389 = t388 * 1024;
        int t390 = t387 - t389;
        float t391 = (int)t150[i];
        int t392 = t391 - 1024;
        int t393 = t392 + 1;
        int t394 = t393 + t390;
        int t395 = t394 + 5119;
        int t396 = t395 % 5119;
        int t397 = t388 * 5119;
        int t398 = t397 + t396;
        float t399 = memory[112217087 + t398];
        int t400 = t214;
        int t401 = t400;
        int t402 = t214 - t401;
        float t403 = memory[1025 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t404 = t399 * t403;
        int t405 = t214;
        int t406 = t405;
        int t407 = t214 - t406;
        int t408 = t405;
        int t409 = t408;
        int t410 = t405 - t409;
        int t411 = t410;
        int t412 = t411;
        int t413 = t410 - t412;
        int t414 = t408 * 2;
        int t415 = 1 + t414;
        float t416 = memory[0 + t415];
        int t417 = t214;
        int t418 = t417;
        int t419 = t214 - t418;
        float t420 = memory[1024 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t421 = t416 * t420;
        float t422 = t404 + t421;
        int t423 = t214;
        int t424 = t423;
        int t425 = t214 - t424;
        int t426 = t423;
        int t427 = t426;
        int t428 = t425;
        int t429 = t427 + t428;
        int t430 = t429;
        int t431 = t430;
        int t432 = t429 - t431;
        int t433 = t432;
        int t434 = t433;
        int t435 = t432 - t434;
        int t436 = t433;
        int t437 = t430 * 2;
        int t438 = t437;
        int t439 = t436;
        int t440 = t438 + t439;
        int t441 = t435;
        int t442 = t440 + t441;
        int t443 = t442;
        int t444 = t443;
        int t445 = t444 / 512;
        int t446 = t445 * 512;
        int t447 = t444 - t446;
        int t448 = t447 / 256;
        int t449 = t448 * 256;
        int t450 = t447 - t449;
        int t451 = t450 / 128;
        int t452 = t451 * 128;
        int t453 = t450 - t452;
        int t454 = t453 / 64;
        int t455 = t454 * 64;
        int t456 = t453 - t455;
        int t457 = t456 / 32;
        int t458 = t457 * 32;
        int t459 = t456 - t458;
        int t460 = t459 / 16;
        int t461 = t460 * 16;
        int t462 = t459 - t461;
        int t463 = t462 / 8;
        int t464 = t463 * 8;
        int t465 = t462 - t464;
        int t466 = t465 / 4;
        int t467 = t466 * 4;
        int t468 = t465 - t467;
        int t469 = t468 / 2;
        int t470 = t469 * 2;
        int t471 = t468 - t470;
        int t472 = t471 * 512;
        int t473 = t472;
        int t474 = t469 * 256;
        int t475 = t473 + t474;
        int t476 = t466 * 128;
        int t477 = t475 + t476;
        int t478 = t463 * 64;
        int t479 = t477 + t478;
        int t480 = t460 * 32;
        int t481 = t479 + t480;
        int t482 = t457 * 16;
        int t483 = t481 + t482;
        int t484 = t454 * 8;
        int t485 = t483 + t484;
        int t486 = t451 * 4;
        int t487 = t485 + t486;
        int t488 = t448 * 2;
        int t489 = t487 + t488;
        int t490 = t445;
        int t491 = t489 + t490;
        int t492 = t491 / 1024;
        int t493 = t492 * 1024;
        int t494 = t491 - t493;
        float t495 = (int)t150[i];
        int t496 = t495 - 1024;
        int t497 = t496 + 1;
        int t498 = t497 + t494;
        int t499 = t498 + 5119;
        int t500 = t499 % 5119;
        int t501 = t492 * 5119;
        int t502 = t501 + t500;
        float t503 = memory[112217087 + t502];
        float t504 = t503 + t318;
        int t505 = i;
        int t506 = t505 * 512;
        int t507 = t506 + t214;
        memory[116422655 + t507] = t504;
        int t509 = t214;
        int t510 = t509;
        int t511 = t214 - t510;
        int t512 = t509;
        int t513 = t512;
        int t514 = t509 - t513;
        int t515 = t514;
        int t516 = t515;
        int t517 = t514 - t516;
        int t518 = t512 * 2;
        float t519 = memory[0 + t518];
        float t520 = t519 + t422;
        int t521 = i;
        int t522 = t521 * 512;
        int t523 = t522 + t214;
        memory[134248447 + t523] = t520;
        int t525 = t214;
        int t526 = t525;
        int t527 = t214 - t526;
        int t528 = t525;
        int t529 = t528;
        int t530 = t527;
        int t531 = t529 + t530;
        int t532 = t531;
        int t533 = t532;
        int t534 = t531 - t533;
        int t535 = t534;
        int t536 = t535;
        int t537 = t534 - t536;
        int t538 = t535;
        int t539 = t532 * 2;
        int t540 = t539;
        int t541 = t538;
        int t542 = t540 + t541;
        int t543 = t537;
        int t544 = t542 + t543;
        int t545 = t544;
        int t546 = t545;
        int t547 = t546 / 512;
        int t548 = t547 * 512;
        int t549 = t546 - t548;
        int t550 = t549 / 256;
        int t551 = t550 * 256;
        int t552 = t549 - t551;
        int t553 = t552 / 128;
        int t554 = t553 * 128;
        int t555 = t552 - t554;
        int t556 = t555 / 64;
        int t557 = t556 * 64;
        int t558 = t555 - t557;
        int t559 = t558 / 32;
        int t560 = t559 * 32;
        int t561 = t558 - t560;
        int t562 = t561 / 16;
        int t563 = t562 * 16;
        int t564 = t561 - t563;
        int t565 = t564 / 8;
        int t566 = t565 * 8;
        int t567 = t564 - t566;
        int t568 = t567 / 4;
        int t569 = t568 * 4;
        int t570 = t567 - t569;
        int t571 = t570 / 2;
        int t572 = t571 * 2;
        int t573 = t570 - t572;
        int t574 = t573 * 512;
        int t575 = t574;
        int t576 = t571 * 256;
        int t577 = t575 + t576;
        int t578 = t568 * 128;
        int t579 = t577 + t578;
        int t580 = t565 * 64;
        int t581 = t579 + t580;
        int t582 = t562 * 32;
        int t583 = t581 + t582;
        int t584 = t559 * 16;
        int t585 = t583 + t584;
        int t586 = t556 * 8;
        int t587 = t585 + t586;
        int t588 = t553 * 4;
        int t589 = t587 + t588;
        int t590 = t550 * 2;
        int t591 = t589 + t590;
        int t592 = t547;
        int t593 = t591 + t592;
        int t594 = t593 / 1024;
        int t595 = t594 * 1024;
        int t596 = t593 - t595;
        float t597 = (int)t150[i];
        int t598 = t597 - 1024;
        int t599 = t598 + 1;
        int t600 = t599 + t596;
        int t601 = t600 + 5119;
        int t602 = t601 % 5119;
        int t603 = t594 * 5119;
        int t604 = t603 + t602;
        float t605 = memory[112217087 + t604];
        float t606 = t605 - t318;
        int t607 = i;
        int t608 = t607 * 512;
        int t609 = t608 + t214;
        memory[171997183 + t609] = t606;
        int t611 = t214;
        int t612 = t611;
        int t613 = t214 - t612;
        int t614 = t611;
        int t615 = t614;
        int t616 = t611 - t615;
        int t617 = t616;
        int t618 = t617;
        int t619 = t616 - t618;
        int t620 = t614 * 2;
        float t621 = memory[0 + t620];
        float t622 = t621 - t422;
        int t623 = i;
        int t624 = t623 * 512;
        int t625 = t624 + t214;
        memory[142374911 + t625] = t622;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
      }
      for (int t627 = 0; t627 < 1024; t627++) {
        int t628 = t627 / 2;
        int t629 = t628 * 2;
        int t630 = t627 - t629;
        int t631 = t628 >= 0;
        int t632 = t628 < 512;
        float t633 = 1.0 * t631;
        float t634 = t633 * t632;
        int t635 = t628;
        int t636 = t630 >= 0;
        int t637 = t630 < 1;
        float t638 = t634 * t636;
        float t639 = t638 * t637;
        int t640 = t630;
        int t641 = t635 + t640;
        float t642 = 0.0;
        if (t639) {
          int t644 = i;
          int t645 = t644 * 512;
          int t646 = t645 + t641;
          float t647 = memory[116422655 + t646];
          t642 = t647;
        }
        int t649 = t627 / 2;
        int t650 = t649 * 2;
        int t651 = t627 - t650;
        int t652 = t649 >= 0;
        int t653 = t649 < 512;
        float t654 = 1.0 * t652;
        float t655 = t654 * t653;
        int t656 = t649;
        int t657 = t651 >= 1;
        int t658 = t651 < 2;
        float t659 = t655 * t657;
        float t660 = t659 * t658;
        int t661 = t651 - 1;
        int t662 = t656 + t661;
        float t663 = 0.0;
        if (t660) {
          int t665 = i;
          int t666 = t665 * 512;
          int t667 = t666 + t662;
          float t668 = memory[171997183 + t667];
          t663 = t668;
        }
        float t670 = t642 + t663;
        int t671 = i;
        int t672 = t671 * 1024;
        int t673 = t672 + t627;
        memory[172259327 + t673] = t670;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
        int t675 = t627 / 2;
        int t676 = t675 * 2;
        int t677 = t627 - t676;
        int t678 = t675 >= 0;
        int t679 = t675 < 512;
        float t680 = 1.0 * t678;
        float t681 = t680 * t679;
        int t682 = t675;
        int t683 = t677 >= 0;
        int t684 = t677 < 1;
        float t685 = t681 * t683;
        float t686 = t685 * t684;
        int t687 = t677;
        int t688 = t682 + t687;
        float t689 = 0.0;
        if (t686) {
          int t691 = i;
          int t692 = t691 * 512;
          int t693 = t692 + t688;
          float t694 = memory[134248447 + t693];
          t689 = t694;
        }
        int t696 = t627 / 2;
        int t697 = t696 * 2;
        int t698 = t627 - t697;
        int t699 = t696 >= 0;
        int t700 = t696 < 512;
        float t701 = 1.0 * t699;
        float t702 = t701 * t700;
        int t703 = t696;
        int t704 = t698 >= 1;
        int t705 = t698 < 2;
        float t706 = t702 * t704;
        float t707 = t706 * t705;
        int t708 = t698 - 1;
        int t709 = t703 + t708;
        float t710 = 0.0;
        if (t707) {
          int t712 = i;
          int t713 = t712 * 512;
          int t714 = t713 + t709;
          float t715 = memory[142374911 + t714];
          t710 = t715;
        }
        float t717 = t689 + t710;
        int t718 = i;
        int t719 = t718 * 1024;
        int t720 = t719 + t627;
        memory[132151295 + t720] = t717;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
      }
    }
  }
  for (int t11 = 0; t11 < 2; t11+=1) {
  }
  for (int simd12 = 0; simd12 < 512; simd12+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
  }
  for (int t13 = 0; t13 < 2; t13+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
      for (int t725 = 0; t725 < 512; t725++) {
        int t726 = t725 / 2;
        int t727 = t726 * 2;
        int t728 = t725 - t727;
        int t729 = t726 * 2;
        int t730 = t729 + t728;
        int t731 = t730 / 2;
        int t732 = t731 * 2;
        int t733 = t730 - t732;
        int t734 = t733 / 2;
        int t735 = t734 * 2;
        int t736 = t733 - t735;
        int t737 = t731 * 4;
        int t738 = 2 + t737;
        int t739 = t738 + t736;
        int t740 = i;
        int t741 = t740 * 1024;
        int t742 = t741 + t739;
        float t743 = memory[172259327 + t742];
        int t744 = t725 / 2;
        int t745 = t744 * 2;
        int t746 = t725 - t745;
        float t747 = memory[1026 + t746];
        float t748 = t743 * t747;
        int t749 = t725 / 2;
        int t750 = t749 * 2;
        int t751 = t725 - t750;
        int t752 = t749 * 2;
        int t753 = t752 + t751;
        int t754 = t753 / 2;
        int t755 = t754 * 2;
        int t756 = t753 - t755;
        int t757 = t756 / 2;
        int t758 = t757 * 2;
        int t759 = t756 - t758;
        int t760 = t754 * 4;
        int t761 = 2 + t760;
        int t762 = t761 + t759;
        int t763 = i;
        int t764 = t763 * 1024;
        int t765 = t764 + t762;
        float t766 = memory[132151295 + t765];
        int t767 = t725 / 2;
        int t768 = t767 * 2;
        int t769 = t725 - t768;
        float t770 = memory[1028 + t769];
        float t771 = t766 * t770;
        float t772 = t748 - t771;
        int t773 = t725 / 2;
        int t774 = t773 * 2;
        int t775 = t725 - t774;
        int t776 = t773 * 2;
        int t777 = t776 + t775;
        int t778 = t777 / 2;
        int t779 = t778 * 2;
        int t780 = t777 - t779;
        int t781 = t780 / 2;
        int t782 = t781 * 2;
        int t783 = t780 - t782;
        int t784 = t778 * 4;
        int t785 = 2 + t784;
        int t786 = t785 + t783;
        int t787 = i;
        int t788 = t787 * 1024;
        int t789 = t788 + t786;
        float t790 = memory[172259327 + t789];
        int t791 = t725 / 2;
        int t792 = t791 * 2;
        int t793 = t725 - t792;
        float t794 = memory[1028 + t793];
        float t795 = t790 * t794;
        int t796 = t725 / 2;
        int t797 = t796 * 2;
        int t798 = t725 - t797;
        int t799 = t796 * 2;
        int t800 = t799 + t798;
        int t801 = t800 / 2;
        int t802 = t801 * 2;
        int t803 = t800 - t802;
        int t804 = t803 / 2;
        int t805 = t804 * 2;
        int t806 = t803 - t805;
        int t807 = t801 * 4;
        int t808 = 2 + t807;
        int t809 = t808 + t806;
        int t810 = i;
        int t811 = t810 * 1024;
        int t812 = t811 + t809;
        float t813 = memory[132151295 + t812];
        int t814 = t725 / 2;
        int t815 = t814 * 2;
        int t816 = t725 - t815;
        float t817 = memory[1026 + t816];
        float t818 = t813 * t817;
        float t819 = t795 + t818;
        int t820 = t725 / 2;
        int t821 = t820 * 2;
        int t822 = t725 - t821;
        int t823 = t820 * 2;
        int t824 = t823 + t822;
        int t825 = t824 / 2;
        int t826 = t825 * 2;
        int t827 = t824 - t826;
        int t828 = t827 / 2;
        int t829 = t828 * 2;
        int t830 = t827 - t829;
        int t831 = t825 * 4;
        int t832 = t831 + t830;
        int t833 = i;
        int t834 = t833 * 1024;
        int t835 = t834 + t832;
        float t836 = memory[172259327 + t835];
        float t837 = t836 + t772;
        int t838 = i;
        int t839 = t838 * 512;
        int t840 = t839 + t725;
        memory[153647103 + t840] = t837;
        int t842 = t725 / 2;
        int t843 = t842 * 2;
        int t844 = t725 - t843;
        int t845 = t842 * 2;
        int t846 = t845 + t844;
        int t847 = t846 / 2;
        int t848 = t847 * 2;
        int t849 = t846 - t848;
        int t850 = t849 / 2;
        int t851 = t850 * 2;
        int t852 = t849 - t851;
        int t853 = t847 * 4;
        int t854 = t853 + t852;
        int t855 = i;
        int t856 = t855 * 1024;
        int t857 = t856 + t854;
        float t858 = memory[132151295 + t857];
        float t859 = t858 + t819;
        int t860 = i;
        int t861 = t860 * 512;
        int t862 = t861 + t725;
        memory[160724991 + t862] = t859;
        int t864 = t725 / 2;
        int t865 = t864 * 2;
        int t866 = t725 - t865;
        int t867 = t864 * 2;
        int t868 = t867 + t866;
        int t869 = t868 / 2;
        int t870 = t869 * 2;
        int t871 = t868 - t870;
        int t872 = t871 / 2;
        int t873 = t872 * 2;
        int t874 = t871 - t873;
        int t875 = t869 * 4;
        int t876 = t875 + t874;
        int t877 = i;
        int t878 = t877 * 1024;
        int t879 = t878 + t876;
        float t880 = memory[172259327 + t879];
        float t881 = t880 - t772;
        int t882 = i;
        int t883 = t882 * 512;
        int t884 = t883 + t725;
        memory[162035711 + t884] = t881;
        int t886 = t725 / 2;
        int t887 = t886 * 2;
        int t888 = t725 - t887;
        int t889 = t886 * 2;
        int t890 = t889 + t888;
        int t891 = t890 / 2;
        int t892 = t891 * 2;
        int t893 = t890 - t892;
        int t894 = t893 / 2;
        int t895 = t894 * 2;
        int t896 = t893 - t895;
        int t897 = t891 * 4;
        int t898 = t897 + t896;
        int t899 = i;
        int t900 = t899 * 1024;
        int t901 = t900 + t898;
        float t902 = memory[132151295 + t901];
        float t903 = t902 - t819;
        int t904 = i;
        int t905 = t904 * 512;
        int t906 = t905 + t725;
        memory[124811263 + t906] = t903;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
      }
      for (int t908 = 0; t908 < 1024; t908++) {
        int t909 = t908 / 4;
        int t910 = t909 * 4;
        int t911 = t908 - t910;
        int t912 = t909 >= 0;
        int t913 = t909 < 256;
        float t914 = 1.0 * t912;
        float t915 = t914 * t913;
        int t916 = t909;
        int t917 = t911 >= 0;
        int t918 = t911 < 2;
        float t919 = t915 * t917;
        float t920 = t919 * t918;
        int t921 = t911;
        int t922 = t916 * 2;
        int t923 = t922 + t921;
        float t924 = 0.0;
        if (t920) {
          int t926 = i;
          int t927 = t926 * 512;
          int t928 = t927 + t923;
          float t929 = memory[153647103 + t928];
          t924 = t929;
        }
        int t931 = t908 / 4;
        int t932 = t931 * 4;
        int t933 = t908 - t932;
        int t934 = t931 >= 0;
        int t935 = t931 < 256;
        float t936 = 1.0 * t934;
        float t937 = t936 * t935;
        int t938 = t931;
        int t939 = t933 >= 2;
        int t940 = t933 < 4;
        float t941 = t937 * t939;
        float t942 = t941 * t940;
        int t943 = t933 - 2;
        int t944 = t938 * 2;
        int t945 = t944 + t943;
        float t946 = 0.0;
        if (t942) {
          int t948 = i;
          int t949 = t948 * 512;
          int t950 = t949 + t945;
          float t951 = memory[162035711 + t950];
          t946 = t951;
        }
        float t953 = t924 + t946;
        int t954 = i;
        int t955 = t954 * 1024;
        int t956 = t955 + t908;
        memory[127956991 + t956] = t953;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
        int t958 = t908 / 4;
        int t959 = t958 * 4;
        int t960 = t908 - t959;
        int t961 = t958 >= 0;
        int t962 = t958 < 256;
        float t963 = 1.0 * t961;
        float t964 = t963 * t962;
        int t965 = t958;
        int t966 = t960 >= 0;
        int t967 = t960 < 2;
        float t968 = t964 * t966;
        float t969 = t968 * t967;
        int t970 = t960;
        int t971 = t965 * 2;
        int t972 = t971 + t970;
        float t973 = 0.0;
        if (t969) {
          int t975 = i;
          int t976 = t975 * 512;
          int t977 = t976 + t972;
          float t978 = memory[160724991 + t977];
          t973 = t978;
        }
        int t980 = t908 / 4;
        int t981 = t980 * 4;
        int t982 = t908 - t981;
        int t983 = t980 >= 0;
        int t984 = t980 < 256;
        float t985 = 1.0 * t983;
        float t986 = t985 * t984;
        int t987 = t980;
        int t988 = t982 >= 2;
        int t989 = t982 < 4;
        float t990 = t986 * t988;
        float t991 = t990 * t989;
        int t992 = t982 - 2;
        int t993 = t987 * 2;
        int t994 = t993 + t992;
        float t995 = 0.0;
        if (t991) {
          int t997 = i;
          int t998 = t997 * 512;
          int t999 = t998 + t994;
          float t1000 = memory[124811263 + t999];
          t995 = t1000;
        }
        float t1002 = t973 + t995;
        int t1003 = i;
        int t1004 = t1003 * 1024;
        int t1005 = t1004 + t908;
        memory[161511423 + t1005] = t1002;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
      }
    }
  }
  for (int simd15 = 0; simd15 < 4; simd15+=4) {
  }
  for (int simd16 = 0; simd16 < 512; simd16+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
  }
  for (int simd17 = 0; simd17 < 4; simd17+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
      for (int t1010 = 0; t1010 < 512; t1010++) {
        int t1011 = t1010 / 4;
        int t1012 = t1011 * 4;
        int t1013 = t1010 - t1012;
        int t1014 = t1011 * 4;
        int t1015 = t1014 + t1013;
        int t1016 = t1015 / 4;
        int t1017 = t1016 * 4;
        int t1018 = t1015 - t1017;
        int t1019 = t1018 / 4;
        int t1020 = t1019 * 4;
        int t1021 = t1018 - t1020;
        int t1022 = t1016 * 8;
        int t1023 = 4 + t1022;
        int t1024 = t1023 + t1021;
        int t1025 = i;
        int t1026 = t1025 * 1024;
        int t1027 = t1026 + t1024;
        float t1028 = memory[127956991 + t1027];
        int t1029 = t1010 / 4;
        int t1030 = t1029 * 4;
        int t1031 = t1010 - t1030;
        float t1032 = memory[1030 + t1031];
        float t1033 = t1028 * t1032;
        int t1034 = t1010 / 4;
        int t1035 = t1034 * 4;
        int t1036 = t1010 - t1035;
        int t1037 = t1034 * 4;
        int t1038 = t1037 + t1036;
        int t1039 = t1038 / 4;
        int t1040 = t1039 * 4;
        int t1041 = t1038 - t1040;
        int t1042 = t1041 / 4;
        int t1043 = t1042 * 4;
        int t1044 = t1041 - t1043;
        int t1045 = t1039 * 8;
        int t1046 = 4 + t1045;
        int t1047 = t1046 + t1044;
        int t1048 = i;
        int t1049 = t1048 * 1024;
        int t1050 = t1049 + t1047;
        float t1051 = memory[161511423 + t1050];
        int t1052 = t1010 / 4;
        int t1053 = t1052 * 4;
        int t1054 = t1010 - t1053;
        float t1055 = memory[1034 + t1054];
        float t1056 = t1051 * t1055;
        float t1057 = t1033 - t1056;
        int t1058 = t1010 / 4;
        int t1059 = t1058 * 4;
        int t1060 = t1010 - t1059;
        int t1061 = t1058 * 4;
        int t1062 = t1061 + t1060;
        int t1063 = t1062 / 4;
        int t1064 = t1063 * 4;
        int t1065 = t1062 - t1064;
        int t1066 = t1065 / 4;
        int t1067 = t1066 * 4;
        int t1068 = t1065 - t1067;
        int t1069 = t1063 * 8;
        int t1070 = 4 + t1069;
        int t1071 = t1070 + t1068;
        int t1072 = i;
        int t1073 = t1072 * 1024;
        int t1074 = t1073 + t1071;
        float t1075 = memory[127956991 + t1074];
        int t1076 = t1010 / 4;
        int t1077 = t1076 * 4;
        int t1078 = t1010 - t1077;
        float t1079 = memory[1034 + t1078];
        float t1080 = t1075 * t1079;
        int t1081 = t1010 / 4;
        int t1082 = t1081 * 4;
        int t1083 = t1010 - t1082;
        int t1084 = t1081 * 4;
        int t1085 = t1084 + t1083;
        int t1086 = t1085 / 4;
        int t1087 = t1086 * 4;
        int t1088 = t1085 - t1087;
        int t1089 = t1088 / 4;
        int t1090 = t1089 * 4;
        int t1091 = t1088 - t1090;
        int t1092 = t1086 * 8;
        int t1093 = 4 + t1092;
        int t1094 = t1093 + t1091;
        int t1095 = i;
        int t1096 = t1095 * 1024;
        int t1097 = t1096 + t1094;
        float t1098 = memory[161511423 + t1097];
        int t1099 = t1010 / 4;
        int t1100 = t1099 * 4;
        int t1101 = t1010 - t1100;
        float t1102 = memory[1030 + t1101];
        float t1103 = t1098 * t1102;
        float t1104 = t1080 + t1103;
        int t1105 = t1010 / 4;
        int t1106 = t1105 * 4;
        int t1107 = t1010 - t1106;
        int t1108 = t1105 * 4;
        int t1109 = t1108 + t1107;
        int t1110 = t1109 / 4;
        int t1111 = t1110 * 4;
        int t1112 = t1109 - t1111;
        int t1113 = t1112 / 4;
        int t1114 = t1113 * 4;
        int t1115 = t1112 - t1114;
        int t1116 = t1110 * 8;
        int t1117 = t1116 + t1115;
        int t1118 = i;
        int t1119 = t1118 * 1024;
        int t1120 = t1119 + t1117;
        float t1121 = memory[127956991 + t1120];
        float t1122 = t1121 + t1057;
        int t1123 = i;
        int t1124 = t1123 * 512;
        int t1125 = t1124 + t1010;
        memory[123762687 + t1125] = t1122;
        int t1127 = t1010 / 4;
        int t1128 = t1127 * 4;
        int t1129 = t1010 - t1128;
        int t1130 = t1127 * 4;
        int t1131 = t1130 + t1129;
        int t1132 = t1131 / 4;
        int t1133 = t1132 * 4;
        int t1134 = t1131 - t1133;
        int t1135 = t1134 / 4;
        int t1136 = t1135 * 4;
        int t1137 = t1134 - t1136;
        int t1138 = t1132 * 8;
        int t1139 = t1138 + t1137;
        int t1140 = i;
        int t1141 = t1140 * 1024;
        int t1142 = t1141 + t1139;
        float t1143 = memory[161511423 + t1142];
        float t1144 = t1143 + t1104;
        int t1145 = i;
        int t1146 = t1145 * 512;
        int t1147 = t1146 + t1010;
        memory[168851455 + t1147] = t1144;
        int t1149 = t1010 / 4;
        int t1150 = t1149 * 4;
        int t1151 = t1010 - t1150;
        int t1152 = t1149 * 4;
        int t1153 = t1152 + t1151;
        int t1154 = t1153 / 4;
        int t1155 = t1154 * 4;
        int t1156 = t1153 - t1155;
        int t1157 = t1156 / 4;
        int t1158 = t1157 * 4;
        int t1159 = t1156 - t1158;
        int t1160 = t1154 * 8;
        int t1161 = t1160 + t1159;
        int t1162 = i;
        int t1163 = t1162 * 1024;
        int t1164 = t1163 + t1161;
        float t1165 = memory[127956991 + t1164];
        float t1166 = t1165 - t1057;
        int t1167 = i;
        int t1168 = t1167 * 512;
        int t1169 = t1168 + t1010;
        memory[128481279 + t1169] = t1166;
        int t1171 = t1010 / 4;
        int t1172 = t1171 * 4;
        int t1173 = t1010 - t1172;
        int t1174 = t1171 * 4;
        int t1175 = t1174 + t1173;
        int t1176 = t1175 / 4;
        int t1177 = t1176 * 4;
        int t1178 = t1175 - t1177;
        int t1179 = t1178 / 4;
        int t1180 = t1179 * 4;
        int t1181 = t1178 - t1180;
        int t1182 = t1176 * 8;
        int t1183 = t1182 + t1181;
        int t1184 = i;
        int t1185 = t1184 * 1024;
        int t1186 = t1185 + t1183;
        float t1187 = memory[161511423 + t1186];
        float t1188 = t1187 - t1104;
        int t1189 = i;
        int t1190 = t1189 * 512;
        int t1191 = t1190 + t1010;
        memory[169637887 + t1191] = t1188;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
      }
      for (int t1193 = 0; t1193 < 1024; t1193++) {
        int t1194 = t1193 / 8;
        int t1195 = t1194 * 8;
        int t1196 = t1193 - t1195;
        int t1197 = t1194 >= 0;
        int t1198 = t1194 < 128;
        float t1199 = 1.0 * t1197;
        float t1200 = t1199 * t1198;
        int t1201 = t1194;
        int t1202 = t1196 >= 0;
        int t1203 = t1196 < 4;
        float t1204 = t1200 * t1202;
        float t1205 = t1204 * t1203;
        int t1206 = t1196;
        int t1207 = t1201 * 4;
        int t1208 = t1207 + t1206;
        float t1209 = 0.0;
        if (t1205) {
          int t1211 = i;
          int t1212 = t1211 * 512;
          int t1213 = t1212 + t1208;
          float t1214 = memory[123762687 + t1213];
          t1209 = t1214;
        }
        int t1216 = t1193 / 8;
        int t1217 = t1216 * 8;
        int t1218 = t1193 - t1217;
        int t1219 = t1216 >= 0;
        int t1220 = t1216 < 128;
        float t1221 = 1.0 * t1219;
        float t1222 = t1221 * t1220;
        int t1223 = t1216;
        int t1224 = t1218 >= 4;
        int t1225 = t1218 < 8;
        float t1226 = t1222 * t1224;
        float t1227 = t1226 * t1225;
        int t1228 = t1218 - 4;
        int t1229 = t1223 * 4;
        int t1230 = t1229 + t1228;
        float t1231 = 0.0;
        if (t1227) {
          int t1233 = i;
          int t1234 = t1233 * 512;
          int t1235 = t1234 + t1230;
          float t1236 = memory[128481279 + t1235];
          t1231 = t1236;
        }
        float t1238 = t1209 + t1231;
        int t1239 = i;
        int t1240 = t1239 * 1024;
        int t1241 = t1240 + t1193;
        memory[154433535 + t1241] = t1238;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
        int t1243 = t1193 / 8;
        int t1244 = t1243 * 8;
        int t1245 = t1193 - t1244;
        int t1246 = t1243 >= 0;
        int t1247 = t1243 < 128;
        float t1248 = 1.0 * t1246;
        float t1249 = t1248 * t1247;
        int t1250 = t1243;
        int t1251 = t1245 >= 0;
        int t1252 = t1245 < 4;
        float t1253 = t1249 * t1251;
        float t1254 = t1253 * t1252;
        int t1255 = t1245;
        int t1256 = t1250 * 4;
        int t1257 = t1256 + t1255;
        float t1258 = 0.0;
        if (t1254) {
          int t1260 = i;
          int t1261 = t1260 * 512;
          int t1262 = t1261 + t1257;
          float t1263 = memory[168851455 + t1262];
          t1258 = t1263;
        }
        int t1265 = t1193 / 8;
        int t1266 = t1265 * 8;
        int t1267 = t1193 - t1266;
        int t1268 = t1265 >= 0;
        int t1269 = t1265 < 128;
        float t1270 = 1.0 * t1268;
        float t1271 = t1270 * t1269;
        int t1272 = t1265;
        int t1273 = t1267 >= 4;
        int t1274 = t1267 < 8;
        float t1275 = t1271 * t1273;
        float t1276 = t1275 * t1274;
        int t1277 = t1267 - 4;
        int t1278 = t1272 * 4;
        int t1279 = t1278 + t1277;
        float t1280 = 0.0;
        if (t1276) {
          int t1282 = i;
          int t1283 = t1282 * 512;
          int t1284 = t1283 + t1279;
          float t1285 = memory[169637887 + t1284];
          t1280 = t1285;
        }
        float t1287 = t1258 + t1280;
        int t1288 = i;
        int t1289 = t1288 * 1024;
        int t1290 = t1289 + t1193;
        memory[121927679 + t1290] = t1287;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
      }
    }
  }
  for (int simd19 = 0; simd19 < 8; simd19+=4) {
  }
  for (int simd20 = 0; simd20 < 512; simd20+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
  }
  for (int simd21 = 0; simd21 < 8; simd21+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
      for (int t1295 = 0; t1295 < 512; t1295++) {
        int t1296 = t1295 / 8;
        int t1297 = t1296 * 8;
        int t1298 = t1295 - t1297;
        int t1299 = t1296 * 8;
        int t1300 = t1299 + t1298;
        int t1301 = t1300 / 8;
        int t1302 = t1301 * 8;
        int t1303 = t1300 - t1302;
        int t1304 = t1303 / 8;
        int t1305 = t1304 * 8;
        int t1306 = t1303 - t1305;
        int t1307 = t1301 * 16;
        int t1308 = 8 + t1307;
        int t1309 = t1308 + t1306;
        int t1310 = i;
        int t1311 = t1310 * 1024;
        int t1312 = t1311 + t1309;
        float t1313 = memory[154433535 + t1312];
        int t1314 = t1295 / 8;
        int t1315 = t1314 * 8;
        int t1316 = t1295 - t1315;
        float t1317 = memory[1038 + t1316];
        float t1318 = t1313 * t1317;
        int t1319 = t1295 / 8;
        int t1320 = t1319 * 8;
        int t1321 = t1295 - t1320;
        int t1322 = t1319 * 8;
        int t1323 = t1322 + t1321;
        int t1324 = t1323 / 8;
        int t1325 = t1324 * 8;
        int t1326 = t1323 - t1325;
        int t1327 = t1326 / 8;
        int t1328 = t1327 * 8;
        int t1329 = t1326 - t1328;
        int t1330 = t1324 * 16;
        int t1331 = 8 + t1330;
        int t1332 = t1331 + t1329;
        int t1333 = i;
        int t1334 = t1333 * 1024;
        int t1335 = t1334 + t1332;
        float t1336 = memory[121927679 + t1335];
        int t1337 = t1295 / 8;
        int t1338 = t1337 * 8;
        int t1339 = t1295 - t1338;
        float t1340 = memory[1046 + t1339];
        float t1341 = t1336 * t1340;
        float t1342 = t1318 - t1341;
        int t1343 = t1295 / 8;
        int t1344 = t1343 * 8;
        int t1345 = t1295 - t1344;
        int t1346 = t1343 * 8;
        int t1347 = t1346 + t1345;
        int t1348 = t1347 / 8;
        int t1349 = t1348 * 8;
        int t1350 = t1347 - t1349;
        int t1351 = t1350 / 8;
        int t1352 = t1351 * 8;
        int t1353 = t1350 - t1352;
        int t1354 = t1348 * 16;
        int t1355 = 8 + t1354;
        int t1356 = t1355 + t1353;
        int t1357 = i;
        int t1358 = t1357 * 1024;
        int t1359 = t1358 + t1356;
        float t1360 = memory[154433535 + t1359];
        int t1361 = t1295 / 8;
        int t1362 = t1361 * 8;
        int t1363 = t1295 - t1362;
        float t1364 = memory[1046 + t1363];
        float t1365 = t1360 * t1364;
        int t1366 = t1295 / 8;
        int t1367 = t1366 * 8;
        int t1368 = t1295 - t1367;
        int t1369 = t1366 * 8;
        int t1370 = t1369 + t1368;
        int t1371 = t1370 / 8;
        int t1372 = t1371 * 8;
        int t1373 = t1370 - t1372;
        int t1374 = t1373 / 8;
        int t1375 = t1374 * 8;
        int t1376 = t1373 - t1375;
        int t1377 = t1371 * 16;
        int t1378 = 8 + t1377;
        int t1379 = t1378 + t1376;
        int t1380 = i;
        int t1381 = t1380 * 1024;
        int t1382 = t1381 + t1379;
        float t1383 = memory[121927679 + t1382];
        int t1384 = t1295 / 8;
        int t1385 = t1384 * 8;
        int t1386 = t1295 - t1385;
        float t1387 = memory[1038 + t1386];
        float t1388 = t1383 * t1387;
        float t1389 = t1365 + t1388;
        int t1390 = t1295 / 8;
        int t1391 = t1390 * 8;
        int t1392 = t1295 - t1391;
        int t1393 = t1390 * 8;
        int t1394 = t1393 + t1392;
        int t1395 = t1394 / 8;
        int t1396 = t1395 * 8;
        int t1397 = t1394 - t1396;
        int t1398 = t1397 / 8;
        int t1399 = t1398 * 8;
        int t1400 = t1397 - t1399;
        int t1401 = t1395 * 16;
        int t1402 = t1401 + t1400;
        int t1403 = i;
        int t1404 = t1403 * 1024;
        int t1405 = t1404 + t1402;
        float t1406 = memory[154433535 + t1405];
        float t1407 = t1406 + t1342;
        int t1408 = i;
        int t1409 = t1408 * 512;
        int t1410 = t1409 + t1295;
        memory[167016447 + t1410] = t1407;
        int t1412 = t1295 / 8;
        int t1413 = t1412 * 8;
        int t1414 = t1295 - t1413;
        int t1415 = t1412 * 8;
        int t1416 = t1415 + t1414;
        int t1417 = t1416 / 8;
        int t1418 = t1417 * 8;
        int t1419 = t1416 - t1418;
        int t1420 = t1419 / 8;
        int t1421 = t1420 * 8;
        int t1422 = t1419 - t1421;
        int t1423 = t1417 * 16;
        int t1424 = t1423 + t1422;
        int t1425 = i;
        int t1426 = t1425 * 1024;
        int t1427 = t1426 + t1424;
        float t1428 = memory[121927679 + t1427];
        float t1429 = t1428 + t1389;
        int t1430 = i;
        int t1431 = t1430 * 512;
        int t1432 = t1431 + t1295;
        memory[149977087 + t1432] = t1429;
        int t1434 = t1295 / 8;
        int t1435 = t1434 * 8;
        int t1436 = t1295 - t1435;
        int t1437 = t1434 * 8;
        int t1438 = t1437 + t1436;
        int t1439 = t1438 / 8;
        int t1440 = t1439 * 8;
        int t1441 = t1438 - t1440;
        int t1442 = t1441 / 8;
        int t1443 = t1442 * 8;
        int t1444 = t1441 - t1443;
        int t1445 = t1439 * 16;
        int t1446 = t1445 + t1444;
        int t1447 = i;
        int t1448 = t1447 * 1024;
        int t1449 = t1448 + t1446;
        float t1450 = memory[154433535 + t1449];
        float t1451 = t1450 - t1342;
        int t1452 = i;
        int t1453 = t1452 * 512;
        int t1454 = t1453 + t1295;
        memory[113276927 + t1454] = t1451;
        int t1456 = t1295 / 8;
        int t1457 = t1456 * 8;
        int t1458 = t1295 - t1457;
        int t1459 = t1456 * 8;
        int t1460 = t1459 + t1458;
        int t1461 = t1460 / 8;
        int t1462 = t1461 * 8;
        int t1463 = t1460 - t1462;
        int t1464 = t1463 / 8;
        int t1465 = t1464 * 8;
        int t1466 = t1463 - t1465;
        int t1467 = t1461 * 16;
        int t1468 = t1467 + t1466;
        int t1469 = i;
        int t1470 = t1469 * 1024;
        int t1471 = t1470 + t1468;
        float t1472 = memory[121927679 + t1471];
        float t1473 = t1472 - t1389;
        int t1474 = i;
        int t1475 = t1474 * 512;
        int t1476 = t1475 + t1295;
        memory[175142911 + t1476] = t1473;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
      }
      for (int t1478 = 0; t1478 < 1024; t1478++) {
        int t1479 = t1478 / 16;
        int t1480 = t1479 * 16;
        int t1481 = t1478 - t1480;
        int t1482 = t1479 >= 0;
        int t1483 = t1479 < 64;
        float t1484 = 1.0 * t1482;
        float t1485 = t1484 * t1483;
        int t1486 = t1479;
        int t1487 = t1481 >= 0;
        int t1488 = t1481 < 8;
        float t1489 = t1485 * t1487;
        float t1490 = t1489 * t1488;
        int t1491 = t1481;
        int t1492 = t1486 * 8;
        int t1493 = t1492 + t1491;
        float t1494 = 0.0;
        if (t1490) {
          int t1496 = i;
          int t1497 = t1496 * 512;
          int t1498 = t1497 + t1493;
          float t1499 = memory[167016447 + t1498];
          t1494 = t1499;
        }
        int t1501 = t1478 / 16;
        int t1502 = t1501 * 16;
        int t1503 = t1478 - t1502;
        int t1504 = t1501 >= 0;
        int t1505 = t1501 < 64;
        float t1506 = 1.0 * t1504;
        float t1507 = t1506 * t1505;
        int t1508 = t1501;
        int t1509 = t1503 >= 8;
        int t1510 = t1503 < 16;
        float t1511 = t1507 * t1509;
        float t1512 = t1511 * t1510;
        int t1513 = t1503 - 8;
        int t1514 = t1508 * 8;
        int t1515 = t1514 + t1513;
        float t1516 = 0.0;
        if (t1512) {
          int t1518 = i;
          int t1519 = t1518 * 512;
          int t1520 = t1519 + t1515;
          float t1521 = memory[113276927 + t1520];
          t1516 = t1521;
        }
        float t1523 = t1494 + t1516;
        int t1524 = i;
        int t1525 = t1524 * 1024;
        int t1526 = t1525 + t1478;
        memory[112228351 + t1526] = t1523;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
        int t1528 = t1478 / 16;
        int t1529 = t1528 * 16;
        int t1530 = t1478 - t1529;
        int t1531 = t1528 >= 0;
        int t1532 = t1528 < 64;
        float t1533 = 1.0 * t1531;
        float t1534 = t1533 * t1532;
        int t1535 = t1528;
        int t1536 = t1530 >= 0;
        int t1537 = t1530 < 8;
        float t1538 = t1534 * t1536;
        float t1539 = t1538 * t1537;
        int t1540 = t1530;
        int t1541 = t1535 * 8;
        int t1542 = t1541 + t1540;
        float t1543 = 0.0;
        if (t1539) {
          int t1545 = i;
          int t1546 = t1545 * 512;
          int t1547 = t1546 + t1542;
          float t1548 = memory[149977087 + t1547];
          t1543 = t1548;
        }
        int t1550 = t1478 / 16;
        int t1551 = t1550 * 16;
        int t1552 = t1478 - t1551;
        int t1553 = t1550 >= 0;
        int t1554 = t1550 < 64;
        float t1555 = 1.0 * t1553;
        float t1556 = t1555 * t1554;
        int t1557 = t1550;
        int t1558 = t1552 >= 8;
        int t1559 = t1552 < 16;
        float t1560 = t1556 * t1558;
        float t1561 = t1560 * t1559;
        int t1562 = t1552 - 8;
        int t1563 = t1557 * 8;
        int t1564 = t1563 + t1562;
        float t1565 = 0.0;
        if (t1561) {
          int t1567 = i;
          int t1568 = t1567 * 512;
          int t1569 = t1568 + t1564;
          float t1570 = memory[175142911 + t1569];
          t1565 = t1570;
        }
        float t1572 = t1543 + t1565;
        int t1573 = i;
        int t1574 = t1573 * 1024;
        int t1575 = t1574 + t1478;
        memory[119044095 + t1575] = t1572;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
      }
    }
  }
  for (int simd23 = 0; simd23 < 16; simd23+=4) {
  }
  for (int simd24 = 0; simd24 < 512; simd24+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
  }
  for (int simd25 = 0; simd25 < 16; simd25+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
      for (int t1580 = 0; t1580 < 512; t1580++) {
        int t1581 = t1580 / 16;
        int t1582 = t1581 * 16;
        int t1583 = t1580 - t1582;
        int t1584 = t1581 * 16;
        int t1585 = t1584 + t1583;
        int t1586 = t1585 / 16;
        int t1587 = t1586 * 16;
        int t1588 = t1585 - t1587;
        int t1589 = t1588 / 16;
        int t1590 = t1589 * 16;
        int t1591 = t1588 - t1590;
        int t1592 = t1586 * 32;
        int t1593 = 16 + t1592;
        int t1594 = t1593 + t1591;
        int t1595 = i;
        int t1596 = t1595 * 1024;
        int t1597 = t1596 + t1594;
        float t1598 = memory[112228351 + t1597];
        int t1599 = t1580 / 16;
        int t1600 = t1599 * 16;
        int t1601 = t1580 - t1600;
        float t1602 = memory[1054 + t1601];
        float t1603 = t1598 * t1602;
        int t1604 = t1580 / 16;
        int t1605 = t1604 * 16;
        int t1606 = t1580 - t1605;
        int t1607 = t1604 * 16;
        int t1608 = t1607 + t1606;
        int t1609 = t1608 / 16;
        int t1610 = t1609 * 16;
        int t1611 = t1608 - t1610;
        int t1612 = t1611 / 16;
        int t1613 = t1612 * 16;
        int t1614 = t1611 - t1613;
        int t1615 = t1609 * 32;
        int t1616 = 16 + t1615;
        int t1617 = t1616 + t1614;
        int t1618 = i;
        int t1619 = t1618 * 1024;
        int t1620 = t1619 + t1617;
        float t1621 = memory[119044095 + t1620];
        int t1622 = t1580 / 16;
        int t1623 = t1622 * 16;
        int t1624 = t1580 - t1623;
        float t1625 = memory[1070 + t1624];
        float t1626 = t1621 * t1625;
        float t1627 = t1603 - t1626;
        int t1628 = t1580 / 16;
        int t1629 = t1628 * 16;
        int t1630 = t1580 - t1629;
        int t1631 = t1628 * 16;
        int t1632 = t1631 + t1630;
        int t1633 = t1632 / 16;
        int t1634 = t1633 * 16;
        int t1635 = t1632 - t1634;
        int t1636 = t1635 / 16;
        int t1637 = t1636 * 16;
        int t1638 = t1635 - t1637;
        int t1639 = t1633 * 32;
        int t1640 = 16 + t1639;
        int t1641 = t1640 + t1638;
        int t1642 = i;
        int t1643 = t1642 * 1024;
        int t1644 = t1643 + t1641;
        float t1645 = memory[112228351 + t1644];
        int t1646 = t1580 / 16;
        int t1647 = t1646 * 16;
        int t1648 = t1580 - t1647;
        float t1649 = memory[1070 + t1648];
        float t1650 = t1645 * t1649;
        int t1651 = t1580 / 16;
        int t1652 = t1651 * 16;
        int t1653 = t1580 - t1652;
        int t1654 = t1651 * 16;
        int t1655 = t1654 + t1653;
        int t1656 = t1655 / 16;
        int t1657 = t1656 * 16;
        int t1658 = t1655 - t1657;
        int t1659 = t1658 / 16;
        int t1660 = t1659 * 16;
        int t1661 = t1658 - t1660;
        int t1662 = t1656 * 32;
        int t1663 = 16 + t1662;
        int t1664 = t1663 + t1661;
        int t1665 = i;
        int t1666 = t1665 * 1024;
        int t1667 = t1666 + t1664;
        float t1668 = memory[119044095 + t1667];
        int t1669 = t1580 / 16;
        int t1670 = t1669 * 16;
        int t1671 = t1580 - t1670;
        float t1672 = memory[1054 + t1671];
        float t1673 = t1668 * t1672;
        float t1674 = t1650 + t1673;
        int t1675 = t1580 / 16;
        int t1676 = t1675 * 16;
        int t1677 = t1580 - t1676;
        int t1678 = t1675 * 16;
        int t1679 = t1678 + t1677;
        int t1680 = t1679 / 16;
        int t1681 = t1680 * 16;
        int t1682 = t1679 - t1681;
        int t1683 = t1682 / 16;
        int t1684 = t1683 * 16;
        int t1685 = t1682 - t1684;
        int t1686 = t1680 * 32;
        int t1687 = t1686 + t1685;
        int t1688 = i;
        int t1689 = t1688 * 1024;
        int t1690 = t1689 + t1687;
        float t1691 = memory[112228351 + t1690];
        float t1692 = t1691 + t1627;
        int t1693 = i;
        int t1694 = t1693 * 512;
        int t1695 = t1694 + t1580;
        memory[169900031 + t1695] = t1692;
        int t1697 = t1580 / 16;
        int t1698 = t1697 * 16;
        int t1699 = t1580 - t1698;
        int t1700 = t1697 * 16;
        int t1701 = t1700 + t1699;
        int t1702 = t1701 / 16;
        int t1703 = t1702 * 16;
        int t1704 = t1701 - t1703;
        int t1705 = t1704 / 16;
        int t1706 = t1705 * 16;
        int t1707 = t1704 - t1706;
        int t1708 = t1702 * 32;
        int t1709 = t1708 + t1707;
        int t1710 = i;
        int t1711 = t1710 * 1024;
        int t1712 = t1711 + t1709;
        float t1713 = memory[119044095 + t1712];
        float t1714 = t1713 + t1674;
        int t1715 = i;
        int t1716 = t1715 * 512;
        int t1717 = t1716 + t1580;
        memory[138442751 + t1717] = t1714;
        int t1719 = t1580 / 16;
        int t1720 = t1719 * 16;
        int t1721 = t1580 - t1720;
        int t1722 = t1719 * 16;
        int t1723 = t1722 + t1721;
        int t1724 = t1723 / 16;
        int t1725 = t1724 * 16;
        int t1726 = t1723 - t1725;
        int t1727 = t1726 / 16;
        int t1728 = t1727 * 16;
        int t1729 = t1726 - t1728;
        int t1730 = t1724 * 32;
        int t1731 = t1730 + t1729;
        int t1732 = i;
        int t1733 = t1732 * 1024;
        int t1734 = t1733 + t1731;
        float t1735 = memory[112228351 + t1734];
        float t1736 = t1735 - t1627;
        int t1737 = i;
        int t1738 = t1737 * 512;
        int t1739 = t1738 + t1580;
        memory[145782783 + t1739] = t1736;
        int t1741 = t1580 / 16;
        int t1742 = t1741 * 16;
        int t1743 = t1580 - t1742;
        int t1744 = t1741 * 16;
        int t1745 = t1744 + t1743;
        int t1746 = t1745 / 16;
        int t1747 = t1746 * 16;
        int t1748 = t1745 - t1747;
        int t1749 = t1748 / 16;
        int t1750 = t1749 * 16;
        int t1751 = t1748 - t1750;
        int t1752 = t1746 * 32;
        int t1753 = t1752 + t1751;
        int t1754 = i;
        int t1755 = t1754 * 1024;
        int t1756 = t1755 + t1753;
        float t1757 = memory[119044095 + t1756];
        float t1758 = t1757 - t1674;
        int t1759 = i;
        int t1760 = t1759 * 512;
        int t1761 = t1760 + t1580;
        memory[148928511 + t1761] = t1758;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
      }
      for (int t1763 = 0; t1763 < 1024; t1763++) {
        int t1764 = t1763 / 32;
        int t1765 = t1764 * 32;
        int t1766 = t1763 - t1765;
        int t1767 = t1764 >= 0;
        int t1768 = t1764 < 32;
        float t1769 = 1.0 * t1767;
        float t1770 = t1769 * t1768;
        int t1771 = t1764;
        int t1772 = t1766 >= 0;
        int t1773 = t1766 < 16;
        float t1774 = t1770 * t1772;
        float t1775 = t1774 * t1773;
        int t1776 = t1766;
        int t1777 = t1771 * 16;
        int t1778 = t1777 + t1776;
        float t1779 = 0.0;
        if (t1775) {
          int t1781 = i;
          int t1782 = t1781 * 512;
          int t1783 = t1782 + t1778;
          float t1784 = memory[169900031 + t1783];
          t1779 = t1784;
        }
        int t1786 = t1763 / 32;
        int t1787 = t1786 * 32;
        int t1788 = t1763 - t1787;
        int t1789 = t1786 >= 0;
        int t1790 = t1786 < 32;
        float t1791 = 1.0 * t1789;
        float t1792 = t1791 * t1790;
        int t1793 = t1786;
        int t1794 = t1788 >= 16;
        int t1795 = t1788 < 32;
        float t1796 = t1792 * t1794;
        float t1797 = t1796 * t1795;
        int t1798 = t1788 - 16;
        int t1799 = t1793 * 16;
        int t1800 = t1799 + t1798;
        float t1801 = 0.0;
        if (t1797) {
          int t1803 = i;
          int t1804 = t1803 * 512;
          int t1805 = t1804 + t1800;
          float t1806 = memory[145782783 + t1805];
          t1801 = t1806;
        }
        float t1808 = t1779 + t1801;
        int t1809 = i;
        int t1810 = t1809 * 1024;
        int t1811 = t1810 + t1763;
        memory[160987135 + t1811] = t1808;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
        int t1813 = t1763 / 32;
        int t1814 = t1813 * 32;
        int t1815 = t1763 - t1814;
        int t1816 = t1813 >= 0;
        int t1817 = t1813 < 32;
        float t1818 = 1.0 * t1816;
        float t1819 = t1818 * t1817;
        int t1820 = t1813;
        int t1821 = t1815 >= 0;
        int t1822 = t1815 < 16;
        float t1823 = t1819 * t1821;
        float t1824 = t1823 * t1822;
        int t1825 = t1815;
        int t1826 = t1820 * 16;
        int t1827 = t1826 + t1825;
        float t1828 = 0.0;
        if (t1824) {
          int t1830 = i;
          int t1831 = t1830 * 512;
          int t1832 = t1831 + t1827;
          float t1833 = memory[138442751 + t1832];
          t1828 = t1833;
        }
        int t1835 = t1763 / 32;
        int t1836 = t1835 * 32;
        int t1837 = t1763 - t1836;
        int t1838 = t1835 >= 0;
        int t1839 = t1835 < 32;
        float t1840 = 1.0 * t1838;
        float t1841 = t1840 * t1839;
        int t1842 = t1835;
        int t1843 = t1837 >= 16;
        int t1844 = t1837 < 32;
        float t1845 = t1841 * t1843;
        float t1846 = t1845 * t1844;
        int t1847 = t1837 - 16;
        int t1848 = t1842 * 16;
        int t1849 = t1848 + t1847;
        float t1850 = 0.0;
        if (t1846) {
          int t1852 = i;
          int t1853 = t1852 * 512;
          int t1854 = t1853 + t1849;
          float t1855 = memory[148928511 + t1854];
          t1850 = t1855;
        }
        float t1857 = t1828 + t1850;
        int t1858 = i;
        int t1859 = t1858 * 1024;
        int t1860 = t1859 + t1763;
        memory[151812095 + t1860] = t1857;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
      }
    }
  }
  for (int simd27 = 0; simd27 < 32; simd27+=4) {
  }
  for (int simd28 = 0; simd28 < 512; simd28+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
  }
  for (int simd29 = 0; simd29 < 32; simd29+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
      for (int t1865 = 0; t1865 < 512; t1865++) {
        int t1866 = t1865 / 32;
        int t1867 = t1866 * 32;
        int t1868 = t1865 - t1867;
        int t1869 = t1866 * 32;
        int t1870 = t1869 + t1868;
        int t1871 = t1870 / 32;
        int t1872 = t1871 * 32;
        int t1873 = t1870 - t1872;
        int t1874 = t1873 / 32;
        int t1875 = t1874 * 32;
        int t1876 = t1873 - t1875;
        int t1877 = t1871 * 64;
        int t1878 = 32 + t1877;
        int t1879 = t1878 + t1876;
        int t1880 = i;
        int t1881 = t1880 * 1024;
        int t1882 = t1881 + t1879;
        float t1883 = memory[160987135 + t1882];
        int t1884 = t1865 / 32;
        int t1885 = t1884 * 32;
        int t1886 = t1865 - t1885;
        float t1887 = memory[1086 + t1886];
        float t1888 = t1883 * t1887;
        int t1889 = t1865 / 32;
        int t1890 = t1889 * 32;
        int t1891 = t1865 - t1890;
        int t1892 = t1889 * 32;
        int t1893 = t1892 + t1891;
        int t1894 = t1893 / 32;
        int t1895 = t1894 * 32;
        int t1896 = t1893 - t1895;
        int t1897 = t1896 / 32;
        int t1898 = t1897 * 32;
        int t1899 = t1896 - t1898;
        int t1900 = t1894 * 64;
        int t1901 = 32 + t1900;
        int t1902 = t1901 + t1899;
        int t1903 = i;
        int t1904 = t1903 * 1024;
        int t1905 = t1904 + t1902;
        float t1906 = memory[151812095 + t1905];
        int t1907 = t1865 / 32;
        int t1908 = t1907 * 32;
        int t1909 = t1865 - t1908;
        float t1910 = memory[1118 + t1909];
        float t1911 = t1906 * t1910;
        float t1912 = t1888 - t1911;
        int t1913 = t1865 / 32;
        int t1914 = t1913 * 32;
        int t1915 = t1865 - t1914;
        int t1916 = t1913 * 32;
        int t1917 = t1916 + t1915;
        int t1918 = t1917 / 32;
        int t1919 = t1918 * 32;
        int t1920 = t1917 - t1919;
        int t1921 = t1920 / 32;
        int t1922 = t1921 * 32;
        int t1923 = t1920 - t1922;
        int t1924 = t1918 * 64;
        int t1925 = 32 + t1924;
        int t1926 = t1925 + t1923;
        int t1927 = i;
        int t1928 = t1927 * 1024;
        int t1929 = t1928 + t1926;
        float t1930 = memory[160987135 + t1929];
        int t1931 = t1865 / 32;
        int t1932 = t1931 * 32;
        int t1933 = t1865 - t1932;
        float t1934 = memory[1118 + t1933];
        float t1935 = t1930 * t1934;
        int t1936 = t1865 / 32;
        int t1937 = t1936 * 32;
        int t1938 = t1865 - t1937;
        int t1939 = t1936 * 32;
        int t1940 = t1939 + t1938;
        int t1941 = t1940 / 32;
        int t1942 = t1941 * 32;
        int t1943 = t1940 - t1942;
        int t1944 = t1943 / 32;
        int t1945 = t1944 * 32;
        int t1946 = t1943 - t1945;
        int t1947 = t1941 * 64;
        int t1948 = 32 + t1947;
        int t1949 = t1948 + t1946;
        int t1950 = i;
        int t1951 = t1950 * 1024;
        int t1952 = t1951 + t1949;
        float t1953 = memory[151812095 + t1952];
        int t1954 = t1865 / 32;
        int t1955 = t1954 * 32;
        int t1956 = t1865 - t1955;
        float t1957 = memory[1086 + t1956];
        float t1958 = t1953 * t1957;
        float t1959 = t1935 + t1958;
        int t1960 = t1865 / 32;
        int t1961 = t1960 * 32;
        int t1962 = t1865 - t1961;
        int t1963 = t1960 * 32;
        int t1964 = t1963 + t1962;
        int t1965 = t1964 / 32;
        int t1966 = t1965 * 32;
        int t1967 = t1964 - t1966;
        int t1968 = t1967 / 32;
        int t1969 = t1968 * 32;
        int t1970 = t1967 - t1969;
        int t1971 = t1965 * 64;
        int t1972 = t1971 + t1970;
        int t1973 = i;
        int t1974 = t1973 * 1024;
        int t1975 = t1974 + t1972;
        float t1976 = memory[160987135 + t1975];
        float t1977 = t1976 + t1912;
        int t1978 = i;
        int t1979 = t1978 * 512;
        int t1980 = t1979 + t1865;
        memory[116160511 + t1980] = t1977;
        int t1982 = t1865 / 32;
        int t1983 = t1982 * 32;
        int t1984 = t1865 - t1983;
        int t1985 = t1982 * 32;
        int t1986 = t1985 + t1984;
        int t1987 = t1986 / 32;
        int t1988 = t1987 * 32;
        int t1989 = t1986 - t1988;
        int t1990 = t1989 / 32;
        int t1991 = t1990 * 32;
        int t1992 = t1989 - t1991;
        int t1993 = t1987 * 64;
        int t1994 = t1993 + t1992;
        int t1995 = i;
        int t1996 = t1995 * 1024;
        int t1997 = t1996 + t1994;
        float t1998 = memory[151812095 + t1997];
        float t1999 = t1998 + t1959;
        int t2000 = i;
        int t2001 = t2000 * 512;
        int t2002 = t2001 + t1865;
        memory[130840575 + t2002] = t1999;
        int t2004 = t1865 / 32;
        int t2005 = t2004 * 32;
        int t2006 = t1865 - t2005;
        int t2007 = t2004 * 32;
        int t2008 = t2007 + t2006;
        int t2009 = t2008 / 32;
        int t2010 = t2009 * 32;
        int t2011 = t2008 - t2010;
        int t2012 = t2011 / 32;
        int t2013 = t2012 * 32;
        int t2014 = t2011 - t2013;
        int t2015 = t2009 * 64;
        int t2016 = t2015 + t2014;
        int t2017 = i;
        int t2018 = t2017 * 1024;
        int t2019 = t2018 + t2016;
        float t2020 = memory[160987135 + t2019];
        float t2021 = t2020 - t1912;
        int t2022 = i;
        int t2023 = t2022 * 512;
        int t2024 = t2023 + t1865;
        memory[174618623 + t2024] = t2021;
        int t2026 = t1865 / 32;
        int t2027 = t2026 * 32;
        int t2028 = t1865 - t2027;
        int t2029 = t2026 * 32;
        int t2030 = t2029 + t2028;
        int t2031 = t2030 / 32;
        int t2032 = t2031 * 32;
        int t2033 = t2030 - t2032;
        int t2034 = t2033 / 32;
        int t2035 = t2034 * 32;
        int t2036 = t2033 - t2035;
        int t2037 = t2031 * 64;
        int t2038 = t2037 + t2036;
        int t2039 = i;
        int t2040 = t2039 * 1024;
        int t2041 = t2040 + t2038;
        float t2042 = memory[151812095 + t2041];
        float t2043 = t2042 - t1959;
        int t2044 = i;
        int t2045 = t2044 * 512;
        int t2046 = t2045 + t1865;
        memory[127694847 + t2046] = t2043;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
      }
      for (int t2048 = 0; t2048 < 1024; t2048++) {
        int t2049 = t2048 / 64;
        int t2050 = t2049 * 64;
        int t2051 = t2048 - t2050;
        int t2052 = t2049 >= 0;
        int t2053 = t2049 < 16;
        float t2054 = 1.0 * t2052;
        float t2055 = t2054 * t2053;
        int t2056 = t2049;
        int t2057 = t2051 >= 0;
        int t2058 = t2051 < 32;
        float t2059 = t2055 * t2057;
        float t2060 = t2059 * t2058;
        int t2061 = t2051;
        int t2062 = t2056 * 32;
        int t2063 = t2062 + t2061;
        float t2064 = 0.0;
        if (t2060) {
          int t2066 = i;
          int t2067 = t2066 * 512;
          int t2068 = t2067 + t2063;
          float t2069 = memory[116160511 + t2068];
          t2064 = t2069;
        }
        int t2071 = t2048 / 64;
        int t2072 = t2071 * 64;
        int t2073 = t2048 - t2072;
        int t2074 = t2071 >= 0;
        int t2075 = t2071 < 16;
        float t2076 = 1.0 * t2074;
        float t2077 = t2076 * t2075;
        int t2078 = t2071;
        int t2079 = t2073 >= 32;
        int t2080 = t2073 < 64;
        float t2081 = t2077 * t2079;
        float t2082 = t2081 * t2080;
        int t2083 = t2073 - 32;
        int t2084 = t2078 * 32;
        int t2085 = t2084 + t2083;
        float t2086 = 0.0;
        if (t2082) {
          int t2088 = i;
          int t2089 = t2088 * 512;
          int t2090 = t2089 + t2085;
          float t2091 = memory[174618623 + t2090];
          t2086 = t2091;
        }
        float t2093 = t2064 + t2086;
        int t2094 = i;
        int t2095 = t2094 * 1024;
        int t2096 = t2095 + t2048;
        memory[121403391 + t2096] = t2093;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
        int t2098 = t2048 / 64;
        int t2099 = t2098 * 64;
        int t2100 = t2048 - t2099;
        int t2101 = t2098 >= 0;
        int t2102 = t2098 < 16;
        float t2103 = 1.0 * t2101;
        float t2104 = t2103 * t2102;
        int t2105 = t2098;
        int t2106 = t2100 >= 0;
        int t2107 = t2100 < 32;
        float t2108 = t2104 * t2106;
        float t2109 = t2108 * t2107;
        int t2110 = t2100;
        int t2111 = t2105 * 32;
        int t2112 = t2111 + t2110;
        float t2113 = 0.0;
        if (t2109) {
          int t2115 = i;
          int t2116 = t2115 * 512;
          int t2117 = t2116 + t2112;
          float t2118 = memory[130840575 + t2117];
          t2113 = t2118;
        }
        int t2120 = t2048 / 64;
        int t2121 = t2120 * 64;
        int t2122 = t2048 - t2121;
        int t2123 = t2120 >= 0;
        int t2124 = t2120 < 16;
        float t2125 = 1.0 * t2123;
        float t2126 = t2125 * t2124;
        int t2127 = t2120;
        int t2128 = t2122 >= 32;
        int t2129 = t2122 < 64;
        float t2130 = t2126 * t2128;
        float t2131 = t2130 * t2129;
        int t2132 = t2122 - 32;
        int t2133 = t2127 * 32;
        int t2134 = t2133 + t2132;
        float t2135 = 0.0;
        if (t2131) {
          int t2137 = i;
          int t2138 = t2137 * 512;
          int t2139 = t2138 + t2134;
          float t2140 = memory[127694847 + t2139];
          t2135 = t2140;
        }
        float t2142 = t2113 + t2135;
        int t2143 = i;
        int t2144 = t2143 * 1024;
        int t2145 = t2144 + t2048;
        memory[138704895 + t2145] = t2142;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 2, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 2, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
      }
    }
  }
  for (int simd31 = 0; simd31 < 64; simd31+=4) {
  }
  for (int simd32 = 0; simd32 < 512; simd32+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([8, 64]), value: empty) */
  }
  for (int simd33 = 0; simd33 < 64; simd33+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([8, 64]), value: empty) */
      for (int t2150 = 0; t2150 < 512; t2150++) {
        int t2151 = t2150 / 64;
        int t2152 = t2151 * 64;
        int t2153 = t2150 - t2152;
        int t2154 = t2151 * 64;
        int t2155 = t2154 + t2153;
        int t2156 = t2155 / 64;
        int t2157 = t2156 * 64;
        int t2158 = t2155 - t2157;
        int t2159 = t2158 / 64;
        int t2160 = t2159 * 64;
        int t2161 = t2158 - t2160;
        int t2162 = t2156 * 128;
        int t2163 = 64 + t2162;
        int t2164 = t2163 + t2161;
        int t2165 = i;
        int t2166 = t2165 * 1024;
        int t2167 = t2166 + t2164;
        float t2168 = memory[121403391 + t2167];
        int t2169 = t2150 / 64;
        int t2170 = t2169 * 64;
        int t2171 = t2150 - t2170;
        float t2172 = memory[1150 + t2171];
        float t2173 = t2168 * t2172;
        int t2174 = t2150 / 64;
        int t2175 = t2174 * 64;
        int t2176 = t2150 - t2175;
        int t2177 = t2174 * 64;
        int t2178 = t2177 + t2176;
        int t2179 = t2178 / 64;
        int t2180 = t2179 * 64;
        int t2181 = t2178 - t2180;
        int t2182 = t2181 / 64;
        int t2183 = t2182 * 64;
        int t2184 = t2181 - t2183;
        int t2185 = t2179 * 128;
        int t2186 = 64 + t2185;
        int t2187 = t2186 + t2184;
        int t2188 = i;
        int t2189 = t2188 * 1024;
        int t2190 = t2189 + t2187;
        float t2191 = memory[138704895 + t2190];
        int t2192 = t2150 / 64;
        int t2193 = t2192 * 64;
        int t2194 = t2150 - t2193;
        float t2195 = memory[1214 + t2194];
        float t2196 = t2191 * t2195;
        float t2197 = t2173 - t2196;
        int t2198 = t2150 / 64;
        int t2199 = t2198 * 64;
        int t2200 = t2150 - t2199;
        int t2201 = t2198 * 64;
        int t2202 = t2201 + t2200;
        int t2203 = t2202 / 64;
        int t2204 = t2203 * 64;
        int t2205 = t2202 - t2204;
        int t2206 = t2205 / 64;
        int t2207 = t2206 * 64;
        int t2208 = t2205 - t2207;
        int t2209 = t2203 * 128;
        int t2210 = 64 + t2209;
        int t2211 = t2210 + t2208;
        int t2212 = i;
        int t2213 = t2212 * 1024;
        int t2214 = t2213 + t2211;
        float t2215 = memory[121403391 + t2214];
        int t2216 = t2150 / 64;
        int t2217 = t2216 * 64;
        int t2218 = t2150 - t2217;
        float t2219 = memory[1214 + t2218];
        float t2220 = t2215 * t2219;
        int t2221 = t2150 / 64;
        int t2222 = t2221 * 64;
        int t2223 = t2150 - t2222;
        int t2224 = t2221 * 64;
        int t2225 = t2224 + t2223;
        int t2226 = t2225 / 64;
        int t2227 = t2226 * 64;
        int t2228 = t2225 - t2227;
        int t2229 = t2228 / 64;
        int t2230 = t2229 * 64;
        int t2231 = t2228 - t2230;
        int t2232 = t2226 * 128;
        int t2233 = 64 + t2232;
        int t2234 = t2233 + t2231;
        int t2235 = i;
        int t2236 = t2235 * 1024;
        int t2237 = t2236 + t2234;
        float t2238 = memory[138704895 + t2237];
        int t2239 = t2150 / 64;
        int t2240 = t2239 * 64;
        int t2241 = t2150 - t2240;
        float t2242 = memory[1150 + t2241];
        float t2243 = t2238 * t2242;
        float t2244 = t2220 + t2243;
        int t2245 = t2150 / 64;
        int t2246 = t2245 * 64;
        int t2247 = t2150 - t2246;
        int t2248 = t2245 * 64;
        int t2249 = t2248 + t2247;
        int t2250 = t2249 / 64;
        int t2251 = t2250 * 64;
        int t2252 = t2249 - t2251;
        int t2253 = t2252 / 64;
        int t2254 = t2253 * 64;
        int t2255 = t2252 - t2254;
        int t2256 = t2250 * 128;
        int t2257 = t2256 + t2255;
        int t2258 = i;
        int t2259 = t2258 * 1024;
        int t2260 = t2259 + t2257;
        float t2261 = memory[121403391 + t2260];
        float t2262 = t2261 + t2197;
        int t2263 = i;
        int t2264 = t2263 * 512;
        int t2265 = t2264 + t2150;
        memory[124024831 + t2265] = t2262;
        int t2267 = t2150 / 64;
        int t2268 = t2267 * 64;
        int t2269 = t2150 - t2268;
        int t2270 = t2267 * 64;
        int t2271 = t2270 + t2269;
        int t2272 = t2271 / 64;
        int t2273 = t2272 * 64;
        int t2274 = t2271 - t2273;
        int t2275 = t2274 / 64;
        int t2276 = t2275 * 64;
        int t2277 = t2274 - t2276;
        int t2278 = t2272 * 128;
        int t2279 = t2278 + t2277;
        int t2280 = i;
        int t2281 = t2280 * 1024;
        int t2282 = t2281 + t2279;
        float t2283 = memory[138704895 + t2282];
        float t2284 = t2283 + t2244;
        int t2285 = i;
        int t2286 = t2285 * 512;
        int t2287 = t2286 + t2150;
        memory[114325503 + t2287] = t2284;
        int t2289 = t2150 / 64;
        int t2290 = t2289 * 64;
        int t2291 = t2150 - t2290;
        int t2292 = t2289 * 64;
        int t2293 = t2292 + t2291;
        int t2294 = t2293 / 64;
        int t2295 = t2294 * 64;
        int t2296 = t2293 - t2295;
        int t2297 = t2296 / 64;
        int t2298 = t2297 * 64;
        int t2299 = t2296 - t2298;
        int t2300 = t2294 * 128;
        int t2301 = t2300 + t2299;
        int t2302 = i;
        int t2303 = t2302 * 1024;
        int t2304 = t2303 + t2301;
        float t2305 = memory[121403391 + t2304];
        float t2306 = t2305 - t2197;
        int t2307 = i;
        int t2308 = t2307 * 512;
        int t2309 = t2308 + t2150;
        memory[129791999 + t2309] = t2306;
        int t2311 = t2150 / 64;
        int t2312 = t2311 * 64;
        int t2313 = t2150 - t2312;
        int t2314 = t2311 * 64;
        int t2315 = t2314 + t2313;
        int t2316 = t2315 / 64;
        int t2317 = t2316 * 64;
        int t2318 = t2315 - t2317;
        int t2319 = t2318 / 64;
        int t2320 = t2319 * 64;
        int t2321 = t2318 - t2320;
        int t2322 = t2316 * 128;
        int t2323 = t2322 + t2321;
        int t2324 = i;
        int t2325 = t2324 * 1024;
        int t2326 = t2325 + t2323;
        float t2327 = memory[138704895 + t2326];
        float t2328 = t2327 - t2244;
        int t2329 = i;
        int t2330 = t2329 * 512;
        int t2331 = t2330 + t2150;
        memory[120092671 + t2331] = t2328;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 64)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (64, 0)]), value: empty) */
      }
      for (int t2333 = 0; t2333 < 1024; t2333++) {
        int t2334 = t2333 / 128;
        int t2335 = t2334 * 128;
        int t2336 = t2333 - t2335;
        int t2337 = t2334 >= 0;
        int t2338 = t2334 < 8;
        float t2339 = 1.0 * t2337;
        float t2340 = t2339 * t2338;
        int t2341 = t2334;
        int t2342 = t2336 >= 0;
        int t2343 = t2336 < 64;
        float t2344 = t2340 * t2342;
        float t2345 = t2344 * t2343;
        int t2346 = t2336;
        int t2347 = t2341 * 64;
        int t2348 = t2347 + t2346;
        float t2349 = 0.0;
        if (t2345) {
          int t2351 = i;
          int t2352 = t2351 * 512;
          int t2353 = t2352 + t2348;
          float t2354 = memory[124024831 + t2353];
          t2349 = t2354;
        }
        int t2356 = t2333 / 128;
        int t2357 = t2356 * 128;
        int t2358 = t2333 - t2357;
        int t2359 = t2356 >= 0;
        int t2360 = t2356 < 8;
        float t2361 = 1.0 * t2359;
        float t2362 = t2361 * t2360;
        int t2363 = t2356;
        int t2364 = t2358 >= 64;
        int t2365 = t2358 < 128;
        float t2366 = t2362 * t2364;
        float t2367 = t2366 * t2365;
        int t2368 = t2358 - 64;
        int t2369 = t2363 * 64;
        int t2370 = t2369 + t2368;
        float t2371 = 0.0;
        if (t2367) {
          int t2373 = i;
          int t2374 = t2373 * 512;
          int t2375 = t2374 + t2370;
          float t2376 = memory[129791999 + t2375];
          t2371 = t2376;
        }
        float t2378 = t2349 + t2371;
        int t2379 = i;
        int t2380 = t2379 * 1024;
        int t2381 = t2380 + t2333;
        memory[137394175 + t2381] = t2378;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 64)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (64, 0)]), value: empty) */
        int t2383 = t2333 / 128;
        int t2384 = t2383 * 128;
        int t2385 = t2333 - t2384;
        int t2386 = t2383 >= 0;
        int t2387 = t2383 < 8;
        float t2388 = 1.0 * t2386;
        float t2389 = t2388 * t2387;
        int t2390 = t2383;
        int t2391 = t2385 >= 0;
        int t2392 = t2385 < 64;
        float t2393 = t2389 * t2391;
        float t2394 = t2393 * t2392;
        int t2395 = t2385;
        int t2396 = t2390 * 64;
        int t2397 = t2396 + t2395;
        float t2398 = 0.0;
        if (t2394) {
          int t2400 = i;
          int t2401 = t2400 * 512;
          int t2402 = t2401 + t2397;
          float t2403 = memory[114325503 + t2402];
          t2398 = t2403;
        }
        int t2405 = t2333 / 128;
        int t2406 = t2405 * 128;
        int t2407 = t2333 - t2406;
        int t2408 = t2405 >= 0;
        int t2409 = t2405 < 8;
        float t2410 = 1.0 * t2408;
        float t2411 = t2410 * t2409;
        int t2412 = t2405;
        int t2413 = t2407 >= 64;
        int t2414 = t2407 < 128;
        float t2415 = t2411 * t2413;
        float t2416 = t2415 * t2414;
        int t2417 = t2407 - 64;
        int t2418 = t2412 * 64;
        int t2419 = t2418 + t2417;
        float t2420 = 0.0;
        if (t2416) {
          int t2422 = i;
          int t2423 = t2422 * 512;
          int t2424 = t2423 + t2419;
          float t2425 = memory[120092671 + t2424];
          t2420 = t2425;
        }
        float t2427 = t2398 + t2420;
        int t2428 = i;
        int t2429 = t2428 * 1024;
        int t2430 = t2429 + t2333;
        memory[133724159 + t2430] = t2427;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 2, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 2, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
      }
    }
  }
  for (int simd35 = 0; simd35 < 128; simd35+=4) {
  }
  for (int simd36 = 0; simd36 < 512; simd36+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([4, 128]), value: empty) */
  }
  for (int simd37 = 0; simd37 < 128; simd37+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([4, 128]), value: empty) */
      for (int t2435 = 0; t2435 < 512; t2435++) {
        int t2436 = t2435 / 128;
        int t2437 = t2436 * 128;
        int t2438 = t2435 - t2437;
        int t2439 = t2436 * 128;
        int t2440 = t2439 + t2438;
        int t2441 = t2440 / 128;
        int t2442 = t2441 * 128;
        int t2443 = t2440 - t2442;
        int t2444 = t2443 / 128;
        int t2445 = t2444 * 128;
        int t2446 = t2443 - t2445;
        int t2447 = t2441 * 256;
        int t2448 = 128 + t2447;
        int t2449 = t2448 + t2446;
        int t2450 = i;
        int t2451 = t2450 * 1024;
        int t2452 = t2451 + t2449;
        float t2453 = memory[137394175 + t2452];
        int t2454 = t2435 / 128;
        int t2455 = t2454 * 128;
        int t2456 = t2435 - t2455;
        float t2457 = memory[1278 + t2456];
        float t2458 = t2453 * t2457;
        int t2459 = t2435 / 128;
        int t2460 = t2459 * 128;
        int t2461 = t2435 - t2460;
        int t2462 = t2459 * 128;
        int t2463 = t2462 + t2461;
        int t2464 = t2463 / 128;
        int t2465 = t2464 * 128;
        int t2466 = t2463 - t2465;
        int t2467 = t2466 / 128;
        int t2468 = t2467 * 128;
        int t2469 = t2466 - t2468;
        int t2470 = t2464 * 256;
        int t2471 = 128 + t2470;
        int t2472 = t2471 + t2469;
        int t2473 = i;
        int t2474 = t2473 * 1024;
        int t2475 = t2474 + t2472;
        float t2476 = memory[133724159 + t2475];
        int t2477 = t2435 / 128;
        int t2478 = t2477 * 128;
        int t2479 = t2435 - t2478;
        float t2480 = memory[1406 + t2479];
        float t2481 = t2476 * t2480;
        float t2482 = t2458 - t2481;
        int t2483 = t2435 / 128;
        int t2484 = t2483 * 128;
        int t2485 = t2435 - t2484;
        int t2486 = t2483 * 128;
        int t2487 = t2486 + t2485;
        int t2488 = t2487 / 128;
        int t2489 = t2488 * 128;
        int t2490 = t2487 - t2489;
        int t2491 = t2490 / 128;
        int t2492 = t2491 * 128;
        int t2493 = t2490 - t2492;
        int t2494 = t2488 * 256;
        int t2495 = 128 + t2494;
        int t2496 = t2495 + t2493;
        int t2497 = i;
        int t2498 = t2497 * 1024;
        int t2499 = t2498 + t2496;
        float t2500 = memory[137394175 + t2499];
        int t2501 = t2435 / 128;
        int t2502 = t2501 * 128;
        int t2503 = t2435 - t2502;
        float t2504 = memory[1406 + t2503];
        float t2505 = t2500 * t2504;
        int t2506 = t2435 / 128;
        int t2507 = t2506 * 128;
        int t2508 = t2435 - t2507;
        int t2509 = t2506 * 128;
        int t2510 = t2509 + t2508;
        int t2511 = t2510 / 128;
        int t2512 = t2511 * 128;
        int t2513 = t2510 - t2512;
        int t2514 = t2513 / 128;
        int t2515 = t2514 * 128;
        int t2516 = t2513 - t2515;
        int t2517 = t2511 * 256;
        int t2518 = 128 + t2517;
        int t2519 = t2518 + t2516;
        int t2520 = i;
        int t2521 = t2520 * 1024;
        int t2522 = t2521 + t2519;
        float t2523 = memory[133724159 + t2522];
        int t2524 = t2435 / 128;
        int t2525 = t2524 * 128;
        int t2526 = t2435 - t2525;
        float t2527 = memory[1278 + t2526];
        float t2528 = t2523 * t2527;
        float t2529 = t2505 + t2528;
        int t2530 = t2435 / 128;
        int t2531 = t2530 * 128;
        int t2532 = t2435 - t2531;
        int t2533 = t2530 * 128;
        int t2534 = t2533 + t2532;
        int t2535 = t2534 / 128;
        int t2536 = t2535 * 128;
        int t2537 = t2534 - t2536;
        int t2538 = t2537 / 128;
        int t2539 = t2538 * 128;
        int t2540 = t2537 - t2539;
        int t2541 = t2535 * 256;
        int t2542 = t2541 + t2540;
        int t2543 = i;
        int t2544 = t2543 * 1024;
        int t2545 = t2544 + t2542;
        float t2546 = memory[137394175 + t2545];
        float t2547 = t2546 + t2482;
        int t2548 = i;
        int t2549 = t2548 * 512;
        int t2550 = t2549 + t2435;
        memory[134510591 + t2550] = t2547;
        int t2552 = t2435 / 128;
        int t2553 = t2552 * 128;
        int t2554 = t2435 - t2553;
        int t2555 = t2552 * 128;
        int t2556 = t2555 + t2554;
        int t2557 = t2556 / 128;
        int t2558 = t2557 * 128;
        int t2559 = t2556 - t2558;
        int t2560 = t2559 / 128;
        int t2561 = t2560 * 128;
        int t2562 = t2559 - t2561;
        int t2563 = t2557 * 256;
        int t2564 = t2563 + t2562;
        int t2565 = i;
        int t2566 = t2565 * 1024;
        int t2567 = t2566 + t2564;
        float t2568 = memory[133724159 + t2567];
        float t2569 = t2568 + t2529;
        int t2570 = i;
        int t2571 = t2570 * 512;
        int t2572 = t2571 + t2435;
        memory[156530687 + t2572] = t2569;
        int t2574 = t2435 / 128;
        int t2575 = t2574 * 128;
        int t2576 = t2435 - t2575;
        int t2577 = t2574 * 128;
        int t2578 = t2577 + t2576;
        int t2579 = t2578 / 128;
        int t2580 = t2579 * 128;
        int t2581 = t2578 - t2580;
        int t2582 = t2581 / 128;
        int t2583 = t2582 * 128;
        int t2584 = t2581 - t2583;
        int t2585 = t2579 * 256;
        int t2586 = t2585 + t2584;
        int t2587 = i;
        int t2588 = t2587 * 1024;
        int t2589 = t2588 + t2586;
        float t2590 = memory[137394175 + t2589];
        float t2591 = t2590 - t2482;
        int t2592 = i;
        int t2593 = t2592 * 512;
        int t2594 = t2593 + t2435;
        memory[130578431 + t2594] = t2591;
        int t2596 = t2435 / 128;
        int t2597 = t2596 * 128;
        int t2598 = t2435 - t2597;
        int t2599 = t2596 * 128;
        int t2600 = t2599 + t2598;
        int t2601 = t2600 / 128;
        int t2602 = t2601 * 128;
        int t2603 = t2600 - t2602;
        int t2604 = t2603 / 128;
        int t2605 = t2604 * 128;
        int t2606 = t2603 - t2605;
        int t2607 = t2601 * 256;
        int t2608 = t2607 + t2606;
        int t2609 = i;
        int t2610 = t2609 * 1024;
        int t2611 = t2610 + t2608;
        float t2612 = memory[133724159 + t2611];
        float t2613 = t2612 - t2529;
        int t2614 = i;
        int t2615 = t2614 * 512;
        int t2616 = t2615 + t2435;
        memory[158103551 + t2616] = t2613;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 128)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (128, 0)]), value: empty) */
      }
      for (int t2618 = 0; t2618 < 1024; t2618++) {
        int t2619 = t2618 / 256;
        int t2620 = t2619 * 256;
        int t2621 = t2618 - t2620;
        int t2622 = t2619 >= 0;
        int t2623 = t2619 < 4;
        float t2624 = 1.0 * t2622;
        float t2625 = t2624 * t2623;
        int t2626 = t2619;
        int t2627 = t2621 >= 0;
        int t2628 = t2621 < 128;
        float t2629 = t2625 * t2627;
        float t2630 = t2629 * t2628;
        int t2631 = t2621;
        int t2632 = t2626 * 128;
        int t2633 = t2632 + t2631;
        float t2634 = 0.0;
        if (t2630) {
          int t2636 = i;
          int t2637 = t2636 * 512;
          int t2638 = t2637 + t2633;
          float t2639 = memory[134510591 + t2638];
          t2634 = t2639;
        }
        int t2641 = t2618 / 256;
        int t2642 = t2641 * 256;
        int t2643 = t2618 - t2642;
        int t2644 = t2641 >= 0;
        int t2645 = t2641 < 4;
        float t2646 = 1.0 * t2644;
        float t2647 = t2646 * t2645;
        int t2648 = t2641;
        int t2649 = t2643 >= 128;
        int t2650 = t2643 < 256;
        float t2651 = t2647 * t2649;
        float t2652 = t2651 * t2650;
        int t2653 = t2643 - 128;
        int t2654 = t2648 * 128;
        int t2655 = t2654 + t2653;
        float t2656 = 0.0;
        if (t2652) {
          int t2658 = i;
          int t2659 = t2658 * 512;
          int t2660 = t2659 + t2655;
          float t2661 = memory[130578431 + t2660];
          t2656 = t2661;
        }
        float t2663 = t2634 + t2656;
        int t2664 = i;
        int t2665 = t2664 * 1024;
        int t2666 = t2665 + t2618;
        memory[154957823 + t2666] = t2663;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 128)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (128, 0)]), value: empty) */
        int t2668 = t2618 / 256;
        int t2669 = t2668 * 256;
        int t2670 = t2618 - t2669;
        int t2671 = t2668 >= 0;
        int t2672 = t2668 < 4;
        float t2673 = 1.0 * t2671;
        float t2674 = t2673 * t2672;
        int t2675 = t2668;
        int t2676 = t2670 >= 0;
        int t2677 = t2670 < 128;
        float t2678 = t2674 * t2676;
        float t2679 = t2678 * t2677;
        int t2680 = t2670;
        int t2681 = t2675 * 128;
        int t2682 = t2681 + t2680;
        float t2683 = 0.0;
        if (t2679) {
          int t2685 = i;
          int t2686 = t2685 * 512;
          int t2687 = t2686 + t2682;
          float t2688 = memory[156530687 + t2687];
          t2683 = t2688;
        }
        int t2690 = t2618 / 256;
        int t2691 = t2690 * 256;
        int t2692 = t2618 - t2691;
        int t2693 = t2690 >= 0;
        int t2694 = t2690 < 4;
        float t2695 = 1.0 * t2693;
        float t2696 = t2695 * t2694;
        int t2697 = t2690;
        int t2698 = t2692 >= 128;
        int t2699 = t2692 < 256;
        float t2700 = t2696 * t2698;
        float t2701 = t2700 * t2699;
        int t2702 = t2692 - 128;
        int t2703 = t2697 * 128;
        int t2704 = t2703 + t2702;
        float t2705 = 0.0;
        if (t2701) {
          int t2707 = i;
          int t2708 = t2707 * 512;
          int t2709 = t2708 + t2704;
          float t2710 = memory[158103551 + t2709];
          t2705 = t2710;
        }
        float t2712 = t2683 + t2705;
        int t2713 = i;
        int t2714 = t2713 * 1024;
        int t2715 = t2714 + t2618;
        memory[117209087 + t2715] = t2712;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
      }
    }
  }
  for (int simd39 = 0; simd39 < 256; simd39+=4) {
  }
  for (int simd40 = 0; simd40 < 512; simd40+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([2, 256]), value: empty) */
  }
  for (int simd41 = 0; simd41 < 256; simd41+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([2, 256]), value: empty) */
      for (int t2720 = 0; t2720 < 512; t2720++) {
        int t2721 = t2720 / 256;
        int t2722 = t2721 * 256;
        int t2723 = t2720 - t2722;
        int t2724 = t2721 * 256;
        int t2725 = t2724 + t2723;
        int t2726 = t2725 / 256;
        int t2727 = t2726 * 256;
        int t2728 = t2725 - t2727;
        int t2729 = t2728 / 256;
        int t2730 = t2729 * 256;
        int t2731 = t2728 - t2730;
        int t2732 = t2726 * 512;
        int t2733 = 256 + t2732;
        int t2734 = t2733 + t2731;
        int t2735 = i;
        int t2736 = t2735 * 1024;
        int t2737 = t2736 + t2734;
        float t2738 = memory[154957823 + t2737];
        int t2739 = t2720 / 256;
        int t2740 = t2739 * 256;
        int t2741 = t2720 - t2740;
        float t2742 = memory[1534 + t2741];
        float t2743 = t2738 * t2742;
        int t2744 = t2720 / 256;
        int t2745 = t2744 * 256;
        int t2746 = t2720 - t2745;
        int t2747 = t2744 * 256;
        int t2748 = t2747 + t2746;
        int t2749 = t2748 / 256;
        int t2750 = t2749 * 256;
        int t2751 = t2748 - t2750;
        int t2752 = t2751 / 256;
        int t2753 = t2752 * 256;
        int t2754 = t2751 - t2753;
        int t2755 = t2749 * 512;
        int t2756 = 256 + t2755;
        int t2757 = t2756 + t2754;
        int t2758 = i;
        int t2759 = t2758 * 1024;
        int t2760 = t2759 + t2757;
        float t2761 = memory[117209087 + t2760];
        int t2762 = t2720 / 256;
        int t2763 = t2762 * 256;
        int t2764 = t2720 - t2763;
        float t2765 = memory[1790 + t2764];
        float t2766 = t2761 * t2765;
        float t2767 = t2743 - t2766;
        int t2768 = t2720 / 256;
        int t2769 = t2768 * 256;
        int t2770 = t2720 - t2769;
        int t2771 = t2768 * 256;
        int t2772 = t2771 + t2770;
        int t2773 = t2772 / 256;
        int t2774 = t2773 * 256;
        int t2775 = t2772 - t2774;
        int t2776 = t2775 / 256;
        int t2777 = t2776 * 256;
        int t2778 = t2775 - t2777;
        int t2779 = t2773 * 512;
        int t2780 = 256 + t2779;
        int t2781 = t2780 + t2778;
        int t2782 = i;
        int t2783 = t2782 * 1024;
        int t2784 = t2783 + t2781;
        float t2785 = memory[154957823 + t2784];
        int t2786 = t2720 / 256;
        int t2787 = t2786 * 256;
        int t2788 = t2720 - t2787;
        float t2789 = memory[1790 + t2788];
        float t2790 = t2785 * t2789;
        int t2791 = t2720 / 256;
        int t2792 = t2791 * 256;
        int t2793 = t2720 - t2792;
        int t2794 = t2791 * 256;
        int t2795 = t2794 + t2793;
        int t2796 = t2795 / 256;
        int t2797 = t2796 * 256;
        int t2798 = t2795 - t2797;
        int t2799 = t2798 / 256;
        int t2800 = t2799 * 256;
        int t2801 = t2798 - t2800;
        int t2802 = t2796 * 512;
        int t2803 = 256 + t2802;
        int t2804 = t2803 + t2801;
        int t2805 = i;
        int t2806 = t2805 * 1024;
        int t2807 = t2806 + t2804;
        float t2808 = memory[117209087 + t2807];
        int t2809 = t2720 / 256;
        int t2810 = t2809 * 256;
        int t2811 = t2720 - t2810;
        float t2812 = memory[1534 + t2811];
        float t2813 = t2808 * t2812;
        float t2814 = t2790 + t2813;
        int t2815 = t2720 / 256;
        int t2816 = t2815 * 256;
        int t2817 = t2720 - t2816;
        int t2818 = t2815 * 256;
        int t2819 = t2818 + t2817;
        int t2820 = t2819 / 256;
        int t2821 = t2820 * 256;
        int t2822 = t2819 - t2821;
        int t2823 = t2822 / 256;
        int t2824 = t2823 * 256;
        int t2825 = t2822 - t2824;
        int t2826 = t2820 * 512;
        int t2827 = t2826 + t2825;
        int t2828 = i;
        int t2829 = t2828 * 1024;
        int t2830 = t2829 + t2827;
        float t2831 = memory[154957823 + t2830];
        float t2832 = t2831 + t2767;
        int t2833 = i;
        int t2834 = t2833 * 512;
        int t2835 = t2834 + t2720;
        memory[163870719 + t2835] = t2832;
        int t2837 = t2720 / 256;
        int t2838 = t2837 * 256;
        int t2839 = t2720 - t2838;
        int t2840 = t2837 * 256;
        int t2841 = t2840 + t2839;
        int t2842 = t2841 / 256;
        int t2843 = t2842 * 256;
        int t2844 = t2841 - t2843;
        int t2845 = t2844 / 256;
        int t2846 = t2845 * 256;
        int t2847 = t2844 - t2846;
        int t2848 = t2842 * 512;
        int t2849 = t2848 + t2847;
        int t2850 = i;
        int t2851 = t2850 * 1024;
        int t2852 = t2851 + t2849;
        float t2853 = memory[117209087 + t2852];
        float t2854 = t2853 + t2814;
        int t2855 = i;
        int t2856 = t2855 * 512;
        int t2857 = t2856 + t2720;
        memory[125073407 + t2857] = t2854;
        int t2859 = t2720 / 256;
        int t2860 = t2859 * 256;
        int t2861 = t2720 - t2860;
        int t2862 = t2859 * 256;
        int t2863 = t2862 + t2861;
        int t2864 = t2863 / 256;
        int t2865 = t2864 * 256;
        int t2866 = t2863 - t2865;
        int t2867 = t2866 / 256;
        int t2868 = t2867 * 256;
        int t2869 = t2866 - t2868;
        int t2870 = t2864 * 512;
        int t2871 = t2870 + t2869;
        int t2872 = i;
        int t2873 = t2872 * 1024;
        int t2874 = t2873 + t2871;
        float t2875 = memory[154957823 + t2874];
        float t2876 = t2875 - t2767;
        int t2877 = i;
        int t2878 = t2877 * 512;
        int t2879 = t2878 + t2720;
        memory[120354815 + t2879] = t2876;
        int t2881 = t2720 / 256;
        int t2882 = t2881 * 256;
        int t2883 = t2720 - t2882;
        int t2884 = t2881 * 256;
        int t2885 = t2884 + t2883;
        int t2886 = t2885 / 256;
        int t2887 = t2886 * 256;
        int t2888 = t2885 - t2887;
        int t2889 = t2888 / 256;
        int t2890 = t2889 * 256;
        int t2891 = t2888 - t2890;
        int t2892 = t2886 * 512;
        int t2893 = t2892 + t2891;
        int t2894 = i;
        int t2895 = t2894 * 1024;
        int t2896 = t2895 + t2893;
        float t2897 = memory[117209087 + t2896];
        float t2898 = t2897 - t2814;
        int t2899 = i;
        int t2900 = t2899 * 512;
        int t2901 = t2900 + t2720;
        memory[173570047 + t2901] = t2898;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 256)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (256, 0)]), value: empty) */
      }
      for (int t2903 = 0; t2903 < 1024; t2903++) {
        int t2904 = t2903 / 512;
        int t2905 = t2904 * 512;
        int t2906 = t2903 - t2905;
        int t2907 = t2904 >= 0;
        int t2908 = t2904 < 2;
        float t2909 = 1.0 * t2907;
        float t2910 = t2909 * t2908;
        int t2911 = t2904;
        int t2912 = t2906 >= 0;
        int t2913 = t2906 < 256;
        float t2914 = t2910 * t2912;
        float t2915 = t2914 * t2913;
        int t2916 = t2906;
        int t2917 = t2911 * 256;
        int t2918 = t2917 + t2916;
        float t2919 = 0.0;
        if (t2915) {
          int t2921 = i;
          int t2922 = t2921 * 512;
          int t2923 = t2922 + t2918;
          float t2924 = memory[163870719 + t2923];
          t2919 = t2924;
        }
        int t2926 = t2903 / 512;
        int t2927 = t2926 * 512;
        int t2928 = t2903 - t2927;
        int t2929 = t2926 >= 0;
        int t2930 = t2926 < 2;
        float t2931 = 1.0 * t2929;
        float t2932 = t2931 * t2930;
        int t2933 = t2926;
        int t2934 = t2928 >= 256;
        int t2935 = t2928 < 512;
        float t2936 = t2932 * t2934;
        float t2937 = t2936 * t2935;
        int t2938 = t2928 - 256;
        int t2939 = t2933 * 256;
        int t2940 = t2939 + t2938;
        float t2941 = 0.0;
        if (t2937) {
          int t2943 = i;
          int t2944 = t2943 * 512;
          int t2945 = t2944 + t2940;
          float t2946 = memory[120354815 + t2945];
          t2941 = t2946;
        }
        float t2948 = t2919 + t2941;
        int t2949 = i;
        int t2950 = t2949 * 1024;
        int t2951 = t2950 + t2903;
        memory[168065023 + t2951] = t2948;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 256)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (256, 0)]), value: empty) */
        int t2953 = t2903 / 512;
        int t2954 = t2953 * 512;
        int t2955 = t2903 - t2954;
        int t2956 = t2953 >= 0;
        int t2957 = t2953 < 2;
        float t2958 = 1.0 * t2956;
        float t2959 = t2958 * t2957;
        int t2960 = t2953;
        int t2961 = t2955 >= 0;
        int t2962 = t2955 < 256;
        float t2963 = t2959 * t2961;
        float t2964 = t2963 * t2962;
        int t2965 = t2955;
        int t2966 = t2960 * 256;
        int t2967 = t2966 + t2965;
        float t2968 = 0.0;
        if (t2964) {
          int t2970 = i;
          int t2971 = t2970 * 512;
          int t2972 = t2971 + t2967;
          float t2973 = memory[125073407 + t2972];
          t2968 = t2973;
        }
        int t2975 = t2903 / 512;
        int t2976 = t2975 * 512;
        int t2977 = t2903 - t2976;
        int t2978 = t2975 >= 0;
        int t2979 = t2975 < 2;
        float t2980 = 1.0 * t2978;
        float t2981 = t2980 * t2979;
        int t2982 = t2975;
        int t2983 = t2977 >= 256;
        int t2984 = t2977 < 512;
        float t2985 = t2981 * t2983;
        float t2986 = t2985 * t2984;
        int t2987 = t2977 - 256;
        int t2988 = t2982 * 256;
        int t2989 = t2988 + t2987;
        float t2990 = 0.0;
        if (t2986) {
          int t2992 = i;
          int t2993 = t2992 * 512;
          int t2994 = t2993 + t2989;
          float t2995 = memory[173570047 + t2994];
          t2990 = t2995;
        }
        float t2997 = t2968 + t2990;
        int t2998 = i;
        int t2999 = t2998 * 1024;
        int t3000 = t2999 + t2903;
        memory[134772735 + t3000] = t2997;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 2, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 2, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
      }
    }
  }
  for (int simd43 = 0; simd43 < 512; simd43+=4) {
  }
  for (int simd44 = 0; simd44 < 512; simd44+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([1, 512]), value: empty) */
  }
  for (int simd45 = 0; simd45 < 512; simd45+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([1, 512]), value: empty) */
      for (int t3005 = 0; t3005 < 512; t3005++) {
        int t3006 = t3005 / 512;
        int t3007 = t3006 * 512;
        int t3008 = t3005 - t3007;
        int t3009 = t3008 / 512;
        int t3010 = t3009 * 512;
        int t3011 = t3008 - t3010;
        int t3012 = t3011 / 512;
        int t3013 = t3012 * 512;
        int t3014 = t3011 - t3013;
        int t3015 = 512 + t3014;
        int t3016 = i;
        int t3017 = t3016 * 1024;
        int t3018 = t3017 + t3015;
        float t3019 = memory[168065023 + t3018];
        float t3020 = memory[2046 + (isfinite((int) t3005) ? (int) t3005 : 0)];
        float t3021 = t3019 * t3020;
        int t3022 = t3005 / 512;
        int t3023 = t3022 * 512;
        int t3024 = t3005 - t3023;
        int t3025 = t3024 / 512;
        int t3026 = t3025 * 512;
        int t3027 = t3024 - t3026;
        int t3028 = t3027 / 512;
        int t3029 = t3028 * 512;
        int t3030 = t3027 - t3029;
        int t3031 = 512 + t3030;
        int t3032 = i;
        int t3033 = t3032 * 1024;
        int t3034 = t3033 + t3031;
        float t3035 = memory[134772735 + t3034];
        float t3036 = memory[2558 + (isfinite((int) t3005) ? (int) t3005 : 0)];
        float t3037 = t3035 * t3036;
        float t3038 = t3021 - t3037;
        int t3039 = t3005 / 512;
        int t3040 = t3039 * 512;
        int t3041 = t3005 - t3040;
        int t3042 = t3041 / 512;
        int t3043 = t3042 * 512;
        int t3044 = t3041 - t3043;
        int t3045 = t3044 / 512;
        int t3046 = t3045 * 512;
        int t3047 = t3044 - t3046;
        int t3048 = 512 + t3047;
        int t3049 = i;
        int t3050 = t3049 * 1024;
        int t3051 = t3050 + t3048;
        float t3052 = memory[168065023 + t3051];
        float t3053 = memory[2558 + (isfinite((int) t3005) ? (int) t3005 : 0)];
        float t3054 = t3052 * t3053;
        int t3055 = t3005 / 512;
        int t3056 = t3055 * 512;
        int t3057 = t3005 - t3056;
        int t3058 = t3057 / 512;
        int t3059 = t3058 * 512;
        int t3060 = t3057 - t3059;
        int t3061 = t3060 / 512;
        int t3062 = t3061 * 512;
        int t3063 = t3060 - t3062;
        int t3064 = 512 + t3063;
        int t3065 = i;
        int t3066 = t3065 * 1024;
        int t3067 = t3066 + t3064;
        float t3068 = memory[134772735 + t3067];
        float t3069 = memory[2046 + (isfinite((int) t3005) ? (int) t3005 : 0)];
        float t3070 = t3068 * t3069;
        float t3071 = t3054 + t3070;
        int t3072 = t3005 / 512;
        int t3073 = t3072 * 512;
        int t3074 = t3005 - t3073;
        int t3075 = t3074 / 512;
        int t3076 = t3075 * 512;
        int t3077 = t3074 - t3076;
        int t3078 = t3077 / 512;
        int t3079 = t3078 * 512;
        int t3080 = t3077 - t3079;
        int t3081 = i;
        int t3082 = t3081 * 1024;
        int t3083 = t3082 + t3080;
        float t3084 = memory[168065023 + t3083];
        float t3085 = t3084 + t3038;
        int t3086 = i;
        int t3087 = t3086 * 512;
        int t3088 = t3087 + t3005;
        memory[142899199 + t3088] = t3085;
        int t3090 = t3005 / 512;
        int t3091 = t3090 * 512;
        int t3092 = t3005 - t3091;
        int t3093 = t3092 / 512;
        int t3094 = t3093 * 512;
        int t3095 = t3092 - t3094;
        int t3096 = t3095 / 512;
        int t3097 = t3096 * 512;
        int t3098 = t3095 - t3097;
        int t3099 = i;
        int t3100 = t3099 * 1024;
        int t3101 = t3100 + t3098;
        float t3102 = memory[134772735 + t3101];
        float t3103 = t3102 + t3071;
        int t3104 = i;
        int t3105 = t3104 * 512;
        int t3106 = t3105 + t3005;
        memory[175929343 + t3106] = t3103;
        int t3108 = t3005 / 512;
        int t3109 = t3108 * 512;
        int t3110 = t3005 - t3109;
        int t3111 = t3110 / 512;
        int t3112 = t3111 * 512;
        int t3113 = t3110 - t3112;
        int t3114 = t3113 / 512;
        int t3115 = t3114 * 512;
        int t3116 = t3113 - t3115;
        int t3117 = i;
        int t3118 = t3117 * 1024;
        int t3119 = t3118 + t3116;
        float t3120 = memory[168065023 + t3119];
        float t3121 = t3120 - t3038;
        int t3122 = i;
        int t3123 = t3122 * 512;
        int t3124 = t3123 + t3005;
        memory[140802047 + t3124] = t3121;
        int t3126 = t3005 / 512;
        int t3127 = t3126 * 512;
        int t3128 = t3005 - t3127;
        int t3129 = t3128 / 512;
        int t3130 = t3129 * 512;
        int t3131 = t3128 - t3130;
        int t3132 = t3131 / 512;
        int t3133 = t3132 * 512;
        int t3134 = t3131 - t3133;
        int t3135 = i;
        int t3136 = t3135 * 1024;
        int t3137 = t3136 + t3134;
        float t3138 = memory[134772735 + t3137];
        float t3139 = t3138 - t3071;
        int t3140 = i;
        int t3141 = t3140 * 512;
        int t3142 = t3141 + t3005;
        memory[131364863 + t3142] = t3139;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 512)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (512, 0)]), value: empty) */
      }
      for (int t3144 = 0; t3144 < 1024; t3144++) {
        int t3145 = t3144 / 1024;
        int t3146 = t3145 * 1024;
        int t3147 = t3144 - t3146;
        int t3148 = t3145 >= 0;
        int t3149 = t3145 < 1;
        float t3150 = 1.0 * t3148;
        float t3151 = t3150 * t3149;
        int t3152 = t3145;
        int t3153 = t3147 >= 0;
        int t3154 = t3147 < 512;
        float t3155 = t3151 * t3153;
        float t3156 = t3155 * t3154;
        int t3157 = t3147;
        int t3158 = t3152 * 512;
        int t3159 = t3158 + t3157;
        float t3160 = 0.0;
        if (t3156) {
          int t3162 = i;
          int t3163 = t3162 * 512;
          int t3164 = t3163 + t3159;
          float t3165 = memory[142899199 + t3164];
          t3160 = t3165;
        }
        int t3167 = t3144 / 1024;
        int t3168 = t3167 * 1024;
        int t3169 = t3144 - t3168;
        int t3170 = t3167 >= 0;
        int t3171 = t3167 < 1;
        float t3172 = 1.0 * t3170;
        float t3173 = t3172 * t3171;
        int t3174 = t3167;
        int t3175 = t3169 >= 512;
        int t3176 = t3169 < 1024;
        float t3177 = t3173 * t3175;
        float t3178 = t3177 * t3176;
        int t3179 = t3169 - 512;
        int t3180 = t3174 * 512;
        int t3181 = t3180 + t3179;
        float t3182 = 0.0;
        if (t3178) {
          int t3184 = i;
          int t3185 = t3184 * 512;
          int t3186 = t3185 + t3181;
          float t3187 = memory[140802047 + t3186];
          t3182 = t3187;
        }
        float t3189 = t3160 + t3182;
        int t3190 = i;
        int t3191 = t3190 * 1024;
        int t3192 = t3191 + t3144;
        memory[170948607 + t3192] = t3189;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 512)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (512, 0)]), value: empty) */
        int t3194 = t3144 / 1024;
        int t3195 = t3194 * 1024;
        int t3196 = t3144 - t3195;
        int t3197 = t3194 >= 0;
        int t3198 = t3194 < 1;
        float t3199 = 1.0 * t3197;
        float t3200 = t3199 * t3198;
        int t3201 = t3194;
        int t3202 = t3196 >= 0;
        int t3203 = t3196 < 512;
        float t3204 = t3200 * t3202;
        float t3205 = t3204 * t3203;
        int t3206 = t3196;
        int t3207 = t3201 * 512;
        int t3208 = t3207 + t3206;
        float t3209 = 0.0;
        if (t3205) {
          int t3211 = i;
          int t3212 = t3211 * 512;
          int t3213 = t3212 + t3208;
          float t3214 = memory[175929343 + t3213];
          t3209 = t3214;
        }
        int t3216 = t3144 / 1024;
        int t3217 = t3216 * 1024;
        int t3218 = t3144 - t3217;
        int t3219 = t3216 >= 0;
        int t3220 = t3216 < 1;
        float t3221 = 1.0 * t3219;
        float t3222 = t3221 * t3220;
        int t3223 = t3216;
        int t3224 = t3218 >= 512;
        int t3225 = t3218 < 1024;
        float t3226 = t3222 * t3224;
        float t3227 = t3226 * t3225;
        int t3228 = t3218 - 512;
        int t3229 = t3223 * 512;
        int t3230 = t3229 + t3228;
        float t3231 = 0.0;
        if (t3227) {
          int t3233 = i;
          int t3234 = t3233 * 512;
          int t3235 = t3234 + t3230;
          float t3236 = memory[131364863 + t3235];
          t3231 = t3236;
        }
        float t3238 = t3209 + t3231;
        int t3239 = i;
        int t3240 = t3239 * 1024;
        int t3241 = t3240 + t3144;
        memory[158365695 + t3241] = t3238;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mtranspose [0m([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
      }
    }
  }
  for (int simd47 = 0; simd47 < 1024; simd47+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      for (int simd48 = 0; simd48 < 1024; simd48+=4) {
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
      }
    }
  }
  for (int t49 = 0; t49 < 1; t49+=1) {
  }
  for (int simd50 = 0; simd50 < 512; simd50+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
  }
  for (int t51 = 0; t51 < 1; t51+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
      for (int t3248 = 0; t3248 < 512; t3248++) {
        int t3249 = t3248;
        int t3250 = t3249;
        int t3251 = t3248 - t3250;
        int t3252 = t3249;
        int t3253 = t3252;
        int t3254 = t3251;
        int t3255 = t3253 + t3254;
        int t3256 = t3255;
        int t3257 = t3256;
        int t3258 = t3255 - t3257;
        int t3259 = t3258;
        int t3260 = t3259;
        int t3261 = t3258 - t3260;
        int t3262 = t3259 + 1;
        int t3263 = t3256 * 2;
        int t3264 = t3263;
        int t3265 = t3262;
        int t3266 = t3264 + t3265;
        int t3267 = t3261;
        int t3268 = t3266 + t3267;
        int t3269 = t3268;
        int t3270 = t3269;
        int t3271 = t3270 / 512;
        int t3272 = t3271 * 512;
        int t3273 = t3270 - t3272;
        int t3274 = t3273 / 256;
        int t3275 = t3274 * 256;
        int t3276 = t3273 - t3275;
        int t3277 = t3276 / 128;
        int t3278 = t3277 * 128;
        int t3279 = t3276 - t3278;
        int t3280 = t3279 / 64;
        int t3281 = t3280 * 64;
        int t3282 = t3279 - t3281;
        int t3283 = t3282 / 32;
        int t3284 = t3283 * 32;
        int t3285 = t3282 - t3284;
        int t3286 = t3285 / 16;
        int t3287 = t3286 * 16;
        int t3288 = t3285 - t3287;
        int t3289 = t3288 / 8;
        int t3290 = t3289 * 8;
        int t3291 = t3288 - t3290;
        int t3292 = t3291 / 4;
        int t3293 = t3292 * 4;
        int t3294 = t3291 - t3293;
        int t3295 = t3294 / 2;
        int t3296 = t3295 * 2;
        int t3297 = t3294 - t3296;
        int t3298 = t3297 * 512;
        int t3299 = t3298;
        int t3300 = t3295 * 256;
        int t3301 = t3299 + t3300;
        int t3302 = t3292 * 128;
        int t3303 = t3301 + t3302;
        int t3304 = t3289 * 64;
        int t3305 = t3303 + t3304;
        int t3306 = t3286 * 32;
        int t3307 = t3305 + t3306;
        int t3308 = t3283 * 16;
        int t3309 = t3307 + t3308;
        int t3310 = t3280 * 8;
        int t3311 = t3309 + t3310;
        int t3312 = t3277 * 4;
        int t3313 = t3311 + t3312;
        int t3314 = t3274 * 2;
        int t3315 = t3313 + t3314;
        int t3316 = t3271;
        int t3317 = t3315 + t3316;
        int t3318 = t3317 / 1024;
        int t3319 = t3318 * 1024;
        int t3320 = t3317 - t3319;
        float t3321 = (int)t188[i];
        int t3322 = t3321 - 1024;
        int t3323 = t3322 + 1;
        int t3324 = t3323 + t3320;
        int t3325 = t3324 + 5119;
        int t3326 = t3325 % 5119;
        int t3327 = t3318 * 5119;
        int t3328 = t3327 + t3326;
        float t3329 = memory[112222208 + t3328];
        int t3330 = t3248;
        int t3331 = t3330;
        int t3332 = t3248 - t3331;
        float t3333 = memory[4094 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t3334 = t3329 * t3333;
        int t3335 = t3248;
        int t3336 = t3335;
        int t3337 = t3248 - t3336;
        int t3338 = t3335;
        int t3339 = t3338;
        int t3340 = t3335 - t3339;
        int t3341 = t3340;
        int t3342 = t3341;
        int t3343 = t3340 - t3342;
        int t3344 = t3338 * 2;
        int t3345 = 1 + t3344;
        float t3346 = memory[3070 + t3345];
        int t3347 = t3248;
        int t3348 = t3347;
        int t3349 = t3248 - t3348;
        float t3350 = memory[4095 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t3351 = t3346 * t3350;
        float t3352 = t3334 - t3351;
        int t3353 = t3248;
        int t3354 = t3353;
        int t3355 = t3248 - t3354;
        int t3356 = t3353;
        int t3357 = t3356;
        int t3358 = t3355;
        int t3359 = t3357 + t3358;
        int t3360 = t3359;
        int t3361 = t3360;
        int t3362 = t3359 - t3361;
        int t3363 = t3362;
        int t3364 = t3363;
        int t3365 = t3362 - t3364;
        int t3366 = t3363 + 1;
        int t3367 = t3360 * 2;
        int t3368 = t3367;
        int t3369 = t3366;
        int t3370 = t3368 + t3369;
        int t3371 = t3365;
        int t3372 = t3370 + t3371;
        int t3373 = t3372;
        int t3374 = t3373;
        int t3375 = t3374 / 512;
        int t3376 = t3375 * 512;
        int t3377 = t3374 - t3376;
        int t3378 = t3377 / 256;
        int t3379 = t3378 * 256;
        int t3380 = t3377 - t3379;
        int t3381 = t3380 / 128;
        int t3382 = t3381 * 128;
        int t3383 = t3380 - t3382;
        int t3384 = t3383 / 64;
        int t3385 = t3384 * 64;
        int t3386 = t3383 - t3385;
        int t3387 = t3386 / 32;
        int t3388 = t3387 * 32;
        int t3389 = t3386 - t3388;
        int t3390 = t3389 / 16;
        int t3391 = t3390 * 16;
        int t3392 = t3389 - t3391;
        int t3393 = t3392 / 8;
        int t3394 = t3393 * 8;
        int t3395 = t3392 - t3394;
        int t3396 = t3395 / 4;
        int t3397 = t3396 * 4;
        int t3398 = t3395 - t3397;
        int t3399 = t3398 / 2;
        int t3400 = t3399 * 2;
        int t3401 = t3398 - t3400;
        int t3402 = t3401 * 512;
        int t3403 = t3402;
        int t3404 = t3399 * 256;
        int t3405 = t3403 + t3404;
        int t3406 = t3396 * 128;
        int t3407 = t3405 + t3406;
        int t3408 = t3393 * 64;
        int t3409 = t3407 + t3408;
        int t3410 = t3390 * 32;
        int t3411 = t3409 + t3410;
        int t3412 = t3387 * 16;
        int t3413 = t3411 + t3412;
        int t3414 = t3384 * 8;
        int t3415 = t3413 + t3414;
        int t3416 = t3381 * 4;
        int t3417 = t3415 + t3416;
        int t3418 = t3378 * 2;
        int t3419 = t3417 + t3418;
        int t3420 = t3375;
        int t3421 = t3419 + t3420;
        int t3422 = t3421 / 1024;
        int t3423 = t3422 * 1024;
        int t3424 = t3421 - t3423;
        float t3425 = (int)t188[i];
        int t3426 = t3425 - 1024;
        int t3427 = t3426 + 1;
        int t3428 = t3427 + t3424;
        int t3429 = t3428 + 5119;
        int t3430 = t3429 % 5119;
        int t3431 = t3422 * 5119;
        int t3432 = t3431 + t3430;
        float t3433 = memory[112222208 + t3432];
        int t3434 = t3248;
        int t3435 = t3434;
        int t3436 = t3248 - t3435;
        float t3437 = memory[4095 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t3438 = t3433 * t3437;
        int t3439 = t3248;
        int t3440 = t3439;
        int t3441 = t3248 - t3440;
        int t3442 = t3439;
        int t3443 = t3442;
        int t3444 = t3439 - t3443;
        int t3445 = t3444;
        int t3446 = t3445;
        int t3447 = t3444 - t3446;
        int t3448 = t3442 * 2;
        int t3449 = 1 + t3448;
        float t3450 = memory[3070 + t3449];
        int t3451 = t3248;
        int t3452 = t3451;
        int t3453 = t3248 - t3452;
        float t3454 = memory[4094 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t3455 = t3450 * t3454;
        float t3456 = t3438 + t3455;
        int t3457 = t3248;
        int t3458 = t3457;
        int t3459 = t3248 - t3458;
        int t3460 = t3457;
        int t3461 = t3460;
        int t3462 = t3459;
        int t3463 = t3461 + t3462;
        int t3464 = t3463;
        int t3465 = t3464;
        int t3466 = t3463 - t3465;
        int t3467 = t3466;
        int t3468 = t3467;
        int t3469 = t3466 - t3468;
        int t3470 = t3467;
        int t3471 = t3464 * 2;
        int t3472 = t3471;
        int t3473 = t3470;
        int t3474 = t3472 + t3473;
        int t3475 = t3469;
        int t3476 = t3474 + t3475;
        int t3477 = t3476;
        int t3478 = t3477;
        int t3479 = t3478 / 512;
        int t3480 = t3479 * 512;
        int t3481 = t3478 - t3480;
        int t3482 = t3481 / 256;
        int t3483 = t3482 * 256;
        int t3484 = t3481 - t3483;
        int t3485 = t3484 / 128;
        int t3486 = t3485 * 128;
        int t3487 = t3484 - t3486;
        int t3488 = t3487 / 64;
        int t3489 = t3488 * 64;
        int t3490 = t3487 - t3489;
        int t3491 = t3490 / 32;
        int t3492 = t3491 * 32;
        int t3493 = t3490 - t3492;
        int t3494 = t3493 / 16;
        int t3495 = t3494 * 16;
        int t3496 = t3493 - t3495;
        int t3497 = t3496 / 8;
        int t3498 = t3497 * 8;
        int t3499 = t3496 - t3498;
        int t3500 = t3499 / 4;
        int t3501 = t3500 * 4;
        int t3502 = t3499 - t3501;
        int t3503 = t3502 / 2;
        int t3504 = t3503 * 2;
        int t3505 = t3502 - t3504;
        int t3506 = t3505 * 512;
        int t3507 = t3506;
        int t3508 = t3503 * 256;
        int t3509 = t3507 + t3508;
        int t3510 = t3500 * 128;
        int t3511 = t3509 + t3510;
        int t3512 = t3497 * 64;
        int t3513 = t3511 + t3512;
        int t3514 = t3494 * 32;
        int t3515 = t3513 + t3514;
        int t3516 = t3491 * 16;
        int t3517 = t3515 + t3516;
        int t3518 = t3488 * 8;
        int t3519 = t3517 + t3518;
        int t3520 = t3485 * 4;
        int t3521 = t3519 + t3520;
        int t3522 = t3482 * 2;
        int t3523 = t3521 + t3522;
        int t3524 = t3479;
        int t3525 = t3523 + t3524;
        int t3526 = t3525 / 1024;
        int t3527 = t3526 * 1024;
        int t3528 = t3525 - t3527;
        float t3529 = (int)t188[i];
        int t3530 = t3529 - 1024;
        int t3531 = t3530 + 1;
        int t3532 = t3531 + t3528;
        int t3533 = t3532 + 5119;
        int t3534 = t3533 % 5119;
        int t3535 = t3526 * 5119;
        int t3536 = t3535 + t3534;
        float t3537 = memory[112222208 + t3536];
        float t3538 = t3537 + t3352;
        int t3539 = i;
        int t3540 = t3539 * 512;
        int t3541 = t3540 + t3248;
        memory[118781951 + t3541] = t3538;
        int t3543 = t3248;
        int t3544 = t3543;
        int t3545 = t3248 - t3544;
        int t3546 = t3543;
        int t3547 = t3546;
        int t3548 = t3543 - t3547;
        int t3549 = t3548;
        int t3550 = t3549;
        int t3551 = t3548 - t3550;
        int t3552 = t3546 * 2;
        float t3553 = memory[3070 + t3552];
        float t3554 = t3553 + t3456;
        int t3555 = i;
        int t3556 = t3555 * 512;
        int t3557 = t3556 + t3248;
        memory[142112767 + t3557] = t3554;
        int t3559 = t3248;
        int t3560 = t3559;
        int t3561 = t3248 - t3560;
        int t3562 = t3559;
        int t3563 = t3562;
        int t3564 = t3561;
        int t3565 = t3563 + t3564;
        int t3566 = t3565;
        int t3567 = t3566;
        int t3568 = t3565 - t3567;
        int t3569 = t3568;
        int t3570 = t3569;
        int t3571 = t3568 - t3570;
        int t3572 = t3569;
        int t3573 = t3566 * 2;
        int t3574 = t3573;
        int t3575 = t3572;
        int t3576 = t3574 + t3575;
        int t3577 = t3571;
        int t3578 = t3576 + t3577;
        int t3579 = t3578;
        int t3580 = t3579;
        int t3581 = t3580 / 512;
        int t3582 = t3581 * 512;
        int t3583 = t3580 - t3582;
        int t3584 = t3583 / 256;
        int t3585 = t3584 * 256;
        int t3586 = t3583 - t3585;
        int t3587 = t3586 / 128;
        int t3588 = t3587 * 128;
        int t3589 = t3586 - t3588;
        int t3590 = t3589 / 64;
        int t3591 = t3590 * 64;
        int t3592 = t3589 - t3591;
        int t3593 = t3592 / 32;
        int t3594 = t3593 * 32;
        int t3595 = t3592 - t3594;
        int t3596 = t3595 / 16;
        int t3597 = t3596 * 16;
        int t3598 = t3595 - t3597;
        int t3599 = t3598 / 8;
        int t3600 = t3599 * 8;
        int t3601 = t3598 - t3600;
        int t3602 = t3601 / 4;
        int t3603 = t3602 * 4;
        int t3604 = t3601 - t3603;
        int t3605 = t3604 / 2;
        int t3606 = t3605 * 2;
        int t3607 = t3604 - t3606;
        int t3608 = t3607 * 512;
        int t3609 = t3608;
        int t3610 = t3605 * 256;
        int t3611 = t3609 + t3610;
        int t3612 = t3602 * 128;
        int t3613 = t3611 + t3612;
        int t3614 = t3599 * 64;
        int t3615 = t3613 + t3614;
        int t3616 = t3596 * 32;
        int t3617 = t3615 + t3616;
        int t3618 = t3593 * 16;
        int t3619 = t3617 + t3618;
        int t3620 = t3590 * 8;
        int t3621 = t3619 + t3620;
        int t3622 = t3587 * 4;
        int t3623 = t3621 + t3622;
        int t3624 = t3584 * 2;
        int t3625 = t3623 + t3624;
        int t3626 = t3581;
        int t3627 = t3625 + t3626;
        int t3628 = t3627 / 1024;
        int t3629 = t3628 * 1024;
        int t3630 = t3627 - t3629;
        float t3631 = (int)t188[i];
        int t3632 = t3631 - 1024;
        int t3633 = t3632 + 1;
        int t3634 = t3633 + t3630;
        int t3635 = t3634 + 5119;
        int t3636 = t3635 % 5119;
        int t3637 = t3628 * 5119;
        int t3638 = t3637 + t3636;
        float t3639 = memory[112222208 + t3638];
        float t3640 = t3639 - t3352;
        int t3641 = i;
        int t3642 = t3641 * 512;
        int t3643 = t3642 + t3248;
        memory[139229183 + t3643] = t3640;
        int t3645 = t3248;
        int t3646 = t3645;
        int t3647 = t3248 - t3646;
        int t3648 = t3645;
        int t3649 = t3648;
        int t3650 = t3645 - t3649;
        int t3651 = t3650;
        int t3652 = t3651;
        int t3653 = t3650 - t3652;
        int t3654 = t3648 * 2;
        float t3655 = memory[3070 + t3654];
        float t3656 = t3655 - t3456;
        int t3657 = i;
        int t3658 = t3657 * 512;
        int t3659 = t3658 + t3248;
        memory[166754303 + t3659] = t3656;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
      }
      for (int t3661 = 0; t3661 < 1024; t3661++) {
        int t3662 = t3661 / 2;
        int t3663 = t3662 * 2;
        int t3664 = t3661 - t3663;
        int t3665 = t3662 >= 0;
        int t3666 = t3662 < 512;
        float t3667 = 1.0 * t3665;
        float t3668 = t3667 * t3666;
        int t3669 = t3662;
        int t3670 = t3664 >= 0;
        int t3671 = t3664 < 1;
        float t3672 = t3668 * t3670;
        float t3673 = t3672 * t3671;
        int t3674 = t3664;
        int t3675 = t3669 + t3674;
        float t3676 = 0.0;
        if (t3673) {
          int t3678 = i;
          int t3679 = t3678 * 512;
          int t3680 = t3679 + t3675;
          float t3681 = memory[118781951 + t3680];
          t3676 = t3681;
        }
        int t3683 = t3661 / 2;
        int t3684 = t3683 * 2;
        int t3685 = t3661 - t3684;
        int t3686 = t3683 >= 0;
        int t3687 = t3683 < 512;
        float t3688 = 1.0 * t3686;
        float t3689 = t3688 * t3687;
        int t3690 = t3683;
        int t3691 = t3685 >= 1;
        int t3692 = t3685 < 2;
        float t3693 = t3689 * t3691;
        float t3694 = t3693 * t3692;
        int t3695 = t3685 - 1;
        int t3696 = t3690 + t3695;
        float t3697 = 0.0;
        if (t3694) {
          int t3699 = i;
          int t3700 = t3699 * 512;
          int t3701 = t3700 + t3696;
          float t3702 = memory[139229183 + t3701];
          t3697 = t3702;
        }
        float t3704 = t3676 + t3697;
        int t3705 = i;
        int t3706 = t3705 * 1024;
        int t3707 = t3706 + t3661;
        memory[152860671 + t3707] = t3704;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
        int t3709 = t3661 / 2;
        int t3710 = t3709 * 2;
        int t3711 = t3661 - t3710;
        int t3712 = t3709 >= 0;
        int t3713 = t3709 < 512;
        float t3714 = 1.0 * t3712;
        float t3715 = t3714 * t3713;
        int t3716 = t3709;
        int t3717 = t3711 >= 0;
        int t3718 = t3711 < 1;
        float t3719 = t3715 * t3717;
        float t3720 = t3719 * t3718;
        int t3721 = t3711;
        int t3722 = t3716 + t3721;
        float t3723 = 0.0;
        if (t3720) {
          int t3725 = i;
          int t3726 = t3725 * 512;
          int t3727 = t3726 + t3722;
          float t3728 = memory[142112767 + t3727];
          t3723 = t3728;
        }
        int t3730 = t3661 / 2;
        int t3731 = t3730 * 2;
        int t3732 = t3661 - t3731;
        int t3733 = t3730 >= 0;
        int t3734 = t3730 < 512;
        float t3735 = 1.0 * t3733;
        float t3736 = t3735 * t3734;
        int t3737 = t3730;
        int t3738 = t3732 >= 1;
        int t3739 = t3732 < 2;
        float t3740 = t3736 * t3738;
        float t3741 = t3740 * t3739;
        int t3742 = t3732 - 1;
        int t3743 = t3737 + t3742;
        float t3744 = 0.0;
        if (t3741) {
          int t3746 = i;
          int t3747 = t3746 * 512;
          int t3748 = t3747 + t3743;
          float t3749 = memory[166754303 + t3748];
          t3744 = t3749;
        }
        float t3751 = t3723 + t3744;
        int t3752 = i;
        int t3753 = t3752 * 1024;
        int t3754 = t3753 + t3661;
        memory[164657151 + t3754] = t3751;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
      }
    }
  }
  for (int t53 = 0; t53 < 2; t53+=1) {
  }
  for (int simd54 = 0; simd54 < 512; simd54+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
  }
  for (int t55 = 0; t55 < 2; t55+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
      for (int t3759 = 0; t3759 < 512; t3759++) {
        int t3760 = t3759 / 2;
        int t3761 = t3760 * 2;
        int t3762 = t3759 - t3761;
        int t3763 = t3760 * 2;
        int t3764 = t3763 + t3762;
        int t3765 = t3764 / 2;
        int t3766 = t3765 * 2;
        int t3767 = t3764 - t3766;
        int t3768 = t3767 / 2;
        int t3769 = t3768 * 2;
        int t3770 = t3767 - t3769;
        int t3771 = t3765 * 4;
        int t3772 = 2 + t3771;
        int t3773 = t3772 + t3770;
        int t3774 = i;
        int t3775 = t3774 * 1024;
        int t3776 = t3775 + t3773;
        float t3777 = memory[152860671 + t3776];
        int t3778 = t3759 / 2;
        int t3779 = t3778 * 2;
        int t3780 = t3759 - t3779;
        float t3781 = memory[4096 + t3780];
        float t3782 = t3777 * t3781;
        int t3783 = t3759 / 2;
        int t3784 = t3783 * 2;
        int t3785 = t3759 - t3784;
        int t3786 = t3783 * 2;
        int t3787 = t3786 + t3785;
        int t3788 = t3787 / 2;
        int t3789 = t3788 * 2;
        int t3790 = t3787 - t3789;
        int t3791 = t3790 / 2;
        int t3792 = t3791 * 2;
        int t3793 = t3790 - t3792;
        int t3794 = t3788 * 4;
        int t3795 = 2 + t3794;
        int t3796 = t3795 + t3793;
        int t3797 = i;
        int t3798 = t3797 * 1024;
        int t3799 = t3798 + t3796;
        float t3800 = memory[164657151 + t3799];
        int t3801 = t3759 / 2;
        int t3802 = t3801 * 2;
        int t3803 = t3759 - t3802;
        float t3804 = memory[4098 + t3803];
        float t3805 = t3800 * t3804;
        float t3806 = t3782 - t3805;
        int t3807 = t3759 / 2;
        int t3808 = t3807 * 2;
        int t3809 = t3759 - t3808;
        int t3810 = t3807 * 2;
        int t3811 = t3810 + t3809;
        int t3812 = t3811 / 2;
        int t3813 = t3812 * 2;
        int t3814 = t3811 - t3813;
        int t3815 = t3814 / 2;
        int t3816 = t3815 * 2;
        int t3817 = t3814 - t3816;
        int t3818 = t3812 * 4;
        int t3819 = 2 + t3818;
        int t3820 = t3819 + t3817;
        int t3821 = i;
        int t3822 = t3821 * 1024;
        int t3823 = t3822 + t3820;
        float t3824 = memory[152860671 + t3823];
        int t3825 = t3759 / 2;
        int t3826 = t3825 * 2;
        int t3827 = t3759 - t3826;
        float t3828 = memory[4098 + t3827];
        float t3829 = t3824 * t3828;
        int t3830 = t3759 / 2;
        int t3831 = t3830 * 2;
        int t3832 = t3759 - t3831;
        int t3833 = t3830 * 2;
        int t3834 = t3833 + t3832;
        int t3835 = t3834 / 2;
        int t3836 = t3835 * 2;
        int t3837 = t3834 - t3836;
        int t3838 = t3837 / 2;
        int t3839 = t3838 * 2;
        int t3840 = t3837 - t3839;
        int t3841 = t3835 * 4;
        int t3842 = 2 + t3841;
        int t3843 = t3842 + t3840;
        int t3844 = i;
        int t3845 = t3844 * 1024;
        int t3846 = t3845 + t3843;
        float t3847 = memory[164657151 + t3846];
        int t3848 = t3759 / 2;
        int t3849 = t3848 * 2;
        int t3850 = t3759 - t3849;
        float t3851 = memory[4096 + t3850];
        float t3852 = t3847 * t3851;
        float t3853 = t3829 + t3852;
        int t3854 = t3759 / 2;
        int t3855 = t3854 * 2;
        int t3856 = t3759 - t3855;
        int t3857 = t3854 * 2;
        int t3858 = t3857 + t3856;
        int t3859 = t3858 / 2;
        int t3860 = t3859 * 2;
        int t3861 = t3858 - t3860;
        int t3862 = t3861 / 2;
        int t3863 = t3862 * 2;
        int t3864 = t3861 - t3863;
        int t3865 = t3859 * 4;
        int t3866 = t3865 + t3864;
        int t3867 = i;
        int t3868 = t3867 * 1024;
        int t3869 = t3868 + t3866;
        float t3870 = memory[152860671 + t3869];
        float t3871 = t3870 + t3806;
        int t3872 = i;
        int t3873 = t3872 * 512;
        int t3874 = t3873 + t3759;
        memory[123238399 + t3874] = t3871;
        int t3876 = t3759 / 2;
        int t3877 = t3876 * 2;
        int t3878 = t3759 - t3877;
        int t3879 = t3876 * 2;
        int t3880 = t3879 + t3878;
        int t3881 = t3880 / 2;
        int t3882 = t3881 * 2;
        int t3883 = t3880 - t3882;
        int t3884 = t3883 / 2;
        int t3885 = t3884 * 2;
        int t3886 = t3883 - t3885;
        int t3887 = t3881 * 4;
        int t3888 = t3887 + t3886;
        int t3889 = i;
        int t3890 = t3889 * 1024;
        int t3891 = t3890 + t3888;
        float t3892 = memory[164657151 + t3891];
        float t3893 = t3892 + t3853;
        int t3894 = i;
        int t3895 = t3894 * 512;
        int t3896 = t3895 + t3759;
        memory[120616959 + t3896] = t3893;
        int t3898 = t3759 / 2;
        int t3899 = t3898 * 2;
        int t3900 = t3759 - t3899;
        int t3901 = t3898 * 2;
        int t3902 = t3901 + t3900;
        int t3903 = t3902 / 2;
        int t3904 = t3903 * 2;
        int t3905 = t3902 - t3904;
        int t3906 = t3905 / 2;
        int t3907 = t3906 * 2;
        int t3908 = t3905 - t3907;
        int t3909 = t3903 * 4;
        int t3910 = t3909 + t3908;
        int t3911 = i;
        int t3912 = t3911 * 1024;
        int t3913 = t3912 + t3910;
        float t3914 = memory[152860671 + t3913];
        float t3915 = t3914 - t3806;
        int t3916 = i;
        int t3917 = t3916 * 512;
        int t3918 = t3917 + t3759;
        memory[152336383 + t3918] = t3915;
        int t3920 = t3759 / 2;
        int t3921 = t3920 * 2;
        int t3922 = t3759 - t3921;
        int t3923 = t3920 * 2;
        int t3924 = t3923 + t3922;
        int t3925 = t3924 / 2;
        int t3926 = t3925 * 2;
        int t3927 = t3924 - t3926;
        int t3928 = t3927 / 2;
        int t3929 = t3928 * 2;
        int t3930 = t3927 - t3929;
        int t3931 = t3925 * 4;
        int t3932 = t3931 + t3930;
        int t3933 = i;
        int t3934 = t3933 * 1024;
        int t3935 = t3934 + t3932;
        float t3936 = memory[164657151 + t3935];
        float t3937 = t3936 - t3853;
        int t3938 = i;
        int t3939 = t3938 * 512;
        int t3940 = t3939 + t3759;
        memory[129267711 + t3940] = t3937;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
      }
      for (int t3942 = 0; t3942 < 1024; t3942++) {
        int t3943 = t3942 / 4;
        int t3944 = t3943 * 4;
        int t3945 = t3942 - t3944;
        int t3946 = t3943 >= 0;
        int t3947 = t3943 < 256;
        float t3948 = 1.0 * t3946;
        float t3949 = t3948 * t3947;
        int t3950 = t3943;
        int t3951 = t3945 >= 0;
        int t3952 = t3945 < 2;
        float t3953 = t3949 * t3951;
        float t3954 = t3953 * t3952;
        int t3955 = t3945;
        int t3956 = t3950 * 2;
        int t3957 = t3956 + t3955;
        float t3958 = 0.0;
        if (t3954) {
          int t3960 = i;
          int t3961 = t3960 * 512;
          int t3962 = t3961 + t3957;
          float t3963 = memory[123238399 + t3962];
          t3958 = t3963;
        }
        int t3965 = t3942 / 4;
        int t3966 = t3965 * 4;
        int t3967 = t3942 - t3966;
        int t3968 = t3965 >= 0;
        int t3969 = t3965 < 256;
        float t3970 = 1.0 * t3968;
        float t3971 = t3970 * t3969;
        int t3972 = t3965;
        int t3973 = t3967 >= 2;
        int t3974 = t3967 < 4;
        float t3975 = t3971 * t3973;
        float t3976 = t3975 * t3974;
        int t3977 = t3967 - 2;
        int t3978 = t3972 * 2;
        int t3979 = t3978 + t3977;
        float t3980 = 0.0;
        if (t3976) {
          int t3982 = i;
          int t3983 = t3982 * 512;
          int t3984 = t3983 + t3979;
          float t3985 = memory[152336383 + t3984];
          t3980 = t3985;
        }
        float t3987 = t3958 + t3980;
        int t3988 = i;
        int t3989 = t3988 * 1024;
        int t3990 = t3989 + t3942;
        memory[165705727 + t3990] = t3987;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
        int t3992 = t3942 / 4;
        int t3993 = t3992 * 4;
        int t3994 = t3942 - t3993;
        int t3995 = t3992 >= 0;
        int t3996 = t3992 < 256;
        float t3997 = 1.0 * t3995;
        float t3998 = t3997 * t3996;
        int t3999 = t3992;
        int t4000 = t3994 >= 0;
        int t4001 = t3994 < 2;
        float t4002 = t3998 * t4000;
        float t4003 = t4002 * t4001;
        int t4004 = t3994;
        int t4005 = t3999 * 2;
        int t4006 = t4005 + t4004;
        float t4007 = 0.0;
        if (t4003) {
          int t4009 = i;
          int t4010 = t4009 * 512;
          int t4011 = t4010 + t4006;
          float t4012 = memory[120616959 + t4011];
          t4007 = t4012;
        }
        int t4014 = t3942 / 4;
        int t4015 = t4014 * 4;
        int t4016 = t3942 - t4015;
        int t4017 = t4014 >= 0;
        int t4018 = t4014 < 256;
        float t4019 = 1.0 * t4017;
        float t4020 = t4019 * t4018;
        int t4021 = t4014;
        int t4022 = t4016 >= 2;
        int t4023 = t4016 < 4;
        float t4024 = t4020 * t4022;
        float t4025 = t4024 * t4023;
        int t4026 = t4016 - 2;
        int t4027 = t4021 * 2;
        int t4028 = t4027 + t4026;
        float t4029 = 0.0;
        if (t4025) {
          int t4031 = i;
          int t4032 = t4031 * 512;
          int t4033 = t4032 + t4028;
          float t4034 = memory[129267711 + t4033];
          t4029 = t4034;
        }
        float t4036 = t4007 + t4029;
        int t4037 = i;
        int t4038 = t4037 * 1024;
        int t4039 = t4038 + t3942;
        memory[115636223 + t4039] = t4036;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
      }
    }
  }
  for (int simd57 = 0; simd57 < 4; simd57+=4) {
  }
  for (int simd58 = 0; simd58 < 512; simd58+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
  }
  for (int simd59 = 0; simd59 < 4; simd59+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
      for (int t4044 = 0; t4044 < 512; t4044++) {
        int t4045 = t4044 / 4;
        int t4046 = t4045 * 4;
        int t4047 = t4044 - t4046;
        int t4048 = t4045 * 4;
        int t4049 = t4048 + t4047;
        int t4050 = t4049 / 4;
        int t4051 = t4050 * 4;
        int t4052 = t4049 - t4051;
        int t4053 = t4052 / 4;
        int t4054 = t4053 * 4;
        int t4055 = t4052 - t4054;
        int t4056 = t4050 * 8;
        int t4057 = 4 + t4056;
        int t4058 = t4057 + t4055;
        int t4059 = i;
        int t4060 = t4059 * 1024;
        int t4061 = t4060 + t4058;
        float t4062 = memory[165705727 + t4061];
        int t4063 = t4044 / 4;
        int t4064 = t4063 * 4;
        int t4065 = t4044 - t4064;
        float t4066 = memory[4100 + t4065];
        float t4067 = t4062 * t4066;
        int t4068 = t4044 / 4;
        int t4069 = t4068 * 4;
        int t4070 = t4044 - t4069;
        int t4071 = t4068 * 4;
        int t4072 = t4071 + t4070;
        int t4073 = t4072 / 4;
        int t4074 = t4073 * 4;
        int t4075 = t4072 - t4074;
        int t4076 = t4075 / 4;
        int t4077 = t4076 * 4;
        int t4078 = t4075 - t4077;
        int t4079 = t4073 * 8;
        int t4080 = 4 + t4079;
        int t4081 = t4080 + t4078;
        int t4082 = i;
        int t4083 = t4082 * 1024;
        int t4084 = t4083 + t4081;
        float t4085 = memory[115636223 + t4084];
        int t4086 = t4044 / 4;
        int t4087 = t4086 * 4;
        int t4088 = t4044 - t4087;
        float t4089 = memory[4104 + t4088];
        float t4090 = t4085 * t4089;
        float t4091 = t4067 - t4090;
        int t4092 = t4044 / 4;
        int t4093 = t4092 * 4;
        int t4094 = t4044 - t4093;
        int t4095 = t4092 * 4;
        int t4096 = t4095 + t4094;
        int t4097 = t4096 / 4;
        int t4098 = t4097 * 4;
        int t4099 = t4096 - t4098;
        int t4100 = t4099 / 4;
        int t4101 = t4100 * 4;
        int t4102 = t4099 - t4101;
        int t4103 = t4097 * 8;
        int t4104 = 4 + t4103;
        int t4105 = t4104 + t4102;
        int t4106 = i;
        int t4107 = t4106 * 1024;
        int t4108 = t4107 + t4105;
        float t4109 = memory[165705727 + t4108];
        int t4110 = t4044 / 4;
        int t4111 = t4110 * 4;
        int t4112 = t4044 - t4111;
        float t4113 = memory[4104 + t4112];
        float t4114 = t4109 * t4113;
        int t4115 = t4044 / 4;
        int t4116 = t4115 * 4;
        int t4117 = t4044 - t4116;
        int t4118 = t4115 * 4;
        int t4119 = t4118 + t4117;
        int t4120 = t4119 / 4;
        int t4121 = t4120 * 4;
        int t4122 = t4119 - t4121;
        int t4123 = t4122 / 4;
        int t4124 = t4123 * 4;
        int t4125 = t4122 - t4124;
        int t4126 = t4120 * 8;
        int t4127 = 4 + t4126;
        int t4128 = t4127 + t4125;
        int t4129 = i;
        int t4130 = t4129 * 1024;
        int t4131 = t4130 + t4128;
        float t4132 = memory[115636223 + t4131];
        int t4133 = t4044 / 4;
        int t4134 = t4133 * 4;
        int t4135 = t4044 - t4134;
        float t4136 = memory[4100 + t4135];
        float t4137 = t4132 * t4136;
        float t4138 = t4114 + t4137;
        int t4139 = t4044 / 4;
        int t4140 = t4139 * 4;
        int t4141 = t4044 - t4140;
        int t4142 = t4139 * 4;
        int t4143 = t4142 + t4141;
        int t4144 = t4143 / 4;
        int t4145 = t4144 * 4;
        int t4146 = t4143 - t4145;
        int t4147 = t4146 / 4;
        int t4148 = t4147 * 4;
        int t4149 = t4146 - t4148;
        int t4150 = t4144 * 8;
        int t4151 = t4150 + t4149;
        int t4152 = i;
        int t4153 = t4152 * 1024;
        int t4154 = t4153 + t4151;
        float t4155 = memory[165705727 + t4154];
        float t4156 = t4155 + t4091;
        int t4157 = i;
        int t4158 = t4157 * 512;
        int t4159 = t4158 + t4044;
        memory[162559999 + t4159] = t4156;
        int t4161 = t4044 / 4;
        int t4162 = t4161 * 4;
        int t4163 = t4044 - t4162;
        int t4164 = t4161 * 4;
        int t4165 = t4164 + t4163;
        int t4166 = t4165 / 4;
        int t4167 = t4166 * 4;
        int t4168 = t4165 - t4167;
        int t4169 = t4168 / 4;
        int t4170 = t4169 * 4;
        int t4171 = t4168 - t4170;
        int t4172 = t4166 * 8;
        int t4173 = t4172 + t4171;
        int t4174 = i;
        int t4175 = t4174 * 1024;
        int t4176 = t4175 + t4173;
        float t4177 = memory[115636223 + t4176];
        float t4178 = t4177 + t4138;
        int t4179 = i;
        int t4180 = t4179 * 512;
        int t4181 = t4180 + t4044;
        memory[123500543 + t4181] = t4178;
        int t4183 = t4044 / 4;
        int t4184 = t4183 * 4;
        int t4185 = t4044 - t4184;
        int t4186 = t4183 * 4;
        int t4187 = t4186 + t4185;
        int t4188 = t4187 / 4;
        int t4189 = t4188 * 4;
        int t4190 = t4187 - t4189;
        int t4191 = t4190 / 4;
        int t4192 = t4191 * 4;
        int t4193 = t4190 - t4192;
        int t4194 = t4188 * 8;
        int t4195 = t4194 + t4193;
        int t4196 = i;
        int t4197 = t4196 * 1024;
        int t4198 = t4197 + t4195;
        float t4199 = memory[165705727 + t4198];
        float t4200 = t4199 - t4091;
        int t4201 = i;
        int t4202 = t4201 * 512;
        int t4203 = t4202 + t4044;
        memory[173832191 + t4203] = t4200;
        int t4205 = t4044 / 4;
        int t4206 = t4205 * 4;
        int t4207 = t4044 - t4206;
        int t4208 = t4205 * 4;
        int t4209 = t4208 + t4207;
        int t4210 = t4209 / 4;
        int t4211 = t4210 * 4;
        int t4212 = t4209 - t4211;
        int t4213 = t4212 / 4;
        int t4214 = t4213 * 4;
        int t4215 = t4212 - t4214;
        int t4216 = t4210 * 8;
        int t4217 = t4216 + t4215;
        int t4218 = i;
        int t4219 = t4218 * 1024;
        int t4220 = t4219 + t4217;
        float t4221 = memory[115636223 + t4220];
        float t4222 = t4221 - t4138;
        int t4223 = i;
        int t4224 = t4223 * 512;
        int t4225 = t4224 + t4044;
        memory[168589311 + t4225] = t4222;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
      }
      for (int t4227 = 0; t4227 < 1024; t4227++) {
        int t4228 = t4227 / 8;
        int t4229 = t4228 * 8;
        int t4230 = t4227 - t4229;
        int t4231 = t4228 >= 0;
        int t4232 = t4228 < 128;
        float t4233 = 1.0 * t4231;
        float t4234 = t4233 * t4232;
        int t4235 = t4228;
        int t4236 = t4230 >= 0;
        int t4237 = t4230 < 4;
        float t4238 = t4234 * t4236;
        float t4239 = t4238 * t4237;
        int t4240 = t4230;
        int t4241 = t4235 * 4;
        int t4242 = t4241 + t4240;
        float t4243 = 0.0;
        if (t4239) {
          int t4245 = i;
          int t4246 = t4245 * 512;
          int t4247 = t4246 + t4242;
          float t4248 = memory[162559999 + t4247];
          t4243 = t4248;
        }
        int t4250 = t4227 / 8;
        int t4251 = t4250 * 8;
        int t4252 = t4227 - t4251;
        int t4253 = t4250 >= 0;
        int t4254 = t4250 < 128;
        float t4255 = 1.0 * t4253;
        float t4256 = t4255 * t4254;
        int t4257 = t4250;
        int t4258 = t4252 >= 4;
        int t4259 = t4252 < 8;
        float t4260 = t4256 * t4258;
        float t4261 = t4260 * t4259;
        int t4262 = t4252 - 4;
        int t4263 = t4257 * 4;
        int t4264 = t4263 + t4262;
        float t4265 = 0.0;
        if (t4261) {
          int t4267 = i;
          int t4268 = t4267 * 512;
          int t4269 = t4268 + t4264;
          float t4270 = memory[173832191 + t4269];
          t4265 = t4270;
        }
        float t4272 = t4243 + t4265;
        int t4273 = i;
        int t4274 = t4273 * 1024;
        int t4275 = t4274 + t4227;
        memory[140015615 + t4275] = t4272;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
        int t4277 = t4227 / 8;
        int t4278 = t4277 * 8;
        int t4279 = t4227 - t4278;
        int t4280 = t4277 >= 0;
        int t4281 = t4277 < 128;
        float t4282 = 1.0 * t4280;
        float t4283 = t4282 * t4281;
        int t4284 = t4277;
        int t4285 = t4279 >= 0;
        int t4286 = t4279 < 4;
        float t4287 = t4283 * t4285;
        float t4288 = t4287 * t4286;
        int t4289 = t4279;
        int t4290 = t4284 * 4;
        int t4291 = t4290 + t4289;
        float t4292 = 0.0;
        if (t4288) {
          int t4294 = i;
          int t4295 = t4294 * 512;
          int t4296 = t4295 + t4291;
          float t4297 = memory[123500543 + t4296];
          t4292 = t4297;
        }
        int t4299 = t4227 / 8;
        int t4300 = t4299 * 8;
        int t4301 = t4227 - t4300;
        int t4302 = t4299 >= 0;
        int t4303 = t4299 < 128;
        float t4304 = 1.0 * t4302;
        float t4305 = t4304 * t4303;
        int t4306 = t4299;
        int t4307 = t4301 >= 4;
        int t4308 = t4301 < 8;
        float t4309 = t4305 * t4307;
        float t4310 = t4309 * t4308;
        int t4311 = t4301 - 4;
        int t4312 = t4306 * 4;
        int t4313 = t4312 + t4311;
        float t4314 = 0.0;
        if (t4310) {
          int t4316 = i;
          int t4317 = t4316 * 512;
          int t4318 = t4317 + t4313;
          float t4319 = memory[168589311 + t4318];
          t4314 = t4319;
        }
        float t4321 = t4292 + t4314;
        int t4322 = i;
        int t4323 = t4322 * 1024;
        int t4324 = t4323 + t4227;
        memory[166230015 + t4324] = t4321;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
      }
    }
  }
  for (int simd61 = 0; simd61 < 8; simd61+=4) {
  }
  for (int simd62 = 0; simd62 < 512; simd62+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
  }
  for (int simd63 = 0; simd63 < 8; simd63+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
      for (int t4329 = 0; t4329 < 512; t4329++) {
        int t4330 = t4329 / 8;
        int t4331 = t4330 * 8;
        int t4332 = t4329 - t4331;
        int t4333 = t4330 * 8;
        int t4334 = t4333 + t4332;
        int t4335 = t4334 / 8;
        int t4336 = t4335 * 8;
        int t4337 = t4334 - t4336;
        int t4338 = t4337 / 8;
        int t4339 = t4338 * 8;
        int t4340 = t4337 - t4339;
        int t4341 = t4335 * 16;
        int t4342 = 8 + t4341;
        int t4343 = t4342 + t4340;
        int t4344 = i;
        int t4345 = t4344 * 1024;
        int t4346 = t4345 + t4343;
        float t4347 = memory[140015615 + t4346];
        int t4348 = t4329 / 8;
        int t4349 = t4348 * 8;
        int t4350 = t4329 - t4349;
        float t4351 = memory[4108 + t4350];
        float t4352 = t4347 * t4351;
        int t4353 = t4329 / 8;
        int t4354 = t4353 * 8;
        int t4355 = t4329 - t4354;
        int t4356 = t4353 * 8;
        int t4357 = t4356 + t4355;
        int t4358 = t4357 / 8;
        int t4359 = t4358 * 8;
        int t4360 = t4357 - t4359;
        int t4361 = t4360 / 8;
        int t4362 = t4361 * 8;
        int t4363 = t4360 - t4362;
        int t4364 = t4358 * 16;
        int t4365 = 8 + t4364;
        int t4366 = t4365 + t4363;
        int t4367 = i;
        int t4368 = t4367 * 1024;
        int t4369 = t4368 + t4366;
        float t4370 = memory[166230015 + t4369];
        int t4371 = t4329 / 8;
        int t4372 = t4371 * 8;
        int t4373 = t4329 - t4372;
        float t4374 = memory[4116 + t4373];
        float t4375 = t4370 * t4374;
        float t4376 = t4352 - t4375;
        int t4377 = t4329 / 8;
        int t4378 = t4377 * 8;
        int t4379 = t4329 - t4378;
        int t4380 = t4377 * 8;
        int t4381 = t4380 + t4379;
        int t4382 = t4381 / 8;
        int t4383 = t4382 * 8;
        int t4384 = t4381 - t4383;
        int t4385 = t4384 / 8;
        int t4386 = t4385 * 8;
        int t4387 = t4384 - t4386;
        int t4388 = t4382 * 16;
        int t4389 = 8 + t4388;
        int t4390 = t4389 + t4387;
        int t4391 = i;
        int t4392 = t4391 * 1024;
        int t4393 = t4392 + t4390;
        float t4394 = memory[140015615 + t4393];
        int t4395 = t4329 / 8;
        int t4396 = t4395 * 8;
        int t4397 = t4329 - t4396;
        float t4398 = memory[4116 + t4397];
        float t4399 = t4394 * t4398;
        int t4400 = t4329 / 8;
        int t4401 = t4400 * 8;
        int t4402 = t4329 - t4401;
        int t4403 = t4400 * 8;
        int t4404 = t4403 + t4402;
        int t4405 = t4404 / 8;
        int t4406 = t4405 * 8;
        int t4407 = t4404 - t4406;
        int t4408 = t4407 / 8;
        int t4409 = t4408 * 8;
        int t4410 = t4407 - t4409;
        int t4411 = t4405 * 16;
        int t4412 = 8 + t4411;
        int t4413 = t4412 + t4410;
        int t4414 = i;
        int t4415 = t4414 * 1024;
        int t4416 = t4415 + t4413;
        float t4417 = memory[166230015 + t4416];
        int t4418 = t4329 / 8;
        int t4419 = t4418 * 8;
        int t4420 = t4329 - t4419;
        float t4421 = memory[4108 + t4420];
        float t4422 = t4417 * t4421;
        float t4423 = t4399 + t4422;
        int t4424 = t4329 / 8;
        int t4425 = t4424 * 8;
        int t4426 = t4329 - t4425;
        int t4427 = t4424 * 8;
        int t4428 = t4427 + t4426;
        int t4429 = t4428 / 8;
        int t4430 = t4429 * 8;
        int t4431 = t4428 - t4430;
        int t4432 = t4431 / 8;
        int t4433 = t4432 * 8;
        int t4434 = t4431 - t4433;
        int t4435 = t4429 * 16;
        int t4436 = t4435 + t4434;
        int t4437 = i;
        int t4438 = t4437 * 1024;
        int t4439 = t4438 + t4436;
        float t4440 = memory[140015615 + t4439];
        float t4441 = t4440 + t4376;
        int t4442 = i;
        int t4443 = t4442 * 512;
        int t4444 = t4443 + t4329;
        memory[169113599 + t4444] = t4441;
        int t4446 = t4329 / 8;
        int t4447 = t4446 * 8;
        int t4448 = t4329 - t4447;
        int t4449 = t4446 * 8;
        int t4450 = t4449 + t4448;
        int t4451 = t4450 / 8;
        int t4452 = t4451 * 8;
        int t4453 = t4450 - t4452;
        int t4454 = t4453 / 8;
        int t4455 = t4454 * 8;
        int t4456 = t4453 - t4455;
        int t4457 = t4451 * 16;
        int t4458 = t4457 + t4456;
        int t4459 = i;
        int t4460 = t4459 * 1024;
        int t4461 = t4460 + t4458;
        float t4462 = memory[166230015 + t4461];
        float t4463 = t4462 + t4423;
        int t4464 = i;
        int t4465 = t4464 * 512;
        int t4466 = t4465 + t4329;
        memory[173307903 + t4466] = t4463;
        int t4468 = t4329 / 8;
        int t4469 = t4468 * 8;
        int t4470 = t4329 - t4469;
        int t4471 = t4468 * 8;
        int t4472 = t4471 + t4470;
        int t4473 = t4472 / 8;
        int t4474 = t4473 * 8;
        int t4475 = t4472 - t4474;
        int t4476 = t4475 / 8;
        int t4477 = t4476 * 8;
        int t4478 = t4475 - t4477;
        int t4479 = t4473 * 16;
        int t4480 = t4479 + t4478;
        int t4481 = i;
        int t4482 = t4481 * 1024;
        int t4483 = t4482 + t4480;
        float t4484 = memory[140015615 + t4483];
        float t4485 = t4484 - t4376;
        int t4486 = i;
        int t4487 = t4486 * 512;
        int t4488 = t4487 + t4329;
        memory[132937727 + t4488] = t4485;
        int t4490 = t4329 / 8;
        int t4491 = t4490 * 8;
        int t4492 = t4329 - t4491;
        int t4493 = t4490 * 8;
        int t4494 = t4493 + t4492;
        int t4495 = t4494 / 8;
        int t4496 = t4495 * 8;
        int t4497 = t4494 - t4496;
        int t4498 = t4497 / 8;
        int t4499 = t4498 * 8;
        int t4500 = t4497 - t4499;
        int t4501 = t4495 * 16;
        int t4502 = t4501 + t4500;
        int t4503 = i;
        int t4504 = t4503 * 1024;
        int t4505 = t4504 + t4502;
        float t4506 = memory[166230015 + t4505];
        float t4507 = t4506 - t4423;
        int t4508 = i;
        int t4509 = t4508 * 512;
        int t4510 = t4509 + t4329;
        memory[113539071 + t4510] = t4507;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
      }
      for (int t4512 = 0; t4512 < 1024; t4512++) {
        int t4513 = t4512 / 16;
        int t4514 = t4513 * 16;
        int t4515 = t4512 - t4514;
        int t4516 = t4513 >= 0;
        int t4517 = t4513 < 64;
        float t4518 = 1.0 * t4516;
        float t4519 = t4518 * t4517;
        int t4520 = t4513;
        int t4521 = t4515 >= 0;
        int t4522 = t4515 < 8;
        float t4523 = t4519 * t4521;
        float t4524 = t4523 * t4522;
        int t4525 = t4515;
        int t4526 = t4520 * 8;
        int t4527 = t4526 + t4525;
        float t4528 = 0.0;
        if (t4524) {
          int t4530 = i;
          int t4531 = t4530 * 512;
          int t4532 = t4531 + t4527;
          float t4533 = memory[169113599 + t4532];
          t4528 = t4533;
        }
        int t4535 = t4512 / 16;
        int t4536 = t4535 * 16;
        int t4537 = t4512 - t4536;
        int t4538 = t4535 >= 0;
        int t4539 = t4535 < 64;
        float t4540 = 1.0 * t4538;
        float t4541 = t4540 * t4539;
        int t4542 = t4535;
        int t4543 = t4537 >= 8;
        int t4544 = t4537 < 16;
        float t4545 = t4541 * t4543;
        float t4546 = t4545 * t4544;
        int t4547 = t4537 - 8;
        int t4548 = t4542 * 8;
        int t4549 = t4548 + t4547;
        float t4550 = 0.0;
        if (t4546) {
          int t4552 = i;
          int t4553 = t4552 * 512;
          int t4554 = t4553 + t4549;
          float t4555 = memory[132937727 + t4554];
          t4550 = t4555;
        }
        float t4557 = t4528 + t4550;
        int t4558 = i;
        int t4559 = t4558 * 1024;
        int t4560 = t4559 + t4512;
        memory[124286975 + t4560] = t4557;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
        int t4562 = t4512 / 16;
        int t4563 = t4562 * 16;
        int t4564 = t4512 - t4563;
        int t4565 = t4562 >= 0;
        int t4566 = t4562 < 64;
        float t4567 = 1.0 * t4565;
        float t4568 = t4567 * t4566;
        int t4569 = t4562;
        int t4570 = t4564 >= 0;
        int t4571 = t4564 < 8;
        float t4572 = t4568 * t4570;
        float t4573 = t4572 * t4571;
        int t4574 = t4564;
        int t4575 = t4569 * 8;
        int t4576 = t4575 + t4574;
        float t4577 = 0.0;
        if (t4573) {
          int t4579 = i;
          int t4580 = t4579 * 512;
          int t4581 = t4580 + t4576;
          float t4582 = memory[173307903 + t4581];
          t4577 = t4582;
        }
        int t4584 = t4512 / 16;
        int t4585 = t4584 * 16;
        int t4586 = t4512 - t4585;
        int t4587 = t4584 >= 0;
        int t4588 = t4584 < 64;
        float t4589 = 1.0 * t4587;
        float t4590 = t4589 * t4588;
        int t4591 = t4584;
        int t4592 = t4586 >= 8;
        int t4593 = t4586 < 16;
        float t4594 = t4590 * t4592;
        float t4595 = t4594 * t4593;
        int t4596 = t4586 - 8;
        int t4597 = t4591 * 8;
        int t4598 = t4597 + t4596;
        float t4599 = 0.0;
        if (t4595) {
          int t4601 = i;
          int t4602 = t4601 * 512;
          int t4603 = t4602 + t4598;
          float t4604 = memory[113539071 + t4603];
          t4599 = t4604;
        }
        float t4606 = t4577 + t4599;
        int t4607 = i;
        int t4608 = t4607 * 1024;
        int t4609 = t4608 + t4512;
        memory[144472063 + t4609] = t4606;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
      }
    }
  }
  for (int simd65 = 0; simd65 < 16; simd65+=4) {
  }
  for (int simd66 = 0; simd66 < 512; simd66+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
  }
  for (int simd67 = 0; simd67 < 16; simd67+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
      for (int t4614 = 0; t4614 < 512; t4614++) {
        int t4615 = t4614 / 16;
        int t4616 = t4615 * 16;
        int t4617 = t4614 - t4616;
        int t4618 = t4615 * 16;
        int t4619 = t4618 + t4617;
        int t4620 = t4619 / 16;
        int t4621 = t4620 * 16;
        int t4622 = t4619 - t4621;
        int t4623 = t4622 / 16;
        int t4624 = t4623 * 16;
        int t4625 = t4622 - t4624;
        int t4626 = t4620 * 32;
        int t4627 = 16 + t4626;
        int t4628 = t4627 + t4625;
        int t4629 = i;
        int t4630 = t4629 * 1024;
        int t4631 = t4630 + t4628;
        float t4632 = memory[124286975 + t4631];
        int t4633 = t4614 / 16;
        int t4634 = t4633 * 16;
        int t4635 = t4614 - t4634;
        float t4636 = memory[4124 + t4635];
        float t4637 = t4632 * t4636;
        int t4638 = t4614 / 16;
        int t4639 = t4638 * 16;
        int t4640 = t4614 - t4639;
        int t4641 = t4638 * 16;
        int t4642 = t4641 + t4640;
        int t4643 = t4642 / 16;
        int t4644 = t4643 * 16;
        int t4645 = t4642 - t4644;
        int t4646 = t4645 / 16;
        int t4647 = t4646 * 16;
        int t4648 = t4645 - t4647;
        int t4649 = t4643 * 32;
        int t4650 = 16 + t4649;
        int t4651 = t4650 + t4648;
        int t4652 = i;
        int t4653 = t4652 * 1024;
        int t4654 = t4653 + t4651;
        float t4655 = memory[144472063 + t4654];
        int t4656 = t4614 / 16;
        int t4657 = t4656 * 16;
        int t4658 = t4614 - t4657;
        float t4659 = memory[4140 + t4658];
        float t4660 = t4655 * t4659;
        float t4661 = t4637 - t4660;
        int t4662 = t4614 / 16;
        int t4663 = t4662 * 16;
        int t4664 = t4614 - t4663;
        int t4665 = t4662 * 16;
        int t4666 = t4665 + t4664;
        int t4667 = t4666 / 16;
        int t4668 = t4667 * 16;
        int t4669 = t4666 - t4668;
        int t4670 = t4669 / 16;
        int t4671 = t4670 * 16;
        int t4672 = t4669 - t4671;
        int t4673 = t4667 * 32;
        int t4674 = 16 + t4673;
        int t4675 = t4674 + t4672;
        int t4676 = i;
        int t4677 = t4676 * 1024;
        int t4678 = t4677 + t4675;
        float t4679 = memory[124286975 + t4678];
        int t4680 = t4614 / 16;
        int t4681 = t4680 * 16;
        int t4682 = t4614 - t4681;
        float t4683 = memory[4140 + t4682];
        float t4684 = t4679 * t4683;
        int t4685 = t4614 / 16;
        int t4686 = t4685 * 16;
        int t4687 = t4614 - t4686;
        int t4688 = t4685 * 16;
        int t4689 = t4688 + t4687;
        int t4690 = t4689 / 16;
        int t4691 = t4690 * 16;
        int t4692 = t4689 - t4691;
        int t4693 = t4692 / 16;
        int t4694 = t4693 * 16;
        int t4695 = t4692 - t4694;
        int t4696 = t4690 * 32;
        int t4697 = 16 + t4696;
        int t4698 = t4697 + t4695;
        int t4699 = i;
        int t4700 = t4699 * 1024;
        int t4701 = t4700 + t4698;
        float t4702 = memory[144472063 + t4701];
        int t4703 = t4614 / 16;
        int t4704 = t4703 * 16;
        int t4705 = t4614 - t4704;
        float t4706 = memory[4124 + t4705];
        float t4707 = t4702 * t4706;
        float t4708 = t4684 + t4707;
        int t4709 = t4614 / 16;
        int t4710 = t4709 * 16;
        int t4711 = t4614 - t4710;
        int t4712 = t4709 * 16;
        int t4713 = t4712 + t4711;
        int t4714 = t4713 / 16;
        int t4715 = t4714 * 16;
        int t4716 = t4713 - t4715;
        int t4717 = t4716 / 16;
        int t4718 = t4717 * 16;
        int t4719 = t4716 - t4718;
        int t4720 = t4714 * 32;
        int t4721 = t4720 + t4719;
        int t4722 = i;
        int t4723 = t4722 * 1024;
        int t4724 = t4723 + t4721;
        float t4725 = memory[124286975 + t4724];
        float t4726 = t4725 + t4661;
        int t4727 = i;
        int t4728 = t4727 * 512;
        int t4729 = t4728 + t4614;
        memory[153909247 + t4729] = t4726;
        int t4731 = t4614 / 16;
        int t4732 = t4731 * 16;
        int t4733 = t4614 - t4732;
        int t4734 = t4731 * 16;
        int t4735 = t4734 + t4733;
        int t4736 = t4735 / 16;
        int t4737 = t4736 * 16;
        int t4738 = t4735 - t4737;
        int t4739 = t4738 / 16;
        int t4740 = t4739 * 16;
        int t4741 = t4738 - t4740;
        int t4742 = t4736 * 32;
        int t4743 = t4742 + t4741;
        int t4744 = i;
        int t4745 = t4744 * 1024;
        int t4746 = t4745 + t4743;
        float t4747 = memory[144472063 + t4746];
        float t4748 = t4747 + t4708;
        int t4749 = i;
        int t4750 = t4749 * 512;
        int t4751 = t4750 + t4614;
        memory[113014783 + t4751] = t4748;
        int t4753 = t4614 / 16;
        int t4754 = t4753 * 16;
        int t4755 = t4614 - t4754;
        int t4756 = t4753 * 16;
        int t4757 = t4756 + t4755;
        int t4758 = t4757 / 16;
        int t4759 = t4758 * 16;
        int t4760 = t4757 - t4759;
        int t4761 = t4760 / 16;
        int t4762 = t4761 * 16;
        int t4763 = t4760 - t4762;
        int t4764 = t4758 * 32;
        int t4765 = t4764 + t4763;
        int t4766 = i;
        int t4767 = t4766 * 1024;
        int t4768 = t4767 + t4765;
        float t4769 = memory[124286975 + t4768];
        float t4770 = t4769 - t4661;
        int t4771 = i;
        int t4772 = t4771 * 512;
        int t4773 = t4772 + t4614;
        memory[153384959 + t4773] = t4770;
        int t4775 = t4614 / 16;
        int t4776 = t4775 * 16;
        int t4777 = t4614 - t4776;
        int t4778 = t4775 * 16;
        int t4779 = t4778 + t4777;
        int t4780 = t4779 / 16;
        int t4781 = t4780 * 16;
        int t4782 = t4779 - t4781;
        int t4783 = t4782 / 16;
        int t4784 = t4783 * 16;
        int t4785 = t4782 - t4784;
        int t4786 = t4780 * 32;
        int t4787 = t4786 + t4785;
        int t4788 = i;
        int t4789 = t4788 * 1024;
        int t4790 = t4789 + t4787;
        float t4791 = memory[144472063 + t4790];
        float t4792 = t4791 - t4708;
        int t4793 = i;
        int t4794 = t4793 * 512;
        int t4795 = t4794 + t4614;
        memory[147617791 + t4795] = t4792;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
      }
      for (int t4797 = 0; t4797 < 1024; t4797++) {
        int t4798 = t4797 / 32;
        int t4799 = t4798 * 32;
        int t4800 = t4797 - t4799;
        int t4801 = t4798 >= 0;
        int t4802 = t4798 < 32;
        float t4803 = 1.0 * t4801;
        float t4804 = t4803 * t4802;
        int t4805 = t4798;
        int t4806 = t4800 >= 0;
        int t4807 = t4800 < 16;
        float t4808 = t4804 * t4806;
        float t4809 = t4808 * t4807;
        int t4810 = t4800;
        int t4811 = t4805 * 16;
        int t4812 = t4811 + t4810;
        float t4813 = 0.0;
        if (t4809) {
          int t4815 = i;
          int t4816 = t4815 * 512;
          int t4817 = t4816 + t4812;
          float t4818 = memory[153909247 + t4817];
          t4813 = t4818;
        }
        int t4820 = t4797 / 32;
        int t4821 = t4820 * 32;
        int t4822 = t4797 - t4821;
        int t4823 = t4820 >= 0;
        int t4824 = t4820 < 32;
        float t4825 = 1.0 * t4823;
        float t4826 = t4825 * t4824;
        int t4827 = t4820;
        int t4828 = t4822 >= 16;
        int t4829 = t4822 < 32;
        float t4830 = t4826 * t4828;
        float t4831 = t4830 * t4829;
        int t4832 = t4822 - 16;
        int t4833 = t4827 * 16;
        int t4834 = t4833 + t4832;
        float t4835 = 0.0;
        if (t4831) {
          int t4837 = i;
          int t4838 = t4837 * 512;
          int t4839 = t4838 + t4834;
          float t4840 = memory[153384959 + t4839];
          t4835 = t4840;
        }
        float t4842 = t4813 + t4835;
        int t4843 = i;
        int t4844 = t4843 * 1024;
        int t4845 = t4844 + t4797;
        memory[158889983 + t4845] = t4842;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
        int t4847 = t4797 / 32;
        int t4848 = t4847 * 32;
        int t4849 = t4797 - t4848;
        int t4850 = t4847 >= 0;
        int t4851 = t4847 < 32;
        float t4852 = 1.0 * t4850;
        float t4853 = t4852 * t4851;
        int t4854 = t4847;
        int t4855 = t4849 >= 0;
        int t4856 = t4849 < 16;
        float t4857 = t4853 * t4855;
        float t4858 = t4857 * t4856;
        int t4859 = t4849;
        int t4860 = t4854 * 16;
        int t4861 = t4860 + t4859;
        float t4862 = 0.0;
        if (t4858) {
          int t4864 = i;
          int t4865 = t4864 * 512;
          int t4866 = t4865 + t4861;
          float t4867 = memory[113014783 + t4866];
          t4862 = t4867;
        }
        int t4869 = t4797 / 32;
        int t4870 = t4869 * 32;
        int t4871 = t4797 - t4870;
        int t4872 = t4869 >= 0;
        int t4873 = t4869 < 32;
        float t4874 = 1.0 * t4872;
        float t4875 = t4874 * t4873;
        int t4876 = t4869;
        int t4877 = t4871 >= 16;
        int t4878 = t4871 < 32;
        float t4879 = t4875 * t4877;
        float t4880 = t4879 * t4878;
        int t4881 = t4871 - 16;
        int t4882 = t4876 * 16;
        int t4883 = t4882 + t4881;
        float t4884 = 0.0;
        if (t4880) {
          int t4886 = i;
          int t4887 = t4886 * 512;
          int t4888 = t4887 + t4883;
          float t4889 = memory[147617791 + t4888];
          t4884 = t4889;
        }
        float t4891 = t4862 + t4884;
        int t4892 = i;
        int t4893 = t4892 * 1024;
        int t4894 = t4893 + t4797;
        memory[157317119 + t4894] = t4891;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
      }
    }
  }
  for (int simd69 = 0; simd69 < 32; simd69+=4) {
  }
  for (int simd70 = 0; simd70 < 512; simd70+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
  }
  for (int simd71 = 0; simd71 < 32; simd71+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
      for (int t4899 = 0; t4899 < 512; t4899++) {
        int t4900 = t4899 / 32;
        int t4901 = t4900 * 32;
        int t4902 = t4899 - t4901;
        int t4903 = t4900 * 32;
        int t4904 = t4903 + t4902;
        int t4905 = t4904 / 32;
        int t4906 = t4905 * 32;
        int t4907 = t4904 - t4906;
        int t4908 = t4907 / 32;
        int t4909 = t4908 * 32;
        int t4910 = t4907 - t4909;
        int t4911 = t4905 * 64;
        int t4912 = 32 + t4911;
        int t4913 = t4912 + t4910;
        int t4914 = i;
        int t4915 = t4914 * 1024;
        int t4916 = t4915 + t4913;
        float t4917 = memory[158889983 + t4916];
        int t4918 = t4899 / 32;
        int t4919 = t4918 * 32;
        int t4920 = t4899 - t4919;
        float t4921 = memory[4156 + t4920];
        float t4922 = t4917 * t4921;
        int t4923 = t4899 / 32;
        int t4924 = t4923 * 32;
        int t4925 = t4899 - t4924;
        int t4926 = t4923 * 32;
        int t4927 = t4926 + t4925;
        int t4928 = t4927 / 32;
        int t4929 = t4928 * 32;
        int t4930 = t4927 - t4929;
        int t4931 = t4930 / 32;
        int t4932 = t4931 * 32;
        int t4933 = t4930 - t4932;
        int t4934 = t4928 * 64;
        int t4935 = 32 + t4934;
        int t4936 = t4935 + t4933;
        int t4937 = i;
        int t4938 = t4937 * 1024;
        int t4939 = t4938 + t4936;
        float t4940 = memory[157317119 + t4939];
        int t4941 = t4899 / 32;
        int t4942 = t4941 * 32;
        int t4943 = t4899 - t4942;
        float t4944 = memory[4188 + t4943];
        float t4945 = t4940 * t4944;
        float t4946 = t4922 - t4945;
        int t4947 = t4899 / 32;
        int t4948 = t4947 * 32;
        int t4949 = t4899 - t4948;
        int t4950 = t4947 * 32;
        int t4951 = t4950 + t4949;
        int t4952 = t4951 / 32;
        int t4953 = t4952 * 32;
        int t4954 = t4951 - t4953;
        int t4955 = t4954 / 32;
        int t4956 = t4955 * 32;
        int t4957 = t4954 - t4956;
        int t4958 = t4952 * 64;
        int t4959 = 32 + t4958;
        int t4960 = t4959 + t4957;
        int t4961 = i;
        int t4962 = t4961 * 1024;
        int t4963 = t4962 + t4960;
        float t4964 = memory[158889983 + t4963];
        int t4965 = t4899 / 32;
        int t4966 = t4965 * 32;
        int t4967 = t4899 - t4966;
        float t4968 = memory[4188 + t4967];
        float t4969 = t4964 * t4968;
        int t4970 = t4899 / 32;
        int t4971 = t4970 * 32;
        int t4972 = t4899 - t4971;
        int t4973 = t4970 * 32;
        int t4974 = t4973 + t4972;
        int t4975 = t4974 / 32;
        int t4976 = t4975 * 32;
        int t4977 = t4974 - t4976;
        int t4978 = t4977 / 32;
        int t4979 = t4978 * 32;
        int t4980 = t4977 - t4979;
        int t4981 = t4975 * 64;
        int t4982 = 32 + t4981;
        int t4983 = t4982 + t4980;
        int t4984 = i;
        int t4985 = t4984 * 1024;
        int t4986 = t4985 + t4983;
        float t4987 = memory[157317119 + t4986];
        int t4988 = t4899 / 32;
        int t4989 = t4988 * 32;
        int t4990 = t4899 - t4989;
        float t4991 = memory[4156 + t4990];
        float t4992 = t4987 * t4991;
        float t4993 = t4969 + t4992;
        int t4994 = t4899 / 32;
        int t4995 = t4994 * 32;
        int t4996 = t4899 - t4995;
        int t4997 = t4994 * 32;
        int t4998 = t4997 + t4996;
        int t4999 = t4998 / 32;
        int t5000 = t4999 * 32;
        int t5001 = t4998 - t5000;
        int t5002 = t5001 / 32;
        int t5003 = t5002 * 32;
        int t5004 = t5001 - t5003;
        int t5005 = t4999 * 64;
        int t5006 = t5005 + t5004;
        int t5007 = i;
        int t5008 = t5007 * 1024;
        int t5009 = t5008 + t5006;
        float t5010 = memory[158889983 + t5009];
        float t5011 = t5010 + t4946;
        int t5012 = i;
        int t5013 = t5012 * 512;
        int t5014 = t5013 + t4899;
        memory[157841407 + t5014] = t5011;
        int t5016 = t4899 / 32;
        int t5017 = t5016 * 32;
        int t5018 = t4899 - t5017;
        int t5019 = t5016 * 32;
        int t5020 = t5019 + t5018;
        int t5021 = t5020 / 32;
        int t5022 = t5021 * 32;
        int t5023 = t5020 - t5022;
        int t5024 = t5023 / 32;
        int t5025 = t5024 * 32;
        int t5026 = t5023 - t5025;
        int t5027 = t5021 * 64;
        int t5028 = t5027 + t5026;
        int t5029 = i;
        int t5030 = t5029 * 1024;
        int t5031 = t5030 + t5028;
        float t5032 = memory[157317119 + t5031];
        float t5033 = t5032 + t4993;
        int t5034 = i;
        int t5035 = t5034 * 512;
        int t5036 = t5035 + t4899;
        memory[151025663 + t5036] = t5033;
        int t5038 = t4899 / 32;
        int t5039 = t5038 * 32;
        int t5040 = t4899 - t5039;
        int t5041 = t5038 * 32;
        int t5042 = t5041 + t5040;
        int t5043 = t5042 / 32;
        int t5044 = t5043 * 32;
        int t5045 = t5042 - t5044;
        int t5046 = t5045 / 32;
        int t5047 = t5046 * 32;
        int t5048 = t5045 - t5047;
        int t5049 = t5043 * 64;
        int t5050 = t5049 + t5048;
        int t5051 = i;
        int t5052 = t5051 * 1024;
        int t5053 = t5052 + t5050;
        float t5054 = memory[158889983 + t5053];
        float t5055 = t5054 - t4946;
        int t5056 = i;
        int t5057 = t5056 * 512;
        int t5058 = t5057 + t4899;
        memory[133199871 + t5058] = t5055;
        int t5060 = t4899 / 32;
        int t5061 = t5060 * 32;
        int t5062 = t4899 - t5061;
        int t5063 = t5060 * 32;
        int t5064 = t5063 + t5062;
        int t5065 = t5064 / 32;
        int t5066 = t5065 * 32;
        int t5067 = t5064 - t5066;
        int t5068 = t5067 / 32;
        int t5069 = t5068 * 32;
        int t5070 = t5067 - t5069;
        int t5071 = t5065 * 64;
        int t5072 = t5071 + t5070;
        int t5073 = i;
        int t5074 = t5073 * 1024;
        int t5075 = t5074 + t5072;
        float t5076 = memory[157317119 + t5075];
        float t5077 = t5076 - t4993;
        int t5078 = i;
        int t5079 = t5078 * 512;
        int t5080 = t5079 + t4899;
        memory[150239231 + t5080] = t5077;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
      }
      for (int t5082 = 0; t5082 < 1024; t5082++) {
        int t5083 = t5082 / 64;
        int t5084 = t5083 * 64;
        int t5085 = t5082 - t5084;
        int t5086 = t5083 >= 0;
        int t5087 = t5083 < 16;
        float t5088 = 1.0 * t5086;
        float t5089 = t5088 * t5087;
        int t5090 = t5083;
        int t5091 = t5085 >= 0;
        int t5092 = t5085 < 32;
        float t5093 = t5089 * t5091;
        float t5094 = t5093 * t5092;
        int t5095 = t5085;
        int t5096 = t5090 * 32;
        int t5097 = t5096 + t5095;
        float t5098 = 0.0;
        if (t5094) {
          int t5100 = i;
          int t5101 = t5100 * 512;
          int t5102 = t5101 + t5097;
          float t5103 = memory[157841407 + t5102];
          t5098 = t5103;
        }
        int t5105 = t5082 / 64;
        int t5106 = t5105 * 64;
        int t5107 = t5082 - t5106;
        int t5108 = t5105 >= 0;
        int t5109 = t5105 < 16;
        float t5110 = 1.0 * t5108;
        float t5111 = t5110 * t5109;
        int t5112 = t5105;
        int t5113 = t5107 >= 32;
        int t5114 = t5107 < 64;
        float t5115 = t5111 * t5113;
        float t5116 = t5115 * t5114;
        int t5117 = t5107 - 32;
        int t5118 = t5112 * 32;
        int t5119 = t5118 + t5117;
        float t5120 = 0.0;
        if (t5116) {
          int t5122 = i;
          int t5123 = t5122 * 512;
          int t5124 = t5123 + t5119;
          float t5125 = memory[133199871 + t5124];
          t5120 = t5125;
        }
        float t5127 = t5098 + t5120;
        int t5128 = i;
        int t5129 = t5128 * 1024;
        int t5130 = t5129 + t5082;
        memory[163346431 + t5130] = t5127;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
        int t5132 = t5082 / 64;
        int t5133 = t5132 * 64;
        int t5134 = t5082 - t5133;
        int t5135 = t5132 >= 0;
        int t5136 = t5132 < 16;
        float t5137 = 1.0 * t5135;
        float t5138 = t5137 * t5136;
        int t5139 = t5132;
        int t5140 = t5134 >= 0;
        int t5141 = t5134 < 32;
        float t5142 = t5138 * t5140;
        float t5143 = t5142 * t5141;
        int t5144 = t5134;
        int t5145 = t5139 * 32;
        int t5146 = t5145 + t5144;
        float t5147 = 0.0;
        if (t5143) {
          int t5149 = i;
          int t5150 = t5149 * 512;
          int t5151 = t5150 + t5146;
          float t5152 = memory[151025663 + t5151];
          t5147 = t5152;
        }
        int t5154 = t5082 / 64;
        int t5155 = t5154 * 64;
        int t5156 = t5082 - t5155;
        int t5157 = t5154 >= 0;
        int t5158 = t5154 < 16;
        float t5159 = 1.0 * t5157;
        float t5160 = t5159 * t5158;
        int t5161 = t5154;
        int t5162 = t5156 >= 32;
        int t5163 = t5156 < 64;
        float t5164 = t5160 * t5162;
        float t5165 = t5164 * t5163;
        int t5166 = t5156 - 32;
        int t5167 = t5161 * 32;
        int t5168 = t5167 + t5166;
        float t5169 = 0.0;
        if (t5165) {
          int t5171 = i;
          int t5172 = t5171 * 512;
          int t5173 = t5172 + t5168;
          float t5174 = memory[150239231 + t5173];
          t5169 = t5174;
        }
        float t5176 = t5147 + t5169;
        int t5177 = i;
        int t5178 = t5177 * 1024;
        int t5179 = t5178 + t5082;
        memory[175405055 + t5179] = t5176;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 2, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 2, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([8, 64]), value: empty) */
      }
    }
  }
  for (int simd73 = 0; simd73 < 64; simd73+=4) {
  }
  for (int simd74 = 0; simd74 < 512; simd74+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([8, 64]), value: empty) */
  }
  for (int simd75 = 0; simd75 < 64; simd75+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([8, 64]), value: empty) */
      for (int t5184 = 0; t5184 < 512; t5184++) {
        int t5185 = t5184 / 64;
        int t5186 = t5185 * 64;
        int t5187 = t5184 - t5186;
        int t5188 = t5185 * 64;
        int t5189 = t5188 + t5187;
        int t5190 = t5189 / 64;
        int t5191 = t5190 * 64;
        int t5192 = t5189 - t5191;
        int t5193 = t5192 / 64;
        int t5194 = t5193 * 64;
        int t5195 = t5192 - t5194;
        int t5196 = t5190 * 128;
        int t5197 = 64 + t5196;
        int t5198 = t5197 + t5195;
        int t5199 = i;
        int t5200 = t5199 * 1024;
        int t5201 = t5200 + t5198;
        float t5202 = memory[163346431 + t5201];
        int t5203 = t5184 / 64;
        int t5204 = t5203 * 64;
        int t5205 = t5184 - t5204;
        float t5206 = memory[4220 + t5205];
        float t5207 = t5202 * t5206;
        int t5208 = t5184 / 64;
        int t5209 = t5208 * 64;
        int t5210 = t5184 - t5209;
        int t5211 = t5208 * 64;
        int t5212 = t5211 + t5210;
        int t5213 = t5212 / 64;
        int t5214 = t5213 * 64;
        int t5215 = t5212 - t5214;
        int t5216 = t5215 / 64;
        int t5217 = t5216 * 64;
        int t5218 = t5215 - t5217;
        int t5219 = t5213 * 128;
        int t5220 = 64 + t5219;
        int t5221 = t5220 + t5218;
        int t5222 = i;
        int t5223 = t5222 * 1024;
        int t5224 = t5223 + t5221;
        float t5225 = memory[175405055 + t5224];
        int t5226 = t5184 / 64;
        int t5227 = t5226 * 64;
        int t5228 = t5184 - t5227;
        float t5229 = memory[4284 + t5228];
        float t5230 = t5225 * t5229;
        float t5231 = t5207 - t5230;
        int t5232 = t5184 / 64;
        int t5233 = t5232 * 64;
        int t5234 = t5184 - t5233;
        int t5235 = t5232 * 64;
        int t5236 = t5235 + t5234;
        int t5237 = t5236 / 64;
        int t5238 = t5237 * 64;
        int t5239 = t5236 - t5238;
        int t5240 = t5239 / 64;
        int t5241 = t5240 * 64;
        int t5242 = t5239 - t5241;
        int t5243 = t5237 * 128;
        int t5244 = 64 + t5243;
        int t5245 = t5244 + t5242;
        int t5246 = i;
        int t5247 = t5246 * 1024;
        int t5248 = t5247 + t5245;
        float t5249 = memory[163346431 + t5248];
        int t5250 = t5184 / 64;
        int t5251 = t5250 * 64;
        int t5252 = t5184 - t5251;
        float t5253 = memory[4284 + t5252];
        float t5254 = t5249 * t5253;
        int t5255 = t5184 / 64;
        int t5256 = t5255 * 64;
        int t5257 = t5184 - t5256;
        int t5258 = t5255 * 64;
        int t5259 = t5258 + t5257;
        int t5260 = t5259 / 64;
        int t5261 = t5260 * 64;
        int t5262 = t5259 - t5261;
        int t5263 = t5262 / 64;
        int t5264 = t5263 * 64;
        int t5265 = t5262 - t5264;
        int t5266 = t5260 * 128;
        int t5267 = 64 + t5266;
        int t5268 = t5267 + t5265;
        int t5269 = i;
        int t5270 = t5269 * 1024;
        int t5271 = t5270 + t5268;
        float t5272 = memory[175405055 + t5271];
        int t5273 = t5184 / 64;
        int t5274 = t5273 * 64;
        int t5275 = t5184 - t5274;
        float t5276 = memory[4220 + t5275];
        float t5277 = t5272 * t5276;
        float t5278 = t5254 + t5277;
        int t5279 = t5184 / 64;
        int t5280 = t5279 * 64;
        int t5281 = t5184 - t5280;
        int t5282 = t5279 * 64;
        int t5283 = t5282 + t5281;
        int t5284 = t5283 / 64;
        int t5285 = t5284 * 64;
        int t5286 = t5283 - t5285;
        int t5287 = t5286 / 64;
        int t5288 = t5287 * 64;
        int t5289 = t5286 - t5288;
        int t5290 = t5284 * 128;
        int t5291 = t5290 + t5289;
        int t5292 = i;
        int t5293 = t5292 * 1024;
        int t5294 = t5293 + t5291;
        float t5295 = memory[163346431 + t5294];
        float t5296 = t5295 + t5231;
        int t5297 = i;
        int t5298 = t5297 * 512;
        int t5299 = t5298 + t5184;
        memory[148666367 + t5299] = t5296;
        int t5301 = t5184 / 64;
        int t5302 = t5301 * 64;
        int t5303 = t5184 - t5302;
        int t5304 = t5301 * 64;
        int t5305 = t5304 + t5303;
        int t5306 = t5305 / 64;
        int t5307 = t5306 * 64;
        int t5308 = t5305 - t5307;
        int t5309 = t5308 / 64;
        int t5310 = t5309 * 64;
        int t5311 = t5308 - t5310;
        int t5312 = t5306 * 128;
        int t5313 = t5312 + t5311;
        int t5314 = i;
        int t5315 = t5314 * 1024;
        int t5316 = t5315 + t5313;
        float t5317 = memory[175405055 + t5316];
        float t5318 = t5317 + t5278;
        int t5319 = i;
        int t5320 = t5319 * 512;
        int t5321 = t5320 + t5184;
        memory[149714943 + t5321] = t5318;
        int t5323 = t5184 / 64;
        int t5324 = t5323 * 64;
        int t5325 = t5184 - t5324;
        int t5326 = t5323 * 64;
        int t5327 = t5326 + t5325;
        int t5328 = t5327 / 64;
        int t5329 = t5328 * 64;
        int t5330 = t5327 - t5329;
        int t5331 = t5330 / 64;
        int t5332 = t5331 * 64;
        int t5333 = t5330 - t5332;
        int t5334 = t5328 * 128;
        int t5335 = t5334 + t5333;
        int t5336 = i;
        int t5337 = t5336 * 1024;
        int t5338 = t5337 + t5335;
        float t5339 = memory[163346431 + t5338];
        float t5340 = t5339 - t5231;
        int t5341 = i;
        int t5342 = t5341 * 512;
        int t5343 = t5342 + t5184;
        memory[128743423 + t5343] = t5340;
        int t5345 = t5184 / 64;
        int t5346 = t5345 * 64;
        int t5347 = t5184 - t5346;
        int t5348 = t5345 * 64;
        int t5349 = t5348 + t5347;
        int t5350 = t5349 / 64;
        int t5351 = t5350 * 64;
        int t5352 = t5349 - t5351;
        int t5353 = t5352 / 64;
        int t5354 = t5353 * 64;
        int t5355 = t5352 - t5354;
        int t5356 = t5350 * 128;
        int t5357 = t5356 + t5355;
        int t5358 = i;
        int t5359 = t5358 * 1024;
        int t5360 = t5359 + t5357;
        float t5361 = memory[175405055 + t5360];
        float t5362 = t5361 - t5278;
        int t5363 = i;
        int t5364 = t5363 * 512;
        int t5365 = t5364 + t5184;
        memory[139753471 + t5365] = t5362;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 64)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (64, 0)]), value: empty) */
      }
      for (int t5367 = 0; t5367 < 1024; t5367++) {
        int t5368 = t5367 / 128;
        int t5369 = t5368 * 128;
        int t5370 = t5367 - t5369;
        int t5371 = t5368 >= 0;
        int t5372 = t5368 < 8;
        float t5373 = 1.0 * t5371;
        float t5374 = t5373 * t5372;
        int t5375 = t5368;
        int t5376 = t5370 >= 0;
        int t5377 = t5370 < 64;
        float t5378 = t5374 * t5376;
        float t5379 = t5378 * t5377;
        int t5380 = t5370;
        int t5381 = t5375 * 64;
        int t5382 = t5381 + t5380;
        float t5383 = 0.0;
        if (t5379) {
          int t5385 = i;
          int t5386 = t5385 * 512;
          int t5387 = t5386 + t5382;
          float t5388 = memory[148666367 + t5387];
          t5383 = t5388;
        }
        int t5390 = t5367 / 128;
        int t5391 = t5390 * 128;
        int t5392 = t5367 - t5391;
        int t5393 = t5390 >= 0;
        int t5394 = t5390 < 8;
        float t5395 = 1.0 * t5393;
        float t5396 = t5395 * t5394;
        int t5397 = t5390;
        int t5398 = t5392 >= 64;
        int t5399 = t5392 < 128;
        float t5400 = t5396 * t5398;
        float t5401 = t5400 * t5399;
        int t5402 = t5392 - 64;
        int t5403 = t5397 * 64;
        int t5404 = t5403 + t5402;
        float t5405 = 0.0;
        if (t5401) {
          int t5407 = i;
          int t5408 = t5407 * 512;
          int t5409 = t5408 + t5404;
          float t5410 = memory[128743423 + t5409];
          t5405 = t5410;
        }
        float t5412 = t5383 + t5405;
        int t5413 = i;
        int t5414 = t5413 * 1024;
        int t5415 = t5414 + t5367;
        memory[114587647 + t5415] = t5412;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 64)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (64, 0)]), value: empty) */
        int t5417 = t5367 / 128;
        int t5418 = t5417 * 128;
        int t5419 = t5367 - t5418;
        int t5420 = t5417 >= 0;
        int t5421 = t5417 < 8;
        float t5422 = 1.0 * t5420;
        float t5423 = t5422 * t5421;
        int t5424 = t5417;
        int t5425 = t5419 >= 0;
        int t5426 = t5419 < 64;
        float t5427 = t5423 * t5425;
        float t5428 = t5427 * t5426;
        int t5429 = t5419;
        int t5430 = t5424 * 64;
        int t5431 = t5430 + t5429;
        float t5432 = 0.0;
        if (t5428) {
          int t5434 = i;
          int t5435 = t5434 * 512;
          int t5436 = t5435 + t5431;
          float t5437 = memory[149714943 + t5436];
          t5432 = t5437;
        }
        int t5439 = t5367 / 128;
        int t5440 = t5439 * 128;
        int t5441 = t5367 - t5440;
        int t5442 = t5439 >= 0;
        int t5443 = t5439 < 8;
        float t5444 = 1.0 * t5442;
        float t5445 = t5444 * t5443;
        int t5446 = t5439;
        int t5447 = t5441 >= 64;
        int t5448 = t5441 < 128;
        float t5449 = t5445 * t5447;
        float t5450 = t5449 * t5448;
        int t5451 = t5441 - 64;
        int t5452 = t5446 * 64;
        int t5453 = t5452 + t5451;
        float t5454 = 0.0;
        if (t5450) {
          int t5456 = i;
          int t5457 = t5456 * 512;
          int t5458 = t5457 + t5453;
          float t5459 = memory[139753471 + t5458];
          t5454 = t5459;
        }
        float t5461 = t5432 + t5454;
        int t5462 = i;
        int t5463 = t5462 * 1024;
        int t5464 = t5463 + t5367;
        memory[172783615 + t5464] = t5461;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 2, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 2, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([4, 128]), value: empty) */
      }
    }
  }
  for (int simd77 = 0; simd77 < 128; simd77+=4) {
  }
  for (int simd78 = 0; simd78 < 512; simd78+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([4, 128]), value: empty) */
  }
  for (int simd79 = 0; simd79 < 128; simd79+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([4, 128]), value: empty) */
      for (int t5469 = 0; t5469 < 512; t5469++) {
        int t5470 = t5469 / 128;
        int t5471 = t5470 * 128;
        int t5472 = t5469 - t5471;
        int t5473 = t5470 * 128;
        int t5474 = t5473 + t5472;
        int t5475 = t5474 / 128;
        int t5476 = t5475 * 128;
        int t5477 = t5474 - t5476;
        int t5478 = t5477 / 128;
        int t5479 = t5478 * 128;
        int t5480 = t5477 - t5479;
        int t5481 = t5475 * 256;
        int t5482 = 128 + t5481;
        int t5483 = t5482 + t5480;
        int t5484 = i;
        int t5485 = t5484 * 1024;
        int t5486 = t5485 + t5483;
        float t5487 = memory[114587647 + t5486];
        int t5488 = t5469 / 128;
        int t5489 = t5488 * 128;
        int t5490 = t5469 - t5489;
        float t5491 = memory[4348 + t5490];
        float t5492 = t5487 * t5491;
        int t5493 = t5469 / 128;
        int t5494 = t5493 * 128;
        int t5495 = t5469 - t5494;
        int t5496 = t5493 * 128;
        int t5497 = t5496 + t5495;
        int t5498 = t5497 / 128;
        int t5499 = t5498 * 128;
        int t5500 = t5497 - t5499;
        int t5501 = t5500 / 128;
        int t5502 = t5501 * 128;
        int t5503 = t5500 - t5502;
        int t5504 = t5498 * 256;
        int t5505 = 128 + t5504;
        int t5506 = t5505 + t5503;
        int t5507 = i;
        int t5508 = t5507 * 1024;
        int t5509 = t5508 + t5506;
        float t5510 = memory[172783615 + t5509];
        int t5511 = t5469 / 128;
        int t5512 = t5511 * 128;
        int t5513 = t5469 - t5512;
        float t5514 = memory[4476 + t5513];
        float t5515 = t5510 * t5514;
        float t5516 = t5492 - t5515;
        int t5517 = t5469 / 128;
        int t5518 = t5517 * 128;
        int t5519 = t5469 - t5518;
        int t5520 = t5517 * 128;
        int t5521 = t5520 + t5519;
        int t5522 = t5521 / 128;
        int t5523 = t5522 * 128;
        int t5524 = t5521 - t5523;
        int t5525 = t5524 / 128;
        int t5526 = t5525 * 128;
        int t5527 = t5524 - t5526;
        int t5528 = t5522 * 256;
        int t5529 = 128 + t5528;
        int t5530 = t5529 + t5527;
        int t5531 = i;
        int t5532 = t5531 * 1024;
        int t5533 = t5532 + t5530;
        float t5534 = memory[114587647 + t5533];
        int t5535 = t5469 / 128;
        int t5536 = t5535 * 128;
        int t5537 = t5469 - t5536;
        float t5538 = memory[4476 + t5537];
        float t5539 = t5534 * t5538;
        int t5540 = t5469 / 128;
        int t5541 = t5540 * 128;
        int t5542 = t5469 - t5541;
        int t5543 = t5540 * 128;
        int t5544 = t5543 + t5542;
        int t5545 = t5544 / 128;
        int t5546 = t5545 * 128;
        int t5547 = t5544 - t5546;
        int t5548 = t5547 / 128;
        int t5549 = t5548 * 128;
        int t5550 = t5547 - t5549;
        int t5551 = t5545 * 256;
        int t5552 = 128 + t5551;
        int t5553 = t5552 + t5550;
        int t5554 = i;
        int t5555 = t5554 * 1024;
        int t5556 = t5555 + t5553;
        float t5557 = memory[172783615 + t5556];
        int t5558 = t5469 / 128;
        int t5559 = t5558 * 128;
        int t5560 = t5469 - t5559;
        float t5561 = memory[4348 + t5560];
        float t5562 = t5557 * t5561;
        float t5563 = t5539 + t5562;
        int t5564 = t5469 / 128;
        int t5565 = t5564 * 128;
        int t5566 = t5469 - t5565;
        int t5567 = t5564 * 128;
        int t5568 = t5567 + t5566;
        int t5569 = t5568 / 128;
        int t5570 = t5569 * 128;
        int t5571 = t5568 - t5570;
        int t5572 = t5571 / 128;
        int t5573 = t5572 * 128;
        int t5574 = t5571 - t5573;
        int t5575 = t5569 * 256;
        int t5576 = t5575 + t5574;
        int t5577 = i;
        int t5578 = t5577 * 1024;
        int t5579 = t5578 + t5576;
        float t5580 = memory[114587647 + t5579];
        float t5581 = t5580 + t5516;
        int t5582 = i;
        int t5583 = t5582 * 512;
        int t5584 = t5583 + t5469;
        memory[131102719 + t5584] = t5581;
        int t5586 = t5469 / 128;
        int t5587 = t5586 * 128;
        int t5588 = t5469 - t5587;
        int t5589 = t5586 * 128;
        int t5590 = t5589 + t5588;
        int t5591 = t5590 / 128;
        int t5592 = t5591 * 128;
        int t5593 = t5590 - t5592;
        int t5594 = t5593 / 128;
        int t5595 = t5594 * 128;
        int t5596 = t5593 - t5595;
        int t5597 = t5591 * 256;
        int t5598 = t5597 + t5596;
        int t5599 = i;
        int t5600 = t5599 * 1024;
        int t5601 = t5600 + t5598;
        float t5602 = memory[172783615 + t5601];
        float t5603 = t5602 + t5563;
        int t5604 = i;
        int t5605 = t5604 * 512;
        int t5606 = t5605 + t5469;
        memory[120879103 + t5606] = t5603;
        int t5608 = t5469 / 128;
        int t5609 = t5608 * 128;
        int t5610 = t5469 - t5609;
        int t5611 = t5608 * 128;
        int t5612 = t5611 + t5610;
        int t5613 = t5612 / 128;
        int t5614 = t5613 * 128;
        int t5615 = t5612 - t5614;
        int t5616 = t5615 / 128;
        int t5617 = t5616 * 128;
        int t5618 = t5615 - t5617;
        int t5619 = t5613 * 256;
        int t5620 = t5619 + t5618;
        int t5621 = i;
        int t5622 = t5621 * 1024;
        int t5623 = t5622 + t5620;
        float t5624 = memory[114587647 + t5623];
        float t5625 = t5624 - t5516;
        int t5626 = i;
        int t5627 = t5626 * 512;
        int t5628 = t5627 + t5469;
        memory[112752639 + t5628] = t5625;
        int t5630 = t5469 / 128;
        int t5631 = t5630 * 128;
        int t5632 = t5469 - t5631;
        int t5633 = t5630 * 128;
        int t5634 = t5633 + t5632;
        int t5635 = t5634 / 128;
        int t5636 = t5635 * 128;
        int t5637 = t5634 - t5636;
        int t5638 = t5637 / 128;
        int t5639 = t5638 * 128;
        int t5640 = t5637 - t5639;
        int t5641 = t5635 * 256;
        int t5642 = t5641 + t5640;
        int t5643 = i;
        int t5644 = t5643 * 1024;
        int t5645 = t5644 + t5642;
        float t5646 = memory[172783615 + t5645];
        float t5647 = t5646 - t5563;
        int t5648 = i;
        int t5649 = t5648 * 512;
        int t5650 = t5649 + t5469;
        memory[146569215 + t5650] = t5647;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 128)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (128, 0)]), value: empty) */
      }
      for (int t5652 = 0; t5652 < 1024; t5652++) {
        int t5653 = t5652 / 256;
        int t5654 = t5653 * 256;
        int t5655 = t5652 - t5654;
        int t5656 = t5653 >= 0;
        int t5657 = t5653 < 4;
        float t5658 = 1.0 * t5656;
        float t5659 = t5658 * t5657;
        int t5660 = t5653;
        int t5661 = t5655 >= 0;
        int t5662 = t5655 < 128;
        float t5663 = t5659 * t5661;
        float t5664 = t5663 * t5662;
        int t5665 = t5655;
        int t5666 = t5660 * 128;
        int t5667 = t5666 + t5665;
        float t5668 = 0.0;
        if (t5664) {
          int t5670 = i;
          int t5671 = t5670 * 512;
          int t5672 = t5671 + t5667;
          float t5673 = memory[131102719 + t5672];
          t5668 = t5673;
        }
        int t5675 = t5652 / 256;
        int t5676 = t5675 * 256;
        int t5677 = t5652 - t5676;
        int t5678 = t5675 >= 0;
        int t5679 = t5675 < 4;
        float t5680 = 1.0 * t5678;
        float t5681 = t5680 * t5679;
        int t5682 = t5675;
        int t5683 = t5677 >= 128;
        int t5684 = t5677 < 256;
        float t5685 = t5681 * t5683;
        float t5686 = t5685 * t5684;
        int t5687 = t5677 - 128;
        int t5688 = t5682 * 128;
        int t5689 = t5688 + t5687;
        float t5690 = 0.0;
        if (t5686) {
          int t5692 = i;
          int t5693 = t5692 * 512;
          int t5694 = t5693 + t5689;
          float t5695 = memory[112752639 + t5694];
          t5690 = t5695;
        }
        float t5697 = t5668 + t5690;
        int t5698 = i;
        int t5699 = t5698 * 1024;
        int t5700 = t5699 + t5652;
        memory[135297023 + t5700] = t5697;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 128)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (128, 0)]), value: empty) */
        int t5702 = t5652 / 256;
        int t5703 = t5702 * 256;
        int t5704 = t5652 - t5703;
        int t5705 = t5702 >= 0;
        int t5706 = t5702 < 4;
        float t5707 = 1.0 * t5705;
        float t5708 = t5707 * t5706;
        int t5709 = t5702;
        int t5710 = t5704 >= 0;
        int t5711 = t5704 < 128;
        float t5712 = t5708 * t5710;
        float t5713 = t5712 * t5711;
        int t5714 = t5704;
        int t5715 = t5709 * 128;
        int t5716 = t5715 + t5714;
        float t5717 = 0.0;
        if (t5713) {
          int t5719 = i;
          int t5720 = t5719 * 512;
          int t5721 = t5720 + t5716;
          float t5722 = memory[120879103 + t5721];
          t5717 = t5722;
        }
        int t5724 = t5652 / 256;
        int t5725 = t5724 * 256;
        int t5726 = t5652 - t5725;
        int t5727 = t5724 >= 0;
        int t5728 = t5724 < 4;
        float t5729 = 1.0 * t5727;
        float t5730 = t5729 * t5728;
        int t5731 = t5724;
        int t5732 = t5726 >= 128;
        int t5733 = t5726 < 256;
        float t5734 = t5730 * t5732;
        float t5735 = t5734 * t5733;
        int t5736 = t5726 - 128;
        int t5737 = t5731 * 128;
        int t5738 = t5737 + t5736;
        float t5739 = 0.0;
        if (t5735) {
          int t5741 = i;
          int t5742 = t5741 * 512;
          int t5743 = t5742 + t5738;
          float t5744 = memory[146569215 + t5743];
          t5739 = t5744;
        }
        float t5746 = t5717 + t5739;
        int t5747 = i;
        int t5748 = t5747 * 1024;
        int t5749 = t5748 + t5652;
        memory[122451967 + t5749] = t5746;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 256]), value: empty) */
      }
    }
  }
  for (int simd81 = 0; simd81 < 256; simd81+=4) {
  }
  for (int simd82 = 0; simd82 < 512; simd82+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([2, 256]), value: empty) */
  }
  for (int simd83 = 0; simd83 < 256; simd83+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([2, 256]), value: empty) */
      for (int t5754 = 0; t5754 < 512; t5754++) {
        int t5755 = t5754 / 256;
        int t5756 = t5755 * 256;
        int t5757 = t5754 - t5756;
        int t5758 = t5755 * 256;
        int t5759 = t5758 + t5757;
        int t5760 = t5759 / 256;
        int t5761 = t5760 * 256;
        int t5762 = t5759 - t5761;
        int t5763 = t5762 / 256;
        int t5764 = t5763 * 256;
        int t5765 = t5762 - t5764;
        int t5766 = t5760 * 512;
        int t5767 = 256 + t5766;
        int t5768 = t5767 + t5765;
        int t5769 = i;
        int t5770 = t5769 * 1024;
        int t5771 = t5770 + t5768;
        float t5772 = memory[135297023 + t5771];
        int t5773 = t5754 / 256;
        int t5774 = t5773 * 256;
        int t5775 = t5754 - t5774;
        float t5776 = memory[4604 + t5775];
        float t5777 = t5772 * t5776;
        int t5778 = t5754 / 256;
        int t5779 = t5778 * 256;
        int t5780 = t5754 - t5779;
        int t5781 = t5778 * 256;
        int t5782 = t5781 + t5780;
        int t5783 = t5782 / 256;
        int t5784 = t5783 * 256;
        int t5785 = t5782 - t5784;
        int t5786 = t5785 / 256;
        int t5787 = t5786 * 256;
        int t5788 = t5785 - t5787;
        int t5789 = t5783 * 512;
        int t5790 = 256 + t5789;
        int t5791 = t5790 + t5788;
        int t5792 = i;
        int t5793 = t5792 * 1024;
        int t5794 = t5793 + t5791;
        float t5795 = memory[122451967 + t5794];
        int t5796 = t5754 / 256;
        int t5797 = t5796 * 256;
        int t5798 = t5754 - t5797;
        float t5799 = memory[4860 + t5798];
        float t5800 = t5795 * t5799;
        float t5801 = t5777 - t5800;
        int t5802 = t5754 / 256;
        int t5803 = t5802 * 256;
        int t5804 = t5754 - t5803;
        int t5805 = t5802 * 256;
        int t5806 = t5805 + t5804;
        int t5807 = t5806 / 256;
        int t5808 = t5807 * 256;
        int t5809 = t5806 - t5808;
        int t5810 = t5809 / 256;
        int t5811 = t5810 * 256;
        int t5812 = t5809 - t5811;
        int t5813 = t5807 * 512;
        int t5814 = 256 + t5813;
        int t5815 = t5814 + t5812;
        int t5816 = i;
        int t5817 = t5816 * 1024;
        int t5818 = t5817 + t5815;
        float t5819 = memory[135297023 + t5818];
        int t5820 = t5754 / 256;
        int t5821 = t5820 * 256;
        int t5822 = t5754 - t5821;
        float t5823 = memory[4860 + t5822];
        float t5824 = t5819 * t5823;
        int t5825 = t5754 / 256;
        int t5826 = t5825 * 256;
        int t5827 = t5754 - t5826;
        int t5828 = t5825 * 256;
        int t5829 = t5828 + t5827;
        int t5830 = t5829 / 256;
        int t5831 = t5830 * 256;
        int t5832 = t5829 - t5831;
        int t5833 = t5832 / 256;
        int t5834 = t5833 * 256;
        int t5835 = t5832 - t5834;
        int t5836 = t5830 * 512;
        int t5837 = 256 + t5836;
        int t5838 = t5837 + t5835;
        int t5839 = i;
        int t5840 = t5839 * 1024;
        int t5841 = t5840 + t5838;
        float t5842 = memory[122451967 + t5841];
        int t5843 = t5754 / 256;
        int t5844 = t5843 * 256;
        int t5845 = t5754 - t5844;
        float t5846 = memory[4604 + t5845];
        float t5847 = t5842 * t5846;
        float t5848 = t5824 + t5847;
        int t5849 = t5754 / 256;
        int t5850 = t5849 * 256;
        int t5851 = t5754 - t5850;
        int t5852 = t5849 * 256;
        int t5853 = t5852 + t5851;
        int t5854 = t5853 / 256;
        int t5855 = t5854 * 256;
        int t5856 = t5853 - t5855;
        int t5857 = t5856 / 256;
        int t5858 = t5857 * 256;
        int t5859 = t5856 - t5858;
        int t5860 = t5854 * 512;
        int t5861 = t5860 + t5859;
        int t5862 = i;
        int t5863 = t5862 * 1024;
        int t5864 = t5863 + t5861;
        float t5865 = memory[135297023 + t5864];
        float t5866 = t5865 + t5801;
        int t5867 = i;
        int t5868 = t5867 * 512;
        int t5869 = t5868 + t5754;
        memory[132675583 + t5869] = t5866;
        int t5871 = t5754 / 256;
        int t5872 = t5871 * 256;
        int t5873 = t5754 - t5872;
        int t5874 = t5871 * 256;
        int t5875 = t5874 + t5873;
        int t5876 = t5875 / 256;
        int t5877 = t5876 * 256;
        int t5878 = t5875 - t5877;
        int t5879 = t5878 / 256;
        int t5880 = t5879 * 256;
        int t5881 = t5878 - t5880;
        int t5882 = t5876 * 512;
        int t5883 = t5882 + t5881;
        int t5884 = i;
        int t5885 = t5884 * 1024;
        int t5886 = t5885 + t5883;
        float t5887 = memory[122451967 + t5886];
        float t5888 = t5887 + t5848;
        int t5889 = i;
        int t5890 = t5889 * 512;
        int t5891 = t5890 + t5754;
        memory[140539903 + t5891] = t5888;
        int t5893 = t5754 / 256;
        int t5894 = t5893 * 256;
        int t5895 = t5754 - t5894;
        int t5896 = t5893 * 256;
        int t5897 = t5896 + t5895;
        int t5898 = t5897 / 256;
        int t5899 = t5898 * 256;
        int t5900 = t5897 - t5899;
        int t5901 = t5900 / 256;
        int t5902 = t5901 * 256;
        int t5903 = t5900 - t5902;
        int t5904 = t5898 * 512;
        int t5905 = t5904 + t5903;
        int t5906 = i;
        int t5907 = t5906 * 1024;
        int t5908 = t5907 + t5905;
        float t5909 = memory[135297023 + t5908];
        float t5910 = t5909 - t5801;
        int t5911 = i;
        int t5912 = t5911 * 512;
        int t5913 = t5912 + t5754;
        memory[157054975 + t5913] = t5910;
        int t5915 = t5754 / 256;
        int t5916 = t5915 * 256;
        int t5917 = t5754 - t5916;
        int t5918 = t5915 * 256;
        int t5919 = t5918 + t5917;
        int t5920 = t5919 / 256;
        int t5921 = t5920 * 256;
        int t5922 = t5919 - t5921;
        int t5923 = t5922 / 256;
        int t5924 = t5923 * 256;
        int t5925 = t5922 - t5924;
        int t5926 = t5920 * 512;
        int t5927 = t5926 + t5925;
        int t5928 = i;
        int t5929 = t5928 * 1024;
        int t5930 = t5929 + t5927;
        float t5931 = memory[122451967 + t5930];
        float t5932 = t5931 - t5848;
        int t5933 = i;
        int t5934 = t5933 * 512;
        int t5935 = t5934 + t5754;
        memory[152598527 + t5935] = t5932;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 256)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (256, 0)]), value: empty) */
      }
      for (int t5937 = 0; t5937 < 1024; t5937++) {
        int t5938 = t5937 / 512;
        int t5939 = t5938 * 512;
        int t5940 = t5937 - t5939;
        int t5941 = t5938 >= 0;
        int t5942 = t5938 < 2;
        float t5943 = 1.0 * t5941;
        float t5944 = t5943 * t5942;
        int t5945 = t5938;
        int t5946 = t5940 >= 0;
        int t5947 = t5940 < 256;
        float t5948 = t5944 * t5946;
        float t5949 = t5948 * t5947;
        int t5950 = t5940;
        int t5951 = t5945 * 256;
        int t5952 = t5951 + t5950;
        float t5953 = 0.0;
        if (t5949) {
          int t5955 = i;
          int t5956 = t5955 * 512;
          int t5957 = t5956 + t5952;
          float t5958 = memory[132675583 + t5957];
          t5953 = t5958;
        }
        int t5960 = t5937 / 512;
        int t5961 = t5960 * 512;
        int t5962 = t5937 - t5961;
        int t5963 = t5960 >= 0;
        int t5964 = t5960 < 2;
        float t5965 = 1.0 * t5963;
        float t5966 = t5965 * t5964;
        int t5967 = t5960;
        int t5968 = t5962 >= 256;
        int t5969 = t5962 < 512;
        float t5970 = t5966 * t5968;
        float t5971 = t5970 * t5969;
        int t5972 = t5962 - 256;
        int t5973 = t5967 * 256;
        int t5974 = t5973 + t5972;
        float t5975 = 0.0;
        if (t5971) {
          int t5977 = i;
          int t5978 = t5977 * 512;
          int t5979 = t5978 + t5974;
          float t5980 = memory[157054975 + t5979];
          t5975 = t5980;
        }
        float t5982 = t5953 + t5975;
        int t5983 = i;
        int t5984 = t5983 * 1024;
        int t5985 = t5984 + t5937;
        memory[113801215 + t5985] = t5982;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 256)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (256, 0)]), value: empty) */
        int t5987 = t5937 / 512;
        int t5988 = t5987 * 512;
        int t5989 = t5937 - t5988;
        int t5990 = t5987 >= 0;
        int t5991 = t5987 < 2;
        float t5992 = 1.0 * t5990;
        float t5993 = t5992 * t5991;
        int t5994 = t5987;
        int t5995 = t5989 >= 0;
        int t5996 = t5989 < 256;
        float t5997 = t5993 * t5995;
        float t5998 = t5997 * t5996;
        int t5999 = t5989;
        int t6000 = t5994 * 256;
        int t6001 = t6000 + t5999;
        float t6002 = 0.0;
        if (t5998) {
          int t6004 = i;
          int t6005 = t6004 * 512;
          int t6006 = t6005 + t6001;
          float t6007 = memory[140539903 + t6006];
          t6002 = t6007;
        }
        int t6009 = t5937 / 512;
        int t6010 = t6009 * 512;
        int t6011 = t5937 - t6010;
        int t6012 = t6009 >= 0;
        int t6013 = t6009 < 2;
        float t6014 = 1.0 * t6012;
        float t6015 = t6014 * t6013;
        int t6016 = t6009;
        int t6017 = t6011 >= 256;
        int t6018 = t6011 < 512;
        float t6019 = t6015 * t6017;
        float t6020 = t6019 * t6018;
        int t6021 = t6011 - 256;
        int t6022 = t6016 * 256;
        int t6023 = t6022 + t6021;
        float t6024 = 0.0;
        if (t6020) {
          int t6026 = i;
          int t6027 = t6026 * 512;
          int t6028 = t6027 + t6023;
          float t6029 = memory[152598527 + t6028];
          t6024 = t6029;
        }
        float t6031 = t6002 + t6024;
        int t6032 = i;
        int t6033 = t6032 * 1024;
        int t6034 = t6033 + t5937;
        memory[155482111 + t6034] = t6031;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 2, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 2, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1, 512]), value: empty) */
      }
    }
  }
  for (int simd85 = 0; simd85 < 512; simd85+=4) {
  }
  for (int simd86 = 0; simd86 < 512; simd86+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([1, 512]), value: empty) */
  }
  for (int simd87 = 0; simd87 < 512; simd87+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd191 = vld1q_f32(t191 + i); /* extra */
    t191[i] = t191[i];
    if (t191[i] == 0.0f) {
      /* skip scalar load */
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([1, 512]), value: empty) */
      for (int t6039 = 0; t6039 < 512; t6039++) {
        int t6040 = t6039 / 512;
        int t6041 = t6040 * 512;
        int t6042 = t6039 - t6041;
        int t6043 = t6042 / 512;
        int t6044 = t6043 * 512;
        int t6045 = t6042 - t6044;
        int t6046 = t6045 / 512;
        int t6047 = t6046 * 512;
        int t6048 = t6045 - t6047;
        int t6049 = 512 + t6048;
        int t6050 = i;
        int t6051 = t6050 * 1024;
        int t6052 = t6051 + t6049;
        float t6053 = memory[113801215 + t6052];
        float t6054 = memory[5116 + (isfinite((int) t6039) ? (int) t6039 : 0)];
        float t6055 = t6053 * t6054;
        int t6056 = t6039 / 512;
        int t6057 = t6056 * 512;
        int t6058 = t6039 - t6057;
        int t6059 = t6058 / 512;
        int t6060 = t6059 * 512;
        int t6061 = t6058 - t6060;
        int t6062 = t6061 / 512;
        int t6063 = t6062 * 512;
        int t6064 = t6061 - t6063;
        int t6065 = 512 + t6064;
        int t6066 = i;
        int t6067 = t6066 * 1024;
        int t6068 = t6067 + t6065;
        float t6069 = memory[155482111 + t6068];
        float t6070 = memory[5628 + (isfinite((int) t6039) ? (int) t6039 : 0)];
        float t6071 = t6069 * t6070;
        float t6072 = t6055 - t6071;
        int t6073 = t6039 / 512;
        int t6074 = t6073 * 512;
        int t6075 = t6039 - t6074;
        int t6076 = t6075 / 512;
        int t6077 = t6076 * 512;
        int t6078 = t6075 - t6077;
        int t6079 = t6078 / 512;
        int t6080 = t6079 * 512;
        int t6081 = t6078 - t6080;
        int t6082 = 512 + t6081;
        int t6083 = i;
        int t6084 = t6083 * 1024;
        int t6085 = t6084 + t6082;
        float t6086 = memory[113801215 + t6085];
        float t6087 = memory[5628 + (isfinite((int) t6039) ? (int) t6039 : 0)];
        float t6088 = t6086 * t6087;
        int t6089 = t6039 / 512;
        int t6090 = t6089 * 512;
        int t6091 = t6039 - t6090;
        int t6092 = t6091 / 512;
        int t6093 = t6092 * 512;
        int t6094 = t6091 - t6093;
        int t6095 = t6094 / 512;
        int t6096 = t6095 * 512;
        int t6097 = t6094 - t6096;
        int t6098 = 512 + t6097;
        int t6099 = i;
        int t6100 = t6099 * 1024;
        int t6101 = t6100 + t6098;
        float t6102 = memory[155482111 + t6101];
        float t6103 = memory[5116 + (isfinite((int) t6039) ? (int) t6039 : 0)];
        float t6104 = t6102 * t6103;
        float t6105 = t6088 + t6104;
        int t6106 = t6039 / 512;
        int t6107 = t6106 * 512;
        int t6108 = t6039 - t6107;
        int t6109 = t6108 / 512;
        int t6110 = t6109 * 512;
        int t6111 = t6108 - t6110;
        int t6112 = t6111 / 512;
        int t6113 = t6112 * 512;
        int t6114 = t6111 - t6113;
        int t6115 = i;
        int t6116 = t6115 * 1024;
        int t6117 = t6116 + t6114;
        float t6118 = memory[113801215 + t6117];
        float t6119 = t6118 + t6072;
        int t6120 = i;
        int t6121 = t6120 * 512;
        int t6122 = t6121 + t6039;
        memory[125335551 + t6122] = t6119;
        int t6124 = t6039 / 512;
        int t6125 = t6124 * 512;
        int t6126 = t6039 - t6125;
        int t6127 = t6126 / 512;
        int t6128 = t6127 * 512;
        int t6129 = t6126 - t6128;
        int t6130 = t6129 / 512;
        int t6131 = t6130 * 512;
        int t6132 = t6129 - t6131;
        int t6133 = i;
        int t6134 = t6133 * 1024;
        int t6135 = t6134 + t6132;
        float t6136 = memory[155482111 + t6135];
        float t6137 = t6136 + t6105;
        int t6138 = i;
        int t6139 = t6138 * 512;
        int t6140 = t6139 + t6039;
        memory[164395007 + t6140] = t6137;
        int t6142 = t6039 / 512;
        int t6143 = t6142 * 512;
        int t6144 = t6039 - t6143;
        int t6145 = t6144 / 512;
        int t6146 = t6145 * 512;
        int t6147 = t6144 - t6146;
        int t6148 = t6147 / 512;
        int t6149 = t6148 * 512;
        int t6150 = t6147 - t6149;
        int t6151 = i;
        int t6152 = t6151 * 1024;
        int t6153 = t6152 + t6150;
        float t6154 = memory[113801215 + t6153];
        float t6155 = t6154 - t6072;
        int t6156 = i;
        int t6157 = t6156 * 512;
        int t6158 = t6157 + t6039;
        memory[174356479 + t6158] = t6155;
        int t6160 = t6039 / 512;
        int t6161 = t6160 * 512;
        int t6162 = t6039 - t6161;
        int t6163 = t6162 / 512;
        int t6164 = t6163 * 512;
        int t6165 = t6162 - t6164;
        int t6166 = t6165 / 512;
        int t6167 = t6166 * 512;
        int t6168 = t6165 - t6167;
        int t6169 = i;
        int t6170 = t6169 * 1024;
        int t6171 = t6170 + t6168;
        float t6172 = memory[155482111 + t6171];
        float t6173 = t6172 - t6105;
        int t6174 = i;
        int t6175 = t6174 * 512;
        int t6176 = t6175 + t6039;
        memory[118257663 + t6176] = t6173;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 512)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (512, 0)]), value: empty) */
      }
      for (int t6178 = 0; t6178 < 1024; t6178++) {
        int t6179 = t6178 / 1024;
        int t6180 = t6179 * 1024;
        int t6181 = t6178 - t6180;
        int t6182 = t6179 >= 0;
        int t6183 = t6179 < 1;
        float t6184 = 1.0 * t6182;
        float t6185 = t6184 * t6183;
        int t6186 = t6179;
        int t6187 = t6181 >= 0;
        int t6188 = t6181 < 512;
        float t6189 = t6185 * t6187;
        float t6190 = t6189 * t6188;
        int t6191 = t6181;
        int t6192 = t6186 * 512;
        int t6193 = t6192 + t6191;
        float t6194 = 0.0;
        if (t6190) {
          int t6196 = i;
          int t6197 = t6196 * 512;
          int t6198 = t6197 + t6193;
          float t6199 = memory[125335551 + t6198];
          t6194 = t6199;
        }
        int t6201 = t6178 / 1024;
        int t6202 = t6201 * 1024;
        int t6203 = t6178 - t6202;
        int t6204 = t6201 >= 0;
        int t6205 = t6201 < 1;
        float t6206 = 1.0 * t6204;
        float t6207 = t6206 * t6205;
        int t6208 = t6201;
        int t6209 = t6203 >= 512;
        int t6210 = t6203 < 1024;
        float t6211 = t6207 * t6209;
        float t6212 = t6211 * t6210;
        int t6213 = t6203 - 512;
        int t6214 = t6208 * 512;
        int t6215 = t6214 + t6213;
        float t6216 = 0.0;
        if (t6212) {
          int t6218 = i;
          int t6219 = t6218 * 512;
          int t6220 = t6219 + t6215;
          float t6221 = memory[174356479 + t6220];
          t6216 = t6221;
        }
        float t6223 = t6194 + t6216;
        int t6224 = i;
        int t6225 = t6224 * 1024;
        int t6226 = t6225 + t6178;
        memory[145258495 + t6226] = t6223;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 512)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (512, 0)]), value: empty) */
        int t6228 = t6178 / 1024;
        int t6229 = t6228 * 1024;
        int t6230 = t6178 - t6229;
        int t6231 = t6228 >= 0;
        int t6232 = t6228 < 1;
        float t6233 = 1.0 * t6231;
        float t6234 = t6233 * t6232;
        int t6235 = t6228;
        int t6236 = t6230 >= 0;
        int t6237 = t6230 < 512;
        float t6238 = t6234 * t6236;
        float t6239 = t6238 * t6237;
        int t6240 = t6230;
        int t6241 = t6235 * 512;
        int t6242 = t6241 + t6240;
        float t6243 = 0.0;
        if (t6239) {
          int t6245 = i;
          int t6246 = t6245 * 512;
          int t6247 = t6246 + t6242;
          float t6248 = memory[164395007 + t6247];
          t6243 = t6248;
        }
        int t6250 = t6178 / 1024;
        int t6251 = t6250 * 1024;
        int t6252 = t6178 - t6251;
        int t6253 = t6250 >= 0;
        int t6254 = t6250 < 1;
        float t6255 = 1.0 * t6253;
        float t6256 = t6255 * t6254;
        int t6257 = t6250;
        int t6258 = t6252 >= 512;
        int t6259 = t6252 < 1024;
        float t6260 = t6256 * t6258;
        float t6261 = t6260 * t6259;
        int t6262 = t6252 - 512;
        int t6263 = t6257 * 512;
        int t6264 = t6263 + t6262;
        float t6265 = 0.0;
        if (t6261) {
          int t6267 = i;
          int t6268 = t6267 * 512;
          int t6269 = t6268 + t6264;
          float t6270 = memory[118257663 + t6269];
          t6265 = t6270;
        }
        float t6272 = t6243 + t6265;
        int t6273 = i;
        int t6274 = t6273 * 1024;
        int t6275 = t6274 + t6178;
        memory[149190655 + t6275] = t6272;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
      }
    }
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      for (int t89 = 0; t89 < 1024; t89+=1) {
        int t6277 = i;
        int t6278 = t6277 * 1024;
        int t6279 = t6278 + t89;
        float t6280 = memory[158365695 + t6279];
        int t6281 = i;
        int t6282 = t6281 * 1024;
        int t6283 = t6282 + t89;
        float t6284 = memory[170948607 + t6283];
        int t6285 = i;
        int t6286 = t6285 * 1024;
        int t6287 = t6286 + t89;
        float t6288 = memory[145258495 + t6287];
        float t6289 = t6284 * t6288;
        int t6290 = i;
        int t6291 = t6290 * 1024;
        int t6292 = t6291 + t89;
        float t6293 = memory[170948607 + t6292];
        int t6294 = i;
        int t6295 = t6294 * 1024;
        int t6296 = t6295 + t89;
        float t6297 = memory[149190655 + t6296];
        float t6298 = t6293 * t6297;
        int t6299 = i;
        int t6300 = t6299 * 1024;
        int t6301 = t6300 + t89;
        float t6302 = memory[158365695 + t6301];
        int t6303 = i;
        int t6304 = t6303 * 1024;
        int t6305 = t6304 + t89;
        float t6306 = memory[145258495 + t6305];
        float t6307 = t6302 * t6306;
        float t6308 = t6289;
        int t6309 = i;
        int t6310 = t6309 * 1024;
        int t6311 = t6310 + t89;
        memory[131627007 + t6311] = t6308;
        float t6313 = t6298 + t6307;
        int t6314 = i;
        int t6315 = t6314 * 1024;
        int t6316 = t6315 + t89;
        memory[127170559 + t6316] = t6313;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mtranspose [0m([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mtranspose [0m([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 2, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([512, 1]), value: empty) */
      }
    }
  }
  for (int t90 = 0; t90 < 1; t90+=1) {
  }
  for (int simd91 = 0; simd91 < 512; simd91+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
  }
  for (int t92 = 0; t92 < 1; t92+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      float32x4_t simd150 = vld1q_f32(t150 + i); /* extra */
    t150[i] = t150[i];
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([512, 1]), value: empty) */
      for (int t6322 = 0; t6322 < 512; t6322++) {
        int t6323 = t6322;
        int t6324 = t6323;
        int t6325 = t6322 - t6324;
        int t6326 = t6323;
        int t6327 = t6326;
        int t6328 = t6323 - t6327;
        int t6329 = t6328;
        int t6330 = t6329;
        int t6331 = t6328 - t6330;
        int t6332 = t6326 * 2;
        int t6333 = 1 + t6332;
        int t6334 = t6333 / 512;
        int t6335 = t6334 * 512;
        int t6336 = t6333 - t6335;
        int t6337 = t6336 / 256;
        int t6338 = t6337 * 256;
        int t6339 = t6336 - t6338;
        int t6340 = t6339 / 128;
        int t6341 = t6340 * 128;
        int t6342 = t6339 - t6341;
        int t6343 = t6342 / 64;
        int t6344 = t6343 * 64;
        int t6345 = t6342 - t6344;
        int t6346 = t6345 / 32;
        int t6347 = t6346 * 32;
        int t6348 = t6345 - t6347;
        int t6349 = t6348 / 16;
        int t6350 = t6349 * 16;
        int t6351 = t6348 - t6350;
        int t6352 = t6351 / 8;
        int t6353 = t6352 * 8;
        int t6354 = t6351 - t6353;
        int t6355 = t6354 / 4;
        int t6356 = t6355 * 4;
        int t6357 = t6354 - t6356;
        int t6358 = t6357 / 2;
        int t6359 = t6358 * 2;
        int t6360 = t6357 - t6359;
        int t6361 = t6337 * 2;
        int t6362 = t6334 + t6361;
        int t6363 = t6340 * 4;
        int t6364 = t6362 + t6363;
        int t6365 = t6343 * 8;
        int t6366 = t6364 + t6365;
        int t6367 = t6346 * 16;
        int t6368 = t6366 + t6367;
        int t6369 = t6349 * 32;
        int t6370 = t6368 + t6369;
        int t6371 = t6352 * 64;
        int t6372 = t6370 + t6371;
        int t6373 = t6355 * 128;
        int t6374 = t6372 + t6373;
        int t6375 = t6358 * 256;
        int t6376 = t6374 + t6375;
        int t6377 = t6360 * 512;
        int t6378 = t6376 + t6377;
        int t6379 = i;
        int t6380 = t6379 * 1024;
        int t6381 = t6380 + t6378;
        float t6382 = memory[131627007 + t6381];
        int t6383 = t6322;
        int t6384 = t6383;
        int t6385 = t6322 - t6384;
        float t6386 = memory[6140 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t6387 = t6382 * t6386;
        int t6388 = t6322;
        int t6389 = t6388;
        int t6390 = t6322 - t6389;
        int t6391 = t6388;
        int t6392 = t6391;
        int t6393 = t6388 - t6392;
        int t6394 = t6393;
        int t6395 = t6394;
        int t6396 = t6393 - t6395;
        int t6397 = t6391 * 2;
        int t6398 = 1 + t6397;
        int t6399 = t6398 / 512;
        int t6400 = t6399 * 512;
        int t6401 = t6398 - t6400;
        int t6402 = t6401 / 256;
        int t6403 = t6402 * 256;
        int t6404 = t6401 - t6403;
        int t6405 = t6404 / 128;
        int t6406 = t6405 * 128;
        int t6407 = t6404 - t6406;
        int t6408 = t6407 / 64;
        int t6409 = t6408 * 64;
        int t6410 = t6407 - t6409;
        int t6411 = t6410 / 32;
        int t6412 = t6411 * 32;
        int t6413 = t6410 - t6412;
        int t6414 = t6413 / 16;
        int t6415 = t6414 * 16;
        int t6416 = t6413 - t6415;
        int t6417 = t6416 / 8;
        int t6418 = t6417 * 8;
        int t6419 = t6416 - t6418;
        int t6420 = t6419 / 4;
        int t6421 = t6420 * 4;
        int t6422 = t6419 - t6421;
        int t6423 = t6422 / 2;
        int t6424 = t6423 * 2;
        int t6425 = t6422 - t6424;
        int t6426 = t6402 * 2;
        int t6427 = t6399 + t6426;
        int t6428 = t6405 * 4;
        int t6429 = t6427 + t6428;
        int t6430 = t6408 * 8;
        int t6431 = t6429 + t6430;
        int t6432 = t6411 * 16;
        int t6433 = t6431 + t6432;
        int t6434 = t6414 * 32;
        int t6435 = t6433 + t6434;
        int t6436 = t6417 * 64;
        int t6437 = t6435 + t6436;
        int t6438 = t6420 * 128;
        int t6439 = t6437 + t6438;
        int t6440 = t6423 * 256;
        int t6441 = t6439 + t6440;
        int t6442 = t6425 * 512;
        int t6443 = t6441 + t6442;
        int t6444 = i;
        int t6445 = t6444 * 1024;
        int t6446 = t6445 + t6443;
        float t6447 = memory[127170559 + t6446];
        int t6448 = t6322;
        int t6449 = t6448;
        int t6450 = t6322 - t6449;
        float t6451 = memory[6141 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t6452 = t6447 * t6451;
        float t6453 = t6387 - t6452;
        int t6454 = t6322;
        int t6455 = t6454;
        int t6456 = t6322 - t6455;
        int t6457 = t6454;
        int t6458 = t6457;
        int t6459 = t6454 - t6458;
        int t6460 = t6459;
        int t6461 = t6460;
        int t6462 = t6459 - t6461;
        int t6463 = t6457 * 2;
        int t6464 = 1 + t6463;
        int t6465 = t6464 / 512;
        int t6466 = t6465 * 512;
        int t6467 = t6464 - t6466;
        int t6468 = t6467 / 256;
        int t6469 = t6468 * 256;
        int t6470 = t6467 - t6469;
        int t6471 = t6470 / 128;
        int t6472 = t6471 * 128;
        int t6473 = t6470 - t6472;
        int t6474 = t6473 / 64;
        int t6475 = t6474 * 64;
        int t6476 = t6473 - t6475;
        int t6477 = t6476 / 32;
        int t6478 = t6477 * 32;
        int t6479 = t6476 - t6478;
        int t6480 = t6479 / 16;
        int t6481 = t6480 * 16;
        int t6482 = t6479 - t6481;
        int t6483 = t6482 / 8;
        int t6484 = t6483 * 8;
        int t6485 = t6482 - t6484;
        int t6486 = t6485 / 4;
        int t6487 = t6486 * 4;
        int t6488 = t6485 - t6487;
        int t6489 = t6488 / 2;
        int t6490 = t6489 * 2;
        int t6491 = t6488 - t6490;
        int t6492 = t6468 * 2;
        int t6493 = t6465 + t6492;
        int t6494 = t6471 * 4;
        int t6495 = t6493 + t6494;
        int t6496 = t6474 * 8;
        int t6497 = t6495 + t6496;
        int t6498 = t6477 * 16;
        int t6499 = t6497 + t6498;
        int t6500 = t6480 * 32;
        int t6501 = t6499 + t6500;
        int t6502 = t6483 * 64;
        int t6503 = t6501 + t6502;
        int t6504 = t6486 * 128;
        int t6505 = t6503 + t6504;
        int t6506 = t6489 * 256;
        int t6507 = t6505 + t6506;
        int t6508 = t6491 * 512;
        int t6509 = t6507 + t6508;
        int t6510 = i;
        int t6511 = t6510 * 1024;
        int t6512 = t6511 + t6509;
        float t6513 = memory[131627007 + t6512];
        int t6514 = t6322;
        int t6515 = t6514;
        int t6516 = t6322 - t6515;
        float t6517 = memory[6141 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t6518 = t6513 * t6517;
        int t6519 = t6322;
        int t6520 = t6519;
        int t6521 = t6322 - t6520;
        int t6522 = t6519;
        int t6523 = t6522;
        int t6524 = t6519 - t6523;
        int t6525 = t6524;
        int t6526 = t6525;
        int t6527 = t6524 - t6526;
        int t6528 = t6522 * 2;
        int t6529 = 1 + t6528;
        int t6530 = t6529 / 512;
        int t6531 = t6530 * 512;
        int t6532 = t6529 - t6531;
        int t6533 = t6532 / 256;
        int t6534 = t6533 * 256;
        int t6535 = t6532 - t6534;
        int t6536 = t6535 / 128;
        int t6537 = t6536 * 128;
        int t6538 = t6535 - t6537;
        int t6539 = t6538 / 64;
        int t6540 = t6539 * 64;
        int t6541 = t6538 - t6540;
        int t6542 = t6541 / 32;
        int t6543 = t6542 * 32;
        int t6544 = t6541 - t6543;
        int t6545 = t6544 / 16;
        int t6546 = t6545 * 16;
        int t6547 = t6544 - t6546;
        int t6548 = t6547 / 8;
        int t6549 = t6548 * 8;
        int t6550 = t6547 - t6549;
        int t6551 = t6550 / 4;
        int t6552 = t6551 * 4;
        int t6553 = t6550 - t6552;
        int t6554 = t6553 / 2;
        int t6555 = t6554 * 2;
        int t6556 = t6553 - t6555;
        int t6557 = t6533 * 2;
        int t6558 = t6530 + t6557;
        int t6559 = t6536 * 4;
        int t6560 = t6558 + t6559;
        int t6561 = t6539 * 8;
        int t6562 = t6560 + t6561;
        int t6563 = t6542 * 16;
        int t6564 = t6562 + t6563;
        int t6565 = t6545 * 32;
        int t6566 = t6564 + t6565;
        int t6567 = t6548 * 64;
        int t6568 = t6566 + t6567;
        int t6569 = t6551 * 128;
        int t6570 = t6568 + t6569;
        int t6571 = t6554 * 256;
        int t6572 = t6570 + t6571;
        int t6573 = t6556 * 512;
        int t6574 = t6572 + t6573;
        int t6575 = i;
        int t6576 = t6575 * 1024;
        int t6577 = t6576 + t6574;
        float t6578 = memory[127170559 + t6577];
        int t6579 = t6322;
        int t6580 = t6579;
        int t6581 = t6322 - t6580;
        float t6582 = memory[6140 + (isfinite((int) 0.0) ? (int) 0.0 : 0)];
        float t6583 = t6578 * t6582;
        float t6584 = t6518 + t6583;
        int t6585 = t6322;
        int t6586 = t6585;
        int t6587 = t6322 - t6586;
        int t6588 = t6585;
        int t6589 = t6588;
        int t6590 = t6585 - t6589;
        int t6591 = t6590;
        int t6592 = t6591;
        int t6593 = t6590 - t6592;
        int t6594 = t6588 * 2;
        int t6595 = t6594 / 512;
        int t6596 = t6595 * 512;
        int t6597 = t6594 - t6596;
        int t6598 = t6597 / 256;
        int t6599 = t6598 * 256;
        int t6600 = t6597 - t6599;
        int t6601 = t6600 / 128;
        int t6602 = t6601 * 128;
        int t6603 = t6600 - t6602;
        int t6604 = t6603 / 64;
        int t6605 = t6604 * 64;
        int t6606 = t6603 - t6605;
        int t6607 = t6606 / 32;
        int t6608 = t6607 * 32;
        int t6609 = t6606 - t6608;
        int t6610 = t6609 / 16;
        int t6611 = t6610 * 16;
        int t6612 = t6609 - t6611;
        int t6613 = t6612 / 8;
        int t6614 = t6613 * 8;
        int t6615 = t6612 - t6614;
        int t6616 = t6615 / 4;
        int t6617 = t6616 * 4;
        int t6618 = t6615 - t6617;
        int t6619 = t6618 / 2;
        int t6620 = t6619 * 2;
        int t6621 = t6618 - t6620;
        int t6622 = t6598 * 2;
        int t6623 = t6595 + t6622;
        int t6624 = t6601 * 4;
        int t6625 = t6623 + t6624;
        int t6626 = t6604 * 8;
        int t6627 = t6625 + t6626;
        int t6628 = t6607 * 16;
        int t6629 = t6627 + t6628;
        int t6630 = t6610 * 32;
        int t6631 = t6629 + t6630;
        int t6632 = t6613 * 64;
        int t6633 = t6631 + t6632;
        int t6634 = t6616 * 128;
        int t6635 = t6633 + t6634;
        int t6636 = t6619 * 256;
        int t6637 = t6635 + t6636;
        int t6638 = t6621 * 512;
        int t6639 = t6637 + t6638;
        int t6640 = i;
        int t6641 = t6640 * 1024;
        int t6642 = t6641 + t6639;
        float t6643 = memory[131627007 + t6642];
        float t6644 = t6643 + t6453;
        int t6645 = i;
        int t6646 = t6645 * 512;
        int t6647 = t6646 + t6322;
        memory[154171391 + t6647] = t6644;
        int t6649 = t6322;
        int t6650 = t6649;
        int t6651 = t6322 - t6650;
        int t6652 = t6649;
        int t6653 = t6652;
        int t6654 = t6649 - t6653;
        int t6655 = t6654;
        int t6656 = t6655;
        int t6657 = t6654 - t6656;
        int t6658 = t6652 * 2;
        int t6659 = t6658 / 512;
        int t6660 = t6659 * 512;
        int t6661 = t6658 - t6660;
        int t6662 = t6661 / 256;
        int t6663 = t6662 * 256;
        int t6664 = t6661 - t6663;
        int t6665 = t6664 / 128;
        int t6666 = t6665 * 128;
        int t6667 = t6664 - t6666;
        int t6668 = t6667 / 64;
        int t6669 = t6668 * 64;
        int t6670 = t6667 - t6669;
        int t6671 = t6670 / 32;
        int t6672 = t6671 * 32;
        int t6673 = t6670 - t6672;
        int t6674 = t6673 / 16;
        int t6675 = t6674 * 16;
        int t6676 = t6673 - t6675;
        int t6677 = t6676 / 8;
        int t6678 = t6677 * 8;
        int t6679 = t6676 - t6678;
        int t6680 = t6679 / 4;
        int t6681 = t6680 * 4;
        int t6682 = t6679 - t6681;
        int t6683 = t6682 / 2;
        int t6684 = t6683 * 2;
        int t6685 = t6682 - t6684;
        int t6686 = t6662 * 2;
        int t6687 = t6659 + t6686;
        int t6688 = t6665 * 4;
        int t6689 = t6687 + t6688;
        int t6690 = t6668 * 8;
        int t6691 = t6689 + t6690;
        int t6692 = t6671 * 16;
        int t6693 = t6691 + t6692;
        int t6694 = t6674 * 32;
        int t6695 = t6693 + t6694;
        int t6696 = t6677 * 64;
        int t6697 = t6695 + t6696;
        int t6698 = t6680 * 128;
        int t6699 = t6697 + t6698;
        int t6700 = t6683 * 256;
        int t6701 = t6699 + t6700;
        int t6702 = t6685 * 512;
        int t6703 = t6701 + t6702;
        int t6704 = i;
        int t6705 = t6704 * 1024;
        int t6706 = t6705 + t6703;
        float t6707 = memory[127170559 + t6706];
        float t6708 = t6707 + t6584;
        int t6709 = i;
        int t6710 = t6709 * 512;
        int t6711 = t6710 + t6322;
        memory[129529855 + t6711] = t6708;
        int t6713 = t6322;
        int t6714 = t6713;
        int t6715 = t6322 - t6714;
        int t6716 = t6713;
        int t6717 = t6716;
        int t6718 = t6713 - t6717;
        int t6719 = t6718;
        int t6720 = t6719;
        int t6721 = t6718 - t6720;
        int t6722 = t6716 * 2;
        int t6723 = t6722 / 512;
        int t6724 = t6723 * 512;
        int t6725 = t6722 - t6724;
        int t6726 = t6725 / 256;
        int t6727 = t6726 * 256;
        int t6728 = t6725 - t6727;
        int t6729 = t6728 / 128;
        int t6730 = t6729 * 128;
        int t6731 = t6728 - t6730;
        int t6732 = t6731 / 64;
        int t6733 = t6732 * 64;
        int t6734 = t6731 - t6733;
        int t6735 = t6734 / 32;
        int t6736 = t6735 * 32;
        int t6737 = t6734 - t6736;
        int t6738 = t6737 / 16;
        int t6739 = t6738 * 16;
        int t6740 = t6737 - t6739;
        int t6741 = t6740 / 8;
        int t6742 = t6741 * 8;
        int t6743 = t6740 - t6742;
        int t6744 = t6743 / 4;
        int t6745 = t6744 * 4;
        int t6746 = t6743 - t6745;
        int t6747 = t6746 / 2;
        int t6748 = t6747 * 2;
        int t6749 = t6746 - t6748;
        int t6750 = t6726 * 2;
        int t6751 = t6723 + t6750;
        int t6752 = t6729 * 4;
        int t6753 = t6751 + t6752;
        int t6754 = t6732 * 8;
        int t6755 = t6753 + t6754;
        int t6756 = t6735 * 16;
        int t6757 = t6755 + t6756;
        int t6758 = t6738 * 32;
        int t6759 = t6757 + t6758;
        int t6760 = t6741 * 64;
        int t6761 = t6759 + t6760;
        int t6762 = t6744 * 128;
        int t6763 = t6761 + t6762;
        int t6764 = t6747 * 256;
        int t6765 = t6763 + t6764;
        int t6766 = t6749 * 512;
        int t6767 = t6765 + t6766;
        int t6768 = i;
        int t6769 = t6768 * 1024;
        int t6770 = t6769 + t6767;
        float t6771 = memory[131627007 + t6770];
        float t6772 = t6771 - t6453;
        int t6773 = i;
        int t6774 = t6773 * 512;
        int t6775 = t6774 + t6322;
        memory[117733375 + t6775] = t6772;
        int t6777 = t6322;
        int t6778 = t6777;
        int t6779 = t6322 - t6778;
        int t6780 = t6777;
        int t6781 = t6780;
        int t6782 = t6777 - t6781;
        int t6783 = t6782;
        int t6784 = t6783;
        int t6785 = t6782 - t6784;
        int t6786 = t6780 * 2;
        int t6787 = t6786 / 512;
        int t6788 = t6787 * 512;
        int t6789 = t6786 - t6788;
        int t6790 = t6789 / 256;
        int t6791 = t6790 * 256;
        int t6792 = t6789 - t6791;
        int t6793 = t6792 / 128;
        int t6794 = t6793 * 128;
        int t6795 = t6792 - t6794;
        int t6796 = t6795 / 64;
        int t6797 = t6796 * 64;
        int t6798 = t6795 - t6797;
        int t6799 = t6798 / 32;
        int t6800 = t6799 * 32;
        int t6801 = t6798 - t6800;
        int t6802 = t6801 / 16;
        int t6803 = t6802 * 16;
        int t6804 = t6801 - t6803;
        int t6805 = t6804 / 8;
        int t6806 = t6805 * 8;
        int t6807 = t6804 - t6806;
        int t6808 = t6807 / 4;
        int t6809 = t6808 * 4;
        int t6810 = t6807 - t6809;
        int t6811 = t6810 / 2;
        int t6812 = t6811 * 2;
        int t6813 = t6810 - t6812;
        int t6814 = t6790 * 2;
        int t6815 = t6787 + t6814;
        int t6816 = t6793 * 4;
        int t6817 = t6815 + t6816;
        int t6818 = t6796 * 8;
        int t6819 = t6817 + t6818;
        int t6820 = t6799 * 16;
        int t6821 = t6819 + t6820;
        int t6822 = t6802 * 32;
        int t6823 = t6821 + t6822;
        int t6824 = t6805 * 64;
        int t6825 = t6823 + t6824;
        int t6826 = t6808 * 128;
        int t6827 = t6825 + t6826;
        int t6828 = t6811 * 256;
        int t6829 = t6827 + t6828;
        int t6830 = t6813 * 512;
        int t6831 = t6829 + t6830;
        int t6832 = i;
        int t6833 = t6832 * 1024;
        int t6834 = t6833 + t6831;
        float t6835 = memory[127170559 + t6834];
        float t6836 = t6835 - t6584;
        int t6837 = i;
        int t6838 = t6837 * 512;
        int t6839 = t6838 + t6322;
        memory[148404223 + t6839] = t6836;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
      }
      for (int t6841 = 0; t6841 < 1024; t6841++) {
        int t6842 = t6841 / 2;
        int t6843 = t6842 * 2;
        int t6844 = t6841 - t6843;
        int t6845 = t6842 >= 0;
        int t6846 = t6842 < 512;
        float t6847 = 1.0 * t6845;
        float t6848 = t6847 * t6846;
        int t6849 = t6842;
        int t6850 = t6844 >= 0;
        int t6851 = t6844 < 1;
        float t6852 = t6848 * t6850;
        float t6853 = t6852 * t6851;
        int t6854 = t6844;
        int t6855 = t6849 + t6854;
        float t6856 = 0.0;
        if (t6853) {
          int t6858 = i;
          int t6859 = t6858 * 512;
          int t6860 = t6859 + t6855;
          float t6861 = memory[154171391 + t6860];
          t6856 = t6861;
        }
        int t6863 = t6841 / 2;
        int t6864 = t6863 * 2;
        int t6865 = t6841 - t6864;
        int t6866 = t6863 >= 0;
        int t6867 = t6863 < 512;
        float t6868 = 1.0 * t6866;
        float t6869 = t6868 * t6867;
        int t6870 = t6863;
        int t6871 = t6865 >= 1;
        int t6872 = t6865 < 2;
        float t6873 = t6869 * t6871;
        float t6874 = t6873 * t6872;
        int t6875 = t6865 - 1;
        int t6876 = t6870 + t6875;
        float t6877 = 0.0;
        if (t6874) {
          int t6879 = i;
          int t6880 = t6879 * 512;
          int t6881 = t6880 + t6876;
          float t6882 = memory[117733375 + t6881];
          t6877 = t6882;
        }
        float t6884 = t6856 + t6877;
        int t6885 = i;
        int t6886 = t6885 * 1024;
        int t6887 = t6886 + t6841;
        memory[146831359 + t6887] = t6884;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 1)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (1, 0)]), value: empty) */
        int t6889 = t6841 / 2;
        int t6890 = t6889 * 2;
        int t6891 = t6841 - t6890;
        int t6892 = t6889 >= 0;
        int t6893 = t6889 < 512;
        float t6894 = 1.0 * t6892;
        float t6895 = t6894 * t6893;
        int t6896 = t6889;
        int t6897 = t6891 >= 0;
        int t6898 = t6891 < 1;
        float t6899 = t6895 * t6897;
        float t6900 = t6899 * t6898;
        int t6901 = t6891;
        int t6902 = t6896 + t6901;
        float t6903 = 0.0;
        if (t6900) {
          int t6905 = i;
          int t6906 = t6905 * 512;
          int t6907 = t6906 + t6902;
          float t6908 = memory[129529855 + t6907];
          t6903 = t6908;
        }
        int t6910 = t6841 / 2;
        int t6911 = t6910 * 2;
        int t6912 = t6841 - t6911;
        int t6913 = t6910 >= 0;
        int t6914 = t6910 < 512;
        float t6915 = 1.0 * t6913;
        float t6916 = t6915 * t6914;
        int t6917 = t6910;
        int t6918 = t6912 >= 1;
        int t6919 = t6912 < 2;
        float t6920 = t6916 * t6918;
        float t6921 = t6920 * t6919;
        int t6922 = t6912 - 1;
        int t6923 = t6917 + t6922;
        float t6924 = 0.0;
        if (t6921) {
          int t6926 = i;
          int t6927 = t6926 * 512;
          int t6928 = t6927 + t6923;
          float t6929 = memory[148404223 + t6928];
          t6924 = t6929;
        }
        float t6931 = t6903 + t6924;
        int t6932 = i;
        int t6933 = t6932 * 1024;
        int t6934 = t6933 + t6841;
        memory[147879935 + t6934] = t6931;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([256, 2]), value: empty) */
      }
    }
  }
  for (int t94 = 0; t94 < 2; t94+=1) {
  }
  for (int simd95 = 0; simd95 < 512; simd95+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
  }
  for (int t96 = 0; t96 < 2; t96+=1) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([256, 2]), value: empty) */
      for (int t6939 = 0; t6939 < 512; t6939++) {
        int t6940 = t6939 / 2;
        int t6941 = t6940 * 2;
        int t6942 = t6939 - t6941;
        int t6943 = t6940 * 2;
        int t6944 = t6943 + t6942;
        int t6945 = t6944 / 2;
        int t6946 = t6945 * 2;
        int t6947 = t6944 - t6946;
        int t6948 = t6947 / 2;
        int t6949 = t6948 * 2;
        int t6950 = t6947 - t6949;
        int t6951 = t6945 * 4;
        int t6952 = 2 + t6951;
        int t6953 = t6952 + t6950;
        int t6954 = i;
        int t6955 = t6954 * 1024;
        int t6956 = t6955 + t6953;
        float t6957 = memory[146831359 + t6956];
        int t6958 = t6939 / 2;
        int t6959 = t6958 * 2;
        int t6960 = t6939 - t6959;
        float t6961 = memory[6142 + t6960];
        float t6962 = t6957 * t6961;
        int t6963 = t6939 / 2;
        int t6964 = t6963 * 2;
        int t6965 = t6939 - t6964;
        int t6966 = t6963 * 2;
        int t6967 = t6966 + t6965;
        int t6968 = t6967 / 2;
        int t6969 = t6968 * 2;
        int t6970 = t6967 - t6969;
        int t6971 = t6970 / 2;
        int t6972 = t6971 * 2;
        int t6973 = t6970 - t6972;
        int t6974 = t6968 * 4;
        int t6975 = 2 + t6974;
        int t6976 = t6975 + t6973;
        int t6977 = i;
        int t6978 = t6977 * 1024;
        int t6979 = t6978 + t6976;
        float t6980 = memory[147879935 + t6979];
        int t6981 = t6939 / 2;
        int t6982 = t6981 * 2;
        int t6983 = t6939 - t6982;
        float t6984 = memory[6144 + t6983];
        float t6985 = t6980 * t6984;
        float t6986 = t6962 - t6985;
        int t6987 = t6939 / 2;
        int t6988 = t6987 * 2;
        int t6989 = t6939 - t6988;
        int t6990 = t6987 * 2;
        int t6991 = t6990 + t6989;
        int t6992 = t6991 / 2;
        int t6993 = t6992 * 2;
        int t6994 = t6991 - t6993;
        int t6995 = t6994 / 2;
        int t6996 = t6995 * 2;
        int t6997 = t6994 - t6996;
        int t6998 = t6992 * 4;
        int t6999 = 2 + t6998;
        int t7000 = t6999 + t6997;
        int t7001 = i;
        int t7002 = t7001 * 1024;
        int t7003 = t7002 + t7000;
        float t7004 = memory[146831359 + t7003];
        int t7005 = t6939 / 2;
        int t7006 = t7005 * 2;
        int t7007 = t6939 - t7006;
        float t7008 = memory[6144 + t7007];
        float t7009 = t7004 * t7008;
        int t7010 = t6939 / 2;
        int t7011 = t7010 * 2;
        int t7012 = t6939 - t7011;
        int t7013 = t7010 * 2;
        int t7014 = t7013 + t7012;
        int t7015 = t7014 / 2;
        int t7016 = t7015 * 2;
        int t7017 = t7014 - t7016;
        int t7018 = t7017 / 2;
        int t7019 = t7018 * 2;
        int t7020 = t7017 - t7019;
        int t7021 = t7015 * 4;
        int t7022 = 2 + t7021;
        int t7023 = t7022 + t7020;
        int t7024 = i;
        int t7025 = t7024 * 1024;
        int t7026 = t7025 + t7023;
        float t7027 = memory[147879935 + t7026];
        int t7028 = t6939 / 2;
        int t7029 = t7028 * 2;
        int t7030 = t6939 - t7029;
        float t7031 = memory[6142 + t7030];
        float t7032 = t7027 * t7031;
        float t7033 = t7009 + t7032;
        int t7034 = t6939 / 2;
        int t7035 = t7034 * 2;
        int t7036 = t6939 - t7035;
        int t7037 = t7034 * 2;
        int t7038 = t7037 + t7036;
        int t7039 = t7038 / 2;
        int t7040 = t7039 * 2;
        int t7041 = t7038 - t7040;
        int t7042 = t7041 / 2;
        int t7043 = t7042 * 2;
        int t7044 = t7041 - t7043;
        int t7045 = t7039 * 4;
        int t7046 = t7045 + t7044;
        int t7047 = i;
        int t7048 = t7047 * 1024;
        int t7049 = t7048 + t7046;
        float t7050 = memory[146831359 + t7049];
        float t7051 = t7050 + t6986;
        int t7052 = i;
        int t7053 = t7052 * 512;
        int t7054 = t7053 + t6939;
        memory[143161343 + t7054] = t7051;
        int t7056 = t6939 / 2;
        int t7057 = t7056 * 2;
        int t7058 = t6939 - t7057;
        int t7059 = t7056 * 2;
        int t7060 = t7059 + t7058;
        int t7061 = t7060 / 2;
        int t7062 = t7061 * 2;
        int t7063 = t7060 - t7062;
        int t7064 = t7063 / 2;
        int t7065 = t7064 * 2;
        int t7066 = t7063 - t7065;
        int t7067 = t7061 * 4;
        int t7068 = t7067 + t7066;
        int t7069 = i;
        int t7070 = t7069 * 1024;
        int t7071 = t7070 + t7068;
        float t7072 = memory[147879935 + t7071];
        float t7073 = t7072 + t7033;
        int t7074 = i;
        int t7075 = t7074 * 512;
        int t7076 = t7075 + t6939;
        memory[142637055 + t7076] = t7073;
        int t7078 = t6939 / 2;
        int t7079 = t7078 * 2;
        int t7080 = t6939 - t7079;
        int t7081 = t7078 * 2;
        int t7082 = t7081 + t7080;
        int t7083 = t7082 / 2;
        int t7084 = t7083 * 2;
        int t7085 = t7082 - t7084;
        int t7086 = t7085 / 2;
        int t7087 = t7086 * 2;
        int t7088 = t7085 - t7087;
        int t7089 = t7083 * 4;
        int t7090 = t7089 + t7088;
        int t7091 = i;
        int t7092 = t7091 * 1024;
        int t7093 = t7092 + t7090;
        float t7094 = memory[146831359 + t7093];
        float t7095 = t7094 - t6986;
        int t7096 = i;
        int t7097 = t7096 * 512;
        int t7098 = t7097 + t6939;
        memory[119568383 + t7098] = t7095;
        int t7100 = t6939 / 2;
        int t7101 = t7100 * 2;
        int t7102 = t6939 - t7101;
        int t7103 = t7100 * 2;
        int t7104 = t7103 + t7102;
        int t7105 = t7104 / 2;
        int t7106 = t7105 * 2;
        int t7107 = t7104 - t7106;
        int t7108 = t7107 / 2;
        int t7109 = t7108 * 2;
        int t7110 = t7107 - t7109;
        int t7111 = t7105 * 4;
        int t7112 = t7111 + t7110;
        int t7113 = i;
        int t7114 = t7113 * 1024;
        int t7115 = t7114 + t7112;
        float t7116 = memory[147879935 + t7115];
        float t7117 = t7116 - t7033;
        int t7118 = i;
        int t7119 = t7118 * 512;
        int t7120 = t7119 + t6939;
        memory[147355647 + t7120] = t7117;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
      }
      for (int t7122 = 0; t7122 < 1024; t7122++) {
        int t7123 = t7122 / 4;
        int t7124 = t7123 * 4;
        int t7125 = t7122 - t7124;
        int t7126 = t7123 >= 0;
        int t7127 = t7123 < 256;
        float t7128 = 1.0 * t7126;
        float t7129 = t7128 * t7127;
        int t7130 = t7123;
        int t7131 = t7125 >= 0;
        int t7132 = t7125 < 2;
        float t7133 = t7129 * t7131;
        float t7134 = t7133 * t7132;
        int t7135 = t7125;
        int t7136 = t7130 * 2;
        int t7137 = t7136 + t7135;
        float t7138 = 0.0;
        if (t7134) {
          int t7140 = i;
          int t7141 = t7140 * 512;
          int t7142 = t7141 + t7137;
          float t7143 = memory[143161343 + t7142];
          t7138 = t7143;
        }
        int t7145 = t7122 / 4;
        int t7146 = t7145 * 4;
        int t7147 = t7122 - t7146;
        int t7148 = t7145 >= 0;
        int t7149 = t7145 < 256;
        float t7150 = 1.0 * t7148;
        float t7151 = t7150 * t7149;
        int t7152 = t7145;
        int t7153 = t7147 >= 2;
        int t7154 = t7147 < 4;
        float t7155 = t7151 * t7153;
        float t7156 = t7155 * t7154;
        int t7157 = t7147 - 2;
        int t7158 = t7152 * 2;
        int t7159 = t7158 + t7157;
        float t7160 = 0.0;
        if (t7156) {
          int t7162 = i;
          int t7163 = t7162 * 512;
          int t7164 = t7163 + t7159;
          float t7165 = memory[119568383 + t7164];
          t7160 = t7165;
        }
        float t7167 = t7138 + t7160;
        int t7168 = i;
        int t7169 = t7168 * 1024;
        int t7170 = t7169 + t7122;
        memory[162822143 + t7170] = t7167;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 2)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (2, 0)]), value: empty) */
        int t7172 = t7122 / 4;
        int t7173 = t7172 * 4;
        int t7174 = t7122 - t7173;
        int t7175 = t7172 >= 0;
        int t7176 = t7172 < 256;
        float t7177 = 1.0 * t7175;
        float t7178 = t7177 * t7176;
        int t7179 = t7172;
        int t7180 = t7174 >= 0;
        int t7181 = t7174 < 2;
        float t7182 = t7178 * t7180;
        float t7183 = t7182 * t7181;
        int t7184 = t7174;
        int t7185 = t7179 * 2;
        int t7186 = t7185 + t7184;
        float t7187 = 0.0;
        if (t7183) {
          int t7189 = i;
          int t7190 = t7189 * 512;
          int t7191 = t7190 + t7186;
          float t7192 = memory[142637055 + t7191];
          t7187 = t7192;
        }
        int t7194 = t7122 / 4;
        int t7195 = t7194 * 4;
        int t7196 = t7122 - t7195;
        int t7197 = t7194 >= 0;
        int t7198 = t7194 < 256;
        float t7199 = 1.0 * t7197;
        float t7200 = t7199 * t7198;
        int t7201 = t7194;
        int t7202 = t7196 >= 2;
        int t7203 = t7196 < 4;
        float t7204 = t7200 * t7202;
        float t7205 = t7204 * t7203;
        int t7206 = t7196 - 2;
        int t7207 = t7201 * 2;
        int t7208 = t7207 + t7206;
        float t7209 = 0.0;
        if (t7205) {
          int t7211 = i;
          int t7212 = t7211 * 512;
          int t7213 = t7212 + t7208;
          float t7214 = memory[147355647 + t7213];
          t7209 = t7214;
        }
        float t7216 = t7187 + t7209;
        int t7217 = i;
        int t7218 = t7217 * 1024;
        int t7219 = t7218 + t7122;
        memory[167278591 + t7219] = t7216;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 2, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([128, 4]), value: empty) */
      }
    }
  }
  for (int simd98 = 0; simd98 < 4; simd98+=4) {
  }
  for (int simd99 = 0; simd99 < 512; simd99+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
  }
  for (int simd100 = 0; simd100 < 4; simd100+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([128, 4]), value: empty) */
      for (int t7224 = 0; t7224 < 512; t7224++) {
        int t7225 = t7224 / 4;
        int t7226 = t7225 * 4;
        int t7227 = t7224 - t7226;
        int t7228 = t7225 * 4;
        int t7229 = t7228 + t7227;
        int t7230 = t7229 / 4;
        int t7231 = t7230 * 4;
        int t7232 = t7229 - t7231;
        int t7233 = t7232 / 4;
        int t7234 = t7233 * 4;
        int t7235 = t7232 - t7234;
        int t7236 = t7230 * 8;
        int t7237 = 4 + t7236;
        int t7238 = t7237 + t7235;
        int t7239 = i;
        int t7240 = t7239 * 1024;
        int t7241 = t7240 + t7238;
        float t7242 = memory[162822143 + t7241];
        int t7243 = t7224 / 4;
        int t7244 = t7243 * 4;
        int t7245 = t7224 - t7244;
        float t7246 = memory[6146 + t7245];
        float t7247 = t7242 * t7246;
        int t7248 = t7224 / 4;
        int t7249 = t7248 * 4;
        int t7250 = t7224 - t7249;
        int t7251 = t7248 * 4;
        int t7252 = t7251 + t7250;
        int t7253 = t7252 / 4;
        int t7254 = t7253 * 4;
        int t7255 = t7252 - t7254;
        int t7256 = t7255 / 4;
        int t7257 = t7256 * 4;
        int t7258 = t7255 - t7257;
        int t7259 = t7253 * 8;
        int t7260 = 4 + t7259;
        int t7261 = t7260 + t7258;
        int t7262 = i;
        int t7263 = t7262 * 1024;
        int t7264 = t7263 + t7261;
        float t7265 = memory[167278591 + t7264];
        int t7266 = t7224 / 4;
        int t7267 = t7266 * 4;
        int t7268 = t7224 - t7267;
        float t7269 = memory[6150 + t7268];
        float t7270 = t7265 * t7269;
        float t7271 = t7247 - t7270;
        int t7272 = t7224 / 4;
        int t7273 = t7272 * 4;
        int t7274 = t7224 - t7273;
        int t7275 = t7272 * 4;
        int t7276 = t7275 + t7274;
        int t7277 = t7276 / 4;
        int t7278 = t7277 * 4;
        int t7279 = t7276 - t7278;
        int t7280 = t7279 / 4;
        int t7281 = t7280 * 4;
        int t7282 = t7279 - t7281;
        int t7283 = t7277 * 8;
        int t7284 = 4 + t7283;
        int t7285 = t7284 + t7282;
        int t7286 = i;
        int t7287 = t7286 * 1024;
        int t7288 = t7287 + t7285;
        float t7289 = memory[162822143 + t7288];
        int t7290 = t7224 / 4;
        int t7291 = t7290 * 4;
        int t7292 = t7224 - t7291;
        float t7293 = memory[6150 + t7292];
        float t7294 = t7289 * t7293;
        int t7295 = t7224 / 4;
        int t7296 = t7295 * 4;
        int t7297 = t7224 - t7296;
        int t7298 = t7295 * 4;
        int t7299 = t7298 + t7297;
        int t7300 = t7299 / 4;
        int t7301 = t7300 * 4;
        int t7302 = t7299 - t7301;
        int t7303 = t7302 / 4;
        int t7304 = t7303 * 4;
        int t7305 = t7302 - t7304;
        int t7306 = t7300 * 8;
        int t7307 = 4 + t7306;
        int t7308 = t7307 + t7305;
        int t7309 = i;
        int t7310 = t7309 * 1024;
        int t7311 = t7310 + t7308;
        float t7312 = memory[167278591 + t7311];
        int t7313 = t7224 / 4;
        int t7314 = t7313 * 4;
        int t7315 = t7224 - t7314;
        float t7316 = memory[6146 + t7315];
        float t7317 = t7312 * t7316;
        float t7318 = t7294 + t7317;
        int t7319 = t7224 / 4;
        int t7320 = t7319 * 4;
        int t7321 = t7224 - t7320;
        int t7322 = t7319 * 4;
        int t7323 = t7322 + t7321;
        int t7324 = t7323 / 4;
        int t7325 = t7324 * 4;
        int t7326 = t7323 - t7325;
        int t7327 = t7326 / 4;
        int t7328 = t7327 * 4;
        int t7329 = t7326 - t7328;
        int t7330 = t7324 * 8;
        int t7331 = t7330 + t7329;
        int t7332 = i;
        int t7333 = t7332 * 1024;
        int t7334 = t7333 + t7331;
        float t7335 = memory[162822143 + t7334];
        float t7336 = t7335 + t7271;
        int t7337 = i;
        int t7338 = t7337 * 512;
        int t7339 = t7338 + t7224;
        memory[164132863 + t7339] = t7336;
        int t7341 = t7224 / 4;
        int t7342 = t7341 * 4;
        int t7343 = t7224 - t7342;
        int t7344 = t7341 * 4;
        int t7345 = t7344 + t7343;
        int t7346 = t7345 / 4;
        int t7347 = t7346 * 4;
        int t7348 = t7345 - t7347;
        int t7349 = t7348 / 4;
        int t7350 = t7349 * 4;
        int t7351 = t7348 - t7350;
        int t7352 = t7346 * 8;
        int t7353 = t7352 + t7351;
        int t7354 = i;
        int t7355 = t7354 * 1024;
        int t7356 = t7355 + t7353;
        float t7357 = memory[167278591 + t7356];
        float t7358 = t7357 + t7318;
        int t7359 = i;
        int t7360 = t7359 * 512;
        int t7361 = t7360 + t7224;
        memory[129005567 + t7361] = t7358;
        int t7363 = t7224 / 4;
        int t7364 = t7363 * 4;
        int t7365 = t7224 - t7364;
        int t7366 = t7363 * 4;
        int t7367 = t7366 + t7365;
        int t7368 = t7367 / 4;
        int t7369 = t7368 * 4;
        int t7370 = t7367 - t7369;
        int t7371 = t7370 / 4;
        int t7372 = t7371 * 4;
        int t7373 = t7370 - t7372;
        int t7374 = t7368 * 8;
        int t7375 = t7374 + t7373;
        int t7376 = i;
        int t7377 = t7376 * 1024;
        int t7378 = t7377 + t7375;
        float t7379 = memory[162822143 + t7378];
        float t7380 = t7379 - t7271;
        int t7381 = i;
        int t7382 = t7381 * 512;
        int t7383 = t7382 + t7224;
        memory[174094335 + t7383] = t7380;
        int t7385 = t7224 / 4;
        int t7386 = t7385 * 4;
        int t7387 = t7224 - t7386;
        int t7388 = t7385 * 4;
        int t7389 = t7388 + t7387;
        int t7390 = t7389 / 4;
        int t7391 = t7390 * 4;
        int t7392 = t7389 - t7391;
        int t7393 = t7392 / 4;
        int t7394 = t7393 * 4;
        int t7395 = t7392 - t7394;
        int t7396 = t7390 * 8;
        int t7397 = t7396 + t7395;
        int t7398 = i;
        int t7399 = t7398 * 1024;
        int t7400 = t7399 + t7397;
        float t7401 = memory[167278591 + t7400];
        float t7402 = t7401 - t7318;
        int t7403 = i;
        int t7404 = t7403 * 512;
        int t7405 = t7404 + t7224;
        memory[119830527 + t7405] = t7402;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
      }
      for (int t7407 = 0; t7407 < 1024; t7407++) {
        int t7408 = t7407 / 8;
        int t7409 = t7408 * 8;
        int t7410 = t7407 - t7409;
        int t7411 = t7408 >= 0;
        int t7412 = t7408 < 128;
        float t7413 = 1.0 * t7411;
        float t7414 = t7413 * t7412;
        int t7415 = t7408;
        int t7416 = t7410 >= 0;
        int t7417 = t7410 < 4;
        float t7418 = t7414 * t7416;
        float t7419 = t7418 * t7417;
        int t7420 = t7410;
        int t7421 = t7415 * 4;
        int t7422 = t7421 + t7420;
        float t7423 = 0.0;
        if (t7419) {
          int t7425 = i;
          int t7426 = t7425 * 512;
          int t7427 = t7426 + t7422;
          float t7428 = memory[164132863 + t7427];
          t7423 = t7428;
        }
        int t7430 = t7407 / 8;
        int t7431 = t7430 * 8;
        int t7432 = t7407 - t7431;
        int t7433 = t7430 >= 0;
        int t7434 = t7430 < 128;
        float t7435 = 1.0 * t7433;
        float t7436 = t7435 * t7434;
        int t7437 = t7430;
        int t7438 = t7432 >= 4;
        int t7439 = t7432 < 8;
        float t7440 = t7436 * t7438;
        float t7441 = t7440 * t7439;
        int t7442 = t7432 - 4;
        int t7443 = t7437 * 4;
        int t7444 = t7443 + t7442;
        float t7445 = 0.0;
        if (t7441) {
          int t7447 = i;
          int t7448 = t7447 * 512;
          int t7449 = t7448 + t7444;
          float t7450 = memory[174094335 + t7449];
          t7445 = t7450;
        }
        float t7452 = t7423 + t7445;
        int t7453 = i;
        int t7454 = t7453 * 1024;
        int t7455 = t7454 + t7407;
        memory[156006399 + t7455] = t7452;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 4)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (4, 0)]), value: empty) */
        int t7457 = t7407 / 8;
        int t7458 = t7457 * 8;
        int t7459 = t7407 - t7458;
        int t7460 = t7457 >= 0;
        int t7461 = t7457 < 128;
        float t7462 = 1.0 * t7460;
        float t7463 = t7462 * t7461;
        int t7464 = t7457;
        int t7465 = t7459 >= 0;
        int t7466 = t7459 < 4;
        float t7467 = t7463 * t7465;
        float t7468 = t7467 * t7466;
        int t7469 = t7459;
        int t7470 = t7464 * 4;
        int t7471 = t7470 + t7469;
        float t7472 = 0.0;
        if (t7468) {
          int t7474 = i;
          int t7475 = t7474 * 512;
          int t7476 = t7475 + t7471;
          float t7477 = memory[129005567 + t7476];
          t7472 = t7477;
        }
        int t7479 = t7407 / 8;
        int t7480 = t7479 * 8;
        int t7481 = t7407 - t7480;
        int t7482 = t7479 >= 0;
        int t7483 = t7479 < 128;
        float t7484 = 1.0 * t7482;
        float t7485 = t7484 * t7483;
        int t7486 = t7479;
        int t7487 = t7481 >= 4;
        int t7488 = t7481 < 8;
        float t7489 = t7485 * t7487;
        float t7490 = t7489 * t7488;
        int t7491 = t7481 - 4;
        int t7492 = t7486 * 4;
        int t7493 = t7492 + t7491;
        float t7494 = 0.0;
        if (t7490) {
          int t7496 = i;
          int t7497 = t7496 * 512;
          int t7498 = t7497 + t7493;
          float t7499 = memory[119830527 + t7498];
          t7494 = t7499;
        }
        float t7501 = t7472 + t7494;
        int t7502 = i;
        int t7503 = t7502 * 1024;
        int t7504 = t7503 + t7407;
        memory[136345599 + t7504] = t7501;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 2, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([64, 8]), value: empty) */
      }
    }
  }
  for (int simd102 = 0; simd102 < 8; simd102+=4) {
  }
  for (int simd103 = 0; simd103 < 512; simd103+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
  }
  for (int simd104 = 0; simd104 < 8; simd104+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([64, 8]), value: empty) */
      for (int t7509 = 0; t7509 < 512; t7509++) {
        int t7510 = t7509 / 8;
        int t7511 = t7510 * 8;
        int t7512 = t7509 - t7511;
        int t7513 = t7510 * 8;
        int t7514 = t7513 + t7512;
        int t7515 = t7514 / 8;
        int t7516 = t7515 * 8;
        int t7517 = t7514 - t7516;
        int t7518 = t7517 / 8;
        int t7519 = t7518 * 8;
        int t7520 = t7517 - t7519;
        int t7521 = t7515 * 16;
        int t7522 = 8 + t7521;
        int t7523 = t7522 + t7520;
        int t7524 = i;
        int t7525 = t7524 * 1024;
        int t7526 = t7525 + t7523;
        float t7527 = memory[156006399 + t7526];
        int t7528 = t7509 / 8;
        int t7529 = t7528 * 8;
        int t7530 = t7509 - t7529;
        float t7531 = memory[6154 + t7530];
        float t7532 = t7527 * t7531;
        int t7533 = t7509 / 8;
        int t7534 = t7533 * 8;
        int t7535 = t7509 - t7534;
        int t7536 = t7533 * 8;
        int t7537 = t7536 + t7535;
        int t7538 = t7537 / 8;
        int t7539 = t7538 * 8;
        int t7540 = t7537 - t7539;
        int t7541 = t7540 / 8;
        int t7542 = t7541 * 8;
        int t7543 = t7540 - t7542;
        int t7544 = t7538 * 16;
        int t7545 = 8 + t7544;
        int t7546 = t7545 + t7543;
        int t7547 = i;
        int t7548 = t7547 * 1024;
        int t7549 = t7548 + t7546;
        float t7550 = memory[136345599 + t7549];
        int t7551 = t7509 / 8;
        int t7552 = t7551 * 8;
        int t7553 = t7509 - t7552;
        float t7554 = memory[6162 + t7553];
        float t7555 = t7550 * t7554;
        float t7556 = t7532 - t7555;
        int t7557 = t7509 / 8;
        int t7558 = t7557 * 8;
        int t7559 = t7509 - t7558;
        int t7560 = t7557 * 8;
        int t7561 = t7560 + t7559;
        int t7562 = t7561 / 8;
        int t7563 = t7562 * 8;
        int t7564 = t7561 - t7563;
        int t7565 = t7564 / 8;
        int t7566 = t7565 * 8;
        int t7567 = t7564 - t7566;
        int t7568 = t7562 * 16;
        int t7569 = 8 + t7568;
        int t7570 = t7569 + t7567;
        int t7571 = i;
        int t7572 = t7571 * 1024;
        int t7573 = t7572 + t7570;
        float t7574 = memory[156006399 + t7573];
        int t7575 = t7509 / 8;
        int t7576 = t7575 * 8;
        int t7577 = t7509 - t7576;
        float t7578 = memory[6162 + t7577];
        float t7579 = t7574 * t7578;
        int t7580 = t7509 / 8;
        int t7581 = t7580 * 8;
        int t7582 = t7509 - t7581;
        int t7583 = t7580 * 8;
        int t7584 = t7583 + t7582;
        int t7585 = t7584 / 8;
        int t7586 = t7585 * 8;
        int t7587 = t7584 - t7586;
        int t7588 = t7587 / 8;
        int t7589 = t7588 * 8;
        int t7590 = t7587 - t7589;
        int t7591 = t7585 * 16;
        int t7592 = 8 + t7591;
        int t7593 = t7592 + t7590;
        int t7594 = i;
        int t7595 = t7594 * 1024;
        int t7596 = t7595 + t7593;
        float t7597 = memory[136345599 + t7596];
        int t7598 = t7509 / 8;
        int t7599 = t7598 * 8;
        int t7600 = t7509 - t7599;
        float t7601 = memory[6154 + t7600];
        float t7602 = t7597 * t7601;
        float t7603 = t7579 + t7602;
        int t7604 = t7509 / 8;
        int t7605 = t7604 * 8;
        int t7606 = t7509 - t7605;
        int t7607 = t7604 * 8;
        int t7608 = t7607 + t7606;
        int t7609 = t7608 / 8;
        int t7610 = t7609 * 8;
        int t7611 = t7608 - t7610;
        int t7612 = t7611 / 8;
        int t7613 = t7612 * 8;
        int t7614 = t7611 - t7613;
        int t7615 = t7609 * 16;
        int t7616 = t7615 + t7614;
        int t7617 = i;
        int t7618 = t7617 * 1024;
        int t7619 = t7618 + t7616;
        float t7620 = memory[156006399 + t7619];
        float t7621 = t7620 + t7556;
        int t7622 = i;
        int t7623 = t7622 * 512;
        int t7624 = t7623 + t7509;
        memory[156792831 + t7624] = t7621;
        int t7626 = t7509 / 8;
        int t7627 = t7626 * 8;
        int t7628 = t7509 - t7627;
        int t7629 = t7626 * 8;
        int t7630 = t7629 + t7628;
        int t7631 = t7630 / 8;
        int t7632 = t7631 * 8;
        int t7633 = t7630 - t7632;
        int t7634 = t7633 / 8;
        int t7635 = t7634 * 8;
        int t7636 = t7633 - t7635;
        int t7637 = t7631 * 16;
        int t7638 = t7637 + t7636;
        int t7639 = i;
        int t7640 = t7639 * 1024;
        int t7641 = t7640 + t7638;
        float t7642 = memory[136345599 + t7641];
        float t7643 = t7642 + t7603;
        int t7644 = i;
        int t7645 = t7644 * 512;
        int t7646 = t7645 + t7509;
        memory[122976255 + t7646] = t7643;
        int t7648 = t7509 / 8;
        int t7649 = t7648 * 8;
        int t7650 = t7509 - t7649;
        int t7651 = t7648 * 8;
        int t7652 = t7651 + t7650;
        int t7653 = t7652 / 8;
        int t7654 = t7653 * 8;
        int t7655 = t7652 - t7654;
        int t7656 = t7655 / 8;
        int t7657 = t7656 * 8;
        int t7658 = t7655 - t7657;
        int t7659 = t7653 * 16;
        int t7660 = t7659 + t7658;
        int t7661 = i;
        int t7662 = t7661 * 1024;
        int t7663 = t7662 + t7660;
        float t7664 = memory[156006399 + t7663];
        float t7665 = t7664 - t7556;
        int t7666 = i;
        int t7667 = t7666 * 512;
        int t7668 = t7667 + t7509;
        memory[136869887 + t7668] = t7665;
        int t7670 = t7509 / 8;
        int t7671 = t7670 * 8;
        int t7672 = t7509 - t7671;
        int t7673 = t7670 * 8;
        int t7674 = t7673 + t7672;
        int t7675 = t7674 / 8;
        int t7676 = t7675 * 8;
        int t7677 = t7674 - t7676;
        int t7678 = t7677 / 8;
        int t7679 = t7678 * 8;
        int t7680 = t7677 - t7679;
        int t7681 = t7675 * 16;
        int t7682 = t7681 + t7680;
        int t7683 = i;
        int t7684 = t7683 * 1024;
        int t7685 = t7684 + t7682;
        float t7686 = memory[136345599 + t7685];
        float t7687 = t7686 - t7603;
        int t7688 = i;
        int t7689 = t7688 * 512;
        int t7690 = t7689 + t7509;
        memory[159414271 + t7690] = t7687;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
      }
      for (int t7692 = 0; t7692 < 1024; t7692++) {
        int t7693 = t7692 / 16;
        int t7694 = t7693 * 16;
        int t7695 = t7692 - t7694;
        int t7696 = t7693 >= 0;
        int t7697 = t7693 < 64;
        float t7698 = 1.0 * t7696;
        float t7699 = t7698 * t7697;
        int t7700 = t7693;
        int t7701 = t7695 >= 0;
        int t7702 = t7695 < 8;
        float t7703 = t7699 * t7701;
        float t7704 = t7703 * t7702;
        int t7705 = t7695;
        int t7706 = t7700 * 8;
        int t7707 = t7706 + t7705;
        float t7708 = 0.0;
        if (t7704) {
          int t7710 = i;
          int t7711 = t7710 * 512;
          int t7712 = t7711 + t7707;
          float t7713 = memory[156792831 + t7712];
          t7708 = t7713;
        }
        int t7715 = t7692 / 16;
        int t7716 = t7715 * 16;
        int t7717 = t7692 - t7716;
        int t7718 = t7715 >= 0;
        int t7719 = t7715 < 64;
        float t7720 = 1.0 * t7718;
        float t7721 = t7720 * t7719;
        int t7722 = t7715;
        int t7723 = t7717 >= 8;
        int t7724 = t7717 < 16;
        float t7725 = t7721 * t7723;
        float t7726 = t7725 * t7724;
        int t7727 = t7717 - 8;
        int t7728 = t7722 * 8;
        int t7729 = t7728 + t7727;
        float t7730 = 0.0;
        if (t7726) {
          int t7732 = i;
          int t7733 = t7732 * 512;
          int t7734 = t7733 + t7729;
          float t7735 = memory[136869887 + t7734];
          t7730 = t7735;
        }
        float t7737 = t7708 + t7730;
        int t7738 = i;
        int t7739 = t7738 * 1024;
        int t7740 = t7739 + t7692;
        memory[141588479 + t7740] = t7737;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 8)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (8, 0)]), value: empty) */
        int t7742 = t7692 / 16;
        int t7743 = t7742 * 16;
        int t7744 = t7692 - t7743;
        int t7745 = t7742 >= 0;
        int t7746 = t7742 < 64;
        float t7747 = 1.0 * t7745;
        float t7748 = t7747 * t7746;
        int t7749 = t7742;
        int t7750 = t7744 >= 0;
        int t7751 = t7744 < 8;
        float t7752 = t7748 * t7750;
        float t7753 = t7752 * t7751;
        int t7754 = t7744;
        int t7755 = t7749 * 8;
        int t7756 = t7755 + t7754;
        float t7757 = 0.0;
        if (t7753) {
          int t7759 = i;
          int t7760 = t7759 * 512;
          int t7761 = t7760 + t7756;
          float t7762 = memory[122976255 + t7761];
          t7757 = t7762;
        }
        int t7764 = t7692 / 16;
        int t7765 = t7764 * 16;
        int t7766 = t7692 - t7765;
        int t7767 = t7764 >= 0;
        int t7768 = t7764 < 64;
        float t7769 = 1.0 * t7767;
        float t7770 = t7769 * t7768;
        int t7771 = t7764;
        int t7772 = t7766 >= 8;
        int t7773 = t7766 < 16;
        float t7774 = t7770 * t7772;
        float t7775 = t7774 * t7773;
        int t7776 = t7766 - 8;
        int t7777 = t7771 * 8;
        int t7778 = t7777 + t7776;
        float t7779 = 0.0;
        if (t7775) {
          int t7781 = i;
          int t7782 = t7781 * 512;
          int t7783 = t7782 + t7778;
          float t7784 = memory[159414271 + t7783];
          t7779 = t7784;
        }
        float t7786 = t7757 + t7779;
        int t7787 = i;
        int t7788 = t7787 * 1024;
        int t7789 = t7788 + t7692;
        memory[135821311 + t7789] = t7786;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 2, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([32, 16]), value: empty) */
      }
    }
  }
  for (int simd106 = 0; simd106 < 16; simd106+=4) {
  }
  for (int simd107 = 0; simd107 < 512; simd107+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
  }
  for (int simd108 = 0; simd108 < 16; simd108+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([32, 16]), value: empty) */
      for (int t7794 = 0; t7794 < 512; t7794++) {
        int t7795 = t7794 / 16;
        int t7796 = t7795 * 16;
        int t7797 = t7794 - t7796;
        int t7798 = t7795 * 16;
        int t7799 = t7798 + t7797;
        int t7800 = t7799 / 16;
        int t7801 = t7800 * 16;
        int t7802 = t7799 - t7801;
        int t7803 = t7802 / 16;
        int t7804 = t7803 * 16;
        int t7805 = t7802 - t7804;
        int t7806 = t7800 * 32;
        int t7807 = 16 + t7806;
        int t7808 = t7807 + t7805;
        int t7809 = i;
        int t7810 = t7809 * 1024;
        int t7811 = t7810 + t7808;
        float t7812 = memory[141588479 + t7811];
        int t7813 = t7794 / 16;
        int t7814 = t7813 * 16;
        int t7815 = t7794 - t7814;
        float t7816 = memory[6170 + t7815];
        float t7817 = t7812 * t7816;
        int t7818 = t7794 / 16;
        int t7819 = t7818 * 16;
        int t7820 = t7794 - t7819;
        int t7821 = t7818 * 16;
        int t7822 = t7821 + t7820;
        int t7823 = t7822 / 16;
        int t7824 = t7823 * 16;
        int t7825 = t7822 - t7824;
        int t7826 = t7825 / 16;
        int t7827 = t7826 * 16;
        int t7828 = t7825 - t7827;
        int t7829 = t7823 * 32;
        int t7830 = 16 + t7829;
        int t7831 = t7830 + t7828;
        int t7832 = i;
        int t7833 = t7832 * 1024;
        int t7834 = t7833 + t7831;
        float t7835 = memory[135821311 + t7834];
        int t7836 = t7794 / 16;
        int t7837 = t7836 * 16;
        int t7838 = t7794 - t7837;
        float t7839 = memory[6186 + t7838];
        float t7840 = t7835 * t7839;
        float t7841 = t7817 - t7840;
        int t7842 = t7794 / 16;
        int t7843 = t7842 * 16;
        int t7844 = t7794 - t7843;
        int t7845 = t7842 * 16;
        int t7846 = t7845 + t7844;
        int t7847 = t7846 / 16;
        int t7848 = t7847 * 16;
        int t7849 = t7846 - t7848;
        int t7850 = t7849 / 16;
        int t7851 = t7850 * 16;
        int t7852 = t7849 - t7851;
        int t7853 = t7847 * 32;
        int t7854 = 16 + t7853;
        int t7855 = t7854 + t7852;
        int t7856 = i;
        int t7857 = t7856 * 1024;
        int t7858 = t7857 + t7855;
        float t7859 = memory[141588479 + t7858];
        int t7860 = t7794 / 16;
        int t7861 = t7860 * 16;
        int t7862 = t7794 - t7861;
        float t7863 = memory[6186 + t7862];
        float t7864 = t7859 * t7863;
        int t7865 = t7794 / 16;
        int t7866 = t7865 * 16;
        int t7867 = t7794 - t7866;
        int t7868 = t7865 * 16;
        int t7869 = t7868 + t7867;
        int t7870 = t7869 / 16;
        int t7871 = t7870 * 16;
        int t7872 = t7869 - t7871;
        int t7873 = t7872 / 16;
        int t7874 = t7873 * 16;
        int t7875 = t7872 - t7874;
        int t7876 = t7870 * 32;
        int t7877 = 16 + t7876;
        int t7878 = t7877 + t7875;
        int t7879 = i;
        int t7880 = t7879 * 1024;
        int t7881 = t7880 + t7878;
        float t7882 = memory[135821311 + t7881];
        int t7883 = t7794 / 16;
        int t7884 = t7883 * 16;
        int t7885 = t7794 - t7884;
        float t7886 = memory[6170 + t7885];
        float t7887 = t7882 * t7886;
        float t7888 = t7864 + t7887;
        int t7889 = t7794 / 16;
        int t7890 = t7889 * 16;
        int t7891 = t7794 - t7890;
        int t7892 = t7889 * 16;
        int t7893 = t7892 + t7891;
        int t7894 = t7893 / 16;
        int t7895 = t7894 * 16;
        int t7896 = t7893 - t7895;
        int t7897 = t7896 / 16;
        int t7898 = t7897 * 16;
        int t7899 = t7896 - t7898;
        int t7900 = t7894 * 32;
        int t7901 = t7900 + t7899;
        int t7902 = i;
        int t7903 = t7902 * 1024;
        int t7904 = t7903 + t7901;
        float t7905 = memory[141588479 + t7904];
        float t7906 = t7905 + t7841;
        int t7907 = i;
        int t7908 = t7907 * 512;
        int t7909 = t7908 + t7794;
        memory[162297855 + t7909] = t7906;
        int t7911 = t7794 / 16;
        int t7912 = t7911 * 16;
        int t7913 = t7794 - t7912;
        int t7914 = t7911 * 16;
        int t7915 = t7914 + t7913;
        int t7916 = t7915 / 16;
        int t7917 = t7916 * 16;
        int t7918 = t7915 - t7917;
        int t7919 = t7918 / 16;
        int t7920 = t7919 * 16;
        int t7921 = t7918 - t7920;
        int t7922 = t7916 * 32;
        int t7923 = t7922 + t7921;
        int t7924 = i;
        int t7925 = t7924 * 1024;
        int t7926 = t7925 + t7923;
        float t7927 = memory[135821311 + t7926];
        float t7928 = t7927 + t7888;
        int t7929 = i;
        int t7930 = t7929 * 512;
        int t7931 = t7930 + t7794;
        memory[165443583 + t7931] = t7928;
        int t7933 = t7794 / 16;
        int t7934 = t7933 * 16;
        int t7935 = t7794 - t7934;
        int t7936 = t7933 * 16;
        int t7937 = t7936 + t7935;
        int t7938 = t7937 / 16;
        int t7939 = t7938 * 16;
        int t7940 = t7937 - t7939;
        int t7941 = t7940 / 16;
        int t7942 = t7941 * 16;
        int t7943 = t7940 - t7942;
        int t7944 = t7938 * 32;
        int t7945 = t7944 + t7943;
        int t7946 = i;
        int t7947 = t7946 * 1024;
        int t7948 = t7947 + t7945;
        float t7949 = memory[141588479 + t7948];
        float t7950 = t7949 - t7841;
        int t7951 = i;
        int t7952 = t7951 * 512;
        int t7953 = t7952 + t7794;
        memory[126384127 + t7953] = t7950;
        int t7955 = t7794 / 16;
        int t7956 = t7955 * 16;
        int t7957 = t7794 - t7956;
        int t7958 = t7955 * 16;
        int t7959 = t7958 + t7957;
        int t7960 = t7959 / 16;
        int t7961 = t7960 * 16;
        int t7962 = t7959 - t7961;
        int t7963 = t7962 / 16;
        int t7964 = t7963 * 16;
        int t7965 = t7962 - t7964;
        int t7966 = t7960 * 32;
        int t7967 = t7966 + t7965;
        int t7968 = i;
        int t7969 = t7968 * 1024;
        int t7970 = t7969 + t7967;
        float t7971 = memory[135821311 + t7970];
        float t7972 = t7971 - t7888;
        int t7973 = i;
        int t7974 = t7973 * 512;
        int t7975 = t7974 + t7794;
        memory[133462015 + t7975] = t7972;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
      }
      for (int t7977 = 0; t7977 < 1024; t7977++) {
        int t7978 = t7977 / 32;
        int t7979 = t7978 * 32;
        int t7980 = t7977 - t7979;
        int t7981 = t7978 >= 0;
        int t7982 = t7978 < 32;
        float t7983 = 1.0 * t7981;
        float t7984 = t7983 * t7982;
        int t7985 = t7978;
        int t7986 = t7980 >= 0;
        int t7987 = t7980 < 16;
        float t7988 = t7984 * t7986;
        float t7989 = t7988 * t7987;
        int t7990 = t7980;
        int t7991 = t7985 * 16;
        int t7992 = t7991 + t7990;
        float t7993 = 0.0;
        if (t7989) {
          int t7995 = i;
          int t7996 = t7995 * 512;
          int t7997 = t7996 + t7992;
          float t7998 = memory[162297855 + t7997];
          t7993 = t7998;
        }
        int t8000 = t7977 / 32;
        int t8001 = t8000 * 32;
        int t8002 = t7977 - t8001;
        int t8003 = t8000 >= 0;
        int t8004 = t8000 < 32;
        float t8005 = 1.0 * t8003;
        float t8006 = t8005 * t8004;
        int t8007 = t8000;
        int t8008 = t8002 >= 16;
        int t8009 = t8002 < 32;
        float t8010 = t8006 * t8008;
        float t8011 = t8010 * t8009;
        int t8012 = t8002 - 16;
        int t8013 = t8007 * 16;
        int t8014 = t8013 + t8012;
        float t8015 = 0.0;
        if (t8011) {
          int t8017 = i;
          int t8018 = t8017 * 512;
          int t8019 = t8018 + t8014;
          float t8020 = memory[126384127 + t8019];
          t8015 = t8020;
        }
        float t8022 = t7993 + t8015;
        int t8023 = i;
        int t8024 = t8023 * 1024;
        int t8025 = t8024 + t7977;
        memory[125597695 + t8025] = t8022;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 16)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (16, 0)]), value: empty) */
        int t8027 = t7977 / 32;
        int t8028 = t8027 * 32;
        int t8029 = t7977 - t8028;
        int t8030 = t8027 >= 0;
        int t8031 = t8027 < 32;
        float t8032 = 1.0 * t8030;
        float t8033 = t8032 * t8031;
        int t8034 = t8027;
        int t8035 = t8029 >= 0;
        int t8036 = t8029 < 16;
        float t8037 = t8033 * t8035;
        float t8038 = t8037 * t8036;
        int t8039 = t8029;
        int t8040 = t8034 * 16;
        int t8041 = t8040 + t8039;
        float t8042 = 0.0;
        if (t8038) {
          int t8044 = i;
          int t8045 = t8044 * 512;
          int t8046 = t8045 + t8041;
          float t8047 = memory[165443583 + t8046];
          t8042 = t8047;
        }
        int t8049 = t7977 / 32;
        int t8050 = t8049 * 32;
        int t8051 = t7977 - t8050;
        int t8052 = t8049 >= 0;
        int t8053 = t8049 < 32;
        float t8054 = 1.0 * t8052;
        float t8055 = t8054 * t8053;
        int t8056 = t8049;
        int t8057 = t8051 >= 16;
        int t8058 = t8051 < 32;
        float t8059 = t8055 * t8057;
        float t8060 = t8059 * t8058;
        int t8061 = t8051 - 16;
        int t8062 = t8056 * 16;
        int t8063 = t8062 + t8061;
        float t8064 = 0.0;
        if (t8060) {
          int t8066 = i;
          int t8067 = t8066 * 512;
          int t8068 = t8067 + t8063;
          float t8069 = memory[133462015 + t8068];
          t8064 = t8069;
        }
        float t8071 = t8042 + t8064;
        int t8072 = i;
        int t8073 = t8072 * 1024;
        int t8074 = t8073 + t7977;
        memory[126646271 + t8074] = t8071;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 2, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((0, 1)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mshrink [0m([nil, Optional((1, 2)), nil]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([16, 32]), value: empty) */
      }
    }
  }
  for (int simd110 = 0; simd110 < 32; simd110+=4) {
  }
  for (int simd111 = 0; simd111 < 512; simd111+=4) {
    /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
  }
  for (int simd112 = 0; simd112 < 32; simd112+=4) {
  }
  for (int i = 0; i < frameCount; i += 1) {
    float32x4_t simd153 = vld1q_f32(t153 + i); /* extra */
    t153[i] = t153[i];
    if (t153[i] == 0.0f) {
      float32x4_t simd188 = vld1q_f32(t188 + i); /* extra */
    t188[i] = t188[i];
      /* skip scalar load */
      /*  [1mUOp [0m(op:  [38;5;51mexpandView [0m([16, 32]), value: empty) */
      for (int t8079 = 0; t8079 < 512; t8079++) {
        int t8080 = t8079 / 32;
        int t8081 = t8080 * 32;
        int t8082 = t8079 - t8081;
        int t8083 = t8080 * 32;
        int t8084 = t8083 + t8082;
        int t8085 = t8084 / 32;
        int t8086 = t8085 * 32;
        int t8087 = t8084 - t8086;
        int t8088 = t8087 / 32;
        int t8089 = t8088 * 32;
        int t8090 = t8087 - t8089;
        int t8091 = t8085 * 64;
        int t8092 = 32 + t8091;
        int t8093 = t8092 + t8090;
        int t8094 = i;
        int t8095 = t8094 * 1024;
        int t8096 = t8095 + t8093;
        float t8097 = memory[125597695 + t8096];
        int t8098 = t8079 / 32;
        int t8099 = t8098 * 32;
        int t8100 = t8079 - t8099;
        float t8101 = memory[6202 + t8100];
        float t8102 = t8097 * t8101;
        int t8103 = t8079 / 32;
        int t8104 = t8103 * 32;
        int t8105 = t8079 - t8104;
        int t8106 = t8103 * 32;
        int t8107 = t8106 + t8105;
        int t8108 = t8107 / 32;
        int t8109 = t8108 * 32;
        int t8110 = t8107 - t8109;
        int t8111 = t8110 / 32;
        int t8112 = t8111 * 32;
        int t8113 = t8110 - t8112;
        int t8114 = t8108 * 64;
        int t8115 = 32 + t8114;
        int t8116 = t8115 + t8113;
        int t8117 = i;
        int t8118 = t8117 * 1024;
        int t8119 = t8118 + t8116;
        float t8120 = memory[126646271 + t8119];
        int t8121 = t8079 / 32;
        int t8122 = t8121 * 32;
        int t8123 = t8079 - t8122;
        float t8124 = memory[6234 + t8123];
        float t8125 = t8120 * t8124;
        float t8126 = t8102 - t8125;
        int t8127 = t8079 / 32;
        int t8128 = t8127 * 32;
        int t8129 = t8079 - t8128;
        int t8130 = t8127 * 32;
        int t8131 = t8130 + t8129;
        int t8132 = t8131 / 32;
        int t8133 = t8132 * 32;
        int t8134 = t8131 - t8133;
        int t8135 = t8134 / 32;
        int t8136 = t8135 * 32;
        int t8137 = t8134 - t8136;
        int t8138 = t8132 * 64;
        int t8139 = 32 + t8138;
        int t8140 = t8139 + t8137;
        int t8141 = i;
        int t8142 = t8141 * 1024;
        int t8143 = t8142 + t8140;
        float t8144 = memory[125597695 + t8143];
        int t8145 = t8079 / 32;
        int t8146 = t8145 * 32;
        int t8147 = t8079 - t8146;
        float t8148 = memory[6234 + t8147];
        float t8149 = t8144 * t8148;
        int t8150 = t8079 / 32;
        int t8151 = t8150 * 32;
        int t8152 = t8079 - t8151;
        int t8153 = t8150 * 32;
        int t8154 = t8153 + t8152;
        int t8155 = t8154 / 32;
        int t8156 = t8155 * 32;
        int t8157 = t8154 - t8156;
        int t8158 = t8157 / 32;
        int t8159 = t8158 * 32;
        int t8160 = t8157 - t8159;
        int t8161 = t8155 * 64;
        int t8162 = 32 + t8161;
        int t8163 = t8162 + t8160;
        int t8164 = i;
        int t8165 = t8164 * 1024;
        int t8166 = t8165 + t8163;
        float t8167 = memory[126646271 + t8166];
        int t8168 = t8079 / 32;
        int t8169 = t8168 * 32;
        int t8170 = t8079 - t8169;
        float t8171 = memory[6202 + t8170];
        float t8172 = t8167 * t8171;
        float t8173 = t8149 + t8172;
        int t8174 = t8079 / 32;
        int t8175 = t8174 * 32;
        int t8176 = t8079 - t8175;
        int t8177 = t8174 * 32;
        int t8178 = t8177 + t8176;
        int t8179 = t8178 / 32;
        int t8180 = t8179 * 32;
        int t8181 = t8178 - t8180;
        int t8182 = t8181 / 32;
        int t8183 = t8182 * 32;
        int t8184 = t8181 - t8183;
        int t8185 = t8179 * 64;
        int t8186 = t8185 + t8184;
        int t8187 = i;
        int t8188 = t8187 * 1024;
        int t8189 = t8188 + t8186;
        float t8190 = memory[125597695 + t8189];
        float t8191 = t8190 + t8126;
        int t8192 = i;
        int t8193 = t8192 * 512;
        int t8194 = t8193 + t8079;
        memory[139491327 + t8194] = t8191;
        int t8196 = t8079 / 32;
        int t8197 = t8196 * 32;
        int t8198 = t8079 - t8197;
        int t8199 = t8196 * 32;
        int t8200 = t8199 + t8198;
        int t8201 = t8200 / 32;
        int t8202 = t8201 * 32;
        int t8203 = t8200 - t8202;
        int t8204 = t8203 / 32;
        int t8205 = t8204 * 32;
        int t8206 = t8203 - t8205;
        int t8207 = t8201 * 64;
        int t8208 = t8207 + t8206;
        int t8209 = i;
        int t8210 = t8209 * 1024;
        int t8211 = t8210 + t8208;
        float t8212 = memory[126646271 + t8211];
        float t8213 = t8212 + t8173;
        int t8214 = i;
        int t8215 = t8214 * 512;
        int t8216 = t8215 + t8079;
        memory[150763519 + t8216] = t8213;
        int t8218 = t8079 / 32;
        int t8219 = t8218 * 32;
        int t8220 = t8079 - t8219;
        int t8221 = t8218 * 32;
        int t8222 = t8221 + t8220;
        int t8223 = t8222 / 32;
        int t8224 = t8223 * 32;
        int t8225 = t8222 - t8224;
        int t8226 = t8225 / 32;
        int t8227 = t8226 * 32;
        int t8228 = t8225 - t8227;
        int t8229 = t8223 * 64;
        int t8230 = t8229 + t8228;
        int t8231 = i;
        int t8232 = t8231 * 1024;
        int t8233 = t8232 + t8230;
        float t8234 = memory[125597695 + t8233];
        float t8235 = t8234 - t8126;
        int t8236 = i;
        int t8237 = t8236 * 512;
        int t8238 = t8237 + t8079;
        memory[137132031 + t8238] = t8235;
        int t8240 = t8079 / 32;
        int t8241 = t8240 * 32;
        int t8242 = t8079 - t8241;
        int t8243 = t8240 * 32;
        int t8244 = t8243 + t8242;
        int t8245 = t8244 / 32;
        int t8246 = t8245 * 32;
        int t8247 = t8244 - t8246;
        int t8248 = t8247 / 32;
        int t8249 = t8248 * 32;
        int t8250 = t8247 - t8249;
        int t8251 = t8245 * 64;
        int t8252 = t8251 + t8250;
        int t8253 = i;
        int t8254 = t8253 * 1024;
        int t8255 = t8254 + t8252;
        float t8256 = memory[126646271 + t8255];
        float t8257 = t8256 - t8173;
        int t8258 = i;
        int t8259 = t8258 * 512;
        int t8260 = t8259 + t8079;
        memory[121141247 + t8260] = t8257;
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
      }
      for (int t8262 = 0; t8262 < 1024; t8262++) {
        int t8263 = t8262 / 64;
        int t8264 = t8263 * 64;
        int t8265 = t8262 - t8264;
        int t8266 = t8263 >= 0;
        int t8267 = t8263 < 16;
        float t8268 = 1.0 * t8266;
        float t8269 = t8268 * t8267;
        int t8270 = t8263;
        int t8271 = t8265 >= 0;
        int t8272 = t8265 < 32;
        float t8273 = t8269 * t8271;
        float t8274 = t8273 * t8272;
        int t8275 = t8265;
        int t8276 = t8270 * 32;
        int t8277 = t8276 + t8275;
        float t8278 = 0.0;
        if (t8274) {
          int t8280 = i;
          int t8281 = t8280 * 512;
          int t8282 = t8281 + t8277;
          float t8283 = memory[139491327 + t8282];
          t8278 = t8283;
        }
        int t8285 = t8262 / 64;
        int t8286 = t8285 * 64;
        int t8287 = t8262 - t8286;
        int t8288 = t8285 >= 0;
        int t8289 = t8285 < 16;
        float t8290 = 1.0 * t8288;
        float t8291 = t8290 * t8289;
        int t8292 = t8285;
        int t8293 = t8287 >= 32;
        int t8294 = t8287 < 64;
        float t8295 = t8291 * t8293;
        float t8296 = t8295 * t8294;
        int t8297 = t8287 - 32;
        int t8298 = t8292 * 32;
        int t8299 = t8298 + t8297;
        float t8300 = 0.0;
        if (t8296) {
          int t8302 = i;
          int t8303 = t8302 * 512;
          int t8304 = t8303 + t8299;
          float t8305 = memory[137132031 + t8304];
          t8300 = t8305;
        }
        float t8307 = t8278 + t8300;
        int t8308 = i;
        int t8309 = t8308 * 1024;
        int t8310 = t8309 + t8262;
        memory[141064191 + t8310] = t8307;
        /*  [1mUOp [0m(op:  [38;5;51mreshape [0m([1024]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (0, 32)]), value: empty) */
        /*  [1mUOp [0m(op:  [38;5;51mpad [0m([(0, 0), (32, 0)]), value: empty) */
        int t8312 = t8262 / 64;
        int t8313 = t8312 * 64;
        int t8314 = t8262 - t8313;
        int t8315 = t8312 >= 0;
        int t8316 = t8312 < 16;
        float t8317 = 1.0 * t8315;
        float t8318 = t8317 * t8316;
        int t8319 = t8312;
        int t8320 = t8314 >= 0;
        int t8321 = t8314 < 32;
        float t8322 = t8318 * t8320;
        float t8323 = t8322 * t8321;
        int t8324 = t8314;
        int t8325 = t8319 * 32;
        int t8326 = t8325 + t8324;
        float t8327 = 0.0;
        if (t8323) {
          int t8329 = i;
          int t8330 = t8329 * 512;
          int t8331 = t8330 + t8326;
          float t8332 = memory[150763519 + t8331];
          t8327 = t8332;
        }
        int t8334 = t8262 / 64;
        int t8335 = t8334 * 64;
        int t8336 = t8262 - t8335;
        int t8337 = t8334 >= 0;
        int t8338 = t8334 < 16;
        float t8339 = 1.0 * t8337;
        float t8340 = t8339 * t8338;
        int t8341 = t8334;
        int t8342 = t8336 >= 32;
        int t8343 = t8336 < 64;
        float t8344 = t8340 * t8342;
        float t8345 = t8344 * t8343;
        int t8346 = t8336 - 32;
        int t8347 = t8341 * 32;
        int t8348 = t8347 + t8346;
        float t8349 = 0.0;
        if (t8345) {
          int t8351 = i;
          int t8352 = t8351 * 512;
          int t8353 = t8352 + t8348;
          float t8354 = memory[1211
