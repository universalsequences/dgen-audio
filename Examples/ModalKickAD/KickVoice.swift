import DGenLazy
import Foundation

// The Latent Kick voice in DGenLazy: VILLAIN Kick's kick-hit math with decoded per-kick
// parameters, frame-parallel (no state cells). Mirrors kick/scripts/kick_np.py render(...,
// mode="trainer"): modal bank (24 slots, VILLAIN drop law with a second drop on slots 0-3,
// time warp, gate masks), 5 transient sines, growl, burst, body shaper/saturator, 18-band
// noise + hum from fixed tables, recording clipper, fade, out gate. The instrument's IIR
// stages (bark HP, 18 Hz body HP, DC blocker, record LP, drive split) are omitted here.

func kClip01(_ x: SignalTensor) -> SignalTensor { max(min(x, 1.0), 0.0) }
func kSmooth01(_ x: SignalTensor) -> SignalTensor {
  let c = kClip01(x)
  return c * c * (c * -2.0 + 3.0)
}
/// x^e for x >= 0 (exp(e log max(x, 1e-30)))
func kPow(_ x: SignalTensor, _ e: Tensor) -> SignalTensor { exp(log(max(x, 1e-30)) * e) }
func kPow(_ x: Tensor, _ e: Tensor) -> Tensor { (log(max(x, 1e-30)) * e).exp() }

enum KickClock {
  /// seconds since the trigger, frame-parallel (non-wrapping phasor at sr/2N)
  static func seconds(frames: Int, sampleRate: Float) -> Signal {
    let rampHz = sampleRate / (2 * Float(frames))
    return Signal.phasor(Tensor([rampHz])).sum() * (2 * Float(frames) / sampleRate)
  }
}

struct KickRender {
  var out: SignalTensor  // [B]
  var modal: SignalTensor  // [B] bank + transients
  var noise: SignalTensor  // [B] texture (vpre vtex (bands + hum)), pre-chain
}

enum KickVoice {
  /// nat: KickLayout.squash(u) ([B, w] each); hits: one per lane.
  static func render(
    nat p: [String: Tensor], layout L: KickLayout, hits: [KickHit], tables: Tensor, hum: Tensor,
    frames: Int, sampleRate sr: Float
  ) -> KickRender {
    let B = hits.count, K = L.K, K2 = L.K2, J = L.J, NB = L.NB
    func lane(_ f: (KickHit) -> Double) -> Tensor { Tensor(hits.map { Float(f($0)) }).reshape([B, 1]) }
    func s(_ name: String) -> Tensor { p[name]! }  // [B,1] or [B,w]
    func bk(_ t: Tensor, _ w: Int) -> Tensor { t.expand([B, w]) }  // [B,1] -> [B,w]
    let time = KickClock.seconds(frames: frames, sampleRate: sr)

    // ---- per hit, post-decoder (static) ----
    let lnv = lane { Foundation.log($0.velClamped) }
    func velp(_ e: Tensor) -> Tensor { (e * lnv).exp() }
    func vcurve(_ k: Tensor, _ e: Tensor) -> Tensor { k * -1.0 + 1.0 + k * velp(e) }
    let one = lane { _ in 1 }
    let tuning = lane { $0.tuning }
    let bendE = lane { $0.k("bend") } * vcurve(s("vb_k"), s("vb_e"))
    let decE = lane { $0.k("decay") } * vcurve(s("vd_k"), one)
    let knockE = lane { $0.k("knock") } * vcurve(s("vk_k"), s("vk_e"))
    let tickE = lane { $0.k("tick") } * vcurve(s("vt_k"), s("vt_e"))
    let bodyE = lane { $0.k("body") }
    let attE = lane { $0.k("attack") } / vcurve(s("va_k"), one)
    let vpreE = velp(s("vpre_e")), vtexE = velp(s("vtex_e")), vpostE = velp(s("vpost_e"))
    // slot transform knobs
    let lf0 = log(s("mode_f"))  // [B,K]
    let lfsub = bk(lf0.shrink([(0, B), (0, 1)]), K)
    var kmulData = [Float](repeating: 0, count: B * K)
    for (b, h) in hits.enumerated() {
      kmulData[b * K] = 1
      kmulData[b * K + 1] = Float(h.k("beat"))
      for i in 2..<K { kmulData[b * K + i] = Float(h.k("spread")) }
    }
    let lf = lfsub + Tensor(kmulData).reshape([B, K]) * (lf0 - lfsub)
    let f = lf.exp()
    let tilt = bk(lane { $0.k("tilt") }, K)
    let idec = ((log(s("mode_dec")) + tilt * (lf - lfsub)) * -1.0).exp() / bk(decE, K)
    let drop = lane { $0.k("drop_time") }
    let tau = s("mode_tau") * bk(drop, K)
    let sh = max(f * -0.9, bk(bendE, K) * s("mode_glide") * f)
    var km = [Float](repeating: 0, count: B * K)
    for b in 0..<B { for i in 14..<K { km[b * K + i] = 1 } }
    let kmask = Tensor(km).reshape([B, K])
    let grp = bk(bodyE, K) * (kmask * -1.0 + 1.0) + bk(knockE, K) * kmask
    let ampE = s("mode_amp") * grp
    let irise = 1.0 / (s("mode_rise") * bk(attE, K))
    let f4 = f.shrink([(0, B), (0, K2)])
    let sh2 = max(f4 * -0.9, bk(bendE, K2) * s("mode_glide2") * f4)
    let tau2 = s("mode_tau2") * bk(drop, K2)
    // gates
    let decay = lane { $0.k("decay") }
    let lenS = s("len") * lane { $0.k("length") } * (s("len_dsc") * (lane { max(1, $0.k("decay")) } - 1.0) + 1.0)
    let holdS = s("hold") * lane { $0.k("hold") }
    let relS = s("rel") * lane { $0.k("release") }
    let dscA = s("gA_dsc") * (decay - 1.0) + 1.0
    let holdA = dscA * (s("gA_hvk") * (lane { $0.velClamped } - 1.0) + 1.0)
      * (s("gA_hL") * lenS + s("gA_hH") * holdS + s("gA_hR") * relS + s("gA_hC"))
    let relA = max(dscA * (s("gA_rR") * relS + s("gA_rC")), 1e-4)
    let dscB = s("gB_dsc") * (decay - 1.0) + 1.0
    let holdB = dscB * (s("gB_hL") * lenS + s("gB_hH") * holdS + s("gB_hC"))
    let relB = max(dscB * (s("gB_rL") * lenS + s("gB_rR") * relS + s("gB_rC")), 1e-4)
    let fadeS = min(lenS * 0.9, max(s("fd_s"), s("fd_frac") * lenS))

    // ---- time bases, gates [B,1] ----
    let Tscale = s("samp") * (tuning - 1.0) + 1.0
    let fmul = s("samp") * (tuning * -1.0 + 1.0) + tuning
    let T = time * Tscale  // [B,1]
    let Tw = T + (exp(T * (-1.0 / s("wtau"))) * -1.0 + 1.0) * (s("wk") * bendE * s("wtau"))
    func kickGate(hold: Tensor, rel: Tensor, wexp: Tensor, wrat: Tensor?, wss: Tensor, pe: Tensor,
                  tmix: Tensor?, tail: Tensor, floorG: Tensor?) -> SignalTensor {
      let gx = relu(time - hold)
      let x = gx * (1.0 / rel)
      let xp = kPow(x, pe)
      var shaped = exp(xp * -1.0) * wexp + (1.0 - kSmooth01(x)) * wss
      if let wrat { shaped = shaped + (1.0 / (xp + 1.0)) * wrat }
      var g = shaped
      if let tmix { g = shaped + (exp(gx * (-1.0 / tail)) - shaped) * tmix }
      if let floorG { g = g * (floorG * -1.0 + 1.0) + floorG }
      return g
    }
    let GA = kickGate(hold: holdA, rel: relA, wexp: s("gA_wexp"), wrat: s("gA_wrat"), wss: s("gA_wss"),
                      pe: s("gA_pe"), tmix: s("gA_tmix"), tail: dscA * s("gA_tail"), floorG: s("gA_floor"))
    let GB = kickGate(hold: holdB, rel: relB, wexp: s("gB_wexp"), wrat: nil, wss: s("gB_wss"),
                      pe: s("gB_pe"), tmix: nil, tail: dscB, floorG: nil)

    // ---- modal bank [B,K] ----
    let TK = T.expand([B, K]), TwK = Tw.expand([B, K])
    let t = relu(TK - s("mode_t0"))
    let tw = relu(TwK - s("mode_t0"))
    let se = exp(tw * (-1.0 / tau))
    let tw4 = tw.shrink([(0, B), (0, K2)])
    let se2 = exp(tw4 * (-1.0 / tau2))
    let hz2 = (se2 * sh2).pad([(0, 0), (0, K - K2)])
    let ph2 = ((se2 * -1.0 + 1.0) * (sh2 * tau2)).pad([(0, 0), (0, K - K2)])
    let fmulK = bk(fmul, K)
    let hz = (se * sh + f + hz2) * fmulK
    let nyq = kClip01((hz * -1.0 + 0.48 * sr) * (1 / (0.08 * sr)))
    let GAK = GA.expand([B, K]), GBK = GB.expand([B, K])
    let g = exp(t * (idec * -1.0)) * (exp(t * (irise * -1.0)) * -1.0 + 1.0) * ampE * nyq
      * ((GAK * -1.0 + 1.0) * s("mode_mga") * -1.0 + 1.0) * ((GBK * -1.0 + 1.0) * s("mode_mgb") * -1.0 + 1.0)
    let phase = (tw * f + (se * -1.0 + 1.0) * (sh * tau) + ph2) * fmulK * (2 * Float.pi) + s("mode_phase")
    let m = g * sin(phase)
    let bank = m.sum(axis: 1)
    let bankMsm = (m * s("mode_msm")).sum(axis: 1)
    let g2: SignalTensor = g * g
    let bankDc = (g2 * s("mode_msm")).sum(axis: 1)
    let esq = g2.sum(axis: 1)

    // ---- transients [B,J] ----
    let TJ = T.expand([B, J])
    let xt = relu(TJ - s("tr_t0"))
    let xsh = max(s("tr_f") * -0.9, bk(bendE, J) * s("tr_dl"))
    let xse = exp(xt * (-1.0 / s("tr_tau")))
    let xidec = 1.0 / (s("tr_dec") * bk(decE, J))
    let xirise = 1.0 / (s("tr_rise") * bk(attE, J))
    let xamp = s("tr_amp") * (s("tr_wb") * bk(bodyE, J) + s("tr_wt") * bk(tickE, J))
    let GBJ = GB.expand([B, J])
    let xm = exp(xt * (xidec * -1.0)) * (exp(xt * (xirise * -1.0)) * -1.0 + 1.0) * xamp
      * ((GBJ * -1.0 + 1.0) * s("tr_mgb") * -1.0 + 1.0)
      * sin((xt * s("tr_f") + (xse * -1.0 + 1.0) * (xsh * s("tr_tau"))) * bk(fmul, J) * (2 * Float.pi) + s("tr_phase"))
    let trans = xm.sum(axis: 1)
    let transMsm = (xm * s("tr_msm")).sum(axis: 1)
    let modal = bank + trans

    // ---- growl, burst [B] (via [B,1]) ----
    func r1(_ x: SignalTensor) -> SignalTensor { x.reshape([B]) }
    let grShift = max(bendE * s("gr_dl"), -31.5)
    let grSettle = exp(T * (-1.0 / s("gr_tau")))
    let grPh = (T * 35.0 + (grSettle * -1.0 + 1.0) * (grShift * s("gr_tau"))) * fmul * (2 * Float.pi) + s("gr_off")
    let grHz = (grSettle * grShift + 35.0) * fmul
    let grAud = kClip01((grHz * -1.0 + 0.02 * sr) * (1 / (0.008 * sr)))
    let grSin = sin(grPh)
    let grWave = min(grSin, lift(s("gr_clip"))) - (grSin * s("gr_b") + s("gr_a"))
    let grEnv = (exp(T * (-1.0 / (decay * s("gr_d1")))) * 0.5 + exp(T * (-1.0 / (decay * s("gr_d2")))) * 0.5)
      * (exp(T * (-1.0 / (lane { $0.k("attack") } * 0.004))) * -1.0 + 1.0)
    let growl = r1(grAud * grEnv * grWave * (lane { $0.k("body") * $0.k("growl") } * s("gr_amp")))
    let buDt = time * (s("bu_samp") * (tuning - 1.0) + 1.0) - s("bu_tc")
    let buZ = buDt * (1.0 / s("bu_w"))
    let buG = exp(min(buZ * buZ, 60.0) * -0.5)
    let buH = cos(kClip01(buZ) * (2 * Float.pi)) * -0.5 + 0.5
    let burst = r1((buG + (buH - buG) * s("bu_hann")) * sin(buDt * (s("bu_f") * (2 * Float.pi)) + s("bu_off"))
      * (tickE * s("bu_amp")))
    let bankAll = modal + growl + burst
    func v(_ t: Tensor) -> Tensor { t.reshape([B]) }  // [B,1] -> [B]
    let bankSh = bankMsm + transMsm + burst * v(s("bu_sm"))

    // ---- body shaper [B] ----
    let shape = Tensor(hits.map { Float($0.k("shape")) })
    let xin = bankSh * v(vpreE)
    let ecl = v(s("sh_ecl"))
    let xcl = min(max(xin, lift(ecl * -1.0)), lift(ecl))
    let shEven = (xcl * xcl * (xcl * (xcl * (xcl * v(s("sh_p5")) + v(s("sh_p4"))) + v(s("sh_p3"))) + v(s("sh_ec")))
      - bankDc * v(s("sh_ec") * s("sh_ed") * vpreE * vpreE * 0.5)) * shape
    let ow = v(s("sh_ow"))
    let shOdd = (tanh(xin * (1.0 / ow)) * ow - xin) * (shape * v(s("sh_oc")))
    let y0 = xin + shEven + shOdd
    let ratC = 1.0 / max(v(s("sh_rg")) * shape, 1e-6)
    let a0 = abs(y0)
    let rat = a0 * exp(log(exp(log(max(a0 * (1.0 / ratC), 1e-9)) * 4.0) + 1.0) * -0.25)
    let y1 = y0 + (sign(y0) * rat - y0) * v(s("sh_rk"))
    let satOn = v(s("sat_on"))
    let satGain = max(v(s("sat_g")) * (satOn * (shape - 1.0) + 1.0), 0.01)
    let satB = v(s("sat_b"))
    let satRef = tanh(satB)
    let satNorm = 0.8 / (tanh(satGain * 0.8 + satB) - satRef)
    let shaped = y1 + ((tanh(y1 * satGain + satB) - satRef) * satNorm - y1) * satOn
    let body = (bankAll - bankSh) * v(vpreE) + shaped

    // ---- noise: 18 bands (fixed table) + hum ----
    let tn = relu(time * (s("nsamp") * (tuning - 1.0) + 1.0) - s("noise_onset"))  // [B,1]
    let att = exp(tn * (-1.0 / s("noise_rise"))) * -1.0 + 1.0
    let ngate = kPow(GA, s("noise_ga")) * kPow(GB, s("noise_gb"))  // [B,1]
    let cgate = kPow(GA, s("couple_ga"))
    let tnB = tn.expand([B, NB])
    let lt = s("band_tail")
    let fastX: SignalTensor = tnB * (1.0 / s("band_d1"))
    let fastE: SignalTensor = exp(kPow(fastX, s("noise_q").expand([B, NB])) * -1.0)
    let envA = (fastE * (lt * -1.0 + 1.0) + exp(tnB * (-1.0 / s("band_d2"))) * lt)
      * s("band_amp") * (att * ngate).expand([B, NB])
    let E = sqrt(esq + 1e-10).reshape([B, 1])
    let envC = (E * cgate).expand([B, NB]) * s("band_couple")
    let playhead = floor(time * sr + 0.5)
    let bandsNow = tables.sampleRow(playhead).reshape([1, NB]).expand([B, NB])
    let nvel = lane { $0.k("noise") } * velp(s("noise_ve"))
    let bandNoise = ((envA + envC) * bandsNow).sum(axis: 1) * v(nvel)
    let humNow: SignalTensor = hum.sampleRow(playhead).reshape([1, 1]).expand([B, 1])
    let humS = sin(time * tuning * (92 * 2 * Float.pi))  // [B,1]
    let humEnv: Signal = (exp(time * -10.0) * -1.0 + 1.0) * exp(time * (-1.0 / 0.22))
    let humAmp = lane { $0.k("noise") } * velp(s("hum_ve")) * s("hum_amt")
    let humWave: SignalTensor = humS + (humS * humS * -2.0 + 1.0) * s("hum_h2")
    let humSig = r1(humWave * ngate * humAmp * humNow * humEnv)
    let tex = (bandNoise + humSig) * v(vpreE * vtexE)
    let npv = v(s("np"))
    let source = body + tex * (npv * -1.0 + 1.0)

    // ---- recording clipper (static nonlinearity) ----
    let ceilP = v(s("ceil") * lane { $0.k("ceiling") })
    let ceilN = ceilP * v(s("cneg"))
    let cx = source + v(s("coff"))
    let up = cx > 0.0
    let mag = abs(cx)
    let cs = (ceilP - ceilN) * up + ceilN
    let ka = (v(s("kap")) - v(s("kan"))) * up + v(s("kan"))
    let kb = (v(s("kbp")) - v(s("kbn"))) * up + v(s("kbn"))
    let kk: SignalTensor = ka * cs
    let kbcs: SignalTensor = kb * cs
    let ww: SignalTensor = max(kbcs, 1e-6)
    let over: SignalTensor = relu(mag - kk) / ww
    let knee: SignalTensor = min(mag, kk) + tanh(over) * ww
    let kn = v(s("kn"))
    let rq: SignalTensor = max(mag / cs, 1e-9)
    let rp: SignalTensor = exp(log(rq) * kn) + 1.0
    let ratc: SignalTensor = mag * exp(log(rp) * (-1.0 / kn))
    let side: SignalTensor = knee + (ratc - knee) * v(s("kr"))
    let clipped: SignalTensor = (up * 2.0 - 1.0) * side - v(s("coff"))
    let dry: SignalTensor = source * v(s("dry"))
    let cmixK = v(s("cmix") * lane { $0.k("clip_amt") })
    let recorded: SignalTensor = dry + (clipped - dry) * cmixK
    let withPost = recorded + tex * npv

    // ---- fade, out gate, level, output clip ----
    let Tf = time * (s("fd_samp") * (tuning - 1.0) + 1.0)
    let fu = r1(kSmooth01((Tf - (lenS - fadeS)) * (1.0 / fadeS)))
    let outGate = r1(kPow(GA, s("oA_e")))
    let gain = v(vpostE * s("lvl") * lane { $0.k("level") })
    let oc = v(s("out_clip"))
    let y = withPost * (fu * -1.0 + 1.0) * outGate * gain
    let out = min(max(y, lift(oc * -1.0)), lift(oc))
    return KickRender(out: out, modal: modal, noise: tex)
  }
}

func lift(_ t: Tensor) -> SignalTensor { SignalTensor.lift(t) }
