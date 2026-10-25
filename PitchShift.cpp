#include "PitchShift.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kFadeMs = 30.0;
constexpr double kFadeMaxMs = 120.0;
constexpr double kFadeMinMs = 6.0;
constexpr double kFadeNccHi = 0.95;
constexpr double kFadeNccLo = 0.6;
constexpr double kOnsetFadeMs = 2.0;
constexpr double kMaxCorrMs = 25.0;
constexpr double kOnsetSpanMs = 4.0;
constexpr double kRefractoryMs = 40.0;
constexpr double kDetectorHpfHz = 600.0;
constexpr double kDetectorSmoothMs = 2.0;
constexpr int kHistoryCells = 50;
constexpr int kHistorySkipCells = 5;
constexpr double kOnsetOverMin = 7.94;
constexpr double kOnsetOverMax = 3.98;
constexpr double kCoarseStepRate = 12000.0;
constexpr double kSearchLeadMs = 4.0;
constexpr double kUpshiftLandShare = 0.5;
constexpr double kSearchKeepRatio = 0.1;
constexpr double kBlendSeconds = 0.025;

// Replacement for juce::LinearSmoothedValue<float>
struct SmoothedValue {
  float current = 0.0f, target = 0.0f, step = 0.0f;
  int countdown = 0, stepsToTarget = 0;

  void reset(double sampleRate, double rampSeconds) {
    stepsToTarget = std::max(1, static_cast<int>(std::floor(sampleRate * rampSeconds)));
    current = target;
    countdown = 0;
  }
  void setCurrentAndTargetValue(float v) {
    current = target = v;
    countdown = 0;
  }
  void setTargetValue(float v) {
    if (v == target) return;
    target = v;
    if (stepsToTarget <= 0) {
      current = target;
      countdown = 0;
    } else {
      countdown = stepsToTarget;
      step = (target - current) / static_cast<float>(countdown);
    }
  }
  float getNextValue() {
    if (countdown <= 0) return current;
    --countdown;
    current = (countdown == 0) ? target : (current + step);
    return current;
  }
  bool isSmoothing() const { return countdown > 0; }
};

// Replacement for juce::dsp::LinkwitzRileyFilter<float> (4th-order LR4 = 2x cascaded Butterworth SVF)
struct LR4Filter {
  enum class Type { Lowpass, Highpass };
  Type type = Type::Lowpass;
  double sampleRate = 48000.0;
  float cutoffHz = 2000.0f;
  float g = 0.0f, h = 0.0f;
  static constexpr float kR2 = 1.4142135623730951f;  // sqrt(2)

  struct State { float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f; };
  std::array<State, PitchShift::kMaxChannels> state{};

  void prepare(double sr, Type t) {
    sampleRate = sr;
    type = t;
    for (auto& s : state) s = State{};
    updateCoeffs();
  }
  void setCutoffFrequency(float hz) {
    cutoffHz = hz;
    updateCoeffs();
  }
  void updateCoeffs() {
    const double fc = std::clamp(static_cast<double>(cutoffHz), 10.0, sampleRate * 0.49);
    g = static_cast<float>(std::tan(kPi * fc / sampleRate));
    h = 1.0f / (1.0f + kR2 * g + g * g);
  }
  float processSample(int ch, float x) {
    auto& s = state[static_cast<size_t>(ch)];
    // Stage 1
    const float yh1 = (x - (kR2 + g) * s.s1 - s.s2) * h;
    const float tb1 = g * yh1;
    const float yb1 = tb1 + s.s1;
    s.s1 = tb1 + yb1;
    const float tl1 = g * yb1;
    const float yl1 = tl1 + s.s2;
    s.s2 = tl1 + yl1;
    const float in2 = (type == Type::Lowpass) ? yl1 : yh1;
    // Stage 2
    const float yh2 = (in2 - (kR2 + g) * s.s3 - s.s4) * h;
    const float tb2 = g * yh2;
    const float yb2 = tb2 + s.s3;
    s.s3 = tb2 + yb2;
    const float tl2 = g * yb2;
    const float yl2 = tl2 + s.s4;
    s.s4 = tl2 + yl2;
    return (type == Type::Lowpass) ? yl2 : yh2;
  }
};
}  // namespace

struct PitchShift::Impl {
  struct Ring {
    std::vector<float> buf;
    uint32_t mask = 0;
    void init(int minSize) {
      int size = 1;
      while (size < minSize) size <<= 1;
      buf.assign(static_cast<size_t>(size), 0.0f);
      mask = static_cast<uint32_t>(size - 1);
    }
    void clear() { std::fill(buf.begin(), buf.end(), 0.0f); }
    void write(int64_t i, float x) { buf[static_cast<uint32_t>(i) & mask] = x; }
    float at(int64_t i) const { return buf[static_cast<uint32_t>(i) & mask]; }
    float read(double pos) const {
      const auto i = static_cast<int64_t>(std::floor(pos));
      const float t = static_cast<float>(pos - static_cast<double>(i));
      const float xm1 = at(i - 1), x0 = at(i), x1 = at(i + 1), x2 = at(i + 2);
      const float c = 0.5f * (x1 - xm1);
      const float v = x0 - x1;
      const float w = c + v;
      const float a = w + v + 0.5f * (x2 - x0);
      const float b = w + a;
      return ((a * t - b) * t + c) * t + x0;
    }
  };

  struct MirroredRing {
    std::vector<float> buf;
    uint32_t mask = 0;
    int size = 0;
    void init(int minSize) {
      size = 1;
      while (size < minSize) size <<= 1;
      buf.assign(static_cast<size_t>(size) * 2, 0.0f);
      mask = static_cast<uint32_t>(size - 1);
    }
    void clear() { std::fill(buf.begin(), buf.end(), 0.0f); }
    void write(int64_t i, float x) {
      const auto k = static_cast<uint32_t>(i) & mask;
      buf[k] = x;
      buf[k + static_cast<uint32_t>(size)] = x;
    }
    const float* windowEndingAt(int64_t i, int n) const {
      const auto k = static_cast<uint32_t>(i) & mask;
      return buf.data() + (static_cast<int>(k) + size - n);
    }
  };

  double sampleRate = 0.0;
  Params params;
  double ratio = 1.0;

  bool enabled = false, running = false;
  SmoothedValue wetMix;
  int primeLeft = 0;

  std::array<Ring, kMaxChannels> rings;
  MirroredRing control;
  std::array<Ring, kMaxChannels> dryRings;
  int64_t written = 0;

  int dMin = 0, dMax = 0, corrLen = 0;
  int fadeLen = 0, fadeMaxLen = 0, fadeMinLen = 0, fadeLo = 0, fadeHi = 0, onsetFadeLen = 0;
  int onsetSpan = 0, refractory = 0, cellLen = 1, coarseStep = 1, searchLead = 1, searchLeadNow = 1;

  double rA = 0.0, rB = 0.0;
  bool fading = false;
  double fade = 0.0, fadeInc = 0.0, fadeNcc = 1.0;

  struct Search {
    bool active = false, ready = false;
    int lo = 0, hi = 0, next = 0, perSample = 1;
    double bestScore = -1e9, bestNcc = 0.0;
    int bestDelay = 0;
    int64_t now0 = 0, a0 = 0;
    int64_t resultJump = 0;
  } search;
  std::vector<float> ref;
  double refXX = 0.0;

  double hpA = 0.0, hpZ1 = 0.0, hpZ2 = 0.0;
  double smoothA = 0.0, eHf = 0.0;
  std::array<float, kHistoryCells> minHist{}, maxHist{};
  double minAcc = 1e9, maxAcc = 0.0;
  int64_t lastOnset = 0;

  LR4Filter wetLowpass, dryHighpass;
  SmoothedValue tonalityMix;

  void applyWindow() {
    dMin = minDelaySamples(sampleRate);
    dMax = windowSamples(params.window, sampleRate);
    corrLen = std::min(dMax, static_cast<int>(sampleRate * kMaxCorrMs * 0.001));
    updateFadeRange();
  }

  void updateFadeRange() {
    const double drift = std::abs(1.0 - ratio);
    double hi = fadeMaxLen;
    double lead = searchLead;
    int floor = fadeMinLen;
    if (ratio > 1.0) {
      hi = std::max<double>(fadeMinLen, std::min(hi, static_cast<double>(dMax - dMin) / (6.0 * drift)));
      const double cost = drift * (2.0 * hi + lead + 4.0) + 6.0;
      const double budget = dMax * (1.0 - kUpshiftLandShare);
      if (cost > budget) {
        const double scale = budget / cost;
        hi *= scale;
        lead *= scale;
        floor = 8;
      }
    }
    fadeHi = std::max(floor, static_cast<int>(hi));
    fadeLo = std::min(fadeLen, fadeHi);
    searchLeadNow = std::max(1, static_cast<int>(lead));
  }

  int lowGuard() const {
    return std::max(dMin, static_cast<int>(std::max(0.0, ratio - 1.0) * fadeHi) + 4);
  }

  int fadeFor(double ncc, double room) const {
    const double t = std::clamp((kFadeNccHi - ncc) / (kFadeNccHi - kFadeNccLo), 0.0, 1.0);
    const double len = fadeLo + t * (fadeHi - fadeLo);
    return std::max(8, static_cast<int>(std::min(len, room)));
  }

  // 0 or >= kTonalityOffHz (20 kHz) means off: crossover blends out for a pure shift.
  void setTonality(float hz) {
    const bool on = hz > 0.0f && hz < (kTonalityOffHz - 1.0f);
    if (on) {
      const float cutoff = std::clamp(hz, kTonalityMinHz, kTonalityOffHz);
      wetLowpass.setCutoffFrequency(cutoff);
      dryHighpass.setCutoffFrequency(cutoff);
    }
    tonalityMix.setTargetValue(on ? 1.0f : 0.0f);
  }

  void reset() {
    for (auto& r : rings) r.clear();
    control.clear();
    for (auto& r : dryRings) r.clear();
    rA = rB = static_cast<double>(written - dMin);
    primeLeft = dMin + 8;
    fading = false;
    search = Search{};
    hpZ1 = hpZ2 = 0.0;
    eHf = 0.0;
    minHist.fill(1e9f);
    maxHist.fill(0.0f);
    minAcc = 1e9;
    maxAcc = 0.0;
    lastOnset = -(1 << 30);
  }

  double nccAt(int64_t b) const {
    const float* y = control.windowEndingAt(b, corrLen);
    const float* x = ref.data();
    float xy[8] = {}, yy[8] = {};
    int i = 0;
    for (; i + 8 <= corrLen; i += 8)
      for (int k = 0; k < 8; ++k) {
        xy[k] += x[i + k] * y[i + k];
        yy[k] += y[i + k] * y[i + k];
      }
    for (; i < corrLen; ++i) {
      xy[0] += x[i] * y[i];
      yy[0] += y[i] * y[i];
    }
    double sxy = 0.0, syy = 0.0;
    for (int k = 0; k < 8; ++k) {
      sxy += xy[k];
      syy += yy[k];
    }
    return sxy / (std::sqrt(refXX * syy) + 1e-9);
  }

  void beginSearch(int lo, int hi, int lead) {
    search.active = true;
    search.ready = false;
    search.now0 = written - 1;
    search.a0 = static_cast<int64_t>(std::floor(rA));
    search.lo = lo;
    search.hi = std::max(lo, hi);
    search.next = lo;
    search.bestScore = -1e9;
    search.bestDelay = lo;
    const float* x = control.windowEndingAt(search.a0, corrLen);
    std::copy(x, x + corrLen, ref.begin());
    refXX = 0.0;
    for (int i = 0; i < corrLen; ++i) refXX += static_cast<double>(x[i]) * x[i];
    const int candidates = (search.hi - search.lo) / coarseStep + 1;
    search.perSample = lead > 0 ? (candidates + lead - 1) / lead : candidates;
  }

  void consider(int d) {
    const double curDelay = static_cast<double>(search.now0 - search.a0);
    const double jump = std::max(1.0, std::abs(curDelay - static_cast<double>(d)));
    const double ncc = nccAt(search.now0 - d);
    const double s = -(1.0 - ncc) / jump;
    if (s > search.bestScore) {
      search.bestScore = s;
      search.bestDelay = d;
      search.bestNcc = ncc;
    }
  }

  void stepSearch(int count) {
    while (count-- > 0 && search.next <= search.hi) {
      consider(search.next);
      search.next += coarseStep;
    }
    if (search.next > search.hi) {
      if (coarseStep > 1) {
        const int c = search.bestDelay;
        for (int d = std::max(search.lo, c - coarseStep + 1); d <= std::min(search.hi, c + coarseStep - 1); ++d)
          if (d != c) consider(d);
      }
      search.resultJump = (search.now0 - search.bestDelay) - search.a0;
      search.active = false;
      search.ready = true;
    }
  }

  void startFade(int64_t jump, int fadeSamples) {
    search.ready = false;
    const double destDelay = static_cast<double>(written - 1) - (rA + static_cast<double>(jump));
    if (destDelay < 2.0 || destDelay > static_cast<double>(rings[0].buf.size()) - 8.0) return;
    rB = rA + static_cast<double>(jump);
    fading = true;
    fade = 0.0;
    fadeInc = 1.0 / fadeSamples;
    fadeNcc = std::clamp(search.bestNcc, 0.0, 1.0);
  }

  bool detectOnset(double x) {
    const int64_t now = written - 1;
    const double hp = x - hpZ1 + hpA * hpZ2;
    hpZ1 = x;
    hpZ2 = hp;
    eHf = smoothA * eHf + (1.0 - smoothA) * hp * hp;
    const int64_t cellNow = now / cellLen;
    if (now % cellLen == 0 && now > 0) {
      const auto prev = static_cast<size_t>((cellNow - 1) % kHistoryCells);
      minHist[prev] = static_cast<float>(minAcc);
      maxHist[prev] = static_cast<float>(maxAcc);
      minAcc = 1e9;
      maxAcc = 0.0;
    }
    minAcc = std::min(minAcc, eHf);
    maxAcc = std::max(maxAcc, eHf);
    double recentMin = 1e9, recentMax = 0.0;
    for (int back = kHistorySkipCells; back < kHistoryCells; ++back) {
      const auto c = static_cast<size_t>((cellNow - back + 4 * kHistoryCells) % kHistoryCells);
      recentMin = std::min(recentMin, static_cast<double>(minHist[c]));
      recentMax = std::max(recentMax, static_cast<double>(maxHist[c]));
    }
    return eHf > 1e-7 && eHf > recentMin * kOnsetOverMin && eHf > recentMax * kOnsetOverMax;
  }
};

PitchShift::PitchShift() : impl_(std::make_unique<Impl>()) {}
PitchShift::~PitchShift() = default;

void PitchShift::prepare(double sampleRate, int /*maxBlockSamples*/) {
  auto& s = *impl_;
  s.sampleRate = sampleRate;
  s.fadeLen = std::max(8, static_cast<int>(sampleRate * kFadeMs * 0.001));
  s.fadeMaxLen = std::max(s.fadeLen, static_cast<int>(sampleRate * kFadeMaxMs * 0.001));
  s.fadeMinLen = std::max(8, static_cast<int>(sampleRate * kFadeMinMs * 0.001));
  s.onsetFadeLen = std::max(8, static_cast<int>(sampleRate * kOnsetFadeMs * 0.001));
  s.onsetSpan = static_cast<int>(sampleRate * kOnsetSpanMs * 0.001);
  s.refractory = static_cast<int>(sampleRate * kRefractoryMs * 0.001);
  s.searchLead = std::max(1, static_cast<int>(sampleRate * kSearchLeadMs * 0.001));
  s.cellLen = std::max(1, static_cast<int>(sampleRate * 0.001));
  s.coarseStep = std::max(1, static_cast<int>(std::lround(sampleRate / kCoarseStepRate)));
  s.hpA = std::exp(-2.0 * kPi * kDetectorHpfHz / sampleRate);
  s.smoothA = std::exp(-1.0 / (sampleRate * kDetectorSmoothMs * 0.001));

  const int maxCorr = static_cast<int>(sampleRate * kMaxCorrMs * 0.001);
  const int reach = windowSamples(Window::ms60, sampleRate) + maxCorr + s.fadeMaxLen + 2 * s.searchLead + 64;
  for (auto& r : s.rings) r.init(reach);
  s.control.init(reach);
  s.ref.assign(static_cast<size_t>(maxCorr), 0.0f);
  for (auto& r : s.dryRings) r.init(minDelaySamples(sampleRate) + 8);

  s.wetLowpass.prepare(sampleRate, LR4Filter::Type::Lowpass);
  s.dryHighpass.prepare(sampleRate, LR4Filter::Type::Highpass);
  s.tonalityMix.reset(sampleRate, kBlendSeconds);
  s.setTonality(s.params.tonalityHz);

  s.wetMix.reset(sampleRate, kBlendSeconds);
  s.wetMix.setCurrentAndTargetValue(0.0f);
  s.running = s.enabled;
  s.applyWindow();
  s.reset();
}

void PitchShift::setEnabled(bool on) {
  auto& s = *impl_;
  if (on == s.enabled) return;
  s.enabled = on;
  if (!on) {
    s.wetMix.setTargetValue(0.0f);
  } else if (s.running) {
    s.wetMix.setTargetValue(1.0f);
  } else {
    s.reset();
    s.running = true;
  }
}

bool PitchShift::isRunning() const { return impl_->running; }

void PitchShift::setParams(const Params& p) {
  auto& s = *impl_;
  if (p.window != s.params.window) {
    s.params.window = p.window;
    s.applyWindow();
    s.search = Impl::Search{};
  }
  const int clampedSemis = std::clamp(p.semitones, -kSemitoneRange, kSemitoneRange);
  if (clampedSemis != s.params.semitones) {
    s.params.semitones = clampedSemis;
    const double ratio = std::pow(2.0, static_cast<double>(clampedSemis) / 12.0);
    const auto direction = [](double r) { return r > 1.0 ? 1 : r < 1.0 ? -1 : 0; };
    if (direction(ratio) != direction(s.ratio) || std::abs(ratio - s.ratio) > kSearchKeepRatio)
      s.search = Impl::Search{};
    s.ratio = ratio;
    s.updateFadeRange();
  }
  if (p.tonalityHz != s.params.tonalityHz) {
    s.params.tonalityHz = p.tonalityHz;
    s.setTonality(p.tonalityHz);
  }
}

void PitchShift::process(float* const* channels, int numChannels, int numSamples) {
  auto& s = *impl_;
  numChannels = std::min(numChannels, kMaxChannels);
  if (s.sampleRate <= 0.0 || numChannels <= 0 || numSamples <= 0 || !s.running) return;

  const float channelScale = 1.0f / static_cast<float>(numChannels);
  const double ratio = s.ratio;
  const double drift = std::abs(1.0 - ratio);
  const double leadDrift = drift * (s.searchLeadNow + 4);
  const int leadDriftI = static_cast<int>(std::ceil(leadDrift));
  const int fadeDrift = static_cast<int>(std::ceil(drift * (ratio > 1.0 ? s.fadeHi : s.fadeMinLen)));

  for (int i = 0; i < numSamples; ++i) {
    const int64_t now = s.written++;
    float dry[kMaxChannels];
    float control = 0.0f;
    for (int ch = 0; ch < numChannels; ++ch) {
      dry[ch] = channels[ch][i];
      s.rings[static_cast<size_t>(ch)].write(now, dry[ch]);
      control += dry[ch];
    }
    control *= channelScale;
    s.control.write(now, control);

    const double delay = static_cast<double>(now) - s.rA;
    const int guard = s.lowGuard();
    const bool onset = s.detectOnset(control);
    const int landLo = ratio > 1.0 ? std::min(guard + fadeDrift + leadDriftI + 2, s.dMax) : s.dMin;
    const int landHi = ratio < 1.0 ? std::max(s.dMin, s.dMax - fadeDrift - leadDriftI - 2) : s.dMax;
    if (!s.fading) {
      if (onset && now - s.lastOnset > s.refractory && delay > landLo + s.onsetSpan) {
        s.lastOnset = now;
        s.beginSearch(landLo, std::min(landLo + s.onsetSpan, s.dMax), 0);
        s.stepSearch(s.search.perSample);
        s.startFade(s.search.resultJump, s.onsetFadeLen);
        s.minHist.fill(static_cast<float>(s.eHf));
        s.maxHist.fill(static_cast<float>(s.eHf));
        s.minAcc = s.maxAcc = s.eHf;
      } else if (ratio < 1.0) {
        if (!s.search.active && !s.search.ready && delay >= s.dMax - leadDrift) {
          const int lo = std::max(4, landLo - leadDriftI);
          s.beginSearch(lo, std::max(lo, landHi - leadDriftI), s.searchLeadNow);
        }
        if (s.search.ready && delay >= s.dMax) {
          const double dest = delay - static_cast<double>(s.search.resultJump);
          const double room = (static_cast<double>(s.dMax) - dest - leadDrift - 2.0) / drift;
          s.startFade(s.search.resultJump, s.fadeFor(s.search.bestNcc, room));
        }
      } else if (ratio > 1.0) {
        if (!s.search.active && !s.search.ready && delay <= guard + leadDrift)
          s.beginSearch(landLo + leadDriftI, landHi + leadDriftI, s.searchLeadNow);
        if (s.search.ready && delay <= guard) s.startFade(s.search.resultJump, s.fadeFor(s.search.bestNcc, s.fadeHi));
      }
    }
    if (s.search.active) s.stepSearch(s.search.perSample);

    float gainA = 1.0f, gainB = 0.0f;
    if (s.fading) {
      gainB = static_cast<float>(0.5 - 0.5 * std::cos(kPi * s.fade));
      gainA = 1.0f - gainB;
      const float r = static_cast<float>(s.fadeNcc);
      const float norm = std::sqrt(gainA * gainA + gainB * gainB + 2.0f * gainA * gainB * r);
      gainA /= norm;
      gainB /= norm;
      s.fade += s.fadeInc;
    }
    if (s.primeLeft > 0 && --s.primeLeft == 0 && s.enabled) s.wetMix.setTargetValue(1.0f);
    const float wetMix = s.wetMix.getNextValue();
    const float tonalityMix = s.tonalityMix.getNextValue();
    for (int ch = 0; ch < numChannels; ++ch) {
      auto& ring = s.rings[static_cast<size_t>(ch)];
      float wet = ring.read(s.rA);
      if (s.fading) wet = wet * gainA + gainB * ring.read(s.rB);
      auto& dryRing = s.dryRings[static_cast<size_t>(ch)];
      dryRing.write(now, s.dryHighpass.processSample(ch, dry[ch]));
      const float split = s.wetLowpass.processSample(ch, wet) + dryRing.at(now - s.dMin);
      wet += tonalityMix * (split - wet);
      channels[ch][i] = dry[ch] + wetMix * (wet - dry[ch]);
    }
    if (s.fading && s.fade >= 1.0) {
      s.rA = s.rB;
      s.fading = false;
    }
    s.rA += ratio;
    s.rB += ratio;
  }
  if (!s.enabled && !s.wetMix.isSmoothing()) s.running = false;
}

int PitchShift::latencySamples() const { return (impl_->dMin + impl_->dMax) / 2; }